# Created by Andrew Silva, andrew.silva@gatech.edu
import sc2
from sc2 import Race, Difficulty
import os
import sys
from sc2.constants import *
from sc2.position import Pointlike, Point2
from sc2.player import Bot, Computer
from sc2.unit import Unit as sc2Unit
sys.path.insert(0, os.path.abspath('../'))
import torch
from agents.prolonet_agent import DeepProLoNet
from agents.non_deep_prolonet_agent import ShallowProLoNet
from agents.random_prolonet_agent import RandomProLoNet
from agents.lstm_agent import LSTMNet
from agents.baseline_agent import FCNet
from runfiles import sc_helpers
import numpy as np
import torch.multiprocessing as mp
import argparse

DEBUG = False
SUPER_DEBUG = False
if SUPER_DEBUG:
    DEBUG = True

FAILED_REWARD = -0.0
SUCCESS_BUILD_REWARD = 1.
SUCCESS_TRAIN_REWARD = 1.
SUCCESS_SCOUT_REWARD = 1.
SUCCESS_ATTACK_REWARD = 1.
SUCCESS_MINING_REWARD = 1.


def discount_reward(reward, value, deeper_value):
    R = 0
    rewards = []
    all_rewards = reward
    reward_sum = sum(all_rewards)
    all_values = value
    deeper_all_values = deeper_value
    # Discount future rewards back to the present using gamma
    advantages = []
    deeper_advantages = []

    for r, v, d_v in zip(all_rewards[::-1], all_values[::-1], deeper_all_values[::-1]):
        R = r + 0.99 * R
        rewards.insert(0, R)
        advantages.insert(0, R - v)
        if d_v is not None:
            deeper_advantages.insert(0, R - d_v)
    advantages = torch.Tensor(advantages)
    rewards = torch.Tensor(rewards)

    if len(deeper_advantages) > 0:
        deeper_advantages = torch.Tensor(deeper_advantages)
        deeper_advantages = (deeper_advantages - deeper_advantages.mean()) / (
                deeper_advantages.std() + torch.Tensor([np.finfo(np.float32).eps]))
        deeper_advantage_list = deeper_advantages.detach().clone().cpu().numpy().tolist()
    else:
        deeper_advantage_list = [None] * len(all_rewards)
    # Scale rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + torch.Tensor([np.finfo(np.float32).eps]))
    advantages = (advantages - advantages.mean()) / (advantages.std() + torch.Tensor([np.finfo(np.float32).eps]))
    rewards_list = rewards.detach().clone().cpu().numpy().tolist()
    advantage_list = advantages.detach().clone().cpu().numpy().tolist()
    return rewards_list, advantage_list, deeper_advantage_list


class SC2MicroBot(sc2.BotAI):
    def __init__(self, rl_agent):
        super(SC2MicroBot, self).__init__()
        self.agent = rl_agent
        self.action_buffer = []
        self.prev_state = None
        self.last_known_enemy_units = []
        self.itercount = 0
        self.last_reward = 0
        self.my_tags = None
        self.agent_list = []
        self.dead_agent_list = []
        self.dead_index_mover = 0

    async def on_step(self, iteration):

        if iteration == 0:
            self.my_tags = [unit.tag for unit in self.units]
            for unit in self.units:
                self.agent_list.append(self.agent.duplicate())
        else:
            self.last_reward = 0
            for unit in self.state.dead_units:
                if unit in self.my_tags:
                    self.last_reward -= 1
                    self.dead_agent_list.append(self.agent_list[self.my_tags.index(unit)])
                    del self.agent_list[self.my_tags.index(unit)]
                    del self.my_tags[self.my_tags.index(unit)]
                    self.dead_agent_list[-1].save_reward(self.last_reward)
                else:
                    self.last_reward += 1
            if len(self.state.dead_units) > 0:
                for agent in self.agent_list:
                    agent.save_reward(self.last_reward)
        # if iteration % 20 != 0:
        #     return
        all_unit_data = []
        for unit in self.units:
            all_unit_data.append(sc_helpers.get_unit_data(unit))
        while len(all_unit_data) < 3:
            all_unit_data.append([-1, -1, -1, -1])
        for unit, agent in zip(self.units, self.agent_list):
            my_index = self.units.index(unit)
            unit_data = all_unit_data[my_index]
            allied_data = all_unit_data[0:my_index] + all_unit_data[my_index+1:]
            nearest_enemies = sc_helpers.get_nearest_enemies(unit, self.known_enemy_units)
            unit_data = np.array(unit_data).reshape(-1)
            allied_data = np.array(allied_data).reshape(-1)
            nearest_enemies = nearest_enemies
            enemy_data = []
            for enemy in nearest_enemies:
                enemy_data.extend(sc_helpers.get_unit_data(enemy))
            enemy_data = np.array(enemy_data).reshape(-1)
            state_in = np.concatenate((unit_data, allied_data, enemy_data))
            action = agent.get_action(state_in)
            await self.execute_unit_action(unit, action, nearest_enemies)
        try:
            await self.do_actions(self.action_buffer)
        except sc2.protocol.ProtocolError:
            print("Not in game?")
            self.action_buffer = []
            return
        self.action_buffer = []

    async def execute_unit_action(self, unit_in, action_in, nearest_enemies):
        if action_in < 4:
            await self.move_unit(unit_in, action_in)
        elif action_in < 9:
            await self.attack_nearest(unit_in, action_in, nearest_enemies)
        else:
            pass

    async def move_unit(self, unit_to_move, direction):
        current_pos = unit_to_move.position
        target_destination = current_pos
        if direction == 0:
            target_destination = [current_pos.x, current_pos.y + 5]
        elif direction == 1:
            target_destination = [current_pos.x + 5, current_pos.y]
        elif direction == 2:
            target_destination = [current_pos.x, current_pos.y - 5]
        elif direction == 3:
            target_destination = [current_pos.x - 5, current_pos.y]
        self.action_buffer.append(unit_to_move.move(Point2(Pointlike(target_destination))))

    async def attack_nearest(self, unit_to_attack, action_in, nearest_enemies_list):
        if len(nearest_enemies_list) > action_in-4:
            target = nearest_enemies_list[action_in-4]
            if target is None:
                return -1
            self.action_buffer.append(unit_to_attack.attack(target))
        else:
            return -1

    def finish_episode(self, game_result):
        print("Game over!")
        if game_result == sc2.Result.Defeat:
            for index in range(len(self.agent_list), 0, -1):
                self.dead_agent_list.append(self.agent_list[index-1])
                self.dead_agent_list[-1].save_reward(-1)
            del self.agent_list[:]
        elif game_result == sc2.Result.Tie:
            reward = 0
        elif game_result == sc2.Result.Victory:
            reward = 0  # - min(self.itercount/500.0, 900) + self.units.amount
        else:
            # ???
            return -13
        if len(self.agent_list) > 0:
            reward_sum = sum(self.agent_list[0].replay_buffer.rewards_list)
        else:
            reward_sum = sum(self.dead_agent_list[-1].replay_buffer.rewards_list)

        for agent_index in range(len(self.agent_list)):
            rewards_list, advantage_list, deeper_advantage_list = discount_reward(
                self.agent_list[agent_index].replay_buffer.rewards_list,
                self.agent_list[agent_index].replay_buffer.value_list,
                self.agent_list[agent_index].replay_buffer.deeper_value_list)
            self.agent_list[agent_index].replay_buffer.rewards_list = rewards_list
            self.agent_list[agent_index].replay_buffer.advantage_list = advantage_list
            self.agent_list[agent_index].replay_buffer.deeper_advantage_list = deeper_advantage_list
        for dead_agent_index in range(len(self.dead_agent_list)):
            rewards_list, advantage_list, deeper_advantage_list = discount_reward(
                self.dead_agent_list[dead_agent_index].replay_buffer.rewards_list,
                self.dead_agent_list[dead_agent_index].replay_buffer.value_list,
                self.dead_agent_list[dead_agent_index].replay_buffer.deeper_value_list)
            self.dead_agent_list[dead_agent_index].replay_buffer.rewards_list = rewards_list
            self.dead_agent_list[dead_agent_index].replay_buffer.advantage_list = advantage_list
            self.dead_agent_list[dead_agent_index].replay_buffer.deeper_advantage_list = deeper_advantage_list
        return reward_sum


def run_episode(q, main_agent):
    result = None
    agent_in = main_agent.duplicate()

    bot = SC2MicroBot(rl_agent=agent_in)

    try:
        result = sc2.run_game(sc2.maps.get("FindAndDefeatZerglings"),
                              [Bot(Race.Protoss, bot)],
                              realtime=False)
    except KeyboardInterrupt:
        result = [-1, -1]
    except Exception as e:
        print(str(e))
        print("No worries", e, " carry on please")
    if type(result) == list and len(result) > 1:
        result = result[0]
    reward_sum = bot.finish_episode(result)
    for agent in bot.agent_list+bot.dead_agent_list:
        agent_in.replay_buffer.extend(agent.replay_buffer.__getstate__())
    if q is not None:
        try:
            q.put([reward_sum, agent_in.replay_buffer.__getstate__()])
        except RuntimeError as e:
            print(e)
            return [reward_sum, agent_in.replay_buffer.__getstate__()]
    return [reward_sum, agent_in.replay_buffer.__getstate__()]


def main(episodes, agent, num_processes):
    running_reward_array = []
    # lowered = False
    mp.set_start_method('spawn')
    for episode in range(episodes):
        successful_runs = 0
        master_reward, reward, running_reward = 0, 0, 0
        processes = []
        queueue = mp.Manager().Queue()
        for proc in range(num_processes):
            p = mp.Process(target=run_episode, args=(queueue, agent))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        while not queueue.empty():
            try:
                fake_out = queueue.get()
            except MemoryError as e:
                print(e)
                fake_out = [-13, None]
            if fake_out[0] != -13:
                master_reward += fake_out[0]
                running_reward_array.append(fake_out[0])
                agent.replay_buffer.extend(fake_out[1])
                successful_runs += 1

        if successful_runs > 0:
            reward = master_reward / float(successful_runs)
            agent.end_episode(reward, num_processes)
            running_reward = sum(running_reward_array[-100:]) / float(min(100.0, len(running_reward_array)))
        if episode % 50 == 0:
            print(f'Episode {episode}  Last Reward: {reward}  Average Reward: {running_reward}')
            print(f"Running {num_processes} concurrent simulations per episode")
        if episode % 500 == 0:
            agent.save('../models/' + str(episode) + 'th')
    return running_reward_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default='fc')
    parser.add_argument("-adv", "--adversary",
                        help="for prolonet, init as adversarial? true for yes, false for no",
                        type=bool, default=False)
    parser.add_argument("-s", "--sl_init", help="sl to rl for fc net?", type=bool, default=False)
    parser.add_argument("-dm", "--deepen_method", help="how to deepen?", type=str, default='random')
    parser.add_argument("-dc", "--deepen_criteria", help="when to deepen?", type=str, default='entropy')
    parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=1000)
    parser.add_argument("-p", "--processes", help="how many processes?", type=int, default=1)

    args = parser.parse_args()
    AGENT_TYPE = args.agent_type  # 'shallow_prolo', 'prolo', 'random', 'fc', 'lstm'
    ADVERSARIAL = args.adversary  # Adversarial prolo, applies for AGENT_TYPE=='shallow_prolo'
    SL_INIT = args.sl_init  # SL->RL fc, applies only for AGENT_TYPE=='fc'
    DEEPEN_METHOD = args.deepen_method  # 'random', 'fc', 'parent', method for deepening, only applies for AGENT_TYPE=='prolo'
    DEEPEN_CRITERIA = args.deepen_criteria  # 'entropy', 'num', 'value', criteria for when to deepen. AGENT_TYPE=='prolo' only
    NUM_EPS = args.episodes
    NUM_PROCS = args.processes
    dim_in = 32
    dim_out = 10
    bot_name = AGENT_TYPE + 'SC_Micro'

    if AGENT_TYPE == 'prolo':
        policy_agent = DeepProLoNet(distribution='one_hot',
                                    bot_name=bot_name,
                                    input_dim=dim_in,
                                    output_dim=dim_out,
                                    deepen_method=DEEPEN_METHOD,
                                    deepen_criteria=DEEPEN_CRITERIA)
    elif AGENT_TYPE == 'fc':
        policy_agent = FCNet(input_dim=dim_in,
                             bot_name=bot_name,
                             output_dim=dim_out,
                             sl_init=SL_INIT)
    elif AGENT_TYPE == 'random':
        policy_agent = RandomProLoNet(input_dim=dim_in,
                                      bot_name=bot_name,
                                      output_dim=dim_out)
    elif AGENT_TYPE == 'lstm':
        policy_agent = LSTMNet(input_dim=dim_in,
                               bot_name=bot_name,
                               output_dim=dim_out)
    elif AGENT_TYPE == 'shallow_prolo':
        policy_agent = ShallowProLoNet(distribution='one_hot',
                                       input_dim=dim_in,
                                       bot_name=bot_name,
                                       output_dim=dim_out,
                                       adversarial=ADVERSARIAL)
    else:
        raise Exception('No valid network selected')
    main(episodes=NUM_EPS, agent=policy_agent, num_processes=NUM_PROCS)
