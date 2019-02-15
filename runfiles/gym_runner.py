# Created by Andrew Silva, andrew.silva@gatech.edu
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import torch
sys.path.insert(0, '../')
from agents.prolonet_agent import DeepProLoNet
from agents.non_deep_prolonet_agent import ShallowProLoNet
from agents.random_prolonet_agent import RandomProLoNet
from agents.lstm_agent import LSTMNet
from agents.baseline_agent import FCNet
import random
import torch.multiprocessing as mp
import argparse


def run_episode(q, agent_in, ENV_NAME):
    agent = agent_in.duplicate()
    if ENV_NAME == 'lunar':
        env = gym.make('LunarLander-v2')
    elif ENV_NAME == 'cart':
        env = gym.make('CartPole-v1')
    else:
        raise Exception('No valid environment selected')

    state = env.reset()  # Reset environment and record the starting state
    done = False

    for timestep in range(1000):
        action = agent.get_action(state)
        # Step through environment using chosen action
        state, reward, done, _ = env.step(action)

        # Save reward
        agent.save_reward(reward)
        if done:
            break
    R = 0
    rewards = []
    all_rewards = agent.replay_buffer.rewards_list

    reward_sum = sum(all_rewards)
    all_values = agent.replay_buffer.value_list
    deeper_all_values = agent.replay_buffer.deeper_value_list
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
                deeper_advantages.std() + np.finfo(np.float32).eps)
        agent.replay_buffer.deeper_advantage_list = deeper_advantages.detach().clone().cpu().numpy().tolist()
    else:
        agent.replay_buffer.deeper_advantage_list = [None] * len(all_rewards)
    # Scale rewards .abs()
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    advantages = (advantages - advantages.mean()) / (advantages.std() + np.finfo(np.float32).eps)

    agent.replay_buffer.rewards_list = rewards.detach().clone().cpu().numpy().tolist()
    agent.replay_buffer.advantage_list = advantages.detach().clone().cpu().numpy().tolist()
    if q is not None:
        try:
            q.put([reward_sum, agent.replay_buffer.__getstate__()])
        except RuntimeError as e:
            print(e)
            return [reward_sum, agent.replay_buffer.__getstate__()]
    return [reward_sum, agent.replay_buffer.__getstate__()]


def main(episodes, agent, num_processes, ENV_NAME):
    running_reward_array = []
    for episode in range(episodes):
        master_reward = 0
        reward, running_reward = 0, 0
        processes = []
        q = mp.Manager().Queue()
        for proc in range(num_processes):
            p = mp.Process(target=run_episode, args=(q, agent, ENV_NAME))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        while not q.empty():
            fake_out = q.get()
            master_reward += fake_out[0]
            running_reward_array.append(fake_out[0])
            agent.replay_buffer.extend(fake_out[1])

        tuple_out = run_episode(None, agent, ENV_NAME)
        master_reward += tuple_out[0]
        running_reward_array.append(tuple_out[0])
        agent.replay_buffer.extend(tuple_out[1])
        reward = master_reward / float(num_processes+1)
        agent.end_episode(reward, num_processes)

        running_reward = sum(running_reward_array[-100:]) / float(min(100.0, len(running_reward_array)))
        print(episode)
        if episode % 50 == 0:
            print(f'Episode {episode}  Last Reward: {reward}  Average Reward: {running_reward}')
        if episode % 500 == 0:
            agent.save('../models/'+str(episode)+'th')
    return running_reward_array


if __name__ == "__main__":
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
    parser.add_argument("-env", "--env_type", help="environment to run on", type=str, default='cart')

    args = parser.parse_args()
    AGENT_TYPE = args.agent_type  # 'shallow_prolo', 'prolo', 'random', 'fc', 'lstm'
    ADVERSARIAL = args.adversary  # Adversarial prolo, applies for AGENT_TYPE=='shallow_prolo'
    SL_INIT = args.sl_init  # SL->RL fc, applies only for AGENT_TYPE=='fc'
    DEEPEN_METHOD = args.deepen_method  # 'random', 'fc', 'parent', method for deepening, only applies for AGENT_TYPE=='prolo'
    DEEPEN_CRITERIA = args.deepen_criteria  # 'entropy', 'num', 'value', criteria for when to deepen. AGENT_TYPE=='prolo' only
    NUM_EPS = args.episodes
    NUM_PROCS = args.processes
    ENV_TYPE = args.env_type
    for NUM_PROCS in [NUM_PROCS]:
        if ENV_TYPE == 'lunar':
            init_env = gym.make('LunarLander-v2')
            dim_in = init_env.observation_space.shape[0]
            dim_out = init_env.action_space.n
        elif ENV_TYPE == 'cart':
            init_env = gym.make('CartPole-v1')
            dim_in = init_env.observation_space.shape[0]
            dim_out = init_env.action_space.n
        else:
            raise Exception('No valid environment selected')

        print(f"Agent {AGENT_TYPE} on {ENV_TYPE} with {NUM_PROCS} runners")
        mp.set_start_method('spawn')

        for i in range(5):
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            bot_name = AGENT_TYPE + ENV_TYPE

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

            if AGENT_TYPE == 'fc' and SL_INIT:
                NUM_EPS += policy_agent.action_loss_threshold
            num_procs = NUM_PROCS
            reward_array = main(NUM_EPS, policy_agent, num_procs, ENV_TYPE)
