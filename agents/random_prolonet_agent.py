# Created by Andrew Silva, andrew.silva@gatech.edu
import torch
from torch.distributions import Categorical
from opt_helpers import replay_buffer, ppo_update
import copy
import os
from agents.prolonet_helpers import init_random_cart_net, init_random_lander_net,\
    save_prolonet, load_prolonet, init_random_sc_net, init_random_micro_net


class RandomProLoNet:
    def __init__(self,
                 bot_name='RandomProLoNet',
                 input_dim=4,
                 output_dim=2):
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        # self.replay_buffer = ReplayMemory(1000)
        self.bot_name = bot_name
        if input_dim == 4 and output_dim == 2:
            self.action_network, self.value_network = init_random_cart_net()

        elif input_dim == 8 and output_dim == 4:
            self.action_network, self.value_network = init_random_lander_net()

        elif input_dim == 194 and output_dim == 44:
            self.action_network, self.value_network = init_random_sc_net()
        elif input_dim == 32 and output_dim == 10:
            self.action_network, self.value_network = init_random_micro_net()

        self.ppo = ppo_update.PPO([self.action_network, self.value_network], two_nets=True)
        self.actor_opt = torch.optim.RMSprop(self.action_network.parameters())
        self.value_opt = torch.optim.RMSprop(self.value_network.parameters())
        self.last_state = [0, 0, 0, 0]
        self.last_action = 0
        self.last_action_probs = torch.Tensor([0])
        self.last_value_pred = torch.Tensor([[0, 0]])
        self.full_probs = None
        self.reward_history = []
        self.num_steps = 0

    def get_action(self, observation):
        with torch.no_grad():
            obs = torch.Tensor(observation)
            obs = obs.view(1, -1)
            self.last_state = obs
            probs = self.action_network(obs)
            value_pred = self.value_network(obs)
            probs = probs.view(-1)
            self.full_probs = probs
            if self.action_network.input_dim > 10:
                probs, inds = torch.topk(probs, 3)
            m = Categorical(probs)
            action = m.sample()
            log_probs = m.log_prob(action)
            self.last_action_probs = log_probs
            self.last_value_pred = value_pred

            if self.action_network.input_dim > 10:
                self.last_action = inds[action]
            else:
                self.last_action = action
        if self.action_network.input_dim > 10:
            action = inds[action].item()
        else:
            action = action.item()
        return action

    def save_reward(self, reward):
        self.replay_buffer.insert(obs=[self.last_state],
                                  action_log_probs=self.last_action_probs,
                                  value_preds=self.last_value_pred[self.last_action.item()],
                                  last_action=self.last_action,
                                  full_probs_vector=self.full_probs,
                                  rewards=reward)
        return True

    def end_episode(self, timesteps, num_processes=1):
        self.reward_history.append(timesteps)
        value_loss, action_loss = self.ppo.cartpole_update(self.replay_buffer, self)
        bot_name = '../txts/' + self.bot_name + str(num_processes) + '_processes'
        with open(bot_name + "_losses.txt", "a") as myfile:
            myfile.write(str(value_loss + action_loss) + '\n')
        with open(bot_name + '_rewards.txt', 'a') as myfile:
            myfile.write(str(timesteps) + '\n')
        self.num_steps += 1

    def lower_lr(self):
        for param_group in self.ppo.actor_opt.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
        for param_group in self.ppo.critic_opt.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5

    def reset(self):
        self.replay_buffer.clear()

    def save(self, fn='last'):
        act_fn = fn + self.bot_name + '_actor_' + '.pth.tar'
        val_fn = fn + self.bot_name + '_critic_' + '.pth.tar'

        save_prolonet(act_fn, self.action_network)
        save_prolonet(val_fn, self.value_network)

    def load(self, fn='last'):
        act_fn = fn + self.bot_name + '_actor_' + '.pth.tar'
        val_fn = fn + self.bot_name + '_critic_' + '.pth.tar'

        if os.path.exists(act_fn):
            self.action_network = load_prolonet(act_fn)
            self.value_network = load_prolonet(val_fn)

    def deepen_networks(self, force_switch=False):
        pass

    def change_to_deeper(self, random=False):
        pass

    def __getstate__(self):
        return {
            'action_network': self.action_network,
            'value_network': self.value_network,
            'ppo': self.ppo,
            'actor_opt': self.actor_opt,
            'value_opt': self.value_opt,
        }

    def __setstate__(self, state):
        self.action_network = copy.deepcopy(state['action_network'])
        self.value_network = copy.deepcopy(state['value_network'])
        self.ppo = copy.deepcopy(state['ppo'])
        self.actor_opt = copy.deepcopy(state['actor_opt'])
        self.value_opt = copy.deepcopy(state['value_opt'])

    def duplicate(self):
        new_agent = RandomProLoNet()
        new_agent.__setstate__(self.__getstate__())
        return new_agent
