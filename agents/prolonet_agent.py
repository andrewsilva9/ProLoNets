# Created by Andrew Silva, andrew.silva@gatech.edu
import torch
from torch.distributions import Categorical
from opt_helpers import replay_buffer, ppo_update
from agents.prolonet_helpers import init_cart_nets, add_level, init_lander_nets, swap_in_node, \
    save_prolonet, load_prolonet, init_sc_nets, init_micro_net
import copy
import os


class DeepProLoNet:
    def __init__(self,
                 distribution='one_hot',
                 bot_name='ProLoNet',
                 input_dim=4,
                 output_dim=2,
                 deepen_method='random',
                 deepen_criteria='entropy'):
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        self.bot_name = bot_name

        if input_dim == 4 and output_dim == 2:
            self.action_network, self.value_network = init_cart_nets(distribution)
        elif input_dim == 8 and output_dim == 4:
            self.action_network, self.value_network = init_lander_nets(distribution)
        elif input_dim == 194 and output_dim == 44:
            self.action_network, self.value_network = init_sc_nets(distribution)
        elif input_dim == 32 and output_dim == 10:
            self.action_network, self.value_network = init_micro_net(distribution)

        self.deepen_method = deepen_method
        self.deeper_action_network = add_level(self.action_network, method=deepen_method)
        self.deeper_value_network = add_level(self.value_network, method=deepen_method)

        self.ppo = ppo_update.PPO([self.action_network, self.value_network], two_nets=True)
        self.actor_opt = torch.optim.RMSprop(self.action_network.parameters())
        self.value_opt = torch.optim.RMSprop(self.value_network.parameters())

        self.deeper_actor_opt = torch.optim.RMSprop(self.deeper_action_network.parameters())
        self.deeper_value_opt = torch.optim.RMSprop(self.deeper_value_network.parameters())

        self.last_state = [0, 0, 0, 0]
        self.last_action = 0
        self.last_action_probs = torch.Tensor([0])
        self.last_value_pred = torch.Tensor([[0, 0]])
        self.last_deep_action_probs = torch.Tensor([0])
        self.last_deep_value_pred = torch.Tensor([[0, 0]])
        self.full_probs = None
        self.deeper_full_probs = None
        self.reward_history = []
        self.num_steps = 0
        self.deepen_criteria = deepen_criteria
        self.deepen_threshold = 350
        self.times_deepened = 0

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

            deeper_obs = torch.Tensor(observation)
            deeper_obs = deeper_obs.view(1, -1)
            deeper_probs = self.deeper_action_network(deeper_obs)
            deeper_value_pred = self.deeper_value_network(deeper_obs)
            deeper_probs = deeper_probs.view(-1)
            self.deeper_full_probs = deeper_probs
            if self.action_network.input_dim > 10:
                deeper_probs, _ = torch.topk(probs, 3)
            deep_m = Categorical(deeper_probs)
            deep_log_probs = deep_m.log_prob(action)
            self.last_deep_action_probs = deep_log_probs
            self.last_deep_value_pred = deeper_value_pred
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
                                  deeper_action_log_probs=self.last_deep_action_probs,
                                  deeper_value_pred=self.last_deep_value_pred[self.last_action.item()],
                                  last_action=self.last_action,
                                  full_probs_vector=self.full_probs,
                                  deeper_full_probs_vector=self.deeper_full_probs,
                                  rewards=reward)
        return True

    def end_episode(self, timesteps, num_processes):
        value_loss, action_loss = self.ppo.cartpole_update(self.replay_buffer, self, go_deeper=True)
        self.num_steps += 1
        bot_name = '../txts/' + self.bot_name + str(num_processes) + '_processes_' + \
                   self.deepen_criteria+'_deepen_'+self.deepen_method
        with open(bot_name + "_losses.txt", "a") as myfile:
            myfile.write(str(value_loss + action_loss) + '\n')
        with open(bot_name + '_rewards.txt', 'a') as myfile:
            myfile.write(str(timesteps) + '\n')

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

        deep_act_fn = fn + self.bot_name + '_deep_actor_' + '.pth.tar'
        deep_val_fn = fn + self.bot_name + '_deep_critic_' + '.pth.tar'
        save_prolonet(act_fn, self.action_network)
        save_prolonet(val_fn, self.value_network)
        save_prolonet(deep_act_fn, self.deeper_action_network)
        save_prolonet(deep_val_fn, self.deeper_value_network)

    def load(self, fn='last'):
        act_fn = fn + self.bot_name + '_actor_' + '.pth.tar'
        val_fn = fn + self.bot_name + '_critic_' + '.pth.tar'

        deep_act_fn = fn + self.bot_name + '_deep_actor_' + '.pth.tar'
        deep_val_fn = fn + self.bot_name + '_deep_critic_' + '.pth.tar'
        if os.path.exists(act_fn):
            self.action_network = load_prolonet(act_fn)
            self.value_network = load_prolonet(val_fn)
            self.deeper_action_network = load_prolonet(deep_act_fn)
            self.deeper_value_network = load_prolonet(deep_val_fn)

    def deepen_networks(self):
        self.entropy_leaf_checks()

    def entropy_leaf_checks(self):
        leaf_max = torch.nn.Softmax(dim=0)
        new_action_network = copy.deepcopy(self.action_network)
        changes_made = []
        for leaf_index in range(len(self.action_network.action_probs)):
            existing_leaf = leaf_max(self.action_network.action_probs[leaf_index])
            new_leaf_1 = leaf_max(self.deeper_action_network.action_probs[2*leaf_index+1])
            new_leaf_2 = leaf_max(self.deeper_action_network.action_probs[2*leaf_index])
            existing_entropy = Categorical(existing_leaf).entropy().item()
            new_entropy = Categorical(new_leaf_1).entropy().item() + \
                Categorical(new_leaf_2).entropy().item()

            if new_entropy+0.1 <= existing_entropy:
                with open(self.bot_name + '_entropy_splits.txt', 'a') as myfile:
                    myfile.write('Split at ' + str(self.num_steps) + ' steps' + ': \n')
                    myfile.write('Leaf: ' + str(leaf_index) + '\n')
                    myfile.write('Prior Probs: ' + str(self.action_network.action_probs[leaf_index]) + '\n')
                    myfile.write('New Probs 1: ' + str(self.deeper_action_network.action_probs[leaf_index*2]) + '\n')
                    myfile.write('New Probs 2: ' + str(self.deeper_action_network.action_probs[leaf_index*2+1]) + '\n')

                new_action_network = swap_in_node(new_action_network, self.deeper_action_network, leaf_index)
                changes_made.append(leaf_index)
        if len(changes_made) > 0:
            self.action_network = new_action_network
            self.actor_opt = torch.optim.RMSprop(self.action_network.parameters(), lr=1e-3)
            self.ppo.actor = self.action_network
            self.ppo.actor_opt = self.actor_opt
            for change in changes_made[::-1]:
                self.deeper_action_network = swap_in_node(self.deeper_action_network, None, change*2+1)
                self.deeper_action_network = swap_in_node(self.deeper_action_network, None, change*2)
            self.deeper_actor_opt = torch.optim.RMSprop(self.deeper_action_network.parameters(), lr=1e-3)

    def __getstate__(self):
        return {
            'action_network': self.action_network,
            'value_network': self.value_network,
            'ppo': self.ppo,
            'deeper_action_network': self.deeper_action_network,
            'deeper_value_network': self.deeper_value_network,
            'actor_opt': self.actor_opt,
            'value_opt': self.value_opt,
            'deeper_actor_opt': self.deeper_actor_opt,
            'deeper_value_opt': self.deeper_value_opt
        }

    def __setstate__(self, state):
        self.action_network = copy.deepcopy(state['action_network'])
        self.value_network = copy.deepcopy(state['value_network'])
        self.ppo = copy.deepcopy(state['ppo'])
        self.deeper_action_network = copy.deepcopy(state['deeper_action_network'])
        self.deeper_value_network = copy.deepcopy(state['deeper_value_network'])
        self.actor_opt = copy.deepcopy(state['actor_opt'])
        self.value_opt = copy.deepcopy(state['value_opt'])
        self.deeper_value_opt = copy.deepcopy(state['deeper_value_opt'])
        self.deeper_actor_opt = copy.deepcopy(state['deeper_actor_opt'])

    def duplicate(self):
        new_agent = DeepProLoNet()
        new_agent.__setstate__(self.__getstate__())
        return new_agent
