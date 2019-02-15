# Created by Andrew Silva, andrew.silva@gatech.edu
import torch
import torch.nn as nn
from torch.distributions import Categorical
from opt_helpers import replay_buffer, ppo_update
import copy
from agents.heuristic_agent import CartPoleHeuristic, LunarHeuristic, \
    StarCraftMacroHeuristic, StarCraftMicroHeuristic


class BaselineFCNet(nn.Module):
    def __init__(self, input_dim, is_value=False, output_dim=2):
        super(BaselineFCNet, self).__init__()
        self.lin1 = nn.Linear(input_dim, input_dim)
        self.lin2 = nn.Linear(input_dim, input_dim)
        self.lin3 = nn.Linear(input_dim, output_dim)
        self.sig = nn.ReLU()
        self.input_dim = input_dim
        if input_dim == 4:
            self.lin2 = nn.Sequential(
                nn.Linear(input_dim, input_dim),
            )
        elif input_dim == 8:
            self.lin2 = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
            )
        elif input_dim > 10:
            self.lin2 = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim),
            )
        self.softmax = nn.Softmax(dim=0)
        self.is_value = is_value

    def forward(self, input_data):
        act_out = self.lin3(self.sig(self.lin2(self.sig(self.lin1(input_data))))).view(-1)
        if self.is_value:
            return act_out
        else:
            return self.softmax(act_out)


class FCNet:
    def __init__(self,
                 bot_name='FCNet',
                 input_dim=4,
                 output_dim=2,
                 sl_init=False):
        self.bot_name = bot_name
        self.sl_init = sl_init

        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        self.action_network = BaselineFCNet(input_dim=input_dim,
                                            output_dim=output_dim,
                                            is_value=False)
        self.value_network = BaselineFCNet(input_dim=input_dim,
                                           output_dim=output_dim,
                                           is_value=True)
        if self.sl_init:
            if input_dim == 4:
                self.teacher = CartPoleHeuristic()
                self.action_loss_threshold = 250
            elif input_dim == 8:
                self.teacher = LunarHeuristic()
                self.action_loss_threshold = 350
            elif input_dim == 32:
                self.teacher = StarCraftMicroHeuristic()
                self.action_loss_threshold = 500
            elif input_dim > 100:
                self.teacher = StarCraftMacroHeuristic()
                self.action_loss_threshold = 1000
            self.bot_name += '_SLtoRL_'
        self.ppo = ppo_update.PPO([self.action_network, self.value_network], two_nets=True)
        self.actor_opt = torch.optim.RMSprop(self.action_network.parameters())
        self.value_opt = torch.optim.RMSprop(self.value_network.parameters())

        self.last_state = [0, 0, 0, 0]
        self.last_action = 0
        self.last_action_probs = torch.Tensor([0])
        self.last_value_pred = torch.Tensor([[0, 0]])
        self.last_deep_action_probs = torch.Tensor([0])
        self.last_deep_value_pred = torch.Tensor([[0, 0]])
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
        if self.sl_init:
            action_loss = self.ppo.sl_updates(self.replay_buffer, self, self.teacher)
            value_loss = action_loss
            if self.num_steps >= self.action_loss_threshold:
                self.sl_init = False
        else:
            value_loss, action_loss = self.ppo.cartpole_update(self.replay_buffer, self)
        bot_name = '../txts/' + self.bot_name + str(num_processes) + '_processes'
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

    def deepen_networks(self):
        pass

    def save(self, fn='last'):
        checkpoint = dict()
        checkpoint['actor'] = self.action_network.state_dict()
        checkpoint['value'] = self.value_network.state_dict()
        torch.save(checkpoint, fn+self.bot_name+'.pth.tar')

    def load(self, fn='last'):
        fn = fn + self.bot_name + '.pth.tar'
        model_checkpoint = torch.load(fn, map_location='cpu')
        actor_data = model_checkpoint['actor']
        value_data = model_checkpoint['value']
        self.action_network.load_state_dict(actor_data)
        self.value_network.load_state_dict(value_data)

    def __getstate__(self):
        return {
            # 'replay_buffer': self.replay_buffer,
            'action_network': self.action_network,
            'value_network': self.value_network,
            'ppo': self.ppo,
            'actor_opt': self.actor_opt,
            'value_opt': self.value_opt
        }

    def __setstate__(self, state):
        self.action_network = copy.deepcopy(state['action_network'])
        self.value_network = copy.deepcopy(state['value_network'])
        self.ppo = copy.deepcopy(state['ppo'])
        self.actor_opt = copy.deepcopy(state['actor_opt'])
        self.value_opt = copy.deepcopy(state['value_opt'])

    def duplicate(self):
        new_agent = FCNet()
        new_agent.__setstate__(self.__getstate__())
        return new_agent
