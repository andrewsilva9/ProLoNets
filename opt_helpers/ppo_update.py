# Created by Andrew Silva, andrew.silva@gatech.edu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class PPO:
    def __init__(self, actor_critic_arr, two_nets=True):

        lr = 1e-3
        eps = 1e-5
        self.clip_param = 0.2
        self.ppo_epoch = 8
        self.num_mini_batch = 4
        self.value_loss_coef = 0.05
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

        if two_nets:
            self.actor = actor_critic_arr[0]
            self.critic = actor_critic_arr[1]
            if self.actor.input_dim > 100:
                self.actor_opt = optim.RMSprop(self.actor.parameters(), lr=1e-4)
                self.critic_opt = optim.RMSprop(self.critic.parameters(), lr=1e-4)
            elif self.actor.input_dim >= 8:
                self.actor_opt = optim.RMSprop(self.actor.parameters(), lr=1e-3)
                self.critic_opt = optim.RMSprop(self.critic.parameters(), lr=1e-3)
            else:
                self.actor_opt = optim.RMSprop(self.actor.parameters(), lr=1e-2)
                self.critic_opt = optim.RMSprop(self.critic.parameters(), lr=1e-2)
        else:
            self.actor = actor_critic_arr
            self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr, eps=eps)

        self.two_nets = two_nets
        self.epoch_counter = 0

    def cartpole_update(self, rollouts, agent_in, go_deeper=False):
        if self.actor.input_dim < 10:
            batch_size = max(rollouts.step // 32, 1)
            num_iters = rollouts.step // batch_size
        else:
            num_iters = 8
            batch_size = 4
        total_action_loss = torch.Tensor([0])
        total_value_loss = torch.Tensor([0])
        for iteration in range(num_iters):
            total_action_loss = torch.Tensor([0])
            total_value_loss = torch.Tensor([0])
            if go_deeper:
                deep_total_action_loss = torch.Tensor([0])
                deep_total_value_loss = torch.Tensor([0])
            for b in range(batch_size):
                sample = rollouts.sample()
                if not sample:
                    break
                state = sample['state']
                action_probs = sample['action_prob']
                adv_targ = torch.Tensor([sample['advantage']])
                reward = sample['reward']
                old_action_probs = sample['full_prob_vector']
                if np.isnan(adv_targ) or np.isnan(reward) or True in np.isnan(old_action_probs):
                    continue
                action_taken = sample['action_taken']
                hidden_state = sample['hidden_state']
                if hidden_state is not None:
                    new_action_probs, _ = self.actor(*state, hidden_state[0])
                    new_value, _ = self.critic(*state, hidden_state[1])
                else:
                    new_action_probs = self.actor(*state)
                    new_value = self.critic(*state)

                if go_deeper:
                    deep_action_probs = sample['deeper_action_prob']
                    deep_adv = torch.Tensor([sample['deeper_advantage']])
                    deeper_old_probs = sample['deeper_full_prob_vector']

                    new_deep_probs = agent_in.deeper_action_network(*state)
                    new_deep_vals = agent_in.deeper_value_network(*state)
                    deep_dist = Categorical(new_deep_probs)
                    deeper_probs = deep_dist.log_prob(action_taken)
                    deeper_val = new_deep_vals[action_taken.item()]
                    deeper_entropy = deep_dist.entropy().mean() * self.entropy_coef
                    # deep_ratio = torch.div(deeper_probs, deep_action_probs)
                    deep_ratio = torch.nn.functional.kl_div(new_deep_probs, deeper_old_probs, reduction='batchmean')
                    deep_surr1 = deep_ratio.mul(deep_adv).mul(deeper_probs)
                    deep_surr2 = torch.clamp(deep_ratio, 1.0 - self.clip_param,
                                             1.0 + self.clip_param).pow(-1).mul(deep_adv).mul(deeper_probs)
                    deep_action_loss = -torch.min(deep_surr1, deep_surr2).mean()
                    deep_total_action_loss = deep_total_action_loss + deep_action_loss - deeper_entropy

                    deeper_val = deeper_val.view(-1, 1)
                    copy_reward = torch.FloatTensor([reward]).view(-1, 1)
                    deeper_value_loss = F.mse_loss(copy_reward, deeper_val)

                    deep_total_value_loss = deep_total_value_loss + deeper_value_loss

                update_m = Categorical(new_action_probs)
                update_log_probs = update_m.log_prob(action_taken)
                new_value = new_value[action_taken.item()]
                entropy = update_m.entropy().mean() * self.entropy_coef
                # ratio = torch.div(update_log_probs, action_probs)
                ratio = torch.nn.functional.kl_div(new_action_probs, old_action_probs, reduction='batchmean')
                clipped = torch.clamp(ratio, 1.0 - self.clip_param,
                                      1.0 + self.clip_param).mul(adv_targ).mul(update_log_probs)
                ratio = ratio.mul(adv_targ).mul(update_log_probs)
                action_loss = -torch.min(ratio, clipped).mean()
                new_value = new_value.view(-1, 1)
                reward = torch.FloatTensor([reward]).view(-1, 1)
                value_loss = F.mse_loss(reward, new_value)

                total_value_loss = total_value_loss + value_loss
                total_action_loss = total_action_loss + action_loss - entropy
            if total_value_loss != 0:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.zero_grad()
                total_value_loss.backward()
                self.critic_opt.step()
            if total_action_loss != 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.zero_grad()
                total_action_loss.backward()
                self.actor_opt.step()
            if go_deeper:
                if deep_total_value_loss != 0:
                    nn.utils.clip_grad_norm_(agent_in.deeper_value_network.parameters(), self.max_grad_norm)
                    agent_in.deeper_value_opt.zero_grad()
                    deep_total_value_loss.backward()
                    agent_in.deeper_value_opt.step()
                if deep_total_action_loss != 0:
                    nn.utils.clip_grad_norm_(agent_in.deeper_action_network.parameters(), self.max_grad_norm)
                    agent_in.deeper_actor_opt.zero_grad()
                    deep_total_action_loss.backward()
                    agent_in.deeper_actor_opt.step()
        agent_in.deepen_networks()
        agent_in.reset()
        return total_action_loss.item(), total_value_loss.item()

    def sl_updates(self, rollouts, agent_in, heuristic_teacher):
        if self.actor.input_dim < 10:
            batch_size = max(rollouts.step // 32, 1)
            num_iters = rollouts.step // batch_size
        else:
            num_iters = 8
            batch_size = 4
        aggregate_actor_loss = 0
        for iteration in range(num_iters):
            total_action_loss = torch.Tensor([0])
            total_value_loss = torch.Tensor([0])
            for b in range(batch_size):
                sample = rollouts.sample()
                if not sample:
                    break
                state = sample['state']
                reward = sample['reward']
                if np.isnan(reward):
                    continue
                new_action_probs = self.actor(*state).view(1, -1)
                new_value = self.critic(*state)
                label = torch.LongTensor([heuristic_teacher.get_action(state[0].detach().clone().data.cpu().numpy()[0])])
                action_loss = torch.nn.functional.cross_entropy(new_action_probs, label)
                new_value = new_value.view(-1, 1)
                reward = torch.Tensor([reward]).view(-1, 1)
                value_loss = F.mse_loss(reward, new_value)

                total_value_loss = total_value_loss + value_loss
                total_action_loss = total_action_loss + action_loss
            if total_value_loss != 0:
                self.critic_opt.zero_grad()
                total_value_loss.backward()
                self.critic_opt.step()
            if total_action_loss != 0:
                self.actor_opt.zero_grad()
                total_action_loss.backward()
                self.actor_opt.step()
            aggregate_actor_loss += total_action_loss.item()
        aggregate_actor_loss /= float(num_iters*batch_size)
        agent_in.reset()
        return aggregate_actor_loss
