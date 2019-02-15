# Created by Andrew Silva, andrew.silva@gatech.edu
import torch.nn as nn
import torch


class ProLoNet(nn.Module):
    def __init__(self, input_dim, weights, comparators, leaves, alpha=1.0, is_value=False):
        super(ProLoNet, self).__init__()
        """
        Initialize the ProLoNet, taking in premade weights for inputs to comparators and sigmoids
        :param leaves: truple of [[left turn indices], [right turn indices], [final_probs]]
        """
        self.layers = nn.ParameterList()
        self.comparators = nn.ParameterList()
        for comparison_val in comparators:
            self.comparators.append(nn.Parameter(torch.Tensor([comparison_val])))
        for weight_layer in weights:
            new_weights = torch.Tensor(weight_layer)
            new_weights.requires_grad = True
            screening = nn.Parameter(new_weights)
            self.layers.append(screening)

        self.alpha = torch.Tensor([alpha])
        self.alpha.requires_grad = True
        self.alpha = nn.Parameter(self.alpha)
        self.leaf_init_information = leaves
        self.input_dim = input_dim
        left_branches = torch.zeros((len(weights), len(leaves)))
        right_branches = torch.zeros((len(weights), len(leaves)))

        for n in range(0, len(leaves)):
            for i in leaves[n][0]:
                left_branches[i][n] = 1.0
            for j in leaves[n][1]:
                right_branches[j][n] = 1.0

        left_branches.requires_grad = False
        right_branches.requires_grad = False
        self.left_path_sigs = left_branches
        self.right_path_sigs = right_branches

        new_leaves = [leaf[-1] for leaf in leaves]
        labels = torch.Tensor(new_leaves)
        labels.requires_grad = True
        self.action_probs = nn.Parameter(labels)

        self.added_levels = nn.Sequential()

        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.is_value = is_value

    def forward(self, input_data, vis_emb=None, grid_emb=None):
        sig_vals = None
        for layer, comparator in zip(self.layers, self.comparators):
            comp = layer.mul(input_data)
            comp = comp.sum(dim=1)
            comp = comp.sub(comparator)
            comp = comp.mul(self.alpha)
            sig_out = self.sig(comp)
            if sig_vals is None:
                sig_vals = sig_out
            else:
                sig_vals = torch.cat((sig_vals, sig_out))

        sig_vals = sig_vals.view(input_data.size(0), -1)
        one_minus_sig = torch.ones(sig_vals.size()).sub(sig_vals)

        left_path_probs = self.left_path_sigs.t()
        right_path_probs = self.right_path_sigs.t()
        left_path_probs = left_path_probs * sig_vals
        right_path_probs = right_path_probs * one_minus_sig
        left_path_probs = left_path_probs.t()
        right_path_probs = right_path_probs.t()

        # We don't want 0s to ruin leaf probabilities, so replace them with 1s so they don't affect the product
        left_filler = torch.zeros(self.left_path_sigs.size())
        left_filler[self.left_path_sigs == 0] = 1
        right_filler = torch.zeros(self.right_path_sigs.size())
        right_filler[self.right_path_sigs == 0] = 1

        left_path_probs = left_path_probs + left_filler
        right_path_probs = right_path_probs + right_filler

        probs = torch.cat((left_path_probs, right_path_probs), dim=0)
        probs = probs.prod(dim=0)

        probs = probs.view(input_data.size(0), -1)
        actions = probs.mm(self.action_probs).view(-1)
        if len(self.added_levels) > 0:
            seq_in = torch.cat((input_data.view(-1), actions), dim=0)
            actions = self.added_levels(seq_in)
        if not self.is_value:
            return self.softmax(actions)
        else:
            return actions
