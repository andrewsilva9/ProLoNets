# Created by Andrew Silva, andrew.silva@gatech.edu
import numpy as np
from agents.prolonet import ProLoNet
import torch.nn as nn
import copy
import torch


def init_cart_nets(distribution):
    dim_in = 4
    dim_out = 2
    w1 = np.zeros(dim_in)
    w1[0] = 1  # cart position
    c1 = -1  # > -1

    w2 = np.zeros(dim_in)
    w2[0] = -1  # negative position
    c2 = -1  # < 1  (so if positive < 4)

    w3 = np.zeros(dim_in)
    w3[2] = -1  # pole angle
    c3 = 0  # < 0

    w4 = np.zeros(dim_in)
    w4[2] = -1
    c4 = 0  # < 0

    w5 = np.zeros(dim_in)
    w5[2] = -1
    c5 = 0  # < 0

    w6 = np.zeros(dim_in)
    w6[1] = -1  # cart velocity
    c6 = 0  # < 0

    w7 = np.zeros(dim_in)
    w7[1] = 1  # cart velocity
    c7 = 0  # > 0

    w8 = np.zeros(dim_in)
    w8[3] = 1  # pole rate
    c8 = 0  # > 0

    w9 = np.zeros(dim_in)
    w9[2] = -1
    c9 = 0

    w10 = np.zeros(dim_in)
    w10[3] = -1
    c10 = 0

    w11 = np.zeros(dim_in)
    w11[2] = -1
    c11 = 0

    init_weights = [
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9,
        w10,
        w11,
    ]
    init_comparators = [
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
    ]
    if distribution == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif distribution == 'soft_hot':
        leaf_base_init_val = 0.1 / dim_out
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * 2

    l1 = [[], [0, 2], leaf_base.copy()]
    l1[-1][1] = leaf_target_init_val  # Right

    l2 = [[0, 1, 3], [], leaf_base.copy()]
    l2[-1][0] = leaf_target_init_val  # Left

    l3 = [[0, 1], [3], leaf_base.copy()]
    l3[-1][1] = leaf_target_init_val  # Right

    l4 = [[0, 4], [1], leaf_base.copy()]
    l4[-1][0] = leaf_target_init_val  # Left

    l5 = [[2, 5, 7], [0], leaf_base.copy()]
    l5[-1][1] = leaf_target_init_val  # Right

    l6 = [[2, 5], [0, 7], leaf_base.copy()]
    l6[-1][0] = leaf_target_init_val  # Left

    l7 = [[2, 8], [0, 5], leaf_base.copy()]
    l7[-1][0] = leaf_target_init_val  # Left

    l8 = [[2], [0, 5, 8], leaf_base.copy()]
    l8[-1][1] = leaf_target_init_val  # Right

    l9 = [[0, 6, 9], [1, 4], leaf_base.copy()]
    l9[-1][0] = leaf_target_init_val  # Left

    l10 = [[0, 6], [1, 4, 9], leaf_base.copy()]
    l10[-1][1] = leaf_target_init_val  # Right

    l11 = [[0, 10], [1, 4, 6], leaf_base.copy()]
    l11[-1][0] = leaf_target_init_val  # Left

    l12 = [[0], [1, 4, 6, 10], leaf_base.copy()]
    l12[-1][1] = leaf_target_init_val  # Right

    init_leaves = [
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
        l12,
    ]
    action_network = ProLoNet(input_dim=dim_in,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1,
                              is_value=False)
    value_network = ProLoNet(input_dim=dim_in,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1,
                             is_value=True)
    return action_network, value_network


def add_level(pro_lo_net, split_noise_scale=0.2, method='parent'):
    old_weights = pro_lo_net.layers  # Get the weights out
    new_weights = [weight.detach().clone().data.cpu().numpy() for weight in old_weights]

    old_comparators = pro_lo_net.comparators  # get the comparator values out
    new_comparators = [comp.detach().clone().data.cpu().numpy()[0] for comp in
                       old_comparators]
    old_leaf_information = pro_lo_net.leaf_init_information  # get the leaf init info out

    if method == 'fc':
        new_network = ProLoNet(input_dim=pro_lo_net.input_dim, weights=new_weights, comparators=new_comparators,
                               leaves=old_leaf_information, alpha=pro_lo_net.alpha.item(), is_value=pro_lo_net.is_value)

        new_network.added_levels = copy.deepcopy(pro_lo_net.added_levels)
        len_levels = len(new_network.added_levels)
        dim_out = pro_lo_net.action_probs.size(-1)
        if len_levels > 1:
            old_dim_in = 64
        else:
            old_dim_in = pro_lo_net.input_dim + dim_out

        if len_levels == 0:
            dim_in = pro_lo_net.input_dim + dim_out
        else:
            dim_in = 64
            new_network.added_levels[-1] = nn.Linear(old_dim_in, 64)
            new_network.added_levels.add_module('relu_'+str(len_levels), nn.ReLU())

        new_network.added_levels.add_module('linear_'+str(len_levels), nn.Linear(dim_in, dim_out))
    else:

        new_leaf_information = []

        for leaf_index in range(len(old_leaf_information)):
            prior_leaf = pro_lo_net.action_probs[leaf_index].detach().clone().cpu().numpy()
            leaf_information = old_leaf_information[leaf_index]
            left_path = leaf_information[0]
            right_path = leaf_information[1]
            # This hideousness is to handle empty sequences... get the index of the node this used to split at
            weight_index = max(
                max(max(left_path,
                        [-1])
                    ),
                max(max(right_path,
                        [-1])
                    )
            )

            new_weight = np.random.normal(scale=split_noise_scale,
                                          size=old_weights[weight_index].size()[0])
            new_comparator = np.random.normal(scale=split_noise_scale,
                                              size=old_comparators[weight_index].size()[0])
            if method == 'parent':
                new_weight += new_weights[weight_index]
                new_comparator += new_comparators[weight_index]
            new_weights.append(new_weight)  # Add it to the list of nodes
            new_comparators.extend(new_comparator)  # Add it to the list of nodes

            new_node_ind = len(new_weights) - 1  # Remember where we put it

            # Create our two new leaves
            new_leaf1 = np.random.normal(scale=split_noise_scale, size=prior_leaf.shape)
            new_leaf2 = np.random.normal(scale=split_noise_scale, size=prior_leaf.shape)
            if method == 'parent':
                new_leaf1 = new_leaf1 + prior_leaf
                new_leaf2 = new_leaf2 + prior_leaf
            # Create the paths, which are copies of the old path but now with a left / right at the new node
            new_leaf1_left = left_path.copy()
            new_leaf1_right = right_path.copy()
            new_leaf2_left = left_path.copy()
            new_leaf2_right = right_path.copy()
            # Leaf 1 goes left at the new node, leaf 2 goes right
            new_leaf1_left.append(new_node_ind)
            new_leaf2_right.append(new_node_ind)

            new_leaf_information.append([new_leaf1_left, new_leaf1_right, new_leaf1])
            new_leaf_information.append([new_leaf2_left, new_leaf2_right, new_leaf2])

        new_network = ProLoNet(input_dim=pro_lo_net.input_dim, weights=new_weights, comparators=new_comparators,
                               leaves=new_leaf_information, alpha=pro_lo_net.alpha.item(), is_value=pro_lo_net.is_value)
    return new_network


def swap_in_node(network, deeper_network, leaf_index):
    """
    Duplicates the network and returns a new one, where the node at leaf_index as been turned into a splitting node
    with two leaves that are slightly noisy copies of the previous node
    :param network: prolonet in
    :param deeper_network: deeper_network to take the new node / leaves from
    :param leaf_index: index of leaf to turn into a split
    :return: new prolonet (value or normal)
    """
    old_weights = network.layers  # Get the weights out
    old_comparators = network.comparators  # get the comparator values out
    leaf_information = network.leaf_init_information[leaf_index]  # get the old leaf init info out
    left_path = leaf_information[0]
    right_path = leaf_information[1]
    if deeper_network is not None:
        deeper_weights = [weight.detach().clone().data.cpu().numpy() for weight in deeper_network.layers]

        deeper_comparators = [comp.detach().clone().data.cpu().numpy()[0] for comp in
                              deeper_network.comparators]

        deeper_leaf_info = deeper_network.leaf_init_information[leaf_index*2]
        deeper_left_path = deeper_leaf_info[0]
        deeper_right_path = deeper_leaf_info[1]
        deeper_weight_index = max(
            max(max(deeper_left_path,
                    [-1])
                ),
            max(max(deeper_right_path,
                    [-1])
                )
        )

        # Make a new weight vector, mostly the same as the old one

        new_weight = deeper_weights[deeper_weight_index]
        new_comparator = deeper_comparators[deeper_weight_index]
        new_leaf1 = deeper_network.action_probs[leaf_index * 2].detach().clone().data.cpu().numpy()
        new_leaf2 = deeper_network.action_probs[leaf_index * 2 + 1].detach().clone().data.cpu().numpy()
    else:
        new_weight = np.random.normal(scale=0.2,
                                      size=old_weights[0].size()[0])
        new_comparator = np.random.normal(scale=0.2,
                                          size=old_comparators[0].size()[0])[0]
        new_leaf1 = np.random.normal(scale=0.2,
                                     size=network.action_probs[leaf_index].size()[0])
        new_leaf2 = np.random.normal(scale=0.2,
                                     size=network.action_probs[leaf_index].size()[0])

    new_weights = [weight.detach().clone().data.cpu().numpy() for weight in old_weights]
    new_weights.append(new_weight)  # Add it to the list of nodes
    new_comparators = [comp.detach().clone().data.cpu().numpy()[0] for comp in old_comparators]
    new_comparators.append(new_comparator)  # Add it to the list of nodes

    new_node_ind = len(new_weights) - 1  # Remember where we put it

    # Create the paths, which are copies of the old path but now with a left / right at the new node
    new_leaf1_left = left_path.copy()
    new_leaf1_right = right_path.copy()
    new_leaf2_left = left_path.copy()
    new_leaf2_right = right_path.copy()
    # Leaf 1 goes left at the new node, leaf 2 goes right
    new_leaf1_left.append(new_node_ind)
    new_leaf2_right.append(new_node_ind)

    new_leaf_information = network.leaf_init_information
    for index, leaf_prob_vec in enumerate(network.action_probs):  # Copy over the learned leaf weight
        new_leaf_information[index][-1] = leaf_prob_vec.detach().clone().data.cpu().numpy()
    new_leaf_information.append([new_leaf1_left, new_leaf1_right, new_leaf1])
    new_leaf_information.append([new_leaf2_left, new_leaf2_right, new_leaf2])
    # Remove the old leaf
    del new_leaf_information[leaf_index]
    new_network = ProLoNet(input_dim=network.input_dim, weights=new_weights, comparators=new_comparators,
                           leaves=new_leaf_information, alpha=network.alpha.item(), is_value=network.is_value)
    return new_network


def init_lander_nets(distribution):
    dim_in = 8
    dim_out = 4
    w0 = np.zeros(dim_in)
    w0[1] = -1
    c0 = -1.1  # < 1.1

    w1 = np.zeros(dim_in)
    w1[3] = -1
    c1 = 0.2  # < -0.2

    w2 = np.zeros(dim_in)
    w2[5] = 1
    c2 = 0.1  # > 0.1

    w3 = np.zeros(dim_in)
    w3[5] = -1
    c3 = -0.1  # < 0.1

    w4 = np.zeros(dim_in)
    w4[6] = 1
    w4[7] = 1
    c4 = 1.2  # both 6 & 7 == 1

    w5 = np.zeros(dim_in)
    w5[5] = -1
    c5 = 0.1  # < -0.1

    w6 = np.zeros(dim_in)
    w6[6] = 1
    w6[7] = 1
    c6 = 1.2  # both 6 & 7 == 1

    w7 = np.zeros(dim_in)
    w7[6] = 1
    w7[7] = 1
    c7 = 1.2  # both 6 & 7 == 1

    w8 = np.zeros(dim_in)
    w8[0] = 1
    c8 = 0.2  # > 0

    w9 = np.zeros(dim_in)
    w9[6] = 1
    w9[7] = 1
    c9 = 1.2  # both 6 & 7 == 1

    w10 = np.zeros(dim_in)
    w10[0] = 1
    c10 = 0.2

    w11 = np.zeros(dim_in)
    w11[5] = 1
    c11 = -0.1  # > -0.1

    w12 = np.zeros(dim_in)
    w12[0] = -1
    c12 = 0.2  # < -0.2

    w13 = np.zeros(dim_in)
    w13[0] = -1
    c13 = 0.2  # < -0.2

    init_weights = [
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9,
        w10,
        w11,
        w12,
        w13
    ]
    init_comparators = [
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
        c12,
        c13
    ]
    if distribution == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif distribution == 'soft_hot':
        leaf_base_init_val = 0.1 / dim_out
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * dim_out

    l0 = [[0, 1], [3], leaf_base.copy()]
    l0[-1][3] = leaf_target_init_val

    l1 = [[0, 4], [1], leaf_base.copy()]
    l1[-1][0] = leaf_target_init_val

    l2 = [[6], [0, 2], leaf_base.copy()]
    l2[-1][0] = leaf_target_init_val

    l3 = [[], [0, 2, 6], leaf_base.copy()]
    l3[-1][3] = leaf_target_init_val

    l4 = [[0, 1, 3, 7], [], leaf_base.copy()]
    l4[-1][0] = leaf_target_init_val

    l5 = [[0, 8], [1, 4], leaf_base.copy()]
    l5[-1][1] = leaf_target_init_val

    l6 = [[2, 5, 9], [0], leaf_base.copy()]
    l6[-1][0] = leaf_target_init_val

    l7 = [[2, 5], [0, 9], leaf_base.copy()]
    l7[-1][1] = leaf_target_init_val

    l8 = [[2, 10], [0, 5], leaf_base.copy()]
    l8[-1][1] = leaf_target_init_val

    l9 = [[0, 1, 3, 11], [7], leaf_base.copy()]
    l9[-1][2] = leaf_target_init_val

    l10 = [[0, 1, 3], [7, 11], leaf_base.copy()]
    l10[-1][1] = leaf_target_init_val

    l11 = [[0, 12], [1, 4, 8], leaf_base.copy()]
    l11[-1][3] = leaf_target_init_val

    l12 = [[0], [1, 4, 8, 12], leaf_base.copy()]
    l12[-1][0] = leaf_target_init_val

    l13 = [[2, 13], [0, 5, 10], leaf_base.copy()]
    l13[-1][3] = leaf_target_init_val

    l14 = [[2], [0, 5, 10, 13], leaf_base.copy()]
    l14[-1][0] = leaf_target_init_val

    init_leaves = [
        l0,
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
        l12,
        l13,
        l14
    ]
    action_network = ProLoNet(input_dim=dim_in,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1,
                              is_value=False)
    value_network = ProLoNet(input_dim=dim_in,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1,
                             is_value=True)
    return action_network, value_network


def init_random_cart_net():
    dim_in = 4
    output_dim = 2
    w1 = np.random.normal(0, 0.1, dim_in)
    c1 = np.random.normal(0, 0.1, 1)[0]

    w2 = np.random.normal(0, 0.1, dim_in)
    c2 = np.random.normal(0, 0.1, 1)[0]

    w3 = np.random.normal(0, 0.1, dim_in)
    c3 = np.random.normal(0, 0.1, 1)[0]

    w4 = np.random.normal(0, 0.1, dim_in)
    c4 = np.random.normal(0, 0.1, 1)[0]

    w5 = np.random.normal(0, 0.1, dim_in)
    c5 = np.random.normal(0, 0.1, 1)[0]

    w6 = np.random.normal(0, 0.1, dim_in)
    c6 = np.random.normal(0, 0.1, 1)[0]

    w7 = np.random.normal(0, 0.1, dim_in)
    c7 = np.random.normal(0, 0.1, 1)[0]

    w8 = np.random.normal(0, 0.1, dim_in)
    c8 = np.random.normal(0, 0.1, 1)[0]

    w9 = np.random.normal(0, 0.1, dim_in)
    c9 = np.random.normal(0, 0.1, 1)[0]

    w10 = np.random.normal(0, 0.1, dim_in)
    c10 = np.random.normal(0, 0.1, 1)[0]

    w11 = np.random.normal(0, 0.1, dim_in)
    c11 = np.random.normal(0, 0.1, 1)[0]

    init_weights = [
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9,
        w10,
        w11,
    ]
    init_comparators = [
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
    ]

    l1 = [[], [0, 2], np.random.normal(0, 0.1, output_dim)]

    l2 = [[0, 1, 3], [], np.random.normal(0, 0.1, output_dim)]

    l3 = [[0, 1], [3], np.random.normal(0, 0.1, output_dim)]

    l4 = [[0, 4], [1], np.random.normal(0, 0.1, output_dim)]

    l5 = [[2, 5, 7], [0], np.random.normal(0, 0.1, output_dim)]

    l6 = [[2, 5], [0, 7], np.random.normal(0, 0.1, output_dim)]

    l7 = [[2, 8], [0, 5], np.random.normal(0, 0.1, output_dim)]

    l8 = [[2], [0, 5, 8], np.random.normal(0, 0.1, output_dim)]

    l9 = [[0, 6, 9], [1, 4], np.random.normal(0, 0.1, output_dim)]

    l10 = [[0, 6], [1, 4, 9], np.random.normal(0, 0.1, output_dim)]

    l11 = [[0, 10], [1, 4, 6], np.random.normal(0, 0.1, output_dim)]

    l12 = [[0], [1, 4, 6, 10], np.random.normal(0, 0.1, output_dim)]

    init_leaves = [
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
        l12,
    ]
    action_network = ProLoNet(input_dim=dim_in,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1,
                              is_value=False)
    value_network = ProLoNet(input_dim=dim_in,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1,
                             is_value=True)
    return action_network, value_network


def init_random_lander_net():
    dim_in = 8
    dim_out = 4
    w0 = np.random.normal(0, 0.1, dim_in)
    c0 = np.random.normal(0, 0.1, 1)[0]

    w1 = np.random.normal(0, 0.1, dim_in)
    c1 = np.random.normal(0, 0.1, 1)[0]

    w2 = np.random.normal(0, 0.1, dim_in)
    c2 = np.random.normal(0, 0.1, 1)[0]

    w3 = np.random.normal(0, 0.1, dim_in)
    c3 = np.random.normal(0, 0.1, 1)[0]

    w4 = np.random.normal(0, 0.1, dim_in)
    c4 = np.random.normal(0, 0.1, 1)[0]

    w5 = np.random.normal(0, 0.1, dim_in)
    c5 = np.random.normal(0, 0.1, 1)[0]

    w6 = np.random.normal(0, 0.1, dim_in)
    c6 = np.random.normal(0, 0.1, 1)[0]

    w7 = np.random.normal(0, 0.1, dim_in)
    c7 = np.random.normal(0, 0.1, 1)[0]

    w8 = np.random.normal(0, 0.1, dim_in)
    c8 = np.random.normal(0, 0.1, 1)[0]

    w9 = np.random.normal(0, 0.1, dim_in)
    c9 = np.random.normal(0, 0.1, 1)[0]

    w10 = np.random.normal(0, 0.1, dim_in)
    c10 = np.random.normal(0, 0.1, 1)[0]

    w11 = np.random.normal(0, 0.1, dim_in)
    c11 = np.random.normal(0, 0.1, 1)[0]

    w12 = np.random.normal(0, 0.1, dim_in)
    c12 = np.random.normal(0, 0.1, 1)[0]

    w13 = np.random.normal(0, 0.1, dim_in)
    c13 = np.random.normal(0, 0.1, 1)[0]

    init_weights = [
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9,
        w10,
        w11,
        w12,
        w13
    ]
    init_comparators = [
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
        c12,
        c13
    ]

    l0 = [[0, 1], [3], np.random.normal(0, 0.1, dim_out)]

    l1 = [[0, 4], [1], np.random.normal(0, 0.1, dim_out)]

    l2 = [[6], [0, 2], np.random.normal(0, 0.1, dim_out)]

    l3 = [[], [0, 2, 6], np.random.normal(0, 0.1, dim_out)]

    l4 = [[0, 1, 3, 7], [], np.random.normal(0, 0.1, dim_out)]

    l5 = [[0, 8], [1, 4], np.random.normal(0, 0.1, dim_out)]

    l6 = [[2, 5, 9], [0], np.random.normal(0, 0.1, dim_out)]

    l7 = [[2, 5], [0, 9], np.random.normal(0, 0.1, dim_out)]

    l8 = [[2, 10], [0, 5], np.random.normal(0, 0.1, dim_out)]

    l9 = [[0, 1, 3, 11], [7], np.random.normal(0, 0.1, dim_out)]

    l10 = [[0, 1, 3], [7, 11], np.random.normal(0, 0.1, dim_out)]

    l11 = [[0, 12], [1, 4, 8], np.random.normal(0, 0.1, dim_out)]

    l12 = [[0], [1, 4, 8, 12], np.random.normal(0, 0.1, dim_out)]

    l13 = [[2, 13], [0, 5, 10], np.random.normal(0, 0.1, dim_out)]

    l14 = [[2], [0, 5, 10, 13], np.random.normal(0, 0.1, dim_out)]

    init_leaves = [
        l0,
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
        l12,
        l13,
        l14
    ]
    action_network = ProLoNet(input_dim=dim_in,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1,
                              is_value=False)
    value_network = ProLoNet(input_dim=dim_in,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1,
                             is_value=True)
    return action_network, value_network


def init_adversarial_net(adv_type='cart', distribution_in='one_hot', adv_prob=0.05):
    """
    initialize networks intelligently but also wrongly
    with p=adv_prob negate weights
    with p=adv_prob negate comparators
    with p=adv_prob negate leaf probabilities
    :param adv_type: which env is this for: cart, lunar, sc
    :param distribution_in: same as init_cart_nets or init_lander_nets. one_hot, soft_hot, or other...
    :param adv_prob: probability to negate the things above
    :return: actor, critic
    """
    cart_act = None
    if adv_type == 'cart':
        cart_act, cart_value = init_cart_nets(distribution=distribution_in)
    elif adv_type == 'lunar':
        cart_act, cart_value = init_lander_nets(distribution=distribution_in)
    elif adv_type == 'sc':
        cart_act, cart_value = init_sc_nets(distribution_in)
    elif adv_type == 'micro':
        cart_act, cart_value = init_micro_net(distribution=distribution_in)
    if cart_act is None:
        return -1
    old_weights = cart_act.layers  # Get the weights out
    new_weights = [weight.detach().clone().data.cpu().numpy() for weight in old_weights]

    old_comparators = cart_act.comparators  # get the comparator values out
    new_comparators = [comp.detach().clone().data.cpu().numpy()[0] for comp in
                       old_comparators]

    prob_flip = adv_prob
    old_leaf_information = cart_act.leaf_init_information  # get the leaf init info out
    weight_flips = 0
    comp_flips = 0
    leaf_flips = 0
    quarter_weights = len(new_weights)*(prob_flip*2)
    quarter_leaves = len(old_leaf_information)*(prob_flip*2)
    for index in range(len(new_weights)):
        if np.random.random() < prob_flip and weight_flips < quarter_weights:
            new_weights[index] *= -1
            weight_flips += 1
        if np.random.random() < prob_flip and comp_flips < quarter_weights:
            new_comparators[index] *= -1
            comp_flips += 1
    for index in range(len(old_leaf_information)):
        if np.random.random() < prob_flip and leaf_flips < quarter_leaves:
            new_leaf_info = np.array(old_leaf_information[index][-1])*-1
            old_leaf_information[index][-1] = new_leaf_info.tolist()
            leaf_flips += 1

    new_actor = ProLoNet(input_dim=cart_act.input_dim, weights=new_weights, comparators=new_comparators,
                         leaves=old_leaf_information, alpha=cart_act.alpha.item(), is_value=False)
    new_value = ProLoNet(input_dim=cart_act.input_dim, weights=new_weights, comparators=new_comparators,
                         leaves=old_leaf_information, alpha=cart_act.alpha.item(), is_value=True)

    return new_actor, new_value


def init_sc_nets(dist='one_hot'):
    dim_in = 194
    dim_out = 44

    w0 = np.zeros(dim_in)
    w0[10] = 1  # zealot
    w0[22] = 1  # voidray
    c0 = 8  # > 8

    w1 = np.zeros(dim_in)
    w1[45:65] = 1  # Enemy Protoss non-buildings
    w1[82:99] = 1  # Enemy Terran non-buildings
    w1[118:139] = 1  # Enemy Zerg non-buildings
    c1 = 2  # > 4 enemies, potentially under attack?

    w2 = np.zeros(dim_in)
    w2[4] = 1  # idle workers
    c2 = 0.5  # > 0.5

    w3 = np.zeros(dim_in)
    w3[65:82] = 1  # Enemy Protoss non-buildings
    w3[99:118] = 1  # Enemy Terran non-buildings
    w3[139:157] = 1  # Enemy Zerg non-buildings
    c3 = 0  # > 0  # know where some enemy structures are

    w4 = np.zeros(dim_in)
    w4[2] = -1  # negative food capacity
    w4[3] = 1  # plus food used = negative food available
    c4 = -4  # > -4  (so if positive < 4)

    w5 = np.zeros(dim_in)
    w5[9] = -1  # probes
    w5[157] = -1
    c5 = -20  # > -16 == #probes<16

    w6 = np.zeros(dim_in)
    w6[30] = 1  # ASSIMILATOR
    w6[178] = 1
    c6 = 0.5  # > 0.5

    w7 = np.zeros(dim_in)
    w7[38] = 1  # STARGATE
    w7[186] = 1
    c7 = 0.5  # > 1.5

    w8 = np.zeros(dim_in)
    w8[31] = 1  # GATEWAY
    w8[179] = 1
    c8 = 0.5  # > 0.5

    w9 = np.zeros(dim_in)
    w9[22] = 1  # VOIDRAY
    w9[170] = 1
    c9 = 7  # > 7

    w10 = np.zeros(dim_in)
    w10[10] = 1  # zealot
    w10[158] = 1
    c10 = 2  # > 3

    w11 = np.zeros(dim_in)
    w11[34] = 10  # CYBERNETICSCORE
    w11[182] = 10
    c11 = 0.5  # > 0.5

    init_weights = [
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9,
        w10,
        w11,
    ]
    init_comparators = [
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
    ]
    if dist == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif dist == 'soft_hot':
        leaf_base_init_val = 0.1/(max(dim_out-1, 1))
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0/dim_out
        leaf_target_init_val = 1.0/dim_out
    leaf_base = [leaf_base_init_val] * dim_out

    l0 = [[0, 1], [], leaf_base.copy()]
    l0[-1][41] = leaf_target_init_val  # Defend
    l0[-1][39] = leaf_target_init_val

    l1 = [[2], [0], leaf_base.copy()]
    l1[-1][40] = leaf_target_init_val  # Mine

    l2 = [[0, 1], [3], leaf_base.copy()]
    l2[-1][39] = leaf_target_init_val  # Attack

    l3 = [[0], [1, 3], leaf_base.copy()]
    l3[-1][42] = leaf_target_init_val  # Scout

    l4 = [[4], [0, 2], leaf_base.copy()]
    l4[-1][1] = leaf_target_init_val  # Pylon

    l5 = [[5], [0, 2, 4], leaf_base.copy()]
    l5[-1][16] = leaf_target_init_val  # Probe

    l6 = [[], [0, 2, 4, 5, 6, 8], leaf_base.copy()]
    l6[-1][3] = leaf_target_init_val  # Gateway

    l7 = [[6, 7, 9], [0, 2, 4, 5], leaf_base.copy()]
    l7[-1][39] = leaf_target_init_val  # Attack

    l8 = [[6, 7], [0, 2, 4, 5, 9], leaf_base.copy()]
    l8[-1][29] = leaf_target_init_val  # Voidray

    l9 = [[6, 10], [0, 2, 4, 5, 7], leaf_base.copy()]
    l9[-1][40] = leaf_target_init_val  # Mine (Get vespene)

    l10 = [[6], [0, 2, 4, 5, 7], leaf_base.copy()]
    l10[-1][10] = leaf_target_init_val  # Stargate

    l11 = [[8], [0, 2, 4, 5, 6, 10], leaf_base.copy()]
    l11[-1][17] = leaf_target_init_val  # Zealot

    l12 = [[8, 10, 11], [0, 2, 4, 5, 6], leaf_base.copy()]
    l12[-1][2] = leaf_target_init_val  # Assimilator

    l13 = [[8, 10], [0, 2, 4, 5, 6, 11], leaf_base.copy()]
    l13[-1][6] = leaf_target_init_val  # Cybernetics Core

    init_leaves = [
        l1,
        l2,
        l3,
        l4,
        l5,
        l10,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
        l12,
        l13,
    ]
    actor = ProLoNet(input_dim=dim_in,
                     weights=init_weights,
                     comparators=init_comparators,
                     leaves=init_leaves,
                     alpha=1,
                     is_value=False)
    critic = ProLoNet(input_dim=dim_in,
                      weights=init_weights,
                      comparators=init_comparators,
                      leaves=init_leaves,
                      alpha=1,
                      is_value=True)
    return actor, critic


def init_random_sc_net():
    dim_in = 194
    dim_out = 44
    w0 = np.random.normal(0, 0.1, dim_in)
    c0 = np.random.normal(0, 0.1, 1)[0]

    w1 = np.random.normal(0, 0.1, dim_in)
    c1 = np.random.normal(0, 0.1, 1)[0]

    w2 = np.random.normal(0, 0.1, dim_in)
    c2 = np.random.normal(0, 0.1, 1)[0]

    w3 = np.random.normal(0, 0.1, dim_in)
    c3 = np.random.normal(0, 0.1, 1)[0]

    w4 = np.random.normal(0, 0.1, dim_in)
    c4 = np.random.normal(0, 0.1, 1)[0]

    w5 = np.random.normal(0, 0.1, dim_in)
    c5 = np.random.normal(0, 0.1, 1)[0]

    w6 = np.random.normal(0, 0.1, dim_in)
    c6 = np.random.normal(0, 0.1, 1)[0]

    w7 = np.random.normal(0, 0.1, dim_in)
    c7 = np.random.normal(0, 0.1, 1)[0]

    w8 = np.random.normal(0, 0.1, dim_in)
    c8 = np.random.normal(0, 0.1, 1)[0]

    w9 = np.random.normal(0, 0.1, dim_in)
    c9 = np.random.normal(0, 0.1, 1)[0]

    w10 = np.random.normal(0, 0.1, dim_in)
    c10 = np.random.normal(0, 0.1, 1)[0]

    w11 = np.random.normal(0, 0.1, dim_in)
    c11 = np.random.normal(0, 0.1, 1)[0]

    w12 = np.random.normal(0, 0.1, dim_in)
    c12 = np.random.normal(0, 0.1, 1)[0]

    init_weights = [
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9,
        w10,
        w11,
        w12,
    ]
    init_comparators = [
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
        c12,
    ]

    l1 = [[2], [0], np.random.normal(0, 0.1, dim_out)]

    l2 = [[0, 1], [3], np.random.normal(0, 0.1, dim_out)]

    l3 = [[0], [1, 3], np.random.normal(0, 0.1, dim_out)]

    l4 = [[4], [0, 2], np.random.normal(0, 0.1, dim_out)]

    l5 = [[5], [0, 2, 4], np.random.normal(0, 0.1, dim_out)]

    l6 = [[], [0, 2, 4, 5, 6, 8], np.random.normal(0, 0.1, dim_out)]

    l7 = [[6, 7, 9], [0, 2, 4, 5], np.random.normal(0, 0.1, dim_out)]

    l8 = [[6, 7], [0, 2, 4, 5, 9], np.random.normal(0, 0.1, dim_out)]

    l9 = [[6, 10], [0, 2, 4, 5, 7], np.random.normal(0, 0.1, dim_out)]

    l10 = [[6], [0, 2, 4, 5, 7, 10], np.random.normal(0, 0.1, dim_out)]

    l11 = [[8], [0, 2, 4, 5, 6, 11], np.random.normal(0, 0.1, dim_out)]

    l12 = [[8, 11, 12], [0, 2, 4, 5, 6], np.random.normal(0, 0.1, dim_out)]

    l13 = [[8, 11], [0, 2, 4, 5, 6, 12], np.random.normal(0, 0.1, dim_out)]

    init_leaves = [
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
        l12,
        l13,
    ]
    actor = ProLoNet(input_dim=dim_in,
                     weights=init_weights,
                     comparators=init_comparators,
                     leaves=init_leaves,
                     alpha=1,
                     is_value=False)
    critic = ProLoNet(input_dim=dim_in,
                      weights=init_weights,
                      comparators=init_comparators,
                      leaves=init_leaves,
                      alpha=1,
                      is_value=True)
    return actor, critic


def save_prolonet(fn, model):
    checkpoint = dict()
    mdl_data = dict()
    mdl_data['weights'] = model.layers
    mdl_data['comparators'] = model.comparators
    mdl_data['leaf_init_information'] = model.leaf_init_information
    mdl_data['action_probs'] = model.action_probs
    mdl_data['alpha'] = model.alpha
    mdl_data['input_dim'] = model.input_dim
    mdl_data['is_value'] = model.is_value
    checkpoint['model_data'] = mdl_data
    torch.save(checkpoint, fn)


def load_prolonet(fn):
    model_checkpoint = torch.load(fn, map_location='cpu')
    model_data = model_checkpoint['model_data']
    init_weights = [weight.detach().clone().data.cpu().numpy() for weight in model_data['weights']]
    init_comparators = [comp.item() for comp in model_data['comparators']]

    new_model = ProLoNet(input_dim=model_data['input_dim'],
                         weights=init_weights,
                         comparators=init_comparators,
                         leaves=model_data['leaf_init_information'],
                         alpha=model_data['alpha'].item(),
                         is_value=model_data['is_value'])
    new_model.action_probs = model_data['action_probs']
    return new_model


def init_shallow_cart_nets(distribution):
    dim_in = 4
    dim_out = 2
    w0 = np.zeros(dim_in)
    w0[2] = -1  # pole angle
    c0 = 0  # < 0
    init_weights = [
        w0
    ]
    init_comparators = [
        c0
    ]
    if distribution == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif distribution == 'soft_hot':
        leaf_base_init_val = 0.1 / dim_out
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * dim_out

    l0 = [[], [0], leaf_base.copy()]
    l0[-1][1] = leaf_target_init_val  # Right

    l1 = [[0], [0], leaf_base.copy()]
    l1[-1][0] = leaf_target_init_val  # Right

    init_leaves = [
        l0,
        l1
    ]
    action_network = ProLoNet(input_dim=dim_in,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1,
                              is_value=False)
    value_network = ProLoNet(input_dim=dim_in,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1,
                             is_value=True)
    return action_network, value_network


def init_random_micro_net():
    dim_in = 32
    dim_out = 10
    w0 = np.random.normal(0, 0.1, dim_in)
    c0 = np.random.normal(0, 0.1, 1)[0]

    w1 = np.random.normal(0, 0.1, dim_in)
    c1 = np.random.normal(0, 0.1, 1)[0]

    w2 = np.random.normal(0, 0.1, dim_in)
    c2 = np.random.normal(0, 0.1, 1)[0]

    w3 = np.random.normal(0, 0.1, dim_in)
    c3 = np.random.normal(0, 0.1, 1)[0]

    w4 = np.random.normal(0, 0.1, dim_in)
    c4 = np.random.normal(0, 0.1, 1)[0]

    w5 = np.random.normal(0, 0.1, dim_in)
    c5 = np.random.normal(0, 0.1, 1)[0]

    w6 = np.random.normal(0, 0.1, dim_in)
    c6 = np.random.normal(0, 0.1, 1)[0]

    w7 = np.random.normal(0, 0.1, dim_in)
    c7 = np.random.normal(0, 0.1, 1)[0]

    w8 = np.random.normal(0, 0.1, dim_in)
    c8 = np.random.normal(0, 0.1, 1)[0]

    w9 = np.random.normal(0, 0.1, dim_in)
    c9 = np.random.normal(0, 0.1, 1)[0]

    init_weights = [
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9
    ]
    init_comparators = [
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9
    ]
    l0 = [[0], [], np.random.normal(0, 0.1, dim_out)]

    l1 = [[1], [0], np.random.normal(0, 0.1, dim_out)]

    l2 = [[2], [0, 1], np.random.normal(0, 0.1, dim_out)]

    l3 = [[3], [0, 1, 2], np.random.normal(0, 0.1, dim_out)]
    l4 = [[4], [0, 1, 2, 3], np.random.normal(0, 0.1, dim_out)]

    l5 = [[5, 6], [0, 1, 2, 3, 4], np.random.normal(0, 0.1, dim_out)]

    l6 = [[7], [0, 1, 2, 3, 4, 5], np.random.normal(0, 0.1, dim_out)]

    l7 = [[5, 8], [0, 1, 2, 3, 4, 6], np.random.normal(0, 0.1, dim_out)]

    l8 = [[5], [0, 1, 2, 3, 4, 6, 8], np.random.normal(0, 0.1, dim_out)]

    l9 = [[9], [0, 1, 2, 3, 4, 5, 7], np.random.normal(0, 0.1, dim_out)]

    l10 = [[], [0, 1, 2, 3, 4, 5, 7, 9], np.random.normal(0, 0.1, dim_out)]

    init_leaves = [
        l0,
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
        l8,
        l9,
        l10
    ]
    action_network = ProLoNet(input_dim=dim_in,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1,
                              is_value=False)
    value_network = ProLoNet(input_dim=dim_in,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1,
                             is_value=True)
    return action_network, value_network


def init_micro_net(distribution='one_hot'):
    dim_in = 32
    dim_out = 10
    w0 = np.zeros(dim_in)
    w0[0] = 1
    w0[4] = -1
    c0 = 3.5

    w1 = np.zeros(dim_in)
    w1[0] = -1
    w1[4] = 1
    c1 = 3.5

    w2 = np.zeros(dim_in)
    w2[1] = 1
    w2[5] = -1
    c2 = 3.5

    w3 = np.zeros(dim_in)
    w3[1] = -1
    w3[5] = 1
    c3 = 3.5

    w4 = np.zeros(dim_in)
    w4[14] = 1  # zergling health
    c4 = 0

    w5 = np.zeros(dim_in)
    w5[1] = 1  # y position
    c5 = 30  # > 30

    w6 = np.zeros(dim_in)
    w6[0] = -1  # y position
    c6 = -20  # < 20

    w7 = np.zeros(dim_in)
    w7[1] = 1  # y position
    c7 = 18  # < 20

    w8 = np.zeros(dim_in)
    w8[0] = 1  # y position
    c8 = 40  # < 20

    w9 = np.zeros(dim_in)
    w9[0] = -1  # y position
    c9 = -40  # < 20

    init_weights = [
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9
    ]
    init_comparators = [
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9
    ]

    if distribution == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif distribution == 'soft_hot':
        leaf_base_init_val = 0.1 / dim_out
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * dim_out

    l0 = [[0], [], leaf_base.copy()]
    l0[-1][3] = leaf_target_init_val  # west

    l1 = [[1], [0], leaf_base.copy()]
    l1[-1][1] = leaf_target_init_val  # east

    l2 = [[2], [0, 1], leaf_base.copy()]
    l2[-1][2] = leaf_target_init_val  # South

    l3 = [[3], [0, 1, 2], leaf_base.copy()]
    l3[-1][0] = leaf_target_init_val  # north

    l4 = [[4], [0, 1, 2, 3], leaf_base.copy()]
    l4[-1][4] = leaf_target_init_val  # Attack

    l5 = [[5, 6], [0, 1, 2, 3, 4], leaf_base.copy()]
    l5[-1][2] = leaf_target_init_val  # S

    l6 = [[7], [0, 1, 2, 3, 4, 5], leaf_base.copy()]
    l6[-1][2] = leaf_target_init_val  # S

    l7 = [[5, 8], [0, 1, 2, 3, 4, 6], leaf_base.copy()]
    l7[-1][0] = leaf_target_init_val  # N

    l8 = [[5], [0, 1, 2, 3, 4, 6, 8], leaf_base.copy()]
    l8[-1][3] = leaf_target_init_val  # N

    l9 = [[9], [0, 1, 2, 3, 4, 5, 7], leaf_base.copy()]
    l9[-1][1] = leaf_target_init_val  # E

    l10 = [[], [0, 1, 2, 3, 4, 5, 7, 9], leaf_base.copy()]
    l10[-1][0] = leaf_target_init_val  # N

    init_leaves = [
        l0,
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
        l8,
        l9,
        l10
    ]
    action_network = ProLoNet(input_dim=dim_in,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1,
                              is_value=False)
    value_network = ProLoNet(input_dim=dim_in,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1,
                             is_value=True)
    return action_network, value_network
