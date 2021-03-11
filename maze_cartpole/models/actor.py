"""Contains custom actor (policy) networks"""
from collections import OrderedDict
from typing import Dict, Union, Sequence

import torch
from torch import nn as nn

from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc


class CartPolePolicyNet(nn.Module):
    """Policy Network for MazeProjectTemplate. In it's initial state this is cartpole where we get 4 observations, and
    want to compute the action indicating whether to go left or right.

    :param obs_shapes: The shapes of all observations as a dict.
    :param action_logits_shapes: The shapes of all actions as a dict structure.
    :param non_lin: The nonlinear activation to be used.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], action_logits_shapes: Dict[str, Sequence[int]],
                 non_lin: Union[str, type(nn.Module)]):

        nn.Module.__init__(self)

        # initialize the perception dictionary
        self.perception_dict = OrderedDict()

        # concatenate all observations in dictionary
        self.perception_dict['concat'] = ConcatenationBlock(
            in_keys=['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity'],
            out_keys='concat',
            in_shapes=[obs_shapes['cart_position'], obs_shapes['cart_velocity'],
                       obs_shapes['pole_angle'], obs_shapes['pole_angular_velocity']],
            concat_dim=-1)

        # process concatenated representation with two dense layers
        self.perception_dict['embedding'] = DenseBlock(
            in_keys='concat', in_shapes=self.perception_dict['concat'].out_shapes(),
            hidden_units=[128, 128], non_lin=non_lin, out_keys='embedding'
        )

        # add a linear output block
        self.perception_dict['action'] = LinearOutputBlock(
            in_keys='embedding', out_keys='action',
            in_shapes=self.perception_dict['embedding'].out_shapes(),
            output_units=action_logits_shapes['action'][-1])

        # compile an inference block
        self.perception_net = InferenceBlock(
            in_keys=['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity'],
            out_keys='action',
            in_shapes=[obs_shapes[key] for key in
                       ['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity']],
            perception_blocks=self.perception_dict)

        # initialize model weights
        self.perception_net.apply(make_module_init_normc(1.0))
        self.perception_dict['action'].apply(make_module_init_normc(0.01))

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network.

        :param tensor_dict: The input tensor dictionary.
        :return: The computed output of the network.
        """
        return self.perception_net(tensor_dict)
