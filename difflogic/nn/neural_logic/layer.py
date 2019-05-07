#! /usr/bin/env python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implement Neural Logic Layers and Machines."""

import torch
import torch.nn as nn

from jacinle.logging import get_logger

from .modules.dimension import Expander, Reducer, Permutation
from .modules.neural_logic import LogicInference

__all__ = ['LogicLayer', 'LogicMachine']

logger = get_logger(__file__)


def _get_tuple_n(x, n, tp):
  """Get a length-n list of type tp."""
  assert tp is not list
  if isinstance(x, tp):
    x = [x,] * n
  assert len(x) == n, 'Parameters should be {} or list of N elements.'.format(
      tp)
  for i in x:
    assert isinstance(i, tp), 'Elements of list should be {}.'.format(tp)
  return x


class LogicLayer(nn.Module):
  """Logic Layers do one-step differentiable logic deduction.

  The predicates grouped by their number of variables. The inter group deduction
  is done by expansion/reduction, the intra group deduction is done by logic
  model.

  Args:
    breadth: The breadth of the logic layer.
    input_dims: the number of input channels of each input group, should consist
                with the inputs. use dims=0 and input=None to indicate no input
                of that group.
    output_dims: the number of output channels of each group, could
                 use a single value.
    logic_hidden_dim: The hidden dim of the logic model.
    exclude_self: Not allow multiple occurrence of same variable when
                  being True.
    residual: Use residual connections when being True.
  """

  def __init__(
      self,
      breadth,
      input_dims,
      output_dims,
      logic_hidden_dim,
      exclude_self=True,
      residual=False,
  ):
    super().__init__()
    assert breadth > 0, 'Does not support breadth <= 0.'
    if breadth > 3:
      logger.warn(
          'Using LogicLayer with breadth > 3 may cause speed and memory issue.')

    self.max_order = breadth
    self.residual = residual

    input_dims = _get_tuple_n(input_dims, self.max_order + 1, int)
    output_dims = _get_tuple_n(output_dims, self.max_order + 1, int)

    self.logic, self.dim_perms, self.dim_expanders, self.dim_reducers = [
        nn.ModuleList() for _ in range(4)
    ]
    for i in range(self.max_order + 1):
      # collect current_dim from group i-1, i and i+1.
      current_dim = input_dims[i]
      if i > 0:
        expander = Expander(i - 1)
        self.dim_expanders.append(expander)
        current_dim += expander.get_output_dim(input_dims[i - 1])
      else:
        self.dim_expanders.append(None)

      if i + 1 < self.max_order + 1:
        reducer = Reducer(i + 1, exclude_self)
        self.dim_reducers.append(reducer)
        current_dim += reducer.get_output_dim(input_dims[i + 1])
      else:
        self.dim_reducers.append(None)

      if current_dim == 0:
        self.dim_perms.append(None)
        self.logic.append(None)
        output_dims[i] = 0
      else:
        perm = Permutation(i)
        self.dim_perms.append(perm)
        current_dim = perm.get_output_dim(current_dim)
        self.logic.append(
            LogicInference(current_dim, output_dims[i], logic_hidden_dim))

    self.input_dims = input_dims
    self.output_dims = output_dims

    if self.residual:
      for i in range(len(input_dims)):
        self.output_dims[i] += input_dims[i]

  def forward(self, inputs):
    assert len(inputs) == self.max_order + 1
    outputs = []
    for i in range(self.max_order + 1):
      # collect input f from group i-1, i and i+1.
      f = []
      if i > 0 and self.input_dims[i - 1] > 0:
        n = inputs[i].size(1) if i == 1 else None
        f.append(self.dim_expanders[i](inputs[i - 1], n))
      if i < len(inputs) and self.input_dims[i] > 0:
        f.append(inputs[i])
      if i + 1 < len(inputs) and self.input_dims[i + 1] > 0:
        f.append(self.dim_reducers[i](inputs[i + 1]))
      if len(f) == 0:
        output = None
      else:
        f = torch.cat(f, dim=-1)
        f = self.dim_perms[i](f)
        output = self.logic[i](f)
      if self.residual and self.input_dims[i] > 0:
        output = torch.cat([inputs[i], output], dim=-1)
      outputs.append(output)
    return outputs

  __hyperparams__ = (
      'breadth',
      'input_dims',
      'output_dims',
      'logic_hidden_dim',
      'exclude_self',
      'residual',
  )

  __hyperparam_defaults__ = {
      'exclude_self': True,
      'residual': False,
  }

  @classmethod
  def make_nlm_parser(cls, parser, defaults, prefix=None):
    for k, v in cls.__hyperparam_defaults__.items():
      defaults.setdefault(k, v)

    if prefix is None:
      prefix = '--'
    else:
      prefix = '--' + str(prefix) + '-'

    parser.add_argument(
        prefix + 'breadth',
        type='int',
        default=defaults['breadth'],
        metavar='N',
        help='breadth of the logic layer')
    parser.add_argument(
        prefix + 'logic-hidden-dim',
        type=int,
        nargs='+',
        default=defaults['logic_hidden_dim'],
        metavar='N',
        help='hidden dim of the logic model')
    parser.add_argument(
        prefix + 'exclude-self',
        type='bool',
        default=defaults['exclude_self'],
        metavar='B',
        help='not allow multiple occurrence of same variable')
    parser.add_argument(
        prefix + 'residual',
        type='bool',
        default=defaults['residual'],
        metavar='B',
        help='use residual connections')

  @classmethod
  def from_args(cls, input_dims, output_dims, args, prefix=None, **kwargs):
    if prefix is None:
      prefix = ''
    else:
      prefix = str(prefix) + '_'

    setattr(args, prefix + 'input_dims', input_dims)
    setattr(args, prefix + 'output_dims', output_dims)
    init_params = {k: getattr(args, prefix + k) for k in cls.__hyperparams__}
    init_params.update(kwargs)

    return cls(**init_params)


class LogicMachine(nn.Module):
  """Neural Logic Machine consists of multiple logic layers."""

  def __init__(
      self,
      depth,
      breadth,
      input_dims,
      output_dims,
      logic_hidden_dim,
      exclude_self=True,
      residual=False,
      io_residual=False,
      recursion=False,
      connections=None,
  ):
    super().__init__()
    self.depth = depth
    self.breadth = breadth
    self.residual = residual
    self.io_residual = io_residual
    self.recursion = recursion
    self.connections = connections

    assert not (self.residual and self.io_residual), \
        'Only one type of residual connection is allowed at the same time.'

    # element-wise addition for vector
    def add_(x, y):
      for i in range(len(y)):
        x[i] += y[i]
      return x

    self.layers = nn.ModuleList()
    current_dims = input_dims
    total_output_dims = [0 for _ in range(self.breadth + 1)
                        ]  # for IO residual only
    for i in range(depth):
      # IO residual is unused.
      if i > 0 and io_residual:
        add_(current_dims, input_dims)
      # Not support output_dims as list or list[list] yet.
      layer = LogicLayer(breadth, current_dims, output_dims, logic_hidden_dim,
                         exclude_self, residual)
      current_dims = layer.output_dims
      current_dims = self._mask(current_dims, i, 0)
      if io_residual:
        add_(total_output_dims, current_dims)
      self.layers.append(layer)

    if io_residual:
      self.output_dims = total_output_dims
    else:
      self.output_dims = current_dims

  # Mask out the specific group-entry in layer i, specified by self.connections.
  # For debug usage.
  def _mask(self, a, i, masked_value):
    if self.connections is not None:
      assert i < len(self.connections)
      mask = self.connections[i]
      if mask is not None:
        assert len(mask) == len(a)
        a = [x if y else masked_value for x, y in zip(a, mask)]
    return a

  def forward(self, inputs, depth=None):
    outputs = [None for _ in range(self.breadth + 1)]
    f = inputs

    # depth: the actual depth used for inference
    if depth is None:
      depth = self.depth
    if not self.recursion:
      depth = min(depth, self.depth)

    def merge(x, y):
      if x is None:
        return y
      if y is None:
        return x
      return torch.cat([x, y], dim=-1)

    layer = None
    last_layer = None
    for i in range(depth):
      if i > 0 and self.io_residual:
        for j, inp in enumerate(inputs):
          f[j] = merge(f[j], inp)
      # To enable recursion, use scroll variables layer/last_layer
      # For weight sharing of period 2, i.e. 0,1,2,1,2,1,2,...
      if self.recursion and i >= 3:
        assert not self.residual
        layer, last_layer = last_layer, layer
      else:
        last_layer = layer
        layer = self.layers[i]

      f = layer(f)
      f = self._mask(f, i, None)
      if self.io_residual:
        for j, out in enumerate(f):
          outputs[j] = merge(outputs[j], out)
    if not self.io_residual:
      outputs = f
    return outputs

  __hyperparams__ = (
      'depth',
      'breadth',
      'input_dims',
      'output_dims',
      'logic_hidden_dim',
      'exclude_self',
      'io_residual',
      'residual',
      'recursion',
  )

  __hyperparam_defaults__ = {
      'exclude_self': True,
      'io_residual': False,
      'residual': False,
      'recursion': False,
  }

  @classmethod
  def make_nlm_parser(cls, parser, defaults, prefix=None):
    for k, v in cls.__hyperparam_defaults__.items():
      defaults.setdefault(k, v)

    if prefix is None:
      prefix = '--'
    else:
      prefix = '--' + str(prefix) + '-'

    parser.add_argument(
        prefix + 'depth',
        type=int,
        default=defaults['depth'],
        metavar='N',
        help='depth of the logic machine')
    parser.add_argument(
        prefix + 'breadth',
        type=int,
        default=defaults['breadth'],
        metavar='N',
        help='breadth of the logic machine')
    parser.add_argument(
        prefix + 'logic-hidden-dim',
        type=int,
        nargs='+',
        default=defaults['logic_hidden_dim'],
        metavar='N',
        help='hidden dim of the logic model')
    parser.add_argument(
        prefix + 'exclude-self',
        type='bool',
        default=defaults['exclude_self'],
        metavar='B',
        help='not allow multiple occurrence of same variable')
    parser.add_argument(
        prefix + 'io-residual',
        type='bool',
        default=defaults['io_residual'],
        metavar='B',
        help='use input/output-only residual connections')
    parser.add_argument(
        prefix + 'residual',
        type='bool',
        default=defaults['residual'],
        metavar='B',
        help='use residual connections')
    parser.add_argument(
        prefix + 'recursion',
        type='bool',
        default=defaults['recursion'],
        metavar='B',
        help='use recursion weight sharing')

  @classmethod
  def from_args(cls, input_dims, output_dims, args, prefix=None, **kwargs):
    if prefix is None:
      prefix = ''
    else:
      prefix = str(prefix) + '_'

    setattr(args, prefix + 'input_dims', input_dims)
    setattr(args, prefix + 'output_dims', output_dims)
    init_params = {k: getattr(args, prefix + k) for k in cls.__hyperparams__}
    init_params.update(kwargs)

    return cls(**init_params)
