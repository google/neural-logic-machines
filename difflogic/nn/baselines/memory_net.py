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

import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jacinle.random as random
from difflogic.nn.neural_logic.modules._utils import meshgrid
from jactorch.quickstart.models import MLPModel

__all__ = ['MemoryNet']


class MemoryNet(nn.Module):

  def __init__(self, input_dim, feature_dim, queries, hidden_dim, key_dim,
               value_dim, id_dim):
    super().__init__()
    self.feature_dim = feature_dim
    assert feature_dim == 1 or feature_dim == 2, \
        'only support attributes or relations'
    self.queries = queries
    self.hidden_dim = hidden_dim
    self.key_dim = key_dim
    self.value_dim = value_dim
    self.id_dim = id_dim

    current_dim = id_dim * 2 + input_dim
    self.key_embed = MLPModel(current_dim, key_dim, [])
    self.value_embed = MLPModel(current_dim, value_dim, [])
    self.query_embed = MLPModel(id_dim * 2, key_dim, [])
    self.to_query = MLPModel(hidden_dim, key_dim, [])
    self.lstm_cell = nn.LSTMCell(value_dim, hidden_dim)

  def forward(self, relations, attributes=None):
    batch_size, nr = relations.size()[:2]
    assert nr == relations.size(2)
    create_zeros = functools.partial(
        torch.zeros, dtype=relations.dtype, device=relations.device)

    id_shape = list(relations.size()[:-1])
    ids = [random.permutation(2 ** self.id_dim - 1)[:nr] + 1 \
        for i in range(batch_size)]
    ids = np.vstack(ids)
    binary_ids = self.binarize_code(ids)
    zeros = create_zeros(binary_ids.shape)
    binary_ids = torch.tensor(
        binary_ids, dtype=relations.dtype, device=relations.device)
    binary_ids2 = torch.cat(meshgrid(binary_ids, dim=1), dim=-1)
    padded_binary_ids = torch.cat([binary_ids, zeros], dim=-1)

    def embed(embed, x):
      input_size = x.size()[:-1]
      input_channel = x.size(-1)
      f = x.view(-1, input_channel)
      f = embed(f)
      return f.view(*input_size, -1)

    if attributes is None:
      rels = [binary_ids2, relations]
    else:
      padding = create_zeros(*binary_ids2.size()[:-1], attributes.size(-1))
      rels = [binary_ids2, padding, relations]
    rels = torch.cat(rels, dim=-1)
    memory = rels.view(batch_size, -1, rels.size(-1))
    if attributes is not None:
      assert nr == attributes.size(1)
      padding = create_zeros(*padded_binary_ids.size()[:-1], relations.size(-1))
      attributes = torch.cat([padded_binary_ids, attributes, padding], dim=-1)
      memory = torch.cat([memory, attributes], dim=1)
    keys = embed(self.key_embed, memory).transpose(1, 2)
    values = embed(self.value_embed, memory)

    query = padded_binary_ids if self.feature_dim == 1 else binary_ids2
    nr_items = nr**self.feature_dim
    query = embed(self.query_embed, query).view(batch_size, nr_items, -1)

    h0 = create_zeros(batch_size * nr_items, self.hidden_dim)
    c0 = create_zeros(batch_size * nr_items, self.hidden_dim)
    for i in range(self.queries):
      attention = F.softmax(torch.bmm(query, keys), dim=-1)
      value = torch.bmm(attention, values)
      value = value.view(-1, value.size(-1))

      h0, c0 = self.lstm_cell(value, (h0, c0))
      query = self.to_query(h0).view(batch_size, nr_items, self.key_dim)

    if self.feature_dim == 1:
      out = h0.view(batch_size, nr, self.hidden_dim)
    else:
      out = h0.view(batch_size, nr, nr, self.hidden_dim)
    return out

  def binarize_code(self, x):
    m = self.id_dim
    code = np.zeros((x.shape + (m,)))
    for i in range(m)[::-1]:
      code[:, :, i] = (x >= 2**i).astype('float')
      x = x - code[:, :, i] * 2**i
    return code

  def get_output_dim(self):
    return self.hidden_dim

  __hyperparams__ = ('queries', 'hidden_dim', 'key_dim', 'value_dim', 'id_dim')

  __hyperparam_defaults__ = {
      'queries': 4,
      'hidden_dim': 64,
      'key_dim': 16,
      'value_dim': 32,
      'id_dim': 8,
  }

  @classmethod
  def make_memnet_parser(cls, parser, defaults, prefix=None):
    for k, v in cls.__hyperparam_defaults__.items():
      defaults.setdefault(k, v)

    if prefix is None:
      prefix = '--'
    else:
      prefix = '--' + str(prefix) + '-'

    parser.add_argument(
        prefix + 'queries',
        type=int,
        default=defaults['queries'],
        metavar='N',
        help='number of queries')
    parser.add_argument(
        prefix + 'hidden-dim',
        type=int,
        default=defaults['hidden_dim'],
        metavar='N',
        help='hidden dimension of LSTM cell')
    parser.add_argument(
        prefix + 'key-dim',
        type=int,
        default=defaults['key_dim'],
        metavar='N',
        help='dimension of key vector')
    parser.add_argument(
        prefix + 'value-dim',
        type=int,
        default=defaults['value_dim'],
        metavar='N',
        help='dimension of value vector')
    parser.add_argument(
        prefix + 'id-dim',
        type=int,
        default=defaults['id_dim'],
        metavar='N',
        help='dimension of id vector')

  @classmethod
  def from_args(cls, input_dim, feature_dim, args, prefix=None, **kwargs):
    if prefix is None:
      prefix = ''
    else:
      prefix = str(prefix) + '_'

    init_params = {k: getattr(args, prefix + k) for k in cls.__hyperparams__}
    init_params.update(kwargs)

    return cls(input_dim, feature_dim, **init_params)
