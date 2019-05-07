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
"""The LSTM baseline."""

import numpy as np
import torch
import torch.nn as nn

import jacinle.random as random
from difflogic.nn.neural_logic.modules._utils import meshgrid
from jactorch.functional.shape import broadcast

__all__ = ['LSTMBaseline']


class LSTMBaseline(nn.Module):
  """LSTM baseline model."""
  def __init__(self,
               input_dim,
               feature_dim,
               num_layers=2,
               hidden_size=512,
               code_length=8):
    super().__init__()
    current_dim = input_dim + code_length * 2
    self.feature_dim = feature_dim
    assert feature_dim == 1 or feature_dim == 2, ('only support attributes or '
                                                  'relations')
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.code_length = code_length
    self.lstm = nn.LSTM(
        current_dim,
        hidden_size,
        num_layers,
        batch_first=True,
        bidirectional=True)

  def forward(self, relations, attributes=None):
    batch_size, nr = relations.size()[:2]
    assert nr == relations.size(2)

    id_shape = list(relations.size()[:-1])
    ids = [
        random.permutation(2**self.code_length - 1)[:nr] + 1
        for i in range(batch_size)
    ]
    ids = np.vstack(ids)
    binary_ids = self.binarize_code(ids)
    zeros = torch.tensor(
        np.zeros(binary_ids.shape),
        dtype=relations.dtype,
        device=relations.device)
    binary_ids = torch.tensor(
        binary_ids, dtype=relations.dtype, device=relations.device)
    binary_ids2 = torch.cat(meshgrid(binary_ids, dim=1), dim=-1)

    if attributes is None:
      rels = [binary_ids2, relations]
    else:
      padding = torch.zeros(
          *binary_ids2.size()[:-1],
          attributes.size(-1),
          dtype=relations.dtype,
          device=relations.device)
      rels = [binary_ids2, padding, relations]
    rels = torch.cat(rels, dim=-1)
    input_seq = rels.view(batch_size, -1, rels.size(-1))
    if attributes is not None:
      assert nr == attributes.size(1)
      padding = torch.zeros(
          *binary_ids.size()[:-1],
          relations.size(-1),
          dtype=relations.dtype,
          device=relations.device)
      attributes = torch.cat([binary_ids, zeros, attributes, padding], dim=-1)
      input_seq = torch.cat([input_seq, attributes], dim=1)

    h0 = torch.zeros(
        self.num_layers * 2,
        batch_size,
        self.hidden_size,
        dtype=relations.dtype,
        device=relations.device)
    c0 = torch.zeros(
        self.num_layers * 2,
        batch_size,
        self.hidden_size,
        dtype=relations.dtype,
        device=relations.device)
    out, _ = self.lstm(input_seq, (h0, c0))
    out = out[:, -1]

    if self.feature_dim == 1:
      expanded_feature = broadcast(out.unsqueeze(dim=1), 1, nr)
      return torch.cat([binary_ids, expanded_feature], dim=-1)
    else:
      expanded_feature = broadcast(out.unsqueeze(dim=1), 1, nr)
      expanded_feature = broadcast(expanded_feature.unsqueeze(dim=1), 1, nr)
      return torch.cat([binary_ids2, expanded_feature], dim=-1)

  def binarize_code(self, x):
    m = self.code_length
    code = np.zeros((x.shape + (m,)))
    for i in range(m)[::-1]:
      code[:, :, i] = (x >= 2**i).astype('float')
      x = x - code[:, :, i] * 2**i
    return code

  def get_output_dim(self):
    return self.hidden_size * 2 + self.code_length * self.feature_dim
