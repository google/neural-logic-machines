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
"""MLP-based implementation for logic and logits inference."""

import torch.nn as nn

from jactorch.quickstart.models import MLPModel

__all__ = ['LogicInference', 'LogitsInference']


class InferenceBase(nn.Module):
  """MLP model with shared parameters among other axies except the channel axis."""

  def __init__(self, input_dim, output_dim, hidden_dim):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.layer = nn.Sequential(MLPModel(input_dim, output_dim, hidden_dim))

  def forward(self, inputs):
    input_size = inputs.size()[:-1]
    input_channel = inputs.size(-1)

    f = inputs.view(-1, input_channel)
    f = self.layer(f)
    f = f.view(*input_size, -1)
    return f

  def get_output_dim(self, input_dim):
    return self.output_dim


class LogicInference(InferenceBase):
  """MLP layer with sigmoid activation."""

  def __init__(self, input_dim, output_dim, hidden_dim):
    super().__init__(input_dim, output_dim, hidden_dim)
    self.layer.add_module(str(len(self.layer)), nn.Sigmoid())


class LogitsInference(InferenceBase):
  pass
