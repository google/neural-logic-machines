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
"""Utility functions for PyTorch."""

import torch
import torch.nn.functional as F

from jactorch.utils.meta import as_float
from jactorch.utils.meta import as_tensor

__all__ = [
    'binary_accuracy', 'rms', 'monitor_saturation', 'monitor_paramrms',
    'monitor_gradrms'
]


def binary_accuracy(label, raw_pred, eps=1e-20, return_float=True):
  """get accuracy for binary classification problem."""
  pred = as_tensor(raw_pred).squeeze(-1)
  pred = (pred > 0.5).float()
  label = as_tensor(label).float()
  # The $acc is micro accuracy = the correct ones / total
  acc = label.eq(pred).float()

  # The $balanced_accuracy is macro accuracy, with class-wide balance.
  nr_total = torch.ones(
      label.size(), dtype=label.dtype, device=label.device).sum(dim=-1)
  nr_pos = label.sum(dim=-1)
  nr_neg = nr_total - nr_pos
  pos_cnt = (acc * label).sum(dim=-1)
  neg_cnt = acc.sum(dim=-1) - pos_cnt
  balanced_acc = ((pos_cnt + eps) / (nr_pos + eps) + (neg_cnt + eps) /
                  (nr_neg + eps)) / 2.0

  # $sat means the saturation rate of the predication,
  # measure how close the predections are to 0 or 1.
  sat = 1 - (raw_pred - pred).abs()
  if return_float:
    acc = as_float(acc.mean())
    balanced_acc = as_float(balanced_acc.mean())
    sat_mean = as_float(sat.mean())
    sat_min = as_float(sat.min())
  else:
    sat_mean = sat.mean(dim=-1)
    sat_min = sat.min(dim=-1)[0]

  return {
      'accuracy': acc,
      'balanced_accuracy': balanced_acc,
      'satuation/mean': sat_mean,
      'satuation/min': sat_min,
  }


def rms(p):
  """Root mean square function."""
  return as_float((as_tensor(p)**2).mean()**0.5)


def monitor_saturation(model):
  """Monitor the saturation rate."""
  monitors = {}
  for name, p in model.named_parameters():
    p = F.sigmoid(p)
    sat = 1 - (p - (p > 0.5).float()).abs()
    monitors['sat/' + name] = sat
  return monitors


def monitor_paramrms(model):
  """Monitor the rms of the parameters."""
  monitors = {}
  for name, p in model.named_parameters():
    monitors['paramrms/' + name] = rms(p)
  return monitors


def monitor_gradrms(model):
  """Monitor the rms of the gradients of the parameters."""
  monitors = {}
  for name, p in model.named_parameters():
    if p.grad is not None:
      monitors['gradrms/' + name] = rms(p.grad) / max(rms(p), 1e-8)
  return monitors
