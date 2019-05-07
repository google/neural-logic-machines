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
"""The script for family tree or general graphs experiments."""

import copy
import collections
import functools
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jacinle.random as random
import jacinle.io as io
import jactorch.nn as jacnn

from difflogic.cli import format_args
from difflogic.dataset.graph import GraphOutDegreeDataset, \
    GraphConnectivityDataset, GraphAdjacentDataset, FamilyTreeDataset
from difflogic.nn.baselines import MemoryNet
from difflogic.nn.neural_logic import LogicMachine, LogicInference, LogitsInference
from difflogic.nn.neural_logic.modules._utils import meshgrid_exclude_self
from difflogic.nn.rl.reinforce import REINFORCELoss
from difflogic.thutils import binary_accuracy
from difflogic.train import TrainerBase

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jactorch.data.dataloader import JacDataLoader
from jactorch.optim.accum_grad import AccumGrad
from jactorch.optim.quickaccess import get_optimizer
from jactorch.train.env import TrainerEnv
from jactorch.utils.meta import as_cuda, as_numpy, as_tensor

TASKS = [
    'outdegree', 'connectivity', 'adjacent', 'adjacent-mnist', 'has-father',
    'has-sister', 'grandparents', 'uncle', 'maternal-great-uncle'
]

parser = JacArgumentParser()

parser.add_argument(
    '--model',
    default='nlm',
    choices=['nlm', 'memnet'],
    help='model choices, nlm: Neural Logic Machine, memnet: Memory Networks')

# NLM parameters, works when model is 'nlm'
nlm_group = parser.add_argument_group('Neural Logic Machines')
LogicMachine.make_nlm_parser(
    nlm_group, {
        'depth': 4,
        'breadth': 3,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='nlm')
nlm_group.add_argument(
    '--nlm-attributes',
    type=int,
    default=8,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)

# MemNN parameters, works when model is 'memnet'
memnet_group = parser.add_argument_group('Memory Networks')
MemoryNet.make_memnet_parser(memnet_group, {}, prefix='memnet')

# task related
task_group = parser.add_argument_group('Task')
task_group.add_argument(
    '--task', required=True, choices=TASKS, help='tasks choices')
task_group.add_argument(
    '--train-number',
    type=int,
    default=10,
    metavar='N',
    help='size of training instances')
task_group.add_argument(
    '--adjacent-pred-colors', type=int, default=4, metavar='N')
task_group.add_argument('--outdegree-n', type=int, default=2, metavar='N')
task_group.add_argument(
    '--connectivity-dist-limit', type=int, default=4, metavar='N')

data_gen_group = parser.add_argument_group('Data Generation')
data_gen_group.add_argument(
    '--gen-graph-method',
    default='edge',
    choices=['dnc', 'edge'],
    help='method use to generate random graph')
data_gen_group.add_argument(
    '--gen-graph-pmin',
    type=float,
    default=0.0,
    metavar='F',
    help='control parameter p reflecting the graph sparsity')
data_gen_group.add_argument(
    '--gen-graph-pmax',
    type=float,
    default=0.3,
    metavar='F',
    help='control parameter p reflecting the graph sparsity')
data_gen_group.add_argument(
    '--gen-graph-colors',
    type=int,
    default=4,
    metavar='N',
    help='number of colors in adjacent task')
data_gen_group.add_argument(
    '--gen-directed', action='store_true', help='directed graph')

train_group = parser.add_argument_group('Train')
train_group.add_argument(
    '--seed',
    type=int,
    default=None,
    metavar='SEED',
    help='seed of jacinle.random')
train_group.add_argument(
    '--use-gpu', action='store_true', help='use GPU or not')
train_group.add_argument(
    '--optimizer',
    default='AdamW',
    choices=['SGD', 'Adam', 'AdamW'],
    help='optimizer choices')
train_group.add_argument(
    '--lr',
    type=float,
    default=0.005,
    metavar='F',
    help='initial learning rate')
train_group.add_argument(
    '--lr-decay',
    type=float,
    default=1.0,
    metavar='F',
    help='exponential decay of learning rate per lesson')
train_group.add_argument(
    '--accum-grad',
    type=int,
    default=1,
    metavar='N',
    help='accumulated gradient for batches (default: 1)')
train_group.add_argument(
    '--ohem-size',
    type=int,
    default=0,
    metavar='N',
    help='size of online hard negative mining')
train_group.add_argument(
    '--batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for training')
train_group.add_argument(
    '--test-batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for testing')
train_group.add_argument(
    '--early-stop-loss-thresh',
    type=float,
    default=1e-5,
    metavar='F',
    help='threshold of loss for early stop')

# Note that nr_examples_per_epoch = epoch_size * batch_size
TrainerBase.make_trainer_parser(
    parser, {
        'epochs': 50,
        'epoch_size': 250,
        'test_epoch_size': 250,
        'test_number_begin': 10,
        'test_number_step': 10,
        'test_number_end': 50,
    })

io_group = parser.add_argument_group('Input/Output')
io_group.add_argument(
    '--dump-dir', type=str, default=None, metavar='DIR', help='dump dir')
io_group.add_argument(
    '--load-checkpoint',
    type=str,
    default=None,
    metavar='FILE',
    help='load parameters from checkpoint')

schedule_group = parser.add_argument_group('Schedule')
schedule_group.add_argument(
    '--runs', type=int, default=1, metavar='N', help='number of runs')
schedule_group.add_argument(
    '--save-interval',
    type=int,
    default=10,
    metavar='N',
    help='the interval(number of epochs) to save checkpoint')
schedule_group.add_argument(
    '--test-interval',
    type=int,
    default=None,
    metavar='N',
    help='the interval(number of epochs) to do test')
schedule_group.add_argument(
    '--test-only', action='store_true', help='test-only mode')

logger = get_logger(__file__)

args = parser.parse_args()

args.use_gpu = args.use_gpu and torch.cuda.is_available()

if args.dump_dir is not None:
  io.mkdir(args.dump_dir)
  args.log_file = os.path.join(args.dump_dir, 'log.log')
  set_output_file(args.log_file)
else:
  args.checkpoints_dir = None
  args.summary_file = None

if args.seed is not None:
  import jacinle.random as random
  random.reset_global_seed(args.seed)

args.task_is_outdegree = args.task in ['outdegree']
args.task_is_connectivity = args.task in ['connectivity']
args.task_is_adjacent = args.task in ['adjacent', 'adjacent-mnist']
args.task_is_family_tree = args.task in [
    'has-father', 'has-sister', 'grandparents', 'uncle', 'maternal-great-uncle'
]
args.task_is_mnist_input = args.task in ['adjacent-mnist']
args.task_is_1d_output = args.task in [
    'outdegree', 'adjacent', 'adjacent-mnist', 'has-father', 'has-sister'
]


class LeNet(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = jacnn.Conv2dLayer(
        1, 10, kernel_size=5, batch_norm=True, activation='relu')
    self.conv2 = jacnn.Conv2dLayer(
        10,
        20,
        kernel_size=5,
        batch_norm=True,
        dropout=False,
        activation='relu')
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2)
    x = F.max_pool2d(self.conv2(x), 2)
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


class Model(nn.Module):
  """The model for family tree or general graphs path tasks."""

  def __init__(self):
    super().__init__()

    # inputs
    input_dim = 4 if args.task_is_family_tree else 1
    self.feature_axis = 1 if args.task_is_1d_output else 2

    # features
    if args.model == 'nlm':
      input_dims = [0 for _ in range(args.nlm_breadth + 1)]
      if args.task_is_adjacent:
        input_dims[1] = args.gen_graph_colors
        if args.task_is_mnist_input:
          self.lenet = LeNet()
      input_dims[2] = input_dim

      self.features = LogicMachine.from_args(
          input_dims, args.nlm_attributes, args, prefix='nlm')
      output_dim = self.features.output_dims[self.feature_axis]

    elif args.model == 'memnet':
      if args.task_is_adjacent:
        input_dim += args.gen_graph_colors
      self.feature = MemoryNet.from_args(
          input_dim, self.feature_axis, args, prefix='memnet')
      output_dim = self.feature.get_output_dim()

    # target
    target_dim = args.adjacent_pred_colors if args.task_is_adjacent else 1
    self.pred = LogicInference(output_dim, target_dim, [])

    # losses
    if args.ohem_size > 0:
      from jactorch.nn.losses import BinaryCrossEntropyLossWithProbs as BCELoss
      self.loss = BCELoss(average='none')
    else:
      self.loss = nn.BCELoss()

  def forward(self, feed_dict):
    feed_dict = GView(feed_dict)

    # properties
    if args.task_is_adjacent:
      states = feed_dict.states.float()
    else:
      states = None

    # relations
    relations = feed_dict.relations.float()
    batch_size, nr = relations.size()[:2]

    if args.model == 'nlm':
      if args.task_is_adjacent and args.task_is_mnist_input:
        states_shape = states.size()
        states = states.view((-1,) + states_shape[2:])
        states = self.lenet(states)
        states = states.view(states_shape[:2] + (-1,))
        states = F.sigmoid(states)

      inp = [None for _ in range(args.nlm_breadth + 1)]
      inp[1] = states
      inp[2] = relations

      depth = None
      if args.nlm_recursion:
        depth = 1
        while 2**depth + 1 < nr:
          depth += 1
        depth = depth * 2 + 1
      feature = self.features(inp, depth=depth)[self.feature_axis]
    elif args.model == 'memnet':
      feature = self.feature(relations, states)
      if args.task_is_adjacent and args.task_is_mnist_input:
        raise NotImplementedError()

    pred = self.pred(feature)
    if not args.task_is_adjacent:
      pred = pred.squeeze(-1)
    if args.task_is_connectivity:
      pred = meshgrid_exclude_self(pred)  # exclude self-cycle

    if self.training:
      monitors = dict()
      target = feed_dict.target.float()

      if args.task_is_adjacent:
        target = target[:, :, :args.adjacent_pred_colors]

      monitors.update(binary_accuracy(target, pred, return_float=False))

      loss = self.loss(pred, target)
      # ohem loss is unused.
      if args.ohem_size > 0:
        loss = loss.view(-1).topk(args.ohem_size)[0].mean()
      return loss, monitors, dict(pred=pred)
    else:
      return dict(pred=pred)


def make_dataset(n, epoch_size, is_train):
  pmin, pmax = args.gen_graph_pmin, args.gen_graph_pmax
  if args.task_is_outdegree:
    return GraphOutDegreeDataset(
        args.outdegree_n,
        epoch_size,
        n,
        pmin=pmin,
        pmax=pmax,
        directed=args.gen_directed,
        gen_method=args.gen_graph_method)
  elif args.task_is_connectivity:
    nmin, nmax = n, n
    if is_train and args.nlm_recursion:
      nmin = 2
    return GraphConnectivityDataset(
        args.connectivity_dist_limit,
        epoch_size,
        nmin,
        pmin,
        nmax,
        pmax,
        directed=args.gen_directed,
        gen_method=args.gen_graph_method)
  elif args.task_is_adjacent:
    return GraphAdjacentDataset(
        args.gen_graph_colors,
        epoch_size,
        n,
        pmin=pmin,
        pmax=pmax,
        directed=args.gen_directed,
        gen_method=args.gen_graph_method,
        is_train=is_train,
        is_mnist_colors=args.task_is_mnist_input)
  else:
    return FamilyTreeDataset(args.task, epoch_size, n, p_marriage=1.0)


class MyTrainer(TrainerBase):
  def save_checkpoint(self, name):
    if args.checkpoints_dir is not None:
      checkpoint_file = os.path.join(args.checkpoints_dir,
                                     'checkpoint_{}.pth'.format(name))
      super().save_checkpoint(checkpoint_file)

  def _dump_meters(self, meters, mode):
    if args.summary_file is not None:
      meters_kv = meters._canonize_values('avg')
      meters_kv['mode'] = mode
      meters_kv['epoch'] = self.current_epoch
      with open(args.summary_file, 'a') as f:
        f.write(io.dumps_json(meters_kv))
        f.write('\n')

  data_iterator = {}

  def _prepare_dataset(self, epoch_size, mode):
    assert mode in ['train', 'test']
    if mode == 'train':
      batch_size = args.batch_size
      number = args.train_number
    else:
      batch_size = args.test_batch_size
      number = self.test_number

    # The actual number of instances in an epoch is epoch_size * batch_size.
    dataset = make_dataset(number, epoch_size * batch_size, mode == 'train')
    dataloader = JacDataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=min(epoch_size, 4))
    self.data_iterator[mode] = dataloader.__iter__()

  def _get_data(self, index, meters, mode):
    feed_dict = self.data_iterator[mode].next()
    meters.update(number=feed_dict['n'].data.numpy().mean())
    if args.use_gpu:
      feed_dict = as_cuda(feed_dict)
    return feed_dict

  def _get_result(self, index, meters, mode):
    feed_dict = self._get_data(index, meters, mode)
    output_dict = self.model(feed_dict)

    target = feed_dict['target']
    if args.task_is_adjacent:
      target = target[:, :, :args.adjacent_pred_colors]
    result = binary_accuracy(target, output_dict['pred'])
    succ = result['accuracy'] == 1.0

    meters.update(succ=succ)
    meters.update(result, n=target.size(0))
    message = '> {} iter={iter}, accuracy={accuracy:.4f}, \
balance_acc={balanced_accuracy:.4f}'.format(
        mode, iter=index, **meters.val)
    return message, dict(succ=succ, feed_dict=feed_dict)

  def _get_train_data(self, index, meters):
    return self._get_data(index, meters, mode='train')

  def _train_epoch(self, epoch_size):
    meters = super()._train_epoch(epoch_size)

    i = self.current_epoch
    if args.save_interval is not None and i % args.save_interval == 0:
      self.save_checkpoint(str(i))
    if args.test_interval is not None and i % args.test_interval == 0:
      self.test()
    return meters

  def _early_stop(self, meters):
    return meters.avg['loss'] < args.early_stop_loss_thresh


def main(run_id):
  if args.dump_dir is not None:
    if args.runs > 1:
      args.current_dump_dir = os.path.join(args.dump_dir,
                                           'run_{}'.format(run_id))
      io.mkdir(args.current_dump_dir)
    else:
      args.current_dump_dir = args.dump_dir

    args.summary_file = os.path.join(args.current_dump_dir, 'summary.json')
    args.checkpoints_dir = os.path.join(args.current_dump_dir, 'checkpoints')
    io.mkdir(args.checkpoints_dir)

  logger.info(format_args(args))

  model = Model()
  if args.use_gpu:
    model.cuda()
  optimizer = get_optimizer(args.optimizer, model, args.lr)
  if args.accum_grad > 1:
    optimizer = AccumGrad(optimizer, args.accum_grad)
  trainer = MyTrainer.from_args(model, optimizer, args)

  if args.load_checkpoint is not None:
    trainer.load_checkpoint(args.load_checkpoint)

  if args.test_only:
    return None, trainer.test()

  final_meters = trainer.train()
  trainer.save_checkpoint('last')

  return trainer.early_stopped, trainer.test()


if __name__ == '__main__':
  stats = []
  nr_graduated = 0

  for i in range(args.runs):
    graduated, test_meters = main(i)
    logger.info('run {}'.format(i + 1))

    if test_meters is not None:
      for j, meters in enumerate(test_meters):
        if len(stats) <= j:
          stats.append(GroupMeters())
        stats[j].update(
            number=meters.avg['number'], test_acc=meters.avg['accuracy'])

      for meters in stats:
        logger.info('number {}, test_acc {}'.format(meters.avg['number'],
                                                    meters.avg['test_acc']))

    if not args.test_only:
      nr_graduated += int(graduated)
      logger.info('graduate_ratio {}'.format(nr_graduated / (i + 1)))
      if graduated:
        for j, meters in enumerate(test_meters):
          stats[j].update(grad_test_acc=meters.avg['accuracy'])
      if nr_graduated > 0:
        for meters in stats:
          logger.info('number {}, grad_test_acc {}'.format(
              meters.avg['number'], meters.avg['grad_test_acc']))
