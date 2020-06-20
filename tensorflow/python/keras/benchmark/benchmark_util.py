# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Common util for benchmark"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timeit
import numpy as np

from tensorflow.python.keras import callbacks
from tensorflow.python.keras.benchmark import distribution_util


class TimerCallBack(callbacks.Callback):

  def __init__(self):
    self.times = []
    self.timer = timeit.default_timer
    self.startup_time = timeit.default_timer()
    self.recorded_startup = False

  def on_epoch_begin(self, e, logs):
    self.epoch_start_time = self.timer()

  def on_epoch_end(self, e, logs):
    self.times.append(self.timer() - self.epoch_start_time)

  def on_batch_end(self, e, logs):
    if not self.recorded_startup:
      self.startup_time = self.timer() - self.startup_time
      self.recorded_startup = True


def _measure_performance(model_fn, x_train, y_train, x_val, y_val,
    test_config):
  optimizer = test_config['optimizer']
  loss_fn = test_config['loss']
  metrics = test_config['metrics']
  warmup_epoch = test_config['warmup_epoch']
  epoch = test_config['epoch']
  batch_size = test_config['batch_size']
  run_iters = test_config['run_iters']
  num_gpus = test_config['num_gpus']
  distribution_strategy = test_config['distribution_strategy']

  avg_epoch_time_list, wall_time_list, exp_per_sec_list = [], [], []
  total_num_examples = (y_train.shape[0] + y_val.shape[0]) * epoch

  strategy = distribution_util.get_distribution_strategy(
      distribution_strategy=distribution_strategy,
      num_gpus=num_gpus)

  for _ in range(run_iters):
    timer = timeit.default_timer
    # each time you have to init scope
    strategy_scope = distribution_util.get_strategy_scope(strategy)
    with strategy_scope:
      model = model_fn()
      model.compile(
          optimizer=optimizer,
          loss=loss_fn,
          metrics=metrics,
      )

    model.fit(x_train, y_train, batch_size=batch_size, epochs=warmup_epoch)
    cbk = TimerCallBack()
    t0 = timer()
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epoch - warmup_epoch,
              callbacks=[cbk],
              verbose=0,
              validation_data=(x_val, y_val))
    end_time = timer()

    avg_epoch_time_list.append(np.mean(cbk.times[1:]))
    wall_time_list.append(end_time - t0)
    exp_per_sec_list.append(total_num_examples / (end_time - t0))

  results = {'avg_epoch_time': np.mean(avg_epoch_time_list),
             'wall_time': np.mean(wall_time_list),
             'exp_per_sec': np.mean(exp_per_sec_list)}
  return results
