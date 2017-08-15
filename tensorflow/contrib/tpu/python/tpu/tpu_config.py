# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ===================================================================

"""A RunConfig subclass with TPU support."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.learn.python.learn.estimators import run_config as run_config_lib


class TPUConfig(collections.namedtuple(
    'TPUConfig', ['iterations_per_loop',
                  'num_shards',
                  'per_host_input_for_training'])):
  """TPU related configuration required by `TPUEstimator`.

  Args:
    iterations_per_loop: This is the number of train steps runnining in TPU
      system before returning to CPU host for each `Session.run`. This means
      global step is increased `iterations_per_loop` times in one `Session.run`.
      It is recommended to be set as number of global steps for next checkpoint.
    num_shards: The number of TPU shards in the system.
    per_host_input_for_training: If `True`, `input_fn` is invoked per host
      rather than per shard. Note: This behavior is going to be default as
      `True` soon, so this flag will be removed after that. Also note that this
      only works for single-host TPU training now.
  """

  def __new__(cls, iterations_per_loop=2, num_shards=2,
              per_host_input_for_training=False):
    return super(TPUConfig, cls).__new__(
        cls,
        iterations_per_loop=iterations_per_loop,
        num_shards=num_shards,
        per_host_input_for_training=per_host_input_for_training)


class RunConfig(run_config_lib.RunConfig):
  """RunConfig with TPU support."""

  def __init__(self, tpu_config=None, **kwargs):
    super(RunConfig, self).__init__(**kwargs)
    self._tpu_config = tpu_config or TPUConfig()

  @property
  def tpu_config(self):
    return self._tpu_config
