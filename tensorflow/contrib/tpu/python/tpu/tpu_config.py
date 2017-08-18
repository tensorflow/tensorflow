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


class TPUConfig(
    collections.namedtuple('TPUConfig', [
        'iterations_per_loop', 'num_shards', 'per_host_input_for_training',
        'shard_dimensions'
    ])):
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
    shard_dimensions: A python tuple of int values describing how each tensor
      produced by the Estimator `input_fn` should be split across the TPU
      compute shards. For example, if your input_fn produced (images, labels)
      where the images tensor is in `HWCN` format, your shard dimensions would
      be: [3, 0], where 3 corresponds to the `N` dimension of your images
      Tensor, and 0 corresponds to the dimension along which to split the labels
      to match up with the corresponding images. If None is supplied, and
      per_host_input_for_training is True, batches will be sharded based on the
      major dimension. If per_host_input_for_training is False, shard_dimensions
      is ignored.
  """
  # TODO(b/64607814): Ensure shard_dimensions works with nested structures.

  def __new__(cls,
              iterations_per_loop=2,
              num_shards=2,
              per_host_input_for_training=False,
              shard_dimensions=None):
    return super(TPUConfig, cls).__new__(
        cls,
        iterations_per_loop=iterations_per_loop,
        num_shards=num_shards,
        per_host_input_for_training=per_host_input_for_training,
        shard_dimensions=shard_dimensions)


class RunConfig(run_config_lib.RunConfig):
  """RunConfig with TPU support."""

  def __init__(self, tpu_config=None, **kwargs):
    super(RunConfig, self).__init__(**kwargs)
    self._tpu_config = tpu_config or TPUConfig()

  @property
  def tpu_config(self):
    return self._tpu_config
