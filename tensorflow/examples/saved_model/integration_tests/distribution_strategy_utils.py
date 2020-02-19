# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utils related to tf.distribute.strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys

from tensorflow.python.distribute import strategy_combinations

_strategies = [
    strategy_combinations.one_device_strategy,
    strategy_combinations.mirrored_strategy_with_one_cpu,
    strategy_combinations.mirrored_strategy_with_one_gpu,
    strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
    strategy_combinations.mirrored_strategy_with_two_gpus,
    strategy_combinations.tpu_strategy,
]

# TODO(b/145386854): The presence of GPU strategies upsets TPU initialization,
# despite their test instances being skipped early on.
if "test_tpu" in sys.argv[0]:
  _strategies = [s for s in _strategies if "GPU" not in str(s)]


named_strategies = collections.OrderedDict(
    [(None, None)] +
    [(str(s), s) for s in _strategies]
)


class MaybeDistributionScope(object):
  """Provides a context allowing no distribution strategy."""

  @staticmethod
  def from_name(name):
    return MaybeDistributionScope(named_strategies[name].strategy if name
                                  else None)

  def __init__(self, distribution):
    self._distribution = distribution
    self._scope = None

  def __enter__(self):
    if self._distribution:
      self._scope = self._distribution.scope()
      self._scope.__enter__()

  def __exit__(self, exc_type, value, traceback):
    if self._distribution:
      self._scope.__exit__(exc_type, value, traceback)
      self._scope = None
