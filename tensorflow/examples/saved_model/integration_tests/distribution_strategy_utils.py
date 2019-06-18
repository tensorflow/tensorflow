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

from tensorflow.python.distribute import strategy_combinations

_strategies = [
    strategy_combinations.one_device_strategy,
    strategy_combinations.mirrored_strategy_with_one_cpu,
    strategy_combinations.mirrored_strategy_with_one_gpu,
    strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
    strategy_combinations.mirrored_strategy_with_two_gpus,
]

named_strategies = collections.OrderedDict()
named_strategies[None] = None


for strategy in _strategies:
  named_strategies[str(strategy)] = strategy
