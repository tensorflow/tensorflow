# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPU Initialization."""

from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_strategy_util


class TPUInitializationTest(parameterized.TestCase, test.TestCase):

  def test_tpu_initialization(self):
    resolver = tpu_cluster_resolver.TPUClusterResolver('')
    tpu_strategy_util.initialize_tpu_system(resolver)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
