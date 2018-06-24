# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for class OneDeviceStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distribute.python import one_device_strategy
from tensorflow.contrib.distribute.python import strategy_test_lib
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util


class OneDeviceStrategyTest(strategy_test_lib.DistributionTestBase):

  def _get_distribution_strategy(self):
    return one_device_strategy.OneDeviceStrategy("/device:CPU:0")

  def testMinimizeLossEager(self):
    self._test_minimize_loss_eager(self._get_distribution_strategy())

  def testMinimizeLossGraph(self):
    self._test_minimize_loss_graph(self._get_distribution_strategy())

  def testMapReduce(self):
    self._test_map_reduce(self._get_distribution_strategy())

  def testDeviceIndex(self):
    self._test_device_index(self._get_distribution_strategy())

  def testTowerId(self):
    self._test_tower_id(self._get_distribution_strategy())

  @test_util.run_in_graph_and_eager_modes()
  def testCallAndMergeExceptions(self):
    self._test_call_and_merge_exceptions(self._get_distribution_strategy())


if __name__ == "__main__":
  test.main()
