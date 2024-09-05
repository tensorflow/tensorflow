# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for TPU device assignment."""

from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python import tpu_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


Layout = layout_lib.Layout
Mesh = layout_lib.Mesh


class DeviceAssignmentTest(test_util.DTensorBaseTest):

  def setUp(self):
    super().setUp()
    accelerator_util.initialize_accelerator_system('TPU')

  def tearDown(self):
    accelerator_util.shutdown_accelerator_system()
    super().tearDown()

  def _build_all_reduce_ring(self, core_locations):
    permutation = tpu_util._build_all_reduce_ring(core_locations)
    return [core_locations[element] for element in permutation]

  # Picture of chips:
  # 0 -- 1
  # |    |
  # 3 -- 2
  def testBuildAllReduceRing4Replicas(self):
    core_locations = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 0),
    ]
    expected = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 0),
    ]
    result = self._build_all_reduce_ring(core_locations)
    self.assertAllEqual(result, expected)

  # Picture of chips with core0/core1 assignments:
  # 0/1 -- 2/3
  #  |      |
  # 6/7 -- 4/5
  def testBuildAllReduceRing8ReplicasUsingTwoCores(self):
    core_locations = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 1),
    ]
    expected = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
    ]
    result = self._build_all_reduce_ring(core_locations)
    self.assertAllEqual(result, expected)

  # Picture of chips:
  # 0 -- 1 -- 2 -- 3
  # |              |
  # 15   6 -- 5 -- 4
  # |    |
  # 14   7 -- 8 -- 9
  # |              |
  # 13-- 12-- 11-- 10
  def testBuildAllReduceRing32Replicas(self):
    core_locations = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(0, 2, 0, 0),
        tpu_util._CoreLocation(0, 2, 0, 1),
        tpu_util._CoreLocation(0, 3, 0, 0),
        tpu_util._CoreLocation(0, 3, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(1, 2, 0, 0),
        tpu_util._CoreLocation(1, 2, 0, 1),
        tpu_util._CoreLocation(1, 3, 0, 0),
        tpu_util._CoreLocation(1, 3, 0, 1),
        tpu_util._CoreLocation(2, 0, 0, 0),
        tpu_util._CoreLocation(2, 0, 0, 1),
        tpu_util._CoreLocation(2, 1, 0, 0),
        tpu_util._CoreLocation(2, 1, 0, 1),
        tpu_util._CoreLocation(2, 2, 0, 0),
        tpu_util._CoreLocation(2, 2, 0, 1),
        tpu_util._CoreLocation(2, 3, 0, 0),
        tpu_util._CoreLocation(2, 3, 0, 1),
        tpu_util._CoreLocation(3, 0, 0, 0),
        tpu_util._CoreLocation(3, 0, 0, 1),
        tpu_util._CoreLocation(3, 1, 0, 0),
        tpu_util._CoreLocation(3, 1, 0, 1),
        tpu_util._CoreLocation(3, 2, 0, 0),
        tpu_util._CoreLocation(3, 2, 0, 1),
        tpu_util._CoreLocation(3, 3, 0, 0),
        tpu_util._CoreLocation(3, 3, 0, 1),
    ]
    expected = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(2, 0, 0, 0),
        tpu_util._CoreLocation(2, 0, 0, 1),
        tpu_util._CoreLocation(3, 0, 0, 0),
        tpu_util._CoreLocation(3, 0, 0, 1),
        tpu_util._CoreLocation(3, 1, 0, 0),
        tpu_util._CoreLocation(3, 1, 0, 1),
        tpu_util._CoreLocation(2, 1, 0, 0),
        tpu_util._CoreLocation(2, 1, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(1, 2, 0, 0),
        tpu_util._CoreLocation(1, 2, 0, 1),
        tpu_util._CoreLocation(2, 2, 0, 0),
        tpu_util._CoreLocation(2, 2, 0, 1),
        tpu_util._CoreLocation(3, 2, 0, 0),
        tpu_util._CoreLocation(3, 2, 0, 1),
        tpu_util._CoreLocation(3, 3, 0, 0),
        tpu_util._CoreLocation(3, 3, 0, 1),
        tpu_util._CoreLocation(2, 3, 0, 0),
        tpu_util._CoreLocation(2, 3, 0, 1),
        tpu_util._CoreLocation(1, 3, 0, 0),
        tpu_util._CoreLocation(1, 3, 0, 1),
        tpu_util._CoreLocation(0, 3, 0, 0),
        tpu_util._CoreLocation(0, 3, 0, 1),
        tpu_util._CoreLocation(0, 2, 0, 0),
        tpu_util._CoreLocation(0, 2, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
    ]
    result = self._build_all_reduce_ring(core_locations)
    self.assertAllEqual(result, expected)

  # Picture of chips:
  # 7 -- 0  6 -- 5
  #      |       |
  # 2 -- 1  3 -- 4
  def testBuildAllReduceRing3D(self):
    core_locations = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(0, 0, 1, 0),
        tpu_util._CoreLocation(0, 0, 1, 1),
        tpu_util._CoreLocation(0, 1, 1, 0),
        tpu_util._CoreLocation(0, 1, 1, 1),
        tpu_util._CoreLocation(1, 0, 1, 0),
        tpu_util._CoreLocation(1, 0, 1, 1),
        tpu_util._CoreLocation(1, 1, 1, 0),
        tpu_util._CoreLocation(1, 1, 1, 1),
    ]
    expected = [
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(0, 1, 1, 1),
        tpu_util._CoreLocation(0, 1, 1, 0),
        tpu_util._CoreLocation(1, 1, 1, 1),
        tpu_util._CoreLocation(1, 1, 1, 0),
        tpu_util._CoreLocation(1, 0, 1, 1),
        tpu_util._CoreLocation(1, 0, 1, 0),
        tpu_util._CoreLocation(0, 0, 1, 0),
        tpu_util._CoreLocation(0, 0, 1, 1),
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
    ]
    result = self._build_all_reduce_ring(core_locations)
    self.assertAllEqual(result, expected)

  # Picture of chips:
  # 31-- 0 -- 1 -- 2  30--29--28--27
  #                |              |
  # 14   5 -- 4 -- 3  15  24--25--26
  # |    |            |   |
  # 13   6 -- 7 -- 8  16  23--22--21
  # |              |  |           |
  # 12-- 11-- 10-- 9  17--18--19--20
  def testBuildAllReduceRing3DLarge(self):
    core_locations = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(2, 0, 0, 0),
        tpu_util._CoreLocation(2, 0, 0, 1),
        tpu_util._CoreLocation(3, 0, 0, 0),
        tpu_util._CoreLocation(3, 0, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(2, 1, 0, 0),
        tpu_util._CoreLocation(2, 1, 0, 1),
        tpu_util._CoreLocation(3, 1, 0, 0),
        tpu_util._CoreLocation(3, 1, 0, 1),
        tpu_util._CoreLocation(0, 2, 0, 0),
        tpu_util._CoreLocation(0, 2, 0, 1),
        tpu_util._CoreLocation(1, 2, 0, 0),
        tpu_util._CoreLocation(1, 2, 0, 1),
        tpu_util._CoreLocation(2, 2, 0, 0),
        tpu_util._CoreLocation(2, 2, 0, 1),
        tpu_util._CoreLocation(3, 2, 0, 0),
        tpu_util._CoreLocation(3, 2, 0, 1),
        tpu_util._CoreLocation(0, 3, 0, 0),
        tpu_util._CoreLocation(0, 3, 0, 1),
        tpu_util._CoreLocation(1, 3, 0, 0),
        tpu_util._CoreLocation(1, 3, 0, 1),
        tpu_util._CoreLocation(2, 3, 0, 0),
        tpu_util._CoreLocation(2, 3, 0, 1),
        tpu_util._CoreLocation(3, 3, 0, 0),
        tpu_util._CoreLocation(3, 3, 0, 1),
        tpu_util._CoreLocation(0, 0, 1, 0),
        tpu_util._CoreLocation(0, 0, 1, 1),
        tpu_util._CoreLocation(1, 0, 1, 0),
        tpu_util._CoreLocation(1, 0, 1, 1),
        tpu_util._CoreLocation(2, 0, 1, 0),
        tpu_util._CoreLocation(2, 0, 1, 1),
        tpu_util._CoreLocation(3, 0, 1, 0),
        tpu_util._CoreLocation(3, 0, 1, 1),
        tpu_util._CoreLocation(0, 1, 1, 0),
        tpu_util._CoreLocation(0, 1, 1, 1),
        tpu_util._CoreLocation(1, 1, 1, 0),
        tpu_util._CoreLocation(1, 1, 1, 1),
        tpu_util._CoreLocation(2, 1, 1, 0),
        tpu_util._CoreLocation(2, 1, 1, 1),
        tpu_util._CoreLocation(3, 1, 1, 0),
        tpu_util._CoreLocation(3, 1, 1, 1),
        tpu_util._CoreLocation(0, 2, 1, 0),
        tpu_util._CoreLocation(0, 2, 1, 1),
        tpu_util._CoreLocation(1, 2, 1, 0),
        tpu_util._CoreLocation(1, 2, 1, 1),
        tpu_util._CoreLocation(2, 2, 1, 0),
        tpu_util._CoreLocation(2, 2, 1, 1),
        tpu_util._CoreLocation(3, 2, 1, 0),
        tpu_util._CoreLocation(3, 2, 1, 1),
        tpu_util._CoreLocation(0, 3, 1, 0),
        tpu_util._CoreLocation(0, 3, 1, 1),
        tpu_util._CoreLocation(1, 3, 1, 0),
        tpu_util._CoreLocation(1, 3, 1, 1),
        tpu_util._CoreLocation(2, 3, 1, 0),
        tpu_util._CoreLocation(2, 3, 1, 1),
        tpu_util._CoreLocation(3, 3, 1, 0),
        tpu_util._CoreLocation(3, 3, 1, 1),
    ]
    expected = [
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(2, 0, 0, 0),
        tpu_util._CoreLocation(2, 0, 0, 1),
        tpu_util._CoreLocation(3, 0, 0, 0),
        tpu_util._CoreLocation(3, 0, 0, 1),
        tpu_util._CoreLocation(3, 1, 0, 0),
        tpu_util._CoreLocation(3, 1, 0, 1),
        tpu_util._CoreLocation(2, 1, 0, 0),
        tpu_util._CoreLocation(2, 1, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(1, 2, 0, 0),
        tpu_util._CoreLocation(1, 2, 0, 1),
        tpu_util._CoreLocation(2, 2, 0, 0),
        tpu_util._CoreLocation(2, 2, 0, 1),
        tpu_util._CoreLocation(3, 2, 0, 0),
        tpu_util._CoreLocation(3, 2, 0, 1),
        tpu_util._CoreLocation(3, 3, 0, 0),
        tpu_util._CoreLocation(3, 3, 0, 1),
        tpu_util._CoreLocation(2, 3, 0, 0),
        tpu_util._CoreLocation(2, 3, 0, 1),
        tpu_util._CoreLocation(1, 3, 0, 0),
        tpu_util._CoreLocation(1, 3, 0, 1),
        tpu_util._CoreLocation(0, 3, 0, 0),
        tpu_util._CoreLocation(0, 3, 0, 1),
        tpu_util._CoreLocation(0, 2, 0, 0),
        tpu_util._CoreLocation(0, 2, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(0, 1, 1, 1),
        tpu_util._CoreLocation(0, 1, 1, 0),
        tpu_util._CoreLocation(0, 2, 1, 1),
        tpu_util._CoreLocation(0, 2, 1, 0),
        tpu_util._CoreLocation(0, 3, 1, 1),
        tpu_util._CoreLocation(0, 3, 1, 0),
        tpu_util._CoreLocation(1, 3, 1, 1),
        tpu_util._CoreLocation(1, 3, 1, 0),
        tpu_util._CoreLocation(2, 3, 1, 1),
        tpu_util._CoreLocation(2, 3, 1, 0),
        tpu_util._CoreLocation(3, 3, 1, 1),
        tpu_util._CoreLocation(3, 3, 1, 0),
        tpu_util._CoreLocation(3, 2, 1, 1),
        tpu_util._CoreLocation(3, 2, 1, 0),
        tpu_util._CoreLocation(2, 2, 1, 1),
        tpu_util._CoreLocation(2, 2, 1, 0),
        tpu_util._CoreLocation(1, 2, 1, 1),
        tpu_util._CoreLocation(1, 2, 1, 0),
        tpu_util._CoreLocation(1, 1, 1, 1),
        tpu_util._CoreLocation(1, 1, 1, 0),
        tpu_util._CoreLocation(2, 1, 1, 1),
        tpu_util._CoreLocation(2, 1, 1, 0),
        tpu_util._CoreLocation(3, 1, 1, 1),
        tpu_util._CoreLocation(3, 1, 1, 0),
        tpu_util._CoreLocation(3, 0, 1, 1),
        tpu_util._CoreLocation(3, 0, 1, 0),
        tpu_util._CoreLocation(2, 0, 1, 1),
        tpu_util._CoreLocation(2, 0, 1, 0),
        tpu_util._CoreLocation(1, 0, 1, 1),
        tpu_util._CoreLocation(1, 0, 1, 0),
        tpu_util._CoreLocation(0, 0, 1, 0),
        tpu_util._CoreLocation(0, 0, 1, 1),
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
    ]
    result = self._build_all_reduce_ring(core_locations)
    self.assertAllEqual(result, expected)

  # Picture of chips:
  # 0 -- 1    4 -- 5
  # |    |    |    |
  # 3 -- 2    7 -- 6
  #
  # 12-- 13   8 -- 9
  # |    |    |    |
  # 15-- 14   11-- 10
  def testBuildOrthogonalAllReduceRings(self):
    core_locations = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(0, 2, 0, 0),
        tpu_util._CoreLocation(0, 2, 0, 1),
        tpu_util._CoreLocation(0, 3, 0, 0),
        tpu_util._CoreLocation(0, 3, 0, 1),
        tpu_util._CoreLocation(1, 2, 0, 0),
        tpu_util._CoreLocation(1, 2, 0, 1),
        tpu_util._CoreLocation(1, 3, 0, 0),
        tpu_util._CoreLocation(1, 3, 0, 1),
        tpu_util._CoreLocation(2, 0, 0, 0),
        tpu_util._CoreLocation(2, 0, 0, 1),
        tpu_util._CoreLocation(2, 1, 0, 0),
        tpu_util._CoreLocation(2, 1, 0, 1),
        tpu_util._CoreLocation(3, 0, 0, 0),
        tpu_util._CoreLocation(3, 0, 0, 1),
        tpu_util._CoreLocation(3, 1, 0, 0),
        tpu_util._CoreLocation(3, 1, 0, 1),
        tpu_util._CoreLocation(2, 2, 0, 0),
        tpu_util._CoreLocation(2, 2, 0, 1),
        tpu_util._CoreLocation(2, 3, 0, 0),
        tpu_util._CoreLocation(2, 3, 0, 1),
        tpu_util._CoreLocation(3, 2, 0, 0),
        tpu_util._CoreLocation(3, 2, 0, 1),
        tpu_util._CoreLocation(3, 3, 0, 0),
        tpu_util._CoreLocation(3, 3, 0, 1),
    ]
    expected = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(2, 0, 0, 0),
        tpu_util._CoreLocation(2, 0, 0, 1),
        tpu_util._CoreLocation(3, 0, 0, 0),
        tpu_util._CoreLocation(3, 0, 0, 1),
        tpu_util._CoreLocation(3, 1, 0, 0),
        tpu_util._CoreLocation(3, 1, 0, 1),
        tpu_util._CoreLocation(2, 1, 0, 0),
        tpu_util._CoreLocation(2, 1, 0, 1),
        tpu_util._CoreLocation(2, 2, 0, 0),
        tpu_util._CoreLocation(2, 2, 0, 1),
        tpu_util._CoreLocation(3, 2, 0, 0),
        tpu_util._CoreLocation(3, 2, 0, 1),
        tpu_util._CoreLocation(3, 3, 0, 0),
        tpu_util._CoreLocation(3, 3, 0, 1),
        tpu_util._CoreLocation(2, 3, 0, 0),
        tpu_util._CoreLocation(2, 3, 0, 1),
        tpu_util._CoreLocation(0, 2, 0, 0),
        tpu_util._CoreLocation(0, 2, 0, 1),
        tpu_util._CoreLocation(1, 2, 0, 0),
        tpu_util._CoreLocation(1, 2, 0, 1),
        tpu_util._CoreLocation(1, 3, 0, 0),
        tpu_util._CoreLocation(1, 3, 0, 1),
        tpu_util._CoreLocation(0, 3, 0, 0),
        tpu_util._CoreLocation(0, 3, 0, 1),
    ]
    result = tpu_util._build_orthogonal_rings(
        core_locations, ring_size=8, rotate_ring_across_rings=False)
    self.assertAllEqual(result, expected)

  # Picture of chips:
  # 0 -- 1    12 -- 13
  # |    |    |     |
  # 3 -- 2    15 -- 14
  #
  # 4 -- 5   8 -- 9
  # |    |    |    |
  # 7 -- 6   11-- 10
  def testBuildOrthogonalRotatedAllReduceRings(self):
    core_locations = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(0, 2, 0, 0),
        tpu_util._CoreLocation(0, 2, 0, 1),
        tpu_util._CoreLocation(0, 3, 0, 0),
        tpu_util._CoreLocation(0, 3, 0, 1),
        tpu_util._CoreLocation(1, 2, 0, 0),
        tpu_util._CoreLocation(1, 2, 0, 1),
        tpu_util._CoreLocation(1, 3, 0, 0),
        tpu_util._CoreLocation(1, 3, 0, 1),
        tpu_util._CoreLocation(2, 0, 0, 0),
        tpu_util._CoreLocation(2, 0, 0, 1),
        tpu_util._CoreLocation(2, 1, 0, 0),
        tpu_util._CoreLocation(2, 1, 0, 1),
        tpu_util._CoreLocation(3, 0, 0, 0),
        tpu_util._CoreLocation(3, 0, 0, 1),
        tpu_util._CoreLocation(3, 1, 0, 0),
        tpu_util._CoreLocation(3, 1, 0, 1),
        tpu_util._CoreLocation(2, 2, 0, 0),
        tpu_util._CoreLocation(2, 2, 0, 1),
        tpu_util._CoreLocation(2, 3, 0, 0),
        tpu_util._CoreLocation(2, 3, 0, 1),
        tpu_util._CoreLocation(3, 2, 0, 0),
        tpu_util._CoreLocation(3, 2, 0, 1),
        tpu_util._CoreLocation(3, 3, 0, 0),
        tpu_util._CoreLocation(3, 3, 0, 1),
    ]
    expected = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(0, 2, 0, 0),
        tpu_util._CoreLocation(0, 2, 0, 1),
        tpu_util._CoreLocation(1, 2, 0, 0),
        tpu_util._CoreLocation(1, 2, 0, 1),
        tpu_util._CoreLocation(1, 3, 0, 0),
        tpu_util._CoreLocation(1, 3, 0, 1),
        tpu_util._CoreLocation(0, 3, 0, 0),
        tpu_util._CoreLocation(0, 3, 0, 1),
        tpu_util._CoreLocation(2, 2, 0, 0),
        tpu_util._CoreLocation(2, 2, 0, 1),
        tpu_util._CoreLocation(3, 2, 0, 0),
        tpu_util._CoreLocation(3, 2, 0, 1),
        tpu_util._CoreLocation(3, 3, 0, 0),
        tpu_util._CoreLocation(3, 3, 0, 1),
        tpu_util._CoreLocation(2, 3, 0, 0),
        tpu_util._CoreLocation(2, 3, 0, 1),
        tpu_util._CoreLocation(2, 0, 0, 0),
        tpu_util._CoreLocation(2, 0, 0, 1),
        tpu_util._CoreLocation(3, 0, 0, 0),
        tpu_util._CoreLocation(3, 0, 0, 1),
        tpu_util._CoreLocation(3, 1, 0, 0),
        tpu_util._CoreLocation(3, 1, 0, 1),
        tpu_util._CoreLocation(2, 1, 0, 0),
        tpu_util._CoreLocation(2, 1, 0, 1),
    ]
    result = tpu_util._build_orthogonal_rings(
        core_locations, ring_size=8, rotate_ring_across_rings=True)
    self.assertAllEqual(result, expected)

  # Create a 4x8 mesh on a 4x4 DF slice, disallowing splitting hosts.
  def testCreateDFMeshNoSplittingHosts(self):
    result = tpu_util._enumerate_core_locations(
        [4, 4, 1, 2], [4, 4, 1, 2], ['core', 'y', 'z', 'x'],
        can_split_host_across_rings=False,
        ring_size=8)
    expected = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(0, 2, 0, 0),
        tpu_util._CoreLocation(0, 2, 0, 1),
        tpu_util._CoreLocation(0, 3, 0, 0),
        tpu_util._CoreLocation(0, 3, 0, 1),
        tpu_util._CoreLocation(1, 2, 0, 0),
        tpu_util._CoreLocation(1, 2, 0, 1),
        tpu_util._CoreLocation(1, 3, 0, 0),
        tpu_util._CoreLocation(1, 3, 0, 1),
        tpu_util._CoreLocation(2, 0, 0, 0),
        tpu_util._CoreLocation(2, 0, 0, 1),
        tpu_util._CoreLocation(2, 1, 0, 0),
        tpu_util._CoreLocation(2, 1, 0, 1),
        tpu_util._CoreLocation(3, 0, 0, 0),
        tpu_util._CoreLocation(3, 0, 0, 1),
        tpu_util._CoreLocation(3, 1, 0, 0),
        tpu_util._CoreLocation(3, 1, 0, 1),
        tpu_util._CoreLocation(2, 2, 0, 0),
        tpu_util._CoreLocation(2, 2, 0, 1),
        tpu_util._CoreLocation(2, 3, 0, 0),
        tpu_util._CoreLocation(2, 3, 0, 1),
        tpu_util._CoreLocation(3, 2, 0, 0),
        tpu_util._CoreLocation(3, 2, 0, 1),
        tpu_util._CoreLocation(3, 3, 0, 0),
        tpu_util._CoreLocation(3, 3, 0, 1),
    ]
    self.assertAllEqual(result, expected)

  # Create a 4x8 mesh on a 4x4 DF slice with at most 2, 2, 1, 2 devices from
  # each dimension, disallowing splitting hosts.
  def testCreateDFMeshWithRingBoundsNoSplittingHosts(self):
    result = tpu_util._enumerate_core_locations(
        [4, 4, 1, 2], [2, 2, 1, 2], ['core', 'x', 'y', 'z'],
        can_split_host_across_rings=False,
        ring_size=8)
    expected = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(2, 0, 0, 0),
        tpu_util._CoreLocation(2, 0, 0, 1),
        tpu_util._CoreLocation(3, 0, 0, 0),
        tpu_util._CoreLocation(3, 0, 0, 1),
        tpu_util._CoreLocation(2, 1, 0, 0),
        tpu_util._CoreLocation(2, 1, 0, 1),
        tpu_util._CoreLocation(3, 1, 0, 0),
        tpu_util._CoreLocation(3, 1, 0, 1),
        tpu_util._CoreLocation(0, 2, 0, 0),
        tpu_util._CoreLocation(0, 2, 0, 1),
        tpu_util._CoreLocation(1, 2, 0, 0),
        tpu_util._CoreLocation(1, 2, 0, 1),
        tpu_util._CoreLocation(0, 3, 0, 0),
        tpu_util._CoreLocation(0, 3, 0, 1),
        tpu_util._CoreLocation(1, 3, 0, 0),
        tpu_util._CoreLocation(1, 3, 0, 1),
        tpu_util._CoreLocation(2, 2, 0, 0),
        tpu_util._CoreLocation(2, 2, 0, 1),
        tpu_util._CoreLocation(3, 2, 0, 0),
        tpu_util._CoreLocation(3, 2, 0, 1),
        tpu_util._CoreLocation(2, 3, 0, 0),
        tpu_util._CoreLocation(2, 3, 0, 1),
        tpu_util._CoreLocation(3, 3, 0, 0),
        tpu_util._CoreLocation(3, 3, 0, 1),
    ]
    self.assertAllEqual(result, expected)

  # Create a 4x8 mesh on a 4x4 DF slice, allowing splitting hosts.
  def testCreateDFMeshSplittingHosts(self):
    result = tpu_util._enumerate_core_locations(
        [4, 4, 1, 2], [4, 4, 1, 2], ['core', 'y', 'z', 'x'],
        can_split_host_across_rings=True,
        ring_size=8)
    expected = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(0, 2, 0, 0),
        tpu_util._CoreLocation(0, 2, 0, 1),
        tpu_util._CoreLocation(0, 3, 0, 0),
        tpu_util._CoreLocation(0, 3, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(1, 2, 0, 0),
        tpu_util._CoreLocation(1, 2, 0, 1),
        tpu_util._CoreLocation(1, 3, 0, 0),
        tpu_util._CoreLocation(1, 3, 0, 1),
        tpu_util._CoreLocation(2, 0, 0, 0),
        tpu_util._CoreLocation(2, 0, 0, 1),
        tpu_util._CoreLocation(2, 1, 0, 0),
        tpu_util._CoreLocation(2, 1, 0, 1),
        tpu_util._CoreLocation(2, 2, 0, 0),
        tpu_util._CoreLocation(2, 2, 0, 1),
        tpu_util._CoreLocation(2, 3, 0, 0),
        tpu_util._CoreLocation(2, 3, 0, 1),
        tpu_util._CoreLocation(3, 0, 0, 0),
        tpu_util._CoreLocation(3, 0, 0, 1),
        tpu_util._CoreLocation(3, 1, 0, 0),
        tpu_util._CoreLocation(3, 1, 0, 1),
        tpu_util._CoreLocation(3, 2, 0, 0),
        tpu_util._CoreLocation(3, 2, 0, 1),
        tpu_util._CoreLocation(3, 3, 0, 0),
        tpu_util._CoreLocation(3, 3, 0, 1),
    ]
    self.assertAllEqual(result, expected)

  # Create a 2x64 mesh on a 4x4x4 PF slice, allowing splitting hosts.
  def testCreateMeshPFSplittingHosts(self):
    result = tpu_util._enumerate_core_locations(
        [4, 4, 4, 2], [4, 4, 4, 2], ['core', 'x', 'y', 'z'],
        can_split_host_across_rings=True,
        ring_size=64)
    expected = [
        tpu_util._CoreLocation(0, 0, 0, 0),
        tpu_util._CoreLocation(0, 0, 0, 1),
        tpu_util._CoreLocation(1, 0, 0, 0),
        tpu_util._CoreLocation(1, 0, 0, 1),
        tpu_util._CoreLocation(2, 0, 0, 0),
        tpu_util._CoreLocation(2, 0, 0, 1),
        tpu_util._CoreLocation(3, 0, 0, 0),
        tpu_util._CoreLocation(3, 0, 0, 1),
        tpu_util._CoreLocation(0, 1, 0, 0),
        tpu_util._CoreLocation(0, 1, 0, 1),
        tpu_util._CoreLocation(1, 1, 0, 0),
        tpu_util._CoreLocation(1, 1, 0, 1),
        tpu_util._CoreLocation(2, 1, 0, 0),
        tpu_util._CoreLocation(2, 1, 0, 1),
        tpu_util._CoreLocation(3, 1, 0, 0),
        tpu_util._CoreLocation(3, 1, 0, 1),
        tpu_util._CoreLocation(0, 2, 0, 0),
        tpu_util._CoreLocation(0, 2, 0, 1),
        tpu_util._CoreLocation(1, 2, 0, 0),
        tpu_util._CoreLocation(1, 2, 0, 1),
        tpu_util._CoreLocation(2, 2, 0, 0),
        tpu_util._CoreLocation(2, 2, 0, 1),
        tpu_util._CoreLocation(3, 2, 0, 0),
        tpu_util._CoreLocation(3, 2, 0, 1),
        tpu_util._CoreLocation(0, 3, 0, 0),
        tpu_util._CoreLocation(0, 3, 0, 1),
        tpu_util._CoreLocation(1, 3, 0, 0),
        tpu_util._CoreLocation(1, 3, 0, 1),
        tpu_util._CoreLocation(2, 3, 0, 0),
        tpu_util._CoreLocation(2, 3, 0, 1),
        tpu_util._CoreLocation(3, 3, 0, 0),
        tpu_util._CoreLocation(3, 3, 0, 1),
        tpu_util._CoreLocation(0, 0, 1, 0),
        tpu_util._CoreLocation(0, 0, 1, 1),
        tpu_util._CoreLocation(1, 0, 1, 0),
        tpu_util._CoreLocation(1, 0, 1, 1),
        tpu_util._CoreLocation(2, 0, 1, 0),
        tpu_util._CoreLocation(2, 0, 1, 1),
        tpu_util._CoreLocation(3, 0, 1, 0),
        tpu_util._CoreLocation(3, 0, 1, 1),
        tpu_util._CoreLocation(0, 1, 1, 0),
        tpu_util._CoreLocation(0, 1, 1, 1),
        tpu_util._CoreLocation(1, 1, 1, 0),
        tpu_util._CoreLocation(1, 1, 1, 1),
        tpu_util._CoreLocation(2, 1, 1, 0),
        tpu_util._CoreLocation(2, 1, 1, 1),
        tpu_util._CoreLocation(3, 1, 1, 0),
        tpu_util._CoreLocation(3, 1, 1, 1),
        tpu_util._CoreLocation(0, 2, 1, 0),
        tpu_util._CoreLocation(0, 2, 1, 1),
        tpu_util._CoreLocation(1, 2, 1, 0),
        tpu_util._CoreLocation(1, 2, 1, 1),
        tpu_util._CoreLocation(2, 2, 1, 0),
        tpu_util._CoreLocation(2, 2, 1, 1),
        tpu_util._CoreLocation(3, 2, 1, 0),
        tpu_util._CoreLocation(3, 2, 1, 1),
        tpu_util._CoreLocation(0, 3, 1, 0),
        tpu_util._CoreLocation(0, 3, 1, 1),
        tpu_util._CoreLocation(1, 3, 1, 0),
        tpu_util._CoreLocation(1, 3, 1, 1),
        tpu_util._CoreLocation(2, 3, 1, 0),
        tpu_util._CoreLocation(2, 3, 1, 1),
        tpu_util._CoreLocation(3, 3, 1, 0),
        tpu_util._CoreLocation(3, 3, 1, 1),
        tpu_util._CoreLocation(0, 0, 2, 0),
        tpu_util._CoreLocation(0, 0, 2, 1),
        tpu_util._CoreLocation(1, 0, 2, 0),
        tpu_util._CoreLocation(1, 0, 2, 1),
        tpu_util._CoreLocation(2, 0, 2, 0),
        tpu_util._CoreLocation(2, 0, 2, 1),
        tpu_util._CoreLocation(3, 0, 2, 0),
        tpu_util._CoreLocation(3, 0, 2, 1),
        tpu_util._CoreLocation(0, 1, 2, 0),
        tpu_util._CoreLocation(0, 1, 2, 1),
        tpu_util._CoreLocation(1, 1, 2, 0),
        tpu_util._CoreLocation(1, 1, 2, 1),
        tpu_util._CoreLocation(2, 1, 2, 0),
        tpu_util._CoreLocation(2, 1, 2, 1),
        tpu_util._CoreLocation(3, 1, 2, 0),
        tpu_util._CoreLocation(3, 1, 2, 1),
        tpu_util._CoreLocation(0, 2, 2, 0),
        tpu_util._CoreLocation(0, 2, 2, 1),
        tpu_util._CoreLocation(1, 2, 2, 0),
        tpu_util._CoreLocation(1, 2, 2, 1),
        tpu_util._CoreLocation(2, 2, 2, 0),
        tpu_util._CoreLocation(2, 2, 2, 1),
        tpu_util._CoreLocation(3, 2, 2, 0),
        tpu_util._CoreLocation(3, 2, 2, 1),
        tpu_util._CoreLocation(0, 3, 2, 0),
        tpu_util._CoreLocation(0, 3, 2, 1),
        tpu_util._CoreLocation(1, 3, 2, 0),
        tpu_util._CoreLocation(1, 3, 2, 1),
        tpu_util._CoreLocation(2, 3, 2, 0),
        tpu_util._CoreLocation(2, 3, 2, 1),
        tpu_util._CoreLocation(3, 3, 2, 0),
        tpu_util._CoreLocation(3, 3, 2, 1),
        tpu_util._CoreLocation(0, 0, 3, 0),
        tpu_util._CoreLocation(0, 0, 3, 1),
        tpu_util._CoreLocation(1, 0, 3, 0),
        tpu_util._CoreLocation(1, 0, 3, 1),
        tpu_util._CoreLocation(2, 0, 3, 0),
        tpu_util._CoreLocation(2, 0, 3, 1),
        tpu_util._CoreLocation(3, 0, 3, 0),
        tpu_util._CoreLocation(3, 0, 3, 1),
        tpu_util._CoreLocation(0, 1, 3, 0),
        tpu_util._CoreLocation(0, 1, 3, 1),
        tpu_util._CoreLocation(1, 1, 3, 0),
        tpu_util._CoreLocation(1, 1, 3, 1),
        tpu_util._CoreLocation(2, 1, 3, 0),
        tpu_util._CoreLocation(2, 1, 3, 1),
        tpu_util._CoreLocation(3, 1, 3, 0),
        tpu_util._CoreLocation(3, 1, 3, 1),
        tpu_util._CoreLocation(0, 2, 3, 0),
        tpu_util._CoreLocation(0, 2, 3, 1),
        tpu_util._CoreLocation(1, 2, 3, 0),
        tpu_util._CoreLocation(1, 2, 3, 1),
        tpu_util._CoreLocation(2, 2, 3, 0),
        tpu_util._CoreLocation(2, 2, 3, 1),
        tpu_util._CoreLocation(3, 2, 3, 0),
        tpu_util._CoreLocation(3, 2, 3, 1),
        tpu_util._CoreLocation(0, 3, 3, 0),
        tpu_util._CoreLocation(0, 3, 3, 1),
        tpu_util._CoreLocation(1, 3, 3, 0),
        tpu_util._CoreLocation(1, 3, 3, 1),
        tpu_util._CoreLocation(2, 3, 3, 0),
        tpu_util._CoreLocation(2, 3, 3, 1),
        tpu_util._CoreLocation(3, 3, 3, 0),
        tpu_util._CoreLocation(3, 3, 3, 1),
    ]
    self.assertAllEqual(result, expected)

  def testCreateMeshNoSplittingHostsUnfulfillable(self):
    with self.assertRaises(ValueError):
      tpu_util.create_tpu_mesh(['x', 'y'], [2, 1],
                               'mesh_unfulfillable_without_splitting_hosts',
                               can_split_host_across_rings=False)

  def testCreateMeshWithDefaultOptions(self):
    mesh = tpu_util.create_tpu_mesh(['x'], [2], 'mesh_with_default_options')
    self.assertAllEqual(mesh.shape(), [2])
    self.assertEqual(mesh.num_local_devices(), 2)

  def testCreateMeshWithWrongShape(self):
    with self.assertRaises(ValueError):
      tpu_util.create_tpu_mesh(['x'], [1], 'mesh_with_wrong_shape')

  # Build rings for the batch dimension.
  def testCreateMeshWithPositiveRingDims(self):
    mesh = tpu_util.create_tpu_mesh(['x', 'y'], [2, 1],
                                    'mesh_with_positive_ring_dims',
                                    ring_dims=1)
    self.assertAllEqual(mesh.shape(), [2, 1])
    self.assertEqual(mesh.num_local_devices(), 2)

  # Build rings for all non-batch dimensions.
  def testCreateMeshWithNegativeRingDims(self):
    mesh = tpu_util.create_tpu_mesh(['x', 'y', 'z'], [1, 2, 1],
                                    'mesh_with_negative_ring_dims',
                                    ring_dims=-2)
    self.assertAllEqual(mesh.shape(), [1, 2, 1])
    self.assertEqual(mesh.num_local_devices(), 2)

  # Build single-core rings.
  def testCreateMeshWithZeroRingDims(self):
    mesh = tpu_util.create_tpu_mesh(['x', 'y'], [2, 1],
                                    'mesh_with_zero_ring_dims',
                                    ring_dims=0)
    self.assertAllEqual(mesh.shape(), [2, 1])
    self.assertEqual(mesh.num_local_devices(), 2)

  def testCreateMeshWithCustomAxes(self):
    mesh = tpu_util.create_tpu_mesh(['x', 'y'], [2, 1],
                                    'mesh_with_custom_axes',
                                    ring_axes=['x', 'z', 'y', 'core'])
    self.assertAllEqual(mesh.shape(), [2, 1])
    self.assertEqual(mesh.num_local_devices(), 2)

  # More cores (2 cores) on the first axis (core) than ring size (1).
  def testCreateMeshWithDividedAxis(self):
    mesh = tpu_util.create_tpu_mesh(['x', 'y'], [2, 1],
                                    'mesh_with_divided_axis',
                                    ring_dims=-1,
                                    ring_axes=['core', 'z', 'y', 'x'])
    self.assertAllEqual(mesh.shape(), [2, 1])
    self.assertEqual(mesh.num_local_devices(), 2)

  # Both meshes should produce the same result despite different `ring_dim`.
  def testCreateMultipleMeshes(self):
    a = constant_op.constant([[0, 1], [2, 3]], dtype=dtypes.int32)
    b_expected = math_ops.reduce_sum(a)

    mesh_1 = tpu_util.create_tpu_mesh(['x', 'y'], [2, 1], 'mesh_1', ring_dims=1)
    a_1 = numpy_util.pack_numpy(a, Layout(['x', 'y'], mesh_1))
    b_1 = math_ops.reduce_sum(a_1)
    self.assertDTensorEqual(b_expected, Layout.replicated(mesh_1, rank=0), b_1)

    mesh_2 = tpu_util.create_tpu_mesh(['x', 'y'], [2, 1],
                                      'mesh_2',
                                      ring_dims=-1)
    a_2 = numpy_util.pack_numpy(a, Layout(['x', 'y'], mesh_2))
    b_2 = math_ops.reduce_sum(a_2)
    self.assertDTensorEqual(b_expected, Layout.replicated(mesh_2, rank=0), b_2)

  def testCreateMeshWithEmptyName(self):
    tpu_util.create_tpu_mesh(['x'], [2], '')

  def testCreateMeshWithExistingName(self):
    tpu_util.create_tpu_mesh(['x'], [2], 'mesh_with_existing_name')
    with self.assertRaises(ValueError):
      tpu_util.create_tpu_mesh(['x'], [2], 'mesh_with_existing_name')

  def testGetDeviceIDs(self):
    mesh = tpu_util.create_tpu_mesh(['x', 'y'], [2, 1],
                                    'mesh_to_get_device_ids')
    self.assertAllEqual(tpu_util.get_device_ids(mesh), [0, 1])

  def testGetDeviceLocations(self):
    mesh = tpu_util.create_tpu_mesh(['x', 'y'], [2, 1],
                                    'mesh_to_get_device_locations')
    self.assertAllEqual(
        tpu_util.get_device_locations(mesh), [{
            'x': 0,
            'y': 0
        }, {
            'x': 1,
            'y': 0
        }])


if __name__ == '__main__':
  test.main()
