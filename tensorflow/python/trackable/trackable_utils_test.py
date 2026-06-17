# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for trackable_utils."""

from tensorflow.python.eager import test
from tensorflow.python.trackable import trackable_utils


class TrackableUtilsTest(test.TestCase):

  def test_order_by_dependency(self):
    """Tests order_by_dependency correctness."""

    # Visual graph (vertical lines point down, so 1 depends on 2):
    #    1
    #  /   \
    # 2 --> 3 <-- 4
    #       |
    #       5
    # One possible order: [5, 3, 4, 2, 1]
    dependencies = {1: [2, 3], 2: [3], 3: [5], 4: [3], 5: []}

    sorted_arr = list(trackable_utils.order_by_dependency(dependencies))
    indices = {x: sorted_arr.index(x) for x in range(1, 6)}
    self.assertEqual(indices[5], 0)
    self.assertEqual(indices[3], 1)
    self.assertGreater(indices[1], indices[2])  # 2 must appear before 1

  def test_order_by_no_dependency(self):
    sorted_arr = list(trackable_utils.order_by_dependency(
        {x: [] for x in range(15)}))
    self.assertEqual(set(sorted_arr), set(range(15)))

  def test_order_by_dependency_invalid_map(self):
    with self.assertRaisesRegex(
        ValueError, "Found values in the dependency map which are not keys"):
      trackable_utils.order_by_dependency({1: [2]})


if __name__ == "__main__":
  test.main()

