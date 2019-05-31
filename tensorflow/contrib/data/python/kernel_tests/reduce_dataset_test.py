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
# ==============================================================================
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.data.python.ops import get_single_element
from tensorflow.contrib.data.python.ops import grouping
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ReduceDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("SumZero", 0),
      ("SumOne", 1),
      ("SumFive", 5),
      ("SumTen", 10),
  )
  def testReduceDataset(self, stop):
    def init_fn(_):
      return np.int64(0)

    def reduce_fn(state, value):
      return state + value

    def finalize_fn(state):
      return state

    sum_reducer = grouping.Reducer(init_fn, reduce_fn, finalize_fn)

    dataset = dataset_ops.Dataset.range(stop)
    element = get_single_element.reduce_dataset(dataset, sum_reducer)

    self.assertEqual(stop * (stop - 1) / 2, self.evaluate(element))


if __name__ == "__main__":
  test.main()
