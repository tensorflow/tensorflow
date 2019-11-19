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
"""Tests for `tf.data.Dataset.take()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class TakeTest(test_base.DatasetTestBase):

  def testTakeTensorDataset(self):
    components = (np.arange(10),)

    def do_test(count):
      dataset = dataset_ops.Dataset.from_tensor_slices(components).take(count)
      self.assertEqual(
          [c.shape[1:] for c in components],
          [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
      num_output = min(count, 10) if count != -1 else 10
      self.assertDatasetProduces(
          dataset, [tuple(components[0][i:i + 1]) for i in range(num_output)])

    # Take fewer than input size
    do_test(4)

    # Take more than input size
    do_test(25)

    # Take all of input
    do_test(-1)

    # Take nothing
    do_test(0)

if __name__ == "__main__":
  test.main()
