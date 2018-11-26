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

import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ZipDatasetTest(test_base.DatasetTestBase):

  def testZipDataset(self):

    def dataset_fn(components):
      datasets = tuple([
          dataset_ops.Dataset.from_tensor_slices(component)
          for component in components
      ])
      return dataset_ops.Dataset.zip(datasets)

    equal_length_components = [
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 22),
        np.array([37.0, 38.0, 39.0, 40.0])
    ]

    get_next = self.getNext(dataset_fn(equal_length_components))
    for i in range(4):
      results = self.evaluate(get_next())
      for component, result_component in zip(equal_length_components, results):
        self.assertAllEqual(component[i], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

    variable_length_components = [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1.0, 2.0]]
    get_next = self.getNext(dataset_fn(variable_length_components))
    for i in range(2):
      results = self.evaluate(get_next())
      for component, result_component in zip(variable_length_components,
                                             results):
        self.assertAllEqual(component[i], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  def testNestedZipDataset(self):

    equal_length_components = [
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 22),
        np.array([37.0, 38.0, 39.0, 40.0])
    ]
    datasets = [
        dataset_ops.Dataset.from_tensor_slices(component)
        for component in equal_length_components
    ]
    dataset = dataset_ops.Dataset.zip((datasets[0], (datasets[1], datasets[2])))

    self.assertEqual(
        dataset.output_shapes,
        (tensor_shape.TensorShape([20]),
         (tensor_shape.TensorShape([22]), tensor_shape.TensorShape([]))))

    get_next = self.getNext(dataset)
    for i in range(4):
      result1, (result2, result3) = self.evaluate(get_next())
      self.assertAllEqual(equal_length_components[0][i], result1)
      self.assertAllEqual(equal_length_components[1][i], result2)
      self.assertAllEqual(equal_length_components[2][i], result3)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())


if __name__ == "__main__":
  test.main()
