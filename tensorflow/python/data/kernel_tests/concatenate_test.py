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
"""Tests for `tf.data.Dataset.concatenate()."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ConcatenateTest(test_base.DatasetTestBase):

  def testConcatenateDataset(self):
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 15),
        np.array([37.0, 38.0, 39.0, 40.0]))
    to_concatenate_components = (
        np.tile(np.array([[1], [2], [3], [4], [5]]), 20),
        np.tile(np.array([[12], [13], [14], [15], [16]]), 15),
        np.array([37.0, 38.0, 39.0, 40.0, 41.0]))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)
    concatenated = input_dataset.concatenate(dataset_to_concatenate)
    self.assertEqual(concatenated.output_shapes, (tensor_shape.TensorShape(
        [20]), tensor_shape.TensorShape([15]), tensor_shape.TensorShape([])))

    get_next = self.getNext(concatenated)

    for i in range(9):
      result = self.evaluate(get_next())
      if i < 4:
        for component, result_component in zip(input_components, result):
          self.assertAllEqual(component[i], result_component)
      else:
        for component, result_component in zip(to_concatenate_components,
                                               result):
          self.assertAllEqual(component[i - 4], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  def testConcatenateDatasetDifferentShape(self):
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (
        np.tile(np.array([[1], [2], [3], [4], [5]]), 20),
        np.tile(np.array([[12], [13], [14], [15], [16]]), 15))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)
    concatenated = input_dataset.concatenate(dataset_to_concatenate)
    self.assertEqual(
        [ts.as_list()
         for ts in nest.flatten(concatenated.output_shapes)], [[20], [None]])
    get_next = self.getNext(concatenated)
    for i in range(9):
      result = self.evaluate(get_next())
      if i < 4:
        for component, result_component in zip(input_components, result):
          self.assertAllEqual(component[i], result_component)
      else:
        for component, result_component in zip(to_concatenate_components,
                                               result):
          self.assertAllEqual(component[i - 4], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  def testConcatenateDatasetDifferentStructure(self):
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 5),
        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (
        np.tile(np.array([[1], [2], [3], [4], [5]]), 20),
        np.tile(np.array([[12], [13], [14], [15], [16]]), 15),
        np.array([37.0, 38.0, 39.0, 40.0, 41.0]))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)

    with self.assertRaisesRegexp(TypeError, "have different types"):
      input_dataset.concatenate(dataset_to_concatenate)

  def testConcatenateDatasetDifferentKeys(self):
    input_components = {
        "foo": np.array([[1], [2], [3], [4]]),
        "bar": np.array([[12], [13], [14], [15]])
    }
    to_concatenate_components = {
        "foo": np.array([[1], [2], [3], [4]]),
        "baz": np.array([[5], [6], [7], [8]])
    }

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)

    with self.assertRaisesRegexp(TypeError, "have different types"):
      input_dataset.concatenate(dataset_to_concatenate)

  def testConcatenateDatasetDifferentType(self):
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 5),
        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (
        np.tile(np.array([[1.0], [2.0], [3.0], [4.0]]), 5),
        np.tile(np.array([[12], [13], [14], [15]]), 15))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)

    with self.assertRaisesRegexp(TypeError, "have different types"):
      input_dataset.concatenate(dataset_to_concatenate)


if __name__ == "__main__":
  test.main()
