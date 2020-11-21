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
"""Tests for `tf.data.Dataset.from_tensors()."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


class FromTensorsTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensors(self):
    """Test a dataset that represents a single tuple of tensors."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))

    dataset = dataset_ops.Dataset.from_tensors(components)

    self.assertEqual(
        [c.shape for c in components],
        nest.flatten(dataset_ops.get_legacy_output_shapes(dataset)))

    self.assertDatasetProduces(dataset, expected_output=[components])

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsDataset(self):
    """Test a dataset that represents a dataset."""
    dataset = dataset_ops.Dataset.from_tensors(dataset_ops.Dataset.range(10))
    dataset = dataset.flat_map(lambda x: x)
    self.assertDatasetProduces(dataset, expected_output=range(10))

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsTensorArray(self):
    """Test a dataset that represents a TensorArray."""
    components = (
        tensor_array_ops.TensorArray(dtypes.float32, element_shape=(), size=2)
        .unstack([1.0, 2.0]))

    dataset = dataset_ops.Dataset.from_tensors(components)

    self.assertDatasetProduces(
        dataset, expected_output=[[1.0, 2.0]], requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsSparse(self):
    """Test a dataset that represents a single tuple of tensors."""
    components = (sparse_tensor.SparseTensorValue(
        indices=np.array([[0]]),
        values=np.array([0]),
        dense_shape=np.array([1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1]]),
                      values=np.array([-1, 1]),
                      dense_shape=np.array([2, 2])))

    dataset = dataset_ops.Dataset.from_tensors(components)

    self.assertEqual(
        [tensor_shape.TensorShape(c.dense_shape) for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    self.assertDatasetProduces(dataset, expected_output=[components])

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsMixed(self):
    """Test an dataset that represents a single tuple of tensors."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0]]),
                      values=np.array([0]),
                      dense_shape=np.array([1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1]]),
                      values=np.array([-1, 1]),
                      dense_shape=np.array([2, 2])))

    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEqual([
        tensor_shape.TensorShape(c.dense_shape)
        if sparse_tensor.is_sparse(c) else c.shape for c in components
    ], [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])

    self.assertDatasetProduces(dataset, expected_output=[components])

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsRagged(self):
    components = (
        ragged_factory_ops.constant_value([[[0]], [[1]], [[2]]]),
        ragged_factory_ops.constant_value([[[3]], [[4]], [[5]]]),
    )

    dataset = dataset_ops.Dataset.from_tensors(components)

    self.assertDatasetProduces(dataset, expected_output=[components])

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsMixedRagged(self):
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0]]),
                      values=np.array([0]),
                      dense_shape=np.array([1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1]]),
                      values=np.array([-1, 1]),
                      dense_shape=np.array([2, 2])),
                  ragged_factory_ops.constant_value([[[0]], [[1]], [[2]]]))

    dataset = dataset_ops.Dataset.from_tensors(components)

    self.assertDatasetProduces(dataset, expected_output=[components])

  @combinations.generate(
      combinations.combine(
          tf_api_version=[1],
          mode=["graph"],
          components=(np.array([1, 2, 3], dtype=np.int64),
                      (np.array([4., 5.]), np.array(
                          [6., 7.])), np.array([8, 9, 10], dtype=np.int64)),
          expected_shapes=[[[None, 3], [None, 3], [None, 2], [None, 2]]]) +
      combinations.combine(
          tf_api_version=[1],
          mode=["eager"],
          components=(np.array([1, 2, 3], dtype=np.int64),
                      (np.array([4., 5.]), np.array(
                          [6., 7.])), np.array([8, 9, 10], dtype=np.int64)),
          expected_shapes=[[[1, 3], [1, 3], [1, 2], [1, 2]]]))
  def testNestedStructure(self, components, expected_shapes):
    dataset = dataset_ops.Dataset.from_tensors(components)
    dataset = dataset.map(lambda x, y, z: ((x, z), (y[0], y[1])))

    dataset = dataset.flat_map(
        lambda x, y: dataset_ops.Dataset.from_tensors(
            ((x[0], x[1]), (y[0], y[1])))).batch(32)

    get_next = self.getNext(dataset)
    (w, x), (y, z) = get_next()
    self.assertEqual(dtypes.int64, w.dtype)
    self.assertEqual(dtypes.int64, x.dtype)
    self.assertEqual(dtypes.float64, y.dtype)
    self.assertEqual(dtypes.float64, z.dtype)
    self.assertEqual(expected_shapes, [
        w.shape.as_list(),
        x.shape.as_list(),
        y.shape.as_list(),
        z.shape.as_list()
    ])

    get_next = self.getNext(dataset)
    (w, x), (y, z) = get_next()
    self.assertEqual(dtypes.int64, w.dtype)
    self.assertEqual(dtypes.int64, x.dtype)
    self.assertEqual(dtypes.float64, y.dtype)
    self.assertEqual(dtypes.float64, z.dtype)
    self.assertEqual(expected_shapes, [
        w.shape.as_list(),
        x.shape.as_list(),
        y.shape.as_list(),
        z.shape.as_list()
    ])

  @combinations.generate(test_base.default_test_combinations())
  def testNestedDict(self):
    components = {"a": {"aa": 1, "ab": [2.0, 2.0]}, "b": [3, 3, 3]}
    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEqual(dtypes.int32,
                     dataset_ops.get_legacy_output_types(dataset)["a"]["aa"])
    self.assertEqual(dtypes.float32,
                     dataset_ops.get_legacy_output_types(dataset)["a"]["ab"])
    self.assertEqual(dtypes.int32,
                     dataset_ops.get_legacy_output_types(dataset)["b"])
    self.assertEqual([],
                     dataset_ops.get_legacy_output_shapes(dataset)["a"]["aa"])
    self.assertEqual([2],
                     dataset_ops.get_legacy_output_shapes(dataset)["a"]["ab"])
    self.assertEqual([3],
                     dataset_ops.get_legacy_output_shapes(dataset)["b"])

  @combinations.generate(test_base.default_test_combinations())
  def testNonSequenceNestedStructure(self):
    components = np.array([1, 2, 3], dtype=np.int64)

    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEqual(dtypes.int64,
                     dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual([3], dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.filter(
        lambda x: math_ops.reduce_all(math_ops.equal(x, components)))
    self.assertEqual(dtypes.int64,
                     dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual([3], dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.map(lambda x: array_ops.stack([x, x]))
    self.assertEqual(dtypes.int64,
                     dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual([2, 3], dataset_ops.get_legacy_output_shapes(dataset))

    dataset = dataset.flat_map(
        lambda x: dataset_ops.Dataset.from_tensor_slices(x))
    self.assertEqual(dtypes.int64,
                     dataset_ops.get_legacy_output_types(dataset))
    self.assertEqual([3], dataset_ops.get_legacy_output_shapes(dataset))

    get_next = self.getNext(dataset)
    self.assertEqual(dtypes.int64, get_next().dtype)
    self.assertEqual([3], get_next().shape)

  # TODO(b/121264236): needs mechanism for multiple device in eager mode.
  @combinations.generate(test_base.graph_only_combinations())
  def testSplitPipeline(self):
    with session.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:

      dataset = dataset_ops.Dataset.from_tensors(0)

      # Define a pipeline that attempts to use variables on two
      # different devices.
      #
      # Initialize the variables before creating to iterator, to avoid the
      # placement algorithm overriding the DT_RESOURCE colocation constraints.
      with ops.device("/cpu:0"):
        var_0 = resource_variable_ops.ResourceVariable(initial_value=1)
      dataset = dataset.map(lambda x: x + var_0.read_value())
      sess.run(var_0.initializer)

      with ops.device("/cpu:1"):
        var_1 = resource_variable_ops.ResourceVariable(initial_value=1)
      dataset = dataset.map(lambda x: x + var_1.read_value())
      sess.run(var_1.initializer)

      iterator = dataset_ops.make_initializable_iterator(dataset)
      sess.run(iterator.initializer)

      self.assertEqual(sess.run(iterator.get_next()), 2)

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetInputSerialization(self):
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset_ops.Dataset.from_tensors(dataset).flat_map(lambda x: x)
    dataset = self.graphRoundTrip(dataset)
    self.assertDatasetProduces(dataset, range(100))


if __name__ == "__main__":
  test.main()
