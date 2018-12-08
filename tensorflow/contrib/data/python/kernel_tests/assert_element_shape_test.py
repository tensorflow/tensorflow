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

from tensorflow.contrib.data.python.ops import batching
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


class AssertElementShapeTest(test_base.DatasetTestBase):

  def test_assert_element_shape(self):

    def create_dataset(_):
      return (array_ops.ones(2, dtype=dtypes.float32),
              array_ops.zeros((3, 4), dtype=dtypes.int32))

    dataset = dataset_ops.Dataset.range(5).map(create_dataset)
    expected_shapes = (tensor_shape.TensorShape(2),
                       tensor_shape.TensorShape((3, 4)))
    self.assertEqual(expected_shapes, dataset.output_shapes)

    result = dataset.apply(batching.assert_element_shape(expected_shapes))
    self.assertEqual(expected_shapes, result.output_shapes)

    iterator = dataset_ops.make_initializable_iterator(result)
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(5):
        sess.run(get_next)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def test_assert_wrong_element_shape(self):

    def create_dataset(_):
      return (array_ops.ones(2, dtype=dtypes.float32),
              array_ops.zeros((3, 4), dtype=dtypes.int32))

    dataset = dataset_ops.Dataset.range(3).map(create_dataset)
    wrong_shapes = (tensor_shape.TensorShape(2),
                    tensor_shape.TensorShape((3, 10)))
    with self.assertRaises(ValueError):
      dataset.apply(batching.assert_element_shape(wrong_shapes))

  def test_assert_element_shape_on_unknown_shape_dataset(self):

    def create_unknown_shape_dataset(x):
      return script_ops.py_func(
          lambda _: (  # pylint: disable=g-long-lambda
              np.ones(2, dtype=np.float32),
              np.zeros((3, 4), dtype=np.int32)),
          [x],
          [dtypes.float32, dtypes.int32])

    dataset = dataset_ops.Dataset.range(5).map(create_unknown_shape_dataset)
    unknown_shapes = (tensor_shape.TensorShape(None),
                      tensor_shape.TensorShape(None))
    self.assertEqual(unknown_shapes, dataset.output_shapes)

    expected_shapes = (tensor_shape.TensorShape(2),
                       tensor_shape.TensorShape((3, 4)))
    result = dataset.apply(batching.assert_element_shape(expected_shapes))
    self.assertEqual(expected_shapes, result.output_shapes)

    iterator = dataset_ops.make_initializable_iterator(result)
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(5):
        sess.run(get_next)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def test_assert_wrong_element_shape_on_unknown_shape_dataset(self):

    def create_unknown_shape_dataset(x):
      return script_ops.py_func(
          lambda _: (  # pylint: disable=g-long-lambda
              np.ones(2, dtype=np.float32),
              np.zeros((3, 4), dtype=np.int32)),
          [x],
          [dtypes.float32, dtypes.int32])

    dataset = dataset_ops.Dataset.range(3).map(create_unknown_shape_dataset)
    unknown_shapes = (tensor_shape.TensorShape(None),
                      tensor_shape.TensorShape(None))
    self.assertEqual(unknown_shapes, dataset.output_shapes)

    wrong_shapes = (tensor_shape.TensorShape(2),
                    tensor_shape.TensorShape((3, 10)))
    iterator = dataset_ops.make_initializable_iterator(
        dataset.apply(batching.assert_element_shape(wrong_shapes)))
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.cached_session() as sess:
      sess.run(init_op)
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(get_next)

  def test_assert_partial_element_shape(self):

    def create_dataset(_):
      return (array_ops.ones(2, dtype=dtypes.float32),
              array_ops.zeros((3, 4), dtype=dtypes.int32))

    dataset = dataset_ops.Dataset.range(5).map(create_dataset)
    partial_expected_shape = (
        tensor_shape.TensorShape(None),  # Unknown shape
        tensor_shape.TensorShape((None, 4)))  # Partial shape
    result = dataset.apply(
        batching.assert_element_shape(partial_expected_shape))
    # Partial shapes are merged with actual shapes:
    actual_shapes = (tensor_shape.TensorShape(2),
                     tensor_shape.TensorShape((3, 4)))
    self.assertEqual(actual_shapes, result.output_shapes)

    iterator = dataset_ops.make_initializable_iterator(result)
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(5):
        sess.run(get_next)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def test_assert_wrong_partial_element_shape(self):

    def create_dataset(_):
      return (array_ops.ones(2, dtype=dtypes.float32),
              array_ops.zeros((3, 4), dtype=dtypes.int32))

    dataset = dataset_ops.Dataset.range(3).map(create_dataset)
    wrong_shapes = (tensor_shape.TensorShape(2),
                    tensor_shape.TensorShape((None, 10)))
    with self.assertRaises(ValueError):
      dataset.apply(batching.assert_element_shape(wrong_shapes))

  def test_assert_partial_element_shape_on_unknown_shape_dataset(self):

    def create_unknown_shape_dataset(x):
      return script_ops.py_func(
          lambda _: (  # pylint: disable=g-long-lambda
              np.ones(2, dtype=np.float32),
              np.zeros((3, 4), dtype=np.int32)),
          [x],
          [dtypes.float32, dtypes.int32])

    dataset = dataset_ops.Dataset.range(5).map(create_unknown_shape_dataset)
    unknown_shapes = (tensor_shape.TensorShape(None),
                      tensor_shape.TensorShape(None))
    self.assertEqual(unknown_shapes, dataset.output_shapes)

    expected_shapes = (tensor_shape.TensorShape(2),
                       tensor_shape.TensorShape((None, 4)))
    result = dataset.apply(batching.assert_element_shape(expected_shapes))
    self.assertEqual(expected_shapes, result.output_shapes)

    iterator = dataset_ops.make_initializable_iterator(result)
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(5):
        sess.run(get_next)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def test_assert_wrong_partial_element_shape_on_unknown_shape_dataset(self):

    def create_unknown_shape_dataset(x):
      return script_ops.py_func(
          lambda _: (  # pylint: disable=g-long-lambda
              np.ones(2, dtype=np.float32),
              np.zeros((3, 4), dtype=np.int32)),
          [x],
          [dtypes.float32, dtypes.int32])

    dataset = dataset_ops.Dataset.range(3).map(create_unknown_shape_dataset)
    unknown_shapes = (tensor_shape.TensorShape(None),
                      tensor_shape.TensorShape(None))
    self.assertEqual(unknown_shapes, dataset.output_shapes)

    wrong_shapes = (tensor_shape.TensorShape(2),
                    tensor_shape.TensorShape((None, 10)))
    iterator = dataset_ops.make_initializable_iterator(
        dataset.apply(batching.assert_element_shape(wrong_shapes)))
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.cached_session() as sess:
      sess.run(init_op)
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
