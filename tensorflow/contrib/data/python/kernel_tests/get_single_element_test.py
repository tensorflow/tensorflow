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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class GetSingleElementTest(test_base.DatasetTestBase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("Zero", 0, 1),
      ("Five", 5, 1),
      ("Ten", 10, 1),
      ("Empty", 100, 1, errors.InvalidArgumentError, "Dataset was empty."),
      ("MoreThanOne", 0, 2, errors.InvalidArgumentError,
       "Dataset had more than one element."),
  )
  def testGetSingleElement(self, skip, take, error=None, error_msg=None):
    skip_t = array_ops.placeholder(dtypes.int64, shape=[])
    take_t = array_ops.placeholder(dtypes.int64, shape=[])

    def make_sparse(x):
      x_1d = array_ops.reshape(x, [1])
      x_2d = array_ops.reshape(x, [1, 1])
      return sparse_tensor.SparseTensor(x_2d, x_1d, x_1d)

    dataset = dataset_ops.Dataset.range(100).skip(skip_t).map(
        lambda x: (x * x, make_sparse(x))).take(take_t)
    element = get_single_element.get_single_element(dataset)

    with self.cached_session() as sess:
      if error is None:
        dense_val, sparse_val = sess.run(
            element, feed_dict={
                skip_t: skip,
                take_t: take
            })
        self.assertEqual(skip * skip, dense_val)
        self.assertAllEqual([[skip]], sparse_val.indices)
        self.assertAllEqual([skip], sparse_val.values)
        self.assertAllEqual([skip], sparse_val.dense_shape)
      else:
        with self.assertRaisesRegexp(error, error_msg):
          sess.run(element, feed_dict={skip_t: skip, take_t: take})

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

    stop_t = array_ops.placeholder(dtypes.int64, shape=[])
    dataset = dataset_ops.Dataset.range(stop_t)
    element = get_single_element.reduce_dataset(dataset, sum_reducer)

    with self.cached_session() as sess:
      value = sess.run(element, feed_dict={stop_t: stop})
      self.assertEqual(stop * (stop - 1) / 2, value)


if __name__ == "__main__":
  test.main()
