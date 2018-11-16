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

from tensorflow.python.data.experimental.ops import get_single_element
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

  def testWindow(self):
    """Test that `get_single_element()` can consume a nested dataset."""
    def flat_map_func(ds):
      batched = ds.batch(2)
      element = get_single_element.get_single_element(batched)
      return dataset_ops.Dataset.from_tensors(element)

    dataset = dataset_ops.Dataset.range(10).window(2).flat_map(flat_map_func)
    self.assertDatasetProduces(
        dataset, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])


if __name__ == "__main__":
  test.main()
