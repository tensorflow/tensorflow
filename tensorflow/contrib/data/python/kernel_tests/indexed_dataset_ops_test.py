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
"""Tests for experimental indexed dataset ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from tensorflow.contrib.data.python.ops import contrib_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.data.python.ops import gen_dataset_ops
from tensorflow.contrib.data.python.ops import indexed_dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class IndexedDatasetOpsTest(test.TestCase):

  def testLowLevelIndexedDatasetOps(self):
    identity = gen_dataset_ops.identity_indexed_dataset(
        ops.convert_to_tensor(16, dtype=dtypes.uint64))
    handle = gen_dataset_ops.materialized_index_dataset_handle(
        container="",
        shared_name="",
        output_types=[dtypes.uint64],
        output_shapes=[[]])
    materialize = gen_dataset_ops.indexed_dataset_materialize(identity, handle)
    index = array_ops.placeholder(dtypes.uint64)
    get_op = gen_dataset_ops.indexed_dataset_get(
        handle, index, output_types=[dtypes.uint64], output_shapes=[[]])

    with self.test_session() as sess:
      sess.run(materialize)
      self.assertEqual([3], sess.run(get_op, feed_dict={index: 3}))

  def testIdentityIndexedDataset(self):
    ds = indexed_dataset_ops.IdentityIndexedDataset(16)
    materialized = ds.materialize()
    with self.test_session() as sess:
      sess.run(materialized.initializer)
      placeholder = array_ops.placeholder(dtypes.uint64, shape=[])
      for i in range(16):
        output = sess.run(
            materialized.get(placeholder), feed_dict={placeholder: i})
        self.assertEqual([i], output)
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(materialized.get(placeholder), feed_dict={placeholder: 16})

  @unittest.skip("Requisite functionality currently unimplemented.")
  def testIdentityIndexedDatasetIterator(self):
    ds = indexed_dataset_ops.IdentityIndexedDataset(16)
    itr = ds.make_initializable_iterator()
    n = itr.get_next()
    with self.test_session() as sess:
      sess.run(itr.initializer)
      for i in range(16):
        output = sess.run(n)
        self.assertEqual(i, output)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(n)

if __name__ == "__main__":
  test.main()
