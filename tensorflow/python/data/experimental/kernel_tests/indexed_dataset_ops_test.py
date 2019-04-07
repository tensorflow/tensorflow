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

from tensorflow.python.data.experimental.ops import indexed_dataset_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class IndexedDatasetOpsTest(test_base.DatasetTestBase):

  def testLowLevelIndexedDatasetOps(self):
    identity = ged_ops.experimental_identity_indexed_dataset(
        ops.convert_to_tensor(16, dtype=dtypes.uint64))
    handle = ged_ops.experimental_materialized_index_dataset_handle(
        container="",
        shared_name="",
        output_types=[dtypes.uint64],
        output_shapes=[[]])
    materialize = ged_ops.experimental_indexed_dataset_materialize(
        identity, handle)
    get_op = ged_ops.experimental_indexed_dataset_get(
        handle, 3, output_types=[dtypes.uint64], output_shapes=[[]])

    self.evaluate(materialize)
    self.assertEqual([3], self.evaluate(get_op))

  # TODO(b/117581999): Eager mode not supported.
  @test_util.run_deprecated_v1
  def testSkipEagerIdentityIndexedDataset(self):
    ds = indexed_dataset_ops.IdentityIndexedDataset(16)
    materialized = ds.materialize()
    self.evaluate(materialized.initializer)
    for i in range(16):
      output = self.evaluate(materialized.get(i))
      self.assertEqual([i], output)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(materialized.get(16))

  @unittest.skip("Requisite functionality currently unimplemented.")
  def testIdentityIndexedDatasetIterator(self):
    ds = indexed_dataset_ops.IdentityIndexedDataset(16)
    n = self.getNext(ds)

    for i in range(16):
      output = self.evaluate(n())
      self.assertEqual(i, output)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(n())


if __name__ == "__main__":
  test.main()
