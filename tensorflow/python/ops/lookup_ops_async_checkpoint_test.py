# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for AsyncCheckpoint for `MutableHashTable` and `DenseHashTable`."""

# Cannot implement these tests in `third_party/tensorflow/python/
# kernel_tests/data_structures/lookup_ops_test.py` because the tests here need
# to import `test` from `tensorflow.python.eager`; while the existing tests
# in `lookup_ops_test.py` need to import `test` from
# `tensorflow.python.platform`.
#
# Both tests in this file are adapted from the `testObjectSaveRestore` test in
# that file, each from the corresponding class.

import os
import tempfile

from absl.testing import parameterized

from tensorflow.python.checkpoint import checkpoint as checkpoint_utils
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables

SKIP_ANONYMOUS_IN_TF1_REASON = (
    "In v1 graph mode, each self.evaluate call will execute the handle "
    "creation op (e.g. AnonymousHashTable) which will create a new table "
    "resource unrelated to other self.evaluate calls, so we can't test "
    "anonymous resources with self.evaluate ."
)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class LookupOpsAsyncCheckpointTest(test.TestCase):
  @test_util.run_in_graph_and_eager_modes
  def testAsyncObjectSaveRestore(self, is_anonymous):

    if is_anonymous and not context.executing_eagerly():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_prefix = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    default_value = -1
    empty_key = 0
    deleted_key = -1
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    save_table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        name="t1",
        checkpoint=True,
        initial_num_buckets=32,
        experimental_is_anonymous=is_anonymous)

    save_checkpoint = checkpoint_utils.Checkpoint(table=save_table)
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=True)

    self.assertAllEqual(0, self.evaluate(save_table.size()))
    self.evaluate(save_table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(save_table.size()))
    self.assertAllEqual(32, len(self.evaluate(save_table.export()[0])))

    save_path = save_checkpoint.save(save_prefix, options=ckpt_options)
    save_checkpoint.sync()  # save_path may not have finished being written
    del save_table, save_checkpoint

    load_table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        name="t1",
        checkpoint=True,
        initial_num_buckets=64,
        experimental_is_anonymous=is_anonymous)
    self.evaluate(
        load_table.insert(
            constant_op.constant([11, 14], dtypes.int64),
            constant_op.constant([12, 24], dtypes.int64)))
    self.assertAllEqual(2, self.evaluate(load_table.size()))
    self.assertAllEqual(64, len(self.evaluate(load_table.export()[0])))

    restore_checkpoint = checkpoint_utils.Checkpoint(table=load_table)

    # Restore the saved values in the parameter nodes.
    restore_checkpoint.restore(save_path).run_restore_ops()

    self.assertAllEqual(3, self.evaluate(load_table.size()))
    self.assertAllEqual(32, len(self.evaluate(load_table.export()[0])))

    input_string = constant_op.constant([10, 11, 12, 13, 14], dtypes.int64)
    output = load_table.lookup(input_string)
    self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  @test_util.run_in_graph_and_eager_modes
  def testMutableHashTable(self, is_anonymous):
    if is_anonymous and not context.executing_eagerly():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    # save_prefix = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(20.0, name="v1")

    default_val = -1
    keys = constant_op.constant(["b", "c", "d"], dtypes.string)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        name="t1",
        checkpoint=True,
        experimental_is_anonymous=is_anonymous)

    checkpoint = checkpoint_utils.Checkpoint(table=table, v0=v0, v1=v1)
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=True)
    self.evaluate([v0.initializer, v1.initializer])

    # Check that the parameter nodes have been initialized.
    self.assertEqual(10.0, self.evaluate(v0))
    self.assertEqual(20.0, self.evaluate(v1))

    self.assertAllEqual(0, self.evaluate(table.size()))
    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))
    save_path = checkpoint.save(save_dir, options=ckpt_options)
    checkpoint.sync()  # Otherwise save_path may not have finished being written
    del table, checkpoint, v0, v1

    v0 = variables.Variable(-1.0, name="v0")
    v1 = variables.Variable(-1.0, name="v1")
    default_val = -1
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        name="t1",
        checkpoint=True,
        experimental_is_anonymous=is_anonymous)
    self.evaluate(
        table.insert(
            constant_op.constant(["a", "c"], dtypes.string),
            constant_op.constant([12, 24], dtypes.int64)))
    self.assertAllEqual(2, self.evaluate(table.size()))

    checkpoint = checkpoint_utils.Checkpoint(table=table, v0=v0, v1=v1)

    # Restore the saved values in the parameter nodes.
    checkpoint.restore(save_path).run_restore_ops()
    # Check that the parameter nodes have been restored.
    self.assertEqual(10.0, self.evaluate(v0))
    self.assertEqual(20.0, self.evaluate(v1))

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["a", "b", "c", "d", "e"],
                                        dtypes.string)
    output = table.lookup(input_string)
    self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))


if __name__ == "__main__":
  test.main()
