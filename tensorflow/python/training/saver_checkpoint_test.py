# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for tensorflow.python.training.saver.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import shutil
import tempfile
import time

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_module
from tensorflow.python.util import compat


class CheckpointedOp(object):
  """Op with a custom checkpointing implementation.

  Defined as part of the test because the MutableHashTable Python code is
  currently in contrib.
  """

  def __init__(self, name):
    self._table_ref = gen_data_flow_ops._mutable_hash_table(
        key_dtype=dtypes.string, value_dtype=dtypes.float32, name=name)
    self._name = name
    self._saveable = CheckpointedOp.CustomSaveable(self, name)
    ops_lib.add_to_collection(ops_lib.GraphKeys.SAVEABLE_OBJECTS,
                              self._saveable)

  @property
  def name(self):
    return self._name

  @property
  def saveable(self):
    return self._saveable

  def insert(self, keys, values):
    return gen_data_flow_ops._lookup_table_insert(self._table_ref, keys, values)

  def keys(self):
    return self._export()[0]

  def values(self):
    return self._export()[1]

  def _export(self):
    return gen_data_flow_ops._lookup_table_export(self._table_ref,
                                                  dtypes.string, dtypes.float32)

  class CustomSaveable(saver_module.BaseSaverBuilder.SaveableObject):

    def __init__(self, table, name):
      tensors = table._export()
      specs = [
          saver_module.BaseSaverBuilder.SaveSpec(tensors[0], "",
                                                 name + "-keys"),
          saver_module.BaseSaverBuilder.SaveSpec(tensors[1], "",
                                                 name + "-values")
      ]
      super(CheckpointedOp.CustomSaveable, self).__init__(table, specs, name)

    def restore(self, restore_tensors, shapes):
      return gen_data_flow_ops._lookup_table_import(self.op._table_ref,
                                                    restore_tensors[0],
                                                    restore_tensors[1])


class KeepCheckpointEveryNHoursTest(test.TestCase):

  def _get_test_dir(self, dirname):
    test_dir = os.path.join(self.get_temp_dir(), dirname)
    gfile.MakeDirs(test_dir)
    return test_dir

  def testNonSharded(self):
    save_dir = self._get_test_dir("keep_checkpoint_every_n_hours")

    with self.test_session() as sess:
      v = variables.Variable([10.0], name="v")
      # Run the initializer NOW to avoid the 0.5s overhead of the first Run()
      # call, which throws the test timing off in fastbuild mode.
      variables.global_variables_initializer().run()
      # Create a saver that will keep the last 2 checkpoints plus one every 0.7
      # seconds.
      start_time = time.time()
      save = saver_module.Saver(
          {
              "v": v
          }, max_to_keep=2, keep_checkpoint_every_n_hours=0.7 / 3600)
      self.assertEqual([], save.last_checkpoints)

      # Wait till 0.7 second have elapsed so s1 will be old enough to keep.
      time.sleep((time.time() + 0.7) - start_time)
      s1 = save.save(sess, os.path.join(save_dir, "s1"))
      self.assertEqual([s1], save.last_checkpoints)

      s2 = save.save(sess, os.path.join(save_dir, "s2"))
      self.assertEqual([s1, s2], save.last_checkpoints)

      # We now have 2 'last_checkpoints': [s1, s2].  The next call to Save(),
      # would normally delete s1, because max_to_keep is 2.  However, s1 is
      # older than 0.7s so we must keep it.
      s3 = save.save(sess, os.path.join(save_dir, "s3"))
      self.assertEqual([s2, s3], save.last_checkpoints)

      # s1 should still be here, we are Not checking now to reduce time
      # variance in the test.

      # We now have 2 'last_checkpoints': [s2, s3], and s1 on disk.  The next
      # call to Save(), will delete s2, because max_to_keep is 2, and because
      # we already kept the old s1. s2 is very close in time to s1 so it gets
      # deleted.
      s4 = save.save(sess, os.path.join(save_dir, "s4"))
      self.assertEqual([s3, s4], save.last_checkpoints)

      # Check that s1 is still here, but s2 is gone.
      self.assertTrue(saver_module.checkpoint_exists(s1))
      self.assertFalse(saver_module.checkpoint_exists(s2))
      self.assertTrue(saver_module.checkpoint_exists(s3))
      self.assertTrue(saver_module.checkpoint_exists(s4))


class LatestCheckpointWithRelativePaths(test.TestCase):

  @staticmethod
  @contextlib.contextmanager
  def tempWorkingDir(temppath):
    cwd = os.getcwd()
    os.chdir(temppath)
    try:
      yield
    finally:
      os.chdir(cwd)

  @staticmethod
  @contextlib.contextmanager
  def tempDir():
    tempdir = tempfile.mkdtemp()
    try:
      yield tempdir
    finally:
      shutil.rmtree(tempdir)

  def testNameCollision(self):
    # Make sure we have a clean directory to work in.
    with self.tempDir() as tempdir:
      # Jump to that directory until this test is done.
      with self.tempWorkingDir(tempdir):
        # Save training snapshots to a relative path.
        traindir = "train/"
        os.mkdir(traindir)
        # Collides with the default name of the checkpoint state file.
        filepath = os.path.join(traindir, "checkpoint")

        with self.test_session() as sess:
          unused_a = variables.Variable(0.0)  # So that Saver saves something.
          variables.global_variables_initializer().run()

          # Should fail.
          saver = saver_module.Saver(sharded=False)
          with self.assertRaisesRegexp(ValueError, "collides with"):
            saver.save(sess, filepath)

          # Succeeds: the file will be named "checkpoint-<step>".
          saver.save(sess, filepath, global_step=1)
          self.assertIsNotNone(saver_module.latest_checkpoint(traindir))

          # Succeeds: the file will be named "checkpoint-<i>-of-<n>".
          saver = saver_module.Saver(sharded=True)
          saver.save(sess, filepath)
          self.assertIsNotNone(saver_module.latest_checkpoint(traindir))

          # Succeeds: the file will be named "checkpoint-<step>-<i>-of-<n>".
          saver = saver_module.Saver(sharded=True)
          saver.save(sess, filepath, global_step=1)
          self.assertIsNotNone(saver_module.latest_checkpoint(traindir))

  def testRelativePath(self):
    # Make sure we have a clean directory to work in.
    with self.tempDir() as tempdir:

      # Jump to that directory until this test is done.
      with self.tempWorkingDir(tempdir):

        # Save training snapshots to a relative path.
        traindir = "train/"
        os.mkdir(traindir)

        filename = "snapshot"
        filepath = os.path.join(traindir, filename)

        with self.test_session() as sess:
          # Build a simple graph.
          v0 = variables.Variable(0.0)
          inc = v0.assign_add(1.0)

          save = saver_module.Saver({"v0": v0})

          # Record a short training history.
          variables.global_variables_initializer().run()
          save.save(sess, filepath, global_step=0)
          inc.eval()
          save.save(sess, filepath, global_step=1)
          inc.eval()
          save.save(sess, filepath, global_step=2)

        with self.test_session() as sess:
          # Build a new graph with different initialization.
          v0 = variables.Variable(-1.0)

          # Create a new saver.
          save = saver_module.Saver({"v0": v0})
          variables.global_variables_initializer().run()

          # Get the most recent checkpoint name from the training history file.
          name = saver_module.latest_checkpoint(traindir)
          self.assertIsNotNone(name)

          # Restore "v0" from that checkpoint.
          save.restore(sess, name)
          self.assertEqual(v0.eval(), 2.0)


class CheckpointStateTest(test.TestCase):

  def _get_test_dir(self, dirname):
    test_dir = os.path.join(self.get_temp_dir(), dirname)
    gfile.MakeDirs(test_dir)
    return test_dir

  def testAbsPath(self):
    save_dir = self._get_test_dir("abs_paths")
    abs_path = os.path.join(save_dir, "model-0")
    ckpt = saver_module.generate_checkpoint_state_proto(save_dir, abs_path)
    self.assertEqual(ckpt.model_checkpoint_path, abs_path)
    self.assertTrue(os.path.isabs(ckpt.model_checkpoint_path))
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 1)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], abs_path)

  def testRelPath(self):
    train_dir = "train"
    model = os.path.join(train_dir, "model-0")
    # model_checkpoint_path should have no "train" directory part.
    new_rel_path = "model-0"
    ckpt = saver_module.generate_checkpoint_state_proto(train_dir, model)
    self.assertEqual(ckpt.model_checkpoint_path, new_rel_path)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 1)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], new_rel_path)

  def testAllModelCheckpointPaths(self):
    save_dir = self._get_test_dir("all_models_test")
    abs_path = os.path.join(save_dir, "model-0")
    for paths in [None, [], ["model-2"]]:
      ckpt = saver_module.generate_checkpoint_state_proto(
          save_dir, abs_path, all_model_checkpoint_paths=paths)
      self.assertEqual(ckpt.model_checkpoint_path, abs_path)
      self.assertTrue(os.path.isabs(ckpt.model_checkpoint_path))
      self.assertEqual(
          len(ckpt.all_model_checkpoint_paths), len(paths) if paths else 1)
      self.assertEqual(ckpt.all_model_checkpoint_paths[-1], abs_path)

  def testUpdateCheckpointState(self):
    save_dir = self._get_test_dir("update_checkpoint_state")
    os.chdir(save_dir)
    # Make a temporary train directory.
    train_dir = "train"
    os.mkdir(train_dir)
    abs_path = os.path.join(save_dir, "model-0")
    rel_path = os.path.join("train", "model-2")
    saver_module.update_checkpoint_state(
        train_dir, rel_path, all_model_checkpoint_paths=[abs_path, rel_path])
    ckpt = saver_module.get_checkpoint_state(train_dir)
    self.assertEqual(ckpt.model_checkpoint_path, rel_path)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], rel_path)
    self.assertEqual(ckpt.all_model_checkpoint_paths[0], abs_path)

  def testCheckPointStateFailsWhenIncomplete(self):
    save_dir = self._get_test_dir("checkpoint_state_fails_when_incomplete")
    os.chdir(save_dir)
    ckpt_path = os.path.join(save_dir, "checkpoint")
    ckpt_file = open(ckpt_path, "w")
    ckpt_file.write("")
    ckpt_file.close()
    with self.assertRaises(ValueError):
      saver_module.get_checkpoint_state(save_dir)

  def testCheckPointCompletesRelativePaths(self):
    save_dir = self._get_test_dir("checkpoint_completes_relative_paths")
    os.chdir(save_dir)
    ckpt_path = os.path.join(save_dir, "checkpoint")
    ckpt_file = open(ckpt_path, "w")
    ckpt_file.write("""
        model_checkpoint_path: "./model.ckpt-687529"
        all_model_checkpoint_paths: "./model.ckpt-687500"
        all_model_checkpoint_paths: "./model.ckpt-687529"
        """)
    ckpt_file.close()
    ckpt = saver_module.get_checkpoint_state(save_dir)
    self.assertEqual(ckpt.model_checkpoint_path,
                     os.path.join(save_dir, "./model.ckpt-687529"))
    self.assertEqual(ckpt.all_model_checkpoint_paths[0],
                     os.path.join(save_dir, "./model.ckpt-687500"))
    self.assertEqual(ckpt.all_model_checkpoint_paths[1],
                     os.path.join(save_dir, "./model.ckpt-687529"))


class CheckpointReaderTest(test.TestCase):

  _WRITE_VERSION = saver_pb2.SaverDef.V1

  def testDebugString(self):
    # Builds a graph.
    v0 = variables.Variable(
        [[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32, name="v0")
    v1 = variables.Variable(
        [[[1], [2]], [[3], [4]], [[5], [6]]], dtype=dtypes.float32, name="v1")
    init_all_op = variables.global_variables_initializer()
    save = saver_module.Saver(
        {
            "v0": v0,
            "v1": v1
        }, write_version=self._WRITE_VERSION)
    save_path = os.path.join(self.get_temp_dir(),
                             "ckpt_for_debug_string" + str(self._WRITE_VERSION))
    with self.test_session() as sess:
      sess.run(init_all_op)
      # Saves a checkpoint.
      save.save(sess, save_path)

      # Creates a reader.
      reader = pywrap_tensorflow.NewCheckpointReader(save_path)
      # Verifies that the tensors exist.
      self.assertTrue(reader.has_tensor("v0"))
      self.assertTrue(reader.has_tensor("v1"))
      debug_string = reader.debug_string()
      # Verifies that debug string contains the right strings.
      self.assertTrue(compat.as_bytes("v0 (DT_FLOAT) [2,3]") in debug_string)
      self.assertTrue(compat.as_bytes("v1 (DT_FLOAT) [3,2,1]") in debug_string)
      # Verifies get_variable_to_shape_map() returns the correct information.
      var_map = reader.get_variable_to_shape_map()
      self.assertEquals([2, 3], var_map["v0"])
      self.assertEquals([3, 2, 1], var_map["v1"])
      # Verifies get_tensor() returns the tensor value.
      v0_tensor = reader.get_tensor("v0")
      v1_tensor = reader.get_tensor("v1")
      self.assertAllEqual(v0.eval(), v0_tensor)
      self.assertAllEqual(v1.eval(), v1_tensor)
      # Verifies get_tensor() fails for non-existent tensors.
      with self.assertRaisesRegexp(errors.NotFoundError,
                                   "v3 not found in checkpoint"):
        reader.get_tensor("v3")

  def testNonexistentPath(self):
    with self.assertRaisesRegexp(errors.NotFoundError,
                                 "Unsuccessful TensorSliceReader"):
      pywrap_tensorflow.NewCheckpointReader("non-existent")


class CheckpointReaderForV2Test(CheckpointReaderTest):
  _WRITE_VERSION = saver_pb2.SaverDef.V2


if __name__ == "__main__":
  test.main()
