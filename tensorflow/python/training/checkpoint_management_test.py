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

from google.protobuf import text_format

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState


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
          self.assertIsNotNone(
              checkpoint_management.latest_checkpoint(traindir))

          # Succeeds: the file will be named "checkpoint-<i>-of-<n>".
          saver = saver_module.Saver(sharded=True)
          saver.save(sess, filepath)
          self.assertIsNotNone(
              checkpoint_management.latest_checkpoint(traindir))

          # Succeeds: the file will be named "checkpoint-<step>-<i>-of-<n>".
          saver = saver_module.Saver(sharded=True)
          saver.save(sess, filepath, global_step=1)
          self.assertIsNotNone(
              checkpoint_management.latest_checkpoint(traindir))

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
          name = checkpoint_management.latest_checkpoint(traindir)
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
    ckpt = checkpoint_management.generate_checkpoint_state_proto(
        save_dir, abs_path)
    self.assertEqual(ckpt.model_checkpoint_path, abs_path)
    self.assertTrue(os.path.isabs(ckpt.model_checkpoint_path))
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 1)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], abs_path)

  def testRelPath(self):
    train_dir = "train"
    model = os.path.join(train_dir, "model-0")
    # model_checkpoint_path should have no "train" directory part.
    new_rel_path = "model-0"
    ckpt = checkpoint_management.generate_checkpoint_state_proto(
        train_dir, model)
    self.assertEqual(ckpt.model_checkpoint_path, new_rel_path)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 1)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], new_rel_path)

  def testAllModelCheckpointPaths(self):
    save_dir = self._get_test_dir("all_models_test")
    abs_path = os.path.join(save_dir, "model-0")
    for paths in [None, [], ["model-2"]]:
      ckpt = checkpoint_management.generate_checkpoint_state_proto(
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
    checkpoint_management.update_checkpoint_state(
        train_dir, rel_path, all_model_checkpoint_paths=[abs_path, rel_path])
    ckpt = checkpoint_management.get_checkpoint_state(train_dir)
    self.assertEqual(ckpt.model_checkpoint_path, rel_path)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], rel_path)
    self.assertEqual(ckpt.all_model_checkpoint_paths[0], abs_path)

  def testUpdateCheckpointStateSaveRelativePaths(self):
    save_dir = self._get_test_dir("update_checkpoint_state")
    os.chdir(save_dir)
    abs_path2 = os.path.join(save_dir, "model-2")
    rel_path2 = "model-2"
    abs_path0 = os.path.join(save_dir, "model-0")
    rel_path0 = "model-0"
    checkpoint_management.update_checkpoint_state_internal(
        save_dir=save_dir,
        model_checkpoint_path=abs_path2,
        all_model_checkpoint_paths=[rel_path0, abs_path2],
        save_relative_paths=True)

    # File should contain relative paths.
    file_content = file_io.read_file_to_string(
        os.path.join(save_dir, "checkpoint"))
    ckpt = CheckpointState()
    text_format.Merge(file_content, ckpt)
    self.assertEqual(ckpt.model_checkpoint_path, rel_path2)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], rel_path2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[0], rel_path0)

    # get_checkpoint_state should return absolute paths.
    ckpt = checkpoint_management.get_checkpoint_state(save_dir)
    self.assertEqual(ckpt.model_checkpoint_path, abs_path2)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], abs_path2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[0], abs_path0)

  def testCheckPointStateFailsWhenIncomplete(self):
    save_dir = self._get_test_dir("checkpoint_state_fails_when_incomplete")
    os.chdir(save_dir)
    ckpt_path = os.path.join(save_dir, "checkpoint")
    ckpt_file = open(ckpt_path, "w")
    ckpt_file.write("")
    ckpt_file.close()
    with self.assertRaises(ValueError):
      checkpoint_management.get_checkpoint_state(save_dir)

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
    ckpt = checkpoint_management.get_checkpoint_state(save_dir)
    self.assertEqual(ckpt.model_checkpoint_path,
                     os.path.join(save_dir, "./model.ckpt-687529"))
    self.assertEqual(ckpt.all_model_checkpoint_paths[0],
                     os.path.join(save_dir, "./model.ckpt-687500"))
    self.assertEqual(ckpt.all_model_checkpoint_paths[1],
                     os.path.join(save_dir, "./model.ckpt-687529"))


class SaverUtilsTest(test.TestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(), "saver_utils_test")
    gfile.MakeDirs(self._base_dir)

  def tearDown(self):
    gfile.DeleteRecursively(self._base_dir)

  def testCheckpointExists(self):
    for sharded in (False, True):
      for version in (saver_pb2.SaverDef.V2, saver_pb2.SaverDef.V1):
        with self.test_session(graph=ops_lib.Graph()) as sess:
          unused_v = variables.Variable(1.0, name="v")
          variables.global_variables_initializer().run()
          saver = saver_module.Saver(sharded=sharded, write_version=version)

          path = os.path.join(self._base_dir, "%s-%s" % (sharded, version))
          self.assertFalse(
              checkpoint_management.checkpoint_exists(path))  # Not saved yet.

          ckpt_prefix = saver.save(sess, path)
          self.assertTrue(checkpoint_management.checkpoint_exists(ckpt_prefix))

          ckpt_prefix = checkpoint_management.latest_checkpoint(self._base_dir)
          self.assertTrue(checkpoint_management.checkpoint_exists(ckpt_prefix))

  def testGetCheckpointMtimes(self):
    prefixes = []
    for version in (saver_pb2.SaverDef.V2, saver_pb2.SaverDef.V1):
      with self.test_session(graph=ops_lib.Graph()) as sess:
        unused_v = variables.Variable(1.0, name="v")
        variables.global_variables_initializer().run()
        saver = saver_module.Saver(write_version=version)
        prefixes.append(
            saver.save(sess, os.path.join(self._base_dir, str(version))))

    mtimes = checkpoint_management.get_checkpoint_mtimes(prefixes)
    self.assertEqual(2, len(mtimes))
    self.assertTrue(mtimes[1] >= mtimes[0])

  def testRemoveCheckpoint(self):
    for sharded in (False, True):
      for version in (saver_pb2.SaverDef.V2, saver_pb2.SaverDef.V1):
        with self.test_session(graph=ops_lib.Graph()) as sess:
          unused_v = variables.Variable(1.0, name="v")
          variables.global_variables_initializer().run()
          saver = saver_module.Saver(sharded=sharded, write_version=version)

          path = os.path.join(self._base_dir, "%s-%s" % (sharded, version))
          ckpt_prefix = saver.save(sess, path)
          self.assertTrue(checkpoint_management.checkpoint_exists(ckpt_prefix))
          checkpoint_management.remove_checkpoint(ckpt_prefix, version)
          self.assertFalse(checkpoint_management.checkpoint_exists(ckpt_prefix))


if __name__ == "__main__":
  test.main()
