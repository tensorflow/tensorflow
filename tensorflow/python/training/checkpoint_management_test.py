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
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorflow.python.training.checkpointable import util


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

        with self.cached_session() as sess:
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

        with self.cached_session() as sess:
          # Build a simple graph.
          v0 = variables.Variable(0.0)
          inc = v0.assign_add(1.0)

          save = saver_module.Saver({"v0": v0})

          # Record a short training history.
          variables.global_variables_initializer().run()
          save.save(sess, filepath, global_step=0)
          self.evaluate(inc)
          save.save(sess, filepath, global_step=1)
          self.evaluate(inc)
          save.save(sess, filepath, global_step=2)

        with self.cached_session() as sess:
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
        with self.session(graph=ops_lib.Graph()) as sess:
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
      with self.session(graph=ops_lib.Graph()) as sess:
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
        with self.session(graph=ops_lib.Graph()) as sess:
          unused_v = variables.Variable(1.0, name="v")
          variables.global_variables_initializer().run()
          saver = saver_module.Saver(sharded=sharded, write_version=version)

          path = os.path.join(self._base_dir, "%s-%s" % (sharded, version))
          ckpt_prefix = saver.save(sess, path)
          self.assertTrue(checkpoint_management.checkpoint_exists(ckpt_prefix))
          checkpoint_management.remove_checkpoint(ckpt_prefix, version)
          self.assertFalse(checkpoint_management.checkpoint_exists(ckpt_prefix))


class CheckpointManagerTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testDeletion(self):
    checkpoint = util.Checkpoint()
    manager = checkpoint_management.CheckpointManager(
        checkpoint, self.get_temp_dir(), max_to_keep=3)
    first_path = manager.save()
    second_path = manager.save()
    third_path = manager.save()
    fourth_path = manager.save()
    self.assertTrue(checkpoint_management.checkpoint_exists(fourth_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertFalse(checkpoint_management.checkpoint_exists(first_path))

  @test_util.run_in_graph_and_eager_modes
  def testKeepAll(self):
    checkpoint = util.Checkpoint()
    directory = os.path.join(
        self.get_temp_dir(),
        # Avoid sharing directories between eager and graph
        # TODO(allenl): stop run_in_graph_and_eager_modes reusing directories
        str(context.executing_eagerly()))
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=None)
    first_path = manager.save()
    second_path = manager.save()
    third_path = manager.save()
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(first_path))
    self.assertEqual(third_path, manager.latest_checkpoint)
    self.assertEqual([first_path, second_path, third_path],
                     manager.checkpoints)
    del manager
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=None)
    fourth_path = manager.save()
    self.assertEqual([first_path, second_path, third_path, fourth_path],
                     manager.checkpoints)
    del manager
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=3)
    self.assertEqual([first_path, second_path, third_path, fourth_path],
                     manager.checkpoints)
    self.assertTrue(checkpoint_management.checkpoint_exists(fourth_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(first_path))
    fifth_path = manager.save()
    self.assertEqual([third_path, fourth_path, fifth_path],
                     manager.checkpoints)
    self.assertTrue(checkpoint_management.checkpoint_exists(fifth_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(fourth_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertFalse(checkpoint_management.checkpoint_exists(second_path))
    self.assertFalse(checkpoint_management.checkpoint_exists(first_path))

  @test_util.run_in_graph_and_eager_modes
  @test.mock.patch.object(checkpoint_management, "time")
  def testSaveRestoreState(self, mock_time):
    directory = self.get_temp_dir()
    mock_time.time.return_value = 3.
    checkpoint = util.Checkpoint()
    first_manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=2)
    first_time = 10000.
    first_name = os.path.join(directory, "ckpt-1")
    mock_time.time.return_value = first_time
    first_manager.save()
    state = checkpoint_management.get_checkpoint_state(directory)
    second_time = first_time + 3610.
    second_name = os.path.join(directory, "ckpt-2")
    mock_time.time.return_value = second_time
    first_manager.save()
    state = checkpoint_management.get_checkpoint_state(directory)
    self.assertEqual([first_time, second_time],
                     state.all_model_checkpoint_timestamps)
    self.assertEqual([first_name, second_name], first_manager.checkpoints)
    self.assertEqual(second_name, first_manager.latest_checkpoint)
    del first_manager

    second_manager = checkpoint_management.CheckpointManager(
        checkpoint, directory,
        max_to_keep=2, keep_checkpoint_every_n_hours=1.5)
    self.assertEqual([first_name, second_name], second_manager.checkpoints)
    self.assertEqual(second_name, second_manager.latest_checkpoint)
    third_name = os.path.join(directory, "ckpt-3")
    third_time = second_time + 3600. * 0.2
    mock_time.time.return_value = third_time
    second_manager.save()
    self.assertTrue(checkpoint_management.checkpoint_exists(first_name))
    self.assertTrue(checkpoint_management.checkpoint_exists(second_name))
    self.assertEqual([second_name, third_name],
                     second_manager.checkpoints)
    state = checkpoint_management.get_checkpoint_state(directory)
    self.assertEqual(first_time, state.last_preserved_timestamp)
    fourth_time = third_time + 3600. * 0.5
    mock_time.time.return_value = fourth_time
    fourth_name = os.path.join(directory, "ckpt-4")
    second_manager.save()
    self.assertTrue(checkpoint_management.checkpoint_exists(first_name))
    self.assertFalse(checkpoint_management.checkpoint_exists(second_name))
    self.assertEqual([third_name, fourth_name],
                     second_manager.checkpoints)
    fifth_time = fourth_time + 3600. * 0.5
    mock_time.time.return_value = fifth_time
    fifth_name = os.path.join(directory, "ckpt-5")
    second_manager.save()
    self.assertEqual([fourth_name, fifth_name],
                     second_manager.checkpoints)
    state = checkpoint_management.get_checkpoint_state(directory)
    self.assertEqual(first_time, state.last_preserved_timestamp)
    del second_manager
    third_manager = checkpoint_management.CheckpointManager(
        checkpoint, directory,
        max_to_keep=2, keep_checkpoint_every_n_hours=1.5)
    self.assertEqual(fifth_name, third_manager.latest_checkpoint)
    mock_time.time.return_value += 10.
    third_manager.save()
    sixth_name = os.path.join(directory, "ckpt-6")
    state = checkpoint_management.get_checkpoint_state(directory)
    self.assertEqual(fourth_time, state.last_preserved_timestamp)
    self.assertTrue(checkpoint_management.checkpoint_exists(first_name))
    self.assertTrue(checkpoint_management.checkpoint_exists(fourth_name))
    self.assertTrue(checkpoint_management.checkpoint_exists(fifth_name))
    self.assertTrue(checkpoint_management.checkpoint_exists(sixth_name))
    self.assertFalse(checkpoint_management.checkpoint_exists(second_name))
    self.assertFalse(checkpoint_management.checkpoint_exists(third_name))
    self.assertEqual([fifth_name, sixth_name],
                     third_manager.checkpoints)

  @test_util.run_in_graph_and_eager_modes
  def testContinueFromUnmanaged(self):
    directory = self.get_temp_dir()
    prefix = os.path.join(directory, "unusual_prefix")
    checkpoint = util.Checkpoint()
    first_path = checkpoint.save(prefix)
    second_path = checkpoint.save(prefix)
    del checkpoint
    checkpoint = util.Checkpoint()
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=2)
    checkpoint.restore(manager.latest_checkpoint).run_restore_ops()
    self.assertEqual(2, self.evaluate(checkpoint.save_counter))
    third_path = manager.save()
    self.assertEqual([third_path], manager.checkpoints)
    fourth_path = manager.save()
    self.assertEqual([third_path, fourth_path],
                     manager.checkpoints)
    fifth_path = manager.save()
    self.assertEqual([fourth_path, fifth_path],
                     manager.checkpoints)
    self.assertTrue(checkpoint_management.checkpoint_exists(first_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertFalse(checkpoint_management.checkpoint_exists(third_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(fourth_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(fifth_path))

  @test_util.run_in_graph_and_eager_modes
  @test.mock.patch.object(checkpoint_management, "time")
  def testClockReset(self, mock_time):
    directory = self.get_temp_dir()
    mock_time.time.return_value = 10000.
    checkpoint = util.Checkpoint()
    first_manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=1, keep_checkpoint_every_n_hours=1.)
    first_path = first_manager.save()
    mock_time.time.return_value += 3600.
    second_path = first_manager.save()
    mock_time.time.return_value += 3600.
    third_path = first_manager.save()
    self.assertFalse(checkpoint_management.checkpoint_exists(first_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertEqual([third_path], first_manager.checkpoints)
    state = checkpoint_management.get_checkpoint_state(directory)
    self.assertEqual(13600., state.last_preserved_timestamp)
    # Set the clock back in time
    mock_time.time.return_value = 5000.
    del first_manager
    with test.mock.patch.object(logging, "warning") as mock_log:
      second_manager = checkpoint_management.CheckpointManager(
          checkpoint, directory, max_to_keep=1)
      self.assertRegexpMatches(
          str(mock_log.call_args),
          "behind the last preserved checkpoint timestamp")
    # We should err on the side of keeping checkpoints around when we're not
    # sure whether they were preserved or not due to clock funkiness.
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    # We know about the existing checkpoints, but they'll never be deleted and
    # so won't go in the CheckpointState proto on save.
    self.assertEqual(third_path, second_manager.latest_checkpoint)
    self.assertEqual([], second_manager.checkpoints)
    mock_time.time.return_value += 10.
    fourth_path = second_manager.save()
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertEqual(fourth_path, second_manager.latest_checkpoint)
    self.assertEqual([fourth_path], second_manager.checkpoints)
    mock_time.time.return_value += 10.
    fifth_path = second_manager.save()
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertEqual([fifth_path], second_manager.checkpoints)
    state = checkpoint_management.get_checkpoint_state(directory)
    self.assertEqual(5000., state.last_preserved_timestamp)
    self.assertEqual([5020.],
                     state.all_model_checkpoint_timestamps)

  @test_util.run_in_graph_and_eager_modes
  def testCustomNumbering(self):
    directory = self.get_temp_dir()
    step = variables.Variable(0, dtype=dtypes.int64)
    checkpoint = util.Checkpoint(step=step)
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=2)
    self.evaluate(step.initializer)
    for i in range(5):
      path = manager.save(checkpoint_number=step)
      expected_suffix = "-%d" % (2 * i,)
      if not path.endswith(expected_suffix):
        self.fail("%s should have suffix %s" % (path, expected_suffix))
      self.evaluate(step.assign_add(2))
    self.assertEqual(5, self.evaluate(checkpoint.save_counter))
    # Test regular integers
    last_path = manager.save(checkpoint_number=32)
    self.assertIn("-32", last_path)
    self.assertEqual(last_path, manager.latest_checkpoint)
    self.assertEqual(
        last_path, checkpoint_management.latest_checkpoint(directory))
    state = checkpoint_management.get_checkpoint_state(directory)
    # Only the most recent two checkpoints are saved
    self.assertEqual([path, last_path], state.all_model_checkpoint_paths)


if __name__ == "__main__":
  test.main()
