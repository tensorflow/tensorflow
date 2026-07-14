# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for checkpoints tools."""

import os
import pathlib

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib


class CheckpointIteratorTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testReturnsEmptyIfNoCheckpointsFound(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), "no_checkpoints_found")

    num_found = 0
    for _ in checkpoint_utils.checkpoints_iterator(checkpoint_dir, timeout=0):
      num_found += 1
    self.assertEqual(num_found, 0)

  @test_util.run_in_graph_and_eager_modes
  def testReturnsSingleCheckpointIfOneCheckpointFound(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), "one_checkpoint_found")
    if not gfile.Exists(checkpoint_dir):
      gfile.MakeDirs(checkpoint_dir)

    save_path = os.path.join(checkpoint_dir, "model.ckpt")

    a = resource_variable_ops.ResourceVariable(5)
    self.evaluate(a.initializer)
    checkpoint = trackable_utils.Checkpoint(a=a)
    checkpoint.save(file_prefix=save_path)

    num_found = 0
    for _ in checkpoint_utils.checkpoints_iterator(checkpoint_dir, timeout=0):
      num_found += 1
    self.assertEqual(num_found, 1)

  @test_util.run_in_graph_and_eager_modes
  def testWorksWithFSPath(self):
    checkpoint_dir = pathlib.Path(self.get_temp_dir()) / "one_checkpoint_found"
    if not gfile.Exists(checkpoint_dir):
      gfile.MakeDirs(checkpoint_dir)

    save_path = checkpoint_dir / "model.ckpt"

    a = resource_variable_ops.ResourceVariable(5)
    self.evaluate(a.initializer)
    checkpoint = trackable_utils.Checkpoint(a=a)
    checkpoint.save(file_prefix=save_path)

    num_found = 0
    for _ in checkpoint_utils.checkpoints_iterator(checkpoint_dir, timeout=0):
      num_found += 1
    self.assertEqual(num_found, 1)

  @test_util.run_v1_only("Tests v1-style checkpoint sharding")
  def testReturnsSingleCheckpointIfOneShardedCheckpoint(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  "one_checkpoint_found_sharded")
    if not gfile.Exists(checkpoint_dir):
      gfile.MakeDirs(checkpoint_dir)

    global_step = variables.Variable(0, name="v0")

    # This will result in 3 different checkpoint shard files.
    with ops.device("/cpu:0"):
      variables.Variable(10, name="v1")
    with ops.device("/cpu:1"):
      variables.Variable(20, name="v2")

    saver = saver_lib.Saver(sharded=True)

    with session_lib.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as session:

      session.run(variables.global_variables_initializer())
      save_path = os.path.join(checkpoint_dir, "model.ckpt")
      saver.save(session, save_path, global_step=global_step)

    num_found = 0
    for _ in checkpoint_utils.checkpoints_iterator(checkpoint_dir, timeout=0):
      num_found += 1
    self.assertEqual(num_found, 1)

  @test_util.run_in_graph_and_eager_modes
  def testTimeoutFn(self):
    timeout_fn_calls = [0]
    def timeout_fn():
      timeout_fn_calls[0] += 1
      return timeout_fn_calls[0] > 3

    results = list(
        checkpoint_utils.checkpoints_iterator(
            "/non-existent-dir", timeout=0.1, timeout_fn=timeout_fn))
    self.assertEqual([], results)
    self.assertEqual(4, timeout_fn_calls[0])


if __name__ == "__main__":
  test.main()
