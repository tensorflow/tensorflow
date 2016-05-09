# Copyright 2015 Google Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.utils import checkpoints


def _create_checkpoints(sess, checkpoint_dir):
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  checkpoint_state_name = "checkpoint"
  v1 = tf.get_variable("var1", [1, 10])
  v2 = tf.get_variable("var2", [10, 10])
  v3 = tf.get_variable("var3", [100, 100])
  with tf.variable_scope("useful_scope"):
    v4 = tf.get_variable("var4", [9, 9])
  sess.run(tf.initialize_all_variables())
  v1_value, v2_value, v3_value, v4_value = sess.run([v1, v2, v3, v4])
  saver = tf.train.Saver()
  saver.save(sess, checkpoint_prefix, global_step=0,
             latest_filename=checkpoint_state_name)
  return v1_value, v2_value, v3_value, v4_value


class CheckpointsTest(tf.test.TestCase):

  def testNoCheckpoints(self):
    checkpoint_dir = self.get_temp_dir() + "/no_checkpoints"
    with self.assertRaises(tf.errors.OpError):
      self.assertAllEqual(checkpoints.load_variable(checkpoint_dir, "var1"), [])

  def testNoTensor(self):
    checkpoint_dir = self.get_temp_dir()
    with self.test_session() as session:
      _, _, _, _ = _create_checkpoints(session, checkpoint_dir)
    with self.assertRaises(tf.errors.OpError):
      self.assertAllEqual(checkpoints.load_variable(checkpoint_dir, "var5"), [])

  def testGetTensor(self):
    checkpoint_dir = self.get_temp_dir()
    with self.test_session() as session:
      v1, v2, v3, v4 = _create_checkpoints(session, checkpoint_dir)
    self.assertAllEqual(checkpoints.load_variable(checkpoint_dir, "var1"), v1)
    self.assertAllEqual(checkpoints.load_variable(checkpoint_dir, "var2"), v2)
    self.assertAllEqual(checkpoints.load_variable(checkpoint_dir, "var3"), v3)
    self.assertAllEqual(
        checkpoints.load_variable(checkpoint_dir, "useful_scope/var4"), v4)

  def testGetAllVariables(self):
    checkpoint_dir = self.get_temp_dir()
    with self.test_session() as session:
      _create_checkpoints(session, checkpoint_dir)
    self.assertEqual(checkpoints.list_variables(checkpoint_dir),
                     [("useful_scope/var4", [9, 9]),
                      ("var1", [1, 10]),
                      ("var2", [10, 10]),
                      ("var3", [100, 100])])

  def testInitFromCheckpoint(self):
    checkpoint_dir = self.get_temp_dir()
    with self.test_session() as session:
      v1, v2, v3, v4 = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with tf.Graph().as_default() as g:
      with self.test_session(graph=g) as session:
        with tf.variable_scope("some_scope"):
          my1 = tf.get_variable("my1", [1, 10])
          with tf.variable_scope("some_other_scope"):
            my2 = tf.get_variable("my2", [10, 10])
            with tf.variable_scope("other_useful_scope"):
              my4 = tf.get_variable("var4", [9, 9])
        my3 = tf.get_variable("my3", [100, 100])

        checkpoints.init_from_checkpoint(checkpoint_dir, {
            "some_scope/my1": "var1",
            "some_scope/some_other_scope/other_useful_scope/": "useful_scope/",
        })
        checkpoints.init_from_checkpoint(checkpoint_dir, {
            "some_scope/some_other_scope/my2": "var2",
            "my3": "var3",
        })

        session.run(tf.initialize_all_variables())
        self.assertAllEqual(my1.eval(session), v1)
        self.assertAllEqual(my2.eval(session), v2)
        self.assertAllEqual(my3.eval(session), v3)
        self.assertAllEqual(my4.eval(session), v4)

        # Check that tensors are not explicitly in the graph.
        self.assertLess(len(str(session.graph.as_graph_def())), 22000)

  def testInitFromCheckpointMissing(self):
    checkpoint_dir = self.get_temp_dir()
    with self.test_session() as session:
      _, _, _, _ = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with tf.Graph().as_default() as g:
      with self.test_session(graph=g) as session:
        with tf.variable_scope("some_scope"):
          _ = tf.get_variable("my1", [10, 10])
          _ = tf.get_variable("my2", [1, 10],
                              dtype=tf.int64, initializer=tf.zeros_initializer)

        # No directory.
        with self.assertRaises(tf.errors.OpError):
          checkpoints.init_from_checkpoint("no_dir", {
              "some_scope/my1": "var1"})

        # No variable in checkpoint.
        with self.assertRaises(ValueError):
          checkpoints.init_from_checkpoint(checkpoint_dir, {
              "some_scope/my1": "no_var"})

        # No variable in the graph.
        with self.assertRaises(ValueError):
          checkpoints.init_from_checkpoint(checkpoint_dir, {
              "some_scope/no_var": "var3"})

        # Shape mismatch.
        with self.assertRaises(ValueError):
          checkpoints.init_from_checkpoint(checkpoint_dir, {
              "some_scope/my1": "var1"})

        # DType mismatch.
        with self.assertRaises(ValueError):
          checkpoints.init_from_checkpoint(checkpoint_dir, {
              "some_scope/my2": "var1"})

        # Variable 'my1' and 'my2' are missing in given checkpoint scope.
        with self.assertRaises(ValueError):
          checkpoints.init_from_checkpoint(checkpoint_dir, {
              "some_scope/": "useful_scope/"})

        # Mapping is not to scope name.
        with self.assertRaises(ValueError):
          checkpoints.init_from_checkpoint(checkpoint_dir, {
              "some_scope/": "useful_scope"})

if __name__ == "__main__":
  tf.test.main()
