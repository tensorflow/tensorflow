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
"""Tests for checkpoint_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.estimator import checkpoint_utils
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib


def _create_checkpoints(sess, checkpoint_dir):
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  checkpoint_state_name = "checkpoint"
  v1 = variable_scope.get_variable("var1", [1, 10])
  v2 = variable_scope.get_variable("var2", [10, 10])
  v3 = variable_scope.get_variable("var3", [100, 100])
  with variable_scope.variable_scope("useful_scope"):
    v4 = variable_scope.get_variable("var4", [9, 9])
  sess.run(variables.global_variables_initializer())
  v1_value, v2_value, v3_value, v4_value = sess.run([v1, v2, v3, v4])
  saver = saver_lib.Saver()
  saver.save(
      sess,
      checkpoint_prefix,
      global_step=0,
      latest_filename=checkpoint_state_name)
  return v1_value, v2_value, v3_value, v4_value


class CheckpointsTest(test.TestCase):

  def testNoCheckpoints(self):
    checkpoint_dir = self.get_temp_dir() + "/no_checkpoints"
    with self.assertRaises(errors_impl.OpError):
      self.assertAllEqual(
          checkpoint_utils.load_variable(checkpoint_dir, "var1"), [])

  def testNoTensor(self):
    checkpoint_dir = self.get_temp_dir()
    with self.test_session() as session:
      _, _, _, _ = _create_checkpoints(session, checkpoint_dir)
    with self.assertRaises(errors_impl.OpError):
      self.assertAllEqual(
          checkpoint_utils.load_variable(checkpoint_dir, "var5"), [])

  def testGetTensor(self):
    checkpoint_dir = self.get_temp_dir()
    with self.test_session() as session:
      v1, v2, v3, v4 = _create_checkpoints(session, checkpoint_dir)
    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "var1"), v1)
    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "var2"), v2)
    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "var3"), v3)
    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "useful_scope/var4"), v4)

  def testGetAllVariables(self):
    checkpoint_dir = self.get_temp_dir()
    with self.test_session() as session:
      _create_checkpoints(session, checkpoint_dir)
    self.assertEqual(
        checkpoint_utils.list_variables(checkpoint_dir),
        [("useful_scope/var4", [9, 9]), ("var1", [1, 10]), ("var2", [10, 10]),
         ("var3", [100, 100])])


if __name__ == "__main__":
  test.main()
