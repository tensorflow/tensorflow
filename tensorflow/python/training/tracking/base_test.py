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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import util


class InterfaceTests(test.TestCase):

  def testOverwrite(self):
    root = base.Trackable()
    leaf = base.Trackable()
    root._track_trackable(leaf, name="leaf")
    (current_name, current_dependency), = root._checkpoint_dependencies
    self.assertIs(leaf, current_dependency)
    self.assertEqual("leaf", current_name)
    duplicate_name_dep = base.Trackable()
    with self.assertRaises(ValueError):
      root._track_trackable(duplicate_name_dep, name="leaf")
    root._track_trackable(duplicate_name_dep, name="leaf", overwrite=True)
    (current_name, current_dependency), = root._checkpoint_dependencies
    self.assertIs(duplicate_name_dep, current_dependency)
    self.assertEqual("leaf", current_name)

  def testAddVariableOverwrite(self):
    root = base.Trackable()
    a = root._add_variable_with_custom_getter(
        name="v", shape=[], getter=variable_scope.get_variable)
    self.assertEqual([root, a], util.list_objects(root))
    with ops.Graph().as_default():
      b = root._add_variable_with_custom_getter(
          name="v", shape=[], overwrite=True,
          getter=variable_scope.get_variable)
      self.assertEqual([root, b], util.list_objects(root))
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          ValueError, "already declared as a dependency"):
        root._add_variable_with_custom_getter(
            name="v", shape=[], overwrite=False,
            getter=variable_scope.get_variable)

  def testAssertConsumedWithUnusedPythonState(self):
    has_config = base.Trackable()
    has_config.get_config = lambda: {}
    saved = util.Checkpoint(obj=has_config)
    save_path = saved.save(os.path.join(self.get_temp_dir(), "ckpt"))
    restored = util.Checkpoint(obj=base.Trackable())
    restored.restore(save_path).assert_consumed()

  def testAssertConsumedFailsWithUsedPythonState(self):
    has_config = base.Trackable()
    attributes = {
        "foo_attr": functools.partial(
            base.PythonStringStateSaveable,
            state_callback=lambda: "",
            restore_callback=lambda x: None)}
    has_config._gather_saveables_for_checkpoint = lambda: attributes
    saved = util.Checkpoint(obj=has_config)
    save_path = saved.save(os.path.join(self.get_temp_dir(), "ckpt"))
    restored = util.Checkpoint(obj=base.Trackable())
    status = restored.restore(save_path)
    with self.assertRaisesRegexp(AssertionError, "foo_attr"):
      status.assert_consumed()

  def testBuggyGetConfig(self):

    class NotSerializable(object):
      pass

    class GetConfigRaisesError(base.Trackable):

      def get_config(self):
        return NotSerializable()

    util.Checkpoint(obj=GetConfigRaisesError()).save(
        os.path.join(self.get_temp_dir(), "ckpt"))


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
