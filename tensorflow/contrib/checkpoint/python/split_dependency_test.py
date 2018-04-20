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

import os

from tensorflow.contrib.checkpoint.python import split_dependency
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import checkpointable
from tensorflow.python.training import checkpointable_utils


def _split_variable_closure(variable):
  def _fill_save_buffer_fn(save_buffer):
    save_buffer["first_half"] = variable[:2]
    save_buffer["second_half"] = variable[2:]
  return _fill_save_buffer_fn


def _combine_variable_closure(variable):
  def _consume_restore_buffer_fn(restore_buffer):
    return variable.assign(
        array_ops.concat([restore_buffer["first_half"],
                          restore_buffer["second_half"]],
                         axis=0))
  return _consume_restore_buffer_fn


class SaveTensorSlicesAsDeps(checkpointable.CheckpointableBase):

  def __init__(self):
    self.combined = resource_variable_ops.ResourceVariable([0., 0., 0., 0.])
    split_dependencies = split_dependency.split_dependency(
        component_names=("first_half", "second_half"),
        component_dtypes=(self.combined.dtype,) * 2,
        fill_save_buffer_fn=_split_variable_closure(
            self.combined),
        consume_restore_buffer_fn=_combine_variable_closure(
            self.combined))
    for name, dep in split_dependencies.items():
      self._track_checkpointable(dep, name=name)


class HasRegularDeps(checkpointable.Checkpointable):

  def __init__(self):
    self.first_half = resource_variable_ops.ResourceVariable([0., 0.])
    self.second_half = resource_variable_ops.ResourceVariable([0., 0.])


class OnlyOneDep(checkpointable.Checkpointable):

  def __init__(self):
    self.first_half = resource_variable_ops.ResourceVariable([0., 0.])


class SplitTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testSaveRestoreSplitDep(self):
    save_checkpoint = checkpointable_utils.Checkpoint(
        dep=SaveTensorSlicesAsDeps())
    self.evaluate(save_checkpoint.dep.combined.assign([1., 2., 3., 4.]))
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = save_checkpoint.save(checkpoint_prefix)

    regular_deps = HasRegularDeps()
    regular_restore_checkpoint = checkpointable_utils.Checkpoint(
        dep=regular_deps)
    regular_restore_checkpoint.restore(
        save_path).assert_consumed().run_restore_ops()
    self.assertAllEqual([1., 2.], self.evaluate(regular_deps.first_half))
    self.assertAllEqual([3., 4.], self.evaluate(regular_deps.second_half))

    one_dep = OnlyOneDep()
    one_dep_restore_checkpoint = checkpointable_utils.Checkpoint(dep=one_dep)
    status = one_dep_restore_checkpoint.restore(save_path)
    with self.assertRaises(AssertionError):
      # Missing the second dependency.
      status.assert_consumed()
    status.run_restore_ops()
    self.assertAllEqual([1., 2.], self.evaluate(one_dep.first_half))

    restore_checkpoint = checkpointable_utils.Checkpoint()
    status = restore_checkpoint.restore(save_path)
    restore_checkpoint.dep = SaveTensorSlicesAsDeps()
    status.assert_consumed().run_restore_ops()
    self.assertAllEqual(
        [1., 2., 3., 4.],
        self.evaluate(restore_checkpoint.dep.combined))


if __name__ == "__main__":
  test.main()
