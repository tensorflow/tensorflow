# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for restore.py."""

import os

from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import restore
from tensorflow.python.eager import test
from tensorflow.python.module import module
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.training.saving import saveable_object


class ExtractSaveablenameTest(test.TestCase):

  def test_standard_saveable_name(self):
    self.assertEqual(
        "object_path/.ATTRIBUTES/",
        restore._extract_saveable_name("object_path/.ATTRIBUTES/123"))
    self.assertEqual(
        "object/path/ATTRIBUTES/.ATTRIBUTES/",
        restore._extract_saveable_name("object/path/ATTRIBUTES/.ATTRIBUTES/"))

  def test_restore_nodes_error_cases_high_level(self):
    root = autotrackable.AutoTrackable()
    root.leaf = autotrackable.AutoTrackable()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = autotrackable.AutoTrackable()
    root2.leaf = autotrackable.AutoTrackable()

    with self.assertRaisesRegex(
        ValueError,
        "Expecting a dictionary of node_id to Trackable for nodes_to_restore."):
      restore.restore_nodes(root_save_path, [0, 1])

    with self.assertRaisesRegex(
        ValueError,
        "The expected node_id: 3 to Trackable <.*?> to restore does not exist "
        "in the checkpoint."):
      restore.restore_nodes(root_save_path, {3: root2})

    with self.assertRaisesRegex(
        ValueError,
        "Expecting a valid Trackable to node_id: 0 but got trackable: None."):
      restore.restore_nodes(root_save_path, {0: None})

  def test_restore_nodes_error_cases_trackable_ckpt_view_mismatch(self):

    class MyTrackable(base.Trackable):

      def __init__(self):
        self.a = module.Module()

    class MyTrackable2(base.Trackable):

      def __init__(self):
        self.a = variables.Variable(5.0)

      def _serialize_to_tensors(self):
        return {"a": variables.Variable(5.0)}

    root = MyTrackable()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackable2()
    with self.assertRaisesRegex(
        ValueError,
        "Trackable <.*?> expects checkpointed values but checkpoint does not "
        "contain serialized tensors for node_id: 0."):
      restore.restore_nodes(root_save_path, {0: root2})

  def test_restore_nodes_has_serialize_to_tensor(self):

    class MyTrackable(base.Trackable):

      def __init__(self):
        self.a = variables.Variable(5.0)

      def _restore_from_tensors(self, restored_tensors):
        self.a.assign(restored_tensors["a"])

      def _serialize_to_tensors(self):
        return {"a": self.a}

    root = MyTrackable()
    leaf = MyTrackable()
    root._track_trackable(leaf, "leaf")
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackable()
    leaf2 = MyTrackable()
    root2._track_trackable(leaf2, "leaf")
    root2.a.assign(3.0)

    # Restore root
    restore.restore_nodes(root_save_path, {0: root2})
    self.assertEqual(root2.a.numpy(), 5.0)  # Restored from 3.0 to 5.0
    self.assertEqual(leaf2.a.numpy(), 5.0)  # Unchanged

    root3 = MyTrackable()
    leaf3 = MyTrackable()
    root3._track_trackable(leaf3, "leaf")
    leaf3.a.assign(3.0)

    # Restore leaf
    restore.restore_nodes(root_save_path, {1: leaf3})
    self.assertEqual(root3.a.numpy(), 5.0)  # Unchanged
    self.assertEqual(leaf3.a.numpy(), 5.0)  # Restored from 3.0 to 5.0.

  def test_restore_nodes_with_different_number_of_serialized_to_tensors(self):

    class MyTrackableA(base.Trackable):

      def __init__(self):
        self.a = variables.Variable(5.0)

      def _restore_from_tensors(self, restored_tensors):
        self.a.assign(restored_tensors["a"])

      def _serialize_to_tensors(self):
        return {"a": self.a}

    class MyTrackableAandB(base.Trackable):

      def __init__(self):
        self.a = variables.Variable(5.0)
        self.b = variables.Variable(6.0)

      def _restore_from_tensors(self, restored_tensors):
        self.a.assign(restored_tensors["a"])
        self.b.assign(restored_tensors["b"])

      def _serialize_to_tensors(self):
        return {"a": self.a, "b": self.b}

    root = MyTrackableA()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackableAandB()

    with self.assertRaisesRegex(
        ValueError,
        "Size for serialized_tensors for Trackable: 2 did not match size for "
        "serialized_tensors for checkpoint: 1."):
      restore.restore_nodes(root_save_path, {0: root2})

    root = MyTrackableAandB()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackableA()

    with self.assertRaisesRegex(
        ValueError,
        "Size for serialized_tensors for Trackable: 1 did not match size for "
        "serialized_tensors for checkpoint: 2."):
      restore.restore_nodes(root_save_path, {0: root2})

  def test_restore_nodes_not_serialize_to_tensor(self):

    class _VarSaveable(saveable_object.SaveableObject):

      def __init__(self, obj, name):
        self.obj = obj
        specs = [saveable_object.SaveSpec(obj.a, "", name + "-a")]
        super(_VarSaveable, self).__init__(None, specs, name)

      def restore(self, restored_tensors, restored_shapes):
        del restored_shapes  # Unused.
        self.obj.a.assign(restored_tensors[0])

    class MyTrackable(base.Trackable):

      def __init__(self):
        self.a = variables.Variable(5.0)

      def _gather_saveables_for_checkpoint(self):
        return {"a": lambda name: _VarSaveable(self, name)}

    root = MyTrackable()
    leaf = MyTrackable()
    root._track_trackable(leaf, "leaf")
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackable()
    leaf2 = MyTrackable()
    root2._track_trackable(leaf2, "leaf")
    root2.a.assign(3.0)

    # Restore root
    restore.restore_nodes(root_save_path, {0: root2})
    self.assertEqual(root2.a.numpy(), 5.0)  # Restored from 3.0 to 5.0
    self.assertEqual(leaf2.a.numpy(), 5.0)  # Unchanged

    root3 = MyTrackable()
    leaf3 = MyTrackable()
    root3._track_trackable(leaf3, "leaf")
    leaf3.a.assign(3.0)

    # Restore leaf
    restore.restore_nodes(root_save_path, {1: leaf3})
    self.assertEqual(root3.a.numpy(), 5.0)  # Unchanged
    self.assertEqual(leaf3.a.numpy(), 5.0)  # Restored from 3.0 to 5.0.

  def test_restore_nodes_not_serialize_to_tensor_error_cases(self):

    class _VarSaveable(saveable_object.SaveableObject):

      def __init__(self, obj, name):
        self.obj = obj
        specs = [saveable_object.SaveSpec(obj.a, "", name + "-a")]
        super(_VarSaveable, self).__init__(None, specs, name)

      def restore(self, restored_tensors, restored_shapes):
        del restored_shapes  # Unused.
        self.obj.a.assign(restored_tensors[0])

    class MyTrackable(base.Trackable):

      def __init__(self):
        self.a = module.Module()

    class MyTrackableWithSingleSaveable(base.Trackable):

      def __init__(self):
        self.a = variables.Variable(1.0)

      def _gather_saveables_for_checkpoint(self):
        return {"foo": lambda name: _VarSaveable(self, name)}

    class MyTrackableWithMultiSaveables(base.Trackable):

      def __init__(self):
        self.a = variables.Variable(1.0)

      def _gather_saveables_for_checkpoint(self):
        return {
            "foo": lambda name: _VarSaveable(self, name),
            "bar": lambda name: _VarSaveable(self, name)
        }

    root = MyTrackable()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackableWithMultiSaveables()
    with self.assertRaisesRegex(
        ValueError,
        "Trackable <.*?> expects checkpointed values but checkpoint does not "
        "contain serialized tensors for node_id: 0."):
      restore.restore_nodes(root_save_path, {0: root2})

    root = MyTrackableWithSingleSaveable()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackableWithMultiSaveables()
    with self.assertRaisesRegex(
        ValueError,
        "Size for saveable_objects for Trackable: 2 did not match the size for "
        "serialized_tensors for checkpoint: 1."):
      restore.restore_nodes(root_save_path, {0: root2})

    root = MyTrackableWithMultiSaveables()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackableWithSingleSaveable()
    with self.assertRaisesRegex(
        ValueError,
        "Size for saveable_objects for Trackable: 1 did not match the size for "
        "serialized_tensors for checkpoint: 2."):
      restore.restore_nodes(root_save_path, {0: root2})

if __name__ == "__main__":
  test.main()
