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

    root = MyTrackable2()
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))

    root2 = MyTrackable()
    with self.assertRaisesRegex(
        ValueError,
        "Trackable <.*?> does not expect checkpointed values but checkpoint "
        "contains serialized tensors: .*?"):
      restore.restore_nodes(root_save_path, {0: root2})


if __name__ == "__main__":
  test.main()
