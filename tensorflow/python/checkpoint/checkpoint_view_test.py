# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the checkpoint view."""

import os

from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_view
from tensorflow.python.eager import test
from tensorflow.python.trackable import base


class CheckpointViewTest(test.TestCase):

  def test_children(self):
    root = base.Trackable()
    leaf = base.Trackable()
    root._track_trackable(leaf, name="leaf")
    root_ckpt = trackable_utils.Checkpoint(root=root)
    root_save_path = root_ckpt.save(
        os.path.join(self.get_temp_dir(), "root_ckpt"))
    current_name, node_id = next(
        iter(
            checkpoint_view.CheckpointView(root_save_path).children(0).items()))
    self.assertEqual("leaf", current_name)
    self.assertEqual(1, node_id)


if __name__ == "__main__":
  test.main()
