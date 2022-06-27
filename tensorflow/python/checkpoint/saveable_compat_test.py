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

"""Tests for SaveableObject compatibility."""

import os

from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint.testdata import generate_checkpoint
from tensorflow.python.eager import test
from tensorflow.python.training import checkpoint_utils


_LEGACY_TABLE_CHECKPOINT_PATH = test.test_src_dir_path(
    "python/checkpoint/testdata/table_legacy_saveable_object")


class SaveableCompatTest(test.TestCase):

  def test_lookup_table_compatibility(self):
    table_module = generate_checkpoint.TableModule()
    ckpt = checkpoint.Checkpoint(table_module)
    checkpoint_directory = self.get_temp_dir()
    checkpoint_path = os.path.join(checkpoint_directory, "ckpt")
    ckpt.write(checkpoint_path)

    # Ensure that the checkpoint metadata and keys are the same.
    legacy_metadata = checkpoint.object_metadata(_LEGACY_TABLE_CHECKPOINT_PATH)
    metadata = checkpoint.object_metadata(checkpoint_path)

    def _get_table_node(object_metadata):
      for child in object_metadata.nodes[0].children:
        if child.local_name == "lookup_table":
          return object_metadata.nodes[child.node_id]

    table_proto = _get_table_node(metadata)
    legacy_table_proto = _get_table_node(legacy_metadata)
    self.assertAllEqual(
        [table_proto.attributes[0].name,
         table_proto.attributes[0].checkpoint_key],
        [legacy_table_proto.attributes[0].name,
         legacy_table_proto.attributes[0].checkpoint_key])

    legacy_reader = checkpoint_utils.load_checkpoint(
        _LEGACY_TABLE_CHECKPOINT_PATH)
    reader = checkpoint_utils.load_checkpoint(checkpoint_path)
    self.assertEqual(
        legacy_reader.get_variable_to_shape_map().keys(),
        reader.get_variable_to_shape_map().keys())

    # Ensure that previous checkpoint can be loaded into current table.
    ckpt.read(_LEGACY_TABLE_CHECKPOINT_PATH).assert_consumed()


if __name__ == "__main__":
  test.main()
