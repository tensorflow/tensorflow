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
"""Tests for SavedModel Reader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.contrib.saved_model.python.saved_model import reader
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants


def tearDownModule():
  file_io.delete_recursively(test.get_temp_dir())


class ReaderTest(test.TestCase):

  def _init_and_validate_variable(self, sess, variable_name, variable_value):
    v = variables.Variable(variable_value, name=variable_name)
    sess.run(variables.global_variables_initializer())
    self.assertEqual(variable_value, v.eval())

  def testReadSavedModelValid(self):
    saved_model_dir = os.path.join(test.get_temp_dir(), "valid_saved_model")
    builder = saved_model_builder.SavedModelBuilder(saved_model_dir)
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)
      builder.add_meta_graph_and_variables(sess, [tag_constants.TRAINING])
    builder.save()

    actual_saved_model_pb = reader.read_saved_model(saved_model_dir)
    self.assertEqual(len(actual_saved_model_pb.meta_graphs), 1)
    self.assertEqual(
        len(actual_saved_model_pb.meta_graphs[0].meta_info_def.tags), 1)
    self.assertEqual(actual_saved_model_pb.meta_graphs[0].meta_info_def.tags[0],
                     tag_constants.TRAINING)

  def testReadSavedModelInvalid(self):
    saved_model_dir = os.path.join(test.get_temp_dir(), "invalid_saved_model")
    with self.assertRaisesRegexp(
        IOError, "SavedModel file does not exist at: %s" % saved_model_dir):
      reader.read_saved_model(saved_model_dir)

  def testGetSavedModelTagSets(self):
    saved_model_dir = os.path.join(test.get_temp_dir(), "test_tags")
    builder = saved_model_builder.SavedModelBuilder(saved_model_dir)

    # Graph with a single variable. SavedModel invoked to:
    # - add with weights.
    # - a single tag (from predefined constants).
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)
      builder.add_meta_graph_and_variables(sess, [tag_constants.TRAINING])

    # Graph that updates the single variable. SavedModel invoked to:
    # - simply add the model (weights are not updated).
    # - a single tag (from predefined constants).
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 43)
      builder.add_meta_graph([tag_constants.SERVING])

    # Graph that updates the single variable. SavedModel is invoked:
    # - to add the model (weights are not updated).
    # - multiple predefined tags.
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 44)
      builder.add_meta_graph([tag_constants.SERVING, tag_constants.GPU])

    # Graph that updates the single variable. SavedModel is invoked:
    # - to add the model (weights are not updated).
    # - multiple predefined tags for serving on TPU.
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 44)
      builder.add_meta_graph([tag_constants.SERVING, tag_constants.TPU])

    # Graph that updates the single variable. SavedModel is invoked:
    # - to add the model (weights are not updated).
    # - multiple custom tags.
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 45)
      builder.add_meta_graph(["foo", "bar"])

    # Save the SavedModel to disk.
    builder.save()

    actual_tags = reader.get_saved_model_tag_sets(saved_model_dir)
    expected_tags = [["train"], ["serve"], ["serve", "gpu"], ["serve", "tpu"],
                     ["foo", "bar"]]
    self.assertEqual(expected_tags, actual_tags)


if __name__ == "__main__":
  test.main()
