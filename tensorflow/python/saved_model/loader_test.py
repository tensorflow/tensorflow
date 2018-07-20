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
# ==============================================================================
"""Tests for SavedModelLoader class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils
from tensorflow.python.training import saver as tf_saver


def _get_export_dir(label):
  return os.path.join(test.get_temp_dir(), label)

SIMPLE_ADD_SAVED_MODEL = _get_export_dir("simple_add_saved_model")
SAVED_MODEL_WITH_MAIN_OP = _get_export_dir("saved_model_with_main_op")


class SavedModelLoaderTest(test.TestCase):

  def setUp(self):
    """Write test SavedModels to a temp directory."""
    with session.Session(graph=ops.Graph()) as sess:
      x = variables.Variable(5, name="x")
      y = variables.Variable(11, name="y")
      z = x + y
      sess.run(variables.global_variables_initializer())

      foo_sig_def = signature_def_utils.build_signature_def(
          {"foo_input": utils.build_tensor_info(x)},
          {"foo_output": utils.build_tensor_info(z)})
      bar_sig_def = signature_def_utils.build_signature_def(
          {"bar_x": utils.build_tensor_info(x),
           "bar_y": utils.build_tensor_info(y)},
          {"bar_z": utils.build_tensor_info(z)})

      builder = saved_model_builder.SavedModelBuilder(SIMPLE_ADD_SAVED_MODEL)
      builder.add_meta_graph_and_variables(
          sess, ["foo_graph"], {"foo": foo_sig_def, "bar": bar_sig_def})
      builder.save()

      # Write SavedModel with a main_op
      assign_op = control_flow_ops.group(state_ops.assign(y, 7))

      builder = saved_model_builder.SavedModelBuilder(SAVED_MODEL_WITH_MAIN_OP)
      builder.add_meta_graph_and_variables(
          sess, ["foo_graph"], {"foo": foo_sig_def, "bar": bar_sig_def},
          main_op=assign_op)
      builder.save()

  def tearDown(self):
    file_io.delete_recursively(test.get_temp_dir())

  def test_load_function(self):
    loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
    with self.test_session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo_graph"])
      self.assertEqual(5, sess.graph.get_tensor_by_name("x:0").eval())
      self.assertEqual(11, sess.graph.get_tensor_by_name("y:0").eval())

    loader2 = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
    with self.test_session(graph=ops.Graph()) as sess:
      loader2.load(sess, ["foo_graph"])
      self.assertEqual(5, sess.graph.get_tensor_by_name("x:0").eval())
      self.assertEqual(7, sess.graph.get_tensor_by_name("y:0").eval())

  def test_load_graph(self):
    loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
    graph = ops.Graph()
    loader.load_graph(graph, ["foo_graph"])

    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")

    with self.assertRaises(KeyError):
      graph.get_tensor_by_name("z:0")

    with self.test_session(graph=graph) as sess:
      # Check that x and y are not initialized
      with self.assertRaises(errors.FailedPreconditionError):
        sess.run(x)
      with self.assertRaises(errors.FailedPreconditionError):
        sess.run(y)

  def test_load_with_import_scope(self):
    loader = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
    with self.test_session(graph=ops.Graph()) as sess:
      saver = loader.load_graph(sess.graph, ["foo_graph"], import_scope="baz")

      # The default saver should not work when the import scope is set.
      with self.assertRaises(errors.NotFoundError):
        loader.restore_variables(sess, tf_saver.Saver())

      loader.restore_variables(sess, saver)
      loader.run_init_ops(sess, ["foo_graph"])

      self.assertEqual(5, sess.graph.get_tensor_by_name("baz/x:0").eval())
      self.assertEqual(7, sess.graph.get_tensor_by_name("baz/y:0").eval())

    # Test combined load function.
    loader = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
    with self.test_session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo_graph"], import_scope="baa")
      self.assertEqual(5, sess.graph.get_tensor_by_name("baa/x:0").eval())
      self.assertEqual(7, sess.graph.get_tensor_by_name("baa/y:0").eval())

  def test_restore_variables(self):
    loader = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
    with self.test_session(graph=ops.Graph()) as sess:
      x = variables.Variable(0, name="x")
      y = variables.Variable(0, name="y")
      z = x * y

      sess.run(variables.global_variables_initializer())

      # There are variables to restore, so a saver must be created.
      with self.assertRaises(ValueError):
        loader.restore_variables(sess, None)

      loader.restore_variables(sess, tf_saver.Saver())
      self.assertEqual(55, z.eval())

  def test_run_init_op(self):
    loader = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
    graph = ops.Graph()
    saver = loader.load_graph(graph, ["foo_graph"])
    with self.test_session(graph=graph) as sess:
      loader.restore_variables(sess, saver)
      self.assertEqual(5, sess.graph.get_tensor_by_name("x:0").eval())
      self.assertEqual(11, sess.graph.get_tensor_by_name("y:0").eval())

      loader.run_init_ops(sess, ["foo_graph"])
      self.assertEqual(5, sess.graph.get_tensor_by_name("x:0").eval())
      self.assertEqual(7, sess.graph.get_tensor_by_name("y:0").eval())

  def test_parse_saved_model(self):
    loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
    meta_graph = loader.get_meta_graph_def_from_tags(["foo_graph"])
    self.assertIsNotNone(meta_graph)
    self.assertIn("foo", meta_graph.signature_def)
    self.assertIn("bar", meta_graph.signature_def)

  def test_load_invalid_meta_graph(self):
    loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
    with self.assertRaises(RuntimeError):
      loader.get_meta_graph_def_from_tags([])
    with self.assertRaises(RuntimeError):
      loader.get_meta_graph_def_from_tags([""])
    with self.assertRaises(RuntimeError):
      loader.get_meta_graph_def_from_tags(["not_a_graph"])

  def test_load_saved_model_with_no_variables(self):
    """Test that SavedModel runs saver when there appear to be no variables.

    When no variables are detected, this may mean that the variables were saved
    to different collections, or the collections weren't saved to the
    SavedModel. If the SavedModel MetaGraphDef contains a saver, it should still
    run in either of these cases.
    """
    path = _get_export_dir("no_variable_saved_model")
    with session.Session(graph=ops.Graph()) as sess:
      x = variables.Variable(5, name="x", collections=["not_global_variable"])
      y = variables.Variable(11, name="y", collections=["not_global_variable"])
      self.assertFalse(variables._all_saveable_objects())
      z = x + y
      sess.run(variables.variables_initializer([x, y]))

      foo_sig_def = signature_def_utils.build_signature_def(
          {"foo_input": utils.build_tensor_info(x)},
          {"foo_output": utils.build_tensor_info(z)})

      builder = saved_model_builder.SavedModelBuilder(path)
      builder.add_meta_graph_and_variables(
          sess, ["foo_graph"], {"foo": foo_sig_def},
          saver=tf_saver.Saver([x, y]))
      builder.save()

    loader = loader_impl.SavedModelLoader(path)
    with self.test_session(graph=ops.Graph()) as sess:
      saver = loader.load_graph(sess.graph, ["foo_graph"])
      self.assertFalse(variables._all_saveable_objects())
      self.assertIsNotNone(saver)

    with self.test_session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo_graph"])
      self.assertEqual(5, sess.graph.get_tensor_by_name("x:0").eval())
      self.assertEqual(11, sess.graph.get_tensor_by_name("y:0").eval())


if __name__ == "__main__":
  test.main()
