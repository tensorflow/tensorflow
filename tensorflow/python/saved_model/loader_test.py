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

import os
import shutil

from absl.testing import parameterized

from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils
from tensorflow.python.training import saver as tf_saver


def _get_export_dir(label):
  return os.path.join(test.get_temp_dir(), label)


def _tensor_name(name):
  if variable_scope.resource_variables_enabled():
    return name + "/Read/ReadVariableOp:0"
  return name + ":0"


SIMPLE_ADD_SAVED_MODEL = _get_export_dir("simple_add_saved_model")
SAVED_MODEL_WITH_MAIN_OP = _get_export_dir("saved_model_with_main_op")


def build_graph_helper():
  g = ops.Graph()
  with g.as_default():
    x = variables.VariableV1(5, name="x")
    y = variables.VariableV1(11, name="y")
    z = x + y

    foo_sig_def = signature_def_utils.build_signature_def({
        "foo_input": utils.build_tensor_info(x)
    }, {"foo_output": utils.build_tensor_info(z)})
    bar_sig_def = signature_def_utils.build_signature_def({
        "bar_x": utils.build_tensor_info(x),
        "bar_y": utils.build_tensor_info(y)
    }, {"bar_z": utils.build_tensor_info(z)})
  return g, {"foo": foo_sig_def, "bar": bar_sig_def}, y


@parameterized.parameters((saved_model_builder.SavedModelBuilder,),
                          (saved_model_builder._SavedModelBuilder,))
class SavedModelLoaderTest(test.TestCase, parameterized.TestCase):

  def export_simple_graph(self, builder_cls):
    g, sig_def_map, _ = build_graph_helper()
    with session.Session(graph=g) as sess:
      self.evaluate(variables.global_variables_initializer())
      builder = builder_cls(SIMPLE_ADD_SAVED_MODEL)
      builder.add_meta_graph_and_variables(sess, ["foo_graph"], sig_def_map)
      builder.save()

  def export_graph_with_main_op(self, builder_cls):
    g, sig_def_map, y = build_graph_helper()
    with session.Session(graph=g) as sess:
      self.evaluate(variables.global_variables_initializer())
      assign_op = control_flow_ops.group(state_ops.assign(y, 7))

      builder = builder_cls(SAVED_MODEL_WITH_MAIN_OP)

      if builder_cls == saved_model_builder._SavedModelBuilder:
        builder.add_meta_graph_and_variables(
            sess, ["foo_graph"], sig_def_map, init_op=assign_op)
      else:
        builder.add_meta_graph_and_variables(
            sess, ["foo_graph"], sig_def_map, main_op=assign_op)
      builder.save()

  def tearDown(self):
    super(SavedModelLoaderTest, self).tearDown()
    shutil.rmtree(test.get_temp_dir(), ignore_errors=True)

  def test_load_function(self, builder_cls):
    # Force test to run in graph mode.
    # The SavedModelLoader.load method is a v1-only API that requires a session
    # to work.
    with ops.Graph().as_default():
      self.export_simple_graph(builder_cls)
      loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
      with self.session(graph=ops.Graph()) as sess:
        loader.load(sess, ["foo_graph"])
        self.assertEqual(5, sess.run(_tensor_name("x")))
        self.assertEqual(11, sess.run(_tensor_name("y")))

      self.export_graph_with_main_op(builder_cls)
      loader2 = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
      with self.session(graph=ops.Graph()) as sess:
        loader2.load(sess, ["foo_graph"])
        self.assertEqual(5, sess.run(_tensor_name("x")))
        self.assertEqual(7, sess.run(_tensor_name("y")))

  def test_load_graph(self, builder_cls):
    self.export_simple_graph(builder_cls)
    loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
    graph = ops.Graph()
    loader.load_graph(graph, ["foo_graph"])

    x = graph.get_tensor_by_name(_tensor_name("x"))
    y = graph.get_tensor_by_name(_tensor_name("y"))

    with self.assertRaises(KeyError):
      graph.get_tensor_by_name(_tensor_name("z"))

    with graph.as_default(), self.session():
      # Check that x and y are not initialized
      with self.assertRaises(errors.FailedPreconditionError):
        self.evaluate(x)
      with self.assertRaises(errors.FailedPreconditionError):
        self.evaluate(y)

  def test_load_with_import_scope(self, builder_cls):
    # Force test to run in graph mode.
    # The SavedModelLoader.restore_variables and SavedModelLoader.run_init_ops
    # methods are v1-only APIs that require a session to work.
    with ops.Graph().as_default():
      self.export_graph_with_main_op(builder_cls)
      loader = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
      with self.session(graph=ops.Graph()) as sess:
        saver, _ = loader.load_graph(
            sess.graph, ["foo_graph"], import_scope="baz")

        # The default saver should not work when the import scope is set.
        with self.assertRaises(errors.NotFoundError):
          loader.restore_variables(sess, tf_saver.Saver())

        loader.restore_variables(sess, saver)

        if builder_cls == saved_model_builder._SavedModelBuilder:
          with self.assertRaises(errors.NotFoundError):
            loader.run_init_ops(sess, ["foo_graph"])
          loader.run_init_ops(sess, ["foo_graph"], import_scope="baz")
        else:
          loader.run_init_ops(sess, ["foo_graph"])

        self.assertEqual(5, sess.run(_tensor_name("baz/x")))
        self.assertEqual(7, sess.run(_tensor_name("baz/y")))

      # Test combined load function.
      loader = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
      with self.session(graph=ops.Graph()) as sess:
        loader.load(sess, ["foo_graph"], import_scope="baa")
        self.assertEqual(5, sess.run(_tensor_name("baa/x")))
        self.assertEqual(7, sess.run(_tensor_name("baa/y")))

  def test_restore_variables(self, builder_cls):
    # Force test to run in graph mode.
    # The SavedModelLoader.restore_variables method is a v1-only API requiring a
    # session to work.
    with ops.Graph().as_default():
      self.export_graph_with_main_op(builder_cls)
      loader = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
      with self.session() as sess:
        x = variables.VariableV1(0, name="x")
        y = variables.VariableV1(0, name="y")
        z = x * y

        self.evaluate(variables.global_variables_initializer())

        # There are variables to restore, so a saver must be created.
        with self.assertRaises(ValueError):
          loader.restore_variables(sess, None)

        loader.restore_variables(sess, tf_saver.Saver())
        self.assertEqual(55, self.evaluate(z))

  def test_run_init_op(self, builder_cls):
    # Force test to run in graph mode.
    # The SavedModelLoader.restore_variables and SavedModelLoader.run_init_ops
    # methods are v1-only APIs that require a session to work.
    with ops.Graph().as_default():
      self.export_graph_with_main_op(builder_cls)
      loader = loader_impl.SavedModelLoader(SAVED_MODEL_WITH_MAIN_OP)
      graph = ops.Graph()
      saver, _ = loader.load_graph(graph, ["foo_graph"])
      with self.session(graph=graph) as sess:
        loader.restore_variables(sess, saver)
        self.assertEqual(5, sess.run(_tensor_name("x")))
        self.assertEqual(11, sess.run(_tensor_name("y")))

        loader.run_init_ops(sess, ["foo_graph"])
        self.assertEqual(5, sess.run(_tensor_name("x")))
        self.assertEqual(7, sess.run(_tensor_name("y")))

  def test_parse_saved_model(self, builder_cls):
    self.export_simple_graph(builder_cls)
    loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
    meta_graph = loader.get_meta_graph_def_from_tags(["foo_graph"])
    self.assertIsNotNone(meta_graph)
    self.assertIn("foo", meta_graph.signature_def)
    self.assertIn("bar", meta_graph.signature_def)

  def test_load_invalid_meta_graph(self, builder_cls):
    self.export_simple_graph(builder_cls)
    loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
    with self.assertRaises(RuntimeError):
      loader.get_meta_graph_def_from_tags([])
    with self.assertRaises(RuntimeError):
      loader.get_meta_graph_def_from_tags([""])
    with self.assertRaises(RuntimeError):
      loader.get_meta_graph_def_from_tags(["not_a_graph"])

  def test_load_saved_model_with_no_variables(self, builder_cls):
    """Test that SavedModel runs saver when there appear to be no variables.

    When no variables are detected, this may mean that the variables were saved
    to different collections, or the collections weren't saved to the
    SavedModel. If the SavedModel MetaGraphDef contains a saver, it should still
    run in either of these cases.

    Args:
      builder_cls: SavedModelBuilder or _SavedModelBuilder class
    """
    # Force test to run in graph mode.
    # The SavedModelBuilder.add_meta_graph_and_variables and
    # SavedModelLoader.load methods are v1-only APIs that require a session to
    # work.
    with ops.Graph().as_default():
      path = _get_export_dir("no_variable_saved_model")
      with session.Session(graph=ops.Graph()) as sess:
        x = variables.VariableV1(
            5, name="x", collections=["not_global_variable"])
        y = variables.VariableV1(
            11, name="y", collections=["not_global_variable"])
        self.assertFalse(variables._all_saveable_objects())
        z = x + y
        self.evaluate(variables.variables_initializer([x, y]))

        foo_sig_def = signature_def_utils.build_signature_def(
            {"foo_input": utils.build_tensor_info(x)},
            {"foo_output": utils.build_tensor_info(z)})

        builder = saved_model_builder.SavedModelBuilder(path)
        builder.add_meta_graph_and_variables(
            sess, ["foo_graph"], {"foo": foo_sig_def},
            saver=tf_saver.Saver([x, y]))
        builder.save()

      loader = loader_impl.SavedModelLoader(path)
      with self.session(graph=ops.Graph()) as sess:
        saver, _ = loader.load_graph(sess.graph, ["foo_graph"])
        self.assertFalse(variables._all_saveable_objects())
        self.assertIsNotNone(saver)

      with self.session(graph=ops.Graph()) as sess:
        loader.load(sess, ["foo_graph"])
        self.assertEqual(5, sess.run(_tensor_name("x")))
        self.assertEqual(11, sess.run(_tensor_name("y")))

  def test_load_saved_model_graph_with_return_elements(self, builder_cls):
    """Ensure that the correct elements are returned."""
    self.export_simple_graph(builder_cls)
    loader = loader_impl.SavedModelLoader(SIMPLE_ADD_SAVED_MODEL)
    graph = ops.Graph()
    _, ret = loader.load_graph(graph, ["foo_graph"],
                               return_elements=["y:0", "x:0"])

    self.assertEqual(graph.get_tensor_by_name("y:0"), ret[0])
    self.assertEqual(graph.get_tensor_by_name("x:0"), ret[1])

    with self.assertRaisesRegex(ValueError, "not found in graph"):
      loader.load_graph(graph, ["foo_graph"], return_elements=["z:0"])

  def test_parse_saved_model_exception(self, builder_cls):
    """Test that error message for not exist model have OS-depend delimiter in path"""
    path = _get_export_dir("not_existing_dir")
    pattern = os.path.sep + "{"
    with self.assertRaises(IOError) as err:
      loader_impl.parse_saved_model(path)
    self.assertTrue(pattern in str(err.exception))


if __name__ == "__main__":
  test.main()
