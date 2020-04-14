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
"""Tests for trackable object SavedModel save."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import graph_debug_info_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import saver
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util
from tensorflow.python.util import compat


class _ModelWithOptimizer(util.Checkpoint):

  def __init__(self):
    self.dense = core.Dense(1)
    self.optimizer = adam.Adam(0.01)

  @def_function.function(
      input_signature=(tensor_spec.TensorSpec([None, 2], dtypes.float32),
                       tensor_spec.TensorSpec([None], dtypes.float32)))
  def call(self, x, y):
    with backprop.GradientTape() as tape:
      loss = math_ops.reduce_mean((self.dense(x) - y) ** 2.)
    trainable_variables = self.dense.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    return {"loss": loss}


def _run_signature(session, meta_graph_def, inputs, signature_key):
  signature = meta_graph_def.signature_def[signature_key]
  assert set(inputs.keys()) == set(signature.inputs.keys())
  feed_dict = {}
  for arg_name in inputs.keys():
    input_tensor = session.graph.get_tensor_by_name(
        signature.inputs[arg_name].name)
    feed_dict[input_tensor] = inputs[arg_name]
  output_dict = {}
  for output_name, output_tensor_info in signature.outputs.items():
    output_dict[output_name] = session.graph.get_tensor_by_name(
        output_tensor_info.name)
  return session.run(output_dict, feed_dict=feed_dict)


def _import_and_infer(
    save_dir, inputs,
    signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
  """Import a SavedModel into a TF 1.x-style graph and run `signature_key`."""
  graph = ops.Graph()
  with graph.as_default(), session_lib.Session() as session:
    model = loader.load(session, [tag_constants.SERVING], save_dir)
    return _run_signature(session, model, inputs, signature_key)


class SaveTest(test.TestCase):

  def test_method_save_signature(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    root.f(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir, root.f)
    self.assertEqual(
        {"output_0": 2.},
        _import_and_infer(save_dir, {"x": 1.}))

  def test_method_save_concrete(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda z: {"out": 2. * z})
    root.f(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(
        root,
        save_dir,
        {"non_default_key": root.f.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.float32))})
    self.assertEqual(
        {"out": 2.},
        _import_and_infer(
            save_dir, {"z": 1.}, signature_key="non_default_key"))

  def test_method_save_annotated_function(self):
    # This test is only meaningful with Python 3 because Python 2's
    # inspect.getargspec doesn't save annotations.

    root = tracking.AutoTrackable()

    class UnknownType(object):  # pylint: disable=unused-variable
      pass

    def annotated_function(z):
      return {"out": 2. * z}

    # Same effect as annotating function like the following.
    # def annotated_function("z": UnknownType) -> UnknownType:
    # This is a workaround since Python 2 does not support annotations and
    # our presubmit linter catches it.
    annotated_function.__annotations__ = {
        "z": UnknownType,
        "return": UnknownType
    }

    root.f = def_function.function(annotated_function)
    root.f(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(
        root, save_dir, {
            "non_default_key":
                root.f.get_concrete_function(
                    tensor_spec.TensorSpec(None, dtypes.float32))
        })
    self.assertEqual({"out": 2.},
                     _import_and_infer(
                         save_dir, {"z": 1.}, signature_key="non_default_key"))

  def test_unbuilt_model_does_not_prevent_saving(self):
    root = util.Checkpoint(model=sequential.Sequential([core.Dense(2)]))
    save.save(root, os.path.join(self.get_temp_dir(), "saved_model"))

  def test_unsaveable_func_graph(self):
    root = module.Module()

    @def_function.function(input_signature=[])
    def nested_f():
      ops.get_default_graph().mark_as_unsaveable("ERROR MSG")
      return 1

    @def_function.function(input_signature=[])
    def f():
      return nested_f()

    root.f = f
    with self.assertRaisesRegexp(ValueError, "ERROR MSG"):
      save.save(root, os.path.join(self.get_temp_dir(), "saved_model"))

  def test_version_information_included(self):
    root = tracking.AutoTrackable()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)
    saved_model_proto = loader_impl.parse_saved_model(save_dir)
    self.assertEqual(
        versions.__version__,
        saved_model_proto.meta_graphs[0].meta_info_def.tensorflow_version)
    self.assertEqual(
        versions.__git_version__,
        saved_model_proto.meta_graphs[0].meta_info_def.tensorflow_git_version)

  def test_non_concrete_error(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: 2. * x)
    root.f(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(
        ValueError, "Expected a TensorFlow function"):
      save.save(root, save_dir, root.f)

  def test_captures_unreachable_variable(self):
    root = tracking.AutoTrackable()
    unreachable_variable = variables.Variable([5.0, 2.0])
    root.reachable_variable = variables.Variable([1.0, 3.0])

    @def_function.function
    def increase_variable(x):
      return 2 * unreachable_variable * x + root.reachable_variable

    root.f = increase_variable

    self.assertAllEqual([101.0, 83.0],
                        root.f(constant_op.constant([10.0, 20.0])).numpy())

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")

    with self.assertRaisesRegexp(KeyError, "not reachable from root"):
      save.save(root, save_dir)

  def test_nested_inputs(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: 2. * x[0],
        input_signature=([tensor_spec.TensorSpec(None, dtypes.float32),
                          tensor_spec.TensorSpec(None, dtypes.float32)],))
    root.f([constant_op.constant(1.), constant_op.constant(1.)])

  def test_nested_outputs(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: (2. * x, (3. * x, 4. * x)))
    root.f(constant_op.constant(1.))
    to_save = root.f.get_concrete_function(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(
        ValueError, "non-flat outputs"):
      save.save(root, save_dir, to_save)

  def test_nested_dict_outputs(self):
    root = util.Checkpoint(
        f=def_function.function(
            lambda x: {"a": 2. * x, "b": (3. * x, 4. * x)}))
    root.f(constant_op.constant(1.))
    to_save = root.f.get_concrete_function(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(
        ValueError, "dictionary containing non-Tensor value"):
      save.save(root, save_dir, to_save)

  def test_variable(self):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(
        lambda x: root.v1 * root.v2 * x)
    root.f(constant_op.constant(1.))
    to_save = root.f.get_concrete_function(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir, to_save)
    self.assertAllEqual({"output_0": 12.},
                        _import_and_infer(save_dir, {"x": 2.}))

  def test_optimizer(self):
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    model = _ModelWithOptimizer()
    first_loss = model.call(x, y)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(model, save_dir, model.call)
    second_loss = model.call(x, y)
    self.assertNotEqual(first_loss, second_loss)
    self.assertAllClose(
        second_loss,
        _import_and_infer(save_dir, {"x": [[3., 4.]], "y": [2.]}))

  def test_single_method_default_signature(self):
    model = _ModelWithOptimizer()
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    model.call(x, y)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(model, save_dir)
    self.assertIn("loss",
                  _import_and_infer(save_dir,
                                    {"x": [[3., 4.]], "y": [2.]}))

  def test_single_function_default_signature(self):
    model = tracking.AutoTrackable()
    model.f = def_function.function(lambda: 3., input_signature=())
    model.f()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(model, save_dir)
    self.assertAllClose({"output_0": 3.},
                        _import_and_infer(save_dir, {}))

  def test_single_function_no_signature(self):
    model = tracking.AutoTrackable()
    model.f = def_function.function(lambda: 3.)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(model, save_dir)

  def test_find_default_save_function(self):

    class ObjWithDefaultSignature(util.Checkpoint):

      @def_function.function(input_signature=[tensor_spec.TensorSpec(
          shape=None, dtype=dtypes.float32)])
      def _default_save_signature(self, x):
        return x + x + 1

    obj = ObjWithDefaultSignature()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(obj, save_dir)
    self.assertAllClose(
        {"output_0": 7.}, _import_and_infer(save_dir, {"x": 3.}))

  def test_docstring(self):

    class Adder(module.Module):

      @def_function.function(input_signature=[tensor_spec.TensorSpec(
          shape=None, dtype=dtypes.float32)])
      def add(self, x):
        return x + x + 1.

    to_save = Adder()
    to_save.add(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(to_save, save_dir)
    self.assertAllClose({"output_0": 7.},
                        _import_and_infer(save_dir, {"x": 3.}))

  def test_datastructures(self):

    class HasDatastructures(util.Checkpoint):

      def __init__(self):
        self.a = [1.]
        self.a.append(variables.Variable(2.))
        self.b = {"a": variables.Variable(3.)}

      @def_function.function(input_signature=[tensor_spec.TensorSpec(
          shape=None, dtype=dtypes.float32)])
      def add(self, x):
        return x + math_ops.add_n(self.a) + self.b["a"]

    to_save = HasDatastructures()
    to_save.add(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(to_save, save_dir)
    self.assertAllClose({"output_0": 10.},
                        _import_and_infer(save_dir, {"x": 4.}))

  def test_default_attr_stripping(self):

    class Complex(util.Checkpoint):

      @def_function.function(input_signature=[])
      def __call__(self):
        return math_ops.complex(
            constant_op.constant(1.),
            constant_op.constant(2.),
            name="complex")

    to_save = Complex()
    to_save()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(to_save, save_dir)
    graph = ops.Graph()
    with graph.as_default(), self.session(graph) as session:
      loader.load(session, [tag_constants.SERVING], save_dir)
      func, = [f for name, f in graph._functions.items() if "call" in name]
      complex_node, = [
          node for node in func.definition.node_def if node.op == "Complex"]
      self.assertNotIn("T", complex_node.attr)
      self.assertNotIn("Tout", complex_node.attr)

  def test_signature_attribute_reserved(self):
    root = util.Checkpoint(signatures=variables.Variable(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(ValueError, "del obj.signatures"):
      save.save(root, save_dir)
    del root.signatures
    save.save(root, save_dir)

  def test_function_with_captured_dataset(self):
    if test_util.is_gpu_available():
      self.skipTest("Currently broken when a GPU is available.")

    class HasDataset(module.Module):

      def __init__(self):
        super(HasDataset, self).__init__()
        self.dataset = (
            dataset_ops.Dataset.range(5)
            .map(lambda x: x ** 2))

      @def_function.function
      def __call__(self, x):
        current_sum = array_ops.zeros([], dtype=dtypes.int64)
        for element in self.dataset:
          current_sum += x * element
        return current_sum

    root = HasDataset()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(
        root, save_dir,
        signatures=root.__call__.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.int64)))
    self.assertAllClose({"output_0": 3 * (1 + 4 + 9 + 16)},
                        _import_and_infer(save_dir, {"x": 3}))

  def test_variable_args_cannot_be_used_as_signature(self):
    @def_function.function(input_signature=[
        resource_variable_ops.VariableSpec(shape=[], dtype=dtypes.int32)])
    def f(unused_v):
      return 1
    root = tracking.AutoTrackable()
    root.f = f.get_concrete_function()
    with self.assertRaisesRegexp(ValueError,
                                 "tf.Variable inputs cannot be exported"):
      save.save(root, os.path.join(self.get_temp_dir(), "saved_model"),
                signatures=root.f)

  def test_export_correct_output_shapes(self):
    """Asserts that nodes are exported with the correct number of output shapes.

    After backpropagation rewrite, functions are rewritten with additional
    outputs. When exporting to SavedModel, the shapes of the additional outputs
    were incorrectly added to the FunctionDef proto (b/133666530).
    """
    obj = tracking.AutoTrackable()
    obj.v = variables.Variable(2.)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(None, dtypes.float32)])
    def f(x):
      return (math_ops.multiply(obj.v, x),
              math_ops.multiply(obj.v, (x+1)),
              None)
    obj.f = f

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(None, dtypes.float32)])
    def g(x):
      return obj.f(x)[1]
    obj.g = g

    # After the following lines, the concrete functions of obj.g and obj.f are
    # rewritten with many extra outputs.
    with backprop.GradientTape():
      obj.g(constant_op.constant(3.0))

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(obj, save_dir, signatures={"g": obj.g})
    graph_def = loader_impl.parse_saved_model(save_dir).meta_graphs[0].graph_def

    def assert_correct_number_of_output_shapes(node):
      if node.op == "StatefulPartitionedCall":
        fn_name = node.attr["f"].func.name
        if fn_name.startswith("__inference_f"):
          self.assertLen(node.attr["_output_shapes"].list.shape, 2)
        if fn_name.startswith("__inference_g"):
          self.assertLen(node.attr["_output_shapes"].list.shape, 1)

    for f in graph_def.library.function:
      if(f.signature.name.startswith("__inference_f") or
         f.signature.name.startswith("__inference_g")):
        for node in f.node_def:
          assert_correct_number_of_output_shapes(node)

  def test_save_cached_variable(self):
    with ops.Graph().as_default(), session_lib.Session() as session:
      obj = tracking.AutoTrackable()
      obj.v = variables.Variable(2., caching_device=lambda op: op.device)
      obj.w = variables.Variable(3.)
      session.run([obj.v.initializer, obj.w.initializer])

      @def_function.function(input_signature=[])
      def f():
        return obj.v + obj.w

      obj.f = f
      save_dir = os.path.join(self.get_temp_dir(), "saved_model")
      save.save(obj, save_dir, signatures=obj.f)
      self.assertAllClose({"output_0": 5}, _import_and_infer(save_dir, {}))


class SavingOptionsTest(test.TestCase):

  def testOpNameSpace(self):
    # TODO(kathywu): Add test that saves out SavedModel with a custom op when
    # the ">" character is allowed in op names.
    graph_def = graph_pb2.GraphDef()
    text_format.Merge("node { name: 'A' op: 'Test>CustomOp' }",
                      graph_def)
    with self.assertRaisesRegexp(
        ValueError, "Attempted to save ops from non-whitelisted namespaces"):
      save._verify_ops(graph_def, [])
    save._verify_ops(graph_def, ["Test"])

    # Test with multiple carrots in op name.
    text_format.Merge("node { name: 'A' op: 'Test>>A>CustomOp' }",
                      graph_def)
    with self.assertRaisesRegexp(
        ValueError, "Attempted to save ops from non-whitelisted namespaces"):
      save._verify_ops(graph_def, [])
    save._verify_ops(graph_def, ["Test"])

  def test_save_debug_info_enabled(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: math_ops.mul(2., x, name="DEBUG_INFO_OP"),
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(
        root,
        save_dir,
        root.f,
        options=save_options.SaveOptions(save_debug_info=True))
    debug_info_file_name = os.path.join(save_dir, "debug",
                                        "saved_model_debug_info.pb")
    self.assertTrue(os.path.exists(debug_info_file_name))
    debug_info = graph_debug_info_pb2.GraphDebugInfo()
    with open(debug_info_file_name, "rb") as f:
      debug_info.ParseFromString(f.read())

    # Verify that there is a trace for DEBUG_INFO_OP just to ensure that
    # function debug info tracing is nominally functioning.
    found_op = False
    for key in debug_info.traces.keys():
      if key.startswith("DEBUG_INFO_OP@"):
        found_op = True
        break
    self.assertTrue(found_op, "Did not find DEBUG_INFO_OP in trace")

  def test_save_debug_info_disabled(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: math_ops.mul(2., x, name="DEBUG_INFO_OP"),
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(
        root,
        save_dir,
        root.f,
        options=save_options.SaveOptions(save_debug_info=False))
    debug_info_file_name = os.path.join(save_dir, "debug",
                                        "saved_model_debug_info.pb")
    self.assertFalse(os.path.exists(debug_info_file_name))

  def test_function_aliases(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    root.f(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    options = save_options.SaveOptions(function_aliases={
        "my_func": root.f,
    })
    save.save(root, save_dir, root.f, options=options)
    function_cache = list(root.f._stateful_fn._function_cache.all_values())
    function_aliases = loader_impl.parse_saved_model(
        save_dir).meta_graphs[0].meta_info_def.function_aliases
    self.assertLen(function_cache, 1)
    self.assertEqual(function_cache[0].name.decode("utf-8"),
                     list(function_aliases.keys())[0])


class AssetTests(test.TestCase):

  def setUp(self):
    super(AssetTests, self).setUp()
    self._vocab_path = os.path.join(self.get_temp_dir(), "vocab.txt")
    with open(self._vocab_path, "w") as f:
      f.write("alpha\nbeta\ngamma\n")

  def test_asset_path_returned(self):
    root = tracking.AutoTrackable()
    root.path = tracking.Asset(self._vocab_path)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    root.get_asset = def_function.function(lambda: root.path.asset_path)
    save.save(root, save_dir, signatures=root.get_asset.get_concrete_function())
    second_dir = os.path.join(self.get_temp_dir(), "second_dir")
    file_io.rename(save_dir, second_dir)
    imported_path = _import_and_infer(second_dir, {})["output_0"]
    self.assertIn(compat.as_str_any(second_dir),
                  compat.as_str_any(imported_path))

  def test_table(self):
    initializer = lookup_ops.TextFileInitializer(
        self._vocab_path,
        key_dtype=dtypes.string,
        key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
        value_dtype=dtypes.int64,
        value_index=lookup_ops.TextFileIndex.LINE_NUMBER)
    root = util.Checkpoint(table=lookup_ops.HashTable(
        initializer, default_value=-1))
    root.table_user = def_function.function(
        root.table.lookup,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.string)])
    self.assertEqual(
        2,
        self.evaluate(root.table_user(constant_op.constant("gamma"))))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)
    file_io.delete_file(self._vocab_path)
    self.assertAllClose(
        {"output_0": [2, 0]},
        _import_and_infer(save_dir, {"keys": ["gamma", "alpha"]}))
    second_dir = os.path.join(self.get_temp_dir(), "second_dir")
    # Asset paths should track the location the SavedModel is loaded from.
    file_io.rename(save_dir, second_dir)
    self.assertAllClose(
        {"output_0": [2, 1]},
        _import_and_infer(second_dir, {"keys": ["gamma", "beta"]}))

  def test_unused_asset(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    root.asset = tracking.Asset(self._vocab_path)

    export_dir = os.path.join(self.get_temp_dir(), "save_dir")
    save.save(root, export_dir)
    self.assertAllClose(
        {"output_0": [0.2]},
        _import_and_infer(export_dir, {"x": [0.1]}))

  def test_sensible_function_building_exception(self):
    root = util.Checkpoint(v=variables.Variable(2.))
    root.f = def_function.function(
        lambda x: 2. * root.v,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    export_dir = os.path.join(self.get_temp_dir(), "save_dir")
    @def_function.function
    def _calls_save():
      save.save(root, export_dir)
    with self.assertRaisesRegexp(AssertionError, "tf.function"):
      _calls_save()


class _ModelWithOptimizerUsingDefun(util.Checkpoint):

  def __init__(self):
    self.dense = core.Dense(1)
    self.optimizer = adam.Adam(0.01)

  # Using defun due to control flow v2 cycles, b/121159261. def_function uses
  # conds to gate variable initialization and so triggers cond reference cycles,
  # but the thing being wrapped here does not use cond itself.
  @function.defun(
      input_signature=(tensor_spec.TensorSpec([None, 2], dtypes.float32),
                       tensor_spec.TensorSpec([None], dtypes.float32)),
  )
  def call(self, x, y):
    with backprop.GradientTape() as tape:
      loss = math_ops.reduce_mean((self.dense(x) - y) ** 2.)
    trainable_variables = self.dense.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    return {"loss": loss}


class MemoryTests(test.TestCase):

  def setUp(self):
    self._model = _ModelWithOptimizerUsingDefun()

  @test_util.assert_no_garbage_created
  def test_no_reference_cycles(self):
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    self._model.call(x, y)
    if sys.version_info[0] < 3:
      # TODO(allenl): debug reference cycles in Python 2.x
      self.skipTest("This test only works in Python 3+. Reference cycles are "
                    "created in older Python versions.")
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(self._model, save_dir, self._model.call)


class ExportMetaGraphTests(test.TestCase):

  def test_export_meta_graph(self):
    root = tracking.AutoTrackable()
    root.variable = resource_variable_ops.UninitializedVariable(
        name="some_variable", dtype=dtypes.float32)

    @def_function.function(input_signature=[tensor_spec.TensorSpec(None)])
    def multiply_var(x):
      return root.variable * x

    @def_function.function(input_signature=[tensor_spec.TensorSpec([])])
    def update(y):
      root.variable.assign_add(y)
      # TODO(b/150393409): All functions exported as signatures must have at
      # least one output.
      return 0

    @def_function.function(input_signature=[])
    def initialize():
      root.variable.assign(1.0)
      # TODO(b/150393409): All functions exported as signatures must have at
      # least one output.
      return 0

    save_path = os.path.join(self.get_temp_dir(), "meta_graph.pb")
    save.export_meta_graph(
        root,
        save_path,
        signatures={
            "multiply_var": multiply_var,
            "initialize": initialize,
            "update": update
        })

    with ops.Graph().as_default(), session_lib.Session() as session:
      saver.import_meta_graph(save_path)
      meta_graph_def = meta_graph.read_meta_graph_file(save_path)

      # Initialize variable to 1
      _run_signature(session, meta_graph_def, {}, "initialize")
      out = _run_signature(session, meta_graph_def, {"x": 3}, "multiply_var")
      self.assertAllEqual(out, {"output_0": 3})

      # Adds 2 to the variable. Variable is now 3
      _run_signature(session, meta_graph_def, {"y": 2}, "update")
      out = _run_signature(session, meta_graph_def, {"x": 4}, "multiply_var")
      self.assertAllEqual(out, {"output_0": 12})


if __name__ == "__main__":
  test.main()
