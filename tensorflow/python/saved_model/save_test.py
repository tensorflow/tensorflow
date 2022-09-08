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

import os

from absl.testing import parameterized

from google.protobuf import text_format

from tensorflow.core.config import flags
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import graph_debug_info_pb2
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training import saver
from tensorflow.python.util import compat


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
    save_dir,
    inputs,
    signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
  """Import a SavedModel into a TF 1.x-style graph and run `signature_key`."""
  graph = ops.Graph()
  with graph.as_default(), session_lib.Session() as session:
    model = loader.load(session, [tag_constants.SERVING], save_dir)
    return _run_signature(session, model, inputs, signature_key)


class SaveTest(test.TestCase, parameterized.TestCase):

  def test_method_save_signature(self):
    root = autotrackable.AutoTrackable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    root.f(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir, root.f)
    self.assertEqual({"output_0": 2.}, _import_and_infer(save_dir, {"x": 1.}))

  def test_method_save_list_func(self):
    root = autotrackable.AutoTrackable()

    @def_function.function
    def case_fn(x):
      branch_index = constant_op.constant(1)
      branches = [lambda: x, lambda: x + 1]
      case_out = control_flow_ops.switch_case(branch_index, branches)
      return case_out

    root.f = def_function.function(
        lambda x: 2. * case_fn(x),
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    root.f(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir, root.f)
    self.assertEqual({"output_0": 4.}, _import_and_infer(save_dir, {"x": 1.}))

  def test_method_save_concrete(self):
    root = autotrackable.AutoTrackable()
    root.f = def_function.function(lambda z: {"out": 2. * z})
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

  def test_method_save_annotated_function(self):
    # This test is only meaningful with Python 3 because Python 2's
    # inspect.getargspec doesn't save annotations.

    root = autotrackable.AutoTrackable()

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
    with self.assertRaisesRegex(ValueError, "ERROR MSG"):
      save.save(root, os.path.join(self.get_temp_dir(), "saved_model"))

  def test_untracked_variable_useful_message(self):
    root = module.Module()
    v = variables.Variable(1., name="some_unique_name")

    @def_function.function(input_signature=[])
    def f():
      return v.read_value()

    root.f = f
    with self.assertRaisesRegex(
        AssertionError, "Trackable referencing this tensor.*some_unique_name"):
      save.save(root, os.path.join(self.get_temp_dir(), "saved_model"))

  def test_version_information_included(self):
    root = autotrackable.AutoTrackable()
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
    root = autotrackable.AutoTrackable()
    root.f = def_function.function(lambda x: 2. * x)
    root.f(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegex(ValueError, "Expected a TensorFlow function"):
      save.save(root, save_dir, root.f)

  def test_captures_unreachable_variable(self):
    root = autotrackable.AutoTrackable()
    unreachable_variable = variables.Variable([5.0, 2.0])
    root.reachable_variable = variables.Variable([1.0, 3.0])

    @def_function.function
    def increase_variable(x):
      return 2 * unreachable_variable * x + root.reachable_variable

    root.f = increase_variable

    self.assertAllEqual([101.0, 83.0],
                        root.f(constant_op.constant([10.0, 20.0])).numpy())

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")

    with self.assertRaisesRegex(KeyError, "not reachable from root"):
      save.save(root, save_dir)

  def test_nested_inputs(self):
    root = autotrackable.AutoTrackable()
    root.f = def_function.function(
        lambda x: 2. * x[0],
        input_signature=([
            tensor_spec.TensorSpec(None, dtypes.float32),
            tensor_spec.TensorSpec(None, dtypes.float32)
        ],))
    root.f([constant_op.constant(1.), constant_op.constant(1.)])

  def test_nested_outputs(self):
    root = autotrackable.AutoTrackable()
    root.f = def_function.function(lambda x: (2. * x, (3. * x, 4. * x)))
    root.f(constant_op.constant(1.))
    to_save = root.f.get_concrete_function(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegex(ValueError, "non-Tensor value"):
      save.save(root, save_dir, to_save)

  def test_nested_dict_outputs(self):
    root = checkpoint.Checkpoint(
        f=def_function.function(lambda x: {  # pylint: disable=g-long-lambda
            "a": 2. * x,
            "b": (3. * x, 4. * x)
        }))
    root.f(constant_op.constant(1.))
    to_save = root.f.get_concrete_function(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegex(ValueError, "non-Tensor value"):
      save.save(root, save_dir, to_save)

  def test_variable(self):
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    root.f(constant_op.constant(1.))
    to_save = root.f.get_concrete_function(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir, to_save)
    self.assertAllEqual({"output_0": 12.},
                        _import_and_infer(save_dir, {"x": 2.}))

  def test_single_function_default_signature(self):
    model = autotrackable.AutoTrackable()
    model.f = def_function.function(lambda: 3., input_signature=())
    model.f()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(model, save_dir)
    self.assertAllClose({"output_0": 3.}, _import_and_infer(save_dir, {}))

  def test_single_function_no_signature(self):
    model = autotrackable.AutoTrackable()
    model.f = def_function.function(lambda: 3.)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(model, save_dir)

  def test_save_function_no_trace(self):

    class ObjWithFunction(module.Module):

      @def_function.function
      def foo(self, a):
        return a

      @def_function.function
      def bar(self, a):
        return a + 1

    root = ObjWithFunction()
    root.bar(1)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertLogs(level="WARNING") as logs:
      save.save(root, save_dir)

    expected_message = (
        "WARNING:absl:Found untraced functions such as foo while saving "
        "(showing 1 of 1). These functions will not be directly callable after "
        "loading.")
    self.assertIn(expected_message, logs.output)

  def test_find_default_save_function(self):

    class ObjWithDefaultSignature(checkpoint.Checkpoint):

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32)
      ])
      def _default_save_signature(self, x):
        return x + x + 1

    obj = ObjWithDefaultSignature()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(obj, save_dir)
    self.assertAllClose({"output_0": 7.},
                        _import_and_infer(save_dir, {"x": 3.}))

  def test_docstring(self):

    class Adder(module.Module):

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32)
      ])
      def add(self, x):
        return x + x + 1.

    to_save = Adder()
    to_save.add(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(to_save, save_dir)
    self.assertAllClose({"output_0": 7.},
                        _import_and_infer(save_dir, {"x": 3.}))

  def test_datastructures(self):

    class HasDatastructures(checkpoint.Checkpoint):

      def __init__(self):
        self.a = [1.]
        self.a.append(variables.Variable(2.))
        self.b = {"a": variables.Variable(3.)}

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32)
      ])
      def add(self, x):
        return x + math_ops.add_n(self.a) + self.b["a"]

    to_save = HasDatastructures()
    to_save.add(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(to_save, save_dir)
    self.assertAllClose({"output_0": 10.},
                        _import_and_infer(save_dir, {"x": 4.}))

  def test_default_attr_stripping(self):

    class Complex(checkpoint.Checkpoint):

      @def_function.function(input_signature=[])
      def __call__(self):
        return math_ops.complex(
            constant_op.constant(1.), constant_op.constant(2.), name="complex")

    to_save = Complex()
    to_save()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(to_save, save_dir)
    graph = ops.Graph()
    with graph.as_default(), self.session(graph) as session:
      loader.load(session, [tag_constants.SERVING], save_dir)
      func, = [f for name, f in graph._functions.items() if "call" in name]
      complex_node, = [
          node for node in func.definition.node_def if node.op == "Complex"
      ]
      self.assertNotIn("T", complex_node.attr)
      self.assertNotIn("Tout", complex_node.attr)

  def test_signature_attribute_reserved(self):
    root = checkpoint.Checkpoint(signatures=variables.Variable(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegex(ValueError, "del obj.signatures"):
      save.save(root, save_dir)
    del root.signatures
    save.save(root, save_dir)

  def test_function_with_captured_dataset(self):
    if test_util.is_gpu_available():
      self.skipTest("Currently broken when a GPU is available.")

    class HasDataset(module.Module):

      def __init__(self):
        super(HasDataset, self).__init__()
        self.dataset = (dataset_ops.Dataset.range(5).map(lambda x: x**2))

      @def_function.function
      def __call__(self, x):
        current_sum = array_ops.zeros([], dtype=dtypes.int64)
        for element in self.dataset:
          current_sum += x * element
        return current_sum

    root = HasDataset()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(
        root,
        save_dir,
        signatures=root.__call__.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.int64)))
    self.assertAllClose({"output_0": 3 * (1 + 4 + 9 + 16)},
                        _import_and_infer(save_dir, {"x": 3}))

  def test_variable_args_cannot_be_used_as_signature(self):

    with self.assertRaises(TypeError):
      @def_function.function(input_signature=[
          resource_variable_ops.VariableSpec(shape=[], dtype=dtypes.int32)
      ])
      def f(unused_v):
        return 1

  def test_export_correct_output_shapes(self):
    """Asserts that nodes are exported with the correct number of output shapes.

    After backpropagation rewrite, functions are rewritten with additional
    outputs. When exporting to SavedModel, the shapes of the additional outputs
    were incorrectly added to the FunctionDef proto (b/133666530).
    """
    obj = autotrackable.AutoTrackable()
    obj.v = variables.Variable(2.)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    def f(x):
      return (math_ops.multiply(obj.v, x), math_ops.multiply(obj.v,
                                                             (x + 1)), None)

    obj.f = f

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
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
      if (f.signature.name.startswith("__inference_f") or
          f.signature.name.startswith("__inference_g")):
        for node in f.node_def:
          assert_correct_number_of_output_shapes(node)

  def test_save_cached_variable(self):
    with ops.Graph().as_default(), session_lib.Session() as session:
      obj = autotrackable.AutoTrackable()
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

  @parameterized.named_parameters(
      ("_SaveDevices_ExportMetaGraph",
       save_options.VariablePolicy.SAVE_VARIABLE_DEVICES, True),
      ("_DiscardDevices_ExportMetaGraph", save_options.VariablePolicy.NONE,
       True), ("_SaveDevices_Save",
               save_options.VariablePolicy.SAVE_VARIABLE_DEVICES, False),
      ("_DiscardDevices_Save", save_options.VariablePolicy.NONE, False))
  def test_save_variable_devices(self, save_devices, meta_graph_only):
    context._reset_context()
    cpus = context.context().list_physical_devices("CPU")
    if len(cpus) == 1:
      context.context().set_logical_device_configuration(
          cpus[0], [
              context.LogicalDeviceConfiguration(),
              context.LogicalDeviceConfiguration()
          ])
    context.ensure_initialized()

    root = autotrackable.AutoTrackable()
    with ops.device("CPU:0"):
      root.v0 = variables.Variable(1., name="v0")
    with ops.device("CPU:1"):
      root.v1 = variables.Variable(1., name="v1")

    options = save_options.SaveOptions(
        experimental_variable_policy=save_devices)
    file_name = os.path.join(self.get_temp_dir(), "saved_model")
    if meta_graph_only:
      save.export_meta_graph(obj=root, filename=file_name, options=options)
    else:
      save.save(obj=root, export_dir=file_name, options=options)

    meta = None
    if meta_graph_only:
      meta = meta_graph.read_meta_graph_file(file_name)
    else:
      meta = loader_impl.parse_saved_model(file_name).meta_graphs[0]

    # Check devices in meta graph nodes.
    graph_def = meta.graph_def
    v0 = next((n for n in graph_def.node if n.name == "v0"), None)
    v1 = next((n for n in graph_def.node if n.name == "v1"), None)
    self.assertIsNotNone(v0)
    self.assertIsNotNone(v1)
    if save_devices == save_options.VariablePolicy.SAVE_VARIABLE_DEVICES:
      self.assertIn("CPU:0", v0.device)
      self.assertIn("CPU:1", v1.device)
    else:
      self.assertEmpty(v0.device)
      self.assertEmpty(v1.device)

    # Check devices in object graph nodes.
    object_graph_def = meta.object_graph_def
    v0 = next((n.variable
               for n in object_graph_def.nodes
               if n.HasField("variable") and n.variable.name == "v0"), None)
    v1 = next((n.variable
               for n in object_graph_def.nodes
               if n.HasField("variable") and n.variable.name == "v1"), None)
    self.assertIsNotNone(v0)
    self.assertIsNotNone(v1)
    if save_devices == save_options.VariablePolicy.SAVE_VARIABLE_DEVICES:
      self.assertIn("CPU:0", v0.device)
      self.assertIn("CPU:1", v1.device)
    else:
      self.assertEmpty(v0.device)
      self.assertEmpty(v1.device)

  @parameterized.named_parameters(
      ("_ExpandDistributedVariablesWithPolicy",
       save_options.VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES, True),
      ("_ExpandDistributedVariablesWithoutPolicy",
       save_options.VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES, False),
      ("_DiscardDistributedVariablesWithPolicy",
       save_options.VariablePolicy.NONE, True),
      ("_DiscardDistributedVariablesWithoutPolicy",
       save_options.VariablePolicy.NONE, False))
  def test_expand_distributed_variables(self, expand_strategy, policy):
    # 1. Create a context with both CPU:0 and CPU:1.
    context._reset_context()
    cpus = context.context().list_physical_devices("CPU")
    if len(cpus) == 1:
      context.context().set_logical_device_configuration(
          cpus[0], [
              context.LogicalDeviceConfiguration(),
              context.LogicalDeviceConfiguration()
          ])
    context.ensure_initialized()

    # 2. Create and save a model under a mirrored strategy.
    file_name = os.path.join(self.get_temp_dir(), "saved_model.pb")
    strategy = mirrored_strategy.MirroredStrategy(["CPU:0", "CPU:1"])
    strategy.extended._use_var_policy = policy
    with strategy.scope():
      root = autotrackable.AutoTrackable()
      root.v = variables.Variable([1., 1.], name="v")

      @def_function.function(input_signature=[])
      def f():
        root.v.assign([2., 2.])

      root.f = f

      save.export_meta_graph(
          obj=root,
          filename=file_name,
          options=save_options.SaveOptions(
              experimental_variable_policy=expand_strategy))

    # 3. Read the output file and test behavior.
    meta_graph_def = meta_graph.read_meta_graph_file(file_name)
    object_graph = meta_graph_def.object_graph_def
    graph_def = meta_graph_def.graph_def
    v = next((n.variable
              for n in object_graph.nodes
              if n.HasField("variable") and n.variable.name == "v"), None)
    saved_function = next((f for f in graph_def.library.function
                           if "inference_f_" in f.signature.name), None)
    self.assertIsNotNone(saved_function)
    if (expand_strategy ==
        save_options.VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES):
      # experimental_save_variable_devices should have been automatically set.
      self.assertIn("CPU:0", v.device)
      components = v.experimental_distributed_variable_components
      self.assertLen(components, 2)
      v0 = next((x for x in components if x.name == "v"), None)
      v1 = next((x for x in components if x.name == "v/replica_1"), None)
      self.assertIsNotNone(v0)
      self.assertIsNotNone(v1)
      self.assertIn("CPU:0", v0.device)
      self.assertIn("CPU:1", v1.device)
      self.assertLen(saved_function.signature.input_arg, 2)
    else:
      self.assertEmpty(v.device)
      self.assertEmpty(v.experimental_distributed_variable_components)
      self.assertLen(saved_function.signature.input_arg, 1)

  def test_save_uninitialized_variable(self):
    root = autotrackable.AutoTrackable()
    root.uninitialized_variable = resource_variable_ops.UninitializedVariable(
        name="uninitialized_variable", dtype=dtypes.float32)
    root.initialized_variable = variables.Variable(
        1.0, name="initialized_variable")

    # TODO(b/149594077): Python loading does not work now partly because it
    # shouldn't, as the public API and semantics of uninitialized variables
    # are not properly defined, and officially supporting loading would end up
    # defining semantics "by usage." We should only allow loading once the API
    # is made official.
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, export_dir)
    with self.assertRaisesRegex(FileNotFoundError,
                                "Key uninitialized_variable"):
      load.load(export_dir)
    with ops.Graph().as_default(), session_lib.Session() as session:
      # The final ValueError here (with "no variables to save") is confusing,
      # but errors upstream give the user the correct information (a
      # NotFoundError stating that the uninitalized_variable was not found in
      # the checkpoint).
      with self.assertRaises(ValueError):
        loader.load(session, [tag_constants.SERVING], export_dir)

  def test_concrete_function_with_set_shape(self,):
    # Serialized concrete function should retain the shape from the TensorSpec,
    # instead of using the shape of the inputs (which are changed by set_shape).
    @def_function.function
    def f(x):
      x.set_shape((5, 1))
      return x

    root = autotrackable.AutoTrackable()
    path = os.path.join(self.get_temp_dir(), "saved_model")
    concrete = f.get_concrete_function(
        tensor_spec.TensorSpec((None, 1), name="name"))
    save.save(root, path, signatures={"key": concrete})
    imported = load.load(path)
    self.assertEqual(imported.signatures["key"].structured_input_signature[1],
                     {"name": tensor_spec.TensorSpec((None, 1), name="name")})

  def test_save_composite_tensor_signature(self):
    @def_function.function(
        input_signature=[ragged_tensor.RaggedTensorSpec(ragged_rank=2)])
    def f(x):
      return {"output_key": x}
    root = autotrackable.AutoTrackable()
    path = os.path.join(self.get_temp_dir(), "saved_model")
    inp = ragged_factory_ops.constant([[[1.0, 2.0], [3.0]], [[5.]]])
    flat_inp = {
        "x": constant_op.constant([1., 2., 3., 5]),
        "x_1": constant_op.constant([0, 2, 3], dtype=dtypes.int64),
        "x_2": constant_op.constant([0, 2, 3, 4], dtype=dtypes.int64)
    }
    save.save(root, path, signatures={"key": f.get_concrete_function()})

    # Test that the ragged signature can be loaded back into Python with V2 APIs
    imported = load.load(path)
    self.assertAllEqual(inp,
                        imported.signatures["key"](**flat_inp)["output_key"])
    graph = ops.Graph()

    # Try running the signature with V1 APIs.
    with graph.as_default(), session_lib.Session() as session:
      meta_graph_def = loader.load(session, [tag_constants.SERVING], path)
      signature = meta_graph_def.signature_def["key"]

      feed_dict = {}
      for arg_name in flat_inp:
        input_tensor = session.graph.get_tensor_by_name(
            signature.inputs[arg_name].name)
        feed_dict[input_tensor] = flat_inp[arg_name].numpy()

      # Get composite tensor components
      output_components = (
          signature.outputs["output_key"].composite_tensor.components)
      fetches = {}
      components_keys = ["x", "x_1", "x_2"]
      for k, output_tensor_info in zip(components_keys, output_components):
        fetches[k] = session.graph.get_tensor_by_name(output_tensor_info.name)

      outputs = session.run(fetches, feed_dict)

    self.assertAllClose(flat_inp, outputs)

  def test_save_uses_sanitized_signature_name(self):

    @def_function.function(
        input_signature=[ragged_tensor.RaggedTensorSpec(ragged_rank=2)])
    def f(x):
      return {"output_key": x}

    # Colons are not usable as name scopes.
    unsanitized_name = "foo:bar"
    root = autotrackable.AutoTrackable()
    path = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(
        root, path, signatures={unsanitized_name: f.get_concrete_function()})
    graph = ops.Graph()
    with graph.as_default(), session_lib.Session() as session:
      meta_graph_def = loader.load(session, [tag_constants.SERVING], path)
      signature = meta_graph_def.signature_def[unsanitized_name]
      tensor_names = [
          session.graph.get_tensor_by_name(signature.inputs[key].name).name
          for key in signature.inputs
      ]
      # The placeholder names will have the sanitized version.
      self.assertCountEqual(tensor_names,
                            ["foo_bar_x:0", "foo_bar_x_1:0", "foo_bar_x_2:0"])

  def test_save_returns_none(self):
    # Test that `tf.saved_model.save` API returns None to user.
    root = autotrackable.AutoTrackable()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    result = save.save(root, save_dir)
    self.assertIsNone(result)


class DependencyTest(test.TestCase):
  """Tests for deserialization dependencies (saving-related only)."""

  def test_validate_dependencies(self):

    class Valid(autotrackable.AutoTrackable):

      def _deserialization_dependencies(self, children):
        return children

    root = Valid()
    root.f = variables.Variable(1.0)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)

  def test_validate_dependencies_error_untracked(self):
    untracked = variables.Variable(1.0)

    class Invalid(autotrackable.AutoTrackable):

      def _deserialization_dependencies(self, children):
        del children  # Unused.
        return {"untracked": untracked}
    invalid_deps = Invalid()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegex(ValueError, "Found an untracked dependency"):
      save.save(invalid_deps, save_dir)

  def test_validate_dependencies_error_cyclic(self):

    class Invalid(autotrackable.AutoTrackable):

      def __init__(self):
        self.cycle_ref = None

      def _deserialization_dependencies(self, children):
        del children  # Unused.
        return {"cycle_ref": self.cycle_ref}
    cycle1 = Invalid()
    cycle2 = Invalid()
    cycle1.cycle_ref = cycle2
    cycle2.cycle_ref = cycle1
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegex(ValueError,
                                "dependency cycle in the saved Trackable"):
      save.save(cycle1, save_dir)


class VariablePolicyEnumTest(test.TestCase):

  def testFromObj(self):
    self.assertEqual(save_options.VariablePolicy.NONE,
                     save_options.VariablePolicy.from_obj(None))
    self.assertEqual(
        save_options.VariablePolicy.SAVE_VARIABLE_DEVICES,
        save_options.VariablePolicy.from_obj(
            save_options.VariablePolicy.SAVE_VARIABLE_DEVICES))
    self.assertEqual(
        save_options.VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES,
        save_options.VariablePolicy.from_obj(
            save_options.VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES))
    self.assertEqual(
        save_options.VariablePolicy.SAVE_VARIABLE_DEVICES,
        save_options.VariablePolicy.from_obj("save_variable_devices"))
    self.assertEqual(
        save_options.VariablePolicy.SAVE_VARIABLE_DEVICES,
        save_options.VariablePolicy.from_obj("SaVe_VaRiAbLe_DeViCeS"))
    self.assertEqual(
        save_options.VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES,
        save_options.VariablePolicy.from_obj("expand_distributed_variables"))
    self.assertEqual(
        save_options.VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES,
        save_options.VariablePolicy.from_obj("eXpAnD_dIsTrIbUtEd_VaRiAbLeS"))
    for invalid in ["not_a_valid_value", 2.0, []]:
      with self.assertRaisesRegex(ValueError, "invalid VariablePolicy value"):
        save_options.VariablePolicy.from_obj(invalid)

  def testNamingConvention(self):
    """Enforces names are uppercase versions of values."""
    for policy in save_options.VariablePolicy:
      if policy == save_options.VariablePolicy.NONE:
        self.assertIsNone(policy.value)
      else:
        self.assertEqual(policy.name, policy.name.upper())
        self.assertEqual(policy.value, policy.value.lower())
        self.assertEqual(policy.name, policy.value.upper())


class SavingOptionsTest(test.TestCase):

  def testOpNameSpace(self):
    # TODO(kathywu): Add test that saves out SavedModel with a custom op when
    # the ">" character is allowed in op names.
    graph_def = graph_pb2.GraphDef()
    text_format.Parse("node { name: 'A' op: 'Test>CustomOp' }", graph_def)
    with self.assertRaisesRegex(
        ValueError, "Attempted to save ops from non-whitelisted namespaces"):
      save._verify_ops(graph_def, [])
    save._verify_ops(graph_def, ["Test"])

    # Test with multiple carrots in op name.
    text_format.Parse("node { name: 'A' op: 'Test>>A>CustomOp' }", graph_def)
    with self.assertRaisesRegex(
        ValueError, "Attempted to save ops from non-whitelisted namespaces"):
      save._verify_ops(graph_def, [])
    save._verify_ops(graph_def, ["Test"])

  def test_save_custom_op_with_no_whitelist_specified(self):
    # Test that we are able to save a model that contains a custom op with a
    # custom namespace when the user has not explicitly specified a namespace
    # whitelist (i.e. that we default to allowing all custom ops when saving
    # and no whitelist is specified, rather than throwing an exception).
    graph_def = graph_pb2.GraphDef()
    text_format.Parse("node { name: 'A' op: 'Test>CustomOp' }", graph_def)
    save._verify_ops(graph_def, namespace_whitelist=None)

    # If the user passes an empty list for the namespace whitelist rather than
    # nothing, we should then throw an exception if a custom op is used.
    with self.assertRaisesRegex(
        ValueError, "Attempted to save ops from non-whitelisted namespaces"):
      save._verify_ops(graph_def, [])

  def test_save_debug_info_enabled(self):
    root = autotrackable.AutoTrackable()
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
    root = autotrackable.AutoTrackable()
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
    root = autotrackable.AutoTrackable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    options = save_options.SaveOptions(function_aliases={
        "my_func": root.f,
    })
    save.save(root, save_dir, root.f, options=options)
    function_cache = root.f._stateful_fn._list_all_concrete_functions()
    function_aliases = loader_impl.parse_saved_model(
        save_dir).meta_graphs[0].meta_info_def.function_aliases
    self.assertLen(function_cache, 1)
    self.assertEqual(function_cache[0].name.decode("utf-8"),
                     list(function_aliases.keys())[0])

  def test_accepts_io_device(self):
    options = save_options.SaveOptions()
    self.assertIsNone(options.experimental_io_device)
    options = save_options.SaveOptions(experimental_io_device="/job:localhost")
    self.assertEqual("/job:localhost", options.experimental_io_device)

  def test_accepts_variable_policy(self):
    options = save_options.SaveOptions()
    self.assertEqual(save_options.VariablePolicy.NONE,
                     options.experimental_variable_policy)
    # VariablePolicy instances.
    options = save_options.SaveOptions(experimental_variable_policy=save_options
                                       .VariablePolicy.SAVE_VARIABLE_DEVICES)
    self.assertEqual(save_options.VariablePolicy.SAVE_VARIABLE_DEVICES,
                     options.experimental_variable_policy)
    options = save_options.SaveOptions(
        experimental_variable_policy=save_options.VariablePolicy
        .EXPAND_DISTRIBUTED_VARIABLES)
    self.assertEqual(save_options.VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES,
                     options.experimental_variable_policy)
    # String conversions.
    options = save_options.SaveOptions(
        experimental_variable_policy="save_variable_devices")
    self.assertEqual(save_options.VariablePolicy.SAVE_VARIABLE_DEVICES,
                     options.experimental_variable_policy)
    options = save_options.SaveOptions(
        experimental_variable_policy="expand_distributed_variables")
    self.assertEqual(save_options.VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES,
                     options.experimental_variable_policy)
    with self.assertRaisesRegex(ValueError, "invalid VariablePolicy value"):
      options = save_options.SaveOptions(
          experimental_variable_policy="not_a_valid_value")


class AssetTests(test.TestCase):

  def setUp(self):
    super(AssetTests, self).setUp()
    self._vocab_path = os.path.join(self.get_temp_dir(), "vocab.txt")
    with open(self._vocab_path, "w") as f:
      f.write("alpha\nbeta\ngamma\n")

  def test_asset_path_returned(self):
    root = autotrackable.AutoTrackable()
    root.path = asset.Asset(self._vocab_path)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    root.get_asset = def_function.function(lambda: root.path.asset_path)
    save.save(root, save_dir, signatures=root.get_asset.get_concrete_function())
    second_dir = os.path.join(self.get_temp_dir(), "second_dir")
    file_io.rename(save_dir, second_dir)
    imported_path = _import_and_infer(second_dir, {})["output_0"]
    self.assertIn(
        compat.as_str_any(second_dir), compat.as_str_any(imported_path))

  def test_table(self):
    initializer = lookup_ops.TextFileInitializer(
        self._vocab_path,
        key_dtype=dtypes.string,
        key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
        value_dtype=dtypes.int64,
        value_index=lookup_ops.TextFileIndex.LINE_NUMBER)
    root = checkpoint.Checkpoint(
        table=lookup_ops.HashTable(initializer, default_value=-1))
    root.table_user = def_function.function(
        root.table.lookup,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.string)])
    self.assertEqual(
        2, self.evaluate(root.table_user(constant_op.constant("gamma"))))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)
    file_io.delete_file(self._vocab_path)
    self.assertAllClose({"output_0": [2, 0]},
                        _import_and_infer(save_dir,
                                          {"keys": ["gamma", "alpha"]}))
    second_dir = os.path.join(self.get_temp_dir(), "second_dir")
    # Asset paths should track the location the SavedModel is loaded from.
    file_io.rename(save_dir, second_dir)
    self.assertAllClose({"output_0": [2, 1]},
                        _import_and_infer(second_dir,
                                          {"keys": ["gamma", "beta"]}))

  def test_untracked_table_useful_message(self):
    root = module.Module()
    initializer = lookup_ops.TextFileInitializer(
        self._vocab_path,
        key_dtype=dtypes.string,
        key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
        value_dtype=dtypes.int64,
        value_index=lookup_ops.TextFileIndex.LINE_NUMBER)
    table = lookup_ops.HashTable(initializer, default_value=-1)
    root.table_user = def_function.function(
        table.lookup,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.string)])
    root.table_user(constant_op.constant("gamma"))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(AssertionError, "HashTable"):
      save.save(root, save_dir)

  def test_unused_asset(self):
    root = autotrackable.AutoTrackable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    root.asset = asset.Asset(self._vocab_path)

    export_dir = os.path.join(self.get_temp_dir(), "save_dir")
    save.save(root, export_dir)
    self.assertAllClose({"output_0": [0.2]},
                        _import_and_infer(export_dir, {"x": [0.1]}))

  def test_sensible_function_building_exception(self):
    root = checkpoint.Checkpoint(v=variables.Variable(2.))
    root.f = def_function.function(
        lambda x: 2. * root.v,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    export_dir = os.path.join(self.get_temp_dir(), "save_dir")

    @def_function.function
    def _calls_save():
      save.save(root, export_dir)

    with self.assertRaisesRegex(AssertionError, "tf.function"):
      _calls_save()


class ExportMetaGraphTests(test.TestCase):

  def test_export_meta_graph(self):
    root = autotrackable.AutoTrackable()
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


class FingerprintingTests(test.TestCase):

  def test_toggle_flag(self):
    self.assertFalse(flags.config().saved_model_fingerprinting.value())
    flags.config().saved_model_fingerprinting.reset(True)
    self.assertTrue(flags.config().saved_model_fingerprinting.value())


if __name__ == "__main__":
  test.main()
