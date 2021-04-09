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
"""Tests for trackable object SavedModel loading."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import functools
import gc
import io
import os
import sys
import tempfile
import weakref

from absl.testing import parameterized
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function as framework_function
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import load_options
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import monitored_session
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util
from tensorflow.python.util import tf_inspect


def cycle(obj, cycles, signatures=None):
  to_save = obj
  # TODO(vbardiovsky): It would be nice if exported protos reached a fixed
  # point w.r.t. saving/restoring, ideally after 2nd saving.
  for _ in range(cycles):
    path = tempfile.mkdtemp(prefix=test.get_temp_dir())
    # If available, we'll run the save and restore preferring the GPU. This
    # just makes sure we aren't throwing errors and have enough
    # device("CPU") blocks to satisfy the placer.
    with test_util.use_gpu():
      save.save(to_save, path, signatures)
      loaded = load.load(path)
      signatures = loaded.signatures
    to_save = loaded
  return loaded


@parameterized.named_parameters(
    dict(testcase_name="ReloadOnce", cycles=1),
    dict(testcase_name="ReloadTwice", cycles=2),
    dict(testcase_name="ReloadThrice", cycles=3)
)
class LoadTest(test.TestCase, parameterized.TestCase):

  def test_structure_import(self, cycles):
    root = tracking.AutoTrackable()
    root.dep_one = tracking.AutoTrackable()
    root.dep_two = tracking.AutoTrackable()
    root.dep_two.dep = tracking.AutoTrackable()
    root.dep_three = root.dep_two.dep
    imported = cycle(root, cycles)
    self.assertIs(imported.dep_three, imported.dep_two.dep)
    self.assertIsNot(imported.dep_one, imported.dep_two)

  @test_util.run_in_graph_and_eager_modes
  def test_variables(self, cycles):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(1., trainable=True)
    root.v2 = variables.Variable(2., trainable=False)
    self.evaluate([root.v1.initializer, root.v2.initializer])

    for _ in range(cycles):
      imported = cycle(root, 1)
      self.evaluate([imported.v1.initializer, imported.v2.initializer])

    if not context.executing_eagerly():
      self.assertIsInstance(imported.v1.initializer, ops.Operation)
      self.assertIsInstance(imported.v2.initializer, ops.Operation)

    self.assertEqual(self.evaluate(imported.v1), 1.0)
    self.assertTrue(imported.v1.trainable)
    self.assertEqual(self.evaluate(imported.v2), 2.0)
    self.assertFalse(imported.v2.trainable)

  def test_variables_name(self, cycles):
    root = tracking.AutoTrackable()
    # Test 2 variables with same name: should work as the checkpoint
    # is based on object name and not on variable name.
    root.v1 = variables.Variable(1., trainable=True, name="v1")
    root.v2 = variables.Variable(2., trainable=False, name="v1")
    imported = cycle(root, cycles)
    self.assertEqual(imported.v1.numpy(), 1.0)
    self.assertEqual(imported.v2.numpy(), 2.0)
    self.assertEqual(imported.v1.name, root.v1.name)
    self.assertEqual(imported.v2.name, root.v2.name)
    with variable_scope.variable_scope("foo"):
      imported = cycle(root, cycles)
      self.assertTrue(imported.v1.name.startswith("foo/"))
      self.assertTrue(imported.v2.name.startswith("foo/"))

  def test_partially_defined_variable_shape(self, cycles):

    class MakeVariable(module.Module):

      def __init__(self):
        self.v = None

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec([None], dtypes.int64)])
      def make_variable(self, initial_value):
        if self.v is None:
          self.v = variables.Variable(initial_value)

    m = MakeVariable()
    m.make_variable([1, 2, 3])
    m = cycle(m, cycles)
    m.v.assign([1, 2, 3, 4])
    self.assertEqual([None], tensor_shape.as_shape(m.v.shape).as_list())

  @test_util.run_in_graph_and_eager_modes
  def test_capture_variables(self, cycles):
    root = tracking.AutoTrackable()
    root.weights = variables.Variable(2.)
    self.evaluate(root.weights.initializer)
    root.f = def_function.function(
        lambda x: root.weights * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    for _ in range(cycles):
      imported = cycle(root, 1)
      self.evaluate(imported.weights.initializer)
    self.assertEqual(4., self.evaluate(imported.f(constant_op.constant(2.))))
    self.evaluate(imported.weights.assign(4.0))
    self.assertEqual(8., self.evaluate(imported.f(constant_op.constant(2.))))

  @test_util.run_in_graph_and_eager_modes
  def test_capture_constant(self, cycles):
    root = tracking.AutoTrackable()
    captured_constant = constant_op.constant(2.)
    root.f = def_function.function(
        lambda x: captured_constant * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    imported = cycle(root, cycles)
    self.assertEqual(4., self.evaluate(imported.f(constant_op.constant(2.))))

  def test_control_outputs(self, cycles):
    exported = tracking.AutoTrackable()
    exported.v = variables.Variable(1.)
    exported.f = def_function.function(
        lambda: exported.v.assign(2., name="should_be_control_output"))
    exported_graph = exported.f.get_concrete_function().graph
    self.assertIn(
        exported_graph.get_operation_by_name("should_be_control_output"),
        exported_graph.control_outputs)

    imported = cycle(exported, cycles)
    # Calling get_concrete_function wraps in a second call operation; we want to
    # inspect the original function body for the control output; digging into
    # graph.as_graph_def() and its FunctionDefLibrary is another option.
    imported_concrete, = imported.f.concrete_functions
    imported_graph = imported_concrete.graph
    self.assertIn(
        imported_graph.get_operation_by_name("should_be_control_output"),
        imported_graph.control_outputs)

  def _make_asset(self, contents):
    filename = tempfile.mktemp(prefix=self.get_temp_dir())
    with open(filename, "w") as f:
      f.write(contents)
    return filename

  @test_util.run_in_graph_and_eager_modes
  def test_assets(self, cycles):
    file1 = self._make_asset("contents 1")
    file2 = self._make_asset("contents 2")

    root = tracking.AutoTrackable()
    root.asset1 = tracking.Asset(file1)
    root.asset2 = tracking.Asset(file2)

    save_dir = os.path.join(self.get_temp_dir(), "save_dir")
    save.save(root, save_dir)

    file_io.delete_file(file1)
    file_io.delete_file(file2)
    load_dir = os.path.join(self.get_temp_dir(), "load_dir")
    file_io.rename(save_dir, load_dir)

    imported = load.load(load_dir)
    with open(self.evaluate(imported.asset1.asset_path), "r") as f:
      self.assertEqual("contents 1", f.read())
    with open(self.evaluate(imported.asset2.asset_path), "r") as f:
      self.assertEqual("contents 2", f.read())

  def test_cond_prune(self, cycles):
    x_in = []
    x_out = []

    def f(x, y):
      x_in.append(x)
      xx = cond_v2.cond_v2(
          math_ops.less(1, 2),
          lambda: x + 1,
          lambda: x + 2,
      )
      x_out.append(xx)
      return xx, 2 * y

    f_wrapped = wrap_function.wrap_function(
        f, [tensor_spec.TensorSpec((), dtypes.float32)] * 2)
    f_pruned = f_wrapped.prune(x_in[0], [x_out[0]])

    class Adder(module.Module):

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32)])
      def add(self, x):
        return f_pruned(x)

    root = Adder()
    root.add(constant_op.constant(1.))
    root = cycle(root, cycles)
    root.add(constant_op.constant(1.))

  def test_capture_assets(self, cycles):
    root = tracking.AutoTrackable()
    root.vocab = tracking.Asset(self._make_asset("contents"))
    root.f = def_function.function(
        lambda: root.vocab.asset_path,
        input_signature=[])
    imported = cycle(root, cycles)
    original_output = root.f().numpy()
    imported_output = imported.f().numpy()
    self.assertNotEqual(original_output, imported_output)
    with open(imported_output, "r") as f:
      self.assertEqual("contents", f.read())

  def test_capture_assets_in_graph(self, cycles):
    root = tracking.AutoTrackable()
    root.vocab = tracking.Asset(self._make_asset("contents"))
    root.f = def_function.function(
        lambda: root.vocab.asset_path,
        input_signature=[])

    original_output = root.f().numpy()

    if cycles > 1:
      root = cycle(root, cycles - 1)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)

    with ops.Graph().as_default():
      imported = load.load(path)
      imported_tensor = imported.f()
      with monitored_session.MonitoredSession() as sess:
        imported_output = sess.run(imported_tensor)
        self.assertLen(ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS), 1)
        self.assertNotEqual(original_output, imported_output)
        with open(imported_output, "r") as f:
          self.assertEqual("contents", f.read())

  def test_dedup_assets(self, cycles):
    vocab = self._make_asset("contents")
    root = tracking.AutoTrackable()
    root.asset1 = tracking.Asset(vocab)
    root.asset2 = tracking.Asset(vocab)
    imported = cycle(root, cycles)
    self.assertEqual(imported.asset1.asset_path.numpy(),
                     imported.asset2.asset_path.numpy())

  def test_implicit_input_signature(self, cycles):
    @def_function.function
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func

    # Add two traces.
    root.f(constant_op.constant(1.))
    root.f(constant_op.constant(1))

    imported = cycle(root, cycles)

    self.assertEqual(4., imported.f(constant_op.constant(2.)).numpy())
    self.assertEqual(14, imported.f(constant_op.constant(7)).numpy())

  def test_explicit_input_signature(self, cycles):
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func

    imported = cycle(root, cycles)
    self.assertEqual(4., imported.f(constant_op.constant(2.0)).numpy())

  def test_explicit_save_signature(self, cycles):
    @def_function.function
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func

    imported = cycle(
        root, cycles, {
            "f":
                root.f.get_concrete_function(
                    tensor_spec.TensorSpec(None, dtypes.float32))
        })
    self.assertEqual(4., imported.f(constant_op.constant(2.0)).numpy())

  def test_nested_functions(self, cycles):
    f = def_function.function(
        lambda x: x*2.0,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    g = def_function.function(
        lambda x: f(x) + 1.0,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])

    root = tracking.AutoTrackable()
    root.g = g
    imported = cycle(root, cycles)
    imported.g(constant_op.constant([1.0]))

  def test_function_with_default_bool_input(self, cycles):

    def func(x, training=False):
      if training:
        return 2 * x
      else:
        return 7

    root = tracking.AutoTrackable()
    root.f = def_function.function(func)

    self.assertEqual(20, root.f(constant_op.constant(10), True).numpy())
    self.assertEqual(7, root.f(constant_op.constant(1)).numpy())
    self.assertEqual(2, root.f(constant_op.constant(1), True).numpy())

    imported = cycle(root, cycles)

    self.assertEqual(4, imported.f(constant_op.constant(2), True).numpy())
    self.assertEqual(7, imported.f(constant_op.constant(2)).numpy())

  def test_function_with_default_none_input(self, cycles):

    def func(x, dtype=None):
      if dtype:
        return array_ops.zeros(shape=x.shape, dtype=dtype)
      else:
        return array_ops.zeros(shape=x.shape, dtype=dtypes.float32)

    root = tracking.AutoTrackable()
    root.f = def_function.function(func)

    self.assertAllEqual([0.0, 0.0, 0.0],
                        root.f(constant_op.constant([1, 2, 3])).numpy())
    self.assertAllEqual([0.0, 0.0, 0.0],
                        root.f(constant_op.constant([1.0, 2.0, 3.0])).numpy())
    self.assertAllEqual([0.0, 0.0, 0.0, 0.0],
                        root.f(constant_op.constant([1, 2, 3, 4])).numpy())
    self.assertAllEqual([0, 0, 0],
                        root.f(
                            constant_op.constant([1.0, 2.0, 3.0]),
                            dtype=dtypes.int32).numpy())

    concrete_functions = root.f._list_all_concrete_functions_for_serialization()  # pylint: disable=protected-access
    self.assertLen(concrete_functions, 4)

    imported = cycle(root, cycles)

    self.assertAllEqual([0.0, 0.0, 0.0],
                        imported.f(constant_op.constant([1, 2, 3]),
                                   None).numpy())
    self.assertAllEqual([0.0, 0.0, 0.0],
                        imported.f(constant_op.constant([1.0, 2.0,
                                                         3.0])).numpy())
    self.assertAllEqual([0.0, 0.0, 0.0, 0.0],
                        imported.f(constant_op.constant([1, 2, 3, 4])).numpy())
    self.assertAllEqual([0, 0, 0],
                        imported.f(
                            constant_op.constant([1.0, 2.0, 3.0]),
                            dtype=dtypes.int32).numpy())

  def test_function_with_str_bytes_input(self, cycles):

    @def_function.function
    def func(x, y):
      return string_ops.string_join([x, y])

    root = tracking.AutoTrackable()
    root.f = func

    self.assertAllEqual(b"ab", root.f("a", "b"))
    self.assertAllEqual(b"ab", root.f("a", constant_op.constant("b")))
    self.assertAllEqual(b"ab", root.f(constant_op.constant("a"), "b"))

    concrete_functions = root.f._list_all_concrete_functions_for_serialization()  # pylint: disable=protected-access
    self.assertLen(concrete_functions, 3)

    imported = cycle(root, cycles)

    self.assertAllEqual(b"ab", imported.f("a", "b"))
    self.assertAllEqual(b"ab", imported.f("a", constant_op.constant("b")))
    self.assertAllEqual(b"ab", imported.f(constant_op.constant("a"), "b"))

  def test_function_no_return(self, cycles):

    class TrackableWithOneVariable(tracking.AutoTrackable):

      def __init__(self, initial_value=0.0):
        super(TrackableWithOneVariable, self).__init__()
        self.variable = variables.Variable(initial_value)

      @def_function.function
      def increase(self, by=1.0):
        self.variable.assign_add(by)

    obj = TrackableWithOneVariable(5.0)

    obj.increase(constant_op.constant(10.0))
    self.assertEqual(15.0, obj.variable.numpy())
    obj.increase()
    self.assertEqual(16.0, obj.variable.numpy())

    imported = cycle(obj, cycles)

    imported.increase(constant_op.constant(10.0))
    self.assertEqual(26.0, imported.variable.numpy())
    imported.increase(constant_op.constant(1.0))
    self.assertEqual(27.0, imported.variable.numpy())

  def test_structured_inputs(self, cycles):

    def func(x, training=True):
      # x is a nested structure, we care about one particular tensor.
      _, (a, b) = x
      if training:
        return 2 * a["a"] + b
      else:
        return 7

    root = tracking.AutoTrackable()
    root.f = def_function.function(func)

    x = constant_op.constant(10)
    y = constant_op.constant(11)

    input1 = [6, ({"a": x}, y)]
    input2 = [7, ({"a": x}, y)]  # Not compatible with input1 signature.
    input3 = [6, ({"a": y}, x)]  # Compatible with input1 signature.

    # Note: by only calling f(input1) before serialization, only inputs with
    # matching signature will be valid on the loaded model.
    self.assertEqual(31, root.f(input1).numpy())

    imported = cycle(root, cycles)

    with self.assertRaisesRegex(ValueError,
                                "Could not find matching function to call"):
      imported.f(input2)

    self.assertEqual(31, imported.f(input1).numpy())
    self.assertEqual(32, imported.f(input3).numpy())

  def test_structured_inputs_bare_concrete_function(self, cycles):

    def func(x, training=True):
      # x is a nested structure, we care about one particular tensor.
      _, (a, b) = x
      if training:
        return 2 * a["a"] + b
      else:
        return 7

    x = constant_op.constant(10)
    y = constant_op.constant(11)

    input1 = [6, ({"a": x}, y)]
    input2 = [7, ({"a": x}, y)]  # Not compatible with input1 signature.
    input3 = [6, ({"a": y}, x)]  # Compatible with input1 signature.

    root = tracking.AutoTrackable()
    root.f = def_function.function(func).get_concrete_function(input1)

    imported = cycle(root, cycles)

    with self.assertRaises(TypeError):
      imported.f(input2)

    self.assertEqual(31, imported.f(input1).numpy())
    self.assertEqual(32, imported.f(input3).numpy())

  def test_structured_output(self, cycles):

    # Use fields with non-alphabetical order
    named_tuple_type = collections.namedtuple("NamedTupleHello", ["b", "a"])

    def func(input1, input2):
      named_tuple = named_tuple_type(a=input1 + input2, b=input1 * input2)
      return [named_tuple, input2, {"x": 0.5}]

    root = tracking.AutoTrackable()
    root.f = def_function.function(func)

    result = root.f(constant_op.constant(2), constant_op.constant(3))

    self.assertEqual(5, result[0].a.numpy())
    self.assertEqual(6, result[0].b.numpy())
    self.assertEqual(["b", "a"], list(result[0]._asdict().keys()))
    self.assertEqual(3, result[1].numpy())
    self.assertEqual(0.5, result[2]["x"].numpy())

    imported = cycle(root, cycles)

    result = imported.f(constant_op.constant(2), constant_op.constant(5))
    self.assertEqual(7, result[0].a.numpy())
    self.assertEqual(10, result[0].b.numpy())
    self.assertEqual(["b", "a"], list(result[0]._asdict().keys()))
    self.assertEqual(5, result[1].numpy())
    self.assertEqual(0.5, result[2]["x"].numpy())

  def test_pretty_print_signature(self, cycles):

    named_tuple_type = collections.namedtuple("NamedTupleHello", ["b", "a"])

    def func(input1, input2):
      named_tuple = named_tuple_type(a=input1 + input2, b=input1 * input2)
      return [named_tuple, input2, {"x": 0.5}]

    root = tracking.AutoTrackable()
    root.f = def_function.function(func).get_concrete_function(
        constant_op.constant(2), constant_op.constant(3))

    imported = cycle(root, cycles)
    self.assertEqual(
        imported.f.pretty_printed_signature(), """func(input1, input2)
  Args:
    input1: int32 Tensor, shape=()
    input2: int32 Tensor, shape=()
  Returns:
    [NamedTupleHello(b=<1>, a=<2>), <3>, {'x': <4>}]
      <1>: int32 Tensor, shape=()
      <2>: int32 Tensor, shape=()
      <3>: int32 Tensor, shape=()
      <4>: float32 Tensor, shape=()""")

  def test_positional_arguments(self, cycles):
    def func(x, training=False, abc=7.1, defg=7.7):
      del abc
      if training:
        return 2 * x
      if defg == 7:
        return 6
      else:
        return 7

    root = tracking.AutoTrackable()
    root.f = def_function.function(func)

    self.assertEqual(20, root.f(constant_op.constant(10), True).numpy())
    self.assertEqual(7, root.f(constant_op.constant(1)).numpy())
    self.assertEqual(2, root.f(constant_op.constant(1), True).numpy())
    self.assertEqual(6, root.f(constant_op.constant(1), defg=7.0).numpy())

    imported = cycle(root, cycles)

    self.assertEqual(4, imported.f(constant_op.constant(2), True).numpy())
    self.assertEqual(7, imported.f(constant_op.constant(2)).numpy())
    self.assertEqual(6, imported.f(constant_op.constant(1), defg=7.0).numpy())

  def test_additional_kwargs(self, cycles):
    def func(x, training=False, **options):
      del options
      if training:
        return 2 * x
      else:
        return 7

    root = tracking.AutoTrackable()
    root.f = def_function.function(func)

    x = constant_op.constant(10)
    self.assertEqual(7, root.f(x, learning_rate=0.5, epochs=3).numpy())

    imported = cycle(root, cycles)

    with self.assertRaisesRegex(ValueError,
                                "Could not find matching function to call.*"):
      imported.f(x, learning_rate=0.5, epochs=4)

    self.assertEqual(7, imported.f(x, learning_rate=0.5, epochs=3).numpy())

  def test_member_function(self, cycles):
    class TrackableWithMember(tracking.AutoTrackable):

      def __init__(self):
        super(TrackableWithMember, self).__init__()
        self._some_value = 20

      @def_function.function
      def f(self, x, training=False):
        if training:
          return 2 * x
        else:
          return 7 + self._some_value

    root = TrackableWithMember()

    self.assertEqual(20, root.f(constant_op.constant(10), True).numpy())
    self.assertEqual(27, root.f(constant_op.constant(1)).numpy())
    self.assertEqual(2, root.f(constant_op.constant(1), True).numpy())

    imported = cycle(root, cycles)

    self.assertEqual(4, imported.f(constant_op.constant(2), True).numpy())
    self.assertEqual(27, imported.f(constant_op.constant(2)).numpy())

  def test_side_effect_listing(self, cycles):
    class M(tracking.AutoTrackable):

      def __init__(self):
        super(M, self).__init__()
        self.var = None

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
      def f(self, x):
        if self.var is None:
          self.var = variables.Variable(2.)
        return x * self.var

    m = M()
    cycle(m, cycles)
    self.assertEqual(4.0, m.f(constant_op.constant(2.0)).numpy())

  def test_basic_backprop(self, cycles):
    weight = variables.Variable(1., trainable=True)
    bias = variables.Variable(0., trainable=True)
    g = def_function.function(
        lambda x: x*weight + bias,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])

    root = tracking.AutoTrackable()
    root.weight = weight
    root.bias = bias
    root.g = g
    imported = cycle(root, cycles)
    with backprop.GradientTape() as t:
      x = constant_op.constant([3.5])
      loss = imported.g(x)
      grad = t.gradient(loss, [imported.weight, imported.bias])
      self.assertAllClose(grad, [3.5, 1.0])

  def test_nested_backprop(self, cycles):
    weight = variables.Variable(1., trainable=True)
    bias = variables.Variable(0., trainable=True)

    # Note: this function gets called from other function defs via a
    # "PartitionedCall" op node.
    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(None, dtypes.float32),
        tensor_spec.TensorSpec(None, dtypes.float32)])
    def mul(x, y):
      return x * y

    # Note: this function gets called from other function defs via a
    # "StatefulPartitionedCall" op node.
    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(None, dtypes.float32)])
    def f(x):
      return mul(weight.read_value(), x)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(None, dtypes.float32)])
    def g(x):
      return f(x) + bias,

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(None, dtypes.float32)])
    def h(x):
      return g(x) + bias,

    root = tracking.AutoTrackable()
    root.weight = weight
    root.bias = bias
    root.g = h

    imported = cycle(root, cycles)
    with backprop.GradientTape() as t:
      x = constant_op.constant([3.5])
      loss = imported.g(x)
    grad = t.gradient(loss, [imported.weight, imported.bias])
    self.assertAllClose(grad, [3.5, 2.0])

  def test_while_loop_backprop(self, cycles):
    weight = variables.Variable(2., trainable=True)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(dtype=dtypes.float32, shape=(None, None))])
    def g(x):
      """Adds rows of matrix x after multiplying each entry by v."""
      i_0 = constant_op.constant(0)
      s_0 = constant_op.constant([0., 0.])
      cond = lambda i, _: i < array_ops.shape(x)[1]
      body = lambda i, s: (i + 1, s + weight * x[:, i])
      i_end, s_end = control_flow_ops.while_loop(cond, body, (i_0, s_0))
      del i_end
      return s_end

    root = tracking.AutoTrackable()
    root.weight = weight
    root.g = g
    imported = cycle(root, cycles)

    def get_gradient(obj):
      with backprop.GradientTape() as t:
        x = constant_op.constant([[1., 2., 3.], [1., -2, 3.]])
        y = obj.g(x)
        self.assertAllClose(y, obj.weight * [6., 2.])
        loss = math_ops.reduce_sum(y)  # weight * 8.
        self.assertAllEqual(t.watched_variables(), [obj.weight])
        return t.gradient(loss, obj.weight)

    imported_gradient = get_gradient(imported)
    original_gradient = get_gradient(root)
    self.assertIsNotNone(original_gradient)
    self.assertAllClose(original_gradient, 8.)
    self.assertIsNotNone(imported_gradient)
    self.assertAllClose(imported_gradient, 8.)

  def _test_restored_func_with_captured_var_backprop(self, cycles, dtype):
    weight = variables.Variable(2., trainable=True, dtype=dtype)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(dtype=dtype, shape=())])
    def g(x):
      return x * weight

    root = tracking.AutoTrackable()
    root.weight = weight
    root.g = g
    imported = cycle(root, cycles)

    def get_gradient(obj):
      with backprop.GradientTape() as t:
        x = constant_op.constant(2.)
        y = obj.g(x)
        self.assertAllClose(y, obj.weight * 2.)
        self.assertAllEqual(t.watched_variables(), [obj.weight])
        return t.gradient(y, obj.weight)

    imported_gradient = get_gradient(imported)
    original_gradient = get_gradient(root)
    self.assertIsNotNone(original_gradient)
    self.assertAllClose(original_gradient, 2.)
    self.assertIsNotNone(imported_gradient)
    self.assertAllClose(imported_gradient, 2.)

  def test_nested_fn_backprop(self, cycles):
    weight = variables.Variable(2., trainable=True)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(dtype=dtypes.float32, shape=(None, None))])
    def g(x):
      weight.read_value()  # Just get the tape to watch the variable
      handle = array_ops.identity(weight.handle)
      @def_function.function
      def launder_var_handle():
        return array_ops.identity(handle)
      return x + resource_variable_ops.read_variable_op(
          launder_var_handle(), dtypes.float32)

    root = tracking.AutoTrackable()
    root.weight = weight
    root.g = g
    imported = cycle(root, cycles)
    def get_gradient(obj, persistent):
      with backprop.GradientTape(persistent=persistent) as t:
        x = constant_op.constant([[1., 2., 3.], [1., -2, 3.]])
        y = obj.g(x)
        self.assertAllClose(y, obj.weight + x)
        loss = math_ops.reduce_sum(y)
        return t.gradient(loss, obj.weight)

    imported_gradient = get_gradient(imported, persistent=False)
    original_gradient = get_gradient(root, persistent=False)
    self.assertIsNotNone(original_gradient)
    self.assertAllClose(original_gradient, 6.)
    self.assertIsNotNone(imported_gradient)
    self.assertAllClose(imported_gradient, 6.)

  def test_restored_func_with_captured_var_backprop_float32(self, cycles):
    self._test_restored_func_with_captured_var_backprop(cycles, dtypes.float32)

  def test_restored_func_with_captured_var_backprop_float64(self, cycles):
    self.skipTest("b/144573917")
    self._test_restored_func_with_captured_var_backprop(cycles, dtypes.float64)

  def test_callable(self, cycles):
    class M1(tracking.AutoTrackable):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
      def __call__(self, x):
        return x

    root = tracking.AutoTrackable()
    root.m1 = M1()
    root.m2 = tracking.AutoTrackable()
    root.m2.__call__ = def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])(
            lambda x: x*3.0)
    imported = cycle(root, cycles)
    x = constant_op.constant(1.0)

    self.assertTrue(callable(imported.m1))
    self.assertAllEqual(root.m1(x), imported.m1(x))

    # Note: `root.m2` was not callable since `__call__` attribute was set
    # into the instance and not on the class. But after a serialization cycle
    # that starts to work.
    self.assertTrue(callable(imported.m2))
    self.assertAllEqual(root.m2.__call__(x), imported.m2(x))

    # Verify that user objects without `__call__` attribute are not callable.
    self.assertFalse(callable(imported))

  def test_chain_callable(self, cycles):
    func = def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])(
            lambda x: x*3.0)
    root = tracking.AutoTrackable()
    root.__call__ = tracking.AutoTrackable()
    root.__call__.__call__ = tracking.AutoTrackable()
    root.__call__.__call__.__call__ = func

    imported = cycle(root, cycles)
    self.assertTrue(callable(imported))
    x = constant_op.constant(1.0)
    self.assertAllEqual(imported(x).numpy(), 3.0)

  def test_load_in_graph_mode(self, cycles):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(1., name="v_one", trainable=False)
    root.v2 = variables.Variable(2., name="v_two", trainable=True)
    root.f = def_function.function(
        lambda x: root.v2 * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])

    if cycles > 1:
      root = cycle(root, cycles - 1)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)

    with ops.Graph().as_default() as g:
      imported = load.load(path)
      var_v1 = imported.v1
      self.assertFalse(var_v1.trainable)
      var_v2 = imported.v2
      self.assertTrue(var_v2.trainable)
      output = imported.f(constant_op.constant(2.))
      with monitored_session.MonitoredSession() as sess:
        self.assertEqual(1.0, sess.run(var_v1))
        self.assertEqual(4.0, sess.run(output))
      self.assertCountEqual([var_v1, var_v2],
                            g.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
      # load() should not add to TRAINABLE_VARIABLES. Higher levels of model
      # building control retraining or frozen use of imported SavedModels.
      self.assertCountEqual([],
                            g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))

  def test_load_in_func_graph(self, cycles):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(1.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(
        lambda x: root.v2 * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])

    if cycles > 1:
      root = cycle(root, cycles - 1)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)

    closure = tracking.AutoTrackable()
    @def_function.function
    def func(x):
      if not hasattr(closure, "model"):
        closure.model = load.load(path)
      return closure.model.f(x)

    inputs = constant_op.constant(2.)
    self.assertEqual(4.0, func(inputs).numpy())

  def test_soft_matching(self, cycles):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)])
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func

    self.assertAllEqual([2], root.f(constant_op.constant([1])).numpy())
    self.assertAllEqual([2, 4], root.f(constant_op.constant([1, 2])).numpy())

    concrete_functions = root.f._list_all_concrete_functions_for_serialization()  # pylint: disable=protected-access
    self.assertLen(concrete_functions, 1)

    imported = cycle(root, cycles)

    with self.assertRaisesRegex(ValueError, "Python inputs incompatible"):
      # We cannot call the function with a constant of shape ().
      imported.f(constant_op.constant(2)).numpy()

    # TODO(vbardiovsky): When classes are revived with input_signatures, we
    # should also check that the calls below are not generating any more
    # concrete functions.
    self.assertAllEqual([2, 4, 6, 8],
                        imported.f(constant_op.constant([1, 2, 3, 4])).numpy())
    self.assertAllEqual([2, 4, 6],
                        imported.f(constant_op.constant([1, 2, 3])).numpy())

  def test_jit_compile(self, cycles):

    # It'd be nice to use parameterize here, but the library does not support
    # having parameterized test methods inside already-parameterized classes.
    for jit_compile in (None, True, False):

      @def_function.function(jit_compile=jit_compile)
      def f(x):
        return x + 1.

      root = module.Module()
      root.f = f
      save_dir = os.path.join(self.get_temp_dir(), "saved_model")
      save.save(root, save_dir)

      imported = cycle(root, cycles)

      self.assertEqual(imported.f._jit_compile, jit_compile)

  def test_get_concrete_function(self, cycles):

    @def_function.function
    def func(x, training=False):
      if training:
        return 2 * x
      else:
        return 3 * x

    func.get_concrete_function(
        tensor_spec.TensorSpec([None], dtypes.int32), True)
    func.get_concrete_function(tensor_spec.TensorSpec([None], dtypes.float32))

    root = tracking.AutoTrackable()
    root.f = func

    imported = cycle(root, cycles)

    concrete = imported.f.get_concrete_function(
        training=True, x=tensor_spec.TensorSpec([None], dtypes.int32))

    self.assertAllEqual([2, 4, 6, 8],
                        concrete(x=constant_op.constant([1, 2, 3, 4])).numpy())
    with self.assertRaisesRegex(ValueError,
                                "Could not find matching function to call"):
      imported.f.get_concrete_function(
          tensor_spec.TensorSpec([None], dtypes.int32))
    imported.f.get_concrete_function(
        tensor_spec.TensorSpec([None], dtypes.int32), True)

  def test_concrete_function(self, cycles):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)])
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function()

    self.assertAllEqual([2], root.f(constant_op.constant([1])).numpy())
    self.assertAllEqual([2, 4], root.f(constant_op.constant([1, 2])).numpy())

    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(root, cycles, signatures={})

    self.assertAllEqual([2, 4, 6, 8],
                        imported.f(constant_op.constant([1, 2, 3, 4])).numpy())
    self.assertAllEqual([2, 4, 6],
                        imported.f(constant_op.constant([1, 2, 3])).numpy())

  def test_concrete_function_captures(self, cycles):

    class Root(module.Module):

      def __init__(self):
        self.v = variables.Variable(1.)
        self.v1 = variables.Variable(1.)

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
      def use_v(self, x):
        return self.v + self.v1 + 1.

    root = Root()
    self.assertIn(root.v.handle,
                  root.use_v.get_concrete_function().graph.external_captures)
    root = cycle(root, cycles, signatures=root.use_v.get_concrete_function())
    func_captures = root.use_v.get_concrete_function().graph.external_captures
    self.assertLen(func_captures, 2)
    self.assertTrue(any(root.v.handle is t for t in func_captures))
    self.assertTrue(any(root.v1.handle is t for t in func_captures))
    signature_captures = root.signatures[
        "serving_default"].graph.external_captures
    self.assertLen(signature_captures, 2)
    self.assertTrue(any(root.v.handle is t for t in signature_captures))
    self.assertTrue(any(root.v1.handle is t for t in signature_captures))

  def test_concrete_function_arg_names(self, cycles):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)])
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function()

    self.assertAllEqual([2], root.f(constant_op.constant([1])).numpy())

    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(root, cycles, signatures={})

    self.assertAllEqual([2, 4, 6],
                        imported.f(x=constant_op.constant([1, 2, 3])).numpy())

  def test_concrete_function_no_signature(self, cycles):
    @def_function.function
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function(constant_op.constant([1]))
    self.assertAllEqual([4], root.f(constant_op.constant([2])).numpy())
    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(root, cycles, signatures={})
    self.assertAllEqual([6],
                        imported.f(constant_op.constant([3])).numpy())

  @test_util.run_in_graph_and_eager_modes
  def test_concrete_function_backprop(self, cycles):
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.float32)])
    def func(x):
      return x ** 2.
    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function()

    def _compute_gradient(function):
      with backprop.GradientTape() as tape:
        inp = constant_op.constant(1.)
        tape.watch(inp)
        output = function(inp)
      return tape.gradient(output, inp)

    self.assertAllEqual(2., _compute_gradient(root.f))
    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(root, cycles, signatures={})
    self.assertAllEqual(2., _compute_gradient(imported.f))

  def test_revived_concrete_function_kwargs(self, cycles):

    @def_function.function
    def func(x, y):
      return x * (y + 1.)
    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function(
        tensor_spec.TensorSpec([], dtypes.float32),
        tensor_spec.TensorSpec([], dtypes.float32))
    self.assertEqual(8., root.f(y=constant_op.constant(3.),
                                x=constant_op.constant(2.)).numpy())
    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(root, cycles, signatures={})
    self.assertEqual(8., imported.f(y=constant_op.constant(3.),
                                    x=constant_op.constant(2.)).numpy())

  def test_revived_concrete_function_tensorspec_kwargs(self, cycles):

    @def_function.function
    def func(*args):
      x, y = args
      return x * (y + 1.)
    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function(
        tensor_spec.TensorSpec([], dtypes.float32, name="x"),
        tensor_spec.TensorSpec([], dtypes.float32, name="y"))
    self.assertEqual(8., root.f(y=constant_op.constant(3.),
                                x=constant_op.constant(2.)).numpy())
    imported = cycle(root, cycles, signatures={})
    self.assertEqual(8., imported.f(y=constant_op.constant(3.),
                                    x=constant_op.constant(2.)).numpy())

  def test_concrete_function_variable_argument(self, cycles):
    capture = variables.Variable(0)

    @def_function.function
    def func(v):
      v.assign_add(1)
      capture.assign_sub(1)

    @def_function.function(input_signature=[
        resource_variable_ops.VariableSpec(shape=[], dtype=dtypes.int32)
    ])
    def func_with_input_signature(v):
      v.assign_add(5)
      capture.assign_sub(5)
      return 1

    vsave = variables.Variable(1)
    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function(vsave)
    root.f_sig = func_with_input_signature.get_concrete_function()
    root.capture = capture

    self.assertEqual(1, vsave.numpy())
    root.f(vsave)
    self.assertEqual(2, vsave.numpy())
    self.assertEqual(-1, capture.numpy())

    root.f_sig(vsave)
    self.assertEqual(7, vsave.numpy())
    self.assertEqual(-6, capture.numpy())

    imported = cycle(root, cycles)

    vload = variables.Variable(1)
    imported.f(vload)
    self.assertEqual(2, vload.numpy())
    imported.f(v=vload)
    self.assertEqual(3, vload.numpy())
    self.assertEqual(-8, imported.capture.numpy())

    imported.f_sig(v=vload)
    self.assertEqual(8, vload.numpy())
    self.assertEqual(-13, imported.capture.numpy())

    self.assertEqual(-6, capture.numpy())

  def test_function_and_component(self, cycles):

    @def_function.function
    def func(v):
      return v + 1

    root = tracking.AutoTrackable()
    root.func = func
    root.concrete_func = func.get_concrete_function(
        tensor_spec.TensorSpec(None, dtypes.int32))
    one = constant_op.constant(1)
    self.assertEqual(2, root.func(one).numpy())
    self.assertEqual(2, root.concrete_func(one).numpy())
    imported = cycle(root, cycles)
    self.assertEqual(2, imported.func(one).numpy())
    self.assertEqual(2, imported.concrete_func(one).numpy())

  def test_dict(self, cycles):
    root = tracking.AutoTrackable()
    root.variables = dict(a=variables.Variable(1.))
    root.variables["b"] = variables.Variable(2.)
    root.variables["c"] = 1
    root.funcs = dict(
        a=def_function.function(lambda: constant_op.constant(100.)))
    root.funcs["conc"] = root.funcs["a"].get_concrete_function()
    imported = cycle(root, cycles)
    self.assertEqual(1., imported.variables["a"].numpy())
    self.assertEqual(2., imported.variables["b"].numpy())
    self.assertEqual(set(["a", "b"]), set(imported.variables.keys()))
    self.assertEqual(100., imported.funcs["a"]().numpy())
    self.assertEqual(100., imported.funcs["conc"]().numpy())

  def test_list(self, cycles):
    root = tracking.AutoTrackable()
    root.variables = [variables.Variable(1.)]
    root.variables.append(1)
    root.variables.append(variables.Variable(3.))
    imported = cycle(root, cycles)
    self.assertEqual(1., imported.variables[0].numpy())
    self.assertEqual(3., imported.variables[2].numpy())
    self.assertIs(None, imported.variables[1])
    self.assertLen(imported.variables, 3)

  def test_tuple(self, cycles):
    root = tracking.AutoTrackable()
    root.variables = (variables.Variable(1.), 1, variables.Variable(3.))
    imported = cycle(root, cycles)
    self.assertEqual(1., imported.variables[0].numpy())
    self.assertEqual(3., imported.variables[2].numpy())
    self.assertIs(None, imported.variables[1])
    self.assertLen(imported.variables, 3)

  def test_functions_list(self, cycles):
    root = tracking.AutoTrackable()
    v1 = variables.Variable(1.)
    root.losses = [def_function.function(lambda: math_ops.reduce_sum(v1 ** 2))]
    root.variables = [v1]

    @def_function.function
    def _v2_loss():
      if len(root.variables) == 1:
        v2 = variables.Variable(2.)
        root.variables.append(v2)
      return math_ops.reduce_sum(root.variables[1] ** 2)

    root.losses.append(_v2_loss)
    self.assertAllClose([1., 4.], [loss() for loss in root.losses])
    imported = cycle(root, cycles)
    self.assertAllClose([1., 4.], [loss() for loss in imported.losses])
    imported.variables[0].assign(3.)
    imported.variables[1].assign(4.)
    self.assertAllClose([9., 16.], [loss() for loss in imported.losses])

  def test_captured_constant(self, cycles):
    const = array_ops.zeros([100])
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda: const + 1.)
    root.g = def_function.function(lambda: const + 2.)
    self.assertAllClose(array_ops.ones([100]), root.f())
    self.assertAllClose(2. * array_ops.ones([100]), root.g())
    imported = cycle(root, cycles)
    self.assertAllClose(array_ops.ones([100]), imported.f())
    self.assertAllClose(2. * array_ops.ones([100]), imported.g())
    # TODO(b/123408994): Use the public get_concrete_function.
    f_concrete = imported.f._list_all_concrete_functions_for_serialization()[0]
    g_concrete = imported.g._list_all_concrete_functions_for_serialization()[0]
    self.assertLen(f_concrete.captured_inputs, 1)
    self.assertLen(g_concrete.captured_inputs, 1)
    # We should be using the same captured EagerTensor in both functions, not
    # duplicating the constant.
    self.assertIs(f_concrete.captured_inputs[0],
                  g_concrete.captured_inputs[0])

  def test_functions_accessed_once(self, cycles):

    class Exported(tracking.AutoTrackable):

      def __init__(self):
        self._counter = 0

      @property
      def make_func(self):
        @def_function.function
        def f():
          return constant_op.constant(self._counter)
        f.get_concrete_function()  # force a trace
        self._counter += 1
        return f

    exported = Exported()
    imported = cycle(exported, cycles)
    self.assertEqual(0, imported.make_func().numpy())
    self.assertEqual(1, exported.make_func().numpy())

  def test_overwritten_signatures_error(self, cycles):
    exported = tracking.AutoTrackable()
    exported.f = def_function.function(lambda: constant_op.constant(1.))
    imported = cycle(
        exported, cycles,
        signatures={"key": exported.f.get_concrete_function()})
    self.assertEqual(1., imported.signatures["key"]()["output_0"].numpy())
    imported.signatures = {"key1": imported.signatures["key"]}
    with self.assertRaisesRegex(ValueError, "signatures"):
      save.save(imported, tempfile.mkdtemp(prefix=self.get_temp_dir()))

  def test_signature_loading(self, cycles):

    class Exported(tracking.AutoTrackable):

      def __init__(self):
        self.v = variables.Variable(3.)

      @def_function.function
      def do(self, x):
        return self.v * x

    exported = Exported()
    imported = cycle(
        exported,
        cycles,
        signatures=exported.do.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.float32)))
    self.assertEqual(["serving_default"], list(imported.signatures.keys()))
    imported_function = imported.signatures["serving_default"]
    two = constant_op.constant(2.)
    self.assertEqual(6., imported_function(x=two)["output_0"].numpy())
    imported.v.assign(4.)
    self.assertEqual(8., imported_function(x=two)["output_0"].numpy())
    self.assertEqual(8., imported_function(two)["output_0"].numpy())
    with self.assertRaises(TypeError):
      # The signatures mapping is immutable
      imported.signatures["random_key"] = 3

  def test_names_normalized(self, cycles):
    class ObjWithFunction(module.Module):

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec([], dtype=dtypes.int32, name="A-b"),
          tensor_spec.TensorSpec([], dtype=dtypes.int32, name="A/D"),
          tensor_spec.TensorSpec([], dtype=dtypes.int32, name="bar"),
          tensor_spec.TensorSpec([], dtype=dtypes.int32, name="e"),
      ])
      def foo(self, a, b, c, d=10, **options):
        del options
        return a + b + c + d

    exported = ObjWithFunction()

    with self.assertLogs(level="WARNING") as logs:
      imported = cycle(exported, cycles)

    expected_message = (
        "WARNING:absl:Function `foo` contains input name(s) A-b, A/D with "
        "unsupported characters which will be renamed to a_b, a_d in the "
        "SavedModel.")
    self.assertIn(expected_message, logs.output)

    loaded_signature = imported.signatures["serving_default"].inputs
    self.assertEqual("a_b:0", loaded_signature[0].name)
    self.assertEqual("a_d:0", loaded_signature[1].name)

  def test_multiple_argument_signatures_no_positional(self, cycles):

    class Exported(tracking.AutoTrackable):

      @def_function.function
      def do(self, x, y):
        return x + y

    exported = Exported()
    imported = cycle(
        exported, cycles, signatures=exported.do.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.float32),
            tensor_spec.TensorSpec(None, dtypes.float32)))
    with self.assertRaises(TypeError):
      imported.signatures["serving_default"](
          constant_op.constant(1.),
          y=constant_op.constant(2.))
    self.assertEqual(
        {"output_0": 3.},
        self.evaluate(imported.signatures["serving_default"](
            x=constant_op.constant(1.),
            y=constant_op.constant(2.))))

  def _make_model_with_tables(self):
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table1_initializer = lookup_ops.KeyValueTensorInitializer(keys, values)
    table1 = lookup_ops.HashTable(table1_initializer, default_val)

    table2_file = self._make_asset("test\nfoo\nbrain\n")
    table2_initializer = lookup_ops.TextFileIdTableInitializer(table2_file)
    table2 = lookup_ops.HashTable(table2_initializer, default_val)

    def _make_lookup_function(table):
      signature = [tensor_spec.TensorSpec(None, dtypes.string)]
      return def_function.function(input_signature=signature)(
          lambda x: table.lookup(x))  # pylint: disable=unnecessary-lambda

    root = tracking.AutoTrackable()
    root.table1 = table1
    root.lookup1 = _make_lookup_function(table1)
    root.table2 = table2
    root.lookup2 = _make_lookup_function(table2)
    return root

  def test_table(self, cycles):
    root = self._make_model_with_tables()
    imported = cycle(root, cycles, signatures={})
    keys = constant_op.constant(["brain", "test", "foo", "surgery"])
    self.assertAllEqual([0, -1, -1, 2], imported.lookup1(keys).numpy())
    self.assertAllEqual([2, 0, 1, -1], imported.lookup2(keys).numpy())

  def test_table_collections_untouched_eager(self, cycles):

    def _gather_nonempty_collections():
      graph = ops.get_default_graph()
      gathered = {}
      for collection in graph.collections:
        collection_contents = graph.get_collection(collection)
        if collection_contents:
          gathered[collection] = collection_contents
      return gathered

    root = self._make_model_with_tables()
    # Warm up collections to ignore those that don't expand every iteration,
    # e.g. the __varscope collection.
    cycle(root, 1)
    original_collections = _gather_nonempty_collections()
    cycle(root, cycles)
    self.assertEqual(original_collections, _gather_nonempty_collections())

  def test_table_in_graph(self, cycles):
    root = self._make_model_with_tables()

    if cycles > 1:
      root = cycle(root, cycles - 1)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)
    imported = cycle(root, 1)

    with ops.Graph().as_default():
      imported = load.load(path)
      keys = constant_op.constant(["brain", "test", "foo", "surgery"])
      output1 = imported.lookup1(keys)
      output2 = imported.lookup2(keys)
      with monitored_session.MonitoredSession() as sess:
        self.assertAllEqual([0, -1, -1, 2], sess.run(output1))
        self.assertAllEqual([2, 0, 1, -1], sess.run(output2))

  def test_preserve_argspec(self, cycles):

    def f(a, b, c):  # pylint: disable=unused-argument
      return None

    original_fullargspec = tf_inspect.getfullargspec(f)

    root = tracking.AutoTrackable()
    root.f = def_function.function(f)
    imported = cycle(root, cycles)

    restored_fullargspec = tf_inspect.getfullargspec(imported.f)
    self.assertEqual(original_fullargspec, restored_fullargspec)

  def test_canonicalize_inputs(self, cycles):
    @def_function.function(autograph=False)
    def func(a=1, b=2, c=3, training=True):
      if training:
        return [a, b, c, training]
      else:
        return [c, b, a, training]

    # TODO(b/123501567): Work-around to trigger generic traces of a function
    # with extra non tensor args.
    signature = 3*[tensor_spec.TensorSpec(None, dtypes.float32)]
    @def_function.function(input_signature=signature)
    def trigger(a, b, c):
      func(a, b, c, True)
      func(a, b, c, False)

    trigger.get_concrete_function()

    root = tracking.AutoTrackable()
    root.f = func
    root = cycle(root, cycles)
    self.assertAllEqual(root.f(), [1.0, 2.0, 3.0, True])
    self.assertAllEqual(root.f(-1.0, training=False), [3.0, 2.0, -1.0, False])

    with self.assertRaisesRegex(ValueError, "Could not find matching function"):
      root.f(["hello", 1.0])

  def test_prefer_specific_trace(self, cycles):
    @def_function.function(autograph=False)
    def func(a):
      if isinstance(a, int):
        return a
      else:
        return a + 1

    self.assertAllEqual(2, func(2).numpy())
    self.assertAllEqual(3, func(constant_op.constant(2)).numpy())

    root = tracking.AutoTrackable()
    root.f = func
    root = cycle(root, cycles)
    self.assertAllEqual(2, root.f(2).numpy())
    self.assertAllEqual(4, root.f(3).numpy())
    self.assertAllEqual(3, root.f(constant_op.constant(2)).numpy())
    self.assertAllEqual(4, root.f(constant_op.constant(3)).numpy())

  def test_partial(self, cycles):
    def f(x, y):
      return x + y

    func = def_function.function(
        functools.partial(f, x=array_ops.zeros([1]), y=array_ops.ones([1])))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(), [1.0])

    root = cycle(root, cycles)
    self.assertAllEqual(root.f(), [1.0])

  def test_partial_with_non_tensor_defaults(self, cycles):

    def f(x, y=3):
      return x + y

    func = def_function.function(functools.partial(f, y=5))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(1), 6)

    root = cycle(root, cycles)
    self.assertAllEqual(root.f(1), 6)

  def test_partial_with_positional(self, cycles):
    def f(x, y):
      return x + y

    func = def_function.function(functools.partial(f, constant_op.constant(5)))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(1), 6)

    root = cycle(root, cycles)
    self.assertAllEqual(root.f(1), 6)

  def test_partial_with_positional_captured_tensors(self, cycles):

    def f(x, y):
      return x + y

    tensor = constant_op.constant(5) + constant_op.constant(7)
    func = def_function.function(functools.partial(f, tensor))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(1), 13)

    root = cycle(root, cycles)
    self.assertAllEqual(root.f(1), 13)

  def test_partial_keyword_hiding_default(self, cycles):

    def f(x=3, training=True, y=7):
      if training:
        return x + y
      else:
        return x + y + 2

    func = def_function.function(functools.partial(f, y=6))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertEqual(root.f().numpy(), 9)
    self.assertEqual(root.f(training=False).numpy(), 11)

    root = cycle(root, cycles)
    self.assertEqual(root.f().numpy(), 9)
    self.assertEqual(root.f(training=False).numpy(), 11)

  def test_partial_with_kwargs(self, cycles):

    def f(a, b, *args, **kwargs):
      args_sum = sum(args)
      return a + b + kwargs["some_tensor"] * kwargs["learning_rate"] + args_sum

    constant_tensor = constant_op.constant(10)
    func = def_function.function(
        functools.partial(
            f, 7, 1, 2, learning_rate=3, some_tensor=constant_tensor))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertEqual(root.f(constant_op.constant(4)).numpy(), 44)

    root = cycle(root, cycles)
    self.assertEqual(root.f(constant_op.constant(5)).numpy(), 45)

  def test_partial_bind_only_first_argument(self, cycles):
    if sys.version_info[0] < 3:
      self.skipTest("Test is only valid in python3. Only then we get some more "
                    "advanced inspection of partials where this is allowed.")

    def f(x, y):
      return x + y

    partial_func = functools.partial(f, x=5)
    tf_func = def_function.function(partial_func)

    root = tracking.AutoTrackable()
    root.f = tf_func
    self.assertAllEqual(root.f(y=constant_op.constant(7)), 12)

    root = cycle(root, cycles)
    self.assertAllEqual(root.f(y=constant_op.constant(9)), 14)

  def test_partial_with_passed_fn_as_default(self, cycles):

    def f(x, y):
      return x(3) + y

    def my_func(a):
      return 2 * a

    func = def_function.function(functools.partial(f, my_func))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertEqual(root.f(constant_op.constant(3)).numpy(), 9)

    root = cycle(root, cycles)
    self.assertEqual(root.f(constant_op.constant(3)).numpy(), 9)

  def test_partial_with_input_signature(self, cycles):

    def full_function(a, b, c=3.0):
      return a, b, c

    partial = functools.partial(full_function, 1, c=4)
    self.assertAllEqual((1, 2.0, 4), partial(2.0))

    signature = [tensor_spec.TensorSpec([], dtypes.float32)]
    func = def_function.function(partial, input_signature=signature)

    root = tracking.AutoTrackable()
    root.f = func
    a, b, c = root.f(2.0)
    self.assertAllEqual([a.numpy(), b.numpy(), c.numpy()], (1, 2.0, 4))

    root = cycle(root, cycles)
    a, b, c = root.f(3.0)
    self.assertAllEqual([a.numpy(), b.numpy(), c.numpy()], (1, 3.0, 4))

  def test_convert_to_input_signature(self, cycles):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)])
    def func(x):
      return x

    root = tracking.AutoTrackable()
    root.f = func

    root = cycle(root, cycles)

    self.assertEqual([2], root.f([2]).numpy())

  def test_named_tuple(self, cycles):

    class NamedTupleType(collections.namedtuple("NamedTupleType", ["a", "b"])):
      pass

    @def_function.function
    def f(x):
      return x.a + x.b

    f.get_concrete_function(
        NamedTupleType(
            a=tensor_spec.TensorSpec(None, dtypes.float32, name="a"),
            b=tensor_spec.TensorSpec(None, dtypes.float32, name="b")))
    obj = tracking.AutoTrackable()
    obj.__call__ = f
    if sys.version_info.major == 3 and sys.version_info.minor < 5:
      # TODO(allenl): figure out why this doesn't work in Python3.4
      self.skipTest("Not working in Python 3.4")
    imported = cycle(obj, cycles)
    self.assertAllClose(3.,
                        imported(NamedTupleType(a=constant_op.constant(1.),
                                                b=constant_op.constant(2.))))

  def test_extra_args(self, cycles):

    @def_function.function
    def f(x):
      return math_ops.add(x["a"], 1.)
    # Trigger a trace.
    f({"a": constant_op.constant(2.0)})

    obj = tracking.AutoTrackable()
    obj.__call__ = f
    imported = cycle(obj, cycles)

    self.assertEqual(4.0, imported({"a": 3.0}).numpy())

    with self.assertRaisesRegex(ValueError,
                                "Could not find matching function to call"):
      imported({"a": 2.0, "b": 3.0})

  def test_shapes_available(self, cycles):

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec([None, 3], dtypes.int32),
        tensor_spec.TensorSpec([None, 2], dtypes.int32)
    ])
    def func(x, y):
      return array_ops.concat([x, y], axis=1)

    root = tracking.AutoTrackable()
    root.f = func

    root = cycle(root, cycles)

    imported_graph = root.f.get_concrete_function().graph
    input_x, input_y = imported_graph.inputs
    self.assertEqual([None, 3], input_x.shape.as_list())
    self.assertEqual([None, 2], input_y.shape.as_list())
    output, = imported_graph.outputs
    self.assertEqual([None, 5], output.shape.as_list())
    signature = root.signatures["serving_default"]
    self.assertEqual(
        [None, 3], signature.inputs[0].shape.as_list())
    self.assertEqual(
        [None, 2], signature.inputs[1].shape.as_list())
    self.assertEqual(
        [None, 5], signature.outputs[0].shape.as_list())

  def test_variables_destroyed(self, cycles):
    v1 = variables.Variable(1.)
    weak_v1 = weakref.ref(v1)
    root = util.Checkpoint(v=v1)
    root = cycle(root, cycles)
    del v1
    self.assertIsNone(weak_v1())
    weak_v2 = weakref.ref(root.v)
    del root
    self.assertIsNone(weak_v2())

  def test_variable_attributes_preserved(self, cycles):
    v = variables.Variable(
        1.,
        trainable=False,
        synchronization=variables.VariableSynchronization.NONE,
        aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)
    self.assertEqual(variables.VariableSynchronization.NONE,
                     v.synchronization)
    self.assertEqual(variables.VariableAggregation.ONLY_FIRST_REPLICA,
                     v.aggregation)
    root = tracking.AutoTrackable()
    root.v = v
    root = cycle(root, cycles)
    self.assertEqual(False, root.v.trainable)
    self.assertEqual(variables.VariableSynchronization.NONE,
                     root.v.synchronization)
    self.assertEqual(variables.VariableAggregation.ONLY_FIRST_REPLICA,
                     root.v.aggregation)

  def test_captured_dataset(self, cycles):

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
    self.assertEqual(
        3 * (1 + 4 + 9 + 16),
        root(constant_op.constant(3, dtype=dtypes.int64)).numpy())
    root = cycle(root, cycles)
    self.assertEqual(
        3 * (1 + 4 + 9 + 16),
        root(constant_op.constant(3, dtype=dtypes.int64)).numpy())

  def test_tuple_signature(self, cycles):
    root = util.Checkpoint()
    root.f = def_function.function(
        lambda: (array_ops.ones([]), array_ops.zeros([])),
        input_signature=())
    root = cycle(root, cycles, signatures=root.f)
    self.assertEqual(({"output_0": 1., "output_1": 0.}),
                     self.evaluate(root.signatures["serving_default"]()))

  def test_version_info(self, cycles):
    root = util.Checkpoint()
    root = cycle(root, cycles)
    self.assertEqual(versions.__version__, root.tensorflow_version)
    self.assertEqual(versions.__git_version__, root.tensorflow_git_version)

  def test_load_grad_save(self, cycles):
    root = util.Checkpoint()
    root.v = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v * x)
    root.g = def_function.function(root.f)
    for _ in range(cycles):
      with backprop.GradientTape() as tape:
        inp = constant_op.constant(2.)
        tape.watch(inp)
        output = root.g(inp)
        self.assertAllClose(4., output)
      self.assertAllClose(2., tape.gradient(output, inp))
      root = cycle(root, 1)

  def test_destroy_resource(self, cycles):

    def get_handle():
      return resource_variable_ops.var_handle_op(
          shape=tensor_shape.as_shape([]),
          dtype=dtypes.float32,
          shared_name="my_var_name",
          name="my_var",
          container="my_container")

    class MyResource(tracking.TrackableResource):

      def _create_resource(self):
        return get_handle()

      def _initialize(self):
        resource_variable_ops.assign_variable_op(
            self.resource_handle, 1.0, name="assign")

      def _destroy_resource(self):
        handle = get_handle()
        resource_variable_ops.destroy_resource_op(
            handle, ignore_lookup_error=True)

    class MyModel(tracking.AutoTrackable):

      def __init__(self):
        super(MyModel, self).__init__()
        self.resource = MyResource()

      @def_function.function(input_signature=[])
      def increase(self):
        handle = self.resource.resource_handle
        resource_variable_ops.assign_add_variable_op(
            handle, 10.0, name="assign_add")
        return resource_variable_ops.read_variable_op(handle, dtypes.float32)

    root = MyModel()
    imported = cycle(root, cycles)
    self.assertEqual(11, imported.increase().numpy())  # Create the resource.

    handle = imported.resource.resource_handle

    # Delete the imported SaveModel. Since we explicitly set the deleter, it
    # should destroy the resource automatically.
    del imported

    # Try to destroy the resource again, should fail.
    with self.assertRaisesRegex(errors.NotFoundError,
                                r"Resource .* does not exist."):
      resource_variable_ops.destroy_resource_op(
          handle, ignore_lookup_error=False)

  def test_function_called_as_operation(self, cycles):

    @framework_function.Defun(dtypes.float32)
    def inner(x):
      return x + 1.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.float32)])
    def outer(x):
      return inner(x)

    root = module.Module()
    root.f = outer
    imported = cycle(root, cycles)
    self.assertAllClose(2., imported.f(constant_op.constant(1.)))

  def test_ragged(self, cycles):

    @def_function.function
    def f(x, c=1):
      """Returns Tensor x incremented by Python constant c."""
      return math_ops.add(x, c)

    for c in (1, 2, 3):
      _ = f.get_concrete_function(
          ragged_tensor.RaggedTensorSpec([None, None], dtype=dtypes.int32),
          c)

    obj = tracking.AutoTrackable()
    obj.f = f

    imported1 = cycle(obj, cycles, signatures={})
    rt = ragged_factory_ops.constant([[1, 2], [3]])
    self.assertAllEqual(imported1.f(rt), [[2, 3], [4]])
    self.assertAllEqual(imported1.f(rt, 2), [[3, 4], [5]])
    self.assertAllEqual(imported1.f(rt, 3), [[4, 5], [6]])

    imported2 = cycle(obj, cycles)
    rt = ragged_factory_ops.constant([[1, 2], [3]])
    self.assertAllEqual(imported2.f(rt, 1), [[2, 3], [4]])
    self.assertAllEqual(imported2.f(rt, 2), [[3, 4], [5]])
    self.assertAllEqual(imported2.f(rt, 3), [[4, 5], [6]])

  def test_accepts_io_device(self, cycles):
    options = load_options.LoadOptions()
    self.assertIsNone(options.experimental_io_device)
    options = load_options.LoadOptions(experimental_io_device="/job:localhost")
    self.assertEqual("/job:localhost", options.experimental_io_device)

  def test_load_custom_saveable_object(self, cycles):
    root = tracking.AutoTrackable()
    root.table = lookup_ops.MutableHashTable(dtypes.string, dtypes.float32, -1)
    root.table.insert("foo", 15)
    root.table2 = lookup_ops.MutableHashTable(dtypes.string, dtypes.float32, -1)
    root.table2.insert("idk", 21)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.string)])
    def lookup(key):
      return root.table.lookup(key)

    root.lookup = lookup

    imported = cycle(root, cycles)
    self.assertEqual(self.evaluate(imported.lookup("foo")), 15)
    self.assertEqual(self.evaluate(imported.lookup("idk")), -1)

  def test_load_resource_with_dependency(self, cycles):
    # Test with StaticHashTable, which has a _initializer attribute that tracks
    # the Asset vocab table.

    class MyLookupModel(tracking.AutoTrackable):

      def __init__(self, vocab_file):

        vocab_initializer = lookup_ops.TextFileInitializer(
            vocab_file,
            key_dtype=dtypes.string,
            key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
            value_dtype=dtypes.int64,
            value_index=lookup_ops.TextFileIndex.LINE_NUMBER)
        self._vocab_table = lookup_ops.StaticHashTable(vocab_initializer,
                                                       default_value=-1)

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec((None,), dtypes.string)])
      def __call__(self, inputs):
        return self._vocab_table.lookup(inputs)

    vocab_file = self._make_asset("\n".join(["a", "b", "c", "d"]))
    root = MyLookupModel(vocab_file)
    imported = cycle(root, cycles)
    file_io.delete_file(vocab_file)
    self.assertAllEqual(imported(constant_op.constant(["d", "b"])),
                        [3, 1])


class SingleCycleTests(test.TestCase, parameterized.TestCase):

  def test_load_with_tags(self):
    root = tracking.AutoTrackable()
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)
    with self.assertRaises(ValueError):
      load.load(path, tags=[tag_constants.EVAL])
    load.load(path, tags=[tag_constants.SERVING])
    load.load(path, tags=tag_constants.SERVING)
    load.load(path, tags=set([tag_constants.SERVING]))

  def test_single_restore_op_used(self):
    root = module.Module()
    root.v1 = variables.Variable(1.)
    root.v2 = variables.Variable(2.)
    root.v3 = variables.Variable(3.)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)
    restore_count = 0

    def _count_restores(op_type, *unused_args, **unused_kwargs):
      nonlocal restore_count
      if op_type == b"RestoreV2":
        restore_count += 1

    op_callbacks.add_op_callback(_count_restores)
    load.load(path)
    op_callbacks.remove_op_callback(_count_restores)
    self.assertEqual(1, restore_count)

  def test_docstring_examples(self):
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    exported = util.Checkpoint(v=variables.Variable(3.))
    exported.f = def_function.function(
        lambda x: exported.v * x,
        input_signature=[
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32)])
    save.save(exported, path)
    imported = load.load(path)
    self.assertEqual(3., imported.v.numpy())
    self.assertEqual(6., imported.f(x=constant_op.constant(2.)).numpy())

    save.save(exported, path, exported.f.get_concrete_function())
    imported = load.load(path)
    f = imported.signatures["serving_default"]
    self.assertAllEqual(
        [[-3.]],
        f(x=constant_op.constant([[-1.]]))["output_0"].numpy())


  def test_object_with_extra_dependencies(self):

    class Extra(tracking.AutoTrackable):

      def _list_extra_dependencies_for_serialization(self, cache):
        if self not in cache:
          cache[self] = {"a": variables.Variable(5.)}
        return cache[self]
    root = Extra()
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)
    imported = load.load(path)
    self.assertEqual(5, self.evaluate(imported.a))

    root.a = variables.Variable(3.)
    with self.assertRaisesRegex(
        ValueError, "object has an attribute named a, which is reserved."):
      save.save(root, path)

  def test_save_cached_variable(self):
    with ops.Graph().as_default(), session_lib.Session() as session:
      obj = tracking.AutoTrackable()
      obj.v = variables.Variable(2., caching_device=lambda op: op.device)
      obj.w = variables.Variable(3.)
      session.run([obj.v.initializer, obj.w.initializer])

      @def_function.function
      def total():
        return obj.v + obj.w

      @def_function.function(input_signature=[tensor_spec.TensorSpec([])])
      def wrapped_total(x):
        return total() + x

      @def_function.function
      def increment_v(x):
        obj.v.assign_add(x)

      session.run(increment_v(constant_op.constant(3.)))  # generate signatures
      self.assertAllClose(8, total())
      self.assertAllClose(13, wrapped_total(constant_op.constant(5.)))

      obj.total = total
      obj.wrapped_total = wrapped_total.get_concrete_function()
      obj.increment_v = increment_v

      save_dir = os.path.join(self.get_temp_dir(), "saved_model")
      save.save(obj, save_dir, signatures=total.get_concrete_function())
      imported = load.load(save_dir)
      session.run(variables.global_variables_initializer())
      self.assertAllClose(8, imported.total())
      session.run(imported.increment_v(4))
      self.assertAllClose(12, imported.total())
      self.assertAllClose(15, imported.wrapped_total(constant_op.constant(3.)))
      self.assertAllClose({"output_0": 12},
                          imported.signatures["serving_default"]())

    # Try loading and running the function in eager mode
    imported = load.load(save_dir)
    self.assertAllClose(8, imported.total())
    imported.increment_v(5)
    self.assertAllClose(13, imported.total())
    self.assertAllClose(13.5, imported.wrapped_total(constant_op.constant(.5)))
    self.assertAllClose({"output_0": 13},
                        imported.signatures["serving_default"]())

  # TODO(allenl, kkb): Use the new memory checker here once it's fast enough (3
  # iterations took hundreds of seconds). It would be really nice to check
  # allocations at a lower level.
  @test_util.assert_no_new_pyobjects_executing_eagerly
  def test_functions_cleaned(self):
    if sys.version_info.major < 3:
      self.skipTest("Not working in Python 2")
    root = module.Module()
    root.v = variables.Variable(1.)
    root.f = def_function.function(
        lambda x: x + root.v,
        input_signature=[
            tensor_spec.TensorSpec(shape=[], dtype=dtypes.float32)])
    cycle(root, 1)

  def test_load_partial_object(self):
    root = module.Module()
    root.variables_holder = module.Module()
    root.variables_holder.v = variables.Variable(1.)

    class Adder(module.Module):

      @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=[])])
      def __call__(self, y):
        root.variables_holder.v.assign_add(y)
        return 1

    root.adder = Adder()

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)

    imported = load.load_partial(save_dir,
                                 ["root.variables_holder.v", "root.adder"])
    v = imported["root.variables_holder.v"]
    adder = imported["root.adder"]
    self.assertEqual(self.evaluate(v), 1)
    adder(5)
    self.assertEqual(self.evaluate(v), 6)

    with self.assertRaisesRegex(ValueError, "requires inputs/variables"):
      imported = load.load_partial(save_dir, ["root.adder"])

  def test_call_untraced_function_raises_error(self):

    class ObjWithFunction(module.Module):

      @def_function.function
      def foo(self, a):
        return a

    root = ObjWithFunction()
    with self.assertLogs(level="WARNING") as logs:
      loaded = cycle(root, 1)

    expected_save_message = (
        "WARNING:absl:Found untraced functions such as foo while saving "
        "(showing 1 of 1). These functions will not be directly callable after "
        "loading.")
    self.assertIn(expected_save_message, logs.output)

    with self.assertRaisesRegex(
        ValueError, "Found zero restored functions for caller function."):
      loaded.foo(1)

  def test_restored_function_execute_eagerly(self):
    try:
      def_function.run_functions_eagerly(True)

      class MyModel(module.Module):

        @def_function.function
        def __call__(self, inputs, training=False):
          return math_ops.multiply(0.5, inputs)

      model = MyModel()
      model.__call__.get_concrete_function(
          tensor_spec.TensorSpec([None], dtypes.float32))
      loaded = cycle(model, 1)

      # Calling the function should not throw an exception.
      loaded(constant_op.constant([1.0]))

    finally:
      def_function.run_functions_eagerly(False)

  def test_restored_model_concrete_function_is_deterministic(self):
    previous_concrete_function = None
    for _ in range(100):

      class MyModel(module.Module):

        @def_function.function
        def __call__(self, x):
          return x * constant_op.constant(3.0)

      model = MyModel()
      model(array_ops.ones((7, 3), dtype=dtypes.float32))
      model.__call__.get_concrete_function(
          tensor_spec.TensorSpec([None, 3], dtypes.float32))
      loaded = cycle(model, 1)

      # Ensure the newly loaded concrete function is the same as the previous
      # after a cycle of serialization / deserialization.
      new_concrete_function = loaded.__call__.get_concrete_function(
          tensor_spec.TensorSpec([None, 3], dtypes.float32))
      if previous_concrete_function is not None:
        self.assertEqual(previous_concrete_function.pretty_printed_signature(),
                         new_concrete_function.pretty_printed_signature())

      previous_concrete_function = new_concrete_function

  def test_garbage_collection_capturable_resource_doesnt_raise_exception(self):
    model = module.Module()
    model.mapping = lookup_ops.StaticHashTable(
        lookup_ops.KeyValueTensorInitializer(
            keys=math_ops.range(1, dtype=dtypes.int32),
            values=["foo"]),
        "default_value")
    loaded = cycle(model, 1)
    del model
    del loaded
    # Exceptions raised during garbage collection are simply printed to stderr
    # and ignored, and we have no way to access them. We'll capture stdout
    # during the garbage collection process and inspect to see if any
    # exceptions were raised.
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
      gc.collect()
    if "Exception ignored in" in stderr.getvalue():
      raise Exception(stderr.getvalue())


if __name__ == "__main__":
  test.main()
