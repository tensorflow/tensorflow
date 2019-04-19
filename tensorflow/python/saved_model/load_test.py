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
import functools
import os
import tempfile
import weakref

from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import monitored_session
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util
from tensorflow.python.util import tf_inspect


@parameterized.named_parameters(
    dict(testcase_name="ReloadOnce", cycles=1),
    dict(testcase_name="ReloadTwice", cycles=2),
    dict(testcase_name="ReloadThrice", cycles=3))
class LoadTest(test.TestCase, parameterized.TestCase):

  def cycle(self, obj, cycles, signatures=None):
    to_save = obj
    # TODO(vbardiovsky): It would be nice if exported protos reached a fixed
    # point w.r.t. saving/restoring, ideally after 2nd saving.
    for _ in range(cycles):
      path = tempfile.mkdtemp(prefix=self.get_temp_dir())
      save.save(to_save, path, signatures)
      loaded = load.load(path)
      to_save = loaded
    return loaded

  def test_structure_import(self, cycles):
    root = tracking.AutoTrackable()
    root.dep_one = tracking.AutoTrackable()
    root.dep_two = tracking.AutoTrackable()
    root.dep_two.dep = tracking.AutoTrackable()
    root.dep_three = root.dep_two.dep
    imported = self.cycle(root, cycles)
    self.assertIs(imported.dep_three, imported.dep_two.dep)
    self.assertIsNot(imported.dep_one, imported.dep_two)

  def test_variables(self, cycles):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(1., trainable=True)
    root.v2 = variables.Variable(2., trainable=False)
    imported = self.cycle(root, cycles)
    self.assertEqual(imported.v1.numpy(), 1.0)
    self.assertTrue(imported.v1.trainable)
    self.assertEqual(imported.v2.numpy(), 2.0)
    self.assertFalse(imported.v2.trainable)

  @test_util.run_in_graph_and_eager_modes
  def test_capture_variables(self, cycles):
    root = tracking.AutoTrackable()
    root.weights = variables.Variable(2.)
    self.evaluate(root.weights.initializer)
    root.f = def_function.function(
        lambda x: root.weights * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    for _ in range(cycles):
      imported = self.cycle(root, 1)
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
    imported = self.cycle(root, cycles)
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

    imported = self.cycle(exported, cycles)
    # Calling get_concrete_function wraps in a second call operation; we want to
    # inspect the original function body for the control output; digging into
    # graph.as_graph_def() and its FunctionDefLibrary is another option.
    imported_concrete, = imported.f._concrete_functions
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
    root.asset1 = tracking.TrackableAsset(file1)
    root.asset2 = tracking.TrackableAsset(file2)

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

  def test_capture_assets(self, cycles):
    root = tracking.AutoTrackable()
    root.vocab = tracking.TrackableAsset(self._make_asset("contents"))
    root.f = def_function.function(
        lambda: root.vocab.asset_path,
        input_signature=[])
    imported = self.cycle(root, cycles)
    original_output = root.f().numpy()
    imported_output = imported.f().numpy()
    self.assertNotEqual(original_output, imported_output)
    with open(imported_output, "r") as f:
      self.assertEqual("contents", f.read())

  def test_capture_assets_in_graph(self, cycles):
    root = tracking.AutoTrackable()
    root.vocab = tracking.TrackableAsset(self._make_asset("contents"))
    root.f = def_function.function(
        lambda: root.vocab.asset_path,
        input_signature=[])

    original_output = root.f().numpy()

    if cycles > 1:
      root = self.cycle(root, cycles - 1)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)

    with ops.Graph().as_default():
      imported = load.load(path)
      imported_tensor = imported.f()
      with monitored_session.MonitoredSession() as sess:
        imported_output = sess.run(imported_tensor)
        self.assertNotEqual(original_output, imported_output)
        with open(imported_output, "r") as f:
          self.assertEqual("contents", f.read())

  def test_dedup_assets(self, cycles):
    vocab = self._make_asset("contents")
    root = tracking.AutoTrackable()
    root.asset1 = tracking.TrackableAsset(vocab)
    root.asset2 = tracking.TrackableAsset(vocab)
    imported = self.cycle(root, cycles)
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

    imported = self.cycle(root, cycles)

    self.assertEqual(4., imported.f(constant_op.constant(2.)).numpy())
    self.assertEqual(14, imported.f(constant_op.constant(7)).numpy())

  def test_explicit_input_signature(self, cycles):
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func

    imported = self.cycle(root, cycles)
    self.assertEqual(4., imported.f(constant_op.constant(2.0)).numpy())

  def test_explicit_save_signature(self, cycles):
    @def_function.function
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func

    imported = self.cycle(
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
    imported = self.cycle(root, cycles)
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

    imported = self.cycle(root, cycles)

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
    self.assertEqual(4, len(concrete_functions))

    imported = self.cycle(root, cycles)

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

    imported = self.cycle(obj, cycles)

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

    imported = self.cycle(root, cycles)

    with self.assertRaisesRegexp(ValueError,
                                 "Could not find matching function to call"):
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

    imported = self.cycle(root, cycles)

    result = imported.f(constant_op.constant(2), constant_op.constant(5))
    self.assertEqual(7, result[0].a.numpy())
    self.assertEqual(10, result[0].b.numpy())
    self.assertEqual(["b", "a"], list(result[0]._asdict().keys()))
    self.assertEqual(5, result[1].numpy())
    self.assertEqual(0.5, result[2]["x"].numpy())

  def test_optimizer(self, cycles):

    class _HasOptimizer(module.Module):

      def __init__(self):
        super(_HasOptimizer, self).__init__()
        self.layer = core.Dense(1)
        self.optimizer = adam.Adam(0.01)

      @def_function.function
      def __call__(self, x):
        return self.layer(x)

      @def_function.function
      def train(self, x, y):
        with backprop.GradientTape() as tape:
          predicted = self(x)
          loss = math_ops.reduce_sum(math_ops.abs(y - predicted))
        train_vars = self.layer.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

    root = _HasOptimizer()
    train_input = dict(x=constant_op.constant([[1.]]),
                       y=constant_op.constant([[2.]]))
    root.train(**train_input)
    imported = self.cycle(root, cycles)
    self.assertAllClose(root.optimizer.learning_rate.numpy(),
                        imported.optimizer.learning_rate.numpy())
    self.assertAllClose(root(constant_op.constant([[-0.5]])),
                        imported(constant_op.constant([[-0.5]])))
    root.train(**train_input)
    imported.train(**train_input)
    self.assertAllClose(root(constant_op.constant([[-0.5]])),
                        imported(constant_op.constant([[-0.5]])))

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

    imported = self.cycle(root, cycles)

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

    imported = self.cycle(root, cycles)

    with self.assertRaisesRegexp(ValueError,
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

    imported = self.cycle(root, cycles)

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
    self.cycle(m, cycles)
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
    imported = self.cycle(root, cycles)
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

    imported = self.cycle(root, cycles)
    with backprop.GradientTape() as t:
      x = constant_op.constant([3.5])
      loss = imported.g(x)
    grad = t.gradient(loss, [imported.weight, imported.bias])
    self.assertAllClose(grad, [3.5, 2.0])

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
    imported = self.cycle(root, cycles)
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

    imported = self.cycle(root, cycles)
    self.assertTrue(callable(imported))
    x = constant_op.constant(1.0)
    self.assertAllEqual(imported(x).numpy(), 3.0)

  def test_load_in_graph_mode(self, cycles):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(1.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(
        lambda x: root.v2 * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])

    if cycles > 1:
      root = self.cycle(root, cycles - 1)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)

    with ops.Graph().as_default():
      imported = load.load(path)
      var_v1 = imported.v1
      output = imported.f(constant_op.constant(2.))
      with monitored_session.MonitoredSession() as sess:
        self.assertEqual(1.0, sess.run(var_v1))
        self.assertEqual(4.0, sess.run(output))

  def test_load_in_func_graph(self, cycles):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(1.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(
        lambda x: root.v2 * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])

    if cycles > 1:
      root = self.cycle(root, cycles - 1)
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
    self.assertEqual(1, len(concrete_functions))

    imported = self.cycle(root, cycles)

    with self.assertRaisesRegexp(ValueError, "Python inputs incompatible"):
      # We cannot call the function with a constant of shape ().
      imported.f(constant_op.constant(2)).numpy()

    # TODO(vbardiovsky): When classes are revived with input_signatures, we
    # should also check that the calls below are not generating any more
    # concrete functions.
    self.assertAllEqual([2, 4, 6, 8],
                        imported.f(constant_op.constant([1, 2, 3, 4])).numpy())
    self.assertAllEqual([2, 4, 6],
                        imported.f(constant_op.constant([1, 2, 3])).numpy())

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

    imported = self.cycle(root, cycles)

    concrete = imported.f.get_concrete_function(
        training=True, x=tensor_spec.TensorSpec([None], dtypes.int32))

    self.assertAllEqual([2, 4, 6, 8],
                        concrete(x=constant_op.constant([1, 2, 3, 4])).numpy())
    with self.assertRaisesRegexp(ValueError,
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
    imported = self.cycle(root, cycles, signatures={})

    self.assertAllEqual([2, 4, 6, 8],
                        imported.f(constant_op.constant([1, 2, 3, 4])).numpy())
    self.assertAllEqual([2, 4, 6],
                        imported.f(constant_op.constant([1, 2, 3])).numpy())

  def test_concrete_function_arg_names(self, cycles):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)])
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function()

    self.assertAllEqual([2], root.f(constant_op.constant([1])).numpy())

    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = self.cycle(root, cycles, signatures={})

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
    imported = self.cycle(root, cycles, signatures={})
    self.assertAllEqual([6],
                        imported.f(constant_op.constant([3])).numpy())

  def test_concrete_function_backprop(self, cycles):
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.float32)])
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

    self.assertEqual(2., _compute_gradient(root.f).numpy())
    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = self.cycle(root, cycles, signatures={})
    self.assertEqual(2., _compute_gradient(imported.f).numpy())

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
    imported = self.cycle(root, cycles, signatures={})
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
    imported = self.cycle(root, cycles, signatures={})
    self.assertEqual(8., imported.f(y=constant_op.constant(3.),
                                    x=constant_op.constant(2.)).numpy())

  def test_concrete_function_variable_argument(self, cycles):
    # TODO(allenl): Fix variables in input signatures.
    self.skipTest("Need to fix encoding of variables in inputs signatures")
    capture = variables.Variable(0)

    @def_function.function
    def func(v):
      v.assign_add(1)
      capture.assign_sub(1)

    vsave = variables.Variable(1)
    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function(vsave)
    root.capture = capture
    self.assertEqual(1, vsave.numpy())
    root.f(vsave)
    self.assertEqual(2, vsave.numpy())
    self.assertEqual(-1, capture.numpy())
    imported = self.cycle(root, cycles)

    vload = variables.Variable(1)
    imported.f(vload)
    self.assertEqual(2, vload.numpy())
    imported.f(v=vload)
    self.assertEqual(3, vload.numpy())
    self.assertEqual(-3, imported.capture.numpy())
    self.assertEqual(-1, capture.numpy())

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
    imported = self.cycle(root, cycles)
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
    imported = self.cycle(root, cycles)
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
    imported = self.cycle(root, cycles)
    self.assertEqual(1., imported.variables[0].numpy())
    self.assertEqual(3., imported.variables[2].numpy())
    self.assertIs(None, imported.variables[1])
    self.assertEqual(3, len(imported.variables))

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
    imported = self.cycle(root, cycles)
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
    imported = self.cycle(root, cycles)
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
    imported = self.cycle(exported, cycles)
    self.assertEqual(0, imported.make_func().numpy())
    self.assertEqual(1, exported.make_func().numpy())

  def test_overwritten_signatures_error(self, cycles):
    exported = tracking.AutoTrackable()
    exported.f = def_function.function(lambda: constant_op.constant(1.))
    imported = self.cycle(
        exported, cycles,
        signatures={"key": exported.f.get_concrete_function()})
    self.assertEqual(1., imported.signatures["key"]()["output_0"].numpy())
    imported.signatures = {"key1": imported.signatures["key"]}
    with self.assertRaisesRegexp(ValueError, "signatures"):
      save.save(imported, tempfile.mkdtemp(prefix=self.get_temp_dir()))

  def test_signature_loading(self, cycles):

    class Exported(tracking.AutoTrackable):

      def __init__(self):
        self.v = variables.Variable(3.)

      @def_function.function
      def do(self, x):
        return self.v * x

    exported = Exported()
    imported = self.cycle(
        exported,
        cycles=1,
        signatures=exported.do.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.float32)))
    for _ in range(cycles - 1):
      imported = self.cycle(imported, cycles=1, signatures=imported.signatures)
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

  def test_multiple_argument_signatures_no_positional(self, cycles):

    class Exported(tracking.AutoTrackable):

      @def_function.function
      def do(self, x, y):
        return x + y

    exported = Exported()
    imported = self.cycle(
        exported, cycles=1, signatures=exported.do.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.float32),
            tensor_spec.TensorSpec(None, dtypes.float32)))
    for _ in range(cycles - 1):
      imported = self.cycle(imported, cycles=1, signatures=imported.signatures)
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
    imported = self.cycle(root, cycles, signatures={})
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
    self.cycle(root, 1)
    original_collections = _gather_nonempty_collections()
    self.cycle(root, cycles)
    self.assertEqual(original_collections, _gather_nonempty_collections())

  def test_table_in_graph(self, cycles):
    root = self._make_model_with_tables()

    if cycles > 1:
      root = self.cycle(root, cycles - 1)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)
    imported = self.cycle(root, 1)

    with ops.Graph().as_default():
      imported = load.load(path)
      keys = constant_op.constant(["brain", "test", "foo", "surgery"])
      output1 = imported.lookup1(keys)
      output2 = imported.lookup2(keys)
      with monitored_session.MonitoredSession() as sess:
        self.assertAllEqual([0, -1, -1, 2], sess.run(output1))
        self.assertAllEqual([2, 0, 1, -1], sess.run(output2))

  def test_perserve_argspec(self, cycles):
    def f(a, b, c):  # pylint: disable=unused-argument
      return None

    original_fullargspec = tf_inspect.getfullargspec(f)

    root = tracking.AutoTrackable()
    root.f = def_function.function(f)
    imported = self.cycle(root, cycles)

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
    root = self.cycle(root, cycles)
    self.assertAllEqual(root.f(), [1.0, 2.0, 3.0, True])
    self.assertAllEqual(root.f(-1.0, training=False), [3.0, 2.0, -1.0, False])

    with self.assertRaisesRegexp(ValueError,
                                 "Could not find matching function"):
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
    root = self.cycle(root, cycles)
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

    root = self.cycle(root, cycles)
    self.assertAllEqual(root.f(), [1.0])

  def test_partial_with_non_tensor_defaults(self, cycles):

    def f(x, y=3):
      return x + y

    func = def_function.function(functools.partial(f, y=5))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(1), 6)

    root = self.cycle(root, cycles)
    self.assertAllEqual(root.f(1), 6)

  def test_partial_with_positional(self, cycles):
    def f(x, y):
      return x + y

    func = def_function.function(functools.partial(f, constant_op.constant(5)))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(1), 6)

    root = self.cycle(root, cycles)
    self.assertAllEqual(root.f(1), 6)

  def test_partial_with_positional_captured_tensors(self, cycles):

    def f(x, y):
      return x + y

    tensor = constant_op.constant(5) + constant_op.constant(7)
    func = def_function.function(functools.partial(f, tensor))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(1), 13)

    root = self.cycle(root, cycles)
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

    root = self.cycle(root, cycles)
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

    root = self.cycle(root, cycles)
    self.assertEqual(root.f(constant_op.constant(5)).numpy(), 45)

  def test_partial_with_passed_fn_as_default(self, cycles):

    def f(x, y):
      return x(3) + y

    def my_func(a):
      return 2 * a

    func = def_function.function(functools.partial(f, my_func))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertEqual(root.f(constant_op.constant(3)).numpy(), 9)

    root = self.cycle(root, cycles)
    self.assertEqual(root.f(constant_op.constant(3)).numpy(), 9)

  def test_convert_to_input_signature(self, cycles):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)])
    def func(x):
      return x

    root = tracking.AutoTrackable()
    root.f = func

    root = self.cycle(root, cycles)

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
    imported = self.cycle(obj, cycles)
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
    imported = self.cycle(obj, cycles)

    self.assertEqual(4.0, imported({"a": 3.0}).numpy())

    with self.assertRaisesRegexp(ValueError,
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

    root = self.cycle(root, cycles)

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
    root = self.cycle(root, cycles)
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
    root = util.Checkpoint(v=v)
    root = self.cycle(root, cycles)
    self.assertEqual(False, root.v.trainable)
    self.assertEqual(variables.VariableSynchronization.NONE,
                     root.v.synchronization)
    self.assertEqual(variables.VariableAggregation.ONLY_FIRST_REPLICA,
                     root.v.aggregation)

  def test_dense_features_layer(self, cycles):
    columns = [feature_column_v2.numeric_column("x"),
               feature_column_v2.numeric_column("y")]
    layer = feature_column_v2.DenseFeatures(columns)
    model = sequential.Sequential([layer])
    model_input = {"x": constant_op.constant([[1.]]),
                   "y": constant_op.constant([[2.]])}
    self.assertAllClose([[1., 2.]], model.predict(model_input))
    loaded = self.cycle(model, cycles)
    output, = loaded._default_save_signature(model_input).values()
    self.assertAllClose([[1., 2.]], output)
    signature_output, = loaded.signatures["serving_default"](
        **model_input).values()
    self.assertAllClose([[1., 2.]], signature_output)

  def test_dense_features_layer_fit(self, cycles):
    columns = [feature_column_v2.numeric_column("x")]
    model = sequential.Sequential(
        [feature_column_v2.DenseFeatures(columns),
         core.Dense(1)])
    model_input = {"x": constant_op.constant([[1.]])}
    model.compile(optimizer="adam", loss="mse")
    model.fit(model_input, constant_op.constant([[3.]]))
    loaded = self.cycle(model, cycles)
    loaded._default_save_signature(model_input)
    loaded.signatures["serving_default"](**model_input)


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


if __name__ == "__main__":
  test.main()
