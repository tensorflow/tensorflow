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
"""Tests for checkpointable object SavedModel loading."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tempfile

from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.training.checkpointable import tracking


@parameterized.named_parameters(
    dict(testcase_name="ReloadOnce", cycles=1),
    dict(testcase_name="ReloadTwice", cycles=2),
    dict(testcase_name="ReloadThrice", cycles=3))
class LoadTest(test.TestCase, parameterized.TestCase):

  def cycle(self, obj, cycles=1, signatures=None):
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
    root = tracking.AutoCheckpointable()
    root.dep_one = tracking.AutoCheckpointable()
    root.dep_two = tracking.AutoCheckpointable()
    root.dep_two.dep = tracking.AutoCheckpointable()
    root.dep_three = root.dep_two.dep
    imported = self.cycle(root, cycles)
    self.assertIs(imported.dep_three, imported.dep_two.dep)
    self.assertIsNot(imported.dep_one, imported.dep_two)

  def test_variables(self, cycles):
    root = tracking.AutoCheckpointable()
    root.v1 = variables.Variable(1., trainable=True)
    root.v2 = variables.Variable(2., trainable=False)
    imported = self.cycle(root, cycles)
    self.assertEqual(imported.v1.numpy(), 1.0)
    self.assertTrue(imported.v1.trainable)
    self.assertEqual(imported.v2.numpy(), 2.0)
    self.assertFalse(imported.v2.trainable)

  def test_capture_variables(self, cycles):
    root = tracking.AutoCheckpointable()
    root.weights = variables.Variable(2.)
    root.f = def_function.function(
        lambda x: root.weights * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    imported = self.cycle(root, cycles)
    self.assertEqual(4., imported.f(constant_op.constant(2.)).numpy())
    imported.weights.assign(4.0)
    self.assertEqual(8., imported.f(constant_op.constant(2.)).numpy())

  def _make_asset(self, contents):
    filename = tempfile.mktemp(prefix=self.get_temp_dir())
    with open(filename, "w") as f:
      f.write(contents)
    return filename

  def test_assets(self, cycles):
    file1 = self._make_asset("contents 1")
    file2 = self._make_asset("contents 2")

    root = tracking.AutoCheckpointable()
    root.asset1 = tracking.TrackableAsset(file1)
    root.asset2 = tracking.TrackableAsset(file2)

    save_dir = os.path.join(self.get_temp_dir(), "save_dir")
    save.save(root, save_dir)

    file_io.delete_file(file1)
    file_io.delete_file(file2)
    load_dir = os.path.join(self.get_temp_dir(), "load_dir")
    file_io.rename(save_dir, load_dir)

    imported = load.load(load_dir)
    with open(imported.asset1.asset_path.numpy(), "r") as f:
      self.assertEqual("contents 1", f.read())
    with open(imported.asset2.asset_path.numpy(), "r") as f:
      self.assertEqual("contents 2", f.read())

  def test_capture_assets(self, cycles):
    root = tracking.AutoCheckpointable()
    root.vocab = tracking.TrackableAsset(self._make_asset("contents"))
    root.f = def_function.function(
        lambda: root.vocab.asset_path,
        input_signature=[])
    imported = self.cycle(root, cycles)
    origin_output = root.f().numpy()
    imported_output = imported.f().numpy()
    self.assertNotEqual(origin_output, imported_output)
    with open(imported_output, "r") as f:
      self.assertEqual("contents", f.read())

  def test_dedup_assets(self, cycles):
    vocab = self._make_asset("contents")
    root = tracking.AutoCheckpointable()
    root.asset1 = tracking.TrackableAsset(vocab)
    root.asset2 = tracking.TrackableAsset(vocab)
    imported = self.cycle(root, cycles)
    self.assertEqual(imported.asset1.asset_path.numpy(),
                     imported.asset2.asset_path.numpy())

  def test_implicit_input_signature(self, cycles):
    @def_function.function
    def func(x):
      return 2 * x

    root = tracking.AutoCheckpointable()
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

    root = tracking.AutoCheckpointable()
    root.f = func

    imported = self.cycle(root, cycles)
    self.assertEqual(4., imported.f(constant_op.constant(2.0)).numpy())

  def test_explicit_save_signature(self, cycles):
    @def_function.function
    def func(x):
      return 2 * x

    root = tracking.AutoCheckpointable()
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

    root = tracking.AutoCheckpointable()
    root.g = g
    imported = self.cycle(root, cycles)
    imported.g(constant_op.constant([1.0]))

  def test_function_with_default_bool_input(self, cycles):

    def func(x, training=False):
      if training:
        return 2 * x
      else:
        return 7

    root = tracking.AutoCheckpointable()
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

    root = tracking.AutoCheckpointable()
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

  def test_structured_inputs(self, cycles):

    def func(x, training=True):
      # x is a nested structure, we care about one particular tensor.
      _, (a, b) = x
      if training:
        return 2 * a["a"] + b
      else:
        return 7

    root = tracking.AutoCheckpointable()
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

    with self.assertRaisesRegexp(AssertionError,
                                 "Could not find matching function to call.*"):
      imported.f(input2)

    self.assertEqual(31, imported.f(input1).numpy())
    self.assertEqual(32, imported.f(input3).numpy())

  def test_structured_output(self, cycles):

    # Use fields with non-alphabetical order
    named_tuple_type = collections.namedtuple("NamedTupleHello", ["b", "a"])

    def func(input1, input2):
      named_tuple = named_tuple_type(a=input1 + input2, b=input1 * input2)
      return [named_tuple, input2, {"x": 0.5}]

    root = tracking.AutoCheckpointable()
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

  def test_positional_arguments(self, cycles):
    def func(x, training=False, abc=7.1, defg=7.7):
      del abc
      if training:
        return 2 * x
      if defg == 7:
        return 6
      else:
        return 7

    root = tracking.AutoCheckpointable()
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

    root = tracking.AutoCheckpointable()
    root.f = def_function.function(func)

    x = constant_op.constant(10)
    self.assertEqual(7, root.f(x, learning_rate=0.5, epochs=3).numpy())

    imported = self.cycle(root, cycles)

    with self.assertRaisesRegexp(AssertionError,
                                 "Could not find matching function to call.*"):
      imported.f(x, learning_rate=0.5, epochs=4)

    self.assertEqual(7, imported.f(x, learning_rate=0.5, epochs=3).numpy())

  def test_member_function(self, cycles):
    class CheckpointableWithMember(tracking.AutoCheckpointable):

      def __init__(self):
        super(CheckpointableWithMember, self).__init__()
        self._some_value = 20

      @def_function.function
      def f(self, x, training=False):
        if training:
          return 2 * x
        else:
          return 7 + self._some_value

    root = CheckpointableWithMember()

    self.assertEqual(20, root.f(constant_op.constant(10), True).numpy())
    self.assertEqual(27, root.f(constant_op.constant(1)).numpy())
    self.assertEqual(2, root.f(constant_op.constant(1), True).numpy())

    imported = self.cycle(root, cycles)

    self.assertEqual(4, imported.f(constant_op.constant(2), True).numpy())
    self.assertEqual(27, imported.f(constant_op.constant(2)).numpy())

  def test_side_effect_listing(self, cycles):
    class M(tracking.AutoCheckpointable):

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
    self.cycle(m)
    self.assertEqual(4.0, m.f(constant_op.constant(2.0)).numpy())

  def test_basic_backprop(self, cycles):
    weight = variables.Variable(1., trainable=True)
    bias = variables.Variable(0., trainable=True)
    g = def_function.function(
        lambda x: x*weight + bias,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])

    root = tracking.AutoCheckpointable()
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

    root = tracking.AutoCheckpointable()
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
    class M1(tracking.AutoCheckpointable):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
      def __call__(self, x):
        return x

    root = tracking.AutoCheckpointable()
    root.m1 = M1()
    root.m2 = tracking.AutoCheckpointable()
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
    root = tracking.AutoCheckpointable()
    root.__call__ = tracking.AutoCheckpointable()
    root.__call__.__call__ = tracking.AutoCheckpointable()
    root.__call__.__call__.__call__ = func

    imported = self.cycle(root, cycles)
    self.assertTrue(callable(imported))
    x = constant_op.constant(1.0)
    self.assertAllEqual(imported(x).numpy(), 3.0)

  def test_soft_matching(self, cycles):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)])
    def func(x):
      return 2 * x

    root = tracking.AutoCheckpointable()
    root.f = func

    self.assertAllEqual([2], root.f(constant_op.constant([1])).numpy())
    self.assertAllEqual([2, 4], root.f(constant_op.constant([1, 2])).numpy())

    concrete_functions = root.f._list_all_concrete_functions_for_serialization()  # pylint: disable=protected-access
    self.assertEqual(1, len(concrete_functions))

    imported = self.cycle(root, cycles)

    with self.assertRaisesRegexp(ValueError, "Cannot canonicalize"):
      # We cannot call the function with a constant of shape ().
      self.assertEqual(7, imported.f(constant_op.constant(2)).numpy())

    # TODO(vbardiovsky): When classes are revived with input_signatures, we
    # should also check that the calls below are not generating any more
    # concrete functions.
    self.assertAllEqual([2, 4, 6, 8],
                        imported.f(constant_op.constant([1, 2, 3, 4])).numpy())
    self.assertAllEqual([2, 4, 6],
                        imported.f(constant_op.constant([1, 2, 3])).numpy())

  def test_concrete_function(self, cycles):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)])
    def func(x):
      return 2 * x

    root = tracking.AutoCheckpointable()
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

    root = tracking.AutoCheckpointable()
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

    root = tracking.AutoCheckpointable()
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
    root = tracking.AutoCheckpointable()
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
    root = tracking.AutoCheckpointable()
    root.f = func.get_concrete_function(
        tensor_spec.TensorSpec([], dtypes.float32),
        tensor_spec.TensorSpec([], dtypes.float32))
    self.assertEqual(8., root.f(y=constant_op.constant(3.),
                                x=constant_op.constant(2.)).numpy())
    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = self.cycle(root, cycles, signatures={})
    self.assertEqual(8., imported.f(y=constant_op.constant(3.),
                                    x=constant_op.constant(2.)).numpy())

  def test_dict(self, cycles):
    root = tracking.AutoCheckpointable()
    root.variables = dict(a=variables.Variable(1.))
    root.variables["b"] = variables.Variable(2.)
    root.variables["c"] = 1
    imported = self.cycle(root, cycles)
    self.assertEqual(1., imported.variables["a"].numpy())
    self.assertEqual(2., imported.variables["b"].numpy())
    self.assertEqual(set(["a", "b"]), set(imported.variables.keys()))

  def test_list(self, cycles):
    root = tracking.AutoCheckpointable()
    root.variables = [variables.Variable(1.)]
    root.variables.append(1)
    root.variables.append(variables.Variable(3.))
    imported = self.cycle(root, cycles)
    self.assertEqual(1., imported.variables[0].numpy())
    self.assertEqual(3., imported.variables[2].numpy())
    self.assertIs(None, imported.variables[1])
    self.assertEqual(3, len(imported.variables))


if __name__ == "__main__":
  test.main()
