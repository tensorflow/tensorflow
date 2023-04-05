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

import collections
import contextlib
import functools
import gc
import io
import os
import pathlib
import sys
import tempfile
import weakref

from absl.testing import parameterized
import numpy as np

# Import for py bindings to runtime
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import config
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
from tensorflow.python.lib.io import tf_record
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import load_options
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import resource
from tensorflow.python.training import monitored_session
from tensorflow.python.util import tf_inspect


def cycle(
    obj,
    cycles,
    signatures=None,
    save_option=None,
    load_option=None,
    use_cpp_bindings=False,
):
  to_save = obj
  # TODO(vbardiovsky): It would be nice if exported protos reached a fixed
  # point w.r.t. saving/restoring, ideally after 2nd saving.
  for _ in range(cycles):
    path = tempfile.mkdtemp(prefix=test.get_temp_dir())
    # If available, we'll run the save and restore preferring the GPU. This
    # just makes sure we aren't throwing errors and have enough
    # device("CPU") blocks to satisfy the placer.
    with test_util.use_gpu():
      save.save(to_save, path, signatures, options=save_option)
      loaded = test_load(
          path, options=load_option, use_cpp_bindings=use_cpp_bindings
      )
      signatures = loaded.signatures
    to_save = loaded
  return loaded


def _test_load_base(path, tags=None, options=None,
                    use_cpp_bindings=False):  # pylint: disable=unused-argument
  return load.load(path, tags=tags, options=options)


def _test_load_internal(path, tags=None, options=None, use_cpp_bindings=False):
  if use_cpp_bindings:
    runtime = runtime_pybind.Runtime()
    return runtime.Import(path)
  return _test_load_base(path, tags=tags, options=options,
                         use_cpp_bindings=use_cpp_bindings)

# replaced by copy.bara.sky
run_external = True


def test_load(path, **kwargs):
  if not run_external:
    return _test_load_internal(path, **kwargs)
  return _test_load_base(path, **kwargs)


def _load_test_params():
  params = [
      dict(testcase_name="ReloadOncePy", cycles=1, use_cpp_bindings=False),
      dict(testcase_name="ReloadTwicePy", cycles=2, use_cpp_bindings=False),
      dict(testcase_name="ReloadThricePy", cycles=3, use_cpp_bindings=False),
  ]
  if not run_external:
    params.append(dict(testcase_name="ReloadOnceCpp", cycles=1,
                       use_cpp_bindings=True))
  return params


def _test_params():
  params = [dict(testcase_name="LoadWithPython", use_cpp_bindings=False)]
  if not run_external:
    params.append(dict(testcase_name="LoadWithCpp", use_cpp_bindings=True))
  return params


@parameterized.named_parameters(*_load_test_params())
class LoadTest(test.TestCase, parameterized.TestCase):

  def test_structure_import(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = autotrackable.AutoTrackable()
    root.dep_one = autotrackable.AutoTrackable()
    root.dep_two = autotrackable.AutoTrackable()
    root.dep_two.dep = autotrackable.AutoTrackable()
    root.dep_three = root.dep_two.dep
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertIs(imported.dep_three, imported.dep_two.dep)
    self.assertIsNot(imported.dep_one, imported.dep_two)

  @test_util.run_in_graph_and_eager_modes
  def test_variables(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(1.0, trainable=True)
    root.v2 = variables.Variable(2.0, trainable=False)
    self.evaluate([root.v1.initializer, root.v2.initializer])

    for _ in range(cycles):
      imported = cycle(root, 1, use_cpp_bindings=use_cpp_bindings)
      self.evaluate([imported.v1.initializer, imported.v2.initializer])

    if not context.executing_eagerly():
      self.assertIsInstance(imported.v1.initializer, ops.Operation)
      self.assertIsInstance(imported.v2.initializer, ops.Operation)

    self.assertEqual(self.evaluate(imported.v1), 1.0)
    self.assertTrue(imported.v1.trainable)
    self.assertEqual(self.evaluate(imported.v2), 2.0)
    self.assertFalse(imported.v2.trainable)

  def test_variables_name(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = autotrackable.AutoTrackable()
    # Test 2 variables with same name: should work as the checkpoint
    # is based on object name and not on variable name.
    root.v1 = variables.Variable(1.0, trainable=True, name="v1")
    root.v2 = variables.Variable(2.0, trainable=False, name="v1")
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(imported.v1.numpy(), 1.0)
    self.assertEqual(imported.v2.numpy(), 2.0)
    self.assertEqual(imported.v1.name, root.v1.name)
    self.assertEqual(imported.v2.name, root.v2.name)
    with variable_scope.variable_scope("foo"):
      imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
      self.assertTrue(imported.v1.name.startswith("foo/"))
      self.assertTrue(imported.v2.name.startswith("foo/"))

  @test_util.disable_xla("This test never passed for XLA")
  def test_partially_defined_variable_shape(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class MakeVariable(module.Module):

      def __init__(self):
        self.v = None

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec([None], dtypes.int64)]
      )
      def make_variable(self, initial_value):
        if self.v is None:
          self.v = variables.Variable(initial_value)

    m = MakeVariable()
    m.make_variable([1, 2, 3])
    m = cycle(m, cycles, use_cpp_bindings=use_cpp_bindings)
    m.v.assign([1, 2, 3, 4])
    self.assertEqual([None], tensor_shape.as_shape(m.v.shape).as_list())

  @test_util.run_in_graph_and_eager_modes
  def test_capture_variables(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = autotrackable.AutoTrackable()
    root.weights = variables.Variable(2.0)
    self.evaluate(root.weights.initializer)
    root.f = def_function.function(
        lambda x: root.weights * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)],
    )
    for _ in range(cycles):
      imported = cycle(root, 1, use_cpp_bindings=use_cpp_bindings)
      self.evaluate(imported.weights.initializer)
    self.assertEqual(4.0, self.evaluate(imported.f(constant_op.constant(2.0))))
    self.evaluate(imported.weights.assign(4.0))
    self.assertEqual(8.0, self.evaluate(imported.f(constant_op.constant(2.0))))

  @test_util.run_in_graph_and_eager_modes
  def test_capture_constant(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = autotrackable.AutoTrackable()
    captured_constant = constant_op.constant(2.0)
    root.f = def_function.function(
        lambda x: captured_constant * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)],
    )
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(4.0, self.evaluate(imported.f(constant_op.constant(2.0))))

  def test_control_outputs(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    exported = autotrackable.AutoTrackable()
    exported.v = variables.Variable(1.0)
    exported.f = def_function.function(
        lambda: exported.v.assign(2.0, name="should_be_control_output")
    )
    exported_graph = exported.f.get_concrete_function().graph
    self.assertIn(
        exported_graph.get_operation_by_name("should_be_control_output"),
        exported_graph.control_outputs,
    )

    imported = cycle(exported, cycles, use_cpp_bindings=use_cpp_bindings)
    # Calling get_concrete_function wraps in a second call operation; we want to
    # inspect the original function body for the control output; digging into
    # graph.as_graph_def() and its FunctionDefLibrary is another option.
    (imported_concrete,) = imported.f.concrete_functions
    imported_graph = imported_concrete.graph
    self.assertIn(
        imported_graph.get_operation_by_name("should_be_control_output"),
        imported_graph.control_outputs,
    )

  def _make_asset(self, contents):
    fd, filename = tempfile.mkstemp(prefix=self.get_temp_dir())
    with os.fdopen(fd, "w") as f:
      f.write(contents)
    return filename

  @test_util.run_in_graph_and_eager_modes
  def test_assets(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    file1 = self._make_asset("contents 1")
    file2 = self._make_asset("contents 2")

    root = autotrackable.AutoTrackable()
    root.asset1 = asset.Asset(file1)
    root.asset2 = asset.Asset(file2)

    save_dir = os.path.join(self.get_temp_dir(), "save_dir")
    save.save(root, save_dir)

    file_io.delete_file(file1)
    file_io.delete_file(file2)
    load_dir = os.path.join(self.get_temp_dir(), "load_dir")
    file_io.rename(save_dir, load_dir)

    imported = test_load(load_dir, use_cpp_bindings=use_cpp_bindings)
    with open(self.evaluate(imported.asset1.asset_path), "r") as f:
      self.assertEqual("contents 1", f.read())
    with open(self.evaluate(imported.asset2.asset_path), "r") as f:
      self.assertEqual("contents 2", f.read())

  def test_cond_prune(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
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
        f, [tensor_spec.TensorSpec((), dtypes.float32)] * 2
    )
    f_pruned = f_wrapped.prune(x_in[0], [x_out[0]])

    class Adder(module.Module):

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32)
          ]
      )
      def add(self, x):
        return f_pruned(x)

    root = Adder()
    root.add(constant_op.constant(1.0))
    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    root.add(constant_op.constant(1.0))

  def test_capture_assets(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = autotrackable.AutoTrackable()
    root.vocab = asset.Asset(self._make_asset("contents"))
    root.f = def_function.function(
        lambda: root.vocab.asset_path, input_signature=[]
    )
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    original_output = root.f().numpy()
    imported_output = imported.f().numpy()
    self.assertNotEqual(original_output, imported_output)
    with open(imported_output, "r") as f:
      self.assertEqual("contents", f.read())

  def test_capture_assets_in_graph(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = autotrackable.AutoTrackable()
    root.vocab = asset.Asset(self._make_asset("contents"))
    root.f = def_function.function(
        lambda: root.vocab.asset_path, input_signature=[]
    )

    original_output = root.f().numpy()

    if cycles > 1:
      root = cycle(root, cycles - 1, use_cpp_bindings=use_cpp_bindings)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)

    with ops.Graph().as_default():
      imported = test_load(path, use_cpp_bindings=use_cpp_bindings)
      imported_tensor = imported.f()
      with monitored_session.MonitoredSession() as sess:
        imported_output = sess.run(imported_tensor)
        self.assertLen(ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS), 1)
        self.assertNotEqual(original_output, imported_output)
        with open(imported_output, "r") as f:
          self.assertEqual("contents", f.read())

  def test_dedup_assets(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    vocab = self._make_asset("contents")
    root = autotrackable.AutoTrackable()
    root.asset1 = asset.Asset(vocab)
    root.asset2 = asset.Asset(vocab)
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(
        imported.asset1.asset_path.numpy(), imported.asset2.asset_path.numpy()
    )

  def test_asset_fspath(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    vocab = pathlib.Path(self._make_asset("contents"))
    root = autotrackable.AutoTrackable()
    root.asset = asset.Asset(vocab)
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertTrue(hasattr(imported, "asset"))

  def test_implicit_input_signature(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function
    def func(x):
      return 2 * x

    root = autotrackable.AutoTrackable()
    root.f = func

    # Add two traces.
    root.f(constant_op.constant(1.0))
    root.f(constant_op.constant(1))

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    self.assertEqual(4.0, imported.f(constant_op.constant(2.0)).numpy())
    self.assertEqual(14, imported.f(constant_op.constant(7)).numpy())

  def test_explicit_input_signature(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)]
    )
    def func(x):
      return 2 * x

    root = autotrackable.AutoTrackable()
    root.f = func

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(4.0, imported.f(constant_op.constant(2.0)).numpy())

  def test_explicit_save_signature(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function
    def func(x):
      return 2 * x

    root = autotrackable.AutoTrackable()
    root.f = func

    imported = cycle(
        root,
        cycles,
        signatures={
            "f": root.f.get_concrete_function(
                tensor_spec.TensorSpec(None, dtypes.float32)
            )
        },
        use_cpp_bindings=use_cpp_bindings,
    )
    self.assertEqual(4.0, imported.f(constant_op.constant(2.0)).numpy())

  def test_nested_functions(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    f = def_function.function(
        lambda x: x * 2.0,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)],
    )
    g = def_function.function(
        lambda x: f(x) + 1.0,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)],
    )

    root = autotrackable.AutoTrackable()
    root.g = g
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    imported.g(constant_op.constant([1.0]))

  def test_function_with_default_bool_input(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def func(x, training=False):
      if training:
        return 2 * x
      else:
        return 7

    root = autotrackable.AutoTrackable()
    root.f = def_function.function(func)

    self.assertEqual(20, root.f(constant_op.constant(10), True).numpy())
    self.assertEqual(7, root.f(constant_op.constant(1)).numpy())
    self.assertEqual(2, root.f(constant_op.constant(1), True).numpy())

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    self.assertEqual(4, imported.f(constant_op.constant(2), True).numpy())
    self.assertEqual(7, imported.f(constant_op.constant(2)).numpy())

  def test_function_with_default_none_input(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def func(x, dtype=None):
      if dtype:
        return array_ops.zeros(shape=x.shape, dtype=dtype)
      else:
        return array_ops.zeros(shape=x.shape, dtype=dtypes.float32)

    root = autotrackable.AutoTrackable()
    root.f = def_function.function(func)

    self.assertAllEqual(
        [0.0, 0.0, 0.0], root.f(constant_op.constant([1, 2, 3])).numpy()
    )
    self.assertAllEqual(
        [0.0, 0.0, 0.0], root.f(constant_op.constant([1.0, 2.0, 3.0])).numpy()
    )
    self.assertAllEqual(
        [0.0, 0.0, 0.0, 0.0], root.f(constant_op.constant([1, 2, 3, 4])).numpy()
    )
    self.assertAllEqual(
        [0, 0, 0],
        root.f(
            constant_op.constant([1.0, 2.0, 3.0]), dtype=dtypes.int32
        ).numpy(),
    )

    concrete_functions = root.f._list_all_concrete_functions_for_serialization()  # pylint: disable=protected-access
    self.assertLen(concrete_functions, 4)

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    self.assertAllEqual(
        [0.0, 0.0, 0.0],
        imported.f(constant_op.constant([1, 2, 3]), None).numpy(),
    )
    self.assertAllEqual(
        [0.0, 0.0, 0.0],
        imported.f(constant_op.constant([1.0, 2.0, 3.0])).numpy(),
    )
    self.assertAllEqual(
        [0.0, 0.0, 0.0, 0.0],
        imported.f(constant_op.constant([1, 2, 3, 4])).numpy(),
    )
    self.assertAllEqual(
        [0, 0, 0],
        imported.f(
            constant_op.constant([1.0, 2.0, 3.0]), dtype=dtypes.int32
        ).numpy(),
    )

  def test_function_with_str_bytes_input(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function
    def func(x, y):
      return string_ops.string_join([x, y])

    root = autotrackable.AutoTrackable()
    root.f = func

    self.assertAllEqual(b"ab", root.f("a", "b"))
    self.assertAllEqual(b"ab", root.f("a", constant_op.constant("b")))
    self.assertAllEqual(b"ab", root.f(constant_op.constant("a"), "b"))

    concrete_functions = root.f._list_all_concrete_functions_for_serialization()  # pylint: disable=protected-access
    self.assertLen(concrete_functions, 3)

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    self.assertAllEqual(b"ab", imported.f("a", "b"))
    self.assertAllEqual(b"ab", imported.f("a", constant_op.constant("b")))
    self.assertAllEqual(b"ab", imported.f(constant_op.constant("a"), "b"))

  def test_function_no_return(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class TrackableWithOneVariable(autotrackable.AutoTrackable):

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

    imported = cycle(obj, cycles, use_cpp_bindings=use_cpp_bindings)

    imported.increase(constant_op.constant(10.0))
    self.assertEqual(26.0, imported.variable.numpy())
    imported.increase(constant_op.constant(1.0))
    self.assertEqual(27.0, imported.variable.numpy())

  def test_structured_inputs(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def func(x, training=True):
      # x is a nested structure, we care about one particular tensor.
      _, (a, b) = x
      if training:
        return 2 * a["a"] + b
      else:
        return 7

    root = autotrackable.AutoTrackable()
    root.f = def_function.function(func)

    x = constant_op.constant(10)
    y = constant_op.constant(11)

    input1 = [6, ({"a": x}, y)]
    input2 = [7, ({"a": x}, y)]  # Not compatible with input1 signature.
    input3 = [6, ({"a": y}, x)]  # Compatible with input1 signature.

    # Note: by only calling f(input1) before serialization, only inputs with
    # matching signature will be valid on the loaded model.
    self.assertEqual(31, root.f(input1).numpy())

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    with self.assertRaisesRegex(
        ValueError, "Could not find matching concrete function to call"
    ):
      imported.f(input2)

    self.assertEqual(31, imported.f(input1).numpy())
    self.assertEqual(32, imported.f(input3).numpy())

  def test_structured_inputs_bare_concrete_function(
      self, cycles, use_cpp_bindings
  ):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

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

    root = autotrackable.AutoTrackable()
    root.f = def_function.function(func).get_concrete_function(input1)

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    with self.assertRaises(TypeError):
      imported.f(input2)

    self.assertEqual(31, imported.f(input1, True).numpy())
    self.assertEqual(32, imported.f(input3, True).numpy())

  def test_structured_output(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    # Use fields with non-alphabetical order
    named_tuple_type = collections.namedtuple("NamedTupleHello", ["b", "a"])

    def func(input1, input2):
      named_tuple = named_tuple_type(a=input1 + input2, b=input1 * input2)
      return [named_tuple, input2, {"x": 0.5}]

    root = autotrackable.AutoTrackable()
    root.f = def_function.function(func)

    result = root.f(constant_op.constant(2), constant_op.constant(3))

    self.assertEqual(5, result[0].a.numpy())
    self.assertEqual(6, result[0].b.numpy())
    self.assertEqual(["b", "a"], list(result[0]._asdict().keys()))
    self.assertEqual(3, result[1].numpy())
    self.assertEqual(0.5, result[2]["x"].numpy())

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    result = imported.f(constant_op.constant(2), constant_op.constant(5))
    self.assertEqual(7, result[0].a.numpy())
    self.assertEqual(10, result[0].b.numpy())
    self.assertEqual(["b", "a"], list(result[0]._asdict().keys()))
    self.assertEqual(5, result[1].numpy())
    self.assertEqual(0.5, result[2]["x"].numpy())

  def test_pretty_print_signature(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    named_tuple_type = collections.namedtuple("NamedTupleHello", ["b", "a"])

    def func(input1, input2):
      named_tuple = named_tuple_type(a=input1 + input2, b=input1 * input2)
      return [named_tuple, input2, {"x": 0.5}]

    root = autotrackable.AutoTrackable()
    root.f = def_function.function(func).get_concrete_function(
        constant_op.constant(2), constant_op.constant(3)
    )

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(
        imported.f.pretty_printed_signature(),
        """func(input1, input2)
  Args:
    input1: int32 Tensor, shape=()
    input2: int32 Tensor, shape=()
  Returns:
    [NamedTupleHello(b=<1>, a=<2>), <3>, {'x': <4>}]
      <1>: int32 Tensor, shape=()
      <2>: int32 Tensor, shape=()
      <3>: int32 Tensor, shape=()
      <4>: float32 Tensor, shape=()""",
    )

  def test_positional_arguments(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def func(x, training=False, abc=7.1, defg=7.7):
      del abc
      if training:
        return 2 * x
      if defg == 7:
        return 6
      else:
        return 7

    root = autotrackable.AutoTrackable()
    root.f = def_function.function(func)

    self.assertEqual(20, root.f(constant_op.constant(10), True).numpy())
    self.assertEqual(7, root.f(constant_op.constant(1)).numpy())
    self.assertEqual(2, root.f(constant_op.constant(1), True).numpy())
    self.assertEqual(6, root.f(constant_op.constant(1), defg=7.0).numpy())

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    self.assertEqual(4, imported.f(constant_op.constant(2), True).numpy())
    self.assertEqual(7, imported.f(constant_op.constant(2)).numpy())
    self.assertEqual(6, imported.f(constant_op.constant(1), defg=7.0).numpy())

  def test_additional_kwargs(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def func(x, training=False, **options):
      del options
      if training:
        return 2 * x
      else:
        return 7

    root = autotrackable.AutoTrackable()
    root.f = def_function.function(func)

    x = constant_op.constant(10)
    self.assertEqual(7, root.f(x, learning_rate=0.5, epochs=3).numpy())

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    with self.assertRaisesRegex(
        ValueError, "Could not find matching concrete function to call.*"
    ):
      imported.f(x, learning_rate=0.5, epochs=4)

    self.assertEqual(7, imported.f(x, learning_rate=0.5, epochs=3).numpy())

  def test_member_function(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class TrackableWithMember(autotrackable.AutoTrackable):

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

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    self.assertEqual(4, imported.f(constant_op.constant(2), True).numpy())
    self.assertEqual(27, imported.f(constant_op.constant(2)).numpy())

  def test_side_effect_listing(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class M(autotrackable.AutoTrackable):

      def __init__(self):
        super(M, self).__init__()
        self.var = None

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)]
      )
      def f(self, x):
        if self.var is None:
          self.var = variables.Variable(2.0)
        return x * self.var

    m = M()
    cycle(m, cycles)
    self.assertEqual(4.0, m.f(constant_op.constant(2.0)).numpy())

  def test_basic_backprop(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    weight = variables.Variable(1.0, trainable=True)
    bias = variables.Variable(0.0, trainable=True)
    g = def_function.function(
        lambda x: x * weight + bias,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)],
    )

    root = autotrackable.AutoTrackable()
    root.weight = weight
    root.bias = bias
    root.g = g
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    with backprop.GradientTape() as t:
      x = constant_op.constant([3.5])
      loss = imported.g(x)
      grad = t.gradient(loss, [imported.weight, imported.bias])
      self.assertAllClose(grad, [3.5, 1.0])

  def test_nested_backprop(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    weight = variables.Variable(1.0, trainable=True)
    bias = variables.Variable(0.0, trainable=True)

    # Note: this function gets called from other function defs via a
    # "PartitionedCall" op node.
    @def_function.function(
        input_signature=[
            tensor_spec.TensorSpec(None, dtypes.float32),
            tensor_spec.TensorSpec(None, dtypes.float32),
        ]
    )
    def mul(x, y):
      return x * y

    # Note: this function gets called from other function defs via a
    # "StatefulPartitionedCall" op node.
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)]
    )
    def f(x):
      return mul(weight.read_value(), x)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)]
    )
    def g(x):
      return (f(x) + bias,)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)]
    )
    def h(x):
      return (g(x) + bias,)

    root = autotrackable.AutoTrackable()
    root.weight = weight
    root.bias = bias
    root.g = h

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    with backprop.GradientTape() as t:
      x = constant_op.constant([3.5])
      loss = imported.g(x)
    grad = t.gradient(loss, [imported.weight, imported.bias])
    self.assertAllClose(grad, [3.5, 2.0])

  def test_while_loop_backprop(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    weight = variables.Variable(2.0, trainable=True)

    @def_function.function(
        input_signature=[
            tensor_spec.TensorSpec(dtype=dtypes.float32, shape=(None, None))
        ]
    )
    def g(x):
      """Adds rows of matrix x after multiplying each entry by v."""
      i_0 = constant_op.constant(0)
      s_0 = constant_op.constant([0.0, 0.0])
      cond = lambda i, _: i < array_ops.shape(x)[1]
      body = lambda i, s: (i + 1, s + weight * x[:, i])
      i_end, s_end = while_loop.while_loop(cond, body, (i_0, s_0))
      del i_end
      return s_end

    root = autotrackable.AutoTrackable()
    root.weight = weight
    root.g = g
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    def get_gradient(obj):
      with backprop.GradientTape() as t:
        x = constant_op.constant([[1.0, 2.0, 3.0], [1.0, -2, 3.0]])
        y = obj.g(x)
        self.assertAllClose(y, obj.weight * [6.0, 2.0])
        loss = math_ops.reduce_sum(y)  # weight * 8.
        self.assertAllEqual(t.watched_variables(), [obj.weight])
        return t.gradient(loss, obj.weight)

    imported_gradient = get_gradient(imported)
    original_gradient = get_gradient(root)
    self.assertIsNotNone(original_gradient)
    self.assertAllClose(original_gradient, 8.0)
    self.assertIsNotNone(imported_gradient)
    self.assertAllClose(imported_gradient, 8.0)

  def _test_restored_func_with_captured_var_backprop(
      self, cycles, use_cpp_bindings, dtype
  ):
    weight = variables.Variable(2.0, trainable=True, dtype=dtype)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(dtype=dtype, shape=())]
    )
    def g(x):
      return x * weight

    root = autotrackable.AutoTrackable()
    root.weight = weight
    root.g = g
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    def get_gradient(obj):
      with backprop.GradientTape() as t:
        x = constant_op.constant(2.0, dtype=dtype)
        y = obj.g(x)
        self.assertAllClose(y, obj.weight * 2.0)
        self.assertAllEqual(t.watched_variables(), [obj.weight])
        return t.gradient(y, obj.weight)

    imported_gradient = get_gradient(imported)
    original_gradient = get_gradient(root)
    self.assertIsNotNone(original_gradient)
    self.assertAllClose(original_gradient, 2.0)
    self.assertIsNotNone(imported_gradient)
    self.assertAllClose(imported_gradient, 2.0)

  def test_nested_fn_backprop(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    weight = variables.Variable(2.0, trainable=True)

    @def_function.function(
        input_signature=[
            tensor_spec.TensorSpec(dtype=dtypes.float32, shape=(None, None))
        ]
    )
    def g(x):
      weight.read_value()  # Just get the tape to watch the variable
      handle = array_ops.identity(weight.handle)

      @def_function.function
      def launder_var_handle():
        return array_ops.identity(handle)

      return x + resource_variable_ops.read_variable_op(
          launder_var_handle(), dtypes.float32
      )

    root = autotrackable.AutoTrackable()
    root.weight = weight
    root.g = g
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    def get_gradient(obj, persistent):
      with backprop.GradientTape(persistent=persistent) as t:
        x = constant_op.constant([[1.0, 2.0, 3.0], [1.0, -2, 3.0]])
        y = obj.g(x)
        self.assertAllClose(y, obj.weight + x)
        loss = math_ops.reduce_sum(y)
        return t.gradient(loss, obj.weight)

    imported_gradient = get_gradient(imported, persistent=False)
    original_gradient = get_gradient(root, persistent=False)
    self.assertIsNotNone(original_gradient)
    self.assertAllClose(original_gradient, 6.0)
    self.assertIsNotNone(imported_gradient)
    self.assertAllClose(imported_gradient, 6.0)

  def test_restored_func_with_captured_var_backprop_float32(
      self, cycles, use_cpp_bindings
  ):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    self._test_restored_func_with_captured_var_backprop(
        cycles, use_cpp_bindings, dtypes.float32
    )

  def test_restored_func_with_captured_var_backprop_float64(
      self, cycles, use_cpp_bindings
  ):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    self._test_restored_func_with_captured_var_backprop(
        cycles, use_cpp_bindings, dtypes.float64
    )

  def test_callable(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class M1(autotrackable.AutoTrackable):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)]
      )
      def __call__(self, x):
        return x

    root = autotrackable.AutoTrackable()
    root.m1 = M1()
    root.m2 = autotrackable.AutoTrackable()
    root.m2.__call__ = def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)]
    )(lambda x: x * 3.0)
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
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

  def test_chain_callable(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    func = def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)]
    )(lambda x: x * 3.0)
    root = autotrackable.AutoTrackable()
    root.__call__ = autotrackable.AutoTrackable()
    root.__call__.__call__ = autotrackable.AutoTrackable()
    root.__call__.__call__.__call__ = func

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertTrue(callable(imported))
    x = constant_op.constant(1.0)
    self.assertAllEqual(imported(x).numpy(), 3.0)

  def test_load_in_graph_mode(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(1.0, name="v_one", trainable=False)
    root.v2 = variables.Variable(2.0, name="v_two", trainable=True)
    root.f = def_function.function(
        lambda x: root.v2 * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)],
    )

    if cycles > 1:
      root = cycle(root, cycles - 1, use_cpp_bindings=use_cpp_bindings)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)

    with ops.Graph().as_default() as g:
      imported = test_load(path, use_cpp_bindings=use_cpp_bindings)
      var_v1 = imported.v1
      self.assertFalse(var_v1.trainable)
      var_v2 = imported.v2
      self.assertTrue(var_v2.trainable)
      output = imported.f(constant_op.constant(2.0))
      with monitored_session.MonitoredSession() as sess:
        self.assertEqual(1.0, sess.run(var_v1))
        self.assertEqual(4.0, sess.run(output))
      self.assertCountEqual(
          [var_v1, var_v2], g.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      )
      # load() should not add to TRAINABLE_VARIABLES. Higher levels of model
      # building control retraining or frozen use of imported SavedModels.
      self.assertCountEqual(
          [], g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      )

  def test_load_in_func_graph(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(1.0)
    root.v2 = variables.Variable(2.0)
    root.f = def_function.function(
        lambda x: root.v2 * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)],
    )

    if cycles > 1:
      root = cycle(root, cycles - 1, use_cpp_bindings=use_cpp_bindings)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)

    closure = autotrackable.AutoTrackable()

    @def_function.function
    def func(x):
      if not hasattr(closure, "model"):
        closure.model = load.load(path)
      return closure.model.f(x)

    inputs = constant_op.constant(2.0)
    self.assertEqual(4.0, func(inputs).numpy())

  def test_soft_matching(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)]
    )
    def func(x):
      return 2 * x

    root = autotrackable.AutoTrackable()
    root.f = func

    self.assertAllEqual([2], root.f(constant_op.constant([1])).numpy())
    self.assertAllEqual([2, 4], root.f(constant_op.constant([1, 2])).numpy())

    concrete_functions = root.f._list_all_concrete_functions_for_serialization()  # pylint: disable=protected-access
    self.assertLen(concrete_functions, 1)

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    with self.assertRaisesRegex(
        TypeError, "Binding inputs to tf.function `f` failed"
    ):
      # We cannot call the function with a constant of shape ().
      imported.f(constant_op.constant(2)).numpy()

    # TODO(vbardiovsky): When classes are revived with input_signatures, we
    # should also check that the calls below are not generating any more
    # concrete functions.
    self.assertAllEqual(
        [2, 4, 6, 8], imported.f(constant_op.constant([1, 2, 3, 4])).numpy()
    )
    self.assertAllEqual(
        [2, 4, 6], imported.f(constant_op.constant([1, 2, 3])).numpy()
    )

  def test_jit_compile(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    # It'd be nice to use parameterize here, but the library does not support
    # having parameterized test methods inside already-parameterized classes.
    for jit_compile in (None, True, False):

      @def_function.function(jit_compile=jit_compile)
      def f(x):
        return x + 1.0

      root = module.Module()
      root.f = f
      save_dir = os.path.join(self.get_temp_dir(), "saved_model")
      save.save(root, save_dir)

      imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

      self.assertEqual(imported.f._jit_compile, jit_compile)

  def test_get_concrete_function(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function
    def func(x, training=False):
      if training:
        return 2 * x
      else:
        return 3 * x

    func.get_concrete_function(
        tensor_spec.TensorSpec([None], dtypes.int32), True
    )
    func.get_concrete_function(tensor_spec.TensorSpec([None], dtypes.float32))

    root = autotrackable.AutoTrackable()
    root.f = func

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    concrete = imported.f.get_concrete_function(
        training=True, x=tensor_spec.TensorSpec([None], dtypes.int32)
    )

    self.assertAllEqual(
        [2, 4, 6, 8], concrete(x=constant_op.constant([1, 2, 3, 4])).numpy()
    )
    with self.assertRaisesRegex(
        ValueError, "Could not find matching concrete function to call"
    ):
      imported.f.get_concrete_function(
          tensor_spec.TensorSpec([None], dtypes.int32)
      )
    imported.f.get_concrete_function(
        tensor_spec.TensorSpec([None], dtypes.int32), True
    )

  def test_concrete_function(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)]
    )
    def func(x):
      return 2 * x

    root = autotrackable.AutoTrackable()
    root.f = func.get_concrete_function()

    self.assertAllEqual([2], root.f(constant_op.constant([1])).numpy())
    self.assertAllEqual([2, 4], root.f(constant_op.constant([1, 2])).numpy())

    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(
        root, cycles, signatures={}, use_cpp_bindings=use_cpp_bindings
    )

    self.assertAllEqual(
        [2, 4, 6, 8], imported.f(constant_op.constant([1, 2, 3, 4])).numpy()
    )
    self.assertAllEqual(
        [2, 4, 6], imported.f(constant_op.constant([1, 2, 3])).numpy()
    )

  def test_concrete_function_captures(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class Root(module.Module):

      def __init__(self):
        self.v = variables.Variable(1.0)
        self.v1 = variables.Variable(1.0)

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)]
      )
      def use_v(self, x):
        return self.v + self.v1 + 1.0

    root = Root()
    self.assertIn(
        root.v.handle,
        root.use_v.get_concrete_function().graph.external_captures,
    )
    root = cycle(
        root,
        cycles,
        signatures=root.use_v.get_concrete_function(),
        use_cpp_bindings=use_cpp_bindings,
    )
    func_captures = root.use_v.get_concrete_function().graph.external_captures
    self.assertLen(func_captures, 2)
    self.assertTrue(any(root.v.handle is t for t in func_captures))
    self.assertTrue(any(root.v1.handle is t for t in func_captures))
    signature_captures = root.signatures[
        "serving_default"
    ].graph.external_captures
    self.assertLen(signature_captures, 2)
    self.assertTrue(any(root.v.handle is t for t in signature_captures))
    self.assertTrue(any(root.v1.handle is t for t in signature_captures))

  def test_concrete_function_arg_names(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)]
    )
    def func(x):
      return 2 * x

    root = autotrackable.AutoTrackable()
    root.f = func.get_concrete_function()

    self.assertAllEqual([2], root.f(constant_op.constant([1])).numpy())

    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(
        root, cycles, signatures={}, use_cpp_bindings=use_cpp_bindings
    )

    self.assertAllEqual(
        [2, 4, 6], imported.f(x=constant_op.constant([1, 2, 3])).numpy()
    )

  def test_concrete_function_no_signature(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function
    def func(x):
      return 2 * x

    root = autotrackable.AutoTrackable()
    root.f = func.get_concrete_function(constant_op.constant([1]))
    self.assertAllEqual([4], root.f(constant_op.constant([2])).numpy())
    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(
        root, cycles, signatures={}, use_cpp_bindings=use_cpp_bindings
    )
    self.assertAllEqual([6], imported.f(constant_op.constant([3])).numpy())

  @test_util.run_in_graph_and_eager_modes
  def test_concrete_function_backprop(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.float32)]
    )
    def func(x):
      return x**2.0

    root = autotrackable.AutoTrackable()
    root.f = func.get_concrete_function()

    def _compute_gradient(function):
      with backprop.GradientTape() as tape:
        inp = constant_op.constant(1.0)
        tape.watch(inp)
        output = function(inp)
      return tape.gradient(output, inp)

    self.assertAllEqual(2.0, _compute_gradient(root.f))
    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(
        root, cycles, signatures={}, use_cpp_bindings=use_cpp_bindings
    )
    self.assertAllEqual(2.0, _compute_gradient(imported.f))

  def test_revived_concrete_function_kwargs(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function
    def func(x, y):
      return x * (y + 1.0)

    root = autotrackable.AutoTrackable()
    root.f = func.get_concrete_function(
        tensor_spec.TensorSpec([], dtypes.float32),
        tensor_spec.TensorSpec([], dtypes.float32),
    )
    self.assertEqual(
        8.0,
        root.f(
            y=constant_op.constant(3.0), x=constant_op.constant(2.0)
        ).numpy(),
    )
    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(
        root, cycles, signatures={}, use_cpp_bindings=use_cpp_bindings
    )
    self.assertEqual(
        8.0,
        imported.f(
            y=constant_op.constant(3.0), x=constant_op.constant(2.0)
        ).numpy(),
    )

  def test_revived_concrete_function_tensorspec_kwargs(
      self, cycles, use_cpp_bindings
  ):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function
    def func(*args):
      x, y = args
      return x * (y + 1.0)

    root = autotrackable.AutoTrackable()
    root.f = func.get_concrete_function(
        tensor_spec.TensorSpec([], dtypes.float32, name="x"),
        tensor_spec.TensorSpec([], dtypes.float32, name="y"),
    )
    self.assertEqual(
        8.0,
        root.f(
            y=constant_op.constant(3.0), x=constant_op.constant(2.0)
        ).numpy(),
    )
    imported = cycle(
        root, cycles, signatures={}, use_cpp_bindings=use_cpp_bindings
    )
    self.assertEqual(
        8.0,
        imported.f(
            y=constant_op.constant(3.0), x=constant_op.constant(2.0)
        ).numpy(),
    )

  def test_concrete_function_variable_argument(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    capture = variables.Variable(0)

    @def_function.function
    def func(v):
      v.assign_add(1)
      capture.assign_sub(1)

    vsave = variables.Variable(1)
    root = autotrackable.AutoTrackable()
    root.f = func.get_concrete_function(vsave)
    root.capture = capture

    self.assertEqual(1, vsave.numpy())
    root.f(vsave)
    self.assertEqual(2, vsave.numpy())
    self.assertEqual(-1, capture.numpy())

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    vload = variables.Variable(1)
    imported.f(vload)
    self.assertEqual(2, vload.numpy())
    self.assertEqual(-2, imported.capture.numpy())
    imported.f(v=vload)
    self.assertEqual(3, vload.numpy())
    self.assertEqual(-3, imported.capture.numpy())

    self.assertEqual(-1, capture.numpy())

  def test_function_and_component(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function
    def func(v):
      return v + 1

    root = autotrackable.AutoTrackable()
    root.func = func
    root.concrete_func = func.get_concrete_function(
        tensor_spec.TensorSpec(None, dtypes.int32)
    )
    one = constant_op.constant(1)
    self.assertEqual(2, root.func(one).numpy())
    self.assertEqual(2, root.concrete_func(one).numpy())
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(2, imported.func(one).numpy())
    self.assertEqual(2, imported.concrete_func(one).numpy())

  def test_dict(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = autotrackable.AutoTrackable()
    root.variables = dict(a=variables.Variable(1.0))
    root.variables["b"] = variables.Variable(2.0)
    root.variables["c"] = 1
    root.funcs = dict(
        a=def_function.function(lambda: constant_op.constant(100.0))
    )
    root.funcs["conc"] = root.funcs["a"].get_concrete_function()
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(1.0, imported.variables["a"].numpy())
    self.assertEqual(2.0, imported.variables["b"].numpy())
    self.assertEqual(set(["a", "b"]), set(imported.variables.keys()))
    self.assertEqual(100.0, imported.funcs["a"]().numpy())
    self.assertEqual(100.0, imported.funcs["conc"]().numpy())

  def test_list(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = autotrackable.AutoTrackable()
    root.variables = [variables.Variable(1.0)]
    root.variables.append(1)
    root.variables.append(variables.Variable(3.0))
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(1.0, imported.variables[0].numpy())
    self.assertEqual(3.0, imported.variables[2].numpy())
    self.assertIs(None, imported.variables[1])
    self.assertLen(imported.variables, 3)

  def test_tuple(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = autotrackable.AutoTrackable()
    root.variables = (variables.Variable(1.0), 1, variables.Variable(3.0))
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(1.0, imported.variables[0].numpy())
    self.assertEqual(3.0, imported.variables[2].numpy())
    self.assertIs(None, imported.variables[1])
    self.assertLen(imported.variables, 3)

  def test_functions_list(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = autotrackable.AutoTrackable()
    v1 = variables.Variable(1.0)
    root.losses = [def_function.function(lambda: math_ops.reduce_sum(v1**2))]
    root.variables = [v1]

    @def_function.function
    def _v2_loss():
      if len(root.variables) == 1:
        v2 = variables.Variable(2.0)
        root.variables.append(v2)
      return math_ops.reduce_sum(root.variables[1] ** 2)

    root.losses.append(_v2_loss)
    self.assertAllClose([1.0, 4.0], [loss() for loss in root.losses])
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertAllClose([1.0, 4.0], [loss() for loss in imported.losses])
    imported.variables[0].assign(3.0)
    imported.variables[1].assign(4.0)
    self.assertAllClose([9.0, 16.0], [loss() for loss in imported.losses])

  def test_captured_constant(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    const = array_ops.zeros([100])
    root = autotrackable.AutoTrackable()
    root.f = def_function.function(lambda: const + 1.0)
    root.g = def_function.function(lambda: const + 2.0)
    self.assertAllClose(array_ops.ones([100]), root.f())
    self.assertAllClose(2.0 * array_ops.ones([100]), root.g())
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertAllClose(array_ops.ones([100]), imported.f())
    self.assertAllClose(2.0 * array_ops.ones([100]), imported.g())
    # TODO(b/123408994): Use the public get_concrete_function.
    f_concrete = imported.f._list_all_concrete_functions_for_serialization()[0]
    g_concrete = imported.g._list_all_concrete_functions_for_serialization()[0]
    self.assertLen(f_concrete.captured_inputs, 1)
    self.assertLen(g_concrete.captured_inputs, 1)
    # We should be using the same captured EagerTensor in both functions, not
    # duplicating the constant.
    self.assertIs(f_concrete.captured_inputs[0], g_concrete.captured_inputs[0])

  def test_functions_accessed_once(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class Exported(autotrackable.AutoTrackable):

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
    imported = cycle(exported, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(0, imported.make_func().numpy())
    self.assertEqual(1, exported.make_func().numpy())

  def test_overwritten_signatures_error(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    exported = autotrackable.AutoTrackable()
    exported.f = def_function.function(lambda: constant_op.constant(1.0))
    imported = cycle(
        exported,
        cycles,
        signatures={"key": exported.f.get_concrete_function()},
        use_cpp_bindings=use_cpp_bindings,
    )
    self.assertEqual(1.0, imported.signatures["key"]()["output_0"].numpy())
    imported.signatures = {"key1": imported.signatures["key"]}
    with self.assertRaisesRegex(ValueError, "signatures"):
      save.save(imported, tempfile.mkdtemp(prefix=self.get_temp_dir()))

  def test_signature_loading(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class Exported(autotrackable.AutoTrackable):

      def __init__(self):
        self.v = variables.Variable(3.0)

      @def_function.function
      def do(self, x):
        return self.v * x

    exported = Exported()
    imported = cycle(
        exported,
        cycles,
        signatures=exported.do.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.float32)
        ),
        use_cpp_bindings=use_cpp_bindings,
    )
    self.assertEqual(["serving_default"], list(imported.signatures.keys()))
    imported_function = imported.signatures["serving_default"]
    two = constant_op.constant(2.0)
    self.assertEqual(6.0, imported_function(x=two)["output_0"].numpy())
    imported.v.assign(4.0)
    self.assertEqual(8.0, imported_function(x=two)["output_0"].numpy())
    self.assertEqual(8.0, imported_function(two)["output_0"].numpy())
    with self.assertRaises(TypeError):
      # The signatures mapping is immutable
      imported.signatures["random_key"] = 3

  def test_names_normalized(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class ObjWithFunction(module.Module):

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec([], dtype=dtypes.int32, name="A-b"),
              tensor_spec.TensorSpec([], dtype=dtypes.int32, name="A/D"),
              tensor_spec.TensorSpec([], dtype=dtypes.int32, name="bar"),
              tensor_spec.TensorSpec([], dtype=dtypes.int32, name="e"),
          ]
      )
      def foo(self, a, b, c, d=10, **options):
        del options
        return a + b + c + d

    exported = ObjWithFunction()

    with self.assertLogs(level="WARNING") as logs:
      imported = cycle(exported, cycles, use_cpp_bindings=use_cpp_bindings)

    expected_message = (
        "WARNING:absl:Function `foo` contains input name(s) A-b, A/D with "
        "unsupported characters which will be renamed to a_b, a_d in the "
        "SavedModel."
    )
    self.assertIn(expected_message, logs.output)

    loaded_signature = imported.signatures["serving_default"].inputs
    self.assertTrue(
        {"a_b:0", "a_d:0"}.issubset({arg.name for arg in loaded_signature}),
    )

  def test_multiple_argument_signatures_no_positional(
      self, cycles, use_cpp_bindings
  ):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class Exported(autotrackable.AutoTrackable):

      @def_function.function
      def do(self, x, y):
        return x + y

    exported = Exported()
    imported = cycle(
        exported,
        cycles,
        signatures=exported.do.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.float32),
            tensor_spec.TensorSpec(None, dtypes.float32),
        ),
        use_cpp_bindings=use_cpp_bindings,
    )
    with self.assertRaises(TypeError):
      imported.signatures["serving_default"](
          constant_op.constant(1.0), y=constant_op.constant(2.0)
      )
    self.assertEqual(
        {"output_0": 3.0},
        self.evaluate(
            imported.signatures["serving_default"](
                x=constant_op.constant(1.0), y=constant_op.constant(2.0)
            )
        ),
    )

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

    root = autotrackable.AutoTrackable()
    root.table1 = table1
    root.lookup1 = _make_lookup_function(table1)
    root.table2 = table2
    root.lookup2 = _make_lookup_function(table2)
    return root

  def test_table(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = self._make_model_with_tables()
    imported = cycle(root, cycles, signatures={})
    keys = constant_op.constant(["brain", "test", "foo", "surgery"])
    self.assertAllEqual([0, -1, -1, 2], imported.lookup1(keys).numpy())
    self.assertAllEqual([2, 0, 1, -1], imported.lookup2(keys).numpy())

  def test_table_collections_untouched_eager(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

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
    cycle(root, 1, use_cpp_bindings=use_cpp_bindings)
    original_collections = _gather_nonempty_collections()
    cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(original_collections, _gather_nonempty_collections())

  def test_table_in_graph(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = self._make_model_with_tables()

    if cycles > 1:
      root = cycle(root, cycles - 1, use_cpp_bindings=use_cpp_bindings)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)
    imported = cycle(root, 1, use_cpp_bindings=use_cpp_bindings)

    with ops.Graph().as_default():
      imported = test_load(path, use_cpp_bindings=use_cpp_bindings)
      keys = constant_op.constant(["brain", "test", "foo", "surgery"])
      output1 = imported.lookup1(keys)
      output2 = imported.lookup2(keys)
      with monitored_session.MonitoredSession() as sess:
        self.assertAllEqual([0, -1, -1, 2], sess.run(output1))
        self.assertAllEqual([2, 0, 1, -1], sess.run(output2))

  def test_preserve_argspec(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def f(a, b, c):  # pylint: disable=unused-argument
      return None

    original_fullargspec = tf_inspect.getfullargspec(f)

    root = autotrackable.AutoTrackable()
    root.f = def_function.function(f)
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    restored_fullargspec = tf_inspect.getfullargspec(imported.f)
    self.assertEqual(original_fullargspec, restored_fullargspec)

  def test_canonicalize_inputs(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function(autograph=False)
    def func(a=1, b=2, c=3, training=True):
      if training:
        return [a, b, c, training]
      else:
        return [c, b, a, training]

    # TODO(b/123501567): Work-around to trigger generic traces of a function
    # with extra non tensor args.
    signature = 3 * [tensor_spec.TensorSpec(None, dtypes.float32)]

    @def_function.function(input_signature=signature)
    def trigger(a, b, c):
      func(a, b, c, True)
      func(a, b, c, False)

    trigger.get_concrete_function()

    root = autotrackable.AutoTrackable()
    root.f = func
    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertAllEqual(root.f(), [1.0, 2.0, 3.0, True])
    self.assertAllEqual(root.f(-1.0, training=False), [3.0, 2.0, -1.0, False])

    with self.assertRaisesRegex(
        ValueError, "Could not find matching concrete function"
    ):
      root.f(["hello", 1.0])

  def test_prefer_specific_trace(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function(autograph=False)
    def func(a):
      if isinstance(a, int):
        return a
      else:
        return a + 1

    self.assertAllEqual(2, func(2).numpy())
    self.assertAllEqual(3, func(constant_op.constant(2)).numpy())

    root = autotrackable.AutoTrackable()
    root.f = func
    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertAllEqual(2, root.f(2).numpy())
    self.assertAllEqual(4, root.f(3).numpy())
    self.assertAllEqual(3, root.f(constant_op.constant(2)).numpy())
    self.assertAllEqual(4, root.f(constant_op.constant(3)).numpy())

  def test_partial_with_non_tensor_defaults(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def f(x, y=3):
      return x + y

    func = def_function.function(functools.partial(f, y=5))

    root = autotrackable.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(1), 6)

    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertAllEqual(root.f(1), 6)

  def test_partial_with_positional(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def f(x, y):
      return x + y

    func = def_function.function(functools.partial(f, constant_op.constant(5)))

    root = autotrackable.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(1), 6)

    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertAllEqual(root.f(1), 6)

  def test_partial_with_positional_captured_tensors(
      self, cycles, use_cpp_bindings
  ):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def f(x, y):
      return x + y

    tensor = constant_op.constant(5) + constant_op.constant(7)
    func = def_function.function(functools.partial(f, tensor))

    root = autotrackable.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(1), 13)

    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertAllEqual(root.f(1), 13)

  def test_partial_keyword_hiding_default(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def f(x=3, training=True, y=7):
      if training:
        return x + y
      else:
        return x + y + 2

    func = def_function.function(functools.partial(f, y=6))

    root = autotrackable.AutoTrackable()
    root.f = func
    self.assertEqual(root.f().numpy(), 9)
    self.assertEqual(root.f(training=False).numpy(), 11)

    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(root.f().numpy(), 9)
    self.assertEqual(root.f(training=False).numpy(), 11)

  def test_partial_with_kwargs(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def f(a, b, *args, **kwargs):
      args_sum = sum(args)
      return a + b + kwargs["some_tensor"] * kwargs["learning_rate"] + args_sum

    constant_tensor = constant_op.constant(10)
    func = def_function.function(
        functools.partial(
            f, 7, 1, 2, learning_rate=3, some_tensor=constant_tensor
        )
    )

    root = autotrackable.AutoTrackable()
    root.f = func
    self.assertEqual(root.f(constant_op.constant(4)).numpy(), 44)

    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(root.f(constant_op.constant(5)).numpy(), 45)

  def test_partial_bind_only_first_argument(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    if sys.version_info[0] < 3:
      self.skipTest(
          "Test is only valid in python3. Only then we get some more "
          "advanced inspection of partials where this is allowed."
      )

    def f(x, y):
      return x + y

    partial_func = functools.partial(f, x=5)
    tf_func = def_function.function(partial_func)

    root = autotrackable.AutoTrackable()
    root.f = tf_func
    self.assertAllEqual(root.f(y=constant_op.constant(7)), 12)

    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertAllEqual(root.f(y=constant_op.constant(9)), 14)

  def test_partial_with_passed_fn_as_default(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def f(x, y):
      return x(3) + y

    def my_func(a):
      return 2 * a

    func = def_function.function(functools.partial(f, my_func))

    root = autotrackable.AutoTrackable()
    root.f = func
    self.assertEqual(root.f(constant_op.constant(3)).numpy(), 9)

    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(root.f(constant_op.constant(3)).numpy(), 9)

  def test_partial_with_input_signature(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def full_function(a, b, c=3.0):
      return a, b, c

    partial = functools.partial(full_function, 1, c=4)
    self.assertAllEqual((1, 2.0, 4), partial(2.0))

    signature = [tensor_spec.TensorSpec([], dtypes.float32)]
    func = def_function.function(partial, input_signature=signature)

    root = autotrackable.AutoTrackable()
    root.f = func
    a, b, c = root.f(2.0)
    self.assertAllEqual([a.numpy(), b.numpy(), c.numpy()], (1, 2.0, 4))

    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    a, b, c = root.f(3.0)
    self.assertAllEqual([a.numpy(), b.numpy(), c.numpy()], (1, 3.0, 4))

  def test_convert_to_input_signature(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)]
    )
    def func(x):
      return x

    root = autotrackable.AutoTrackable()
    root.f = func

    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    self.assertEqual([2], root.f([2]).numpy())

  def test_named_tuple(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class NamedTupleType(collections.namedtuple("NamedTupleType", ["a", "b"])):
      pass

    @def_function.function
    def f(x):
      return x.a + x.b

    f.get_concrete_function(
        NamedTupleType(
            a=tensor_spec.TensorSpec(None, dtypes.float32, name="a"),
            b=tensor_spec.TensorSpec(None, dtypes.float32, name="b"),
        )
    )
    obj = autotrackable.AutoTrackable()
    obj.__call__ = f
    if sys.version_info.major == 3 and sys.version_info.minor < 5:
      # TODO(allenl): figure out why this doesn't work in Python3.4
      self.skipTest("Not working in Python 3.4")
    imported = cycle(obj, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertAllClose(
        3.0,
        imported(
            NamedTupleType(
                a=constant_op.constant(1.0), b=constant_op.constant(2.0)
            )
        ),
    )

  def test_extra_args(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function
    def f(x):
      return math_ops.add(x["a"], 1.0)

    # Trigger a trace.
    f({"a": constant_op.constant(2.0)})

    obj = autotrackable.AutoTrackable()
    obj.__call__ = f
    imported = cycle(obj, cycles, use_cpp_bindings=use_cpp_bindings)

    self.assertEqual(4.0, imported({"a": 3.0}).numpy())

    with self.assertRaisesRegex(
        ValueError, "Could not find matching concrete function to call"
    ):
      imported({"a": 2.0, "b": 3.0})

  def test_shapes_available(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function(
        input_signature=[
            tensor_spec.TensorSpec([None, 3], dtypes.int32),
            tensor_spec.TensorSpec([None, 2], dtypes.int32),
        ]
    )
    def func(x, y):
      return array_ops.concat([x, y], axis=1)

    root = autotrackable.AutoTrackable()
    root.f = func

    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)

    imported_graph = root.f.get_concrete_function().graph
    input_x, input_y = imported_graph.inputs
    self.assertEqual([None, 3], input_x.shape.as_list())
    self.assertEqual([None, 2], input_y.shape.as_list())
    (output,) = imported_graph.outputs
    self.assertEqual([None, 5], output.shape.as_list())
    signature = root.signatures["serving_default"]
    self.assertEqual([None, 3], signature.inputs[0].shape.as_list())
    self.assertEqual([None, 2], signature.inputs[1].shape.as_list())
    self.assertEqual([None, 5], signature.outputs[0].shape.as_list())

  def test_variables_destroyed(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    v1 = variables.Variable(1.0)
    weak_v1 = weakref.ref(v1)
    root = checkpoint.Checkpoint(v=v1)
    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    del v1
    self.assertIsNone(weak_v1())
    weak_v2 = weakref.ref(root.v)
    del root
    self.assertIsNone(weak_v2())

  def test_variable_attributes_preserved(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    v = variables.Variable(
        1.0,
        trainable=False,
        synchronization=variables.VariableSynchronization.NONE,
        aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA,
    )
    self.assertEqual(variables.VariableSynchronization.NONE, v.synchronization)
    self.assertEqual(
        variables.VariableAggregation.ONLY_FIRST_REPLICA, v.aggregation
    )
    root = autotrackable.AutoTrackable()
    root.v = v
    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(False, root.v.trainable)
    self.assertEqual(
        variables.VariableSynchronization.NONE, root.v.synchronization
    )
    self.assertEqual(
        variables.VariableAggregation.ONLY_FIRST_REPLICA, root.v.aggregation
    )

  def test_captured_dataset(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class HasDataset(module.Module):

      def __init__(self):
        super(HasDataset, self).__init__()
        self.dataset = dataset_ops.Dataset.range(5).map(lambda x: x**2)

      @def_function.function
      def __call__(self, x):
        current_sum = array_ops.zeros([], dtype=dtypes.int64)
        for element in self.dataset:
          current_sum += x * element
        return current_sum

    root = HasDataset()
    self.assertEqual(
        3 * (1 + 4 + 9 + 16),
        root(constant_op.constant(3, dtype=dtypes.int64)).numpy(),
    )
    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(
        3 * (1 + 4 + 9 + 16),
        root(constant_op.constant(3, dtype=dtypes.int64)).numpy(),
    )

  def test_tuple_signature(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = checkpoint.Checkpoint()
    root.f = def_function.function(
        lambda: (array_ops.ones([]), array_ops.zeros([])), input_signature=()
    )
    root = cycle(
        root, cycles, signatures=root.f, use_cpp_bindings=use_cpp_bindings
    )
    self.assertEqual(
        ({"output_0": 1.0, "output_1": 0.0}),
        self.evaluate(root.signatures["serving_default"]()),
    )

  def test_version_info(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = checkpoint.Checkpoint()
    root = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(versions.__version__, root.tensorflow_version)
    self.assertEqual(versions.__git_version__, root.tensorflow_git_version)

  def test_load_grad_save(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = checkpoint.Checkpoint()
    root.v = variables.Variable(2.0)
    root.f = def_function.function(lambda x: root.v * x)
    root.g = def_function.function(root.f)
    for _ in range(cycles):
      with backprop.GradientTape() as tape:
        inp = constant_op.constant(2.0)
        tape.watch(inp)
        output = root.g(inp)
        self.assertAllClose(4.0, output)
      self.assertAllClose(2.0, tape.gradient(output, inp))
      root = cycle(root, 1, use_cpp_bindings=use_cpp_bindings)

  def test_destroy_resource(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    def get_handle():
      return resource_variable_ops.var_handle_op(
          shape=tensor_shape.as_shape([]),
          dtype=dtypes.float32,
          shared_name="my_var_name",
          name="my_var",
          container="my_container",
      )

    class MyResource(resource.TrackableResource):

      def _create_resource(self):
        return get_handle()

      def _initialize(self):
        resource_variable_ops.assign_variable_op(
            self.resource_handle, 1.0, name="assign"
        )

      def _destroy_resource(self):
        handle = get_handle()
        resource_variable_ops.destroy_resource_op(
            handle, ignore_lookup_error=True
        )

    class MyModel(autotrackable.AutoTrackable):

      def __init__(self):
        super(MyModel, self).__init__()
        self.resource = MyResource()

      @def_function.function(input_signature=[])
      def increase(self):
        handle = self.resource.resource_handle
        resource_variable_ops.assign_add_variable_op(
            handle, 10.0, name="assign_add"
        )
        return resource_variable_ops.read_variable_op(handle, dtypes.float32)

    root = MyModel()
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(11, imported.increase().numpy())  # Create the resource.

    handle = imported.resource.resource_handle

    # Delete the imported SaveModel. Since we explicitly set the deleter, it
    # should destroy the resource automatically.
    del imported

    # Try to destroy the resource again, should fail.
    with self.assertRaisesRegex(
        errors.NotFoundError, r"Resource .* does not exist."
    ):
      resource_variable_ops.destroy_resource_op(
          handle, ignore_lookup_error=False
      )

  def test_function_called_as_operation(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @framework_function.Defun(dtypes.float32)
    def inner(x):
      return x + 1.0

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.float32)]
    )
    def outer(x):
      return inner(x)

    root = module.Module()
    root.f = outer
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertAllClose(2.0, imported.f(constant_op.constant(1.0)))

  def test_ragged(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @def_function.function
    def f(x, c=1):
      """Returns Tensor x incremented by Python constant c."""
      return math_ops.add(x, c)

    for c in (1, 2, 3):
      _ = f.get_concrete_function(
          ragged_tensor.RaggedTensorSpec([None, None], dtype=dtypes.int32), c
      )

    obj = autotrackable.AutoTrackable()
    obj.f = f

    imported1 = cycle(
        obj, cycles, signatures={}, use_cpp_bindings=use_cpp_bindings
    )
    rt = ragged_factory_ops.constant([[1, 2], [3]])
    self.assertAllEqual(imported1.f(rt), [[2, 3], [4]])
    self.assertAllEqual(imported1.f(rt, 2), [[3, 4], [5]])
    self.assertAllEqual(imported1.f(rt, 3), [[4, 5], [6]])

    imported2 = cycle(obj, cycles, use_cpp_bindings=use_cpp_bindings)
    rt = ragged_factory_ops.constant([[1, 2], [3]])
    self.assertAllEqual(imported2.f(rt, 1), [[2, 3], [4]])
    self.assertAllEqual(imported2.f(rt, 2), [[3, 4], [5]])
    self.assertAllEqual(imported2.f(rt, 3), [[4, 5], [6]])

  def test_accepts_io_device(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    options = load_options.LoadOptions()
    self.assertIsNone(options.experimental_io_device)
    options = load_options.LoadOptions(experimental_io_device="/job:localhost")
    self.assertEqual("/job:localhost", options.experimental_io_device)

  def _custom_saveable_object(self, cycles, use_cpp_bindings):
    if context.is_tfrt_enabled():
      self.skipTest("Disable due to b/190539415.")
    root = autotrackable.AutoTrackable()
    root.table = lookup_ops.MutableHashTable(dtypes.string, dtypes.float32, -1)
    root.table.insert("foo", 15)
    root.table2 = lookup_ops.MutableHashTable(dtypes.string, dtypes.float32, -1)
    root.table2.insert("idk", 21)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.string)]
    )
    def lookup(key):
      return root.table.lookup(key)

    root.lookup = lookup

    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(self.evaluate(imported.lookup("foo")), 15)
    self.assertEqual(self.evaluate(imported.lookup("idk")), -1)

    if not saveable_compat.force_checkpoint_conversion_enabled():
      self.assertEqual(
          {"table"}, imported.table._self_saveable_object_factories.keys()
      )

  def test_load_custom_saveable_object(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    self._custom_saveable_object(cycles, use_cpp_bindings=use_cpp_bindings)

  def test_load_custom_saveable_object_ckpt_conversion(
      self, cycles, use_cpp_bindings
  ):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    # Tests custom saveable object with checkpoint conversion enabled (forces
    # Trackable-based checkpoint implementation).
    saveable_compat.force_checkpoint_conversion()
    self._custom_saveable_object(cycles, use_cpp_bindings=use_cpp_bindings)

  def test_load_resource_with_dependency(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    # Test with StaticHashTable, which has a _initializer attribute that tracks
    # the Asset vocab table.

    class MyLookupModel(autotrackable.AutoTrackable):

      def __init__(self, vocab_file):
        vocab_initializer = lookup_ops.TextFileInitializer(
            vocab_file,
            key_dtype=dtypes.string,
            key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
            value_dtype=dtypes.int64,
            value_index=lookup_ops.TextFileIndex.LINE_NUMBER,
        )
        self._vocab_table = lookup_ops.StaticHashTable(
            vocab_initializer, default_value=-1
        )

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec((None,), dtypes.string)]
      )
      def __call__(self, inputs):
        return self._vocab_table.lookup(inputs)

    vocab_file = self._make_asset("\n".join(["a", "b", "c", "d"]))
    root = MyLookupModel(vocab_file)
    imported = cycle(root, cycles, use_cpp_bindings=use_cpp_bindings)
    file_io.delete_file(vocab_file)
    self.assertAllEqual(imported(constant_op.constant(["d", "b"])), [3, 1])

  def test_custom_gradients(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    @custom_gradient.custom_gradient
    def log1pexp(x):
      e = math_ops.exp(x)

      def grad(dy):
        return dy * e  # incorrect to check the custom gradients is respected.

      return math_ops.log(1 + e), grad

    @def_function.function
    def g(x):
      y = log1pexp(x)

      @def_function.function
      def g_nest():
        return log1pexp(y)

      return g_nest()

    @def_function.function
    def f(x):
      return log1pexp(g(x * x))

    v = variables.Variable(1.)

    with backprop.GradientTape() as tape2:
      with backprop.GradientTape() as tape:
        tape.watch(v)
        y = f(v)
        expected_grads = tape.gradient(y, v)
      expected_grad_grads = tape2.gradient(expected_grads, v)

    root = autotrackable.AutoTrackable()
    root.f = f
    loaded = cycle(
        root,
        cycles,
        save_option=save_options.SaveOptions(
            experimental_custom_gradients=True
        ),
        use_cpp_bindings=use_cpp_bindings,
    )
    with backprop.GradientTape() as tape2:
      with backprop.GradientTape() as tape:
        tape.watch(v)
        y = loaded.f(v)
        grads = tape.gradient(y, v)
      grad_grads = tape2.gradient(grads, v)

    self.assertAllClose(grads, expected_grads)
    self.assertAllClose(grad_grads, expected_grad_grads)

  def test_custom_gradients_with_none_grad(self, cycles, use_cpp_bindings):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    # https://github.com/google/jax/issues/7123

    @custom_gradient.custom_gradient
    def f(params, state):
      def grad_fn(*args):
        return args

      return (params, state), grad_fn

    @def_function.function(
        input_signature=[
            tensor_spec.TensorSpec([], dtypes.float32),
            tensor_spec.TensorSpec([], dtypes.int32),
        ]
    )
    def predict(params, state):
      return f(params, state)

    params = variables.Variable(1.0)
    # None grads only appear when state is an int.
    state = constant_op.constant(3, dtype=dtypes.int32)
    with backprop.GradientTape() as tape:
      tape.watch(params)
      y = predict(params, state)
      expected_grads = tape.gradient(y, params)

    root = autotrackable.AutoTrackable()
    root.fn = predict
    loaded = cycle(
        root,
        cycles,
        save_option=save_options.SaveOptions(
            experimental_custom_gradients=True
        ),
        use_cpp_bindings=use_cpp_bindings,
    )

    with backprop.GradientTape() as tape:
      tape.watch(params)
      y = loaded.fn(params, state)
      grads = tape.gradient(y, params)

    self.assertAllClose(grads, expected_grads)

  def test_custom_gradients_with_none_grad_and_partial_shape(
      self, cycles, use_cpp_bindings
  ):
    # TODO(b/264869228) Fix LoadTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    # https://github.com/google/jax/issues/7123

    @custom_gradient.custom_gradient
    def f(params, state):
      def grad_fn(*args):
        return args

      return (params, state), grad_fn

    @def_function.function(
        input_signature=[
            tensor_spec.TensorSpec(None, dtypes.float32),
            tensor_spec.TensorSpec(None, dtypes.int32),
        ]
    )
    def predict(params, state):
      return f(params, state)

    params = variables.Variable(1.0)
    # None grads only appear when state is an int.
    state = constant_op.constant(3, dtype=dtypes.int32)
    with backprop.GradientTape() as tape:
      tape.watch(params)
      y = predict(params, state)
      expected_grads = tape.gradient(y, params)

    root = autotrackable.AutoTrackable()
    root.fn = predict
    loaded = cycle(
        root,
        cycles,
        save_option=save_options.SaveOptions(
            experimental_custom_gradients=True
        ),
        use_cpp_bindings=use_cpp_bindings,
    )

    with backprop.GradientTape() as tape:
      tape.watch(params)
      y = loaded.fn(params, state)
      grads = tape.gradient(y, params)

    self.assertAllClose(grads, expected_grads)


@parameterized.named_parameters(*_test_params())
class SingleCycleTests(test.TestCase, parameterized.TestCase):

  def test_load_with_tags(self, use_cpp_bindings):
    if use_cpp_bindings:
      self.skipTest("Cpp bindings do not support Tags.")
    root = autotrackable.AutoTrackable()
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)

    with self.assertRaises(ValueError):
      load.load(path, tags=[tag_constants.EVAL])
    load.load(path, tags=[tag_constants.SERVING])
    load.load(path, tags=tag_constants.SERVING)
    load.load(path, tags=set([tag_constants.SERVING]))

  def test_save_load_contains_with_fspath(self, use_cpp_bindings):
    if use_cpp_bindings:
      self.skipTest("Cpp bindings cannot work with pathlib object.")
    root = autotrackable.AutoTrackable()
    path = pathlib.Path(tempfile.mkdtemp(prefix=self.get_temp_dir()))
    save.save(root, path)
    self.assertTrue(loader_impl.contains_saved_model(path))

    test_load(path, use_cpp_bindings=use_cpp_bindings)

  def test_single_restore_op_used(self, use_cpp_bindings):
    # TODO(b/264869753) Fix SingleCycleTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = module.Module()
    root.v1 = variables.Variable(1.0)
    root.v2 = variables.Variable(2.0)
    root.v3 = variables.Variable(3.0)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)
    restore_count = 0

    def _count_restores(op_type, *unused_args, **unused_kwargs):
      nonlocal restore_count
      if op_type == b"RestoreV2":
        restore_count += 1

    op_callbacks.add_op_callback(_count_restores)
    save.save(root, path)
    test_load(path, use_cpp_bindings=use_cpp_bindings)
    op_callbacks.remove_op_callback(_count_restores)
    self.assertEqual(1, restore_count)

  def test_docstring_examples(self, use_cpp_bindings):
    # TODO(b/264869753) Fix SingleCycleTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    exported = checkpoint.Checkpoint(v=variables.Variable(3.0))
    exported.f = def_function.function(
        lambda x: exported.v * x,
        input_signature=[
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32)
        ],
    )
    save.save(exported, path)
    imported = test_load(path)
    self.assertEqual(3.0, imported.v.numpy())
    self.assertEqual(6.0, imported.f(x=constant_op.constant(2.0)).numpy())

    save.save(exported, path, exported.f.get_concrete_function())
    imported = test_load(path, use_cpp_bindings=use_cpp_bindings)
    f = imported.signatures["serving_default"]
    self.assertAllEqual(
        [[-3.0]], f(x=constant_op.constant([[-1.0]]))["output_0"].numpy()
    )

  def test_object_with_extra_dependencies(self, use_cpp_bindings):
    # TODO(b/264869753) Fix SingleCycleTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class Extra(autotrackable.AutoTrackable):

      def _trackable_children(self, save_type, **kwargs):
        children = super(Extra, self)._trackable_children(save_type, **kwargs)
        children["a"] = variables.Variable(5.0)
        return children

    root = Extra()
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)
    imported = test_load(path)
    self.assertEqual(5, self.evaluate(imported.a))

  def test_save_cached_variable(self, use_cpp_bindings):
    # TODO(b/264869753) Fix SingleCycleTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    with ops.Graph().as_default(), session_lib.Session() as session:
      obj = autotrackable.AutoTrackable()
      obj.v = variables.Variable(2.0, caching_device=lambda op: op.device)
      obj.w = variables.Variable(3.0)
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

      session.run(increment_v(constant_op.constant(3.0)))  # generate signatures
      self.assertAllClose(8, total())
      self.assertAllClose(13, wrapped_total(constant_op.constant(5.0)))

      obj.total = total
      obj.wrapped_total = wrapped_total.get_concrete_function()
      obj.increment_v = increment_v

      save_dir = os.path.join(self.get_temp_dir(), "saved_model")
      save.save(obj, save_dir, signatures=total.get_concrete_function())
      imported = test_load(save_dir)
      session.run(variables.global_variables_initializer())
      self.assertAllClose(8, imported.total())
      session.run(imported.increment_v(4))
      self.assertAllClose(12, imported.total())
      self.assertAllClose(15, imported.wrapped_total(constant_op.constant(3.0)))
      self.assertAllClose(
          {"output_0": 12}, imported.signatures["serving_default"]()
      )

    # Try loading and running the function in eager mode
    imported = test_load(save_dir)
    self.assertAllClose(8, imported.total())
    imported.increment_v(5)
    self.assertAllClose(13, imported.total())
    self.assertAllClose(13.5, imported.wrapped_total(constant_op.constant(0.5)))
    self.assertAllClose(
        {"output_0": 13}, imported.signatures["serving_default"]()
    )

  # TODO(allenl, kkb): Use the new memory checker here once it's fast enough (3
  # iterations took hundreds of seconds). It would be really nice to check
  # allocations at a lower level.
  @test_util.assert_no_new_pyobjects_executing_eagerly
  def test_functions_cleaned(self, use_cpp_bindings):
    # TODO(b/264869753) Fix SingleCycleTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    if sys.version_info.major < 3:
      self.skipTest("Not working in Python 2")
    if sys.version_info.major == 3 and sys.version_info.minor == 11:
      # TODO(b/264948173)
      self.skipTest("Not working in Python 3.11")
    root = module.Module()
    root.v = variables.Variable(1.0)
    root.f = def_function.function(
        lambda x: x + root.v,
        input_signature=[
            tensor_spec.TensorSpec(shape=[], dtype=dtypes.float32)
        ],
    )
    cycle(root, 1, use_cpp_bindings=use_cpp_bindings)

  def test_load_partial_object(self, use_cpp_bindings):
    # TODO(b/264869753) Fix SingleCycleTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = module.Module()
    root.variables_holder = module.Module()
    root.variables_holder.v = variables.Variable(1.0)

    class Adder(module.Module):

      @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=[])])
      def __call__(self, y):
        root.variables_holder.v.assign_add(y)
        return 1

    root.adder = Adder()

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)

    imported = load.load_partial(
        save_dir, ["root.variables_holder.v", "root.adder"]
    )
    v = imported["root.variables_holder.v"]
    adder = imported["root.adder"]
    self.assertEqual(self.evaluate(v), 1)
    adder(5)
    self.assertEqual(self.evaluate(v), 6)

    with self.assertRaisesRegex(
        ValueError, "does not include all required objects for loading"
    ):
      imported = load.load_partial(save_dir, ["root.adder"])

  def test_load_partial_checkpoint(self, use_cpp_bindings):
    # TODO(b/264869753) Fix SingleCycleTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    root = module.Module()
    root.variables_holder = module.Module()
    root.variables_holder.v = variables.Variable(1.0)

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)

    loaded = module.Module()
    loaded.v = variables.Variable(2.0)

    load.load_partial(
        save_dir,
        {"root": loaded},
        options=load_options.LoadOptions(allow_partial_checkpoint=True),
    )
    self.assertEqual(loaded.variables_holder.v.numpy(), 1)
    with self.assertRaisesRegex(AssertionError, "were not bound"):
      load.load_partial(save_dir, {"root": loaded})

  def test_call_untraced_function_raises_error(self, use_cpp_bindings):
    # TODO(b/264869753) Fix SingleCycleTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class ObjWithFunction(module.Module):

      @def_function.function
      def foo(self, a):
        return a

    root = ObjWithFunction()
    with self.assertLogs(level="WARNING") as logs:
      loaded = cycle(root, 1, use_cpp_bindings=use_cpp_bindings)

    expected_save_message = (
        "WARNING:absl:Found untraced functions such as foo while saving "
        "(showing 1 of 1). These functions will not be directly callable after "
        "loading."
    )
    self.assertIn(expected_save_message, logs.output)

    with self.assertRaisesRegex(
        ValueError, "Found zero restored functions for caller function."
    ):
      loaded.foo(1)

  def test_restored_function_execute_eagerly(self, use_cpp_bindings):
    # TODO(b/264869753) Fix SingleCycleTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    try:
      def_function.run_functions_eagerly(True)

      class MyModel(module.Module):

        @def_function.function
        def __call__(self, inputs, training=False):
          return math_ops.multiply(0.5, inputs)

      model = MyModel()
      model.__call__.get_concrete_function(
          tensor_spec.TensorSpec([None], dtypes.float32)
      )
      loaded = cycle(model, 1, use_cpp_bindings=use_cpp_bindings)

      # Calling the function should not throw an exception.
      loaded(constant_op.constant([1.0]))

    finally:
      def_function.run_functions_eagerly(False)

  def test_restored_model_concrete_function_is_deterministic(
      self, use_cpp_bindings
  ):
    # TODO(b/264869753) Fix SingleCycleTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    previous_concrete_function = None
    for _ in range(100):

      class MyModel(module.Module):

        @def_function.function
        def __call__(self, x):
          return x * constant_op.constant(3.0)

      model = MyModel()
      model(array_ops.ones((7, 3), dtype=dtypes.float32))
      model.__call__.get_concrete_function(
          tensor_spec.TensorSpec([None, 3], dtypes.float32)
      )
      loaded = cycle(model, 1, use_cpp_bindings=use_cpp_bindings)

      # Ensure the newly loaded concrete function is the same as the previous
      # after a cycle of serialization / deserialization.
      new_concrete_function = loaded.__call__.get_concrete_function(
          tensor_spec.TensorSpec([None, 3], dtypes.float32)
      )
      if previous_concrete_function is not None:
        self.assertEqual(
            previous_concrete_function.pretty_printed_signature(),
            new_concrete_function.pretty_printed_signature(),
        )

      previous_concrete_function = new_concrete_function

  def test_garbage_collection_capturable_resource_doesnt_raise_exception(
      self, use_cpp_bindings
  ):
    # TODO(b/264869753) Fix SingleCycleTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    model = module.Module()
    model.mapping = lookup_ops.StaticHashTable(
        lookup_ops.KeyValueTensorInitializer(
            keys=math_ops.range(1, dtype=dtypes.int32), values=["foo"]
        ),
        "default_value",
    )
    loaded = cycle(model, 1, use_cpp_bindings=use_cpp_bindings)
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

  def test_captured_dataset_with_asset(self, use_cpp_bindings):
    # TODO(b/264869753) Fix SingleCycleTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class HasDataset(module.Module):

      def __init__(self, temp_dir, file_name):
        super(HasDataset, self).__init__()
        file = os.path.join(temp_dir, file_name)
        with tf_record.TFRecordWriter(file, "GZIP") as f:
          for v in ["a", "aa", "aaa"]:
            f.write(str(v))
        self.dataset = readers.TFRecordDataset([file], compression_type="GZIP")

      @def_function.function
      def __call__(self, x):
        current_sum = array_ops.zeros([], dtype=dtypes.int32)
        for element in self.dataset:
          current_sum += x * string_ops.string_length(element)
        return current_sum

    temp_dir = self.get_temp_dir()
    file_name = "tf_record_asset.tfrecord.gz"
    root = HasDataset(temp_dir, file_name)
    self.assertEqual(
        18,  # 3 * (1 + 2 + 3)
        root(constant_op.constant(3, dtype=dtypes.int32)).numpy(),
    )

    save_dir = os.path.join(self.get_temp_dir(), "save_dir")
    save.save(root, save_dir)

    file_io.delete_file(os.path.join(temp_dir, file_name))
    asset_path = os.path.join(save_dir, "assets/{}".format(file_name))
    self.assertTrue(file_io.file_exists(asset_path))
    load_dir = os.path.join(self.get_temp_dir(), "load_dir")
    file_io.rename(save_dir, load_dir)

    loaded = test_load(load_dir, use_cpp_bindings=use_cpp_bindings)
    self.assertEqual(
        18,  # 3 * (1 + 2 + 3)
        loaded(constant_op.constant(3, dtype=dtypes.int32)).numpy(),
    )


# TODO(b/264882754) Support Cpp bindings DeferredInitModuleVariablesTest
class DeferredInitModuleVariablesTest(test.TestCase, parameterized.TestCase):

  def test_deferred_init_module_variables(self):
    """Defer initialization of variables in a module to the load stage."""

    class MyModule(module.Module):

      def __init__(self, size):
        super().__init__()
        self.size = size
        # variable initialized by a Tensor-compatible value
        self.w1 = variables.Variable(
            constant_op.constant(1., shape=[self.size]), trainable=False)
        # variable initialized by a function
        self.w2 = variables.Variable(
            lambda: constant_op.constant(2., shape=[self.size]))
        # variable instantiated lazily in call()
        self.w3 = None

      def call(self):
        if self.w3 is None:
          self.w3 = variables.Variable(
              constant_op.constant(3., shape=[self.size]))
        for w in (self.w1, self.w2, self.w3):
          w.assign_add(constant_op.constant(1., shape=[self.size]))
        return self.w1, self.w2, self.w3

    def export_initializer(initial_value, export_dir):

      class Initializer(module.Module):

        @def_function.function(input_signature=[])
        def call(self):
          if callable(initial_value):
            return initial_value()
          return initial_value

      save.save(Initializer(), export_dir)

    def create_and_save_module(weight_size):

      initial_values = {}  # For storing initial_value of created variables

      def variable_creator(next_creator, **kwargs):
        variable = next_creator(**kwargs)
        variable_name = variable.name
        if ":" in variable_name:
          variable_name = variable_name[:variable_name.index(":")]
        initial_values[variable_name] = kwargs["initial_value"]
        return variable

      export_dir = self.create_tempdir().full_path

      with ops.Graph().as_default():
        with variable_scope.variable_creator_scope(variable_creator):
          exported = MyModule(weight_size)
          exported.call = def_function.function(input_signature=[])(
              exported.call)

          module_dir = f"{export_dir}/module"
          file_io.recursive_create_dir(module_dir)
          save.save_and_return_nodes(
              exported, module_dir, experimental_skip_checkpoint=True)

      # Save the initializer of the created variables.
      for variable_name, initial_value in initial_values.items():
        export_initializer(initial_value,
                           f"{export_dir}/variables/{variable_name}")

      return export_dir

    def load_and_run_module(export_dir, weight_size):

      # pylint: disable=unused-argument
      def layer_variable_creator(next_creator, **kwargs):
        variable_dir = f"{export_dir}/variables/{kwargs['name']}"
        initializer = load.load(variable_dir)
        kwargs["initial_value"] = initializer.call
        variable = resource_variable_ops.ResourceVariable(**kwargs)
        return variable

      with ops.Graph().as_default():
        with variable_scope.variable_creator_scope(layer_variable_creator):
          imported = load.load(
              f"{export_dir}/module",
              options=load_options.LoadOptions(
                  experimental_skip_checkpoint=True))
        outputs = imported.call()

        with self.cached_session() as sess:
          variables.global_variables_initializer().run()
          # Check if variables work as expected across multiple iterations.
          for i in range(3):
            np_outputs = sess.run(outputs)
            for j, np_output in enumerate(np_outputs):
              self.assertAllClose(np_output, np.full(weight_size, i + j + 2))

    # The size of the serialized content (both module and variables) stays
    # small even with a large weight_size as the initial values are not stored
    # in checkpoints.
    weight_size = 1024
    export_dir = create_and_save_module(weight_size)
    load_and_run_module(export_dir, weight_size)

  def _make_asset(self, contents):
    fd, filename = tempfile.mkstemp(prefix=self.get_temp_dir())
    with os.fdopen(fd, "w") as f:
      f.write(contents)
    return filename

  @parameterized.named_parameters(*_test_params())
  def test_assets(self, use_cpp_bindings):
    # TODO(b/264882754) Fix DeferredInitModuleVariablesTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")

    class MyLookupModel(autotrackable.AutoTrackable):

      def __init__(self, vocab_file):
        vocab_initializer = lookup_ops.TextFileInitializer(
            vocab_file,
            key_dtype=dtypes.string,
            key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
            value_dtype=dtypes.int64,
            value_index=lookup_ops.TextFileIndex.LINE_NUMBER,
        )
        self._vocab_table = lookup_ops.StaticHashTable(
            vocab_initializer, default_value=-1
        )

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec((None,), dtypes.string)]
      )
      def __call__(self, inputs):
        return self._vocab_table.lookup(inputs)

    vocab_file = self._make_asset("\n".join(["a", "b", "c", "d"]))
    root = MyLookupModel(vocab_file)

    save_dir = os.path.join(self.get_temp_dir(), "save_dir")
    save.save_and_return_nodes(
        root, save_dir, experimental_skip_checkpoint=True
    )
    file_io.delete_file(vocab_file)
    load_dir = os.path.join(self.get_temp_dir(), "load_dir")
    file_io.rename(save_dir, load_dir)

    imported = test_load(
        load_dir,
        options=load_options.LoadOptions(experimental_skip_checkpoint=True),
        use_cpp_bindings=use_cpp_bindings,
    )
    self.assertAllEqual(imported(constant_op.constant(["d", "b"])), [3, 1])


class _TestModel(module.Module):

  def __init__(self, rows, cols):
    super().__init__()
    self.rows = rows
    self.cols = cols
    self.table = None

  def __call__(self, x):
    with ops.device("/cpu:0"):
      self.table = variables.Variable(
          constant_op.constant(1.0, shape=[self.rows, self.cols])
      )
      x = math_ops.matmul(self.table, x)
      x = math_ops.reduce_sum(x, axis=0)
    return x


@parameterized.named_parameters(*_test_params())
class SavedModelLoadMemoryTests(test.TestCase, parameterized.TestCase):

  @test_util.run_gpu_only
  def test_no_oom_loading_large_tenor(self, use_cpp_bindings):
    # TODO(b/264882686) Fix DeferredInitModuleVariablesTest
    if use_cpp_bindings:
      self.skipTest("Not implemented for cpp.")
    if not config.get_soft_device_placement():
      self.skipTest("This test only works for soft device placement is on")
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    ncols = 16
    nrows = 32
    model = _TestModel(rows=nrows, cols=ncols)
    x = array_ops.zeros(shape=(ncols, 2), dtype=dtypes.float32)
    y = model(x)
    save.save(
        model,
        save_dir,
        options=save_options.SaveOptions(
            experimental_variable_policy=save_options.VariablePolicy.SAVE_VARIABLE_DEVICES
        ),
    )
    loaded_on_cpu = test_load(
        path=save_dir,
        options=load_options.LoadOptions(
            experimental_variable_policy=save_options.VariablePolicy.SAVE_VARIABLE_DEVICES
        ),
        use_cpp_bindings=use_cpp_bindings,
    )
    loaded_on_gpu = test_load(save_dir)
    self.assertIn("CPU", loaded_on_cpu.table.device)
    self.assertIn("GPU", loaded_on_gpu.table.device)


if __name__ == "__main__":
  test.main()
