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
"""Tests for checkpointable object SavedModel save."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras.layers import core
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import adam
from tensorflow.python.training.checkpointable import tracking
from tensorflow.python.training.checkpointable import util


class _ModelWithOptimizer(util.Checkpoint):

  def __init__(self):
    self.dense = core.Dense(1)
    self.optimizer = adam.AdamOptimizer(0.01)

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


def _import_and_infer(
    save_dir, inputs,
    signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
  """Import a SavedModel into a TF 1.x-style graph and run `signature_key`."""
  graph = ops.Graph()
  with graph.as_default(), session_lib.Session() as session:
    model = loader.load(session, [tag_constants.SERVING], save_dir)
    signature = model.signature_def[signature_key]
    assert set(inputs.keys()) == set(signature.inputs.keys())
    feed_dict = {}
    for arg_name in inputs.keys():
      feed_dict[graph.get_tensor_by_name(signature.inputs[arg_name].name)] = (
          inputs[arg_name])
    output_dict = {}
    for output_name, output_tensor_info in signature.outputs.items():
      output_dict[output_name] = graph.get_tensor_by_name(
          output_tensor_info.name)
    return session.run(output_dict, feed_dict=feed_dict)


class SaveTest(test.TestCase):

  def test_method_save_signature(self):
    root = tracking.Checkpointable()
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
    root = tracking.Checkpointable()
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

  def test_non_concrete_error(self):
    root = tracking.Checkpointable()
    root.f = def_function.function(lambda x: 2. * x)
    root.f(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(
        ValueError, "must be converted to concrete functions"):
      save.save(root, save_dir, root.f)

  def test_nested_inputs(self):
    root = tracking.Checkpointable()
    root.f = def_function.function(
        lambda x: 2. * x[0],
        input_signature=([tensor_spec.TensorSpec(None, dtypes.float32),
                          tensor_spec.TensorSpec(None, dtypes.float32)],))
    root.f([constant_op.constant(1.), constant_op.constant(1.)])
    # Concrete functions must always have uniquely named Tensor inputs. Save
    # relies on this.
    with self.assertRaisesRegexp(
        ValueError, "two arguments named 'x'"):
      root.f.get_concrete_function()

  def test_nested_outputs(self):
    root = tracking.Checkpointable()
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
    root = tracking.Checkpointable()
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

  def test_trivial_save_exception(self):
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(ValueError, "signature"):
      save.save(tracking.Checkpointable(), save_dir)

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
    model = tracking.Checkpointable()
    model.f = def_function.function(lambda: 3., input_signature=())
    model.f()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(model, save_dir)
    self.assertAllClose({"output_0": 3.},
                        _import_and_infer(save_dir, {}))

  def test_ambiguous_signatures(self):
    model = _ModelWithOptimizer()
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    model.call(x, y)
    model.second_function = def_function.function(lambda: 1.)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(ValueError, "call.*second_function"):
      save.save(model, save_dir)

  def test_no_signature(self):

    class Model(util.Checkpoint):

      def call(self, inputs):
        return inputs * 2.

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    model = Model()
    with self.assertRaisesRegexp(
        ValueError, "no @tf.function-decorated methods"):
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

    class Adder(util.Checkpoint):

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


class AssetTests(test.TestCase):

  def setUp(self):
    super(AssetTests, self).setUp()
    self._vocab_path = os.path.join(self.get_temp_dir(), "vocab.txt")
    with open(self._vocab_path, "w") as f:
      f.write("alpha\nbeta\ngamma\n")

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
    root = tracking.Checkpointable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    root.asset = tracking.TrackableAsset(self._vocab_path)

    export_dir = os.path.join(self.get_temp_dir(), "save_dir")
    save.save(root, export_dir)
    self.assertAllClose(
        {"output_0": [0.2]},
        _import_and_infer(export_dir, {"x": [0.1]}))

  def test_sensible_graph_building_exception(self):
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
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(AssertionError, "enable_eager_execution"):
        save.save(root, export_dir)


class MemoryTests(test.TestCase):

  def setUp(self):
    self._model = _ModelWithOptimizer()

  @test_util.assert_no_garbage_created
  def test_no_reference_cycles(self):
    self.skipTest("b/121159261")
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    self._model.call(x, y)
    if sys.version_info[0] < 3:
      # TODO(allenl): debug reference cycles in Python 2.x
      self.skipTest("This test only works in Python 3+. Reference cycles are "
                    "created in older Python versions.")
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(self._model, save_dir, self._model.call)


if __name__ == "__main__":
  test.main()
