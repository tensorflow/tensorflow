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
"""Tests for checkpointable object SavedModel export."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import export
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import adam
from tensorflow.python.training.checkpointable import tracking


class _ModelWithOptimizer(training.Model):

  def __init__(self):
    super(_ModelWithOptimizer, self).__init__()
    self.dense = core.Dense(1)
    self.optimizer = adam.AdamOptimizer(0.01)

  @def_function.function(
      input_signature=(tensor_spec.TensorSpec([None, 2], dtypes.float32),
                       tensor_spec.TensorSpec([None], dtypes.float32)))
  def call(self, x, y):
    with backprop.GradientTape() as tape:
      loss = math_ops.reduce_mean((self.dense(x) - y) ** 2.)
    trainable_variables = self.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    return {"loss": loss}


class ExportTest(test.TestCase):

  def _import_and_infer(
      self, export_dir, inputs,
      signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
    """Import a SavedModel into a TF 1.x-style graph and run `signature_key`."""
    graph = ops.Graph()
    with graph.as_default(), self.session(graph) as session:
      model = loader.load(session, [], export_dir)
      signature = model.signature_def[signature_key]
      self.assertEqual(set(inputs.keys()), set(signature.inputs.keys()))
      feed_dict = {}
      for arg_name in inputs.keys():
        feed_dict[graph.get_tensor_by_name(signature.inputs[arg_name].name)] = (
            inputs[arg_name])
      output_dict = {}
      for output_name, output_tensor_info in signature.outputs.items():
        output_dict[output_name] = graph.get_tensor_by_name(
            output_tensor_info.name)
      return session.run(output_dict, feed_dict=feed_dict)

  def test_method_export_signature(self):
    root = tracking.Checkpointable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    root.f(constant_op.constant(1.))
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    export.export(root, export_dir, root.f)
    self.assertEqual(
        {"output_0": 2.},
        self._import_and_infer(export_dir, {"x": 1.}))

  def test_method_export_concrete(self):
    root = tracking.Checkpointable()
    root.f = def_function.function(
        lambda z: {"out": 2. * z})
    root.f(constant_op.constant(1.))
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    export.export(
        root,
        export_dir,
        {"non_default_key": root.f.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.float32))})
    self.assertEqual(
        {"out": 2.},
        self._import_and_infer(
            export_dir, {"z": 1.}, signature_key="non_default_key"))

  def test_non_concrete_error(self):
    root = tracking.Checkpointable()
    root.f = def_function.function(lambda x: 2. * x)
    root.f(constant_op.constant(1.))
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(
        ValueError, "must be converted to concrete functions"):
      export.export(root, export_dir, root.f)

  def test_nested_inputs(self):
    root = tracking.Checkpointable()
    root.f = def_function.function(
        lambda x: 2. * x[0],
        input_signature=([tensor_spec.TensorSpec(None, dtypes.float32),
                          tensor_spec.TensorSpec(None, dtypes.float32)],))
    root.f([constant_op.constant(1.), constant_op.constant(1.)])
    # Concrete functions must always have uniquely named Tensor inputs. Export
    # relies on this.
    with self.assertRaisesRegexp(
        ValueError, "two arguments named 'x'"):
      root.f.get_concrete_function()

  def test_nested_outputs(self):
    root = tracking.Checkpointable()
    root.f = def_function.function(lambda x: (2. * x, (3. * x, 4. * x)))
    root.f(constant_op.constant(1.))
    to_export = root.f.get_concrete_function(constant_op.constant(1.))
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(
        ValueError, "non-flat outputs"):
      export.export(root, export_dir, to_export)

  def test_nested_dict_outputs(self):
    root = tracking.Checkpointable()
    root.f = def_function.function(
        lambda x: {"a": 2. * x, "b": (3. * x, 4. * x)})
    root.f(constant_op.constant(1.))
    to_export = root.f.get_concrete_function(constant_op.constant(1.))
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(
        ValueError, "dictionary containing non-Tensor value"):
      export.export(root, export_dir, to_export)

  def test_variable(self):
    root = tracking.Checkpointable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(
        lambda x: root.v1 * root.v2 * x)
    root.f(constant_op.constant(1.))
    to_export = root.f.get_concrete_function(constant_op.constant(1.))
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    export.export(root, export_dir, to_export)
    self.assertAllEqual({"output_0": 12.},
                        self._import_and_infer(export_dir, {"x": 2.}))

  def test_optimizer(self):
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    model = _ModelWithOptimizer()
    first_loss = model(x, y)
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    export.export(model, export_dir, model.call)
    second_loss = model(x, y)
    self.assertNotEqual(first_loss, second_loss)
    self.assertAllClose(
        second_loss,
        self._import_and_infer(export_dir, {"x": [[3., 4.]], "y": [2.]}))

  def test_trivial_export_exception(self):
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(ValueError, "signature"):
      export.export(tracking.Checkpointable(), export_dir)

  def test_single_method_default_signature(self):
    model = _ModelWithOptimizer()
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    model(x, y)
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    export.export(model, export_dir)
    self.assertIn("loss",
                  self._import_and_infer(export_dir,
                                         {"x": [[3., 4.]], "y": [2.]}))

  def test_single_function_default_signature(self):
    model = tracking.Checkpointable()
    model.f = def_function.function(lambda: 3., input_signature=())
    model.f()
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    export.export(model, export_dir)
    self.assertAllClose({"output_0": 3.},
                        self._import_and_infer(export_dir, {}))

  def test_ambiguous_signatures(self):
    model = _ModelWithOptimizer()
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    model(x, y)
    model.second_function = def_function.function(lambda: 1.)
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(ValueError, "call.*second_function"):
      export.export(model, export_dir)


class MemoryTests(test.TestCase):

  def setUp(self):
    self._model = _ModelWithOptimizer()

  @test_util.assert_no_garbage_created
  def test_no_reference_cycles(self):
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    self._model(x, y)
    if sys.version_info[0] < 3:
      # TODO(allenl): debug reference cycles in Python 2.x
      self.skipTest("This test only works in Python 3+. Reference cycles are "
                    "created in older Python versions.")
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    export.export(self._model, export_dir, self._model.call)


if __name__ == "__main__":
  test.main()
