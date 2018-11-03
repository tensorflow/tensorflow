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

from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.saved_model import export
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training.checkpointable import tracking


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
    root.f = def_function.function(lambda x: 2. * x[0])
    root.f([constant_op.constant(1.)])
    to_export = root.f.get_concrete_function(
        [constant_op.constant(1.), constant_op.constant(2.)])
    export_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(
        ValueError, "non-unique argument names"):
      export.export(root, export_dir, to_export)

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


if __name__ == "__main__":
  test.main()
