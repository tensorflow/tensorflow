# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for importing a TF v1-style SavedModel when executing eagerly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import simple_save
from tensorflow.python.saved_model import utils_impl


class LoadTest(test.TestCase):

  def _v1_single_metagraph_saved_model(self, use_resource):
    export_graph = ops.Graph()
    with export_graph.as_default():
      start = array_ops.placeholder(
          shape=[None], dtype=dtypes.float32, name="start")
      if use_resource:
        v = resource_variable_ops.ResourceVariable(3.)
      else:
        v = variables.RefVariable(3.)
      local_variable = variables.VariableV1(
          1.,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          trainable=False,
          use_resource=True)
      output = array_ops.identity(start * v * local_variable, name="output")
      with session_lib.Session() as session:
        session.run([v.initializer, local_variable.initializer])
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"start": start},
            outputs={"output": output},
            legacy_init_op=local_variable.initializer)
    return path

  def test_resource_variable_import(self):
    imported = load.load(self._v1_single_metagraph_saved_model(
        use_resource=True))
    fn = imported.signatures["serving_default"]
    with self.assertRaisesRegexp(TypeError, "positional"):
      fn(constant_op.constant(2.))
    self.assertEqual({"output": 6.},
                     self.evaluate(fn(start=constant_op.constant(2.))))
    self.assertAllEqual([3., 1.], self.evaluate(imported.variables))
    imported.variables[0].assign(4.)
    self.assertEqual({"output": 8.},
                     self.evaluate(fn(start=constant_op.constant(2.))))
    imported.variables[1].assign(2.)
    self.assertEqual({"output": 24.},
                     self.evaluate(fn(start=constant_op.constant(3.))))
    self.assertTrue(imported.variables[0].trainable)
    self.assertFalse(imported.variables[1].trainable)
    with backprop.GradientTape() as tape:
      output = fn(start=constant_op.constant(4.))
    self.assertEqual(imported.variables[:1], list(tape.watched_variables()))
    self.assertEqual(8., tape.gradient(output, imported.variables[0]).numpy())

  def test_ref_variable_import(self):
    with self.assertRaises(NotImplementedError):
      imported = load.load(self._v1_single_metagraph_saved_model(
          use_resource=False))
    # TODO(allenl): Support ref variables
    self.skipTest("Ref variables aren't working yet")
    fn = imported.signatures["serving_default"]
    self.assertEqual(6., fn(start=constant_op.constant(2.)))

  def _v1_multi_metagraph_saved_model(self):
    export_graph = ops.Graph()
    with export_graph.as_default():
      start = array_ops.placeholder(
          shape=[None], dtype=dtypes.float32, name="start")
      v = resource_variable_ops.ResourceVariable(21.)
      first_output = array_ops.identity(start * v, name="first_output")
      second_output = array_ops.identity(v, name="second_output")
      with session_lib.Session() as session:
        session.run(v.initializer)
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        builder = builder_impl.SavedModelBuilder(path)
        builder.add_meta_graph_and_variables(
            session, tags=["first"],
            signature_def_map={
                "first_key": signature_def_utils.build_signature_def(
                    {"first_start": utils_impl.build_tensor_info(start)},
                    {"first_output": utils_impl.build_tensor_info(
                        first_output)})})
        builder.add_meta_graph(
            tags=["second"],
            signature_def_map={
                "second_key": signature_def_utils.build_signature_def(
                    {"second_start": utils_impl.build_tensor_info(start)},
                    {"second_output": utils_impl.build_tensor_info(
                        second_output)})})
        builder.save()
    return path

  def test_multi_meta_graph_loading(self):
    with self.assertRaisesRegexp(ValueError, "2 MetaGraphs"):
      load.load(self._v1_multi_metagraph_saved_model())
    first_imported = load.load(self._v1_multi_metagraph_saved_model(),
                               tags=["first"])
    self.assertEqual({"first_output": 42.},
                     self.evaluate(first_imported.signatures["first_key"](
                         first_start=constant_op.constant(2.))))
    second_imported = load.load(self._v1_multi_metagraph_saved_model(),
                                tags=["second"])
    with self.assertRaisesRegexp(TypeError, "second_start"):
      second_imported.signatures["second_key"](x=constant_op.constant(2.))
    with self.assertRaisesRegexp(TypeError, "second_start"):
      second_imported.signatures["second_key"](
          second_start=constant_op.constant(2.),
          x=constant_op.constant(2.))
    self.assertEqual({"second_output": 21.},
                     self.evaluate(second_imported.signatures["second_key"](
                         second_start=constant_op.constant(2.))))


if __name__ == "__main__":
  test.main()
