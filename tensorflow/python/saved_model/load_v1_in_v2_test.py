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
import shutil

from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import simple_save
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils_impl


class LoadTest(test.TestCase):

  def _v1_single_metagraph_saved_model(self, use_resource):
    export_graph = ops.Graph()
    with export_graph.as_default():
      start = array_ops.placeholder(
          shape=[None], dtype=dtypes.float32, name="start")
      if use_resource:
        distractor = variables.RefVariable(-1., name="distractor")
        v = resource_variable_ops.ResourceVariable(3., name="v")
      else:
        # "distractor" gets saved in the checkpoint and so used in the restore
        # function, but not in the pruned function for the signature. This tests
        # node naming: it needs to be consistent (and ideally always the same as
        # the node in the original GraphDef) for the resource manager to find
        # the right variable.
        distractor = variables.RefVariable(-1., name="distractor")
        v = variables.RefVariable(3., name="v")
      local_variable = variables.VariableV1(
          1.,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          trainable=False,
          use_resource=True)
      output = array_ops.identity(start * v * local_variable, name="output")
      with session_lib.Session() as session:
        session.run([v.initializer, distractor.initializer,
                     local_variable.initializer])
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
    self.assertEqual({"output": 6.},
                     self.evaluate(fn(constant_op.constant(2.))))
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
    saved = self._v1_single_metagraph_saved_model(use_resource=False)
    imported = load.load(saved)
    fn = imported.signatures["serving_default"]
    self.assertEqual(6., fn(start=constant_op.constant(2.))["output"].numpy())

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
                                tags=set(["second"]))
    with self.assertRaisesRegexp(TypeError, "second_start"):
      second_imported.signatures["second_key"](x=constant_op.constant(2.))
    with self.assertRaisesRegexp(TypeError, "second_start"):
      second_imported.signatures["second_key"](
          second_start=constant_op.constant(2.),
          x=constant_op.constant(2.))
    self.assertEqual({"second_output": 21.},
                     self.evaluate(second_imported.signatures["second_key"](
                         second_start=constant_op.constant(2.))))

  def _v1_asset_saved_model(self):
    export_graph = ops.Graph()
    vocab_path = os.path.join(self.get_temp_dir(), "vocab.txt")
    with open(vocab_path, "w") as f:
      f.write("alpha\nbeta\ngamma\n")
    with export_graph.as_default():
      initializer = lookup_ops.TextFileInitializer(
          vocab_path,
          key_dtype=dtypes.string,
          key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
          value_dtype=dtypes.int64,
          value_index=lookup_ops.TextFileIndex.LINE_NUMBER)
      table = lookup_ops.HashTable(
          initializer, default_value=-1)
      start = array_ops.placeholder(
          shape=None, dtype=dtypes.string, name="in")
      output = table.lookup(start, name="out")
      with session_lib.Session() as session:
        session.run([table.initializer])
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"start": start},
            outputs={"output": output},
            legacy_init_op=table.initializer)
    file_io.delete_file(vocab_path)
    return path

  def test_asset_loading(self):
    first_path = self._v1_asset_saved_model()
    imported = load.load(first_path)
    fn = imported.signatures["serving_default"]
    self.assertAllClose({"output": [2, 0]},
                        fn(start=constant_op.constant(["gamma", "alpha"])))
    second_path = os.path.join(self.get_temp_dir(), "saved_model",
                               str(ops.uid()))
    save.save(imported, second_path, signatures=imported.signatures)
    shutil.rmtree(first_path)
    second_import = load.load(second_path)
    fn = second_import.signatures["serving_default"]
    self.assertAllClose({"output": [2, 0]},
                        fn(start=constant_op.constant(["gamma", "alpha"])))

    third_path = os.path.join(self.get_temp_dir(), "saved_model",
                              str(ops.uid()))
    save.save(second_import, third_path, signatures=second_import.signatures)
    shutil.rmtree(second_path)
    third_import = load.load(third_path)
    fn = third_import.signatures["serving_default"]
    self.assertAllClose({"output": [2, 0]},
                        fn(start=constant_op.constant(["gamma", "alpha"])))

  def _v1_cond_saved_model(self):
    export_graph = ops.Graph()
    with export_graph.as_default():
      branch_selector = array_ops.placeholder(
          name="branch_selector", shape=[], dtype=dtypes.bool)
      output = control_flow_ops.cond(
          branch_selector,
          lambda: array_ops.ones([]),
          lambda: array_ops.zeros([]))
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"branch_selector": branch_selector},
            outputs={"output": output})
    return path

  def test_cond(self):
    first_path = self._v1_cond_saved_model()
    imported = load.load(first_path)
    function = imported.signatures["serving_default"]
    self.assertAllClose({"output": 1.}, function(constant_op.constant(True)))
    self.assertAllClose({"output": 0.}, function(constant_op.constant(False)))

  def _v1_while_saved_model(self):
    export_graph = ops.Graph()
    with export_graph.as_default():
      loop_iterations = array_ops.placeholder(
          name="loop_iterations", shape=[], dtype=dtypes.int32)
      _, output = control_flow_ops.while_loop(
          lambda index, accum: index <= loop_iterations,
          lambda index, accum: (index + 1, accum + index),
          [constant_op.constant(0), constant_op.constant(0)])
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"loop_iterations": loop_iterations},
            outputs={"output": output})
    return path

  def test_while(self):
    first_path = self._v1_while_saved_model()
    imported = load.load(first_path)
    function = imported.signatures["serving_default"]
    self.assertAllClose({"output": 10}, function(constant_op.constant(4)))
    self.assertAllClose({"output": 15}, function(constant_op.constant(5)))

  def _v1_nested_while_saved_model(self):
    export_graph = ops.Graph()
    with export_graph.as_default():

      def _inner_while(loop_iterations):
        _, output = control_flow_ops.while_loop(
            lambda index, accum: index <= loop_iterations,
            lambda index, accum: (index + 1, accum + index),
            [constant_op.constant(0), constant_op.constant(0)])
        return output

      loop_iterations = array_ops.placeholder(
          name="loop_iterations", shape=[], dtype=dtypes.int32)
      _, output = control_flow_ops.while_loop(
          lambda index, accum: index <= loop_iterations,
          lambda index, accum: (index + 1, accum + _inner_while(index)),
          [constant_op.constant(0), constant_op.constant(0)])
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"loop_iterations": loop_iterations},
            outputs={"output": output})
    return path

  def test_nested_while(self):
    first_path = self._v1_nested_while_saved_model()
    imported = load.load(first_path)
    function = imported.signatures["serving_default"]
    self.assertAllClose({"output": 20}, function(constant_op.constant(4)))
    self.assertAllClose({"output": 35}, function(constant_op.constant(5)))

  def _no_signatures_model(self):
    export_graph = ops.Graph()
    with export_graph.as_default():
      array_ops.placeholder(name="x", shape=[], dtype=dtypes.float32)

      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        b = builder_impl.SavedModelBuilder(path)
        b.add_meta_graph_and_variables(
            session,
            tags=[tag_constants.SERVING],
            signature_def_map={},
            assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS))
        b.save()
    return path

  def test_no_signature(self):
    path = self._no_signatures_model()
    imported = load.load(path)
    self.assertEqual([], list(imported.signatures.keys()))

  def _signature_with_no_inputs(self):
    export_graph = ops.Graph()
    with export_graph.as_default():
      array_ops.placeholder(name="x", shape=[], dtype=dtypes.float32)
      output = random_ops.random_normal([2])
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        b = builder_impl.SavedModelBuilder(path)
        b.add_meta_graph_and_variables(
            session,
            tags=[tag_constants.SERVING],
            signature_def_map={
                "key": signature_def_utils.build_signature_def(
                    {}, dict(value=utils_impl.build_tensor_info(output)))})
        b.save()
    return path

  def test_signature_with_no_inputs(self):
    path = self._signature_with_no_inputs()
    imported = load.load(path)
    self.assertEqual([2], imported.signatures["key"]()["value"].shape)

if __name__ == "__main__":
  test.main()
