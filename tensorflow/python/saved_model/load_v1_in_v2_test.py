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

from tensorflow.core.framework import variable_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function as framework_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
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
          shape=None, dtype=dtypes.float32, name="start")
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

  @test_util.run_in_graph_and_eager_modes
  def test_resource_variable_import(self):
    imported = load.load(self._v1_single_metagraph_saved_model(
        use_resource=True))
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(variables.local_variables_initializer())
    fn = imported.signatures["serving_default"]
    self.assertEqual({"output": 6.},
                     self.evaluate(fn(constant_op.constant(2.))))
    self.assertAllEqual([3., 1.], self.evaluate(imported.variables))
    self.evaluate(imported.variables[0].assign(4.))
    self.assertEqual({"output": 8.},
                     self.evaluate(fn(start=constant_op.constant(2.))))
    self.evaluate(imported.variables[1].assign(2.))
    self.assertEqual({"output": 24.},
                     self.evaluate(fn(start=constant_op.constant(3.))))
    self.assertTrue(imported.variables[0].trainable)
    self.assertFalse(imported.variables[1].trainable)
    with backprop.GradientTape() as tape:
      output = fn(start=constant_op.constant(4.))
    self.assertEqual(imported.variables[:1], list(tape.watched_variables()))
    self.assertEqual(
        8.,
        self.evaluate(tape.gradient(output, imported.variables[0])))

  @test_util.run_in_graph_and_eager_modes
  def test_ref_variable_import(self):
    saved = self._v1_single_metagraph_saved_model(use_resource=False)
    imported = load.load(saved)
    fn = imported.signatures["serving_default"]
    self.evaluate(lookup_ops.tables_initializer())
    self.assertEqual(
        6., self.evaluate(fn(start=constant_op.constant(2.))["output"]))

  def _v1_output_shape_saved_model(self):
    export_graph = ops.Graph()
    with export_graph.as_default():
      start = array_ops.placeholder(
          shape=[None], dtype=dtypes.float32, name="start")
      output = array_ops.identity(start, name="output")
      output.set_shape([1])  # Ok to use [1] because shape is only informational
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        builder = builder_impl.SavedModelBuilder(path)
        builder.add_meta_graph_and_variables(
            session,
            tags=[tag_constants.SERVING],
            signature_def_map={
                "serving_default":
                    signature_def_utils.build_signature_def(
                        {"start": utils_impl.build_tensor_info(start)},
                        {"output": utils_impl.build_tensor_info(output)})
            })
        builder.save()
    return path

  def test_restore_output_shapes(self):
    saved = self._v1_output_shape_saved_model()
    imported = load.load(saved)
    fn = imported.signatures["serving_default"]
    self.assertEqual(tensor_shape.TensorShape([1]), fn.outputs[0].shape)

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

  def _v1_asset_saved_model(self, clear_shared_name):
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
      if clear_shared_name:
        export_graph.get_operation_by_name("hash_table")._clear_attr(
            "shared_name")
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

  @test_util.run_in_graph_and_eager_modes
  def test_asset_loading(self):
    first_path = self._v1_asset_saved_model(clear_shared_name=False)
    imported = load.load(first_path)
    self.evaluate(lookup_ops.tables_initializer())
    fn = imported.signatures["serving_default"]
    self.assertAllClose({"output": [2, 0]},
                        fn(start=constant_op.constant(["gamma", "alpha"])))
    second_path = os.path.join(self.get_temp_dir(), "saved_model",
                               str(ops.uid()))
    save.save(imported, second_path, signatures=imported.signatures)
    shutil.rmtree(first_path)
    del ops.get_collection_ref(ops.GraphKeys.TABLE_INITIALIZERS)[:]
    second_import = load.load(second_path)
    self.evaluate(lookup_ops.tables_initializer())
    fn = second_import.signatures["serving_default"]
    self.assertAllClose({"output": [2, 0]},
                        fn(start=constant_op.constant(["gamma", "alpha"])))

    third_path = os.path.join(self.get_temp_dir(), "saved_model",
                              str(ops.uid()))
    save.save(second_import, third_path, signatures=second_import.signatures)
    shutil.rmtree(second_path)
    del ops.get_collection_ref(ops.GraphKeys.TABLE_INITIALIZERS)[:]
    third_import = load.load(third_path)
    self.evaluate(lookup_ops.tables_initializer())
    fn = third_import.signatures["serving_default"]
    self.assertAllClose({"output": [2, 0]},
                        fn(start=constant_op.constant(["gamma", "alpha"])))

  @test_util.run_in_graph_and_eager_modes
  def test_node_name_sharing(self):
    fourth_path = self._v1_asset_saved_model(clear_shared_name=True)
    fourth_import = load.load(fourth_path)
    self.evaluate(lookup_ops.tables_initializer())
    fn = fourth_import.signatures["serving_default"]
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
      inp = array_ops.placeholder(name="x", shape=[], dtype=dtypes.float32)
      array_ops.identity(inp + 1., name="out")
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

  def test_version_info(self):
    path = self._signature_with_no_inputs()
    imported = load.load(path)
    self.assertEqual(versions.__version__, imported.tensorflow_version)
    self.assertEqual(versions.__git_version__,
                     imported.tensorflow_git_version)

  def _unfed_placeholder_signature(self):
    export_graph = ops.Graph()
    with export_graph.as_default():
      x = array_ops.placeholder(name="x", shape=[], dtype=dtypes.float32)
      output = x * random_ops.random_normal([2])
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

  def test_unfed_placeholder_exception(self):
    path = self._unfed_placeholder_signature()
    with self.assertRaisesRegexp(
        lift_to_graph.UnliftableError,
        "signature needs an input for each placeholder.*\n\nUnable to lift"):
      load.load(path)

  def test_custom_pruning(self):
    path = self._no_signatures_model()
    root = load.load(path)
    fn = root.prune("x:0", "out:0")
    self.assertEqual(2., self.evaluate(fn(x=array_ops.ones([]))))
    root.graph.as_graph_element("x:0")

  def _no_trainable_variable_attribute(self, trainable):
    """A SavedModel where the VariableDef has no 'trainable' (it's false)."""

    class _MissingFieldsVariable(resource_variable_ops.ResourceVariable):

      def to_proto(self, export_scope=None):
        full_proto = super(_MissingFieldsVariable, self).to_proto(export_scope)
        return variable_pb2.VariableDef(
            variable_name=full_proto.variable_name,
            initial_value_name=full_proto.initial_value_name,
            initializer_name=full_proto.snapshot_name,
            save_slice_info_def=full_proto.save_slice_info_def,
            is_resource=full_proto.is_resource)

    export_graph = ops.Graph()
    with export_graph.as_default():
      v = _MissingFieldsVariable(3., trainable=trainable)
      with session_lib.Session() as session:
        session.run([v.initializer])
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        b = builder_impl.SavedModelBuilder(path)
        b.add_meta_graph_and_variables(
            session,
            tags=[tag_constants.SERVING],
            signature_def_map={})
        b.save()

    return path

  def test_trainable_not_set_in_proto(self):
    """If a VariableDef has no 'trainable', we fall back to collections."""
    real_tf_version = versions.__version__
    # Pretend to be exported from an older version of TensorFlow, so trainable
    # will follow collections instead of checking VariableDefs.
    versions.__version__ = "1.7.0"
    path = self._no_trainable_variable_attribute(trainable=True)
    root = load.load(path)
    self.assertTrue(root.variables[0].trainable)
    path = self._no_trainable_variable_attribute(trainable=False)
    root = load.load(path)
    self.assertFalse(root.variables[0].trainable)
    versions.__version__ = real_tf_version

  def _export_variable(self, **kwargs_for_variable):
    """A 1.x SavedModel with a single variable."""
    export_graph = ops.Graph()
    with export_graph.as_default():
      v = resource_variable_ops.ResourceVariable(3., **kwargs_for_variable)
      with session_lib.Session() as session:
        session.run([v.initializer])
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        b = builder_impl.SavedModelBuilder(path)
        b.add_meta_graph_and_variables(
            session,
            tags=[tag_constants.SERVING],
            signature_def_map={})
        b.save()

    return path

  def test_trainable_in_proto(self):
    """If a VariableDef has a trainable property, we do not use collections."""
    path = self._export_variable(
        trainable=True,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES])
    root = load.load(path)
    self.assertTrue(root.variables[0].trainable)
    path = self._export_variable(
        trainable=False,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                     ops.GraphKeys.TRAINABLE_VARIABLES])
    root = load.load(path)
    self.assertFalse(root.variables[0].trainable)

  def _model_with_sparse_output(self):
    """Generate a graph with a SparseTensor output and serialize in V1 format"""
    export_graph = ops.Graph()
    with export_graph.as_default():
      in_placeholder = array_ops.placeholder(dtype=dtypes.int64, shape=[1])
      out_sparse_tensor = sparse_tensor.SparseTensor(
          indices=[[0]], values=in_placeholder, dense_shape=[1]) * 2
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"start": in_placeholder},
            outputs={"output": out_sparse_tensor})
    return path

  def test_load_sparse_outputs(self):
    path = self._model_with_sparse_output()
    imported = load.load(path)
    imported_fn = imported.signatures["serving_default"]
    forty_two = constant_op.constant([42], dtype=dtypes.int64)
    self.assertEqual([84], imported_fn(forty_two)["output"].values.numpy())

  def _model_with_sparse_input(self):
    """Generate a graph with a SparseTensor input and serialize in V1 format."""
    export_graph = ops.Graph()
    with export_graph.as_default():
      in_sparse_placeholder = array_ops.sparse_placeholder(
          dtype=dtypes.int64, shape=[2, 2])
      out_sparse_tensor = sparse_tensor.SparseTensor(
          indices=in_sparse_placeholder.indices,
          values=in_sparse_placeholder.values,
          dense_shape=in_sparse_placeholder.dense_shape) * 2
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"start": in_sparse_placeholder},
            outputs={"output": out_sparse_tensor})
    return path

  def test_load_sparse_inputs(self):
    path = self._model_with_sparse_input()
    imported = load.load(path)
    imported_fn = imported.signatures["serving_default"]
    indices = constant_op.constant([[0, 0], [0, 1], [1, 1]], dtype=dtypes.int64)
    values = constant_op.constant([42, 43, 44], dtype=dtypes.int64)
    dense_shape = constant_op.constant([2, 2], dtype=dtypes.int64)
    result = imported_fn(
        start_indices=indices,
        start_values=values,
        start_dense_shape=dense_shape)
    self.assertAllEqual([84, 86, 88], result["output"].values.numpy())

  def _model_with_defun(self):
    """Generate a graph with a Defun and serialize in V1 format."""
    export_graph = ops.Graph()
    with export_graph.as_default():
      @framework_function.Defun(dtypes.int64)
      def z(x):
        return x + 1

      @framework_function.Defun(dtypes.int64)
      def g(x):
        return z(x) + 1

      @framework_function.Defun(dtypes.int64)
      def f(x):
        return g(x) + 1
      in_placeholder = array_ops.placeholder(dtype=dtypes.int64, shape=[1])
      out = f(in_placeholder)
      with session_lib.Session() as session:
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"start": in_placeholder},
            outputs={"output": out})
    return path

  def test_load_defun(self):
    path = self._model_with_defun()
    imported = load.load(path)
    imported_fn = imported.signatures["serving_default"]
    forty_two = constant_op.constant([42], dtype=dtypes.int64)
    self.assertEqual([45], imported_fn(forty_two)["output"].numpy())


if __name__ == "__main__":
  test.main()
