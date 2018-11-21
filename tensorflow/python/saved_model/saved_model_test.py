# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SavedModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import main_op
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import saver_test_utils
from tensorflow.python.training import training
from tensorflow.python.util import compat

SAVED_MODEL_PATH = ("cc/saved_model/testdata/half_plus_two/00000123")


def tearDownModule():
  file_io.delete_recursively(test.get_temp_dir())


class SavedModelTestBase(test.TestCase):

  def _get_export_dir(self, label):
    return os.path.join(test.get_temp_dir(), label)

  def _init_and_validate_variable(self, sess, variable_name, variable_value):
    v = variables.VariableV1(variable_value, name=variable_name)
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(variable_value, self.evaluate(v))

  def _build_asset_collection(self, asset_file_name, asset_file_contents,
                              asset_file_tensor_name, asset_subdir=""):
    parent_dir = os.path.join(
        compat.as_bytes(test.get_temp_dir()), compat.as_bytes(asset_subdir))
    file_io.recursive_create_dir(parent_dir)
    asset_filepath = os.path.join(
        compat.as_bytes(parent_dir), compat.as_bytes(asset_file_name))
    file_io.write_string_to_file(asset_filepath, asset_file_contents)
    asset_file_tensor = constant_op.constant(
        asset_filepath, name=asset_file_tensor_name)
    ops.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, asset_file_tensor)
    asset_collection = ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)
    return asset_collection


class SavedModelTest(SavedModelTestBase):

  def _validate_assets(self,
                       export_dir,
                       asset_file_def,
                       expected_asset_file_name,
                       expected_asset_file_contents,
                       expected_asset_tensor_name,
                       asset_id=0):
    assets_path = os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes(constants.ASSETS_DIRECTORY),
        compat.as_bytes(expected_asset_file_name))
    actual_asset_contents = file_io.read_file_to_string(assets_path)
    self.assertEqual(expected_asset_file_contents,
                     compat.as_text(actual_asset_contents))
    self.assertEqual(expected_asset_file_name,
                     asset_file_def[asset_id].filename)
    self.assertEqual(expected_asset_tensor_name,
                     asset_file_def[asset_id].tensor_info.name)

  def _validate_inputs_tensor_info_fail(self, builder, tensor_info):
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      foo_signature = signature_def_utils.build_signature_def({
          "foo_inputs": tensor_info
      }, dict(), "foo")
      self.assertRaises(
          AssertionError,
          builder.add_meta_graph_and_variables,
          sess, ["foo"],
          signature_def_map={"foo_key": foo_signature})

  def _validate_inputs_tensor_info_accept(self, builder, tensor_info):
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      foo_signature = signature_def_utils.build_signature_def({
          "foo_inputs": tensor_info
      }, dict(), "foo")
      builder.add_meta_graph_and_variables(
          sess, ["foo"],
          signature_def_map={"foo_key": foo_signature})

  def _validate_outputs_tensor_info_fail(self, builder, tensor_info):
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      foo_signature = signature_def_utils.build_signature_def(
          dict(), {"foo_outputs": tensor_info}, "foo")
      self.assertRaises(
          AssertionError,
          builder.add_meta_graph_and_variables,
          sess, ["foo"],
          signature_def_map={"foo_key": foo_signature})

  def _validate_outputs_tensor_info_accept(self, builder, tensor_info):
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      foo_signature = signature_def_utils.build_signature_def(
          dict(), {"foo_outputs": tensor_info}, "foo")
      builder.add_meta_graph_and_variables(
          sess, ["foo"],
          signature_def_map={"foo_key": foo_signature})

  def testMaybeSavedModelDir(self):
    base_path = test.test_src_dir_path("/python/saved_model")
    self.assertFalse(loader.maybe_saved_model_directory(base_path))
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    self.assertTrue(loader.maybe_saved_model_directory(base_path))
    base_path = "complete_garbage"
    self.assertFalse(loader.maybe_saved_model_directory(base_path))

  def testBadSavedModelFileFormat(self):
    export_dir = self._get_export_dir("test_bad_saved_model_file_format")
    # Attempt to load a SavedModel from an export directory that does not exist.
    with self.session(graph=ops.Graph()) as sess:
      with self.assertRaisesRegexp(IOError,
                                   "SavedModel file does not exist at: %s" %
                                   export_dir):
        loader.load(sess, ["foo"], export_dir)

    os.makedirs(export_dir)
    # Write an invalid binary proto to saved_model.pb.
    path_to_pb = os.path.join(export_dir, constants.SAVED_MODEL_FILENAME_PB)
    with open(path_to_pb, "w") as f:
      f.write("invalid content")
    with self.session(graph=ops.Graph()) as sess:
      with self.assertRaisesRegexp(IOError, "Cannot parse file.*%s" %
                                   constants.SAVED_MODEL_FILENAME_PB):
        loader.load(sess, ["foo"], export_dir)

    # Cleanup the directory and start again.
    file_io.delete_recursively(export_dir)

    os.makedirs(export_dir)
    # Write an invalid text proto to saved_model.pbtxt
    path_to_pbtxt = os.path.join(export_dir,
                                 constants.SAVED_MODEL_FILENAME_PBTXT)
    with open(path_to_pbtxt, "w") as f:
      f.write("invalid content")
    with self.session(graph=ops.Graph()) as sess:
      with self.assertRaisesRegexp(IOError, "Cannot parse file.*%s" %
                                   constants.SAVED_MODEL_FILENAME_PBTXT):
        loader.load(sess, ["foo"], export_dir)

  def testVerifySessionGraphUsage(self):
    export_dir = self._get_export_dir("test_verify_session_graph_usage")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)
      builder.add_meta_graph_and_variables(sess, [tag_constants.TRAINING])

    # Save the SavedModel to disk.
    builder.save()

    # Build a session and supply it to the load operation.
    sess = session.Session(graph=ops.Graph())
    loader.load(sess, [tag_constants.TRAINING], export_dir)

    # Check the variable within the scope of the session and its graph.
    with sess:
      self.assertEqual(
          42, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].eval())

  def testSequence(self):
    export_dir = self._get_export_dir("test_sequence")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    # Expect an assertion error since add_meta_graph_and_variables() should be
    # invoked before any add_meta_graph() calls.
    with self.session(graph=ops.Graph()) as sess:
      self.assertRaises(AssertionError, builder.add_meta_graph, ["foo"])

    # Expect an assertion error for multiple calls of
    # add_meta_graph_and_variables() since weights should be saved exactly once.
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)
      builder.add_meta_graph_and_variables(sess, ["bar"])
      self.assertRaises(AssertionError, builder.add_meta_graph_and_variables,
                        sess, ["baz"])

  def testTags(self):
    export_dir = self._get_export_dir("test_tags")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    # Graph with a single variable. SavedModel invoked to:
    # - add with weights.
    # - a single tag (from predefined constants).
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)
      builder.add_meta_graph_and_variables(sess, [tag_constants.TRAINING])

    # Graph that updates the single variable. SavedModel invoked to:
    # - simply add the model (weights are not updated).
    # - a single tag (from predefined constants).
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 43)
      builder.add_meta_graph([tag_constants.SERVING])

    # Graph that updates the single variable. SavedModel invoked to:
    # - simply add the model (weights are not updated).
    # - multiple tags (from predefined constants).
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 45)
      builder.add_meta_graph([tag_constants.SERVING, tag_constants.GPU])

    # Graph that updates the single variable. SavedModel invoked to:
    # - simply add the model (weights are not updated).
    # - multiple tags (from predefined constants for serving on TPU).
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 45)
      builder.add_meta_graph([tag_constants.SERVING, tag_constants.TPU])

    # Graph that updates the single variable. SavedModel is invoked:
    # - to add the model (weights are not updated).
    # - multiple custom tags.
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 44)
      builder.add_meta_graph(["foo", "bar"])

    # Save the SavedModel to disk.
    builder.save()

    # Restore the graph with a single predefined tag whose variables were saved.
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, [tag_constants.TRAINING], export_dir)
      self.assertEqual(
          42, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].eval())

    # Restore the graph with a single predefined tag whose variables were not
    # saved.
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, [tag_constants.SERVING], export_dir)
      self.assertEqual(
          42, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].eval())

    # Restore the graph with multiple predefined tags whose variables were not
    # saved.
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, [tag_constants.SERVING, tag_constants.GPU], export_dir)
      self.assertEqual(
          42, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].eval())

    # Restore the graph with multiple predefined tags (for serving on TPU)
    # whose variables were not saved.
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, [tag_constants.SERVING, tag_constants.TPU], export_dir)
      self.assertEqual(
          42, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].eval())

    # Restore the graph with multiple tags. Provide duplicate tags to test set
    # semantics.
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo", "bar", "foo"], export_dir)
      self.assertEqual(
          42, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].eval())

    # Try restoring a graph with a non-existent tag. This should yield a runtime
    # error.
    with self.session(graph=ops.Graph()) as sess:
      self.assertRaises(RuntimeError, loader.load, sess, ["INVALID"],
                        export_dir)

    # Try restoring a graph where a subset of the tags match. Since tag matching
    # for meta graph defs follows "all" semantics, this should yield a runtime
    # error.
    with self.session(graph=ops.Graph()) as sess:
      self.assertRaises(RuntimeError, loader.load, sess, ["foo", "baz"],
                        export_dir)

  def testVariables(self):
    export_dir = self._get_export_dir("test_variables")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    # Graph with two variables. SavedModel invoked to:
    # - add with weights.
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v1", 1)
      self._init_and_validate_variable(sess, "v2", 2)
      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Graph with a single variable (subset of the variables from the previous
    # graph whose weights were saved). SavedModel invoked to:
    # - simply add the model (weights are not updated).
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v2", 3)
      builder.add_meta_graph(["bar"])

    # Graph with a single variable (disjoint set of variables from the previous
    # graph whose weights were saved). SavedModel invoked to:
    # - simply add the model (weights are not updated).
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v3", 4)
      builder.add_meta_graph(["baz"])

    # Save the SavedModel to disk.
    builder.save()

    # Restore the graph with tag "foo", whose variables were saved.
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo"], export_dir)
      collection_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertEqual(len(collection_vars), 2)
      self.assertEqual(1, collection_vars[0].eval())
      self.assertEqual(2, collection_vars[1].eval())

    # Restore the graph with tag "bar", whose variables were not saved. Only the
    # subset of the variables added to the graph will be restored with the
    # checkpointed value.
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["bar"], export_dir)
      collection_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertEqual(len(collection_vars), 1)
      self.assertEqual(2, collection_vars[0].eval())

    # Try restoring the graph with tag "baz", whose variables were not saved.
    # Since this graph has a disjoint set of variables from the set that was
    # saved, this should raise an error.
    with self.session(graph=ops.Graph()) as sess:
      self.assertRaises(errors.NotFoundError, loader.load, sess, ["baz"],
                        export_dir)

  def testGraphWithoutVariables(self):
    export_dir = self._get_export_dir("test_graph_has_variables")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    # Graph with no variables.
    with self.session(graph=ops.Graph()) as sess:
      constant_5_name = constant_op.constant(5.0).name
      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Second graph with no variables
    with self.session(graph=ops.Graph()) as sess:
      constant_6_name = constant_op.constant(6.0).name
      builder.add_meta_graph(["bar"])

    # Save the SavedModel to disk.
    builder.save()

    # Restore the graph with tag "foo".
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo"], export_dir)
      # Read the constant a from the graph.
      a = ops.get_default_graph().get_tensor_by_name(constant_5_name)
      b = constant_op.constant(6.0)
      c = a * b
      self.assertEqual(30.0, self.evaluate(c))

    # Restore the graph with tag "bar".
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["bar"], export_dir)
      # Read the constant a from the graph.
      a = ops.get_default_graph().get_tensor_by_name(constant_6_name)
      b = constant_op.constant(5.0)
      c = a * b
      self.assertEqual(30.0, self.evaluate(c))

  def testNoOverwrite(self):
    export_dir = self._get_export_dir("test_no_overwrite")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    # Graph with a single variable. SavedModel invoked to:
    # - add with weights.
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)
      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Save the SavedModel to disk in text format.
    builder.save(as_text=True)

    # Restore the graph with tag "foo", whose variables were saved.
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo"], export_dir)
      self.assertEqual(
          42, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].eval())

    # An attempt to create another builder with the same export directory should
    # result in an assertion error.
    self.assertRaises(AssertionError, saved_model_builder._SavedModelBuilder,
                      export_dir)

  def testSaveAsText(self):
    export_dir = self._get_export_dir("test_astext")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    # Graph with a single variable. SavedModel invoked to:
    # - add with weights.
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)
      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Graph with the same single variable. SavedModel invoked to:
    # - simply add the model (weights are not updated).
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 43)
      builder.add_meta_graph(["bar"])

    # Save the SavedModel to disk in text format.
    builder.save(as_text=True)

    # Restore the graph with tag "foo", whose variables were saved.
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo"], export_dir)
      self.assertEqual(
          42, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].eval())

    # Restore the graph with tag "bar", whose variables were not saved.
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["bar"], export_dir)
      self.assertEqual(
          42, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].eval())

  def testCollections(self):
    export_dir = self._get_export_dir("test_collections")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    # Graph with a single variable added to a collection. SavedModel invoked to:
    # - add with weights.
    with self.session(graph=ops.Graph()) as sess:
      v = variables.VariableV1(42, name="v")
      ops.add_to_collection("foo_vars", v)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(42, self.evaluate(v))
      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Graph with the same single variable added to a different collection.
    # SavedModel invoked to:
    # - simply add the model (weights are not updated).
    with self.session(graph=ops.Graph()) as sess:
      v = variables.VariableV1(43, name="v")
      ops.add_to_collection("bar_vars", v)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(43, self.evaluate(v))
      builder.add_meta_graph(["bar"])

    # Save the SavedModel to disk.
    builder.save()

    # Restore the graph with tag "foo", whose variables were saved. The
    # collection 'foo_vars' should contain a single element. The collection
    # 'bar_vars' should not be found.
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo"], export_dir)
      collection_foo_vars = ops.get_collection("foo_vars")
      self.assertEqual(len(collection_foo_vars), 1)
      self.assertEqual(42, collection_foo_vars[0].eval())

      self.assertEqual(len(ops.get_collection("bar_vars")), 0)

    # Restore the graph with tag "bar", whose variables were not saved. The
    # collection-def exported as part of the meta graph def is updated to
    # reflect the new collection. The value of the variable in the
    # collection-def corresponds to the saved value (from the previous graph
    # with tag "foo").
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["bar"], export_dir)
      collection_bar_vars = ops.get_collection("bar_vars")
      self.assertEqual(len(collection_bar_vars), 1)
      self.assertEqual(42, collection_bar_vars[0].eval())

      self.assertEqual(len(ops.get_collection("foo_vars")), 0)

  def testSignatureDefs(self):
    export_dir = self._get_export_dir("test_signature_defs")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    # Graph with a single variable and a single entry in the signature def map.
    # SavedModel is invoked to add with weights.
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)
      # Build and populate an empty SignatureDef for testing.
      foo_signature = signature_def_utils.build_signature_def(dict(),
                                                              dict(), "foo")
      builder.add_meta_graph_and_variables(
          sess, ["foo"], signature_def_map={"foo_key": foo_signature})

    # Graph with the same single variable and multiple entries in the signature
    # def map. No weights are saved by SavedModel.
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 43)
      # Build and populate a different SignatureDef for testing.
      bar_signature = signature_def_utils.build_signature_def(dict(),
                                                              dict(), "bar")
      # Also, build a different SignatureDef corresponding to "foo_key" defined
      # in the previous graph.
      foo_new_signature = signature_def_utils.build_signature_def(dict(),
                                                                  dict(),
                                                                  "foo_new")
      builder.add_meta_graph(
          ["bar"],
          signature_def_map={
              "bar_key": bar_signature,
              "foo_key": foo_new_signature
          })

    # Save the SavedModel to disk.
    builder.save()

    # Restore the graph with tag "foo". The single entry in the SignatureDef map
    # corresponding to "foo_key" should exist.
    with self.session(graph=ops.Graph()) as sess:
      foo_graph = loader.load(sess, ["foo"], export_dir)
      self.assertEqual(
          42, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].eval())

      foo_signature = foo_graph.signature_def
      self.assertEqual(len(foo_signature), 1)
      self.assertEqual("foo", foo_signature["foo_key"].method_name)

    # Restore the graph with tag "bar". The SignatureDef map should have two
    # entries. One corresponding to "bar_key" and another corresponding to the
    # new value of "foo_key".
    with self.session(graph=ops.Graph()) as sess:
      bar_graph = loader.load(sess, ["bar"], export_dir)
      self.assertEqual(
          42, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].eval())

      bar_signature = bar_graph.signature_def
      self.assertEqual(len(bar_signature), 2)
      self.assertEqual("bar", bar_signature["bar_key"].method_name)
      self.assertEqual("foo_new", bar_signature["foo_key"].method_name)

  def testSignatureDefValidationFails(self):
    export_dir = self._get_export_dir("test_signature_def_validation_fail")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    tensor_without_encoding = meta_graph_pb2.TensorInfo()
    tensor_without_encoding.dtype = types_pb2.DT_FLOAT
    self._validate_inputs_tensor_info_fail(builder, tensor_without_encoding)
    self._validate_outputs_tensor_info_fail(builder, tensor_without_encoding)

    tensor_without_dtype = meta_graph_pb2.TensorInfo()
    tensor_without_dtype.name = "x"
    self._validate_inputs_tensor_info_fail(builder, tensor_without_dtype)
    self._validate_outputs_tensor_info_fail(builder, tensor_without_dtype)

    tensor_empty = meta_graph_pb2.TensorInfo()
    self._validate_inputs_tensor_info_fail(builder, tensor_empty)
    self._validate_outputs_tensor_info_fail(builder, tensor_empty)

  def testSignatureDefValidationSucceedsWithName(self):
    tensor_with_name = meta_graph_pb2.TensorInfo()
    tensor_with_name.name = "foo"
    tensor_with_name.dtype = types_pb2.DT_FLOAT

    export_dir = self._get_export_dir("test_signature_def_validation_name_1")
    builder = saved_model_builder._SavedModelBuilder(export_dir)
    self._validate_inputs_tensor_info_accept(builder, tensor_with_name)

    export_dir = self._get_export_dir("test_signature_def_validation_name_2")
    builder = saved_model_builder._SavedModelBuilder(export_dir)
    self._validate_outputs_tensor_info_accept(builder, tensor_with_name)

  def testSignatureDefValidationSucceedsWithCoo(self):
    tensor_with_coo = meta_graph_pb2.TensorInfo()
    # TODO(soergel) test validation of each of the fields of coo_sparse
    tensor_with_coo.coo_sparse.values_tensor_name = "foo"
    tensor_with_coo.dtype = types_pb2.DT_FLOAT

    export_dir = self._get_export_dir("test_signature_def_validation_coo_1")
    builder = saved_model_builder._SavedModelBuilder(export_dir)
    self._validate_inputs_tensor_info_accept(builder, tensor_with_coo)

    export_dir = self._get_export_dir("test_signature_def_validation_coo_2")
    builder = saved_model_builder._SavedModelBuilder(export_dir)
    self._validate_outputs_tensor_info_accept(builder, tensor_with_coo)

  def testAssets(self):
    export_dir = self._get_export_dir("test_assets")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      # Build an asset collection.
      ignored_filepath = os.path.join(
          compat.as_bytes(test.get_temp_dir()), compat.as_bytes("ignored.txt"))
      file_io.write_string_to_file(ignored_filepath, "will be ignored")

      asset_list = self._build_asset_collection("hello42.txt", "foo bar baz",
                                                "asset_file_tensor")

      builder.add_meta_graph_and_variables(
          sess, ["foo"], assets_list=asset_list)

    # Save the SavedModel to disk.
    builder.save()

    with self.session(graph=ops.Graph()) as sess:
      foo_graph = loader.load(sess, ["foo"], export_dir)
      self._validate_assets(export_dir, foo_graph.asset_file_def, "hello42.txt",
                            "foo bar baz", "asset_file_tensor:0")
      ignored_asset_path = os.path.join(
          compat.as_bytes(export_dir),
          compat.as_bytes(constants.ASSETS_DIRECTORY),
          compat.as_bytes("ignored.txt"))
      self.assertFalse(file_io.file_exists(ignored_asset_path))

  def testAssetsNameCollisionDiffFile(self):
    export_dir = self._get_export_dir("test_assets_name_collision_diff_file")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      asset_list = self._build_asset_collection(
          "hello42.txt", "foo bar bak", "asset_file_tensor", asset_subdir="1")

      asset_list = self._build_asset_collection(
          "hello42.txt", "foo bar baz", "asset_file_tensor_1", asset_subdir="2")

      builder.add_meta_graph_and_variables(
          sess, ["foo"], assets_list=asset_list)

    # Save the SavedModel to disk.
    builder.save()

    with self.session(graph=ops.Graph()) as sess:
      foo_graph = loader.load(sess, ["foo"], export_dir)
      self._validate_assets(export_dir, foo_graph.asset_file_def, "hello42.txt",
                            "foo bar bak", "asset_file_tensor:0")
      self._validate_assets(
          export_dir,
          foo_graph.asset_file_def,
          "hello42.txt_1",
          "foo bar baz",
          "asset_file_tensor_1:0",
          asset_id=1)

  def testAssetsNameCollisionSameFilepath(self):
    export_dir = self._get_export_dir("test_assets_name_collision_same_path")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      asset_list = self._build_asset_collection("hello42.txt", "foo bar baz",
                                                "asset_file_tensor")

      asset_list = self._build_asset_collection("hello42.txt", "foo bar baz",
                                                "asset_file_tensor_1")

      builder.add_meta_graph_and_variables(
          sess, ["foo"], assets_list=asset_list)

    # Save the SavedModel to disk.
    builder.save()

    with self.session(graph=ops.Graph()) as sess:
      foo_graph = loader.load(sess, ["foo"], export_dir)
      self._validate_assets(export_dir, foo_graph.asset_file_def, "hello42.txt",
                            "foo bar baz", "asset_file_tensor:0")
      # The second tensor should be recorded, but the same.
      self._validate_assets(
          export_dir,
          foo_graph.asset_file_def,
          "hello42.txt",
          "foo bar baz",
          "asset_file_tensor_1:0",
          asset_id=1)
      ignored_asset_path = os.path.join(
          compat.as_bytes(export_dir),
          compat.as_bytes(constants.ASSETS_DIRECTORY),
          compat.as_bytes("hello42.txt_1"))
      self.assertFalse(file_io.file_exists(ignored_asset_path))

  def testAssetsNameCollisionSameFile(self):
    export_dir = self._get_export_dir("test_assets_name_collision_same_file")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      asset_list = self._build_asset_collection(
          "hello42.txt", "foo bar baz", "asset_file_tensor", asset_subdir="1")

      asset_list = self._build_asset_collection(
          "hello42.txt", "foo bar baz", "asset_file_tensor_1", asset_subdir="2")

      builder.add_meta_graph_and_variables(
          sess, ["foo"], assets_list=asset_list)

    # Save the SavedModel to disk.
    builder.save()

    with self.session(graph=ops.Graph()) as sess:
      foo_graph = loader.load(sess, ["foo"], export_dir)
      self._validate_assets(export_dir, foo_graph.asset_file_def, "hello42.txt",
                            "foo bar baz", "asset_file_tensor:0")
      # The second tensor should be recorded, but the same.
      self._validate_assets(
          export_dir,
          foo_graph.asset_file_def,
          "hello42.txt",
          "foo bar baz",
          "asset_file_tensor_1:0",
          asset_id=1)
      ignored_asset_path = os.path.join(
          compat.as_bytes(export_dir),
          compat.as_bytes(constants.ASSETS_DIRECTORY),
          compat.as_bytes("hello42.txt_1"))
      self.assertFalse(file_io.file_exists(ignored_asset_path))

  def testAssetsNameCollisionManyFiles(self):
    export_dir = self._get_export_dir("test_assets_name_collision_many_files")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      for i in range(5):
        idx = str(i)
        asset_list = self._build_asset_collection(
            "hello42.txt",
            "foo bar baz " + idx,
            "asset_file_tensor_" + idx,
            asset_subdir=idx)

      builder.add_meta_graph_and_variables(
          sess, ["foo"], assets_list=asset_list)

    # Save the SavedModel to disk.
    builder.save()

    with self.session(graph=ops.Graph()) as sess:
      foo_graph = loader.load(sess, ["foo"], export_dir)
      for i in range(1, 5):
        idx = str(i)
        self._validate_assets(
            export_dir,
            foo_graph.asset_file_def,
            "hello42.txt_" + idx,
            "foo bar baz " + idx,
            "asset_file_tensor_{}:0".format(idx),
            asset_id=i)

      self._validate_assets(export_dir, foo_graph.asset_file_def, "hello42.txt",
                            "foo bar baz 0", "asset_file_tensor_0:0")

  def testCustomMainOp(self):
    export_dir = self._get_export_dir("test_main_op")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      # Add `v1` and `v2` variables to the graph.
      v1 = variables.VariableV1(1, name="v1")
      ops.add_to_collection("v", v1)
      v2 = variables.VariableV1(2, name="v2")
      ops.add_to_collection("v", v2)

      # Initialize another variable `v3` to 42.
      v3 = variables.VariableV1(42, name="v3")
      ops.add_to_collection("v", v3)

      # Set up an assignment op to be run as part of the main_op.
      with ops.control_dependencies([main_op.main_op()]):
        add_v1_v2 = math_ops.add(v1._ref(), v2._ref())
        custom_main_op = control_flow_ops.group(state_ops.assign(v3, add_v1_v2))

      self.evaluate(custom_main_op)
      builder.add_meta_graph_and_variables(
          sess, ["foo"], main_op=custom_main_op)

    # Save the SavedModel to disk.
    builder.save()

    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo"], export_dir)
      self.assertEqual(1, ops.get_collection("v")[0].eval())
      self.assertEqual(2, ops.get_collection("v")[1].eval())
      # Evaluates to the sum of the first two variables and assigned as part of
      # the main_op, following a restore.
      self.assertEqual(3, ops.get_collection("v")[2].eval())

  def testLegacyInitOp(self):
    export_dir = self._get_export_dir("test_legacy_init_op")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      # Add `v1` and `v2` variables to the graph.
      v1 = variables.VariableV1(1, name="v1")
      ops.add_to_collection("v", v1)
      v2 = variables.VariableV1(2, name="v2")
      ops.add_to_collection("v", v2)

      # Initialize another variable `v3` to 42.
      v3 = variables.VariableV1(42, name="v3", trainable=False, collections=[])
      ops.add_to_collection("v", v3)

      # Set up an assignment op to be run as part of the legacy_init_op.
      assign_v3 = state_ops.assign(v3, math_ops.add(v1, v2))
      legacy_init_op = control_flow_ops.group(assign_v3, name="legacy_init_op")

      self.evaluate(variables.global_variables_initializer())
      builder.add_meta_graph_and_variables(
          sess, ["foo"], legacy_init_op=legacy_init_op)

    # Save the SavedModel to disk.
    builder.save()

    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo"], export_dir)
      self.assertEqual(1, ops.get_collection("v")[0].eval())
      self.assertEqual(2, ops.get_collection("v")[1].eval())
      # Evaluates to the sum of the first two variables and assigned as part of
      # the legacy_init_op, following a restore.
      self.assertEqual(3, ops.get_collection("v")[2].eval())

  def testLegacyInitOpWithNonEmptyCollection(self):
    export_dir = self._get_export_dir(
        "test_legacy_init_op_with_non_empty_collection")
    self._testInitOpsWithNonEmptyCollection(
        export_dir, constants.LEGACY_INIT_OP_KEY)

  def testMainOpWithNonEmptyCollection(self):
    export_dir = self._get_export_dir(
        "test_main_op_with_non_empty_collection")
    self._testInitOpsWithNonEmptyCollection(export_dir, constants.MAIN_OP_KEY)

  def _testInitOpsWithNonEmptyCollection(self, export_dir, key):
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    g = ops.Graph()
    with self.session(graph=g) as sess:
      # Initialize variable `v1` to 1.
      v1 = variables.VariableV1(1, name="v1")
      ops.add_to_collection("v", v1)

      # Initialize another variable `v2` to 42.
      v2 = variables.VariableV1(42, name="v2", trainable=False, collections=[])
      ops.add_to_collection("v", v2)

      # Set up an assignment op to be run as part of the init op.
      assign_v2 = state_ops.assign(v2, v1)
      init_op = control_flow_ops.group(assign_v2, name="init_op")

      self.evaluate(variables.global_variables_initializer())

      ops.add_to_collection(key, control_flow_ops.no_op())
      # ValueError should be raised since the LEGACY_INIT_OP_KEY collection
      # is not empty and we don't support multiple init ops.
      with self.assertRaisesRegexp(ValueError, "Graph already contains"):
        builder.add_meta_graph_and_variables(
            sess, ["foo"], legacy_init_op=init_op)
      # We shouldn't be able to add as MAIN_OP, either.
      with self.assertRaisesRegexp(ValueError, "Graph already contains"):
        builder.add_meta_graph_and_variables(sess, ["foo"], main_op=init_op)

  def testTrainOp(self):
    export_dir = self._get_export_dir("test_train_op")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      # Add `v1` and `v2` variables to the graph.
      v1 = variables.VariableV1(1, name="v1")
      ops.add_to_collection("v", v1)
      v2 = variables.VariableV1(2, name="v2")
      ops.add_to_collection("v", v2)

      self.evaluate(variables.global_variables_initializer())
      train_op = state_ops.assign_add(v1, v2)

      self.evaluate(train_op)
      # TODO(karmel): remove explicit call when in the public method.
      builder._add_train_op(train_op)
      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Save the SavedModel to disk.
    builder.save()

    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo"], export_dir)
      self.assertEqual(3, ops.get_collection("v")[0].eval())
      self.assertEqual(2, ops.get_collection("v")[1].eval())
      self.assertIsInstance(
          ops.get_collection(constants.TRAIN_OP_KEY)[0], ops.Tensor)

  def testTrainOpGroup(self):
    export_dir = self._get_export_dir("test_train_op_group")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      # Add `v1` and `v2` variables to the graph.
      v1 = variables.VariableV1(1, name="v1")
      ops.add_to_collection("v", v1)
      v2 = variables.VariableV1(2, name="v2")
      ops.add_to_collection("v", v2)

      self.evaluate(variables.global_variables_initializer())
      train_op = control_flow_ops.group()

      self.evaluate(train_op)
      # TODO(karmel): remove explicit call when in the public method.
      builder._add_train_op(train_op)
      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Save the SavedModel to disk.
    builder.save()

    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo"], export_dir)
      self.assertEqual(1, ops.get_collection("v")[0].eval())
      self.assertEqual(2, ops.get_collection("v")[1].eval())
      self.assertIsInstance(
          ops.get_collection(constants.TRAIN_OP_KEY)[0], ops.Operation)

  def testTrainOpAfterVariables(self):
    export_dir = self._get_export_dir("test_train_op_after_variables")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      # Add `v1` and `v2` variables to the graph.
      v1 = variables.VariableV1(1, name="v1")
      ops.add_to_collection("v", v1)
      v2 = variables.VariableV1(2, name="v2")
      ops.add_to_collection("v", v2)

      self.evaluate(variables.global_variables_initializer())
      builder.add_meta_graph_and_variables(sess, ["pre_foo"])

      train_op = state_ops.assign_add(v1, v2)
      self.evaluate(train_op)
      # TODO(karmel): remove explicit call when in the public method.
      builder._add_train_op(train_op)
      builder.add_meta_graph(["foo"])

    # Save the SavedModel to disk.
    builder.save()

    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo"], export_dir)
      self.assertIsInstance(
          ops.get_collection(constants.TRAIN_OP_KEY)[0], ops.Tensor)

    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["pre_foo"], export_dir)
      self.assertFalse(ops.get_collection(constants.TRAIN_OP_KEY))

  def testMultipleAssets(self):
    export_dir = self._get_export_dir("test_multiple_assets")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      # Build an asset collection specific to `foo` graph.
      asset_list = self._build_asset_collection("foo.txt", "content_foo",
                                                "asset_file_tensor")

      # Add the asset collection as part of the graph with tag "foo".
      builder.add_meta_graph_and_variables(
          sess, ["foo"], assets_list=asset_list)

    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      # Build an asset collection specific to `bar` graph.
      asset_list = self._build_asset_collection("bar.txt", "content_bar",
                                                "asset_file_tensor")

      # Add the asset collection as part of the graph with tag "bar".
      builder.add_meta_graph(["bar"], assets_list=asset_list)

    # Save the SavedModel to disk.
    builder.save()

    # Check assets restored for graph with tag "foo".
    with self.session(graph=ops.Graph()) as sess:
      foo_graph = loader.load(sess, ["foo"], export_dir)
      self._validate_assets(export_dir, foo_graph.asset_file_def, "foo.txt",
                            "content_foo", "asset_file_tensor:0")

    # Check assets restored for graph with tag "bar".
    with self.session(graph=ops.Graph()) as sess:
      bar_graph = loader.load(sess, ["bar"], export_dir)
      self._validate_assets(export_dir, bar_graph.asset_file_def, "bar.txt",
                            "content_bar", "asset_file_tensor:0")

  def testDuplicateAssets(self):
    export_dir = self._get_export_dir("test_duplicate_assets")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      # Build an asset collection with `foo.txt` that has `foo` specific
      # content.
      asset_list = self._build_asset_collection("foo.txt", "content_foo",
                                                "asset_file_tensor")

      # Add the asset collection as part of the graph with tag "foo".
      builder.add_meta_graph_and_variables(
          sess, ["foo"], assets_list=asset_list)

    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      # Build an asset collection with `foo.txt` that has `bar` specific
      # content.
      asset_list = self._build_asset_collection("foo.txt", "content_bar",
                                                "asset_file_tensor")

      # Add the asset collection as part of the graph with tag "bar".
      builder.add_meta_graph(["bar"], assets_list=asset_list)

    # Save the SavedModel to disk.
    builder.save()

    # Check assets restored for graph with tag "foo".
    with self.session(graph=ops.Graph()) as sess:
      foo_graph = loader.load(sess, ["foo"], export_dir)
      self._validate_assets(export_dir, foo_graph.asset_file_def, "foo.txt",
                            "content_foo", "asset_file_tensor:0")

    # Check assets restored for graph with tag "bar".
    with self.session(graph=ops.Graph()) as sess:
      bar_graph = loader.load(sess, ["bar"], export_dir)

      # Validate the assets for `bar` graph. `foo.txt` should contain the
      # original contents corresponding to `foo` graph since an asset with the
      # same name across multiple graphs is only stored the first time
      self._validate_assets(export_dir, bar_graph.asset_file_def, "foo.txt",
                            "content_foo", "asset_file_tensor:0")

  def testOp(self):
    export_dir = self._get_export_dir("test_op")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with session.Session(
        graph=ops.Graph(),
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v1 = variables.VariableV1(1, name="v1")
      with sess.graph.device("/cpu:1"):
        v2 = variables.VariableV1(2, name="v2")

      # v3 is an unsaved variable derived from v1 and v2.  It is used to
      # exercise the ability to run an init op when restoring a graph.
      v3 = variables.VariableV1(1, name="v3", trainable=False, collections=[])
      assign_v3 = state_ops.assign(v3, math_ops.add(v1, v2))
      init_op = control_flow_ops.group(assign_v3, name="init_op")

      ops.add_to_collection("v", v1)
      ops.add_to_collection("v", v2)
      ops.add_to_collection("v", v3)
      ops.add_to_collection("init_op", init_op)

      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1, ops.get_collection("v")[0].eval())
      self.assertEqual(2, ops.get_collection("v")[1].eval())

      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Save the SavedModel to disk.
    builder.save()

    with session.Session(
        graph=ops.Graph(),
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      loader.load(sess, ["foo"], export_dir)

      # Validate variables, run the init op and verify result.
      self.assertEqual(1, ops.get_collection("v")[0].eval())
      self.assertEqual(2, ops.get_collection("v")[1].eval())
      ops.get_collection("init_op")[0].run()
      self.assertEqual(3, ops.get_collection("v")[2].eval())

  def testCustomSaveable(self):
    export_dir = self._get_export_dir("custom_saveable")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with session.Session(
        graph=ops.Graph(),
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      # CheckpointedOp is a key-value table that can be saved across sessions.
      # The table register itself in SAVEABLE_OBJECTS collection.
      v1 = saver_test_utils.CheckpointedOp(name="v1")
      variables.global_variables_initializer().run()
      v1.insert("k1", 3.0).run()
      # Once the table is restored, we can access it through this reference.
      ops.add_to_collection("table_ref", v1.table_ref)
      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Save the SavedModel to disk.
    builder.save()

    with session.Session(
        graph=ops.Graph(),
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      loader.load(sess, ["foo"], export_dir)
      # Instantiate a wrapper object from the checkpointed reference.
      v1 = saver_test_utils.CheckpointedOp(
          name="v1", table_ref=ops.get_collection("table_ref")[0])
      self.assertEqual(b"k1", v1.keys().eval())
      self.assertEqual(3.0, v1.values().eval())

  def testCustomSaver(self):
    export_dir = self._get_export_dir("test_custom_saver")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      variables.VariableV1(1, name="v1")
      self.evaluate(variables.global_variables_initializer())
      custom_saver = training.Saver(name="my_saver")
      builder.add_meta_graph_and_variables(sess, ["tag"], saver=custom_saver)

    # Save the SavedModel to disk.
    builder.save()

    with ops.Graph().as_default() as graph:
      with self.session(graph=graph) as sess:
        saved_graph = loader.load(sess, ["tag"], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue("my_saver/restore_all" in graph_ops)
        self.assertFalse("save/restore_all" in graph_ops)
        self.assertEqual(
            saved_graph.saver_def.restore_op_name, "my_saver/restore_all")

  def testNoCustomSaver(self):
    export_dir = self._get_export_dir("test_no_custom_saver")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      variables.VariableV1(1, name="v1")
      self.evaluate(variables.global_variables_initializer())
      training.Saver(name="my_saver")
      builder.add_meta_graph_and_variables(sess, ["tag"])

    # Save the SavedModel to disk.
    builder.save()

    with ops.Graph().as_default() as graph:
      with self.session(graph=graph) as sess:
        saved_graph = loader.load(sess, ["tag"], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue("my_saver/restore_all" in graph_ops)
        self.assertTrue("save/restore_all" in graph_ops)
        self.assertEqual(
            saved_graph.saver_def.restore_op_name, "save/restore_all")

  def testMultipleCustomSavers(self):
    export_dir = self._get_export_dir("test_multiple_custom_savers")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      variables.VariableV1(1, name="v1")
      self.evaluate(variables.global_variables_initializer())
      builder.add_meta_graph_and_variables(sess, ["tag_0"])

      saver_1 = training.Saver()
      builder.add_meta_graph(["tag_1"], saver=saver_1)

      saver_2 = training.Saver()
      builder.add_meta_graph(["tag_2"], saver=saver_2)

    # Save the SavedModel to disk.
    builder.save()

    def _validate_custom_saver(tag_name, saver_name):
      with ops.Graph().as_default() as graph:
        with self.session(graph=graph) as sess:
          saved_graph = loader.load(sess, [tag_name], export_dir)
          self.assertEqual(
              saved_graph.saver_def.restore_op_name,
              saver_name)

    _validate_custom_saver("tag_0", "save/restore_all")
    _validate_custom_saver("tag_1", "save_1/restore_all")
    _validate_custom_saver("tag_2", "save_2/restore_all")

  def testImportScope(self):
    export_dir = self._get_export_dir("test_scoped_assets")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    # Build a SavedModel with a variable, an asset, and a constant tensor.
    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)
      asset_list = self._build_asset_collection("foo.txt", "content_foo",
                                                "asset_file_tensor")
      constant_op.constant("constant value", name="constant_tensor_name")
      builder.add_meta_graph_and_variables(
          sess, ["tag_name"], assets_list=asset_list)

      # Save the asset file path for later comparison.
      asset_file_path = asset_list[0].eval()

    # Save the SavedModel to disk.
    builder.save()

    with self.session(graph=ops.Graph()) as sess:
      # Restore the SavedModel under an import_scope in a new graph/session.
      graph_proto = loader.load(
          sess, ["tag_name"], export_dir, import_scope="scope_name")

      # The loaded variable tensor should be scoped, but its contents should be
      # unchanged.
      self.assertEqual(
          "scope_name/v:0",
          ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].name)
      self.assertEqual(
          42,
          ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].eval())

      # The loaded asset tensor should be scoped, but the asset file path and
      # contents should be unchanged.
      asset_list = ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)
      self.assertEqual(1, len(asset_list))
      self.assertEqual(asset_file_path, asset_list[0].eval())
      self.assertEqual("scope_name/asset_file_tensor:0", asset_list[0].name)
      # The static asset data inside graph_proto.collection_def should not be
      # scoped.
      self._validate_assets(export_dir, graph_proto.asset_file_def, "foo.txt",
                            "content_foo", "asset_file_tensor:0")

      # The constant tensor should be scoped, but its contents should be
      # unchanged.
      self.assertEqual(
          compat.as_bytes("constant value"),
          ops.get_default_graph().get_tensor_by_name(
              "scope_name/constant_tensor_name:0").eval())

  def testClearDevices(self):
    export_dir = self._get_export_dir("test_clear_devices")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    # Specify a device and save a variable.
    ops.reset_default_graph()
    with session.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        self._init_and_validate_variable(sess, "v", 42)
        builder.add_meta_graph_and_variables(
            sess, [tag_constants.TRAINING], clear_devices=True)

    # Save the SavedModel to disk.
    builder.save()

    # Restore the graph with a single predefined tag whose variables were saved
    # without any device information.
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, [tag_constants.TRAINING], export_dir)
      self.assertEqual(
          42, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)[0].eval())

  def testStripDefaultAttrs(self):
    export_dir = self._get_export_dir("test_strip_default_attrs")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    # Add a graph with two float32 variables and a Complex Op composing them
    # with strip_default_attrs enabled.
    with session.Session(graph=ops.Graph()) as sess:
      real_num = variables.VariableV1(1.0, dtype=dtypes.float32, name="real")
      imag_num = variables.VariableV1(2.0, dtype=dtypes.float32, name="imag")
      math_ops.complex(real_num, imag_num, name="complex")
      self.evaluate(variables.global_variables_initializer())
      builder.add_meta_graph_and_variables(
          sess, ["foo"], strip_default_attrs=True)

    # Add a graph with the same float32 variables and a Complex Op composing
    # them with strip_default_attrs disabled.
    with session.Session(graph=ops.Graph()) as sess:
      real_num = variables.VariableV1(1.0, dtype=dtypes.float32, name="real")
      imag_num = variables.VariableV1(2.0, dtype=dtypes.float32, name="imag")
      math_ops.complex(real_num, imag_num, name="complex")
      self.evaluate(variables.global_variables_initializer())
      builder.add_meta_graph(["bar"], strip_default_attrs=False)

    # Save the SavedModel to disk in text format.
    builder.save(as_text=True)

    # Loading graph "foo" via the loader must restore the defaults for the
    # "Complex" node based on the "Complex" OpDef in the Op registry.
    sess = session.Session(graph=ops.Graph())
    meta_graph_def = loader.load(sess, ["foo"], export_dir)
    complex_node = test_util.get_node_def_from_graph("complex",
                                                     meta_graph_def.graph_def)
    self.assertIn("T", complex_node.attr)
    self.assertIn("Tout", complex_node.attr)

    # Load graph "foo" from disk as-is to verify default attrs are stripped.
    # pylint: disable=protected-access
    saved_model_pb = loader_impl._parse_saved_model(export_dir)
    self.assertIsNotNone(saved_model_pb)
    # pylint: enable=protected-access

    meta_graph_foo_def = None
    meta_graph_bar_def = None
    for meta_graph_def in saved_model_pb.meta_graphs:
      if set(meta_graph_def.meta_info_def.tags) == set(["foo"]):
        meta_graph_foo_def = meta_graph_def
      elif set(meta_graph_def.meta_info_def.tags) == set(["bar"]):
        meta_graph_bar_def = meta_graph_def

    self.assertIsNotNone(meta_graph_foo_def)
    self.assertIsNotNone(meta_graph_bar_def)

    # "Complex" Op has 2 attributes with defaults:
    #   o "T"    : float32.   (input type)
    #   o "Tout" : complex64. (output type)

    # "Complex" Op in graph "foo" shouldn't have attributes "T" and "Tout".
    # Graph "foo" was saved with strip_default_attrs set to True.
    node_def = test_util.get_node_def_from_graph("complex",
                                                 meta_graph_foo_def.graph_def)
    self.assertNotIn("T", node_def.attr)
    self.assertNotIn("Tout", node_def.attr)

    # "Complex" Op in graph "bar" must have attributes "T" and "Tout".
    # Graph "bar" was saved with strip_default_attrs set to False.
    node_def = test_util.get_node_def_from_graph("complex",
                                                 meta_graph_bar_def.graph_def)
    self.assertIn("T", node_def.attr)
    self.assertIn("Tout", node_def.attr)

  # Tests the behavior of loading SavedModels that having missing attrs or attrs
  # with incorrect types.
  def testInconsistentConsumerDefaultAttrs(self):
    export_dir = self._get_export_dir(
        "test_strip_default_attrs_no_consumer_defaults")
    builder = saved_model_builder._SavedModelBuilder(export_dir)

    # Add a graph with a single variable and a test op with a defaultless
    # float32 attr, "test_attr".
    with session.Session(graph=ops.Graph()) as sess:
      variables.VariableV1(1.0, dtype=dtypes.float64, name="var")
      test_ops.test_attr(T=dtypes.float32, name="test_attr")
      self.evaluate(variables.global_variables_initializer())
      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Save the SavedModel to disk in text format.
    builder.save(as_text=True)

    # Rewrite the SavedModel to remove the T attr from "test_attr".
    saved_model_file = os.path.join(
        export_dir, constants.SAVED_MODEL_FILENAME_PBTXT)
    with open(saved_model_file) as f:
      original_saved_model = f.read()

    no_attr_saved_model = original_saved_model.replace("""
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }""", "")
    with open(saved_model_file, "w") as f:
      f.write(no_attr_saved_model)

    # Loading the SavedModel via the loader must fail because the SavedModel
    # does not have any attr values for the "TestAttr" node, and there is no
    # default specified in the TestAttr OpDef.
    sess = session.Session(graph=ops.Graph())
    with self.assertRaisesRegexp(
        ValueError, "NodeDef missing attr 'T' from Op<name=TestAttr"):
      loader.load(sess, ["foo"], export_dir)

    # Rewrite the SavedModel to change the type of the T attr in "test_attr"
    bad_type_saved_model = original_saved_model.replace("""
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }""", """
      attr {
        key: "T"
        value {
          type: DT_DOUBLE
        }
      }""")
    with open(saved_model_file, "w") as f:
      f.write(bad_type_saved_model)

    # Loading the SavedModel via the loader must fail because there is no
    # OpKernel registered to handle T = double.
    sess = session.Session(graph=ops.Graph())
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "No OpKernel was registered to support Op 'TestAttr' used by node "
        "test_attr \\(defined at .*\\) with these attrs: \\[.*\\]\n"
        "Registered devices:.*\n"
        "Registered kernels:.*"
    ):
      loader.load(sess, ["foo"], export_dir)


class SavedModelV1Test(SavedModelTestBase):

  def _validate_asset_collection(self,
                                 export_dir,
                                 graph_collection_def,
                                 expected_asset_file_name,
                                 expected_asset_file_contents,
                                 expected_asset_tensor_name,
                                 asset_id=0):
    assets_any = graph_collection_def[constants.ASSETS_KEY].any_list.value
    asset = meta_graph_pb2.AssetFileDef()
    assets_any[asset_id].Unpack(asset)
    assets_path = os.path.join(
        compat.as_bytes(export_dir),
        compat.as_bytes(constants.ASSETS_DIRECTORY),
        compat.as_bytes(expected_asset_file_name))
    actual_asset_contents = file_io.read_file_to_string(assets_path)
    self.assertEqual(expected_asset_file_contents,
                     compat.as_text(actual_asset_contents))
    self.assertEqual(expected_asset_file_name, asset.filename)
    self.assertEqual(expected_asset_tensor_name, asset.tensor_info.name)

  def testWritingAssetsToCollection(self):
    export_dir = self._get_export_dir("test_writing_assets_to_collection")
    builder = saved_model_builder.SavedModelBuilder(export_dir)

    with self.session(graph=ops.Graph()) as sess:
      self._init_and_validate_variable(sess, "v", 42)

      # Build an asset list.
      ignored_filepath = os.path.join(
          compat.as_bytes(test.get_temp_dir()), compat.as_bytes("ignored.txt"))
      file_io.write_string_to_file(ignored_filepath, "will be ignored")

      asset_collection = self._build_asset_collection(
          "hello42.txt", "foo bar baz", "asset_file_tensor")

      builder.add_meta_graph_and_variables(
          sess, ["foo"], assets_collection=asset_collection)

    # Save the SavedModel to disk.
    builder.save()

    with self.session(graph=ops.Graph()) as sess:
      foo_graph = loader.load(sess, ["foo"], export_dir)
      self._validate_asset_collection(export_dir, foo_graph.collection_def,
                                      "hello42.txt", "foo bar baz",
                                      "asset_file_tensor:0")
      ignored_asset_path = os.path.join(
          compat.as_bytes(export_dir),
          compat.as_bytes(constants.ASSETS_DIRECTORY),
          compat.as_bytes("ignored.txt"))
      self.assertFalse(file_io.file_exists(ignored_asset_path))


if __name__ == "__main__":
  test.main()
