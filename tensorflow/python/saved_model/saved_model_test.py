## Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
import tensorflow as tf

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat


def tearDownModule():
  file_io.delete_recursively(tf.test.get_temp_dir())


class SavedModelTest(tf.test.TestCase):

  def testSequence(self):
    export_dir = os.path.join(tf.test.get_temp_dir(), "sequence")
    builder = saved_model_builder.SavedModelBuilder(export_dir)

    # Expect an assertion error since add_meta_graph_and_variables() should be
    # invoked before any add_meta_graph() calls.
    with self.test_session(graph=tf.Graph()) as sess:
      self.assertRaises(AssertionError, builder.add_meta_graph, ["foo"])

    # Expect an assertion error for multiple calls of
    # add_meta_graph_and_variables() since weights should be saved exactly once.
    with self.test_session(graph=tf.Graph()) as sess:
      v = tf.Variable(42, name="v")
      sess.run(tf.initialize_all_variables())
      self.assertEqual(42, v.eval())
      builder.add_meta_graph_and_variables(sess, ["bar"])
      self.assertRaises(AssertionError, builder.add_meta_graph_and_variables,
                        sess, ["baz"])

  def testTags(self):
    export_dir = os.path.join(
        compat.as_bytes(tf.test.get_temp_dir()), compat.as_bytes("tags"))
    builder = saved_model_builder.SavedModelBuilder(export_dir)

    # Graph with a single variable. SavedModel invoked to:
    # - add with weights.
    # - a single tag (from predefined constants).
    with self.test_session(graph=tf.Graph()) as sess:
      v = tf.Variable(42, name="v")
      sess.run(tf.initialize_all_variables())
      self.assertEqual(42, v.eval())
      builder.add_meta_graph_and_variables(sess, [constants.TAG_TRAINING])

    # Graph that updates the single variable. SavedModel invoked to:
    # - simply add the model (weights are not updated).
    # - a single tag (from predefined constants).
    with self.test_session(graph=tf.Graph()) as sess:
      v = tf.Variable(43, name="v")
      sess.run(tf.initialize_all_variables())
      self.assertEqual(43, v.eval())
      builder.add_meta_graph([constants.TAG_SERVING])

    # Graph that updates the single variable. SavedModel is invoked:
    # - to add the model (weights are not updated).
    # - multiple custom tags.
    with self.test_session(graph=tf.Graph()) as sess:
      v = tf.Variable(44, name="v")
      sess.run(tf.initialize_all_variables())
      self.assertEqual(44, v.eval())
      builder.add_meta_graph(["foo", "bar"])

    # Save the SavedModel to disk.
    builder.save()

    # Restore the graph with a single predefined tag whose variables were saved.
    with self.test_session(graph=tf.Graph()) as sess:
      loader.load(sess, [constants.TAG_TRAINING], export_dir)
      self.assertEqual(42, tf.get_collection(tf.GraphKeys.VARIABLES)[0].eval())

    # Restore the graph with a single predefined tag whose variables were not
    # saved.
    with self.test_session(graph=tf.Graph()) as sess:
      loader.load(sess, [constants.TAG_SERVING], export_dir)
      self.assertEqual(42, tf.get_collection(tf.GraphKeys.VARIABLES)[0].eval())

    # Restore the graph with multiple tags. Provide duplicate tags to test set
    # semantics.
    with self.test_session(graph=tf.Graph()) as sess:
      loader.load(sess, ["foo", "bar", "foo"], export_dir)
      self.assertEqual(42, tf.get_collection(tf.GraphKeys.VARIABLES)[0].eval())

    # Try restoring a graph with a non-existent tag. This should yield a runtime
    # error.
    with self.test_session(graph=tf.Graph()) as sess:
      self.assertRaises(RuntimeError, loader.load, sess, ["INVALID"],
                        export_dir)

    # Try restoring a graph where a subset of the tags match. Since tag matching
    # for meta graph defs follows "all" semantics, this should yield a runtime
    # error.
    with self.test_session(graph=tf.Graph()) as sess:
      self.assertRaises(RuntimeError, loader.load, sess, ["foo", "baz"],
                        export_dir)

  def testVariables(self):
    export_dir = os.path.join(
        compat.as_bytes(tf.test.get_temp_dir()), compat.as_bytes("variables"))
    builder = saved_model_builder.SavedModelBuilder(export_dir)

    # Graph with two variables. SavedModel invoked to:
    # - add with weights.
    with self.test_session(graph=tf.Graph()) as sess:
      v1 = tf.Variable(1, name="v1")
      v2 = tf.Variable(2, name="v2")
      sess.run(tf.initialize_all_variables())
      self.assertEqual(1, v1.eval())
      self.assertEqual(2, v2.eval())
      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Graph with a single variable (subset of the variables from the previous
    # graph whose weights were saved). SavedModel invoked to:
    # - simply add the model (weights are not updated).
    with self.test_session(graph=tf.Graph()) as sess:
      v2 = tf.Variable(3, name="v2")
      sess.run(tf.initialize_all_variables())
      self.assertEqual(3, v2.eval())
      builder.add_meta_graph(["bar"])

    # Graph with a single variable (disjoint set of variables from the previous
    # graph whose weights were saved). SavedModel invoked to:
    # - simply add the model (weights are not updated).
    with self.test_session(graph=tf.Graph()) as sess:
      v3 = tf.Variable(4, name="v3")
      sess.run(tf.initialize_all_variables())
      self.assertEqual(4, v3.eval())
      builder.add_meta_graph(["baz"])

    # Save the SavedModel to disk.
    builder.save()

    # Restore the graph with tag "foo", whose variables were saved.
    with self.test_session(graph=tf.Graph()) as sess:
      loader.load(sess, ["foo"], export_dir)
      collection_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
      self.assertEqual(len(collection_vars), 2)
      self.assertEqual(1, collection_vars[0].eval())
      self.assertEqual(2, collection_vars[1].eval())

    # Restore the graph with tag "bar", whose variables were not saved. Only the
    # subset of the variables added to the graph will be restored with the
    # checkpointed value.
    with self.test_session(graph=tf.Graph()) as sess:
      loader.load(sess, ["bar"], export_dir)
      collection_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
      self.assertEqual(len(collection_vars), 1)
      self.assertEqual(2, collection_vars[0].eval())

    # Try restoring the graph with tag "baz", whose variables were not saved.
    # Since this graph has a disjoint set of variables from the set that was
    # saved, this should raise an error.
    with self.test_session(graph=tf.Graph()) as sess:
      self.assertRaises(errors.NotFoundError, loader.load, sess, ["baz"],
                        export_dir)

  def testSaveAsText(self):
    export_dir = os.path.join(
        compat.as_bytes(tf.test.get_temp_dir()), compat.as_bytes("astext"))
    builder = saved_model_builder.SavedModelBuilder(export_dir)

    # Graph with a single variable. SavedModel invoked to:
    # - add with weights.
    with self.test_session(graph=tf.Graph()) as sess:
      v = tf.Variable(42, name="v")
      sess.run(tf.initialize_all_variables())
      self.assertEqual(42, v.eval())
      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Graph with the same single variable. SavedModel invoked to:
    # - simply add the model (weights are not updated).
    with self.test_session(graph=tf.Graph()) as sess:
      v = tf.Variable(43, name="v")
      sess.run(tf.initialize_all_variables())
      self.assertEqual(43, v.eval())
      builder.add_meta_graph(["bar"])

    # Save the SavedModel to disk in text format.
    builder.save(as_text=True)

    # Restore the graph with tag "foo", whose variables were saved.
    with self.test_session(graph=tf.Graph()) as sess:
      loader.load(sess, ["foo"], export_dir)
      self.assertEqual(42, tf.get_collection(tf.GraphKeys.VARIABLES)[0].eval())

    # Restore the graph with tag "bar", whose variables were not saved.
    with self.test_session(graph=tf.Graph()) as sess:
      loader.load(sess, ["bar"], export_dir)
      self.assertEqual(42, tf.get_collection(tf.GraphKeys.VARIABLES)[0].eval())

  def testCollections(self):
    export_dir = os.path.join(
        compat.as_bytes(tf.test.get_temp_dir()), compat.as_bytes("collections"))
    builder = saved_model_builder.SavedModelBuilder(export_dir)

    # Graph with a single variable added to a collection. SavedModel invoked to:
    # - add with weights.
    with self.test_session(graph=tf.Graph()) as sess:
      v = tf.Variable(42, name="v")
      tf.add_to_collection("foo_vars", v)
      sess.run(tf.initialize_all_variables())
      self.assertEqual(42, v.eval())
      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Graph with the same single variable added to a different collection.
    # SavedModel invoked to:
    # - simply add the model (weights are not updated).
    with self.test_session(graph=tf.Graph()) as sess:
      v = tf.Variable(43, name="v")
      tf.add_to_collection("bar_vars", v)
      sess.run(tf.initialize_all_variables())
      self.assertEqual(43, v.eval())
      builder.add_meta_graph(["bar"])

    # Save the SavedModel to disk.
    builder.save()

    # Restore the graph with tag "foo", whose variables were saved. The
    # collection 'foo_vars' should contain a single element. The collection
    # 'bar_vars' should not be found.
    with self.test_session(graph=tf.Graph()) as sess:
      loader.load(sess, ["foo"], export_dir)
      collection_foo_vars = tf.get_collection("foo_vars")
      self.assertEqual(len(collection_foo_vars), 1)
      self.assertEqual(42, collection_foo_vars[0].eval())

      self.assertEqual(len(tf.get_collection("bar_vars")), 0)

    # Restore the graph with tag "bar", whose variables were not saved. The
    # collection-def exported as part of the meta graph def is updated to
    # reflect the new collection. The value of the variable in the
    # collection-def corresponds to the saved value (from the previous graph
    # with tag "foo").
    with self.test_session(graph=tf.Graph()) as sess:
      loader.load(sess, ["bar"], export_dir)
      collection_bar_vars = tf.get_collection("bar_vars")
      self.assertEqual(len(collection_bar_vars), 1)
      self.assertEqual(42, collection_bar_vars[0].eval())

      self.assertEqual(len(tf.get_collection("foo_vars")), 0)

  def testSignatureDefs(self):
    export_dir = os.path.join(
        compat.as_bytes(tf.test.get_temp_dir()),
        compat.as_bytes("signature_defs"))
    builder = saved_model_builder.SavedModelBuilder(export_dir)

    # Graph with a single variable and a single entry in the signature def map.
    # SavedModel is invoked to add with weights.
    with self.test_session(graph=tf.Graph()) as sess:
      v = tf.Variable(42, name="v")
      sess.run(tf.initialize_all_variables())
      self.assertEqual(42, v.eval())
      # Build and populate an empty SignatureDef for testing.
      foo_signature = utils.build_signature_def(dict(), dict(), "foo")
      builder.add_meta_graph_and_variables(
          sess, ["foo"], signature_def_map={"foo_key": foo_signature})

    # Graph with the same single variable and multiple entries in the signature
    # def map. No weights are saved by SavedModel.
    with self.test_session(graph=tf.Graph()) as sess:
      v = tf.Variable(43, name="v")
      sess.run(tf.initialize_all_variables())
      self.assertEqual(43, v.eval())

      # Build and populate a different SignatureDef for testing.
      bar_signature = utils.build_signature_def(dict(), dict(), "bar")
      # Also, build a different SignatureDef corresponding to "foo_key" defined
      # in the previous graph.
      foo_new_signature = utils.build_signature_def(dict(), dict(), "foo_new")
      builder.add_meta_graph(
          ["bar"],
          signature_def_map={"bar_key": bar_signature,
                             "foo_key": foo_new_signature})

    # Save the SavedModel to disk.
    builder.save()

    # Restore the graph with tag "foo". The single entry in the SignatureDef map
    # corresponding to "foo_key" should exist.
    with self.test_session(graph=tf.Graph()) as sess:
      foo_graph = loader.load(sess, ["foo"], export_dir)
      self.assertEqual(42, tf.get_collection(tf.GraphKeys.VARIABLES)[0].eval())

      foo_signature = foo_graph.signature_def
      self.assertEqual(len(foo_signature), 1)
      self.assertEqual("foo", foo_signature["foo_key"].method_name)

    # Restore the graph with tag "bar". The SignatureDef map should have two
    # entries. One corresponding to "bar_key" and another corresponding to the
    # new value of "foo_key".
    with self.test_session(graph=tf.Graph()) as sess:
      bar_graph = loader.load(sess, ["bar"], export_dir)
      self.assertEqual(42, tf.get_collection(tf.GraphKeys.VARIABLES)[0].eval())

      bar_signature = bar_graph.signature_def
      self.assertEqual(len(bar_signature), 2)
      self.assertEqual("bar", bar_signature["bar_key"].method_name)
      self.assertEqual("foo_new", bar_signature["foo_key"].method_name)

  def testAssets(self):
    export_dir = os.path.join(
        compat.as_bytes(tf.test.get_temp_dir()), compat.as_bytes("with-assets"))
    builder = saved_model_builder.SavedModelBuilder(export_dir)

    with self.test_session(graph=tf.Graph()) as sess:
      v = tf.Variable(42, name="v")
      sess.run(tf.initialize_all_variables())
      self.assertEqual(42, v.eval())

      # Build an asset collection.
      asset_filepath = os.path.join(
          compat.as_bytes(tf.test.get_temp_dir()),
          compat.as_bytes("hello42.txt"))
      file_io.write_string_to_file(asset_filepath, "foo bar baz")
      asset_file_tensor = tf.constant(asset_filepath, name="asset_file_tensor")
      tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, asset_file_tensor)

      ignored_filepath = os.path.join(
          compat.as_bytes(tf.test.get_temp_dir()),
          compat.as_bytes("ignored.txt"))
      file_io.write_string_to_file(ignored_filepath, "will be ignored")

      asset_collection = tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)

      builder.add_meta_graph_and_variables(
          sess, ["foo"], assets_collection=asset_collection)

    # Save the SavedModel to disk.
    builder.save()

    with self.test_session(graph=tf.Graph()) as sess:
      foo_graph = loader.load(sess, ["foo"], export_dir)

      # Validate the assets.
      collection_def = foo_graph.collection_def
      assets_any = collection_def[constants.ASSETS_KEY].any_list.value
      self.assertEqual(len(assets_any), 1)
      asset = meta_graph_pb2.AssetFileDef()
      assets_any[0].Unpack(asset)
      assets_path = os.path.join(
          compat.as_bytes(export_dir),
          compat.as_bytes(constants.ASSETS_DIRECTORY),
          compat.as_bytes("hello42.txt"))
      asset_contents = file_io.read_file_to_string(assets_path)
      self.assertEqual("foo bar baz", compat.as_text(asset_contents))
      self.assertEqual("hello42.txt", asset.filename)
      self.assertEqual("asset_file_tensor:0", asset.tensor_info.name)
      ignored_asset_path = os.path.join(
          compat.as_bytes(export_dir),
          compat.as_bytes(constants.ASSETS_DIRECTORY),
          compat.as_bytes("ignored.txt"))
      self.assertFalse(file_io.file_exists(ignored_asset_path))

  def testOp(self):
    export_dir = os.path.join(
        compat.as_bytes(tf.test.get_temp_dir()), compat.as_bytes("op"))
    builder = saved_model_builder.SavedModelBuilder(export_dir)

    with tf.Session(
        graph=tf.Graph(),
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      with sess.graph.device("/cpu:0"):
        v1 = tf.Variable(1, name="v1")
      with sess.graph.device("/cpu:1"):
        v2 = tf.Variable(2, name="v2")

      # v3 is an unsaved variable derived from v1 and v2.  It is used to
      # exercise the ability to run an init op when restoring a graph.
      v3 = tf.Variable(1, name="v3", trainable=False, collections=[])
      assign_v3 = tf.assign(v3, tf.add(v1, v2))
      init_op = tf.group(assign_v3, name="init_op")

      tf.add_to_collection("v", v1)
      tf.add_to_collection("v", v2)
      tf.add_to_collection("v", v3)
      tf.add_to_collection("init_op", init_op)

      sess.run(tf.initialize_all_variables())
      self.assertEqual(1, tf.get_collection("v")[0].eval())
      self.assertEqual(2, tf.get_collection("v")[1].eval())

      builder.add_meta_graph_and_variables(sess, ["foo"])

    # Save the SavedModel to disk.
    builder.save()

    with tf.Session(
        graph=tf.Graph(),
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:
      loader.load(sess, ["foo"], export_dir)

      # Validate variables, run the init op and verify result.
      self.assertEqual(1, tf.get_collection("v")[0].eval())
      self.assertEqual(2, tf.get_collection("v")[1].eval())
      tf.get_collection("init_op")[0].run()
      self.assertEqual(3, tf.get_collection("v")[2].eval())


if __name__ == "__main__":
  tf.test.main()
