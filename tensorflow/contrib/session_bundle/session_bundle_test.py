# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for session_bundle.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib.session_bundle import constants

from tensorflow.contrib.session_bundle import manifest_pb2
from tensorflow.contrib.session_bundle import session_bundle
from tensorflow.core.example.example_pb2 import Example
from tensorflow.python.framework import graph_util
from tensorflow.python.util import compat

SAVED_MODEL_PATH = (
    "python/saved_model/example/saved_model_half_plus_two/00000123")
SESSION_BUNDLE_PATH = "contrib/session_bundle/testdata/half_plus_two/00000123"


def _make_serialized_example(x):
  example = Example()
  example.features.feature["x"].float_list.value.append(x)
  return example.SerializeToString()


class SessionBundleLoadTest(tf.test.TestCase):

  def _checkRegressionSignature(self, signatures, sess):
    default_signature = signatures.default_signature
    input_name = default_signature.regression_signature.input.tensor_name
    output_name = default_signature.regression_signature.output.tensor_name
    tf_example = [_make_serialized_example(x) for x in [0, 1, 2, 3]]
    y = sess.run([output_name], {input_name: tf_example})
    # The operation is y = 0.5 * x + 2
    self.assertEqual(y[0][0], 2)
    self.assertEqual(y[0][1], 2.5)
    self.assertEqual(y[0][2], 3)
    self.assertEqual(y[0][3], 3.5)

  def _checkNamedSignatures(self, signatures, sess):
    named_signatures = signatures.named_signatures
    input_name = (named_signatures["inputs"].generic_signature.map["x"]
                  .tensor_name)
    output_name = (named_signatures["outputs"].generic_signature.map["y"]
                   .tensor_name)
    y = sess.run([output_name], {input_name: np.array([[0], [1], [2], [3]])})
    # The operation is y = 0.5 * x + 2
    self.assertEqual(y[0][0], 2)
    self.assertEqual(y[0][1], 2.5)
    self.assertEqual(y[0][2], 3)
    self.assertEqual(y[0][3], 3.5)

  def testMaybeSessionBundleDir(self):
    base_path = tf.test.test_src_dir_path(SESSION_BUNDLE_PATH)
    self.assertTrue(session_bundle.maybe_session_bundle_dir(base_path))
    base_path = tf.test.test_src_dir_path(SAVED_MODEL_PATH)
    self.assertFalse(session_bundle.maybe_session_bundle_dir(base_path))
    base_path = "complete_garbage"
    self.assertFalse(session_bundle.maybe_session_bundle_dir(base_path))

  def testBasic(self):
    base_path = tf.test.test_src_dir_path(SESSION_BUNDLE_PATH)
    tf.reset_default_graph()
    sess, meta_graph_def = session_bundle.load_session_bundle_from_path(
        base_path, target="", config=tf.ConfigProto(device_count={"CPU": 2}))

    self.assertTrue(sess)
    asset_path = os.path.join(base_path, constants.ASSETS_DIRECTORY)
    with sess.as_default():
      path1, path2 = sess.run(["filename1:0", "filename2:0"])
      self.assertEqual(
          compat.as_bytes(os.path.join(asset_path, "hello1.txt")), path1)
      self.assertEqual(
          compat.as_bytes(os.path.join(asset_path, "hello2.txt")), path2)

      collection_def = meta_graph_def.collection_def

      signatures_any = collection_def[constants.SIGNATURES_KEY].any_list.value
      self.assertEquals(len(signatures_any), 1)

      signatures = manifest_pb2.Signatures()
      signatures_any[0].Unpack(signatures)
      self._checkRegressionSignature(signatures, sess)
      self._checkNamedSignatures(signatures, sess)

  def testBadPath(self):
    base_path = tf.test.test_src_dir_path("/no/such/a/dir")
    tf.reset_default_graph()
    with self.assertRaises(RuntimeError) as cm:
      _, _ = session_bundle.load_session_bundle_from_path(
          base_path, target="local",
          config=tf.ConfigProto(device_count={"CPU": 2}))
    self.assertTrue("Expected meta graph file missing" in str(cm.exception))

  def testVarCheckpointV2(self):
    base_path = tf.test.test_src_dir_path(
        "contrib/session_bundle/testdata/half_plus_two_ckpt_v2/00000123")
    tf.reset_default_graph()
    sess, meta_graph_def = session_bundle.load_session_bundle_from_path(
        base_path, target="", config=tf.ConfigProto(device_count={"CPU": 2}))

    self.assertTrue(sess)
    asset_path = os.path.join(base_path, constants.ASSETS_DIRECTORY)
    with sess.as_default():
      path1, path2 = sess.run(["filename1:0", "filename2:0"])
      self.assertEqual(
          compat.as_bytes(os.path.join(asset_path, "hello1.txt")), path1)
      self.assertEqual(
          compat.as_bytes(os.path.join(asset_path, "hello2.txt")), path2)

      collection_def = meta_graph_def.collection_def

      signatures_any = collection_def[constants.SIGNATURES_KEY].any_list.value
      self.assertEquals(len(signatures_any), 1)

      signatures = manifest_pb2.Signatures()
      signatures_any[0].Unpack(signatures)
      self._checkRegressionSignature(signatures, sess)
      self._checkNamedSignatures(signatures, sess)


class SessionBundleLoadNoVarsTest(tf.test.TestCase):
  """Test the case where there are no variables in the graph."""

  def setUp(self):
    self.base_path = os.path.join(tf.test.get_temp_dir(), "no_vars")
    if not os.path.exists(self.base_path):
      os.mkdir(self.base_path)

    # Create a simple graph with a variable, then convert variables to
    # constants and export the graph.
    with tf.Graph().as_default() as g:
      x = tf.placeholder(tf.float32, name="x")
      w = tf.Variable(3.0)
      y = tf.sub(w * x, 7.0, name="y")  # pylint: disable=unused-variable
      tf.add_to_collection("meta", "this is meta")

      with self.test_session(graph=g) as session:
        tf.global_variables_initializer().run()
        new_graph_def = graph_util.convert_variables_to_constants(
            session, g.as_graph_def(), ["y"])

      filename = os.path.join(self.base_path, constants.META_GRAPH_DEF_FILENAME)
      tf.train.export_meta_graph(
          filename, graph_def=new_graph_def, collection_list=["meta"])

  def tearDown(self):
    shutil.rmtree(self.base_path)

  def testGraphWithoutVarsLoadsCorrectly(self):
    session, _ = session_bundle.load_session_bundle_from_path(self.base_path)
    got = session.run(["y:0"], {"x:0": 5.0})[0]
    self.assertEquals(got, 5.0 * 3.0 - 7.0)
    self.assertEquals(tf.get_collection("meta"), [b"this is meta"])


if __name__ == "__main__":
  tf.test.main()
