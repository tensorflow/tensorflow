# Copyright 2016 Google Inc. All Rights Reserved.
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
from tensorflow.python.framework import graph_util
from tensorflow.python.util import compat


class SessionBundleLoadTest(tf.test.TestCase):

  def testBasic(self):
    base_path = tf.test.test_src_dir_path(
        "contrib/session_bundle/example/half_plus_two/00000123")
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
      default_signature = signatures.default_signature
      input_name = default_signature.regression_signature.input.tensor_name
      output_name = default_signature.regression_signature.output.tensor_name
      y = sess.run([output_name], {input_name: np.array([[0], [1], [2], [3]])})
      # The operation is y = 0.5 * x + 2
      self.assertEqual(y[0][0], 2)
      self.assertEqual(y[0][1], 2.5)
      self.assertEqual(y[0][2], 3)
      self.assertEqual(y[0][3], 3.5)

  def testBadPath(self):
    base_path = tf.test.test_src_dir_path("/no/such/a/dir")
    tf.reset_default_graph()
    with self.assertRaises(RuntimeError) as cm:
      _, _ = session_bundle.load_session_bundle_from_path(
          base_path, target="local",
          config=tf.ConfigProto(device_count={"CPU": 2}))
    self.assertTrue("Expected meta graph file missing" in str(cm.exception))


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
        tf.initialize_all_variables().run()
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
