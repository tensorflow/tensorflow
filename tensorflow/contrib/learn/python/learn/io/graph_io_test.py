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

"""Tests for learn.io.graph_io."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tempfile

import tensorflow as tf

from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile

_VALID_FILE_PATTERN = "VALID"
_FILE_NAMES = [b"abc", b"def", b"ghi", b"jkl"]
_INVALID_FILE_PATTERN = "INVALID"


class GraphIOTest(tf.test.TestCase):

  def _mock_glob(self, pattern):
    if _VALID_FILE_PATTERN == pattern:
      return _FILE_NAMES
    self.assertEqual(_INVALID_FILE_PATTERN, pattern)
    return []

  def setUp(self):
    super(GraphIOTest, self).setUp()
    random.seed(42)
    self._orig_glob = gfile.Glob
    gfile.Glob = self._mock_glob

  def tearDown(self):
    gfile.Glob = self._orig_glob
    super(GraphIOTest, self).tearDown()

  def test_dequeue_batch_value_errors(self):
    default_batch_size = 17
    queue_capacity = 1234
    num_threads = 3
    name = "my_batch"

    self.assertRaisesRegexp(
        ValueError, "No files match",
        tf.contrib.learn.io.read_batch_examples,
        _INVALID_FILE_PATTERN, default_batch_size, tf.TFRecordReader,
        False, num_epochs=None, queue_capacity=queue_capacity,
        num_threads=num_threads, name=name)
    self.assertRaisesRegexp(
        ValueError, "Invalid batch_size",
        tf.contrib.learn.io.read_batch_examples,
        _VALID_FILE_PATTERN, None, tf.TFRecordReader,
        False, num_epochs=None, queue_capacity=queue_capacity,
        num_threads=num_threads, name=name)
    self.assertRaisesRegexp(
        ValueError, "Invalid batch_size",
        tf.contrib.learn.io.read_batch_examples,
        _VALID_FILE_PATTERN, -1, tf.TFRecordReader,
        False, num_epochs=None, queue_capacity=queue_capacity,
        num_threads=num_threads, name=name)
    self.assertRaisesRegexp(
        ValueError, "Invalid queue_capacity",
        tf.contrib.learn.io.read_batch_examples,
        _VALID_FILE_PATTERN, default_batch_size, tf.TFRecordReader,
        False, num_epochs=None, queue_capacity=None,
        num_threads=num_threads, name=name)
    self.assertRaisesRegexp(
        ValueError, "Invalid num_threads",
        tf.contrib.learn.io.read_batch_examples,
        _VALID_FILE_PATTERN, default_batch_size, tf.TFRecordReader,
        False, num_epochs=None, queue_capacity=queue_capacity,
        num_threads=None, name=name)
    self.assertRaisesRegexp(
        ValueError, "Invalid num_threads",
        tf.contrib.learn.io.read_batch_examples,
        _VALID_FILE_PATTERN, default_batch_size, tf.TFRecordReader,
        False, num_epochs=None, queue_capacity=queue_capacity,
        num_threads=-1, name=name)
    self.assertRaisesRegexp(
        ValueError, "Invalid batch_size",
        tf.contrib.learn.io.read_batch_examples,
        _VALID_FILE_PATTERN, queue_capacity + 1, tf.TFRecordReader,
        False, num_epochs=None, queue_capacity=queue_capacity,
        num_threads=1, name=name)
    self.assertRaisesRegexp(
        ValueError, "Invalid num_epochs",
        tf.contrib.learn.io.read_batch_examples,
        _VALID_FILE_PATTERN, default_batch_size, tf.TFRecordReader,
        False, num_epochs=-1, queue_capacity=queue_capacity, num_threads=1,
        name=name)

  def test_batch_record_features(self):
    batch_size = 17
    queue_capacity = 1234
    name = "my_batch"
    features = {"feature": tf.FixedLenFeature(shape=[0], dtype=tf.float32)}

    with tf.Graph().as_default() as g, self.test_session(graph=g) as sess:
      features = tf.contrib.learn.io.read_batch_record_features(
          _VALID_FILE_PATTERN, batch_size, features, randomize_input=False,
          queue_capacity=queue_capacity, reader_num_threads=2,
          parser_num_threads=2, name=name)
      self.assertEqual("%s/parse_example_batch_join:0" % name,
                       features["feature"].name)
      file_name_queue_name = "%s/file_name_queue" % name
      file_names_name = "%s/input" % file_name_queue_name
      example_queue_name = "%s/fifo_queue" % name
      parse_example_queue_name = "%s/parse_example_batch_join" % name
      op_nodes = test_util.assert_ops_in_graph({
          file_names_name: "Const",
          file_name_queue_name: "FIFOQueue",
          "%s/read/TFRecordReader" % name: "TFRecordReader",
          example_queue_name: "FIFOQueue",
          parse_example_queue_name: "QueueDequeueMany",
          name: "QueueDequeueMany"
      }, g)
      self.assertAllEqual(_FILE_NAMES, sess.run(["%s:0" % file_names_name])[0])
      self.assertEqual(
          queue_capacity, op_nodes[example_queue_name].attr["capacity"].i)

  def test_one_epoch(self):
    batch_size = 17
    queue_capacity = 1234
    name = "my_batch"

    with tf.Graph().as_default() as g, self.test_session(graph=g) as sess:
      inputs = tf.contrib.learn.io.read_batch_examples(
          _VALID_FILE_PATTERN, batch_size,
          reader=tf.TFRecordReader, randomize_input=True,
          num_epochs=1,
          queue_capacity=queue_capacity, name=name)
      self.assertEqual("%s:0" % name, inputs.name)
      file_name_queue_name = "%s/file_name_queue" % name
      file_name_queue_limit_name = (
          "%s/limit_epochs/epochs" % file_name_queue_name)
      file_names_name = "%s/input" % file_name_queue_name
      example_queue_name = "%s/random_shuffle_queue" % name
      op_nodes = test_util.assert_ops_in_graph({
          file_names_name: "Const",
          file_name_queue_name: "FIFOQueue",
          "%s/read/TFRecordReader" % name: "TFRecordReader",
          example_queue_name: "RandomShuffleQueue",
          name: "QueueDequeueMany",
          file_name_queue_limit_name: "Variable"
      }, g)
      self.assertEqual(
          set(_FILE_NAMES), set(sess.run(["%s:0" % file_names_name])[0]))
      self.assertEqual(
          queue_capacity, op_nodes[example_queue_name].attr["capacity"].i)

  def test_batch_randomized(self):
    batch_size = 17
    queue_capacity = 1234
    name = "my_batch"

    with tf.Graph().as_default() as g, self.test_session(graph=g) as sess:
      inputs = tf.contrib.learn.io.read_batch_examples(
          _VALID_FILE_PATTERN, batch_size,
          reader=tf.TFRecordReader, randomize_input=True,
          queue_capacity=queue_capacity, name=name)
      self.assertEqual("%s:0" % name, inputs.name)
      file_name_queue_name = "%s/file_name_queue" % name
      file_names_name = "%s/input" % file_name_queue_name
      example_queue_name = "%s/random_shuffle_queue" % name
      op_nodes = test_util.assert_ops_in_graph({
          file_names_name: "Const",
          file_name_queue_name: "FIFOQueue",
          "%s/read/TFRecordReader" % name: "TFRecordReader",
          example_queue_name: "RandomShuffleQueue",
          name: "QueueDequeueMany"
      }, g)
      self.assertEqual(
          set(_FILE_NAMES), set(sess.run(["%s:0" % file_names_name])[0]))
      self.assertEqual(
          queue_capacity, op_nodes[example_queue_name].attr["capacity"].i)

  def test_read_csv(self):
    gfile.Glob = self._orig_glob
    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, "file.csv")
    gfile.Open(filename, "w").write("ABC\nDEF\nGHK\n")

    batch_size = 1
    queue_capacity = 5
    name = "my_batch"

    with tf.Graph().as_default() as g, self.test_session(graph=g) as session:
      inputs = tf.contrib.learn.io.read_batch_examples(
          filename, batch_size,
          reader=tf.TextLineReader, randomize_input=False,
          num_epochs=1, queue_capacity=queue_capacity, name=name)
      session.run(tf.initialize_all_variables())

      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(session, coord=coord)

      self.assertAllEqual(session.run(inputs), [b"ABC"])
      self.assertAllEqual(session.run(inputs), [b"DEF"])
      self.assertAllEqual(session.run(inputs), [b"GHK"])
      with self.assertRaises(errors.OutOfRangeError):
        session.run(inputs)

      coord.request_stop()


if __name__ == "__main__":
  tf.test.main()
