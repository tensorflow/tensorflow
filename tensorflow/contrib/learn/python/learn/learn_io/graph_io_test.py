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

import base64
import os
import random
import tempfile

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.learn_io.graph_io import _read_keyed_batch_examples_shared_queue
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
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
    self.assertRaisesRegexp(
        ValueError, "Invalid read_batch_size",
        tf.contrib.learn.io.read_batch_examples,
        _VALID_FILE_PATTERN, default_batch_size, tf.TFRecordReader,
        False, num_epochs=None, queue_capacity=queue_capacity,
        num_threads=1, read_batch_size=0, name=name)

  def test_batch_record_features(self):
    batch_size = 17
    queue_capacity = 1234
    name = "my_batch"
    shape = (0,)
    features = {"feature": tf.FixedLenFeature(shape=shape, dtype=tf.float32)}

    with tf.Graph().as_default() as g, self.test_session(graph=g) as sess:
      features = tf.contrib.learn.io.read_batch_record_features(
          _VALID_FILE_PATTERN,
          batch_size,
          features,
          randomize_input=False,
          queue_capacity=queue_capacity,
          reader_num_threads=2,
          name=name)
      self.assertTrue(
          "feature" in features, "'feature' missing from %s." % features.keys())
      feature = features["feature"]
      self.assertEqual("%s/fifo_queue_1_Dequeue:0" % name, feature.name)
      self.assertAllEqual((batch_size,) + shape, feature.get_shape().as_list())
      file_name_queue_name = "%s/file_name_queue" % name
      file_names_name = "%s/input" % file_name_queue_name
      example_queue_name = "%s/fifo_queue" % name
      parse_example_queue_name = "%s/fifo_queue" % name
      op_nodes = test_util.assert_ops_in_graph({
          file_names_name: "Const",
          file_name_queue_name: "FIFOQueue",
          "%s/read/TFRecordReader" % name: "TFRecordReader",
          example_queue_name: "FIFOQueue",
          parse_example_queue_name: "FIFOQueue",
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
      self.assertAllEqual((None,), inputs.get_shape().as_list())
      self.assertEqual("%s:1" % name, inputs.name)
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
          name: "QueueDequeueUpTo",
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
      self.assertAllEqual((batch_size,), inputs.get_shape().as_list())
      self.assertEqual("%s:1" % name, inputs.name)
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

  def _create_temp_file(self, lines):
    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, "temp_file")
    gfile.Open(filename, "w").write(lines)
    return filename

  def _create_sorted_temp_files(self, lines_list):
    tempdir = tempfile.mkdtemp()
    filenames = []
    for i, lines in enumerate(lines_list):
      filename = os.path.join(tempdir, "temp_file%05d" % i)
      gfile.Open(filename, "w").write(lines)
      filenames.append(filename)
    return filenames

  def test_read_text_lines(self):
    gfile.Glob = self._orig_glob
    filename = self._create_temp_file("ABC\nDEF\nGHK\n")

    batch_size = 1
    queue_capacity = 5
    name = "my_batch"

    with tf.Graph().as_default() as g, self.test_session(graph=g) as session:
      inputs = tf.contrib.learn.io.read_batch_examples(
          filename, batch_size, reader=tf.TextLineReader,
          randomize_input=False, num_epochs=1, queue_capacity=queue_capacity,
          name=name)
      self.assertAllEqual((None,), inputs.get_shape().as_list())
      session.run(tf.local_variables_initializer())

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(session, coord=coord)

      self.assertAllEqual(session.run(inputs), [b"ABC"])
      self.assertAllEqual(session.run(inputs), [b"DEF"])
      self.assertAllEqual(session.run(inputs), [b"GHK"])
      with self.assertRaises(errors.OutOfRangeError):
        session.run(inputs)

      coord.request_stop()
      coord.join(threads)

  def test_read_text_lines_large(self):
    gfile.Glob = self._orig_glob
    sequence_prefix = "abcdefghijklmnopqrstuvwxyz123456789"
    num_records = 49999
    lines = ["".join([sequence_prefix, str(l)]).encode("ascii")
             for l in xrange(num_records)]
    json_lines = ["".join(['{"features": { "feature": { "sequence": {',
                           '"bytes_list": { "value": ["',
                           base64.b64encode(l).decode("ascii"),
                           '"]}}}}}\n']) for l in lines]
    filename = self._create_temp_file("".join(json_lines))
    batch_size = 10000
    queue_capacity = 10000
    name = "my_large_batch"

    features = {"sequence": tf.FixedLenFeature([], tf.string)}

    with tf.Graph().as_default() as g, self.test_session(graph=g) as session:
      keys, result = tf.contrib.learn.read_keyed_batch_features(
          filename, batch_size, features, tf.TextLineReader,
          randomize_input=False, num_epochs=1, queue_capacity=queue_capacity,
          num_enqueue_threads=2, parse_fn=tf.decode_json_example, name=name)
      self.assertAllEqual((None,), keys.get_shape().as_list())
      self.assertEqual(1, len(result))
      self.assertAllEqual((None,), result["sequence"].get_shape().as_list())
      session.run(tf.local_variables_initializer())
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(session, coord=coord)

      data = []
      try:
        while not coord.should_stop():
          data.append(session.run(result))
      except errors.OutOfRangeError:
        pass
      finally:
        coord.request_stop()

      coord.join(threads)

    parsed_records = [item for sublist in [d["sequence"] for d in data]
                      for item in sublist]
    # Check that the number of records matches expected and all records
    # are present.
    self.assertEqual(len(parsed_records), num_records)
    self.assertEqual(set(parsed_records), set(lines))

  def test_read_text_lines_multifile(self):
    gfile.Glob = self._orig_glob
    filenames = self._create_sorted_temp_files(["ABC\n", "DEF\nGHK\n"])

    batch_size = 1
    queue_capacity = 5
    name = "my_batch"

    with tf.Graph().as_default() as g, self.test_session(graph=g) as session:
      inputs = tf.contrib.learn.io.read_batch_examples(
          filenames, batch_size, reader=tf.TextLineReader,
          randomize_input=False, num_epochs=1, queue_capacity=queue_capacity,
          name=name)
      self.assertAllEqual((None,), inputs.get_shape().as_list())
      session.run(tf.local_variables_initializer())

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(session, coord=coord)

      self.assertEqual("%s:1" % name, inputs.name)
      file_name_queue_name = "%s/file_name_queue" % name
      file_names_name = "%s/input" % file_name_queue_name
      example_queue_name = "%s/fifo_queue" % name
      test_util.assert_ops_in_graph({
          file_names_name: "Const",
          file_name_queue_name: "FIFOQueue",
          "%s/read/TextLineReader" % name: "TextLineReader",
          example_queue_name: "FIFOQueue",
          name: "QueueDequeueUpTo"
      }, g)

      self.assertAllEqual(session.run(inputs), [b"ABC"])
      self.assertAllEqual(session.run(inputs), [b"DEF"])
      self.assertAllEqual(session.run(inputs), [b"GHK"])
      with self.assertRaises(errors.OutOfRangeError):
        session.run(inputs)

      coord.request_stop()
      coord.join(threads)

  def test_read_text_lines_multifile_with_shared_queue(self):
    gfile.Glob = self._orig_glob
    filenames = self._create_sorted_temp_files(["ABC\n", "DEF\nGHK\n"])

    batch_size = 1
    queue_capacity = 5
    name = "my_batch"

    with tf.Graph().as_default() as g, self.test_session(graph=g) as session:
      keys, inputs = _read_keyed_batch_examples_shared_queue(
          filenames,
          batch_size,
          reader=tf.TextLineReader,
          randomize_input=False,
          num_epochs=1,
          queue_capacity=queue_capacity,
          name=name)
      self.assertAllEqual((None,), keys.get_shape().as_list())
      self.assertAllEqual((None,), inputs.get_shape().as_list())
      session.run(tf.local_variables_initializer())

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(session, coord=coord)

      self.assertEqual("%s:1" % name, inputs.name)
      shared_file_name_queue_name = "%s/file_name_queue" % name
      file_names_name = "%s/input" % shared_file_name_queue_name
      example_queue_name = "%s/fifo_queue" % name
      worker_file_name_queue_name = "%s/file_name_queue/fifo_queue" % name
      test_util.assert_ops_in_graph({
          file_names_name: "Const",
          shared_file_name_queue_name: "FIFOQueue",
          "%s/read/TextLineReader" % name: "TextLineReader",
          example_queue_name: "FIFOQueue",
          worker_file_name_queue_name: "FIFOQueue",
          name: "QueueDequeueUpTo"
      }, g)

      self.assertAllEqual(session.run(inputs), [b"ABC"])
      self.assertAllEqual(session.run(inputs), [b"DEF"])
      self.assertAllEqual(session.run(inputs), [b"GHK"])
      with self.assertRaises(errors.OutOfRangeError):
        session.run(inputs)

      coord.request_stop()
      coord.join(threads)

  def _get_qr(self, name):
    for qr in ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS):
      if qr.name == name:
        return qr

  def _run_queue(self, name, session):
    qr = self._get_qr(name)
    for op in qr.enqueue_ops:
      session.run(op)

  def test_multiple_workers_with_shared_queue(self):
    gfile.Glob = self._orig_glob
    filenames = self._create_sorted_temp_files([
        "ABC\n", "DEF\n", "GHI\n", "JKL\n", "MNO\n", "PQR\n", "STU\n", "VWX\n",
        "YZ\n"
    ])

    batch_size = 1
    queue_capacity = 5
    name = "my_batch"
    shared_file_name_queue_name = "%s/file_name_queue" % name
    example_queue_name = "%s/fifo_queue" % name
    worker_file_name_queue_name = "%s/file_name_queue/fifo_queue" % name

    server = tf.train.Server.create_local_server()

    with tf.Graph().as_default() as g1, tf.Session(
        server.target, graph=g1) as session:
      keys, inputs = _read_keyed_batch_examples_shared_queue(
          filenames,
          batch_size,
          reader=tf.TextLineReader,
          randomize_input=False,
          num_epochs=1,
          queue_capacity=queue_capacity,
          name=name)
      self.assertAllEqual((None,), keys.get_shape().as_list())
      self.assertAllEqual((None,), inputs.get_shape().as_list())
      session.run(tf.local_variables_initializer())

      # Run the three queues once manually.
      self._run_queue(shared_file_name_queue_name, session)
      self._run_queue(worker_file_name_queue_name, session)
      self._run_queue(example_queue_name, session)

      self.assertAllEqual(session.run(inputs), [b"ABC"])

      # Run the worker and the example queue.
      self._run_queue(worker_file_name_queue_name, session)
      self._run_queue(example_queue_name, session)

      self.assertAllEqual(session.run(inputs), [b"DEF"])

    with tf.Graph().as_default() as g2, tf.Session(
        server.target, graph=g2) as session:
      keys, inputs = _read_keyed_batch_examples_shared_queue(
          filenames,
          batch_size,
          reader=tf.TextLineReader,
          randomize_input=False,
          num_epochs=1,
          queue_capacity=queue_capacity,
          name=name)
      self.assertAllEqual((None,), keys.get_shape().as_list())
      self.assertAllEqual((None,), inputs.get_shape().as_list())

      # Run the worker and the example queue.
      self._run_queue(worker_file_name_queue_name, session)
      self._run_queue(example_queue_name, session)

      self.assertAllEqual(session.run(inputs), [b"GHI"])

    self.assertTrue(g1 is not g2)

  def test_batch_text_lines(self):
    gfile.Glob = self._orig_glob
    filename = self._create_temp_file("A\nB\nC\nD\nE\n")

    batch_size = 3
    queue_capacity = 10
    name = "my_batch"

    with tf.Graph().as_default() as g, self.test_session(graph=g) as session:
      inputs = tf.contrib.learn.io.read_batch_examples(
          [filename], batch_size, reader=tf.TextLineReader,
          randomize_input=False, num_epochs=1, queue_capacity=queue_capacity,
          read_batch_size=10, name=name)
      self.assertAllEqual((None,), inputs.get_shape().as_list())
      session.run(tf.local_variables_initializer())

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(session, coord=coord)

      self.assertAllEqual(session.run(inputs), [b"A", b"B", b"C"])
      self.assertAllEqual(session.run(inputs), [b"D", b"E"])
      with self.assertRaises(errors.OutOfRangeError):
        session.run(inputs)

      coord.request_stop()
      coord.join(threads)

  def test_keyed_read_text_lines(self):
    gfile.Glob = self._orig_glob
    filename = self._create_temp_file("ABC\nDEF\nGHK\n")

    batch_size = 1
    queue_capacity = 5
    name = "my_batch"

    with tf.Graph().as_default() as g, self.test_session(graph=g) as session:
      keys, inputs = tf.contrib.learn.io.read_keyed_batch_examples(
          filename, batch_size,
          reader=tf.TextLineReader, randomize_input=False,
          num_epochs=1, queue_capacity=queue_capacity, name=name)
      self.assertAllEqual((None,), keys.get_shape().as_list())
      self.assertAllEqual((None,), inputs.get_shape().as_list())
      session.run(tf.local_variables_initializer())

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(session, coord=coord)

      self.assertAllEqual(session.run([keys, inputs]),
                          [[filename.encode("utf-8") + b":1"], [b"ABC"]])
      self.assertAllEqual(session.run([keys, inputs]),
                          [[filename.encode("utf-8") + b":2"], [b"DEF"]])
      self.assertAllEqual(session.run([keys, inputs]),
                          [[filename.encode("utf-8") + b":3"], [b"GHK"]])
      with self.assertRaises(errors.OutOfRangeError):
        session.run(inputs)

      coord.request_stop()
      coord.join(threads)

  def test_keyed_parse_json(self):
    gfile.Glob = self._orig_glob
    filename = self._create_temp_file(
        '{"features": {"feature": {"age": {"int64_list": {"value": [0]}}}}}\n'
        '{"features": {"feature": {"age": {"int64_list": {"value": [1]}}}}}\n'
        '{"features": {"feature": {"age": {"int64_list": {"value": [2]}}}}}\n'
    )

    batch_size = 1
    queue_capacity = 5
    name = "my_batch"

    with tf.Graph().as_default() as g, self.test_session(graph=g) as session:
      dtypes = {"age": tf.FixedLenFeature([1], tf.int64)}
      parse_fn = lambda example: tf.parse_single_example(  # pylint: disable=g-long-lambda
          tf.decode_json_example(example), dtypes)
      keys, inputs = tf.contrib.learn.io.read_keyed_batch_examples(
          filename, batch_size,
          reader=tf.TextLineReader, randomize_input=False,
          num_epochs=1, queue_capacity=queue_capacity,
          parse_fn=parse_fn, name=name)
      self.assertAllEqual((None,), keys.get_shape().as_list())
      self.assertEqual(1, len(inputs))
      self.assertAllEqual((None, 1), inputs["age"].get_shape().as_list())
      session.run(tf.local_variables_initializer())

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(session, coord=coord)

      key, age = session.run([keys, inputs["age"]])
      self.assertAllEqual(age, [[0]])
      self.assertAllEqual(key, [filename.encode("utf-8") + b":1"])
      key, age = session.run([keys, inputs["age"]])
      self.assertAllEqual(age, [[1]])
      self.assertAllEqual(key, [filename.encode("utf-8") + b":2"])
      key, age = session.run([keys, inputs["age"]])
      self.assertAllEqual(age, [[2]])
      self.assertAllEqual(key, [filename.encode("utf-8") + b":3"])
      with self.assertRaises(errors.OutOfRangeError):
        session.run(inputs)

      coord.request_stop()
      coord.join(threads)


if __name__ == "__main__":
  tf.test.main()
