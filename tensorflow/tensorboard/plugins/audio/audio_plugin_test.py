# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests the Tensorboard audio plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import shutil
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from werkzeug import test as werkzeug_test
from werkzeug import wrappers

from tensorflow.tensorboard.backend import application
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
from tensorflow.tensorboard.plugins.audio import audio_plugin


class AudioPluginTest(tf.test.TestCase):

  def setUp(self):
    self.log_dir = tempfile.mkdtemp()

    # We use numpy.random to generate audio. We seed to avoid non-determinism
    # in this test.
    numpy.random.seed(42)

    # Create audio summaries for run foo.
    tf.reset_default_graph()
    sess = tf.Session()
    placeholder = tf.placeholder(tf.float32)
    tf.summary.audio(name="baz", tensor=placeholder, sample_rate=44100)
    merged_summary_op = tf.summary.merge_all()
    foo_directory = os.path.join(self.log_dir, "foo")
    writer = tf.summary.FileWriter(foo_directory)
    writer.add_graph(sess.graph)
    for step in xrange(2):
      # The floats (sample data) range from -1 to 1.
      writer.add_summary(sess.run(merged_summary_op, feed_dict={
          placeholder: numpy.random.rand(42, 22050) * 2 - 1
      }), global_step=step)
    writer.close()

    # Create audio summaries for run bar.
    tf.reset_default_graph()
    sess = tf.Session()
    placeholder = tf.placeholder(tf.float32)
    tf.summary.audio(name="quux", tensor=placeholder, sample_rate=44100)
    merged_summary_op = tf.summary.merge_all()
    bar_directory = os.path.join(self.log_dir, "bar")
    writer = tf.summary.FileWriter(bar_directory)
    writer.add_graph(sess.graph)
    for step in xrange(2):
      # The floats (sample data) range from -1 to 1.
      writer.add_summary(sess.run(merged_summary_op, feed_dict={
          placeholder: numpy.random.rand(42, 11025) * 2 - 1
      }), global_step=step)
    writer.close()

    # Start a server with the plugin.
    multiplexer = event_multiplexer.EventMultiplexer({
        "foo": foo_directory,
        "bar": bar_directory,
    })
    plugin = audio_plugin.AudioPlugin()
    wsgi_app = application.TensorBoardWSGIApp(
        self.log_dir, [plugin], multiplexer, reload_interval=0)
    self.server = werkzeug_test.Client(wsgi_app, wrappers.BaseResponse)
    self.routes = plugin.get_plugin_apps(multiplexer, self.log_dir)

  def tearDown(self):
    shutil.rmtree(self.log_dir, ignore_errors=True)

  def _DeserializeResponse(self, byte_content):
    """Deserializes byte content that is a JSON encoding.

    Args:
      byte_content: The byte content of a response.

    Returns:
      The deserialized python object decoded from JSON.
    """
    return json.loads(byte_content.decode("utf-8"))

  def testRoutesProvided(self):
    """Tests that the plugin offers the correct routes."""
    self.assertIsInstance(self.routes["/audio"], collections.Callable)
    self.assertIsInstance(self.routes["/individualAudio"], collections.Callable)
    self.assertIsInstance(self.routes["/tags"], collections.Callable)

  def testAudioRoute(self):
    """Tests that the /audio routes returns with the correct data."""
    response = self.server.get(
        "/data/plugin/audio/audio?run=foo&tag=baz/audio/0")
    self.assertEqual(200, response.status_code)

    # Verify that the correct entries are returned.
    entries = self._DeserializeResponse(response.get_data())
    self.assertEqual(2, len(entries))

    # Verify that the 1st entry is correct.
    entry = entries[0]
    self.assertEqual(0, entry["step"])
    parsed_query = urllib.parse.parse_qs(entry["query"])
    self.assertListEqual(["0"], parsed_query["index"])
    self.assertListEqual(["foo"], parsed_query["run"])
    self.assertListEqual(["baz/audio/0"], parsed_query["tag"])

    # Verify that the 2nd entry is correct.
    entry = entries[1]
    self.assertEqual(1, entry["step"])
    parsed_query = urllib.parse.parse_qs(entry["query"])
    self.assertListEqual(["1"], parsed_query["index"])
    self.assertListEqual(["foo"], parsed_query["run"])
    self.assertListEqual(["baz/audio/0"], parsed_query["tag"])

  def testIndividualAudioRoute(self):
    """Tests fetching an individual audio."""
    response = self.server.get(
        "/data/plugin/audio/individualAudio?run=bar&tag=quux/audio/0&index=0")
    self.assertEqual(200, response.status_code)
    self.assertEqual("audio/wav", response.headers.get("content-type"))

  def testRunsRoute(self):
    """Tests that the /runs route offers the correct run to tag mapping."""
    response = self.server.get("/data/plugin/audio/tags")
    self.assertEqual(200, response.status_code)
    run_to_tags = self._DeserializeResponse(response.get_data())
    self.assertItemsEqual(("foo", "bar"), run_to_tags.keys())
    self.assertItemsEqual(
        ["baz/audio/0", "baz/audio/1", "baz/audio/2"], run_to_tags["foo"])
    self.assertItemsEqual(
        ["quux/audio/0", "quux/audio/1", "quux/audio/2"], run_to_tags["bar"])


if __name__ == "__main__":
  tf.test.main()
