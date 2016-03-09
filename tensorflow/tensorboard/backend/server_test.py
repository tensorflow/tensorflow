# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Integration tests for TensorBoard.

These tests start up a full-fledged TensorBoard server.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import gzip
import json
import os
import shutil
import threading
import zlib

from six import BytesIO
from six.moves import http_client
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.summary import event_multiplexer
from tensorflow.tensorboard.backend import server


class TensorboardServerTest(tf.test.TestCase):

  # Number of scalar-containing events to make.
  _SCALAR_COUNT = 99

  def setUp(self):
    self._GenerateTestData()
    self._multiplexer = event_multiplexer.EventMultiplexer(
        size_guidance=server.TENSORBOARD_SIZE_GUIDANCE)
    server.ReloadMultiplexer(self._multiplexer, {self.get_temp_dir(): None})
    # 0 to pick an unused port.
    self._server = server.BuildServer(self._multiplexer, 'localhost', 0)
    self._server_thread = threading.Thread(target=self._server.serve_forever)
    self._server_thread.daemon = True
    self._server_thread.start()
    self._connection = http_client.HTTPConnection(
        'localhost', self._server.server_address[1])

  def tearDown(self):
    self._connection.close()
    self._server.shutdown()
    self._server.server_close()

  def _get(self, path):
    """Perform a GET request for the given path."""
    self._connection.request('GET', path)
    return self._connection.getresponse()

  def _getJson(self, path):
    """Perform a GET request and decode the result as JSON."""
    self._connection.request('GET', path)
    response = self._connection.getresponse()
    self.assertEqual(response.status, 200)
    return json.loads(response.read().decode('utf-8'))

  def _decodeResponse(self, response):
    """Decompresses (if necessary) the response from the server."""
    encoding = response.getheader('Content-Encoding')
    content = response.read()
    if encoding in ('gzip', 'x-gzip', 'deflate'):
      if encoding == 'deflate':
        data = BytesIO(zlib.decompress(content))
      else:
        data = gzip.GzipFile('', 'rb', 9, BytesIO(content))
      content = data.read()
    return content

  def testBasicStartup(self):
    """Start the server up and then shut it down immediately."""
    pass

  def testRequestMainPage(self):
    """Navigate to the main page and verify that it returns a 200."""
    response = self._get('/')
    self.assertEqual(response.status, 200)

  def testRequestNonexistentPage(self):
    """Request a page that doesn't exist; it should 404."""
    response = self._get('/asdf')
    self.assertEqual(response.status, 404)

  def testDirectoryTraversal(self):
    """Attempt a directory traversal attack."""
    response = self._get('/..' * 30 + '/etc/passwd')
    self.assertEqual(response.status, 404)

  def testRuns(self):
    """Test the format of the /data/runs endpoint."""
    self.assertEqual(
        self._getJson('/data/runs'),
        {'run1': {'compressedHistograms': ['histogram'],
                  'scalars': ['simple_values'],
                  'histograms': ['histogram'],
                  'images': ['image'],
                  'graph': True}})

  def testHistograms(self):
    """Test the format of /data/histograms."""
    self.assertEqual(
        self._getJson('/data/histograms?tag=histogram&run=run1'),
        [[0, 0, [0, 2.0, 3.0, 6.0, 5.0, [0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]]])

  def testSampleScalars(self):
    """Test the sample_count parameter of /data/scalars."""
    for i in xrange(10, self._SCALAR_COUNT, 10):
      samples = self._getJson('/data/scalars?sample_count=%d' % i)
      values = samples['run1']['simple_values']
      # Verify that we got the right amount of values and that we got the
      # endpoints.
      self.assertEqual(len(values), i)
      self.assertEqual(values[0], [100, 10, 1])
      self.assertEqual(values[-1], [9900, 990, 99])

  def testSampleScalarsWithLargeSampleCount(self):
    """Test using a large sample_count."""
    samples = self._getJson('/data/scalars?sample_count=999999')
    values = samples['run1']['simple_values']
    self.assertEqual(len(values), self._SCALAR_COUNT)

  def testImages(self):
    """Test listing images and retrieving an individual image."""
    image_json = self._getJson('/data/images?tag=image&run=run1')
    image_query = image_json[0]['query']
    # We don't care about the format of the image query.
    del image_json[0]['query']
    self.assertEqual(image_json, [{
        'wall_time': 0,
        'step': 0,
        'height': 1,
        'width': 1
    }])
    response = self._get('/data/individualImage?%s' % image_query)
    self.assertEqual(response.status, 200)

  def testGraph(self):
    """Test retrieving the graph definition."""
    response = self._get('/data/graph?run=run1&limit_attr_size=1024'
                         '&large_attrs_key=_very_large_attrs')
    self.assertEqual(response.status, 200)
    # Decompress (unzip) the response, since graphs come gzipped.
    graph_pbtxt = self._decodeResponse(response)
    # Parse the graph from pbtxt into a graph message.
    graph = tf.GraphDef()
    graph = text_format.Parse(graph_pbtxt, graph)
    self.assertEqual(len(graph.node), 2)
    self.assertEqual(graph.node[0].name, 'a')
    self.assertEqual(graph.node[1].name, 'b')
    # Make sure the second node has an attribute that was filtered out because
    # it was too large and was added to the "too large" attributes list.
    self.assertEqual(graph.node[1].attr.keys(), ['_very_large_attrs'])
    self.assertEqual(graph.node[1].attr['_very_large_attrs'].list.s,
                     ['very_large_attr'])

  def _GenerateTestData(self):
    """Generates the test data directory.

    The test data has a single run named run1 which contains:
     - a histogram
     - an image at timestamp and step 0
     - scalar events containing the value i at step 10 * i and wall time
         100 * i, for i in [1, _SCALAR_COUNT).
     - a graph definition
    """
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    run1_path = os.path.join(temp_dir, 'run1')
    os.makedirs(run1_path)
    writer = tf.train.SummaryWriter(run1_path)

    histogram_value = tf.HistogramProto(min=0,
                                        max=2,
                                        num=3,
                                        sum=6,
                                        sum_squares=5,
                                        bucket_limit=[0, 1, 2],
                                        bucket=[1, 1, 1])
    # Add a simple graph event.
    graph_def = tf.GraphDef()
    node1 = graph_def.node.add()
    node1.name = 'a'
    node2 = graph_def.node.add()
    node2.name = 'b'
    node2.attr['very_large_attr'].s = b'a' * 2048  # 2 KB attribute
    writer.add_event(tf.Event(graph_def=graph_def.SerializeToString()))

    # 1x1 transparent GIF.
    encoded_image = base64.b64decode(
        'R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7')
    image_value = tf.Summary.Image(height=1,
                                   width=1,
                                   colorspace=1,
                                   encoded_image_string=encoded_image)
    writer.add_event(tf.Event(wall_time=0,
                              step=0,
                              summary=tf.Summary(value=[tf.Summary.Value(
                                  tag='histogram',
                                  histo=histogram_value), tf.Summary.Value(
                                      tag='image',
                                      image=image_value)])))

    # Write 100 simple values.
    for i in xrange(1, self._SCALAR_COUNT + 1):
      writer.add_event(tf.Event(
          # We use different values for wall time, step, and the value so we can
          # tell them apart.
          wall_time=100 * i,
          step=10 * i,
          summary=tf.Summary(value=[tf.Summary.Value(tag='simple_values',
                                                     simple_value=i)])))
    writer.flush()
    writer.close()


class ParseEventFilesSpecTest(tf.test.TestCase):

  def testRespectsGCSPath(self):
    logdir_string = 'gs://foo/path'
    expected = {'gs://foo/path': None}
    self.assertEqual(server.ParseEventFilesSpec(logdir_string), expected)


if __name__ == '__main__':
  tf.test.main()
