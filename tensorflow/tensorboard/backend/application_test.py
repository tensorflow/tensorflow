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
"""Integration tests for TensorBoard.

These tests start up a full-fledged TensorBoard server.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import gzip
import json
import numbers
import os
import shutil
import tempfile
import threading

from six import BytesIO
from six.moves import http_client
from six.moves import xrange  # pylint: disable=redefined-builtin

from werkzeug import serving
from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.summary import event_multiplexer
from tensorflow.python.summary.writer import writer as writer_lib
from tensorflow.tensorboard.backend import application


class TensorboardServerTest(test.TestCase):
  _only_use_meta_graph = False  # Server data contains only a GraphDef

  # Number of scalar-containing events to make.
  _SCALAR_COUNT = 99

  def setUp(self):
    self.temp_dir = self._GenerateTestData()
    multiplexer = event_multiplexer.EventMultiplexer(
        size_guidance=application.DEFAULT_SIZE_GUIDANCE,
        purge_orphaned_data=True)
    plugins = {}
    app = application.TensorBoardWSGIApp(
        self.temp_dir, plugins, multiplexer, reload_interval=0)
    self._server = serving.BaseWSGIServer('localhost', 0, app)
    # 0 to pick an unused port.
    self._server_thread = threading.Thread(target=self._server.serve_forever)
    self._server_thread.daemon = True
    self._server_thread.start()
    self._connection = http_client.HTTPConnection(
        'localhost', self._server.server_address[1])

  def tearDown(self):
    self._connection.close()
    self._server.shutdown()
    self._server.server_close()

  def _get(self, path, headers=None):
    """Perform a GET request for the given path."""
    if headers is None:
      headers = {}
    self._connection.request('GET', path, None, headers)
    return self._connection.getresponse()

  def _getJson(self, path):
    """Perform a GET request and decode the result as JSON."""
    self._connection.request('GET', path)
    response = self._connection.getresponse()
    self.assertEqual(response.status, 200)
    data = response.read()
    if response.getheader('Content-Encoding') == 'gzip':
      data = gzip.GzipFile('', 'rb', 9, BytesIO(data)).read()
    return json.loads(data.decode('utf-8'))

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
    self.assertEqual(response.status, 400)

  def testLogdir(self):
    """Test the format of the data/logdir endpoint."""
    parsed_object = self._getJson('/data/logdir')
    self.assertEqual(parsed_object, {'logdir': self.temp_dir})

  def testRuns(self):
    """Test the format of the /data/runs endpoint."""
    run_json = self._getJson('/data/runs')

    # Don't check the actual timestamp since it's time-dependent.
    self.assertTrue(
        isinstance(run_json['run1']['firstEventTimestamp'], numbers.Number))
    del run_json['run1']['firstEventTimestamp']
    self.assertEqual(
        run_json,
        {
            'run1': {
                'compressedHistograms': ['histogram'],
                'scalars': ['simple_values'],
                'histograms': ['histogram'],
                'images': ['image'],
                'audio': ['audio'],
                # if only_use_meta_graph, the graph is from the metagraph
                'graph': True,
                'meta_graph': self._only_use_meta_graph,
                'run_metadata': ['test run']
            }
        })

  def testApplicationPaths_getCached(self):
    """Test the format of the /data/runs endpoint."""
    for path in ('/',):  # TODO(jart): '/app.js' in open source
      connection = http_client.HTTPConnection('localhost',
                                              self._server.server_address[1])
      connection.request('GET', path)
      response = connection.getresponse()
      self.assertEqual(response.status, 200, msg=path)
      self.assertEqual(
          response.getheader('Cache-Control'),
          'private, max-age=3600',
          msg=path)
      connection.close()

  def testDataPaths_disableAllCaching(self):
    """Test the format of the /data/runs endpoint."""
    for path in ('/data/runs', '/data/logdir',
                 '/data/scalars?run=run1&tag=simple_values',
                 '/data/scalars?run=run1&tag=simple_values&format=csv',
                 '/data/images?run=run1&tag=image',
                 '/data/individualImage?run=run1&tag=image&index=0',
                 '/data/audio?run=run1&tag=audio',
                 '/data/run_metadata?run=run1&tag=test%20run'):
      connection = http_client.HTTPConnection('localhost',
                                              self._server.server_address[1])
      connection.request('GET', path)
      response = connection.getresponse()
      self.assertEqual(response.status, 200, msg=path)
      self.assertEqual(response.getheader('Expires'), '0', msg=path)
      response.read()
      connection.close()

  def testHistograms(self):
    """Test the format of /data/histograms."""
    self.assertEqual(
        self._getJson('/data/histograms?tag=histogram&run=run1'),
        [[0, 0, [0, 2.0, 3.0, 6.0, 5.0, [0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]]])

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

  def testAudio(self):
    """Test listing audio and retrieving an individual audio clip."""
    audio_json = self._getJson('/data/audio?tag=audio&run=run1')
    audio_query = audio_json[0]['query']
    # We don't care about the format of the audio query.
    del audio_json[0]['query']
    self.assertEqual(audio_json, [{
        'wall_time': 0,
        'step': 0,
        'content_type': 'audio/wav'
    }])
    response = self._get('/data/individualAudio?%s' % audio_query)
    self.assertEqual(response.status, 200)

  def testGraph(self):
    """Test retrieving the graph definition."""
    response = self._get('/data/graph?run=run1&limit_attr_size=1024'
                         '&large_attrs_key=_very_large_attrs')
    self.assertEqual(response.status, 200)
    graph_pbtxt = response.read()
    # Parse the graph from pbtxt into a graph message.
    graph = graph_pb2.GraphDef()
    graph = text_format.Parse(graph_pbtxt, graph)
    self.assertEqual(len(graph.node), 2)
    self.assertEqual(graph.node[0].name, 'a')
    self.assertEqual(graph.node[1].name, 'b')
    # Make sure the second node has an attribute that was filtered out because
    # it was too large and was added to the "too large" attributes list.
    self.assertEqual(list(graph.node[1].attr.keys()), ['_very_large_attrs'])
    self.assertEqual(graph.node[1].attr['_very_large_attrs'].list.s,
                     [b'very_large_attr'])

  def testAcceptGzip_compressesResponse(self):
    response = self._get('/data/graph?run=run1&limit_attr_size=1024'
                         '&large_attrs_key=_very_large_attrs',
                         {'Accept-Encoding': 'gzip'})
    self.assertEqual(response.status, 200)
    self.assertEqual(response.getheader('Content-Encoding'), 'gzip')
    pbtxt = gzip.GzipFile('', 'rb', 9, BytesIO(response.read())).read()
    graph = text_format.Parse(pbtxt, graph_pb2.GraphDef())
    self.assertEqual(len(graph.node), 2)

  def testAcceptAnyEncoding_compressesResponse(self):
    response = self._get('/data/graph?run=run1&limit_attr_size=1024'
                         '&large_attrs_key=_very_large_attrs',
                         {'Accept-Encoding': '*'})
    self.assertEqual(response.status, 200)
    self.assertEqual(response.getheader('Content-Encoding'), 'gzip')
    pbtxt = gzip.GzipFile('', 'rb', 9, BytesIO(response.read())).read()
    graph = text_format.Parse(pbtxt, graph_pb2.GraphDef())
    self.assertEqual(len(graph.node), 2)

  def testAcceptDoodleEncoding_doesNotCompressResponse(self):
    response = self._get('/data/graph?run=run1&limit_attr_size=1024'
                         '&large_attrs_key=_very_large_attrs',
                         {'Accept-Encoding': 'doodle'})
    self.assertEqual(response.status, 200)
    self.assertIsNone(response.getheader('Content-Encoding'))
    graph = text_format.Parse(response.read(), graph_pb2.GraphDef())
    self.assertEqual(len(graph.node), 2)

  def testAcceptGzip_doesNotCompressImage(self):
    response = self._get('/data/individualImage?run=run1&tag=image&index=0',
                         {'Accept-Encoding': 'gzip'})
    self.assertEqual(response.status, 200)
    self.assertEqual(response.getheader('Content-Encoding'), None)

  def testRunMetadata(self):
    """Test retrieving the run metadata information."""
    response = self._get('/data/run_metadata?run=run1&tag=test%20run')
    self.assertEqual(response.status, 200)
    run_metadata_pbtxt = response.read()
    # Parse from pbtxt into a message.
    run_metadata = config_pb2.RunMetadata()
    text_format.Parse(run_metadata_pbtxt, run_metadata)
    self.assertEqual(len(run_metadata.step_stats.dev_stats), 1)
    self.assertEqual(run_metadata.step_stats.dev_stats[0].device, 'test device')

  def _GenerateTestData(self):
    """Generates the test data directory.

    The test data has a single run named run1 which contains:
     - a histogram
     - an image at timestamp and step 0
     - scalar events containing the value i at step 10 * i and wall time
         100 * i, for i in [1, _SCALAR_COUNT).
     - a graph definition

    Returns:
      temp_dir: The directory the test data is generated under.
    """
    temp_dir = tempfile.mkdtemp(prefix=self.get_temp_dir())
    self.addCleanup(shutil.rmtree, temp_dir)
    run1_path = os.path.join(temp_dir, 'run1')
    os.makedirs(run1_path)
    writer = writer_lib.FileWriter(run1_path)

    histogram_value = summary_pb2.HistogramProto(
        min=0,
        max=2,
        num=3,
        sum=6,
        sum_squares=5,
        bucket_limit=[0, 1, 2],
        bucket=[1, 1, 1])
    # Add a simple graph event.
    graph_def = graph_pb2.GraphDef()
    node1 = graph_def.node.add()
    node1.name = 'a'
    node2 = graph_def.node.add()
    node2.name = 'b'
    node2.attr['very_large_attr'].s = b'a' * 2048  # 2 KB attribute

    meta_graph_def = meta_graph_pb2.MetaGraphDef(graph_def=graph_def)

    if self._only_use_meta_graph:
      writer.add_meta_graph(meta_graph_def)
    else:
      writer.add_graph(graph_def)

    # Add a simple run metadata event.
    run_metadata = config_pb2.RunMetadata()
    device_stats = run_metadata.step_stats.dev_stats.add()
    device_stats.device = 'test device'
    writer.add_run_metadata(run_metadata, 'test run')

    # 1x1 transparent GIF.
    encoded_image = base64.b64decode(
        'R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7')
    image_value = summary_pb2.Summary.Image(
        height=1, width=1, colorspace=1, encoded_image_string=encoded_image)

    audio_value = summary_pb2.Summary.Audio(
        sample_rate=44100,
        length_frames=22050,
        num_channels=2,
        encoded_audio_string=b'',
        content_type='audio/wav')
    writer.add_event(
        event_pb2.Event(
            wall_time=0,
            step=0,
            summary=summary_pb2.Summary(value=[
                summary_pb2.Summary.Value(
                    tag='histogram', histo=histogram_value),
                summary_pb2.Summary.Value(
                    tag='image', image=image_value), summary_pb2.Summary.Value(
                        tag='audio', audio=audio_value)
            ])))

    # Write 100 simple values.
    for i in xrange(1, self._SCALAR_COUNT + 1):
      writer.add_event(
          event_pb2.Event(
              # We use different values for wall time, step, and the value so we
              # can tell them apart.
              wall_time=100 * i,
              step=10 * i,
              summary=summary_pb2.Summary(value=[
                  summary_pb2.Summary.Value(
                      tag='simple_values', simple_value=i)
              ])))
    writer.flush()
    writer.close()

    return temp_dir


class TensorboardServerUsingMetagraphOnlyTest(TensorboardServerTest):
  # Tests new ability to use only the MetaGraphDef
  _only_use_meta_graph = True  # Server data contains only a MetaGraphDef


class ParseEventFilesSpecTest(test.TestCase):

  def testRunName(self):
    logdir = 'lol:/cat'
    expected = {'/cat': 'lol'}
    self.assertEqual(application.parse_event_files_spec(logdir), expected)

  def testPathWithColonThatComesAfterASlash_isNotConsideredARunName(self):
    logdir = '/lol:/cat'
    expected = {'/lol:/cat': None}
    self.assertEqual(application.parse_event_files_spec(logdir), expected)

  def testMultipleDirectories(self):
    logdir = '/a,/b'
    expected = {'/a': None, '/b': None}
    self.assertEqual(application.parse_event_files_spec(logdir), expected)

  def testNormalizesPaths(self):
    logdir = '/lol/.//cat/../cat'
    expected = {'/lol/cat': None}
    self.assertEqual(application.parse_event_files_spec(logdir), expected)

  def testAbsolutifies(self):
    logdir = 'lol/cat'
    expected = {os.path.realpath('lol/cat'): None}
    self.assertEqual(application.parse_event_files_spec(logdir), expected)

  def testRespectsGCSPath(self):
    logdir = 'gs://foo/path'
    expected = {'gs://foo/path': None}
    self.assertEqual(application.parse_event_files_spec(logdir), expected)

  def testRespectsHDFSPath(self):
    logdir = 'hdfs://foo/path'
    expected = {'hdfs://foo/path': None}
    self.assertEqual(application.parse_event_files_spec(logdir), expected)

  def testDoesNotExpandUserInGCSPath(self):
    logdir = 'gs://~/foo/path'
    expected = {'gs://~/foo/path': None}
    self.assertEqual(application.parse_event_files_spec(logdir), expected)

  def testDoesNotNormalizeGCSPath(self):
    logdir = 'gs://foo/./path//..'
    expected = {'gs://foo/./path//..': None}
    self.assertEqual(application.parse_event_files_spec(logdir), expected)

  def testRunNameWithGCSPath(self):
    logdir = 'lol:gs://foo/path'
    expected = {'gs://foo/path': 'lol'}
    self.assertEqual(application.parse_event_files_spec(logdir), expected)


class TensorBoardAssetsTest(test.TestCase):

  def testTagFound(self):
    tag = resource_loader.load_resource('tensorboard/TAG')
    self.assertTrue(tag)


if __name__ == '__main__':
  test.main()
