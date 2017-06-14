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

import gzip
import json
import os
import shutil
import socket
import tempfile
import threading

from six import BytesIO
from six.moves import http_client
import tensorflow as tf

from werkzeug import serving

from tensorflow.tensorboard import main as tensorboard
from tensorflow.tensorboard.backend import application
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
from tensorflow.tensorboard.plugins import base_plugin


class FakePlugin(base_plugin.TBPlugin):
  """A plugin with no functionality."""

  def __init__(self, plugin_name, is_active_value, routes_mapping):
    """Constructs a fake plugin.

    Args:
      plugin_name: The name of this plugin.
      is_active_value: Whether the plugin is active.
      routes_mapping: A dictionary mapping from route (string URL path) to the
        method called when a user issues a request to that route.
    """
    self.plugin_name = plugin_name
    self._is_active_value = is_active_value
    self._routes_mapping = routes_mapping

  def get_plugin_apps(self, multiplexer, logdir):
    """Returns a mapping from routes to handlers offered by this plugin.

    Args:
      multiplexer: The event multiplexer.
      logdir: The path to the directory containing logs.

    Returns:
      A dictionary mapping from routes to handlers offered by this plugin.
    """
    return self._routes_mapping

  def is_active(self):
    """Returns whether this plugin is active.

    Returns:
      A boolean. Whether this plugin is active.
    """
    return self._is_active_value


class TensorboardServerTest(tf.test.TestCase):
  _only_use_meta_graph = False  # Server data contains only a GraphDef

  def setUp(self):
    self.logdir = self.get_temp_dir()

    self._GenerateTestData(run_name='run1')
    self._multiplexer = event_multiplexer.EventMultiplexer(
        size_guidance=application.DEFAULT_SIZE_GUIDANCE,
        purge_orphaned_data=True)
    plugins = [
        FakePlugin(plugin_name='foo', is_active_value=True, routes_mapping={}),
        FakePlugin(plugin_name='bar', is_active_value=False, routes_mapping={})
    ]
    app = application.TensorBoardWSGIApp(
        self.logdir, plugins, self._multiplexer, reload_interval=0)
    try:
      self._server = serving.BaseWSGIServer('localhost', 0, app)
      # 0 to pick an unused port.
    except IOError:
      # BaseWSGIServer has a preference for IPv4. If that didn't work, try again
      # with an explicit IPv6 address.
      self._server = serving.BaseWSGIServer('::1', 0, app)
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

  def testLogdir(self):
    """Test the format of the data/logdir endpoint."""
    parsed_object = self._getJson('/data/logdir')
    self.assertEqual(parsed_object, {'logdir': self.logdir})

  def testPluginsListing(self):
    """Test the format of the data/plugins_listing endpoint."""
    parsed_object = self._getJson('/data/plugins_listing')
    # Plugin foo is active. Plugin bar is not.
    self.assertEqual(parsed_object, {'foo': True, 'bar': False})

  def testRuns(self):
    """Test the format of the /data/runs endpoint."""
    run_json = self._getJson('/data/runs')
    self.assertEqual(run_json, ['run1'])

  def testRunsAppendOnly(self):
    """Test that new runs appear after old ones in /data/runs."""
    # We use three runs: the 'run1' that we already created in our
    # `setUp` method, plus runs with names lexicographically before and
    # after it (so that just sorting by name doesn't have a chance of
    # working).
    fake_wall_times = {
        'run1': 1234.0,
        'avocado': 2345.0,
        'zebra': 3456.0,
        'mysterious': None,
    }

    stubs = tf.test.StubOutForTesting()
    # pylint: disable=invalid-name
    def FirstEventTimestamp_stub(multiplexer_self, run_name):
      del multiplexer_self
      matches = [candidate_name
                 for candidate_name in fake_wall_times
                 if run_name.endswith(candidate_name)]
      self.assertEqual(len(matches), 1, '%s (%s)' % (matches, run_name))
      wall_time = fake_wall_times[matches[0]]
      if wall_time is None:
        raise ValueError('No event timestamp could be found')
      else:
        return wall_time
    # pylint: enable=invalid-name

    stubs.SmartSet(self._multiplexer,
                   'FirstEventTimestamp',
                   FirstEventTimestamp_stub)

    def add_run(run_name):
      self._GenerateTestData(run_name)
      self._multiplexer.AddRunsFromDirectory(self.logdir)
      self._multiplexer.Reload()

    # Add one run: it should come last.
    add_run('avocado')
    self.assertEqual(self._getJson('/data/runs'),
                     ['run1', 'avocado'])

    # Add another run: it should come last, too.
    add_run('zebra')
    self.assertEqual(self._getJson('/data/runs'),
                     ['run1', 'avocado', 'zebra'])

    # And maybe there's a run for which we somehow have no timestamp.
    add_run('mysterious')
    self.assertEqual(self._getJson('/data/runs'),
                     ['run1', 'avocado', 'zebra', 'mysterious'])

    stubs.UnsetAll()

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
    for path in ('/data/runs', '/data/logdir'):
      connection = http_client.HTTPConnection('localhost',
                                              self._server.server_address[1])
      connection.request('GET', path)
      response = connection.getresponse()
      self.assertEqual(response.status, 200, msg=path)
      self.assertEqual(response.getheader('Expires'), '0', msg=path)
      response.read()
      connection.close()

  def _GenerateTestData(self, run_name):
    """Generates the test data directory.

    The test data has a single run of the given name, containing:
      - a graph definition and metagraph definition

    Arguments:
      run_name: the directory under self.logdir into which to write
        events
    """
    run_path = os.path.join(self.logdir, run_name)
    os.makedirs(run_path)

    writer = tf.summary.FileWriter(run_path)

    # Add a simple graph event.
    graph_def = tf.GraphDef()
    node1 = graph_def.node.add()
    node1.name = 'a'
    node2 = graph_def.node.add()
    node2.name = 'b'
    node2.attr['very_large_attr'].s = b'a' * 2048  # 2 KB attribute

    meta_graph_def = tf.MetaGraphDef(graph_def=graph_def)

    if self._only_use_meta_graph:
      writer.add_meta_graph(meta_graph_def)
    else:
      writer.add_graph(graph_def)

    writer.flush()
    writer.close()


class TensorboardServerPluginNameTest(tf.test.TestCase):

  def _test(self, name, should_be_okay):
    temp_dir = tempfile.mkdtemp(prefix=self.get_temp_dir())
    self.addCleanup(shutil.rmtree, temp_dir)
    multiplexer = event_multiplexer.EventMultiplexer(
        size_guidance=application.DEFAULT_SIZE_GUIDANCE,
        purge_orphaned_data=True)
    plugins = [
        FakePlugin(plugin_name='foo', is_active_value=True, routes_mapping={}),
        FakePlugin(plugin_name=name, is_active_value=True, routes_mapping={}),
        FakePlugin(plugin_name='bar', is_active_value=False, routes_mapping={})
    ]
    if should_be_okay:
      application.TensorBoardWSGIApp(
          temp_dir, plugins, multiplexer, reload_interval=0)
    else:
      with self.assertRaisesRegexp(ValueError, r'invalid name'):
        application.TensorBoardWSGIApp(
            temp_dir, plugins, multiplexer, reload_interval=0)

  def testEmptyName(self):
    self._test('', False)

  def testNameWithSlashes(self):
    self._test('scalars/data', False)

  def testNameWithSpaces(self):
    self._test('my favorite plugin', False)

  def testSimpleName(self):
    self._test('scalars', True)

  def testComprehensiveName(self):
    self._test('Scalar-Dashboard_3000.1', True)


class TensorboardServerPluginRouteTest(tf.test.TestCase):

  def _test(self, route, should_be_okay):
    temp_dir = tempfile.mkdtemp(prefix=self.get_temp_dir())
    self.addCleanup(shutil.rmtree, temp_dir)
    multiplexer = event_multiplexer.EventMultiplexer(
        size_guidance=application.DEFAULT_SIZE_GUIDANCE,
        purge_orphaned_data=True)
    plugins = [
        FakePlugin(
            plugin_name='foo',
            is_active_value=True,
            routes_mapping={route: lambda environ, start_response: None}),
    ]
    if should_be_okay:
      application.TensorBoardWSGIApp(
          temp_dir, plugins, multiplexer, reload_interval=0)
    else:
      with self.assertRaisesRegexp(ValueError, r'invalid route'):
        application.TensorBoardWSGIApp(
            temp_dir, plugins, multiplexer, reload_interval=0)

  def testNormalRoute(self):
    self._test('/runs', True)

  def testEmptyRoute(self):
    self._test('', False)

  def testSlashlessRoute(self):
    self._test('runaway', False)


class TensorboardServerUsingMetagraphOnlyTest(TensorboardServerTest):
  # Tests new ability to use only the MetaGraphDef
  _only_use_meta_graph = True  # Server data contains only a MetaGraphDef


class ParseEventFilesSpecTest(tf.test.TestCase):

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


class TensorBoardAssetsTest(tf.test.TestCase):

  def testTagFound(self):
    tag = application.get_tensorboard_tag()
    self.assertTrue(tag)
    app = application.standard_tensorboard_wsgi('', True, 60, [])
    self.assertEqual(app.tag, tag)


class TensorBoardPluginsTest(tf.test.TestCase):

  def testPluginsAdded(self):

    def foo_handler():
      pass

    def bar_handler():
      pass

    plugins = [
        FakePlugin(
            plugin_name='foo',
            is_active_value=True,
            routes_mapping={'/foo_route': foo_handler}),
        FakePlugin(
            plugin_name='bar',
            is_active_value=True,
            routes_mapping={'/bar_route': bar_handler}),
    ]

    # The application should have added routes for both plugins.
    app = application.standard_tensorboard_wsgi('', True, 60, plugins)

    # The routes are prefixed with /data/plugin/[plugin name].
    self.assertDictContainsSubset({
        '/data/plugin/foo/foo_route': foo_handler,
        '/data/plugin/bar/bar_route': bar_handler,
    }, app.data_applications)


class TensorboardSimpleServerConstructionTest(tf.test.TestCase):
  """Tests that the default HTTP server is constructed without error.

  Mostly useful for IPv4/IPv6 testing. This test should run with only IPv4, only
  IPv6, and both IPv4 and IPv6 enabled.
  """

  class _StubApplication(object):
    tag = ''

  def testMakeServerBlankHost(self):
    # Test that we can bind to all interfaces without throwing an error
    server, url = tensorboard.make_simple_server(
        self._StubApplication(),
        host='',
        port=0)  # Grab any available port
    self.assertTrue(server)
    self.assertTrue(url)

  def testSpecifiedHost(self):
    one_passed = False
    try:
      _, url = tensorboard.make_simple_server(
          self._StubApplication(),
          host='127.0.0.1',
          port=0)
      self.assertStartsWith(actual=url, expected_start='http://127.0.0.1:')
      one_passed = True
    except socket.error:
      # IPv4 is not supported
      pass
    try:
      _, url = tensorboard.make_simple_server(
          self._StubApplication(),
          host='::1',
          port=0)
      self.assertStartsWith(actual=url, expected_start='http://[::1]:')
      one_passed = True
    except socket.error:
      # IPv6 is not supported
      pass
    self.assertTrue(one_passed)  # We expect either IPv4 or IPv6 to be supported


class TensorBoardApplcationConstructionTest(tf.test.TestCase):

  def testExceptions(self):
    logdir = '/fake/foo'
    multiplexer = event_multiplexer.EventMultiplexer()

    # Fails if there is an unnamed plugin
    with self.assertRaises(ValueError):
      # This plugin lacks a name.
      plugins = [
          FakePlugin(plugin_name=None, is_active_value=True, routes_mapping={})
      ]
      application.TensorBoardWSGIApp(logdir, plugins, multiplexer, 0)

    # Fails if there are two plugins with same name
    with self.assertRaises(ValueError):
      plugins = [
          FakePlugin(
              plugin_name='foo', is_active_value=True, routes_mapping={}),
          FakePlugin(
              plugin_name='foo', is_active_value=True, routes_mapping={}),
      ]
      application.TensorBoardWSGIApp(logdir, plugins, multiplexer, 0)


if __name__ == '__main__':
  tf.test.main()
