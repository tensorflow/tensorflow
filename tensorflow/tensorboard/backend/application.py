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
"""TensorBoard WSGI Application Logic.

TensorBoardApplication constructs TensorBoard as a WSGI application.
It handles serving static assets, and implements TensorBoard data APIs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import threading
import time

import six
from six.moves.urllib import parse as urlparse
import tensorflow as tf
from werkzeug import wrappers

from tensorflow.tensorboard.backend import http_util
from tensorflow.tensorboard.backend.event_processing import event_accumulator
from tensorflow.tensorboard.backend.event_processing import event_multiplexer


DEFAULT_SIZE_GUIDANCE = {
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 10,
    event_accumulator.AUDIO: 10,
    event_accumulator.SCALARS: 1000,
    event_accumulator.HEALTH_PILLS: 100,
    event_accumulator.HISTOGRAMS: 50,
}

DATA_PREFIX = '/data'
LOGDIR_ROUTE = '/logdir'
RUNS_ROUTE = '/runs'
PLUGIN_PREFIX = '/plugin'
PLUGINS_LISTING_ROUTE = '/plugins_listing'
TAB_ROUTES = ['', '/events', '/images', '/audio', '/graphs', '/histograms']

# Slashes in a plugin name could throw the router for a loop. An empty
# name would be confusing, too. To be safe, let's restrict the valid
# names as follows.
_VALID_PLUGIN_RE = re.compile(r'^[A-Za-z0-9_.-]+$')


def standard_tensorboard_wsgi(
    logdir,
    purge_orphaned_data,
    reload_interval,
    plugins):
  """Construct a TensorBoardWSGIApp with standard plugins and multiplexer.

  Args:
    logdir: The path to the directory containing events files.
    purge_orphaned_data: Whether to purge orphaned data.
    reload_interval: The interval at which the backend reloads more data in
        seconds.
    plugins: A list of plugins for TensorBoard to initialize.

  Returns:
    The new TensorBoard WSGI application.
  """
  multiplexer = event_multiplexer.EventMultiplexer(
      size_guidance=DEFAULT_SIZE_GUIDANCE,
      purge_orphaned_data=purge_orphaned_data)

  return TensorBoardWSGIApp(logdir, plugins, multiplexer, reload_interval)


class TensorBoardWSGIApp(object):
  """The TensorBoard application, conforming to WSGI spec."""

  # How many samples to include in sampling API calls by default.
  DEFAULT_SAMPLE_COUNT = 10

  # NOTE TO MAINTAINERS: An accurate Content-Length MUST be specified on all
  #                      responses using send_header.
  protocol_version = 'HTTP/1.1'

  def __init__(self, logdir, plugins, multiplexer, reload_interval):
    """Constructs the TensorBoard application.

    Args:
      logdir: the logdir spec that describes where data will be loaded.
        may be a directory, or comma,separated list of directories, or colons
        can be used to provide named directories
      plugins: List of plugins that extend tensorboard.plugins.BasePlugin
      multiplexer: The EventMultiplexer with TensorBoard data to serve
      reload_interval: How often (in seconds) to reload the Multiplexer

    Returns:
      A WSGI application that implements the TensorBoard backend.

    Raises:
      ValueError: If some plugin has no plugin_name
      ValueError: If some plugin has an invalid plugin_name (plugin
          names must only contain [A-Za-z0-9_.-])
      ValueError: If two plugins have the same plugin_name
      ValueError: If some plugin handles a route that does not start
          with a slash
    """
    self._logdir = logdir
    self._plugins = plugins
    self._multiplexer = multiplexer
    self.tag = get_tensorboard_tag()

    path_to_run = parse_event_files_spec(self._logdir)
    if reload_interval:
      start_reloading_multiplexer(self._multiplexer, path_to_run,
                                  reload_interval)
    else:
      reload_multiplexer(self._multiplexer, path_to_run)

    self.data_applications = {
        DATA_PREFIX + LOGDIR_ROUTE:
            self._serve_logdir,
        # TODO(chizeng): Delete this RPC once we have skylark rules that obviate
        # the need for the frontend to determine which plugins are active.
        DATA_PREFIX + PLUGINS_LISTING_ROUTE: self._serve_plugins_listing,
        DATA_PREFIX + RUNS_ROUTE: self._serve_runs,
    }

    # Serve the routes from the registered plugins using their name as the route
    # prefix. For example if plugin z has two routes /a and /b, they will be
    # served as /data/plugin/z/a and /data/plugin/z/b.
    plugin_names_encountered = set()
    for plugin in self._plugins:
      if plugin.plugin_name is None:
        raise ValueError('Plugin %s has no plugin_name' % plugin)
      if not _VALID_PLUGIN_RE.match(plugin.plugin_name):
        raise ValueError('Plugin %s has invalid name %r' % (plugin,
                                                            plugin.plugin_name))
      if plugin.plugin_name in plugin_names_encountered:
        raise ValueError('Duplicate plugins for name %s' % plugin.plugin_name)
      plugin_names_encountered.add(plugin.plugin_name)

      try:
        plugin_apps = plugin.get_plugin_apps(self._multiplexer, self._logdir)
      except Exception as e:  # pylint: disable=broad-except
        tf.logging.warning('Plugin %s failed. Exception: %s',
                           plugin.plugin_name, str(e))
        continue
      for route, app in plugin_apps.items():
        if not route.startswith('/'):
          raise ValueError('Plugin named %r handles invalid route %r: '
                           'route does not start with a slash' %
                           (plugin.plugin_name, route))
        path = DATA_PREFIX + PLUGIN_PREFIX + '/' + plugin.plugin_name + route
        self.data_applications[path] = app

  def _path_is_safe(self, path):
    """Check path is safe (stays within current directory).

    This is for preventing directory-traversal attacks.

    Args:
      path: The path to check for safety.

    Returns:
      True if the given path stays within the current directory, and false
      if it would escape to a higher directory. E.g. _path_is_safe('index.html')
      returns true, but _path_is_safe('../../../etc/password') returns false.
    """
    base = os.path.abspath(os.curdir)
    absolute_path = os.path.abspath(path)
    prefix = os.path.commonprefix([base, absolute_path])
    return prefix == base

  @wrappers.Request.application
  def _serve_logdir(self, request):
    """Respond with a JSON object containing this TensorBoard's logdir."""
    return http_util.Respond(
        request, {'logdir': self._logdir}, 'application/json')

  @wrappers.Request.application
  def _serve_plugins_listing(self, request):
    """Serves an object mapping plugin name to whether it is enabled.

    Args:
      request: The werkzeug.Request object.

    Returns:
      A werkzeug.Response object.
    """
    return http_util.Respond(
        request,
        {plugin.plugin_name: plugin.is_active() for plugin in self._plugins},
        'application/json')

  @wrappers.Request.application
  def _serve_runs(self, request):
    """WSGI app serving a JSON object about runs and tags.

    Returns a mapping from runs to tagType to list of tags for that run.

    Args:
      request: A werkzeug request

    Returns:
      A werkzeug Response with the following content:
      {runName: {firstEventTimestamp: 123456.789}}
    """
    run_names = sorted(self._multiplexer.Runs())  # Why `sorted`? See below.
    def get_first_event_timestamp(run_name):
      try:
        return self._multiplexer.FirstEventTimestamp(run_name)
      except ValueError:
        tf.logging.warning('Unable to get first event timestamp for run %s',
                           run_name)
        # Put runs without a timestamp at the end. Their internal
        # ordering would be nondeterministic, but Python's sorts are
        # stable, so `sorted`ing the initial list above provides a
        # deterministic ordering. Of course, we cannot guarantee that
        # this will be append-only for new event-less runs.
        return float('inf')
    first_event_timestamps = {
        run_name: get_first_event_timestamp(run_name)
        for run_name in run_names
    }
    run_names.sort(key=first_event_timestamps.get)
    return http_util.Respond(request, run_names, 'application/json')

  @wrappers.Request.application
  def _serve_index(self, request):
    """Serves the index page (i.e., the tensorboard app itself)."""
    contents = tf.resource_loader.load_resource(
        'tensorboard/components/index.html')
    return http_util.Respond(request, contents, 'text/html', expires=3600)

  def __call__(self, environ, start_response):  # pylint: disable=invalid-name
    """Central entry point for the TensorBoard application.

    This method handles routing to sub-applications. It does simple routing
    using regular expression matching.

    This __call__ method conforms to the WSGI spec, so that instances of this
    class are WSGI applications.

    Args:
      environ: See WSGI spec.
      start_response: See WSGI spec.

    Returns:
      A werkzeug Response.
    """
    request = wrappers.Request(environ)
    parsed_url = urlparse.urlparse(request.path)

    # Remove a trailing slash, if present.
    clean_path = parsed_url.path
    if clean_path.endswith('/'):
      clean_path = clean_path[:-1]
    # pylint: disable=too-many-function-args
    if clean_path in self.data_applications:
      return self.data_applications[clean_path](environ, start_response)
    elif clean_path in TAB_ROUTES:
      return self._serve_index(environ, start_response)
    else:
      tf.logging.warning('path %s not found, sending 404', clean_path)
      return http_util.Respond(request, 'Not found', 'text/plain', code=404)(
          environ, start_response)
    # pylint: enable=too-many-function-args


def parse_event_files_spec(logdir):
  """Parses `logdir` into a map from paths to run group names.

  The events files flag format is a comma-separated list of path specifications.
  A path specification either looks like 'group_name:/path/to/directory' or
  '/path/to/directory'; in the latter case, the group is unnamed. Group names
  cannot start with a forward slash: /foo:bar/baz will be interpreted as a
  spec with no name and path '/foo:bar/baz'.

  Globs are not supported.

  Args:
    logdir: A comma-separated list of run specifications.
  Returns:
    A dict mapping directory paths to names like {'/path/to/directory': 'name'}.
    Groups without an explicit name are named after their path. If logdir is
    None, returns an empty dict, which is helpful for testing things that don't
    require any valid runs.
  """
  files = {}
  if logdir is None:
    return files
  # Make sure keeping consistent with ParseURI in core/lib/io/path.cc
  uri_pattern = re.compile('[a-zA-Z][0-9a-zA-Z.]*://.*')
  for specification in logdir.split(','):
    # Check if the spec contains group. A spec start with xyz:// is regarded as
    # URI path spec instead of group spec. If the spec looks like /foo:bar/baz,
    # then we assume it's a path with a colon.
    if (uri_pattern.match(specification) is None and ':' in specification and
        specification[0] != '/'):
      # We split at most once so run_name:/path:with/a/colon will work.
      run_name, _, path = specification.partition(':')
    else:
      run_name = None
      path = specification
    if uri_pattern.match(path) is None:
      path = os.path.realpath(path)
    files[path] = run_name
  return files


def reload_multiplexer(multiplexer, path_to_run):
  """Loads all runs into the multiplexer.

  Args:
    multiplexer: The `EventMultiplexer` to add runs to and reload.
    path_to_run: A dict mapping from paths to run names, where `None` as the run
      name is interpreted as a run name equal to the path.
  """
  start = time.time()
  tf.logging.info('TensorBoard reload process beginning')
  for (path, name) in six.iteritems(path_to_run):
    multiplexer.AddRunsFromDirectory(path, name)
  tf.logging.info('TensorBoard reload process: Reload the whole Multiplexer')
  multiplexer.Reload()
  duration = time.time() - start
  tf.logging.info('TensorBoard done reloading. Load took %0.3f secs', duration)


def start_reloading_multiplexer(multiplexer, path_to_run, load_interval):
  """Starts a thread to automatically reload the given multiplexer.

  The thread will reload the multiplexer by calling `ReloadMultiplexer` every
  `load_interval` seconds, starting immediately.

  Args:
    multiplexer: The `EventMultiplexer` to add runs to and reload.
    path_to_run: A dict mapping from paths to run names, where `None` as the run
      name is interpreted as a run name equal to the path.
    load_interval: How many seconds to wait after one load before starting the
      next load.

  Returns:
    A started `threading.Thread` that reloads the multiplexer.
  """

  # We don't call multiplexer.Reload() here because that would make
  # AddRunsFromDirectory block until the runs have all loaded.
  def _reload_forever():
    while True:
      reload_multiplexer(multiplexer, path_to_run)
      time.sleep(load_interval)

  thread = threading.Thread(target=_reload_forever)
  thread.daemon = True
  thread.start()
  return thread


def get_tensorboard_tag():
  """Read the TensorBoard TAG number, and return it or an empty string."""
  tag = tf.resource_loader.load_resource('tensorboard/TAG').strip()
  return tag
