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

import imghdr
import mimetypes
import os
import re
import threading
import time

import six
from six.moves import urllib
from six.moves.urllib import parse as urlparse
from werkzeug import wrappers

from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import event_accumulator
from tensorflow.python.summary import event_multiplexer
from tensorflow.tensorboard.backend import process_graph
from tensorflow.tensorboard.lib.python import http_util
from tensorflow.tensorboard.plugins.debugger import debugger_plugin
from tensorflow.tensorboard.plugins.projector import projector_plugin


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
SCALARS_ROUTE = '/' + event_accumulator.SCALARS
IMAGES_ROUTE = '/' + event_accumulator.IMAGES
AUDIO_ROUTE = '/' + event_accumulator.AUDIO
HISTOGRAMS_ROUTE = '/' + event_accumulator.HISTOGRAMS
COMPRESSED_HISTOGRAMS_ROUTE = '/' + event_accumulator.COMPRESSED_HISTOGRAMS
INDIVIDUAL_IMAGE_ROUTE = '/individualImage'
INDIVIDUAL_AUDIO_ROUTE = '/individualAudio'
GRAPH_ROUTE = '/' + event_accumulator.GRAPH
RUN_METADATA_ROUTE = '/' + event_accumulator.RUN_METADATA
TAB_ROUTES = ['', '/events', '/images', '/audio', '/graphs', '/histograms']

_IMGHDR_TO_MIMETYPE = {
    'bmp': 'image/bmp',
    'gif': 'image/gif',
    'jpeg': 'image/jpeg',
    'png': 'image/png'
}
_DEFAULT_IMAGE_MIMETYPE = 'application/octet-stream'


def _content_type_for_image(encoded_image_string):
  image_type = imghdr.what(None, encoded_image_string)
  return _IMGHDR_TO_MIMETYPE.get(image_type, _DEFAULT_IMAGE_MIMETYPE)


class _OutputFormat(object):
  """An enum used to list the valid output formats for API calls.

  Not all API calls support all formats (for example, only scalars and
  compressed histograms support CSV).
  """
  JSON = 'json'


def standard_tensorboard_wsgi(logdir, purge_orphaned_data, reload_interval):
  """Construct a TensorBoardWSGIApp with standard plugins and multiplexer."""
  multiplexer = event_multiplexer.EventMultiplexer(
      size_guidance=DEFAULT_SIZE_GUIDANCE,
      purge_orphaned_data=purge_orphaned_data)

  plugins = {
      debugger_plugin.PLUGIN_PREFIX_ROUTE:
          debugger_plugin.DebuggerPlugin(multiplexer),
      projector_plugin.PLUGIN_PREFIX_ROUTE:
          projector_plugin.ProjectorPlugin(),
  }

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
      plugins: Map from plugin name to plugin application
      multiplexer: The EventMultiplexer with TensorBoard data to serve
      reload_interval: How often (in seconds) to reload the Multiplexer

    Returns:
      A WSGI application that implements the TensorBoard backend.
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
        DATA_PREFIX + SCALARS_ROUTE:
            self._serve_scalars,
        DATA_PREFIX + GRAPH_ROUTE:
            self._serve_graph,
        DATA_PREFIX + RUN_METADATA_ROUTE:
            self._serve_run_metadata,
        DATA_PREFIX + HISTOGRAMS_ROUTE:
            self._serve_histograms,
        DATA_PREFIX + COMPRESSED_HISTOGRAMS_ROUTE:
            self._serve_compressed_histograms,
        DATA_PREFIX + IMAGES_ROUTE:
            self._serve_images,
        DATA_PREFIX + INDIVIDUAL_IMAGE_ROUTE:
            self._serve_image,
        DATA_PREFIX + AUDIO_ROUTE:
            self._serve_audio,
        DATA_PREFIX + INDIVIDUAL_AUDIO_ROUTE:
            self._serve_individual_audio,
        DATA_PREFIX + RUNS_ROUTE:
            self._serve_runs,
        '/app.js':
            self._serve_js
    }

    # Serve the routes from the registered plugins using their name as the route
    # prefix. For example if plugin z has two routes /a and /b, they will be
    # served as /data/plugin/z/a and /data/plugin/z/b.
    for name in self._plugins:
      try:
        plugin = self._plugins[name]
        plugin_apps = plugin.get_plugin_apps(self._multiplexer.RunPaths(),
                                             self._logdir)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning('Plugin %s failed. Exception: %s', name, str(e))
        continue
      for route, app in plugin_apps.items():
        path = DATA_PREFIX + PLUGIN_PREFIX + '/' + name + route
        self.data_applications[path] = app

  # We use underscore_names for consistency with inherited methods.

  def _image_response_for_run(self, run_images, run, tag):
    """Builds a JSON-serializable object with information about run_images.

    Args:
      run_images: A list of event_accumulator.ImageValueEvent objects.
      run: The name of the run.
      tag: The name of the tag the images all belong to.

    Returns:
      A list of dictionaries containing the wall time, step, URL, width, and
      height for each image.
    """
    response = []
    for index, run_image in enumerate(run_images):
      response.append({
          'wall_time': run_image.wall_time,
          'step': run_image.step,
          # We include the size so that the frontend can add that to the <img>
          # tag so that the page layout doesn't change when the image loads.
          'width': run_image.width,
          'height': run_image.height,
          'query': self._query_for_individual_image(run, tag, index)
      })
    return response

  def _audio_response_for_run(self, run_audio, run, tag):
    """Builds a JSON-serializable object with information about run_audio.

    Args:
      run_audio: A list of event_accumulator.AudioValueEvent objects.
      run: The name of the run.
      tag: The name of the tag the images all belong to.

    Returns:
      A list of dictionaries containing the wall time, step, URL, and
      content_type for each audio clip.
    """
    response = []
    for index, run_audio_clip in enumerate(run_audio):
      response.append({
          'wall_time': run_audio_clip.wall_time,
          'step': run_audio_clip.step,
          'content_type': run_audio_clip.content_type,
          'query': self._query_for_individual_audio(run, tag, index)
      })
    return response

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
  def _serve_scalars(self, request):
    """Given a tag and single run, return array of ScalarEvents."""
    # TODO(cassandrax): return HTTP status code for malformed requests
    tag = request.args.get('tag')
    run = request.args.get('run')
    values = self._multiplexer.Scalars(run, tag)

    return http_util.Respond(request, values, 'application/json')

  @wrappers.Request.application
  def _serve_graph(self, request):
    """Given a single run, return the graph definition in json format."""
    run = request.args.get('run', None)
    if run is None:
      return http_util.Respond(
          request, 'query parameter "run" is required', 'text/plain', 400)

    try:
      graph = self._multiplexer.Graph(run)
    except ValueError:
      return http_util.Respond(request, '404 Not Found', code=404)

    limit_attr_size = request.args.get('limit_attr_size', None)
    if limit_attr_size is not None:
      try:
        limit_attr_size = int(limit_attr_size)
      except ValueError:
        return http_util.Respond(
            request, 'query parameter `limit_attr_size` must be integer',
            'text/plain', 400)

    large_attrs_key = request.args.get('large_attrs_key', None)
    try:
      process_graph.prepare_graph_for_ui(graph, limit_attr_size,
                                         large_attrs_key)
    except ValueError as e:
      return http_util.Respond(request, e.message, 'text/plain', 400)

    return http_util.Respond(request, str(graph), 'text/x-protobuf')  # pbtxt

  @wrappers.Request.application
  def _serve_run_metadata(self, request):
    """Given a tag and a TensorFlow run, return the session.run() metadata."""
    tag = request.args.get('tag', None)
    run = request.args.get('run', None)
    if tag is None:
      return http_util.Respond(
          request, 'query parameter "tag" is required', 'text/plain', 400)
    if run is None:
      return http_util.Respond(
          request, 'query parameter "run" is required', 'text/plain', 400)
    try:
      run_metadata = self._multiplexer.RunMetadata(run, tag)
    except ValueError:
      return http_util.Respond(request, '404 Not Found', code=404)
    return http_util.Respond(
        request, str(run_metadata), 'text/x-protobuf')  # pbtxt

  @wrappers.Request.application
  def _serve_histograms(self, request):
    """Given a tag and single run, return an array of histogram values."""
    tag = request.args.get('tag')
    run = request.args.get('run')
    values = self._multiplexer.Histograms(run, tag)
    return http_util.Respond(request, values, 'application/json')

  @wrappers.Request.application
  def _serve_compressed_histograms(self, request):
    """Given a tag and single run, return an array of compressed histograms."""
    tag = request.args.get('tag')
    run = request.args.get('run')
    compressed_histograms = self._multiplexer.CompressedHistograms(run, tag)
    return http_util.Respond(request, compressed_histograms, 'application/json')

  @wrappers.Request.application
  def _serve_images(self, request):
    """Given a tag and list of runs, serve a list of images.

    Note that the images themselves are not sent; instead, we respond with URLs
    to the images. The frontend should treat these URLs as opaque and should not
    try to parse information about them or generate them itself, as the format
    may change.

    Args:
      request: A werkzeug.wrappers.Request object.

    Returns:
      A werkzeug.Response application.
    """
    tag = request.args.get('tag')
    run = request.args.get('run')

    images = self._multiplexer.Images(run, tag)
    response = self._image_response_for_run(images, run, tag)
    return http_util.Respond(request, response, 'application/json')

  @wrappers.Request.application
  def _serve_image(self, request):
    """Serves an individual image."""
    tag = request.args.get('tag')
    run = request.args.get('run')
    index = int(request.args.get('index'))
    image = self._multiplexer.Images(run, tag)[index]
    encoded_image_string = image.encoded_image_string
    content_type = _content_type_for_image(encoded_image_string)
    return http_util.Respond(request, encoded_image_string, content_type)

  def _query_for_individual_image(self, run, tag, index):
    """Builds a URL for accessing the specified image.

    This should be kept in sync with _serve_image. Note that the URL is *not*
    guaranteed to always return the same image, since images may be unloaded
    from the reservoir as new images come in.

    Args:
      run: The name of the run.
      tag: The tag.
      index: The index of the image. Negative values are OK.

    Returns:
      A string representation of a URL that will load the index-th
      sampled image in the given run with the given tag.
    """
    query_string = urllib.parse.urlencode({
        'run': run,
        'tag': tag,
        'index': index
    })
    return query_string

  @wrappers.Request.application
  def _serve_audio(self, request):
    """Given a tag and list of runs, serve a list of audio.

    Note that the audio clips themselves are not sent; instead, we respond with
    URLs to the audio. The frontend should treat these URLs as opaque and should
    not try to parse information about them or generate them itself, as the
    format may change.

    Args:
      request: A werkzeug.wrappers.Request object.

    Returns:
      A werkzeug.Response application.
    """
    tag = request.args.get('tag')
    run = request.args.get('run')

    audio_list = self._multiplexer.Audio(run, tag)
    response = self._audio_response_for_run(audio_list, run, tag)
    return http_util.Respond(request, response, 'application/json')

  @wrappers.Request.application
  def _serve_individual_audio(self, request):
    """Serves an individual audio clip."""
    tag = request.args.get('tag')
    run = request.args.get('run')
    index = int(request.args.get('index'))
    audio = self._multiplexer.Audio(run, tag)[index]
    return http_util.Respond(
        request, audio.encoded_audio_string, audio.content_type)

  def _query_for_individual_audio(self, run, tag, index):
    """Builds a URL for accessing the specified audio.

    This should be kept in sync with _serve_individual_audio. Note that the URL
    is *not* guaranteed to always return the same audio, since audio may be
    unloaded from the reservoir as new audio comes in.

    Args:
      run: The name of the run.
      tag: The tag.
      index: The index of the audio. Negative values are OK.

    Returns:
      A string representation of a URL that will load the index-th
      sampled audio in the given run with the given tag.
    """
    query_string = urllib.parse.urlencode({
        'run': run,
        'tag': tag,
        'index': index
    })
    return query_string

  @wrappers.Request.application
  def _serve_runs(self, request):
    """WSGI app serving a JSON object about runs and tags.

    Returns a mapping from runs to tagType to list of tags for that run.

    Args:
      request: A werkzeug request

    Returns:
      A werkzeug Response with the following content:
      {runName: {images: [tag1, tag2, tag3],
                 audio: [tag4, tag5, tag6],
                 scalars: [tagA, tagB, tagC],
                 histograms: [tagX, tagY, tagZ],
                 firstEventTimestamp: 123456.789}}
    """
    runs = self._multiplexer.Runs()
    for run_name, run_data in runs.items():
      try:
        run_data['firstEventTimestamp'] = self._multiplexer.FirstEventTimestamp(
            run_name)
      except ValueError:
        logging.warning('Unable to get first event timestamp for run %s',
                        run_name)
        run_data['firstEventTimestamp'] = None
    return http_util.Respond(request, runs, 'application/json')

  @wrappers.Request.application
  def _serve_index(self, request):
    """Serves the index page (i.e., the tensorboard app itself)."""
    return self._serve_static_file(request, '/dist/index.html')

  @wrappers.Request.application
  def _serve_js(self, request):
    """Serves the JavaScript for the index page."""
    return self._serve_static_file(request, '/dist/app.js')

  def _serve_static_file(self, request, path):
    """Serves the static file located at the given path.

    Args:
      request: A werkzeug Request
      path: The path of the static file, relative to the tensorboard/ directory.

    Returns:
      A werkzeug.Response application.
    """
    # Strip off the leading forward slash.
    orig_path = path.lstrip('/')
    if not self._path_is_safe(orig_path):
      logging.warning('path not safe: %s', orig_path)
      return http_util.Respond(request, 'Naughty naughty!', 'text/plain', 400)
      # Resource loader wants a path relative to //WORKSPACE/tensorflow.
    path = os.path.join('tensorboard', orig_path)
    # Open the file and read it.
    try:
      contents = resource_loader.load_resource(path)
    except IOError:
      # For compatibility with latest version of Bazel, we renamed bower
      # packages to use '_' rather than '-' in their package name.
      # This means that the directory structure is changed too.
      # So that all our recursive imports work, we need to modify incoming
      # requests to map onto the new directory structure.
      path = orig_path
      components = path.split('/')
      components[0] = components[0].replace('-', '_')
      path = ('/').join(components)
      # Bazel keeps all the external dependencies in //WORKSPACE/external.
      # and resource loader wants a path relative to //WORKSPACE/tensorflow/.
      path = os.path.join('../external', path)
      try:
        contents = resource_loader.load_resource(path)
      except IOError:
        logging.warning('path %s not found, sending 404', path)
        return http_util.Respond(request, 'Not found', 'text/plain', code=404)
    mimetype, content_encoding = mimetypes.guess_type(path)
    mimetype = mimetype or 'application/octet-stream'
    return http_util.Respond(
        request,
        contents,
        mimetype,
        expires=3600,
        content_encoding=content_encoding)

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
      return self._serve_static_file(request, clean_path)(environ,
                                                          start_response)
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
  logging.info('TensorBoard reload process beginning')
  for (path, name) in six.iteritems(path_to_run):
    multiplexer.AddRunsFromDirectory(path, name)
  logging.info('TensorBoard reload process: Reload the whole Multiplexer')
  multiplexer.Reload()
  duration = time.time() - start
  logging.info('TensorBoard done reloading. Load took %0.3f secs', duration)


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
  tag = resource_loader.load_resource('tensorboard/TAG').strip()
  return tag
