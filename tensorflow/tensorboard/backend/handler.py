# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""TensorBoard server handler logic.

TensorboardHandler contains all the logic for serving static files off of disk
and for handling the API calls to endpoints like /tags that require information
about loaded events.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import functools
import imghdr
import mimetypes
import os

from six import StringIO
from six.moves import BaseHTTPServer
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves.urllib import parse as urlparse

from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import event_accumulator
from tensorflow.tensorboard.backend import process_graph
from tensorflow.tensorboard.lib.python import http
from tensorflow.tensorboard.plugins.projector import plugin as projector_plugin

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

REGISTERED_PLUGINS = {
    'projector': projector_plugin.ProjectorPlugin(),
}

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
  CSV = 'csv'


class TensorboardHandler(BaseHTTPServer.BaseHTTPRequestHandler):
  """Handler class for use with BaseHTTPServer.HTTPServer.

  This is essentially a thin wrapper around calls to an EventMultiplexer object
  as well as serving files off disk.
  """

  # How many samples to include in sampling API calls by default.
  DEFAULT_SAMPLE_COUNT = 10

  # NOTE TO MAINTAINERS: An accurate Content-Length MUST be specified on all
  #                      responses using send_header.
  protocol_version = 'HTTP/1.1'

  def __init__(self, multiplexer, logdir, *args):
    self._multiplexer = multiplexer
    self._logdir = logdir
    self._setup_data_handlers()
    BaseHTTPServer.BaseHTTPRequestHandler.__init__(self, *args)

  def _setup_data_handlers(self):
    self.data_handlers = {
        DATA_PREFIX + LOGDIR_ROUTE: self._serve_logdir,
        DATA_PREFIX + SCALARS_ROUTE: self._serve_scalars,
        DATA_PREFIX + GRAPH_ROUTE: self._serve_graph,
        DATA_PREFIX + RUN_METADATA_ROUTE: self._serve_run_metadata,
        DATA_PREFIX + HISTOGRAMS_ROUTE: self._serve_histograms,
        DATA_PREFIX + COMPRESSED_HISTOGRAMS_ROUTE:
            self._serve_compressed_histograms,
        DATA_PREFIX + IMAGES_ROUTE: self._serve_images,
        DATA_PREFIX + INDIVIDUAL_IMAGE_ROUTE: self._serve_image,
        DATA_PREFIX + AUDIO_ROUTE: self._serve_audio,
        DATA_PREFIX + INDIVIDUAL_AUDIO_ROUTE: self._serve_individual_audio,
        DATA_PREFIX + RUNS_ROUTE: self._serve_runs,
        '/app.js': self._serve_js
    }

    # Serve the routes from the registered plugins using their name as the route
    # prefix. For example if plugin z has two routes /a and /b, they will be
    # served as /data/plugin/z/a and /data/plugin/z/b.
    for name in REGISTERED_PLUGINS:
      try:
        plugin = REGISTERED_PLUGINS[name]
        plugin_handlers = plugin.get_plugin_handlers(
            self._multiplexer.RunPaths(), self._logdir)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning('Plugin %s failed. Exception: %s', name, str(e))
        continue
      for route, handler in plugin_handlers.items():
        path = DATA_PREFIX + PLUGIN_PREFIX + '/' + name + route
        self.data_handlers[path] = functools.partial(handler, self)

  def respond(self, *args, **kwargs):
    """Delegates to http.Respond."""
    http.Respond(self, *args, **kwargs)

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

  def _serve_logdir(self, unused_query_params):
    """Writes out the logdir argument with which this tensorboard was started.
    """
    self.respond({'logdir': self._logdir}, 'application/json')

  def _serve_scalars(self, query_params):
    """Given a tag and single run, return array of ScalarEvents.

    Alternately, if both the tag and the run are omitted, returns JSON object
    where obj[run][tag] contains sample values for the given tag in the given
    run.

    Args:
      query_params: The query parameters as a dict.
    """
    # TODO(cassandrax): return HTTP status code for malformed requests
    tag = query_params.get('tag')
    run = query_params.get('run')
    if tag is None and run is None:
      if query_params.get('format') == _OutputFormat.CSV:
        self.respond('Scalar sample values only supports JSON output',
                     'text/plain', 400)
        return

      sample_count = int(query_params.get('sample_count',
                                          self.DEFAULT_SAMPLE_COUNT))
      values = {}
      for run_name, tags in self._multiplexer.Runs().items():
        values[run_name] = {
            tag: _uniform_sample(
                self._multiplexer.Scalars(run_name, tag), sample_count)
            for tag in tags['scalars']
        }
    else:
      values = self._multiplexer.Scalars(run, tag)

    if query_params.get('format') == _OutputFormat.CSV:
      string_io = StringIO()
      writer = csv.writer(string_io)
      writer.writerow(['Wall time', 'Step', 'Value'])
      writer.writerows(values)
      self.respond(string_io.getvalue(), 'text/csv')
    else:
      self.respond(values, 'application/json')

  def _serve_graph(self, query_params):
    """Given a single run, return the graph definition in json format."""
    run = query_params.get('run', None)
    if run is None:
      self.respond('query parameter "run" is required', 'text/plain', 400)
      return

    try:
      graph = self._multiplexer.Graph(run)
    except ValueError:
      self.send_response(404)
      return

    limit_attr_size = query_params.get('limit_attr_size', None)
    if limit_attr_size is not None:
      try:
        limit_attr_size = int(limit_attr_size)
      except ValueError:
        self.respond('query parameter `limit_attr_size` must be integer',
                     'text/plain', 400)
        return

    large_attrs_key = query_params.get('large_attrs_key', None)
    try:
      process_graph.prepare_graph_for_ui(graph, limit_attr_size,
                                         large_attrs_key)
    except ValueError as e:
      self.respond(e.message, 'text/plain', 400)
      return

    self.respond(str(graph), 'text/x-protobuf')  # pbtxt

  def _serve_run_metadata(self, query_params):
    """Given a tag and a TensorFlow run, return the session.run() metadata."""
    tag = query_params.get('tag', None)
    run = query_params.get('run', None)
    if tag is None:
      self.respond('query parameter "tag" is required', 'text/plain', 400)
      return
    if run is None:
      self.respond('query parameter "run" is required', 'text/plain', 400)
      return
    try:
      run_metadata = self._multiplexer.RunMetadata(run, tag)
    except ValueError:
      self.send_response(404)
      return
    self.respond(str(run_metadata), 'text/x-protobuf')  # pbtxt

  def _serve_histograms(self, query_params):
    """Given a tag and single run, return an array of histogram values."""
    tag = query_params.get('tag')
    run = query_params.get('run')
    values = self._multiplexer.Histograms(run, tag)
    self.respond(values, 'application/json')

  def _serve_compressed_histograms(self, query_params):
    """Given a tag and single run, return an array of compressed histograms."""
    tag = query_params.get('tag')
    run = query_params.get('run')
    compressed_histograms = self._multiplexer.CompressedHistograms(run, tag)
    if query_params.get('format') == _OutputFormat.CSV:
      string_io = StringIO()
      writer = csv.writer(string_io)

      # Build the headers; we have two columns for timing and two columns for
      # each compressed histogram bucket.
      headers = ['Wall time', 'Step']
      if compressed_histograms:
        bucket_count = len(compressed_histograms[0].compressed_histogram_values)
        for i in xrange(bucket_count):
          headers += ['Edge %d basis points' % i, 'Edge %d value' % i]
      writer.writerow(headers)

      for compressed_histogram in compressed_histograms:
        row = [compressed_histogram.wall_time, compressed_histogram.step]
        for value in compressed_histogram.compressed_histogram_values:
          row += [value.rank_in_bps, value.value]
        writer.writerow(row)
      self.respond(string_io.getvalue(), 'text/csv')
    else:
      self.respond(compressed_histograms, 'application/json')

  def _serve_images(self, query_params):
    """Given a tag and list of runs, serve a list of images.

    Note that the images themselves are not sent; instead, we respond with URLs
    to the images. The frontend should treat these URLs as opaque and should not
    try to parse information about them or generate them itself, as the format
    may change.

    Args:
      query_params: The query parameters as a dict.
    """
    tag = query_params.get('tag')
    run = query_params.get('run')

    images = self._multiplexer.Images(run, tag)
    response = self._image_response_for_run(images, run, tag)
    self.respond(response, 'application/json')

  def _serve_image(self, query_params):
    """Serves an individual image."""
    tag = query_params.get('tag')
    run = query_params.get('run')
    index = int(query_params.get('index'))
    image = self._multiplexer.Images(run, tag)[index]
    encoded_image_string = image.encoded_image_string
    content_type = _content_type_for_image(encoded_image_string)
    self.respond(encoded_image_string, content_type)

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

  def _serve_audio(self, query_params):
    """Given a tag and list of runs, serve a list of audio.

    Note that the audio clips themselves are not sent; instead, we respond with
    URLs to the audio. The frontend should treat these URLs as opaque and should
    not try to parse information about them or generate them itself, as the
    format may change.

    Args:
      query_params: The query parameters as a dict.

    """
    tag = query_params.get('tag')
    run = query_params.get('run')

    audio_list = self._multiplexer.Audio(run, tag)
    response = self._audio_response_for_run(audio_list, run, tag)
    self.respond(response, 'application/json')

  def _serve_individual_audio(self, query_params):
    """Serves an individual audio clip."""
    tag = query_params.get('tag')
    run = query_params.get('run')
    index = int(query_params.get('index'))
    audio = self._multiplexer.Audio(run, tag)[index]
    self.respond(audio.encoded_audio_string, audio.content_type)

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

  def _serve_runs(self, unused_query_params):
    """Return a JSON object about runs and tags.

    Returns a mapping from runs to tagType to list of tags for that run.

    Returns:
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
    self.respond(runs, 'application/json')

  def _serve_index(self, unused_query_params):
    """Serves the index page (i.e., the tensorboard app itself)."""
    self._serve_static_file('/dist/index.html')

  def _serve_js(self, unused_query_params):
    """Serves the JavaScript for the index page."""
    self._serve_static_file('/dist/app.js')

  def _serve_static_file(self, path):
    """Serves the static file located at the given path.

    Args:
      path: The path of the static file, relative to the tensorboard/ directory.
    """
    # Strip off the leading forward slash.
    orig_path = path.lstrip('/')
    if not self._path_is_safe(orig_path):
      logging.warning('path not safe: %s', orig_path)
      self.respond('Naughty naughty!', 'text/plain', 400)
      return
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
        logging.info('path %s not found, sending 404', path)
        self.respond('Not found', 'text/plain', 404)
        return
    mimetype, content_encoding = mimetypes.guess_type(path)
    mimetype = mimetype or 'application/octet-stream'
    self.respond(contents, mimetype, expires=3600,
                 content_encoding=content_encoding)

  def do_GET(self):  # pylint: disable=invalid-name
    """Handler for all get requests."""
    parsed_url = urlparse.urlparse(self.path)

    # Remove a trailing slash, if present.
    clean_path = parsed_url.path
    if clean_path.endswith('/'):
      clean_path = clean_path[:-1]

    query_params = urlparse.parse_qs(parsed_url.query)
    # parse_qs returns a list of values for each key; we're only interested in
    # the first.
    for key in query_params:
      value_count = len(query_params[key])
      if value_count != 1:
        self.respond(
            'query parameter %s should have exactly one value, had %d' %
            (key, value_count), 'text/plain', 400)
        return
      query_params[key] = query_params[key][0]

    if clean_path in self.data_handlers:
      self.data_handlers[clean_path](query_params)
    elif clean_path in TAB_ROUTES:
      self._serve_index(query_params)
    else:
      self._serve_static_file(clean_path)

  # @Override
  def log_message(self, *args):
    """Logs message."""
    # By default, BaseHTTPRequestHandler logs to stderr.
    logging.info(*args)

  # @Override
  def log_request(self, *args):
    """Does nothing."""
    # This is called by BaseHTTPRequestHandler.send_response() which causes it
    # to log every request. We've configured http.Respond() to only log
    # requests with >=400 status code.
    pass


def _uniform_sample(values, count):
  """Samples `count` values uniformly from `values`.

  Args:
    values: The values to sample from.
    count: The number of values to sample. Must be at least 2.

  Raises:
    ValueError: If `count` is not at least 2.
    TypeError: If `type(count) != int`.

  Returns:
    A list of values from `values`. The first and the last element will always
    be included. If `count > len(values)`, then all values will be returned.
  """

  if count < 2:
    raise ValueError('Must sample at least 2 elements, %d requested' % count)

  if count >= len(values):
    # Copy the list in case the caller mutates it.
    return list(values)

  return [
      # We divide by count - 1 to make sure we always get the first and the last
      # element.
      values[(len(values) - 1) * i // (count - 1)] for i in xrange(count)
  ]
