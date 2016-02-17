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

"""TensorBoard server handler logic.

TensorboardHandler contains all the logic for serving static files off of disk
and for handling the API calls to endpoints like /tags that require information
about loaded events.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import gzip
import imghdr
import json
import mimetypes
import os

from six import BytesIO
from six.moves import BaseHTTPServer
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves.urllib import parse as urlparse

from google.protobuf import text_format

from tensorflow.python.platform import logging
from tensorflow.python.platform import resource_loader
from tensorflow.python.summary import event_accumulator
from tensorflow.python.util import compat
from tensorflow.tensorboard.backend import float_wrapper


DATA_PREFIX = '/data'
RUNS_ROUTE = '/runs'
SCALARS_ROUTE = '/' + event_accumulator.SCALARS
IMAGES_ROUTE = '/' + event_accumulator.IMAGES
HISTOGRAMS_ROUTE = '/' + event_accumulator.HISTOGRAMS
COMPRESSED_HISTOGRAMS_ROUTE = '/' + event_accumulator.COMPRESSED_HISTOGRAMS
INDIVIDUAL_IMAGE_ROUTE = '/individualImage'
GRAPH_ROUTE = '/' + event_accumulator.GRAPH
TAB_ROUTES = ['', '/events', '/images', '/graphs', '/histograms']

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

  def __init__(self, multiplexer, *args):
    self._multiplexer = multiplexer
    BaseHTTPServer.BaseHTTPRequestHandler.__init__(self, *args)

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

  def _send_gzip_response(self, content, content_type, code=200):
    """Writes the given content as gzip response using the given content type.

    Args:
      content: The content to respond with.
      content_type: The mime type of the content.
      code: The numeric HTTP status code to use.
    """
    out = BytesIO()
    f = gzip.GzipFile(fileobj=out, mode='wb')
    f.write(compat.as_bytes(content))
    f.close()
    gzip_content = out.getvalue()
    self.send_response(code)
    self.send_header('Content-Type', content_type)
    self.send_header('Content-Length', len(gzip_content))
    self.send_header('Content-Encoding', 'gzip')
    self.end_headers()
    self.wfile.write(gzip_content)

  def _send_json_response(self, obj, code=200):
    """Writes out the given object as JSON using the given HTTP status code.

    This also replaces special float values with stringified versions.

    Args:
      obj: The object to respond with.
      code: The numeric HTTP status code to use.
    """

    output = json.dumps(float_wrapper.WrapSpecialFloats(obj))

    self.send_response(code)
    self.send_header('Content-Type', 'application/json')
    self.send_header('Content-Length', len(output))
    self.end_headers()
    self.wfile.write(compat.as_bytes(output))

  def _send_csv_response(self, serialized_csv, code=200):
    """Writes out the given string, which represents CSV data.

    Unlike _send_json_response, this does *not* perform the CSV serialization
    for you. It only sets the proper headers.

    Args:
      serialized_csv: A string containing some CSV data.
      code: The numeric HTTP status code to use.
    """

    self.send_response(code)
    self.send_header('Content-Type', 'text/csv')
    self.send_header('Content-Length', len(serialized_csv))
    self.end_headers()
    self.wfile.write(serialized_csv)

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
        self.send_error(400, 'Scalar sample values only supports JSON output')
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
      string_io = BytesIO()
      writer = csv.writer(string_io)
      writer.writerow(['Wall time', 'Step', 'Value'])
      writer.writerows(values)
      self._send_csv_response(string_io.getvalue())
    else:
      self._send_json_response(values)

  def _serve_graph(self, query_params):
    """Given a single run, return the graph definition in json format."""
    run = query_params.get('run', None)
    if run is None:
      self.send_error(400, 'query parameter "run" is required')
      return

    try:
      graph = self._multiplexer.Graph(run)
    except ValueError:
      self.send_response(404)
      return

    # Serialize the graph to pbtxt format.
    graph_pbtxt = text_format.MessageToString(graph)
    # Gzip it and send it to the user.
    self._send_gzip_response(graph_pbtxt, 'text/plain')

  def _serve_histograms(self, query_params):
    """Given a tag and single run, return an array of histogram values."""
    tag = query_params.get('tag')
    run = query_params.get('run')
    values = self._multiplexer.Histograms(run, tag)
    self._send_json_response(values)

  def _serve_compressed_histograms(self, query_params):
    """Given a tag and single run, return an array of compressed histograms."""
    tag = query_params.get('tag')
    run = query_params.get('run')
    compressed_histograms = self._multiplexer.CompressedHistograms(run, tag)
    if query_params.get('format') == _OutputFormat.CSV:
      string_io = BytesIO()
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
      self._send_csv_response(string_io.getvalue())
    else:
      self._send_json_response(compressed_histograms)

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
    self._send_json_response(response)

  def _serve_image(self, query_params):
    """Serves an individual image."""
    tag = query_params.get('tag')
    run = query_params.get('run')
    index = int(query_params.get('index'))
    image = self._multiplexer.Images(run, tag)[index]
    encoded_image_string = image.encoded_image_string
    content_type = _content_type_for_image(encoded_image_string)

    self.send_response(200)
    self.send_header('Content-Type', content_type)
    self.send_header('Content-Length', len(encoded_image_string))
    self.end_headers()
    self.wfile.write(encoded_image_string)

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

  def _serve_runs(self, unused_query_params):
    """Return a JSON object about runs and tags.

    Returns a mapping from runs to tagType to list of tags for that run.

    Returns:
      {runName: {images: [tag1, tag2, tag3],
                 scalars: [tagA, tagB, tagC],
                 histograms: [tagX, tagY, tagZ]}}
    """
    self._send_json_response(self._multiplexer.Runs())

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
    path = path.lstrip('/')
    if not self._path_is_safe(path):
      logging.info('path %s not safe, sending 404', path)
      # Traversal attack, so 404.
      self.send_error(404)
      return

    if path.startswith('external'):
      # For compatibility with latest version of Bazel, we renamed bower
      # packages to use '_' rather than '-' in their package name.
      # This means that the directory structure is changed too.
      # So that all our recursive imports work, we need to modify incoming
      # requests to map onto the new directory structure.
      components = path.split('/')
      components[1] = components[1].replace('-', '_')
      path = ('/').join(components)
      path = os.path.join('../', path)
    else:
      path = os.path.join('tensorboard', path)
    # Open the file and read it.
    try:
      contents = resource_loader.load_resource(path)
    except IOError:
      logging.info('path %s not found, sending 404', path)
      self.send_error(404)
      return

    self.send_response(200)

    mimetype = mimetypes.guess_type(path)[0] or 'application/octet-stream'
    self.send_header('Content-Type', mimetype)
    self.end_headers()
    self.wfile.write(contents)

  def do_GET(self):  # pylint: disable=invalid-name
    """Handler for all get requests."""
    parsed_url = urlparse.urlparse(self.path)

    # Remove a trailing slash, if present.
    clean_path = parsed_url.path
    if clean_path.endswith('/'):
      clean_path = clean_path[:-1]

    data_handlers = {
        DATA_PREFIX + SCALARS_ROUTE: self._serve_scalars,
        DATA_PREFIX + GRAPH_ROUTE: self._serve_graph,
        DATA_PREFIX + HISTOGRAMS_ROUTE: self._serve_histograms,
        DATA_PREFIX + COMPRESSED_HISTOGRAMS_ROUTE:
            self._serve_compressed_histograms,
        DATA_PREFIX + IMAGES_ROUTE: self._serve_images,
        DATA_PREFIX + INDIVIDUAL_IMAGE_ROUTE: self._serve_image,
        DATA_PREFIX + RUNS_ROUTE: self._serve_runs,
        '/app.js': self._serve_js
    }

    query_params = urlparse.parse_qs(parsed_url.query)
    # parse_qs returns a list of values for each key; we're only interested in
    # the first.
    for key in query_params:
      value_count = len(query_params[key])
      if value_count != 1:
        self.send_error(
            400,
            'query parameter %s should have exactly one value, had %d' %
            (key, value_count))
        return
      query_params[key] = query_params[key][0]

    if clean_path in data_handlers:
      data_handlers[clean_path](query_params)
    elif clean_path in TAB_ROUTES:
      self._serve_index(query_params)
    else:
      self._serve_static_file(clean_path)


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
