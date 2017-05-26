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
"""The TensorBoard Images plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imghdr

from six.moves import urllib
from werkzeug import wrappers

from tensorflow.tensorboard.backend import http_util
from tensorflow.tensorboard.backend.event_processing import event_accumulator
from tensorflow.tensorboard.plugins import base_plugin

_PLUGIN_PREFIX_ROUTE = event_accumulator.IMAGES

_IMGHDR_TO_MIMETYPE = {
    'bmp': 'image/bmp',
    'gif': 'image/gif',
    'jpeg': 'image/jpeg',
    'png': 'image/png'
}

_DEFAULT_IMAGE_MIMETYPE = 'application/octet-stream'


class ImagesPlugin(base_plugin.TBPlugin):
  """Images Plugin for TensorBoard."""

  plugin_name = _PLUGIN_PREFIX_ROUTE

  def get_plugin_apps(self, multiplexer, unused_logdir):
    self._multiplexer = multiplexer
    return {
        '/images': self._serve_image_metadata,
        '/individualImage': self._serve_individual_image,
        '/tags': self._serve_tags,
    }

  def is_active(self):
    """The images plugin is active iff any run has at least one relevant tag."""
    return any(self.index_impl().values())

  def _index_impl(self):
    return {
        run_name: run_data[event_accumulator.IMAGES]
        for (run_name, run_data) in self._multiplexer.Runs().items()
        if event_accumulator.IMAGES in run_data
    }

  @wrappers.Request.application
  def _serve_image_metadata(self, request):
    """Given a tag and list of runs, serve a list of metadata for images.

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

  def _query_for_individual_image(self, run, tag, index):
    """Builds a URL for accessing the specified image.

    This should be kept in sync with _serve_image_metadata. Note that the URL is
    *not* guaranteed to always return the same image, since images may be
    unloaded from the reservoir as new images come in.

    Args:
      run: The name of the run.
      tag: The tag.
      index: The index of the image. Negative values are OK.

    Returns:
      A string representation of a URL that will load the index-th sampled image
      in the given run with the given tag.
    """
    query_string = urllib.parse.urlencode({
        'run': run,
        'tag': tag,
        'index': index
    })
    return query_string

  @wrappers.Request.application
  def _serve_individual_image(self, request):
    """Serves an individual image."""
    tag = request.args.get('tag')
    run = request.args.get('run')
    index = int(request.args.get('index'))
    image = self._multiplexer.Images(run, tag)[index]
    image_type = imghdr.what(None, image.encoded_image_string)
    content_type = _IMGHDR_TO_MIMETYPE.get(image_type, _DEFAULT_IMAGE_MIMETYPE)
    return http_util.Respond(request, image.encoded_image_string, content_type)

  @wrappers.Request.application
  def _serve_tags(self, request):
    index = self._index_impl()
    return http_util.Respond(request, index, 'application/json')
