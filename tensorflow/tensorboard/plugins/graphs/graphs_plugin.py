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
"""The TensorBoard Graphs plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from werkzeug import wrappers

from tensorflow.tensorboard.backend import http_util
from tensorflow.tensorboard.backend import process_graph
from tensorflow.tensorboard.backend.event_processing import event_accumulator
from tensorflow.tensorboard.plugins import base_plugin

_PLUGIN_PREFIX_ROUTE = 'graphs'


class GraphsPlugin(base_plugin.TBPlugin):
  """Graphs Plugin for TensorBoard."""

  plugin_name = _PLUGIN_PREFIX_ROUTE

  def get_plugin_apps(self, multiplexer, unused_logdir):
    self._multiplexer = multiplexer
    return {
        '/graph': self.graph_route,
        '/runs': self.runs_route,
        '/run_metadata': self.run_metadata_route,
        '/run_metadata_tags': self.run_metadata_tags_route,
    }

  def is_active(self):
    """The graphs plugin is active iff any run has a graph."""
    return bool(self.index_impl())

  def index_impl(self):
    """Returns a list of all runs that have a graph."""
    return [run_name
            for (run_name, run_data) in self._multiplexer.Runs().items()
            if run_data.get(event_accumulator.GRAPH)]

  def run_metadata_index_impl(self):
    """Returns a run-to-tag mapping for metadata."""
    return {
        run_name: run_data[event_accumulator.RUN_METADATA]
        for (run_name, run_data) in self._multiplexer.Runs().items()
        if event_accumulator.RUN_METADATA in run_data
    }

  def graph_impl(self, run, limit_attr_size=None, large_attrs_key=None):
    """Result of the form `(body, mime_type)`, or `None` if no graph exists."""
    try:
      graph = self._multiplexer.Graph(run)
    except ValueError:
      return None
    # This next line might raise a ValueError if the limit parameters
    # are invalid (size is negative, size present but key absent, etc.).
    process_graph.prepare_graph_for_ui(graph, limit_attr_size, large_attrs_key)
    return (str(graph), 'text/x-protobuf')  # pbtxt

  def run_metadata_impl(self, run, tag):
    """Result of the form `(body, mime_type)`, or `None` if no data exists."""
    try:
      run_metadata = self._multiplexer.RunMetadata(run, tag)
    except ValueError:
      return None
    return (str(run_metadata), 'text/x-protobuf')  # pbtxt

  @wrappers.Request.application
  def runs_route(self, request):
    index = self.index_impl()
    return http_util.Respond(request, index, 'application/json')

  @wrappers.Request.application
  def run_metadata_tags_route(self, request):
    index = self.run_metadata_index_impl()
    return http_util.Respond(request, index, 'application/json')

  @wrappers.Request.application
  def graph_route(self, request):
    """Given a single run, return the graph definition in protobuf format."""
    run = request.args.get('run')
    if run is None:
      return http_util.Respond(
          request, 'query parameter "run" is required', 'text/plain', 400)

    limit_attr_size = request.args.get('limit_attr_size', None)
    if limit_attr_size is not None:
      try:
        limit_attr_size = int(limit_attr_size)
      except ValueError:
        return http_util.Respond(
            request, 'query parameter `limit_attr_size` must be an integer',
            'text/plain', 400)

    large_attrs_key = request.args.get('large_attrs_key', None)

    try:
      result = self.graph_impl(run, limit_attr_size, large_attrs_key)
    except ValueError as e:
      return http_util.Respond(request, e.message, 'text/plain', code=400)
    else:
      if result is not None:
        (body, mime_type) = result  # pylint: disable=unpacking-non-sequence
        return http_util.Respond(request, body, mime_type)
      else:
        return http_util.Respond(request, '404 Not Found', 'text/plain',
                                 code=404)

  @wrappers.Request.application
  def run_metadata_route(self, request):
    """Given a tag and a run, return the session.run() metadata."""
    tag = request.args.get('tag')
    run = request.args.get('run')
    if tag is None:
      return http_util.Respond(
          request, 'query parameter "tag" is required', 'text/plain', 400)
    if run is None:
      return http_util.Respond(
          request, 'query parameter "run" is required', 'text/plain', 400)
    result = self.run_metadata_impl(run, tag)
    if result is not None:
      (body, mime_type) = result  # pylint: disable=unpacking-non-sequence
      return http_util.Respond(request, body, mime_type)
    else:
      return http_util.Respond(request, '404 Not Found', 'text/plain',
                               code=404)
