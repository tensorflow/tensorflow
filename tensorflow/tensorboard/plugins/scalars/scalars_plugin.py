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
"""The TensorBoard Scalars plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

from six import StringIO
from werkzeug import wrappers

from tensorflow.tensorboard.backend import http_util
from tensorflow.tensorboard.backend.event_processing import event_accumulator
from tensorflow.tensorboard.plugins import base_plugin

_PLUGIN_PREFIX_ROUTE = event_accumulator.SCALARS


class OutputFormat(object):
  """An enum used to list the valid output formats for API calls."""
  JSON = 'json'
  CSV = 'csv'


class ScalarsPlugin(base_plugin.TBPlugin):
  """Scalars Plugin for TensorBoard."""

  plugin_name = _PLUGIN_PREFIX_ROUTE

  def get_plugin_apps(self, multiplexer, unused_logdir):
    self._multiplexer = multiplexer
    return {
        '/scalars': self.scalars_route,
        '/tags': self.tags_route,
    }

  def is_active(self):
    """The scalars plugin is active iff any run has at least one scalar tag."""
    return any(self.index_impl().values())

  def index_impl(self):
    return {
        run_name: run_data[event_accumulator.SCALARS]
        for (run_name, run_data) in self._multiplexer.Runs().items()
        if event_accumulator.SCALARS in run_data
    }

  def scalars_impl(self, tag, run, output_format):
    """Result of the form `(body, mime_type)`."""
    values = self._multiplexer.Scalars(run, tag)
    if output_format == OutputFormat.CSV:
      string_io = StringIO()
      writer = csv.writer(string_io)
      writer.writerow(['Wall time', 'Step', 'Value'])
      writer.writerows(values)
      return (string_io.getvalue(), 'text/csv')
    else:
      return (values, 'application/json')

  @wrappers.Request.application
  def tags_route(self, request):
    index = self.index_impl()
    return http_util.Respond(request, index, 'application/json')

  @wrappers.Request.application
  def scalars_route(self, request):
    """Given a tag and single run, return array of ScalarEvents."""
    # TODO(cassandrax): return HTTP status code for malformed requests
    tag = request.args.get('tag')
    run = request.args.get('run')
    output_format = request.args.get('format')
    (body, mime_type) = self.scalars_impl(tag, run, output_format)
    return http_util.Respond(request, body, mime_type)
