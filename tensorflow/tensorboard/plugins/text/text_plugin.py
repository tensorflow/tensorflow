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
"""The TensorBoard Text plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import bleach
# pylint: disable=g-bad-import-order
# Google-only: import markdown_freewisdom
import markdown
# pylint: enable=g-bad-import-order
from werkzeug import wrappers

from tensorflow.python.summary import text_summary
from tensorflow.tensorboard.backend import http_util
from tensorflow.tensorboard.plugins import base_plugin

# The prefix of routes provided by this plugin.
PLUGIN_PREFIX_ROUTE = 'text'

# HTTP routes
RUNS_ROUTE = '/index'
TEXT_ROUTE = '/text'

ALLOWED_TAGS = [
    'ul',
    'ol',
    'li',
    'p',
    'pre',
    'code',
    'blockquote',
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'hr',
    'br',
    'strong',
    'em',
    'a',
    'img',
    'table',
    'thead',
    'tbody',
    'td',
    'tr',
    'th',
]

ALLOWED_ATTRIBUTES = {'a': ['href', 'title'], 'img': ['src', 'title', 'alt']}


def markdown_and_sanitize(markdown_string):
  """Takes a markdown string and converts it into sanitized html.

  It uses the table extension; while that's not a part of standard
  markdown, it is sure to be useful for TensorBoard users.

  The sanitizer uses the allowed_tags and attributes specified above. Mostly,
  we ensure that our standard use cases like tables and links are supported.

  Args:
    markdown_string: Markdown string to sanitize

  Returns:
    a string containing sanitized html for input markdown
  """
  # Convert to utf-8 because we get a bytearray in python3
  if not isinstance(markdown_string, str):
    markdown_string = markdown_string.decode('utf-8')
  string_html = markdown.markdown(
      markdown_string, extensions=['markdown.extensions.tables'])
  string_sanitized = bleach.clean(
      string_html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES)
  return string_sanitized


def process_string_tensor_event(event):
  """Convert a TensorEvent into a JSON-compatible response."""
  return {
      'wall_time': event.wall_time,
      'step': event.step,
      'text': markdown_and_sanitize(event.tensor_proto.string_val[0]),
  }


class TextPlugin(base_plugin.TBPlugin):
  """Text Plugin for TensorBoard."""

  def index_impl(self):
    run_to_series = {}
    name = text_summary.TextSummaryPluginAsset.plugin_name
    run_to_assets = self.multiplexer.PluginAssets(name)

    for run, assets in run_to_assets.items():
      if 'tensors.json' in assets:
        tensors_json = self.multiplexer.RetrievePluginAsset(
            run, name, 'tensors.json')
        tensors = json.loads(tensors_json)
        run_to_series[run] = tensors
    return run_to_series

  @wrappers.Request.application
  def runs_route(self, request):
    index = self.index_impl()
    return http_util.Respond(request, index, 'application/json')

  def text_impl(self, run, tag):
    try:
      text_events = self.multiplexer.Tensors(run, tag)
    except KeyError:
      text_events = []
    responses = [process_string_tensor_event(ev) for ev in text_events]
    return responses

  @wrappers.Request.application
  def text_route(self, request):
    run = request.args.get('run')
    tag = request.args.get('tag')
    response = self.text_impl(run, tag)
    return http_util.Respond(request, response, 'application/json')

  def get_plugin_apps(self, multiplexer, unused_logdir):
    self.multiplexer = multiplexer
    return {
        RUNS_ROUTE: self.runs_route,
        TEXT_ROUTE: self.text_route,
    }
