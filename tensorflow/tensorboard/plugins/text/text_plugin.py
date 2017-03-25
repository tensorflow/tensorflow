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
from werkzeug import wrappers

from tensorflow.python.summary import text_summary
from tensorflow.tensorboard.backend import http_util
from tensorflow.tensorboard.plugins import base_plugin

# The prefix of routes provided by this plugin.
PLUGIN_PREFIX_ROUTE = 'text'

# HTTP routes
RUNS_ROUTE = '/index'
TEXT_ROUTE = '/text'


def process_string_tensor_event(event):
  return {
      'wall_time': event.wall_time,
      'step': event.step,
      'text': event.tensor_proto.string_val[0],
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
