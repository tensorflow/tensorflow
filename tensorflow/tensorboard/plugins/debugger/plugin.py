# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""The plugin for serving data from a TensorFlow debugger."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from werkzeug import wrappers

from tensorflow.python.platform import tf_logging as logging
from tensorflow.tensorboard.lib.python import http_util
from tensorflow.tensorboard.plugins import base_plugin

# The prefix of routes provided by this plugin.
PLUGIN_PREFIX_ROUTE = 'debugger'

# HTTP routes.
_HEALTH_PILLS_ROUTE = '/health_pills'

# The POST key value of the HEALTH_PILLS_ROUTE for a JSON list of node names.
_NODE_NAMES_POST_KEY = 'node_names'


class DebuggerPlugin(base_plugin.TBPlugin):
  """TensorFlow Debugger plugin. Receives requests for debugger-related data.

  That data could include health pills, which unveil the status of tensor
  values.
  """

  def get_plugin_apps(self, unused_run_paths, unused_logdir):
    """Obtains a mapping between routes and handlers.

    Args:
      unused_run_paths: A mapping between run paths and handlers.
      unused_logdir: The logdir string - the directory of events files.

    Returns:
      A mapping between routes and handlers (functions that respond to
      requests).
    """
    return {
        _HEALTH_PILLS_ROUTE: self._serve_health_pills_handler,
    }

  @wrappers.Request.application
  def _serve_health_pills_handler(self, request):
    """A (wrapped) werkzeug handler for serving health pills.

    NOTE(chizeng): This handler is currently not useful and does not behave as
    expected. It currently merely responds with the provided list of op names
    (instead of the health pills for those ops). Very soon, that will change.

    We defer to another method for actually performing the main logic because
    the @wrappers.Request.application decorator makes this logic hard to access
    in tests.

    Args:
      request: The request issued by the client for health pills.

    Returns:
      A werkzeug BaseResponse object.
    """
    return self._serve_health_pills_helper(request)

  def _serve_health_pills_helper(self, request):
    """Responds with health pills.

    Accepts POST requests and responds with health pills. Specifically, the
    handler expects a "node_names" POST data key. The value of that key should
    be a JSON-ified list of node names for which the client would like to
    request health pills. This data is sent via POST instead of GET because URL
    length is limited.

    This handler responds with a JSON-ified object mapping from node names to a
    list of HealthPillEvents. Node names for which there are no health pills to
    be found are excluded from the mapping.

    Args:
      request: The request issued by the client for health pills.

    Returns:
      A werkzeug BaseResponse object.
    """
    if request.method != 'POST':
      logging.error(
          '%s requests are forbidden by the debugger plugin.', request.method)
      return wrappers.Response(status=405)

    if _NODE_NAMES_POST_KEY not in request.form:
      logging.error(
          'The %s POST key was not found in the request for health pills.',
          _NODE_NAMES_POST_KEY)
      return wrappers.Response(status=400)

    jsonified_node_names = request.form[_NODE_NAMES_POST_KEY]
    try:
      node_names = json.loads(jsonified_node_names)
    except Exception as e:  # pylint: disable=broad-except
      # Different JSON libs raise different exceptions, so we just do a
      # catch-all here. This problem is complicated by how Tensorboard might be
      # run in many different environments, as it is open-source.
      logging.error(
          'Could not decode node name JSON string %s: %s',
          jsonified_node_names, e)
      return wrappers.Response(status=400)

    if not isinstance(node_names, list):
      logging.error(
          '%s is not a JSON list of node names:', jsonified_node_names)
      return wrappers.Response(status=400)

    # TODO(chizeng): Actually respond with the health pills per node name.
    return http_util.Respond(request, node_names, mimetype='application/json')
