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
"""Tests for the Tensorboard debugger data plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json

from tensorflow.python.platform import test
from tensorflow.tensorboard.plugins.debugger import plugin as debugger_plugin


class FakeRequest(object):
  """A fake shell of a werkzeug request.

  We fake instead of using a real request because the real request requires a
  WSGI environment.
  """

  def __init__(self, method, post_data):
    """Constructs a fake request, a simple version of a werkzeug request.

    Args:
      method: The uppercase method of the request, ie POST.
      post_data: A dictionary of POST data.
    """
    self.method = method
    self.form = post_data


class DebuggerPluginTest(test.TestCase):

  def setUp(self):
    self.debugger_plugin = debugger_plugin.DebuggerPlugin()
    self.unused_run_paths = {}
    self.unused_logdir = '/logdir'

  def testHealthPillsRouteProvided(self):
    """Tests that the plugin offers the route for requesting health pills."""
    apps = self.debugger_plugin.get_plugin_apps(self.unused_run_paths,
                                                self.unused_logdir)
    self.assertIn('/health_pills', apps)
    self.assertIsInstance(apps['/health_pills'], collections.Callable)

  def testGetRequestsUnsupported(self):
    """Tests that GET requests are unsupported."""
    request = FakeRequest('GET', {
        'node_names': json.dumps(['layers/Matmul', 'logits/Add']),
    })
    self.assertEqual(
        405,
        self.debugger_plugin._serve_health_pills_helper(request).status_code)

  def testRequestsWithoutProperPostKeyUnsupported(self):
    """Tests that requests lacking the node_names POST key are unsupported."""
    request = FakeRequest('POST', {})
    self.assertEqual(
        400,
        self.debugger_plugin._serve_health_pills_helper(request).status_code)

  def testRequestsWithBadJsonUnsupported(self):
    """Tests that requests with undecodable JSON are unsupported."""
    request = FakeRequest('POST',
                          {'node_names': 'some obviously non JSON text',})
    self.assertEqual(
        400,
        self.debugger_plugin._serve_health_pills_helper(request).status_code)

  def testRequestsWithNonListPostDataUnsupported(self):
    """Tests that requests with loads lacking lists of ops are unsupported."""
    request = FakeRequest('POST', {
        'node_names': json.dumps({
            'this is a dict': 'and not a list.'
        }),
    })
    self.assertEqual(
        400,
        self.debugger_plugin._serve_health_pills_helper(request).status_code)


if __name__ == '__main__':
  test.main()
