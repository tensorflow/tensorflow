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
from tensorflow.python.summary import event_accumulator
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

    # http_util.Respond requires a headers property.
    self.headers = {}


class FakeEventMultiplexer(object):
  """A fake event multiplexer we can populate with custom health pills."""

  def __init__(self, run_to_node_name_to_health_pills):
    """Constructs a fake event multiplexer.

    Args:
      run_to_node_name_to_health_pills: A dict mapping run to a dict mapping
        node name to a list of health pills.
    """
    self._run_to_node_name_to_health_pills = run_to_node_name_to_health_pills

  def HealthPills(self, run, node_name):
    """Retrieve the health pill events associated with a run and node name.

    Args:
      run: A string name of the run for which health pills are retrieved.
      node_name: A string name of the node for which health pills are retrieved.

    Raises:
      KeyError: If the run is not found, or the node name is not available for
        the given run.

    Returns:
      An array of strings (that substitute for
      event_accumulator.HealthPillEvents) that represent health pills.
    """
    return self._run_to_node_name_to_health_pills[run][node_name]


class DebuggerPluginTest(test.TestCase):

  def setUp(self):
    self.fake_event_multiplexer = FakeEventMultiplexer({
        '.': {
            'layers/Matmul': [
                event_accumulator.HealthPillEvent(
                    wall_time=42,
                    step=2,
                    node_name='layers/Matmul',
                    output_slot=0,
                    value=[1, 2, 3]),
                event_accumulator.HealthPillEvent(
                    wall_time=43,
                    step=3,
                    node_name='layers/Matmul',
                    output_slot=1,
                    value=[4, 5, 6]),
            ],
            'logits/Add': [
                event_accumulator.HealthPillEvent(
                    wall_time=1337,
                    step=7,
                    node_name='logits/Add',
                    output_slot=0,
                    value=[7, 8, 9]),
                event_accumulator.HealthPillEvent(
                    wall_time=1338,
                    step=8,
                    node_name='logits/Add',
                    output_slot=0,
                    value=[10, 11, 12]),
            ],
        },
        'run_foo': {
            'layers/Variable': [
                event_accumulator.HealthPillEvent(
                    wall_time=4242,
                    step=42,
                    node_name='layers/Variable',
                    output_slot=0,
                    value=[13, 14, 15]),
            ],
        },
    })
    self.debugger_plugin = debugger_plugin.DebuggerPlugin(
        self.fake_event_multiplexer)
    self.unused_run_paths = {}
    self.unused_logdir = '/logdir'

  def _DeserializeResponse(self, byte_content):
    """Deserializes byte content that is a JSON encoding.

    Args:
      byte_content: The byte content of a JSON response.

    Returns:
      The deserialized python object.
    """
    return json.loads(byte_content.decode('utf-8'))

  def testRequestHealthPillsForRunFoo(self):
    """Tests that the plugin produces health pills for a specified run."""
    request = FakeRequest('POST', {
        'node_names': json.dumps(['layers/Variable', 'unavailable_node']),
        'run': 'run_foo',
    })
    response = self.debugger_plugin._serve_health_pills_helper(request)
    self.assertEqual(200, response.status_code)
    self.assertDictEqual({
        'layers/Variable': [{
            'wall_time': 4242,
            'step': 42,
            'node_name': 'layers/Variable',
            'output_slot': 0,
            'value': [13, 14, 15],
        }],
    }, self._DeserializeResponse(response.get_data()))

  def testRequestHealthPillsForDefaultRun(self):
    """Tests that the plugin produces health pills for the default '.' run."""
    # Do not provide a 'run' parameter in POST data.
    request = FakeRequest('POST', {
        'node_names': json.dumps(['logits/Add', 'unavailable_node']),
    })
    response = self.debugger_plugin._serve_health_pills_helper(request)
    self.assertEqual(200, response.status_code)
    # The health pills for 'layers/Matmul' should not be included since the
    # request excluded that node name.
    self.assertDictEqual({
        'logits/Add': [
            {
                'wall_time': 1337,
                'step': 7,
                'node_name': 'logits/Add',
                'output_slot': 0,
                'value': [7, 8, 9],
            },
            {
                'wall_time': 1338,
                'step': 8,
                'node_name': 'logits/Add',
                'output_slot': 0,
                'value': [10, 11, 12],
            },
        ],
    }, self._DeserializeResponse(response.get_data()))

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
