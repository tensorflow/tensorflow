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
"""Tests the Tensorboard debugger data plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import shutil

import numpy as np
from werkzeug import test as werkzeug_test
from werkzeug import wrappers

from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import test
from tensorflow.python.util import compat
from tensorflow.tensorboard.backend import application
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
from tensorflow.tensorboard.plugins.debugger import debugger_plugin


class DebuggerPluginTest(test.TestCase):

  def setUp(self):
    # Populate the log directory with debugger event for run '.'.
    self.log_dir = self.get_temp_dir()
    file_prefix = compat.as_bytes(os.path.join(self.log_dir, 'events.debugger'))
    writer = pywrap_tensorflow.EventsWriter(file_prefix)
    writer.WriteEvent(
        self._CreateEventWithDebugNumericSummary(
            op_name='layers/Matmul',
            output_slot=0,
            wall_time=42,
            step=2,
            list_of_values=[1, 2, 3]))
    writer.WriteEvent(
        self._CreateEventWithDebugNumericSummary(
            op_name='layers/Matmul',
            output_slot=1,
            wall_time=43,
            step=7,
            list_of_values=[4, 5, 6]))
    writer.WriteEvent(
        self._CreateEventWithDebugNumericSummary(
            op_name='logits/Add',
            output_slot=0,
            wall_time=1337,
            step=7,
            list_of_values=[7, 8, 9]))
    writer.WriteEvent(
        self._CreateEventWithDebugNumericSummary(
            op_name='logits/Add',
            output_slot=0,
            wall_time=1338,
            step=8,
            list_of_values=[10, 11, 12]))
    writer.Close()

    # Populate the log directory with debugger event for run 'run_foo'.
    run_foo_directory = os.path.join(self.log_dir, 'run_foo')
    os.mkdir(run_foo_directory)
    file_prefix = compat.as_bytes(
        os.path.join(run_foo_directory, 'events.debugger'))
    writer = pywrap_tensorflow.EventsWriter(file_prefix)
    writer.WriteEvent(
        self._CreateEventWithDebugNumericSummary(
            op_name='layers/Variable',
            output_slot=0,
            wall_time=4242,
            step=42,
            list_of_values=[13, 14, 15]))
    writer.Close()

    # Start a server that will receive requests and respond with health pills.
    self.multiplexer = event_multiplexer.EventMultiplexer({
        '.': self.log_dir,
        'run_foo': run_foo_directory,
    })
    self.plugin = debugger_plugin.DebuggerPlugin()
    wsgi_app = application.TensorBoardWSGIApp(
        self.log_dir, {'debugger': self.plugin},
        self.multiplexer,
        reload_interval=0)
    self.server = werkzeug_test.Client(wsgi_app, wrappers.BaseResponse)

  def tearDown(self):
    # Remove the directory with debugger-related events files.
    shutil.rmtree(self.log_dir, ignore_errors=True)

  def _CreateEventWithDebugNumericSummary(
      self, op_name, output_slot, wall_time, step, list_of_values):
    """Creates event with a health pill summary.

    Args:
      op_name: The name of the op to which a DebugNumericSummary was attached.
      output_slot: The numeric output slot for the tensor.
      wall_time: The numeric wall time of the event.
      step: The step of the event.
      list_of_values: A python list of values within the tensor.

    Returns:
      A event_pb2.Event with a health pill summary.
    """
    event = event_pb2.Event(step=step, wall_time=wall_time)
    value = event.summary.value.add(
        tag='__health_pill__',
        node_name='%s:%d:DebugNumericSummary' % (op_name, output_slot))
    value.tensor.tensor_shape.dim.add(size=len(list_of_values))
    value.tensor.dtype = types_pb2.DT_DOUBLE
    value.tensor.tensor_content = np.array(
        list_of_values, dtype=np.float64).tobytes()
    return event

  def _DeserializeResponse(self, byte_content):
    """Deserializes byte content that is a JSON encoding.

    Args:
      byte_content: The byte content of a JSON response.

    Returns:
      The deserialized python object decoded from JSON.
    """
    return json.loads(byte_content.decode('utf-8'))

  def testHealthPillsRouteProvided(self):
    """Tests that the plugin offers the route for requesting health pills."""
    apps = self.plugin.get_plugin_apps(self.multiplexer, self.log_dir)
    self.assertIn('/health_pills', apps)
    self.assertIsInstance(apps['/health_pills'], collections.Callable)

  def testRequestHealthPillsForRunFoo(self):
    """Tests that the plugin produces health pills for a specified run."""
    response = self.server.post(
        '/data/plugin/debugger/health_pills',
        data={
            'node_names': json.dumps(['layers/Variable', 'unavailable_node']),
            'run': 'run_foo',
        })
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
    response = self.server.post(
        '/data/plugin/debugger/health_pills',
        data={
            'node_names': json.dumps(['logits/Add', 'unavailable_node']),
        })
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

  def testGetRequestsUnsupported(self):
    """Tests that GET requests are unsupported."""
    response = self.server.get('/data/plugin/debugger/health_pills')
    self.assertEqual(405, response.status_code)

  def testRequestsWithoutProperPostKeyUnsupported(self):
    """Tests that requests lacking the node_names POST key are unsupported."""
    response = self.server.post('/data/plugin/debugger/health_pills')
    self.assertEqual(400, response.status_code)

  def testRequestsWithBadJsonUnsupported(self):
    """Tests that requests with undecodable JSON are unsupported."""
    response = self.server.post(
        '/data/plugin/debugger/health_pills',
        data={
            'node_names': 'some obviously non JSON text',
        })
    self.assertEqual(400, response.status_code)

  def testRequestsWithNonListPostDataUnsupported(self):
    """Tests that requests with loads lacking lists of ops are unsupported."""
    response = self.server.post(
        '/data/plugin/debugger/health_pills',
        data={
            'node_names': json.dumps({
                'this is a dict': 'and not a list.'
            }),
        })
    self.assertEqual(400, response.status_code)

  def testFetchHealthPillsForSpecificStep(self):
    """Tests that requesting health pills at a specific steps works.

    This path may be slow in real life because it reads from disk.
    """
    # Request health pills for these nodes at step 7 specifically.
    response = self.server.post(
        '/data/plugin/debugger/health_pills',
        data={
            'node_names': json.dumps(['logits/Add', 'layers/Matmul']),
            'step': 7
        })
    self.assertEqual(200, response.status_code)
    # The response should only include health pills at step 7.
    self.assertDictEqual({
        'logits/Add': [
            {
                'wall_time': 1337,
                'step': 7,
                'node_name': 'logits/Add',
                'output_slot': 0,
                'value': [7, 8, 9],
            },
        ],
        'layers/Matmul': [
            {
                'wall_time': 43,
                'step': 7,
                'node_name': 'layers/Matmul',
                'output_slot': 1,
                'value': [4, 5, 6],
            },
        ],
    }, self._DeserializeResponse(response.get_data()))

  def testNoHealthPillsForSpecificStep(self):
    """Tests that an empty mapping is returned for no health pills at a step."""
    response = self.server.post(
        '/data/plugin/debugger/health_pills',
        data={
            'node_names': json.dumps(['some/clearly/non-existent/op']),
            'step': 7
        })
    self.assertEqual(200, response.status_code)
    self.assertDictEqual({}, self._DeserializeResponse(response.get_data()))

  def testNoHealthPillsForOutOfRangeStep(self):
    """Tests that an empty mapping is returned for an out of range step."""
    response = self.server.post(
        '/data/plugin/debugger/health_pills',
        data={
            'node_names': json.dumps(['logits/Add', 'layers/Matmul']),
            # This step higher than that of any event written to disk.
            'step': 42424242
        })
    self.assertEqual(200, response.status_code)
    self.assertDictEqual({}, self._DeserializeResponse(response.get_data()))

if __name__ == '__main__':
  test.main()
