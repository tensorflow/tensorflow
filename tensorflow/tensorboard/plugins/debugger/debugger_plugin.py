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

import collections
import glob
import json
import os
import re

from werkzeug import wrappers

from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.tensorboard.backend import http_util
from tensorflow.tensorboard.backend.event_processing import event_accumulator
from tensorflow.tensorboard.backend.event_processing import event_file_loader
from tensorflow.tensorboard.plugins import base_plugin

# The prefix of routes provided by this plugin.
_PLUGIN_PREFIX_ROUTE = 'debugger'

# HTTP routes.
_HEALTH_PILLS_ROUTE = '/health_pills'

# The POST key of HEALTH_PILLS_ROUTE for a JSON list of node names.
_NODE_NAMES_POST_KEY = 'node_names'

# The POST key of HEALTH_PILLS_ROUTE for the run to retrieve health pills for.
_RUN_POST_KEY = 'run'

# The default run to retrieve health pills for.
_DEFAULT_RUN = '.'

# The POST key of HEALTH_PILLS_ROUTE for the specific step to retrieve health
# pills for.
_STEP_POST_KEY = 'step'

# A glob pattern for files containing debugger-related events.
_DEBUGGER_EVENTS_GLOB_PATTERN = 'events.debugger*'


class DebuggerPlugin(base_plugin.TBPlugin):
  """TensorFlow Debugger plugin. Receives requests for debugger-related data.

  That data could include health pills, which unveil the status of tensor
  values.
  """

  plugin_name = _PLUGIN_PREFIX_ROUTE

  def get_plugin_apps(self, multiplexer, logdir):
    """Obtains a mapping between routes and handlers. Stores the logdir.

    Args:
      multiplexer: The EventMultiplexer that provides TB data.
      logdir: The logdir string - the directory of events files.

    Returns:
      A mapping between routes and handlers (functions that respond to
      requests).
    """
    self._event_multiplexer = multiplexer
    self._logdir = logdir
    return {
        _HEALTH_PILLS_ROUTE: self._serve_health_pills_handler,
    }

  @wrappers.Request.application
  def _serve_health_pills_handler(self, request):
    """A (wrapped) werkzeug handler for serving health pills.

    Accepts POST requests and responds with health pills. The request accepts
    several POST parameters:

      node_names: (required string) A JSON-ified list of node names for which
          the client would like to request health pills.
      run: (optional string) The run to retrieve health pills for. Defaults to
          '.'. This data is sent via POST (not GET) since URL length is limited.
      step: (optional integer): The session run step for which to
          retrieve health pills. If provided, the handler reads the health pills
          of that step from disk (which is slow) and produces a response with
          only health pills at that step. If not provided, the handler returns a
          response with health pills at all steps sampled by the event
          multiplexer (the fast path). The motivation here is that, sometimes,
          one desires to examine health pills at a specific step (to say find
          the first step that causes a model to blow up with NaNs).
          get_plugin_apps must be called before this slower feature is used
          because that method passes the logdir (directory path) to this plugin.

    This handler responds with a JSON-ified object mapping from node names to a
    list (of size 1) of health pill event objects, each of which has these
    properties.

    {
        'wall_time': float,
        'step': int,
        'node_name': string,
        'output_slot': int,
        # A list of 12 floats that summarizes the elements of the tensor.
        'value': float[],
    }

    Node names for which there are no health pills to be found are excluded from
    the mapping.

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
          'The %r POST key was not found in the request for health pills.',
          _NODE_NAMES_POST_KEY)
      return wrappers.Response(status=400)

    jsonified_node_names = request.form[_NODE_NAMES_POST_KEY]
    try:
      node_names = json.loads(jsonified_node_names)
    except Exception as e:  # pylint: disable=broad-except
      # Different JSON libs raise different exceptions, so we just do a
      # catch-all here. This problem is complicated by how Tensorboard might be
      # run in many different environments, as it is open-source.
      logging.error('Could not decode node name JSON string %r: %s',
                    jsonified_node_names, e)
      return wrappers.Response(status=400)

    if not isinstance(node_names, list):
      logging.error('%r is not a JSON list of node names:',
                    jsonified_node_names)
      return wrappers.Response(status=400)

    run = request.form.get(_RUN_POST_KEY, _DEFAULT_RUN)
    step_string = request.form.get(_STEP_POST_KEY, None)
    if step_string is None:
      # Use all steps sampled by the event multiplexer (Relatively fast).
      mapping = self._obtain_sampled_health_pills(run, node_names)
    else:
      # Read disk to obtain the health pills for that step (Relatively slow).
      # Make sure that the directory for the run exists.
      # Determine the directory of events file to read.
      events_directory = self._logdir
      if run != _DEFAULT_RUN:
        # Use the directory for the specific run.
        events_directory = os.path.join(events_directory, run)

      step = int(step_string)
      try:
        mapping = self._obtain_health_pills_at_step(
            events_directory, node_names, step)
      except IOError as error:
        logging.error(
            'Error retrieving health pills for step %d: %s', step, error)
        return wrappers.Response(status=404)

    # Convert event_accumulator.HealthPillEvents to JSON-able dicts.
    jsonable_mapping = {}
    for node_name, events in mapping.items():
      jsonable_mapping[node_name] = [e._asdict() for e in events]
    return http_util.Respond(request, jsonable_mapping, 'application/json')

  def _obtain_sampled_health_pills(self, run, node_names):
    """Obtains the health pills for a run sampled by the event multiplexer.

    This is much faster than the alternative path of reading health pills from
    disk.

    Args:
      run: The run to fetch health pills for.
      node_names: A list of node names for which to retrieve health pills.

    Returns:
      A dictionary mapping from node name to a list of
      event_accumulator.HealthPillEvents.
    """
    mapping = {}
    for node_name in node_names:
      try:
        mapping[node_name] = self._event_multiplexer.HealthPills(run, node_name)
      except KeyError:
        logging.info('No health pills found for node %r.', node_name)
        continue

    return mapping

  def _obtain_health_pills_at_step(self, events_directory, node_names, step):
    """Reads disk to obtain the health pills for a run at a specific step.

    This could be much slower than the alternative path of just returning all
    health pills sampled by the event multiplexer. It could take tens of minutes
    to complete this call for large graphs for big step values (in the
    thousands).

    Args:
      events_directory: The directory containing events for the desired run.
      node_names: A list of node names for which to retrieve health pills.
      step: The step to obtain health pills for.

    Returns:
      A dictionary mapping from node name to a list of health pill objects (see
      docs for _serve_health_pills_handler for properties of those objects).

    Raises:
      IOError: If no files with health pill events could be found.
    """
    # Obtain all files with debugger-related events.
    pattern = os.path.join(events_directory, _DEBUGGER_EVENTS_GLOB_PATTERN)
    file_paths = glob.glob(pattern)

    if not file_paths:
      raise IOError(
          'No events files found that matches the pattern %r.', pattern)

    # Sort by name (and thus by timestamp).
    file_paths.sort()

    mapping = collections.defaultdict(list)
    node_name_set = frozenset(node_names)

    for file_path in file_paths:
      should_stop = self._process_health_pill_event(
          node_name_set, mapping, step, file_path)
      if should_stop:
        break

    return mapping

  def _process_health_pill_event(self, node_name_set, mapping, target_step,
                                 file_path):
    """Creates health pills out of data in an event.

    Creates health pills out of the event and adds them to the mapping.

    Args:
      node_name_set: A set of node names that are relevant.
      mapping: The mapping from node name to event_accumulator.HealthPillEvents.
          This object may be destructively modified.
      target_step: The target step at which to obtain health pills.
      file_path: The path to the file with health pill events.

    Returns:
      Whether we should stop reading events because future events are no longer
      relevant.
    """
    events_loader = event_file_loader.EventFileLoader(file_path)
    for event in events_loader.Load():
      if not event.HasField('summary'):
        logging.warning('An event in a debugger events file lacks a summary.')
        continue

      if event.step < target_step:
        # This event is not of the relevant step. We perform this check
        # first because the majority of events will be eliminated from
        # consideration by this check.
        continue

      if event.step > target_step:
        # We have passed the relevant step. No need to read more events.
        return True

      for value in event.summary.value:
        # Since we seek health pills for a specific step, this function
        # returns 1 health pill per node per step. The wall time is the
        # seconds since the epoch.
        health_pill = self._process_health_pill_value(
            node_name_set, event.wall_time, event.step, value)
        if not health_pill:
          continue
        mapping[health_pill.node_name].append(health_pill)

    # Keep reading events.
    return False

  def _process_health_pill_value(self, node_name_set, wall_time, step, value):
    """Creates a dict containing various properties of a health pill.

    Args:
      node_name_set: A set of node names that are relevant.
      wall_time: The wall time in seconds.
      step: The session run step of the event.
      value: The health pill value.

    Returns:
      An event_accumulator.HealthPillEvent. Or None if one could not be created.
    """
    if not value.HasField('tensor'):
      logging.warning(
          'An event in a debugger events file lacks a tensor value.')
      return None

    if value.tag != event_accumulator.HEALTH_PILL_EVENT_TAG:
      logging.warning(
          ('A debugger-related event lacks the %r tag. It instead has '
           'the %r tag.'), event_accumulator.HEALTH_PILL_EVENT_TAG, value.tag)
      return None

    match = re.match(r'^(.*):(\d+):DebugNumericSummary$', value.node_name)
    if not match:
      logging.warning(
          ('A event with a health pill has an invalid watch, (i.e., an '
           'unexpected debug op): %r'), value.node_name)
      return None

    node_name = match.group(1)
    if node_name not in node_name_set:
      # This event is not relevant.
      return None

    # Since we seek health pills for a specific step, this function
    # returns 1 health pill per node per step. The wall time is the
    # seconds since the epoch.
    return event_accumulator.HealthPillEvent(
        wall_time=wall_time,
        step=step,
        node_name=node_name,
        output_slot=int(match.group(2)),
        value=list(tensor_util.MakeNdarray(value.tensor)))
