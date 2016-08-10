# pylint: disable=g-bad-file-header
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
"""A wrapper of Session API which runs monitors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.learn.python.learn.wrapped_session import WrappedSession
from tensorflow.python.framework import ops


class MonitoredSession(WrappedSession):
  """A WrappedSession that calls monitors during calls to run().

  The list of monitors to call is passed in the constructor.  Before each call
  to `run()` the session calls the `step_begin()` method of the monitors, which
  can return additional ops or tensors to run.  These are added to the arguments
  of the call to `run()`.

  When the `run()` call finishes, the session calls the `step_end()` methods of
  the monitors, passing the values returned by the `run()` call corresponding to
  the ops and tensors that each monitor requested.

  If any call to the `step_end()` methods returns `True` the session will be
  marked as needing to stop and its `should_stop()` method will now return
  `True`.

  This wrapped session requires a "global step" tensor on construction.  This
  should return a scalar int value.  It is added to the list of tensors to fetch
  in calls to `run()`
  """

  def __init__(self, sess, monitors, global_step_tensor):
    """Initializes a MonitoredSession object.

    Args:
      sess: A `tf.Session` or a `WrappedSession` object.
      monitors: An iterable of `tf.contrib.learn.BaseMonitor' objects.
      global_step_tensor: A 'Tensor' which holds a scalar int value.
    """

    WrappedSession.__init__(self, sess)
    self._monitors = monitors
    self._should_stop = False
    self._global_step_tensor = global_step_tensor
    self._last_step = None

  def _check_stop(self):
    """See base class."""
    return self._should_stop

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    """See base class."""
    if self.should_stop():
      raise RuntimeError('Run called even after should_stop requested.')

    if self._last_step is None:
      self._last_step = WrappedSession.run(self, self._global_step_tensor)

    monitors_step = self._last_step + 1
    monitor_fetches = []
    for monitor in self._monitors:
      monitor_requests = monitor.step_begin(monitors_step)
      if monitor_requests:
        # TODO(ispir): remove following restriction after b/30136815 fixed
        if not isinstance(monitor_requests, list):
          raise ValueError('Monitor.step_begin should return a list.')
        monitor_fetches.extend(monitor_requests)

    actual_fetches = {
        'caller': fetches,
        self._global_step_tensor: self._global_step_tensor,
        'monitors': [_as_graph_element(f, self.graph) for f in monitor_fetches]
    }

    # Do session run.
    outputs = WrappedSession.run(self,
                                 fetches=actual_fetches,
                                 feed_dict=feed_dict,
                                 options=options,
                                 run_metadata=run_metadata)
    self._last_step = outputs[self._global_step_tensor]

    # Call monitors step_end and stop if one of them tells to stop.
    if monitor_fetches:
      monitor_outputs = dict(zip(monitor_fetches, outputs['monitors']))
    else:
      monitor_outputs = {}

    for monitor in self._monitors:
      induce_stop = monitor.step_end(monitors_step, monitor_outputs)
      self._should_stop = self._should_stop or induce_stop

    # Call the post_step methods.
    for monitor in self._monitors:
      monitor.post_step(monitors_step, self._sess)

    return outputs['caller']


# TODO(ispir): Remove following logic after forcing monitors returns tensors.
def _as_graph_element(obj, graph):
  """Retrieves Graph element."""
  graph = graph or ops.get_default_graph()
  if not isinstance(obj, six.string_types):
    if not hasattr(obj, 'graph') or obj.graph != graph:
      raise ValueError('Passed %s should have graph attribute that is equal '
                       'to current graph %s.' % (obj, graph))
    return obj
  if ':' in obj:
    element = graph.as_graph_element(obj)
  else:
    element = graph.as_graph_element(obj + ':0')
    # Check that there is no :1 (e.g. it's single output).
    try:
      graph.as_graph_element(obj + ':1')
    except (KeyError, ValueError):
      pass
    else:
      raise ValueError('Name %s is ambiguous, '
                       'as this `Operation` has multiple outputs '
                       '(at least 2).' % obj)
  return element
