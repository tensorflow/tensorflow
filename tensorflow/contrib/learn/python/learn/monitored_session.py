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

from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.learn.python.learn import session_run_hook
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

  def __init__(self, sess, hooks):
    """Initializes a MonitoredSession object.

    Args:
      sess: A `tf.Session` or a `WrappedSession` object.
      hooks: An iterable of `tf.contrib.learn.SessionRunHook' objects.
    """

    WrappedSession.__init__(self, sess)
    self._hooks = [m for m in hooks
                   if isinstance(m, session_run_hook.SessionRunHook)]
    self._monitors = [m for m in hooks
                      if not isinstance(m, session_run_hook.SessionRunHook)]
    self._should_stop = False
    self._monitors_init()

  def _check_stop(self):
    """See base class."""
    return self._should_stop

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    """See base class."""
    if self.should_stop():
      raise RuntimeError('Run called even after should_stop requested.')

    actual_fetches = {'caller': fetches}

    self._monitors_step_begin(actual_fetches)

    run_context = session_run_hook.SessionRunContext(
        original_args=session_run_hook.SessionRunArgs(fetches, feed_dict),
        session=self._sess)
    feed_dict = self._call_hook_before_run(
        run_context, actual_fetches, feed_dict)

    # Do session run.
    outputs = WrappedSession.run(self,
                                 fetches=actual_fetches,
                                 feed_dict=feed_dict,
                                 options=options,
                                 run_metadata=run_metadata)

    for hook in self._hooks:
      hook.after_run(
          run_context,
          session_run_hook.SessionRunValues(results=outputs[hook] if
                                            hook in outputs else None))
    self._should_stop = self._should_stop or run_context.stop_requested

    self._monitors_step_end(outputs)

    return outputs['caller']

  def _call_hook_before_run(self, run_context, fetch_dict, user_feed_dict):
    hook_feeds = {}
    for hook in self._hooks:
      request = hook.before_run(run_context)
      if request is not None:
        if request.fetches is not None:
          fetch_dict[hook] = request.fetches
        if request.feed_dict:
          self._raise_if_feeds_intersects(
              hook_feeds, request.feed_dict,
              'Same tensor is fed by two hooks.')
          hook_feeds.update(request.feed_dict)

    if hook_feeds:
      if user_feed_dict:
        self._raise_if_feeds_intersects(
            user_feed_dict, hook_feeds,
            'Same tensor is fed by a SessionRunHook and user.')
        hook_feeds.update(user_feed_dict)
        user_feed_dict = hook_feeds
      else:
        user_feed_dict = hook_feeds

    return user_feed_dict

  def _raise_if_feeds_intersects(self, feeds1, feeds2, message):
    intersection = set(feeds1.keys()) & set(feeds2.keys())
    if intersection:
      raise RuntimeError(message + ' Conflict(s): ' + str(list(intersection)))

  # TODO(ispir): Delete all of following functions after deprecating Monitors.
  def _monitors_init(self):
    if self._monitors:
      self._global_step_tensor = contrib_variables.get_global_step()
      self._last_step = None

  def _monitors_step_begin(self, actual_fetches):
    if self._monitors:
      actual_fetches[self._global_step_tensor] = self._global_step_tensor
      if self._last_step is None:
        self._last_step = WrappedSession.run(self, self._global_step_tensor)

      self.monitors_step = self._last_step + 1
      self.monitor_fetches = []
      for monitor in self._monitors:
        monitor_requests = monitor.step_begin(self.monitors_step)
        if monitor_requests:
          if not isinstance(monitor_requests, list):
            raise ValueError('Monitor.step_begin should return a list.')
          self.monitor_fetches.extend(monitor_requests)
      actual_fetches['monitors'] = [
          _as_graph_element(f, self.graph) for f in self.monitor_fetches
      ]

  def _monitors_step_end(self, outputs):
    if self._monitors:
      self._last_step = outputs[self._global_step_tensor]
      # Call monitors step_end and stop if one of them tells to stop.
      if self.monitor_fetches:
        monitor_outputs = dict(zip(self.monitor_fetches, outputs['monitors']))
      else:
        monitor_outputs = {}

      for monitor in self._monitors:
        induce_stop = monitor.step_end(self.monitors_step, monitor_outputs)
        self._should_stop = self._should_stop or induce_stop

      # Call the post_step methods.
      for monitor in self._monitors:
        monitor.post_step(self.monitors_step, self._sess)


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



