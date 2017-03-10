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
"""tfdbg CLI as SessionRunHook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import stepper
from tensorflow.python.debug.wrappers import dumping_wrapper
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.training import session_run_hook


class LocalCLIDebugHook(session_run_hook.SessionRunHook,
                        local_cli_wrapper.LocalCLIDebugWrapperSession):
  """Command-line-interface debugger hook.

  Can be used as a monitor/hook for `tf.train.MonitoredSession`s and
  `tf.contrib.learn`'s `Estimator`s and `Experiment`s.
  """

  def __init__(self, ui_type="curses"):
    """Create a local debugger command-line interface (CLI) hook.

    Args:
      ui_type: (str) user-interface type.
    """

    self._ui_type = ui_type
    self._wrapper_initialized = False
    self._pending_tensor_filters = {}

  def add_tensor_filter(self, filter_name, tensor_filter):
    """Add a tensor filter.

    See doc of `LocalCLIDebugWrapperSession.add_tensor_filter()` for details.
    Override default behavior to accomodate the possibility of this method being
    called prior to the initialization of the underlying
    `LocalCLIDebugWrapperSession` object.

    Args:
      filter_name: See doc of `LocalCLIDebugWrapperSession.add_tensor_filter()`
        for details.
      tensor_filter: See doc of
        `LocalCLIDebugWrapperSession.add_tensor_filter()` for details.
    """

    if self._wrapper_initialized:
      local_cli_wrapper.LocalCLIDebugWrapperSession.add_tensor_filter(
          self, filter_name, tensor_filter)
    else:
      self._pending_tensor_filters[filter_name] = tensor_filter

  def begin(self):
    pass

  def before_run(self, run_context):
    if not self._wrapper_initialized:
      local_cli_wrapper.LocalCLIDebugWrapperSession.__init__(
          self, run_context.session, ui_type=self._ui_type)

      # Actually register tensor filters registered prior to the construction
      # of the underlying LocalCLIDebugWrapperSession object.
      for filter_name in self._pending_tensor_filters:
        local_cli_wrapper.LocalCLIDebugWrapperSession.add_tensor_filter(
            self, filter_name, self._pending_tensor_filters[filter_name])

      self._wrapper_initialized = True

    # Increment run call counter.
    self._run_call_count += 1

    # Adapt run_context to an instance of OnRunStartRequest for invoking
    # superclass on_run_start().
    on_run_start_request = framework.OnRunStartRequest(
        run_context.original_args.fetches, run_context.original_args.feed_dict,
        None, None, self._run_call_count)

    on_run_start_response = self.on_run_start(on_run_start_request)
    self._performed_action = on_run_start_response.action

    run_args = session_run_hook.SessionRunArgs(
        None, feed_dict=None, options=config_pb2.RunOptions())
    if self._performed_action == framework.OnRunStartAction.DEBUG_RUN:
      self._decorate_options_for_debug(run_args.options,
                                       run_context.session.graph)
    elif self._performed_action == framework.OnRunStartAction.INVOKE_STEPPER:
      # The _finalized property must be set to False so that the NodeStepper
      # can insert ops for retrieving TensorHandles.
      # pylint: disable=protected-access
      run_context.session.graph._finalized = False
      # pylint: enable=protected-access

      with stepper.NodeStepper(
          run_context.session,
          run_context.original_args.
          fetches,
          run_context.original_args.feed_dict) as node_stepper:
        self.invoke_node_stepper(
            node_stepper, restore_variable_values_on_exit=True)

    return run_args

  def after_run(self, run_context, run_values):
    # Adapt run_context and run_values to OnRunEndRequest and invoke superclass
    # on_run_end()
    on_run_end_request = framework.OnRunEndRequest(self._performed_action,
                                                   run_values.run_metadata)
    self.on_run_end(on_run_end_request)

  def _decorate_options_for_debug(self, options, graph):
    """Modify RunOptions.debug_options.debug_tensor_watch_opts for debugging.

    Args:
      options: (config_pb2.RunOptions) The RunOptions instance to be modified.
      graph: A TensorFlow Graph object.
    """

    debug_utils.watch_graph(
        options, graph, debug_urls=self._get_run_debug_urls())
    options.output_partition_graphs = True


class DumpingDebugHook(session_run_hook.SessionRunHook,
                       dumping_wrapper.DumpingDebugWrapperSession):
  """A debugger hook that dumps debug data to filesystem.

  Can be used as a monitor/hook for `tf.train.MonitoredSession`s and
  `tf.contrib.learn`'s `Estimator`s and `Experiment`s.
  """

  def __init__(self, session_root, watch_fn=None, log_usage=True):
    """Create a local debugger command-line interface (CLI) hook.

    Args:
      session_root: See doc of
        `dumping_wrapper.DumpingDebugWrapperSession.__init__`.
      watch_fn: See doc of
        `dumping_wrapper.DumpingDebugWrapperSession.__init__`.
      log_usage: (bool) Whether usage is to be logged.
    """

    self._session_root = session_root
    self._watch_fn = watch_fn
    self._log_usage = log_usage
    self._wrapper_initialized = False

  def begin(self):
    pass

  def before_run(self, run_context):
    if not self._wrapper_initialized:
      dumping_wrapper.DumpingDebugWrapperSession.__init__(
          self,
          run_context.session,
          self._session_root,
          watch_fn=self._watch_fn,
          log_usage=self._log_usage)
      self._wrapper_initialized = True

    self._run_call_count += 1

    (debug_urls, debug_ops, node_name_regex_whitelist,
     op_type_regex_whitelist) = self._prepare_run_watch_config(
         run_context.original_args.fetches, run_context.original_args.feed_dict)
    run_options = config_pb2.RunOptions()
    debug_utils.watch_graph(
        run_options,
        run_context.session.graph,
        debug_urls=debug_urls,
        debug_ops=debug_ops,
        node_name_regex_whitelist=node_name_regex_whitelist,
        op_type_regex_whitelist=op_type_regex_whitelist)

    run_args = session_run_hook.SessionRunArgs(
        None, feed_dict=None, options=run_options)
    return run_args

  def after_run(self, run_context, run_values):
    pass
