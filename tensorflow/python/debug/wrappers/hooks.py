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
from tensorflow.python.debug import debug_utils
from tensorflow.python.debug import stepper
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.training import session_run_hook


class LocalCLIDebugHook(session_run_hook.SessionRunHook,
                        local_cli_wrapper.LocalCLIDebugWrapperSession):
  """Command-line-interface debugger hook.

  Can be used as a monitor/hook for tf.train.MonitoredSession.
  """

  def __init__(self):
    """Create a local debugger command-line interface (CLI) hook."""

    self._wrapper_initialized = False

  def begin(self):
    pass

  def before_run(self, run_context):
    if not self._wrapper_initialized:
      local_cli_wrapper.LocalCLIDebugWrapperSession.__init__(
          self, run_context.session)
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

      self.invoke_node_stepper(
          stepper.NodeStepper(run_context.session, run_context.original_args.
                              fetches, run_context.original_args.feed_dict),
          restore_variable_values_on_exit=True)

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
