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
from tensorflow.python.debug.wrappers import grpc_wrapper
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.training import session_run_hook

# The prefix for GRPC endpoint URLs.
_GRPC_ENDPOINT_PREFIX = "grpc://"


class LocalCLIDebugHook(session_run_hook.SessionRunHook,
                        local_cli_wrapper.LocalCLIDebugWrapperSession):
  """Command-line-interface debugger hook.

  Can be used as a monitor/hook for `tf.train.MonitoredSession`s and
  `tf.contrib.learn`'s `Estimator`s and `Experiment`s.
  """

  def __init__(self, ui_type="curses", dump_root=None):
    """Create a local debugger command-line interface (CLI) hook.

    Args:
      ui_type: (str) user-interface type.
      dump_root: (`str`) optional path to the dump root directory. Must be a
        directory that does not exist or an empty directory. If the directory
        does not exist, it will be created by the debugger core during debug
        `run()` calls and removed afterwards.
    """

    self._ui_type = ui_type
    self._dump_root = dump_root
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
          self,
          run_context.session,
          ui_type=self._ui_type,
          dump_root=self._dump_root)

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
      # TODO(cais): Make this hook have a DumpingDebugWrapperSession property
      # instead of subclassing DumpingDebugWrapperSession.
      dumping_wrapper.DumpingDebugWrapperSession.__init__(
          self,
          run_context.session,
          self._session_root,
          watch_fn=self._watch_fn,
          log_usage=self._log_usage)
      self._wrapper_initialized = True

    self._run_call_count += 1

    debug_urls, watch_options = self._prepare_run_watch_config(
        run_context.original_args.fetches, run_context.original_args.feed_dict)
    run_options = config_pb2.RunOptions()
    debug_utils.watch_graph(
        run_options,
        run_context.session.graph,
        debug_urls=debug_urls,
        debug_ops=watch_options.debug_ops,
        node_name_regex_whitelist=watch_options.node_name_regex_whitelist,
        op_type_regex_whitelist=watch_options.op_type_regex_whitelist,
        tensor_dtype_regex_whitelist=watch_options.tensor_dtype_regex_whitelist,
        tolerate_debug_op_creation_failures=(
            watch_options.tolerate_debug_op_creation_failures))

    run_args = session_run_hook.SessionRunArgs(
        None, feed_dict=None, options=run_options)
    return run_args

  def after_run(self, run_context, run_values):
    pass


class GrpcDebugHook(session_run_hook.SessionRunHook):
  """A hook that streams debugger-related events to any grpc_debug_server.

  For example, the debugger data server is a grpc_debug_server. The debugger
  data server writes debugger-related events it receives via GRPC to logdir.
  This enables debugging features in Tensorboard such as health pills.

  When the arguments of debug_utils.watch_graph changes, strongly consider
  changing arguments here too so that features are available to tflearn users.

  Can be used as a monitor/hook for `tf.train.MonitoredSession`s and
  `tf.contrib.learn`'s `Estimator`s and `Experiment`s.
  """

  def __init__(self,
               grpc_debug_server_addresses,
               watch_fn=None,
               log_usage=True):
    """Constructs a GrpcDebugHook.

    Args:
      grpc_debug_server_addresses: (`list` of `str`) A list of the gRPC debug
        server addresses, in the format of <host:port>, without the "grpc://"
        prefix. For example: ["localhost:7000", "192.168.0.2:8000"]
      watch_fn: A function that allows for customizing which ops to watch at
        which specific steps. See doc of
        `dumping_wrapper.DumpingDebugWrapperSession.__init__` for details.
      log_usage: (bool) Whether usage is to be logged.

    Raises:
      ValueError: if any debugger server addresses start with grpc://.
    """

    for address in grpc_debug_server_addresses:
      if address.startswith(_GRPC_ENDPOINT_PREFIX):
        raise ValueError(
            ("Debug server address %r starts with %r. It should not because "
             "the hook already automatically adds the prefix.") % (
                 address, _GRPC_ENDPOINT_PREFIX))

    # A wrapper session responsible for GRPC communication.
    self._grpc_debug_wrapper_session = None

    self._grpc_debug_server_addresses = grpc_debug_server_addresses
    self._watch_fn = watch_fn
    self._log_usage = log_usage

  def before_run(self, run_context):
    """Called right before a session is run.

    Args:
      run_context: A session_run_hook.SessionRunContext. Encapsulates
        information on the run.

    Returns:
      A session_run_hook.SessionRunArgs object.
    """

    if not self._grpc_debug_wrapper_session:
      self._grpc_debug_wrapper_session = grpc_wrapper.GrpcDebugWrapperSession(
          run_context.session,
          self._grpc_debug_server_addresses,
          watch_fn=self._watch_fn,
          log_usage=self._log_usage)

    fetches = run_context.original_args.fetches
    feed_dict = run_context.original_args.feed_dict
    watch_options = self._watch_fn(fetches, feed_dict)
    run_options = config_pb2.RunOptions()
    debug_utils.watch_graph(
        run_options,
        run_context.session.graph,
        debug_urls=self._grpc_debug_wrapper_session.prepare_run_debug_urls(
            fetches, feed_dict),
        debug_ops=watch_options.debug_ops,
        node_name_regex_whitelist=watch_options.node_name_regex_whitelist,
        op_type_regex_whitelist=watch_options.op_type_regex_whitelist,
        tensor_dtype_regex_whitelist=watch_options.tensor_dtype_regex_whitelist,
        tolerate_debug_op_creation_failures=(
            watch_options.tolerate_debug_op_creation_failures))

    return session_run_hook.SessionRunArgs(
        None, feed_dict=None, options=run_options)
