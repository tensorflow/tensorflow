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
"""Debugger Wrapper Session Consisting of a Local Curses-based CLI."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys
import tempfile

# Google-internal import(s).
from tensorflow.python.debug import debug_data
from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import curses_ui
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import stepper_cli
from tensorflow.python.debug.wrappers import framework


_DUMP_ROOT_PREFIX = "tfdbg_"


class LocalCLIDebugWrapperSession(framework.BaseDebugWrapperSession):
  """Concrete subclass of BaseDebugWrapperSession implementing a local CLI."""

  def __init__(self, sess, dump_root=None, log_usage=True):
    """Constructor of LocalCLIDebugWrapperSession.

    Args:
      sess: (BaseSession subtypes) The TensorFlow Session object being wrapped.
      dump_root: (str) Optional path to the dump root directory. Must be either
        a directory that does not exist or an empty directory. If the directory
        does not exist, it will be created by the debugger core during debug
        run() calls and removed afterwards.
      log_usage: (bool) Whether the usage of this class is to be logged.

    Raises:
      ValueError: If dump_root is an existing and non-empty directory or if
        dump_root is a file.
    """

    if log_usage:
      pass  # No logging for open-source.

    framework.BaseDebugWrapperSession.__init__(self, sess)

    if dump_root is None:
      self._dump_root = tempfile.mktemp(prefix=_DUMP_ROOT_PREFIX)
    else:
      if os.path.isfile(dump_root):
        raise ValueError("dump_root path points to a file: %s" % dump_root)
      elif os.path.isdir(dump_root) and os.listdir(dump_root):
        raise ValueError("dump_root path points to a non-empty directory: %s" %
                         dump_root)

      self._dump_root = dump_root

    self._initialize_argparsers()

    # Registered tensor filters.
    self._tensor_filters = {}

    # Below are the state variables of this wrapper object.
    # _active_tensor_filter: what (if any) tensor filter is in effect. If such
    #   a filter is in effect, this object will call run() method of the
    #   underlying TensorFlow Session object until the filter passes. This is
    #   activated by the "-f" flag of the "run" command.
    # _run_through_times: keeps track of how many times the wrapper needs to
    #   run through without stopping at the run-end CLI. It is activated by the
    #   "-t" option of the "run" command.
    # _skip_debug: keeps track of whether the current run should be executed
    #   without debugging. It is activated by the "-n" option of the "run"
    #   command.
    #
    # _run_start_response: keeps track what OnRunStartResponse the wrapper
    #   should return at the next run-start callback. If this information is
    #   unavailable (i.e., is None), the run-start CLI will be launched to ask
    #   the user. This is the case, e.g., right before the first run starts.
    self._active_tensor_filter = None
    self._run_through_times = 1
    self._skip_debug = False
    self._run_start_response = None

  def _initialize_argparsers(self):
    self._argparsers = {}
    ap = argparse.ArgumentParser(
        description="Run through, with or without debug tensor watching.",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "-t",
        "--times",
        dest="times",
        type=int,
        default=1,
        help="How many Session.run() calls to proceed with.")
    ap.add_argument(
        "-n",
        "--no_debug",
        dest="no_debug",
        action="store_true",
        help="Run through without debug tensor watching.")
    ap.add_argument(
        "-f",
        "--till_filter_pass",
        dest="till_filter_pass",
        type=str,
        default="",
        help="Run until a tensor in the graph passes the specified filter.")
    self._argparsers["run"] = ap

    ap = argparse.ArgumentParser(
        description="Invoke stepper (cont, step, breakpoint, etc.)",
        usage=argparse.SUPPRESS)
    self._argparsers["invoke_stepper"] = ap

    ap = argparse.ArgumentParser(
        description="Display information about this Session.run() call.",
        usage=argparse.SUPPRESS)
    self._argparsers["run_info"] = ap

  def add_tensor_filter(self, filter_name, tensor_filter):
    """Add a tensor filter.

    The signature of this command is identical to that of
    debug_data.DebugDumpDir.add_tensor_filter(). This method is a thin wrapper
    around that method.

    Args:
      filter_name: (str) Name of the filter.
      tensor_filter: (callable) The filter callable. See the doc string of
        debug_data.DebugDumpDir.add_tensor_filter() for more details.
    """

    self._tensor_filters[filter_name] = tensor_filter

  def on_session_init(self, request):
    """Overrides on-session-init callback.

    Args:
      request: An instance of OnSessionInitRequest.

    Returns:
      An instance of OnSessionInitResponse.
    """

    return framework.OnSessionInitResponse(
        framework.OnSessionInitAction.PROCEED)

  def on_run_start(self, request):
    """Overrides on-run-start callback.

    Invoke the CLI to let user choose what action to take:
      run / run --no_debug / step.

    Args:
      request: An instance of OnSessionInitRequest.

    Returns:
      An instance of OnSessionInitResponse.

    Raises:
      RuntimeError: If user chooses to prematurely exit the debugger.
    """

    self._update_run_calls_state(request.run_call_count, request.fetches,
                                 request.feed_dict)

    if self._active_tensor_filter:
      # If we are running till a filter passes, we just need to keep running
      # with the DEBUG_RUN option.
      return framework.OnRunStartResponse(framework.OnRunStartAction.DEBUG_RUN,
                                          self._get_run_debug_urls())

    if self._run_call_count > 1 and not self._skip_debug:
      if self._run_through_times > 0:
        # Just run through without debugging.
        return framework.OnRunStartResponse(
            framework.OnRunStartAction.NON_DEBUG_RUN, [])
      elif self._run_through_times == 0:
        # It is the run at which the run-end CLI will be launched: activate
        # debugging.
        return framework.OnRunStartResponse(
            framework.OnRunStartAction.DEBUG_RUN,
            self._get_run_debug_urls())

    if self._run_start_response is None:
      self._prep_cli_for_run_start()

      self._run_start_response = self._launch_cli(is_run_start=True)
      if self._run_through_times > 1:
        self._run_through_times -= 1

    if self._run_start_response == debugger_cli_common.EXPLICIT_USER_EXIT:
      # Explicit user "exit" command leads to sys.exit(1).
      print(
          "Note: user exited from debugger CLI: Calling sys.exit(1).",
          file=sys.stderr)
      sys.exit(1)

    return self._run_start_response

  def _prep_cli_for_run_start(self):
    """Prepare (but not launch) the CLI for run-start."""

    self._run_cli = curses_ui.CursesUI()

    help_intro = debugger_cli_common.RichTextLines([])
    if self._run_call_count == 1:
      # Show logo at the onset of the first run.
      help_intro.extend(cli_shared.get_tfdbg_logo())
    help_intro.extend(debugger_cli_common.RichTextLines("Upcoming run:"))
    help_intro.extend(self._run_info)

    self._run_cli.set_help_intro(help_intro)

    # Create initial screen output detailing the run.
    self._title = "run-start: " + self._run_description
    self._init_command = "help"
    self._title_color = "blue_on_white"

  def on_run_end(self, request):
    """Overrides on-run-end callback.

    Actions taken:
      1) Load the debug dump.
      2) Bring up the Analyzer CLI.

    Args:
      request: An instance of OnSessionInitRequest.

    Returns:
      An instance of OnSessionInitResponse.
    """

    if request.performed_action == framework.OnRunStartAction.DEBUG_RUN:
      partition_graphs = None
      if request.run_metadata and request.run_metadata.partition_graphs:
        partition_graphs = request.run_metadata.partition_graphs
      elif request.client_graph_def:
        partition_graphs = [request.client_graph_def]

      debug_dump = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=partition_graphs)

      passed_filter = None
      if self._active_tensor_filter:
        if not debug_dump.find(
            self._tensor_filters[self._active_tensor_filter], first_n=1):
          # No dumped tensor passes the filter in this run. Clean up the dump
          # directory and move on.
          self._remove_dump_root()
          return framework.OnRunEndResponse()
        else:
          # Some dumped tensor(s) from this run passed the filter.
          passed_filter = self._active_tensor_filter
          self._active_tensor_filter = None

      self._prep_cli_for_run_end(debug_dump, request.tf_error, passed_filter)

      self._run_start_response = self._launch_cli()

      # Clean up the dump generated by this run.
      self._remove_dump_root()
    else:
      # No debug information to show following a non-debug run() call.
      self._run_start_response = None

    # Return placeholder response that currently holds no additional
    # information.
    return framework.OnRunEndResponse()

  def _remove_dump_root(self):
    if os.path.isdir(self._dump_root):
      shutil.rmtree(self._dump_root)

  def _prep_cli_for_run_end(self, debug_dump, tf_error, passed_filter):
    """Prepare (but not launch) CLI for run-end, with debug dump from the run.

    Args:
      debug_dump: (debug_data.DebugDumpDir) The debug dump directory from this
        run.
      tf_error: (None or OpError) OpError that happened during the run() call
        (if any).
      passed_filter: (None or str) Name of the tensor filter that just passed
        and caused the preparation of this run-end CLI (if any).
    """

    if tf_error:
      help_intro = cli_shared.get_error_intro(tf_error)

      self._init_command = "help"
      self._title_color = "red_on_white"
    else:
      help_intro = None
      self._init_command = "lt"

      self._title_color = "black_on_white"
      if passed_filter is not None:
        # Some dumped tensor(s) from this run passed the filter.
        self._init_command = "lt -f %s" % passed_filter
        self._title_color = "red_on_white"

    analyzer = analyzer_cli.DebugAnalyzer(debug_dump)

    # Supply all the available tensor filters.
    for filter_name in self._tensor_filters:
      analyzer.add_tensor_filter(filter_name,
                                 self._tensor_filters[filter_name])

    self._run_cli = curses_ui.CursesUI()
    self._run_cli.register_command_handler(
        "list_tensors",
        analyzer.list_tensors,
        analyzer.get_help("list_tensors"),
        prefix_aliases=["lt"])
    self._run_cli.register_command_handler(
        "node_info",
        analyzer.node_info,
        analyzer.get_help("node_info"),
        prefix_aliases=["ni"])
    self._run_cli.register_command_handler(
        "list_inputs",
        analyzer.list_inputs,
        analyzer.get_help("list_inputs"),
        prefix_aliases=["li"])
    self._run_cli.register_command_handler(
        "list_outputs",
        analyzer.list_outputs,
        analyzer.get_help("list_outputs"),
        prefix_aliases=["lo"])
    self._run_cli.register_command_handler(
        "print_tensor",
        analyzer.print_tensor,
        analyzer.get_help("print_tensor"),
        prefix_aliases=["pt"])

    # Get names of all dumped tensors.
    dumped_tensor_names = []
    for datum in debug_dump.dumped_tensor_data:
      dumped_tensor_names.append("%s:%d" %
                                 (datum.node_name, datum.output_slot))

    # Tab completions for command "print_tensors".
    self._run_cli.register_tab_comp_context(["print_tensor", "pt"],
                                            dumped_tensor_names)

    # Tab completion for commands "node_info", "list_inputs" and
    # "list_outputs". The list comprehension is used below because nodes()
    # output can be unicodes and they need to be converted to strs.
    self._run_cli.register_tab_comp_context(
        ["node_info", "ni", "list_inputs", "li", "list_outputs", "lo"],
        [str(node_name) for node_name in debug_dump.nodes()])
    # TODO(cais): Reduce API surface area for aliases vis-a-vis tab
    #    completion contexts and registered command handlers.

    self._title = "run-end: " + self._run_description

    if help_intro:
      self._run_cli.set_help_intro(help_intro)

  def _launch_cli(self, is_run_start=False):
    """Launch the interactive command-line interface.

    Args:
      is_run_start: (bool) whether this CLI launch occurs at a run-start
        callback.

    Returns:
      The OnRunStartResponse specified by the user using the "run" command.
    """

    self._register_this_run_info(self._run_cli)
    response = self._run_cli.run_ui(
        init_command=self._init_command,
        title=self._title,
        title_color=self._title_color)

    return response

  def _run_info_handler(self, args, screen_info=None):
    return self._run_info

  def _run_handler(self, args, screen_info=None):
    """Command handler for "run" command during on-run-start."""

    _ = screen_info  # Currently unused.

    parsed = self._argparsers["run"].parse_args(args)

    if parsed.till_filter_pass:
      # For the run-till-bad-numerical-value-appears mode, use the DEBUG_RUN
      # option to access the intermediate tensors, and set the corresponding
      # state flag of the class itself to True.
      if parsed.till_filter_pass in self._tensor_filters:
        action = framework.OnRunStartAction.DEBUG_RUN
        self._active_tensor_filter = parsed.till_filter_pass
      else:
        # Handle invalid filter name.
        return debugger_cli_common.RichTextLines(
            ["ERROR: tensor filter \"%s\" does not exist." %
             parsed.till_filter_pass])

    self._skip_debug = parsed.no_debug
    self._run_through_times = parsed.times

    if parsed.times > 1 or parsed.no_debug:
      # If requested -t times > 1, the very next run will be a non-debug run.
      action = framework.OnRunStartAction.NON_DEBUG_RUN
      debug_urls = []
    else:
      action = framework.OnRunStartAction.DEBUG_RUN
      debug_urls = self._get_run_debug_urls()

    # Raise CommandLineExit exception to cause the CLI to exit.
    raise debugger_cli_common.CommandLineExit(
        exit_token=framework.OnRunStartResponse(action, debug_urls))

  def _register_this_run_info(self, curses_cli):
    curses_cli.register_command_handler(
        "run",
        self._run_handler,
        self._argparsers["run"].format_help(),
        prefix_aliases=["r"])
    curses_cli.register_command_handler(
        "invoke_stepper",
        self._on_run_start_step_handler,
        self._argparsers["invoke_stepper"].format_help(),
        prefix_aliases=["s"])
    curses_cli.register_command_handler(
        "run_info",
        self._run_info_handler,
        self._argparsers["run_info"].format_help(),
        prefix_aliases=["ri"])

    if self._tensor_filters:
      # Register tab completion for the filter names.
      curses_cli.register_tab_comp_context(["run", "r"],
                                           list(self._tensor_filters.keys()))

  def _on_run_start_step_handler(self, args, screen_info=None):
    """Command handler for "invoke_stepper" command during on-run-start."""

    _ = screen_info  # Currently unused.

    # No parsing is currently necessary for invoke_stepper. This may change
    # in the future when the command has arguments.

    # Raise CommandLineExit exception to cause the CLI to exit.
    raise debugger_cli_common.CommandLineExit(
        exit_token=framework.OnRunStartResponse(
            framework.OnRunStartAction.INVOKE_STEPPER, []))

  def _get_run_debug_urls(self):
    """Get the debug_urls value for the current run() call.

    Returns:
      debug_urls: (list of str) Debug URLs for the current run() call.
        Currently, the list consists of only one URL that is a file:// URL.
    """

    return ["file://" + self._dump_root]

  def _update_run_calls_state(self, run_call_count, fetches, feed_dict):
    """Update the internal state with regard to run() call history.

    Args:
      run_call_count: (int) Number of run() calls that have occurred.
      fetches: a node/tensor or a list of node/tensor that are the fetches of
        the run() call. This is the same as the fetches argument to the run()
        call.
      feed_dict: None of a dict. This is the feed_dict argument to the run()
        call.
    """

    self._run_call_count = run_call_count
    self._run_description = cli_shared.get_run_short_description(run_call_count,
                                                                 fetches,
                                                                 feed_dict)
    self._run_through_times -= 1

    self._run_info = cli_shared.get_run_start_intro(run_call_count,
                                                    fetches,
                                                    feed_dict,
                                                    self._tensor_filters)

  def invoke_node_stepper(self,
                          node_stepper,
                          restore_variable_values_on_exit=True):
    """Overrides method in base class to implement interactive node stepper.

    Args:
      node_stepper: (stepper.NodeStepper) The underlying NodeStepper API object.
      restore_variable_values_on_exit: (bool) Whether any variables whose values
        have been altered during this node-stepper invocation should be restored
        to their old values when this invocation ends.

    Returns:
      The same return values as the `Session.run()` call on the same fetches as
        the NodeStepper.
    """

    stepper = stepper_cli.NodeStepperCLI(node_stepper)

    # On exiting the node-stepper CLI, the finalize method of the node_stepper
    # object will be called, ensuring that the state of the graph will be the
    # same as if the stepping did not happen.
    # TODO(cais): Perhaps some users will want the effect of the interactive
    # stepping and value injection to persist. When that happens, make the call
    # to finalize optional.
    stepper_ui = curses_ui.CursesUI(
        on_ui_exit=(node_stepper.restore_variable_values
                    if restore_variable_values_on_exit else None))

    stepper_ui.register_command_handler(
        "list_sorted_nodes",
        stepper.list_sorted_nodes,
        stepper.arg_parsers["list_sorted_nodes"].format_help(),
        prefix_aliases=["lt", "lsn"])
    stepper_ui.register_command_handler(
        "cont",
        stepper.cont,
        stepper.arg_parsers["cont"].format_help(),
        prefix_aliases=["ct", "c"])
    stepper_ui.register_command_handler(
        "step",
        stepper.step,
        stepper.arg_parsers["step"].format_help(),
        prefix_aliases=["st", "s"])
    stepper_ui.register_command_handler(
        "print_tensor",
        stepper.print_tensor,
        stepper.arg_parsers["print_tensor"].format_help(),
        prefix_aliases=["pt"])
    stepper_ui.register_command_handler(
        "inject_value",
        stepper.inject_value,
        stepper.arg_parsers["inject_value"].format_help(),
        prefix_aliases=["inject", "override_value", "override"])

    # Register tab completion candidates.
    stepper_ui.register_tab_comp_context([
        "cont", "ct", "c", "pt", "inject_value", "inject", "override_value",
        "override"
    ], [str(elem) for elem in node_stepper.sorted_nodes()])
    # TODO(cais): Tie up register_tab_comp_context to a single alias to shorten
    # calls like this.

    return stepper_ui.run_ui(
        init_command="lt",
        title="Node Stepper: " + self._run_description,
        title_color="blue_on_white")
