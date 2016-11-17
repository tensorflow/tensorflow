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

import six

# Google-internal import(s).
from tensorflow.python.debug import debug_data
from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.debug.cli import curses_ui
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables


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

    # State flag for running till a tensor filter is passed.
    self._run_till_filter_pass = None

    # State related to tensor filters.
    self._tensor_filters = {}

    # Options for the on-run-start hook:
    #   1) run (DEBUG_RUN)
    #   2) run --nodebug (NON_DEBUG_RUN)
    #   3) invoke_stepper (INVOKE_STEPPER, not implemented)
    self._on_run_start_parsers = {}
    ap = argparse.ArgumentParser(
        description="Run through, with or without debug tensor watching.",
        usage=argparse.SUPPRESS)
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
    self._on_run_start_parsers["run"] = ap

    ap = argparse.ArgumentParser(
        description="Invoke stepper (cont, step, breakpoint, etc.)",
        usage=argparse.SUPPRESS)
    self._on_run_start_parsers["invoke_stepper"] = ap

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

    if self._run_till_filter_pass:
      # If we are running till a filter passes, we just need to keep running
      # with the DEBUG_RUN option.
      return framework.OnRunStartResponse(framework.OnRunStartAction.DEBUG_RUN,
                                          self._get_run_debug_urls())

    run_start_cli = curses_ui.CursesUI()

    run_start_cli.register_command_handler(
        "run",
        self._on_run_start_run_handler,
        self._on_run_start_parsers["run"].format_help(),
        prefix_aliases=["r"])
    run_start_cli.register_command_handler(
        "invoke_stepper",
        self._on_run_start_step_handler,
        self._on_run_start_parsers["invoke_stepper"].format_help(),
        prefix_aliases=["s"])

    if isinstance(request.fetches, list) or isinstance(request.fetches, tuple):
      fetch_lines = [fetch.name for fetch in request.fetches]
    else:
      fetch_lines = [repr(request.fetches)]

    if not request.feed_dict:
      feed_dict_lines = ["(Empty)"]
    else:
      feed_dict_lines = []
      for feed_key in request.feed_dict:
        if isinstance(feed_key, six.string_types):
          feed_dict_lines.append(feed_key)
        else:
          feed_dict_lines.append(feed_key.name)

    # TODO(cais): Refactor into its own function.
    help_intro = [
        "======================================",
        "About to enter Session run() call #%d:" % request.run_call_count, "",
        "Fetch(es):"
    ]
    help_intro.extend(["  " + line for line in fetch_lines])
    help_intro.extend(["", "Feed dict(s):"])
    help_intro.extend(["  " + line for line in feed_dict_lines])
    help_intro.extend([
        "======================================", "",
        "Select one of the following commands to proceed ---->", "  run:",
        "      Execute the run() call with the debug tensor-watching",
        "  run -n:",
        "      Execute the run() call without the debug tensor-watching",
        "  run -f <filter_name>:",
        "      Keep executing run() calls until a dumped tensor passes ",
        "      a given, registered filter emerge. Registered filter(s):"
    ])

    if self._tensor_filters:
      filter_names = []
      for filter_name in self._tensor_filters:
        filter_names.append(filter_name)
        help_intro.append("        * " + filter_name)

      # Register tab completion for the filter names.
      run_start_cli.register_tab_comp_context(["run", "r"], filter_names)
    else:
      help_intro.append("        (None)")

    help_intro.extend(["",
                       "For more details, see help below:"
                       "",])
    run_start_cli.set_help_intro(help_intro)

    # Create initial screen output detailing the run.
    title = "run-start: " + self._run_description
    response = run_start_cli.run_ui(
        init_command="help", title=title, title_color="yellow")
    if response == debugger_cli_common.EXPLICIT_USER_EXIT:
      # Explicit user "exit" command leads to sys.exit(1).
      print(
          "Note: user exited from debugger CLI: Calling sys.exit(1).",
          file=sys.stderr)
      sys.exit(1)

    return response

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

      if request.tf_error:
        op_name = request.tf_error.op.name

        # Prepare help introduction for the TensorFlow error that occurred
        # during the run.
        help_intro = [
            "--------------------------------------",
            "!!! An error occurred during the run !!!",
            "",
            "  * Use command \"ni %s\" to see the information about the "
            "failing op." % op_name,
            "  * Use command \"li -r %s\" to see the inputs to the "
            "failing op." % op_name,
            "  * Use command \"lt\" to view the dumped tensors.",
            "",
            "Op name:    " + op_name,
            "Error type: " + str(type(request.tf_error)),
            "",
            "Details:",
            str(request.tf_error),
            "",
            "WARNING: Using client GraphDef due to the error, instead of "
            "executor GraphDefs.",
            "--------------------------------------",
            "",
        ]
        init_command = "help"
        title_color = "red"
      else:
        help_intro = None
        init_command = "lt"

        title_color = "green"
        if self._run_till_filter_pass:
          if not debug_dump.find(
              self._tensor_filters[self._run_till_filter_pass], first_n=1):
            # No dumped tensor passes the filter in this run. Clean up the dump
            # directory and move on.
            shutil.rmtree(self._dump_root)
            return framework.OnRunEndResponse()
          else:
            # Some dumped tensor(s) from this run passed the filter.
            init_command = "lt -f %s" % self._run_till_filter_pass
            title_color = "red"
            self._run_till_filter_pass = None

      analyzer = analyzer_cli.DebugAnalyzer(debug_dump)

      # Supply all the available tensor filters.
      for filter_name in self._tensor_filters:
        analyzer.add_tensor_filter(filter_name,
                                   self._tensor_filters[filter_name])

      run_end_cli = curses_ui.CursesUI()
      run_end_cli.register_command_handler(
          "list_tensors",
          analyzer.list_tensors,
          analyzer.get_help("list_tensors"),
          prefix_aliases=["lt"])
      run_end_cli.register_command_handler(
          "node_info",
          analyzer.node_info,
          analyzer.get_help("node_info"),
          prefix_aliases=["ni"])
      run_end_cli.register_command_handler(
          "list_inputs",
          analyzer.list_inputs,
          analyzer.get_help("list_inputs"),
          prefix_aliases=["li"])
      run_end_cli.register_command_handler(
          "list_outputs",
          analyzer.list_outputs,
          analyzer.get_help("list_outputs"),
          prefix_aliases=["lo"])
      run_end_cli.register_command_handler(
          "print_tensor",
          analyzer.print_tensor,
          analyzer.get_help("print_tensor"),
          prefix_aliases=["pt"])

      run_end_cli.register_command_handler(
          "run",
          self._run_end_run_command_handler,
          "Helper command for incorrectly entered run command at the run-end "
          "prompt.",
          prefix_aliases=["r"]
      )

      # Get names of all dumped tensors.
      dumped_tensor_names = []
      for datum in debug_dump.dumped_tensor_data:
        dumped_tensor_names.append("%s:%d" %
                                   (datum.node_name, datum.output_slot))

      # Tab completions for command "print_tensors".
      run_end_cli.register_tab_comp_context(["print_tensor", "pt"],
                                            dumped_tensor_names)

      # Tab completion for commands "node_info", "list_inputs" and
      # "list_outputs". The list comprehension is used below because nodes()
      # output can be unicodes and they need to be converted to strs.
      run_end_cli.register_tab_comp_context(
          ["node_info", "ni", "list_inputs", "li", "list_outputs", "lo"],
          [str(node_name) for node_name in debug_dump.nodes()])
      # TODO(cais): Reduce API surface area for aliases vis-a-vis tab
      #    completion contexts and registered command handlers.

      title = "run-end: " + self._run_description
      run_end_cli.set_help_intro(help_intro)
      run_end_cli.run_ui(
          init_command=init_command, title=title, title_color=title_color)

      # Clean up the dump directory.
      shutil.rmtree(self._dump_root)
    else:
      print("No debug information to show following a non-debug run() call.")

    # Return placeholder response that currently holds no additional
    # information.
    return framework.OnRunEndResponse()

  def _on_run_start_run_handler(self, args, screen_info=None):
    """Command handler for "run" command during on-run-start."""

    _ = screen_info  # Currently unused.

    parsed = self._on_run_start_parsers["run"].parse_args(args)

    if parsed.till_filter_pass:
      # For the run-till-bad-numerical-value-appears mode, use the DEBUG_RUN
      # option to access the intermediate tensors, and set the corresponding
      # state flag of the class itself to True.
      if parsed.till_filter_pass in self._tensor_filters:
        action = framework.OnRunStartAction.DEBUG_RUN
        self._run_till_filter_pass = parsed.till_filter_pass
      else:
        # Handle invalid filter name.
        return debugger_cli_common.RichTextLines(
            ["ERROR: tensor filter \"%s\" does not exist." %
             parsed.till_filter_pass])

    if parsed.no_debug:
      action = framework.OnRunStartAction.NON_DEBUG_RUN
      debug_urls = []
    else:
      action = framework.OnRunStartAction.DEBUG_RUN
      debug_urls = self._get_run_debug_urls()

    # Raise CommandLineExit exception to cause the CLI to exit.
    raise debugger_cli_common.CommandLineExit(
        exit_token=framework.OnRunStartResponse(action, debug_urls))

  def _on_run_start_step_handler(self, args, screen_info=None):
    """Command handler for "invoke_stepper" command during on-run-start."""

    _ = screen_info  # Currently unused.

    # No parsing is currently necessary for invoke_stepper. This may change
    # in the future when the command has arguments.

    # Raise CommandLineExit exception to cause the CLI to exit.
    raise debugger_cli_common.CommandLineExit(
        exit_token=framework.OnRunStartResponse(
            framework.OnRunStartAction.INVOKE_STEPPER, []))

  def _run_end_run_command_handler(self, args, screen_info=None):
    """Handler for incorrectly entered run command at run-end prompt."""

    _ = screen_info  # Currently unused.

    return debugger_cli_common.RichTextLines([
        "ERROR: the \"run\" command is invalid for the run-end prompt.", "",
        "To proceed to the next run, ",
        "  1) exit this run-end prompt using the command \"exit\"",
        "  2) enter the command \"run\" at the next run-start prompt.",
    ])

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
    self._run_description = "run #%d: " % self._run_call_count

    if isinstance(fetches, (ops.Tensor, ops.Operation, variables.Variable)):
      self._run_description += "fetch: %s; " % fetches.name
    else:
      # Could be list, tuple, dict or namedtuple.
      self._run_description += "%d fetch(es); " % len(fetches)

    if not feed_dict:
      self._run_description += "0 feeds"
    else:
      if len(feed_dict) == 1:
        self._run_description += "1 feed"
      else:
        self._run_description += "%d feeds" % len(feed_dict)
