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
import sys
import tempfile

from tensorflow.python.debug import debug_data
from tensorflow.python.debug import framework
from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.debug.cli import curses_ui
from tensorflow.python.debug.cli import debugger_cli_common


_DUMP_ROOT_PREFIX = "tfdbg_"


class LocalCLIDebugWrapperSession(framework.BaseDebugWrapperSession):
  """Concrete subclass of BaseDebugWrapperSession implementing a local CLI."""

  def __init__(self, sess, dump_root=None):
    framework.BaseDebugWrapperSession.__init__(self, sess)

    self._dump_root = dump_root or tempfile.mktemp(prefix=_DUMP_ROOT_PREFIX)

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
    self._on_run_start_parsers["run"] = ap

    ap = argparse.ArgumentParser(
        description="Invoke stepper (cont, step, breakpoint, etc.)",
        usage=argparse.SUPPRESS)
    self._on_run_start_parsers["invoke_stepper"] = ap

  def on_session_init(self, request):
    """Overrides on-session-init callback.

    Actions taken:
      1) Start the CLI.

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

    help_intro = [
        "======================================",
        "About to enter Session run() call #%d:" % request.run_call_count,
        "",
        "Fetch(es):",
        "  " + repr(request.fetches),
        "",
        "Feed dict(s):",
        "  " + repr(request.feed_dict),
        "======================================",
        "",
        "Select one of the following commands to proceed ---->",
        "  run:",
        "      Execute the run() call with the debug tensor-watching",
        "  run --no_debug:",
        "      Execute the run() call without the debug tensor-watching",
        "",
        "For more details, see help below:"
        "",
    ]
    run_start_cli.set_help_intro(help_intro)

    # Create initial screen output detailing the run.
    response = run_start_cli.run_ui(init_command="help")
    if response == debugger_cli_common.EXPLICIT_USER_EXIT:
      # Explicit user "exit" command leads to sys.exit(1).
      print("Note: user exited from debugger CLI: sys.exit(1) called.",
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
      debug_dump = debug_data.DebugDumpDir(
          self._dump_root,
          partition_graphs=request.run_metadata.partition_graphs)

      analyzer = analyzer_cli.DebugAnalyzer(debug_dump)

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

      run_end_cli.run_ui(init_command="list_tensors")
    else:
      print("No debug information to show following a non-debug run() call.")

    # Return placeholder response that currently holds no additional
    # information.
    return framework.OnRunEndResponse()

  def _on_run_start_run_handler(self, args, screen_info=None):
    """Command handler for "run" command during on-run-start."""

    _ = screen_info  # Currently unused.

    parsed = self._on_run_start_parsers["run"].parse_args(args)

    if parsed.no_debug:
      action = framework.OnRunStartAction.NON_DEBUG_RUN
      debug_urls = []
    else:
      action = framework.OnRunStartAction.DEBUG_RUN
      debug_urls = ["file://" + self._dump_root]

    annotations = {
        debugger_cli_common.EXIT_TOKEN_KEY: framework.OnRunStartResponse(
            action, debug_urls)
    }

    return debugger_cli_common.RichTextLines([], annotations=annotations)

  def _on_run_start_step_handler(self, args, screen_info=None):
    """Command handler for "invoke_stepper" command during on-run-start."""

    _ = screen_info  # Currently unused.

    # No parsing is currently necessary for invoke_stepper. This may change
    # in the future when the command has arguments.

    action = framework.OnRunStartAction.INVOKE_STEPPER
    annotations = {
        debugger_cli_common.EXIT_TOKEN_KEY: framework.OnRunStartResponse(action,
                                                                         [])
    }

    return debugger_cli_common.RichTextLines([], annotations=annotations)
