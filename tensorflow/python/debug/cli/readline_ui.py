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
"""Readline-Based Command-Line Interface of TensorFlow Debugger (tfdbg)."""
import readline

import six

from tensorflow.python.debug.cli import base_ui
from tensorflow.python.debug.cli import debugger_cli_common


class ReadlineUI(base_ui.BaseUI):
  """Readline-based Command-line UI."""

  def __init__(self, on_ui_exit=None, config=None):
    base_ui.BaseUI.__init__(self, on_ui_exit=on_ui_exit, config=config)
    self._init_input()

  def _init_input(self):
    readline.parse_and_bind("set editing-mode emacs")

    # Disable default readline delimiter in order to receive the full text
    # (not just the last word) in the completer.
    readline.set_completer_delims("\n")
    readline.set_completer(self._readline_complete)
    readline.parse_and_bind("tab: complete")

    self._input = six.moves.input

  def _readline_complete(self, text, state):
    context, prefix, except_last_word = self._analyze_tab_complete_input(text)
    candidates, _ = self._tab_completion_registry.get_completions(context,
                                                                  prefix)
    candidates = [(except_last_word + candidate) for candidate in candidates]
    return candidates[state]

  def run_ui(self,
             init_command=None,
             title=None,
             title_color=None,
             enable_mouse_on_start=True):
    """Run the CLI: See the doc of base_ui.BaseUI.run_ui for more details."""

    print(title)

    if init_command is not None:
      self._dispatch_command(init_command)

    exit_token = self._ui_loop()

    if self._on_ui_exit:
      self._on_ui_exit()

    return exit_token

  def _ui_loop(self):
    while True:
      command = self._get_user_command()

      exit_token = self._dispatch_command(command)
      if exit_token is not None:
        return exit_token

  def _get_user_command(self):
    print("")
    return self._input(self.CLI_PROMPT).strip()

  def _dispatch_command(self, command):
    """Dispatch user command.

    Args:
      command: (str) Command to dispatch.

    Returns:
      An exit token object. None value means that the UI loop should not exit.
      A non-None value means the UI loop should exit.
    """

    if command in self.CLI_EXIT_COMMANDS:
      # Explicit user command-triggered exit: EXPLICIT_USER_EXIT as the exit
      # token.
      return debugger_cli_common.EXPLICIT_USER_EXIT

    try:
      prefix, args, output_file_path = self._parse_command(command)
    except SyntaxError as e:
      print(str(e))
      return

    if self._command_handler_registry.is_registered(prefix):
      try:
        screen_output = self._command_handler_registry.dispatch_command(
            prefix, args, screen_info=None)
      except debugger_cli_common.CommandLineExit as e:
        return e.exit_token
    else:
      screen_output = debugger_cli_common.RichTextLines([
          self.ERROR_MESSAGE_PREFIX + "Invalid command prefix \"%s\"" % prefix
      ])

    self._display_output(screen_output)
    if output_file_path:
      try:
        screen_output.write_to_file(output_file_path)
        print("Wrote output to %s" % output_file_path)
      except Exception:  # pylint: disable=broad-except
        print("Failed to write output to %s" % output_file_path)

  def _display_output(self, screen_output):
    for line in screen_output.lines:
      print(line)
