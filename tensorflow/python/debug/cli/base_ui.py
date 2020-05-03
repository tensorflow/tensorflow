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
"""Base Class of TensorFlow Debugger (tfdbg) Command-Line Interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common


class BaseUI(object):
  """Base class of tfdbg user interface."""

  CLI_PROMPT = "tfdbg> "
  CLI_EXIT_COMMANDS = ["exit", "quit"]
  ERROR_MESSAGE_PREFIX = "ERROR: "
  INFO_MESSAGE_PREFIX = "INFO: "

  def __init__(self, on_ui_exit=None, config=None):
    """Constructor of the base class.

    Args:
      on_ui_exit: (`Callable`) the callback to be called when the UI exits.
      config: An instance of `cli_config.CLIConfig()` carrying user-facing
        configurations.
    """

    self._on_ui_exit = on_ui_exit

    self._command_handler_registry = (
        debugger_cli_common.CommandHandlerRegistry())

    self._tab_completion_registry = debugger_cli_common.TabCompletionRegistry()

    # Create top-level tab-completion context and register the exit and help
    # commands.
    self._tab_completion_registry.register_tab_comp_context(
        [""], self.CLI_EXIT_COMMANDS +
        [debugger_cli_common.CommandHandlerRegistry.HELP_COMMAND] +
        debugger_cli_common.CommandHandlerRegistry.HELP_COMMAND_ALIASES)

    self._config = config or cli_config.CLIConfig()
    self._config_argparser = argparse.ArgumentParser(
        description="config command", usage=argparse.SUPPRESS)
    subparsers = self._config_argparser.add_subparsers()
    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("property_name", type=str)
    set_parser.add_argument("property_value", type=str)
    set_parser = subparsers.add_parser("show")
    self.register_command_handler(
        "config",
        self._config_command_handler,
        self._config_argparser.format_help(),
        prefix_aliases=["cfg"])

  def set_help_intro(self, help_intro):
    """Set an introductory message to the help output of the command registry.

    Args:
      help_intro: (RichTextLines) Rich text lines appended to the beginning of
        the output of the command "help", as introductory information.
    """

    self._command_handler_registry.set_help_intro(help_intro=help_intro)

  def register_command_handler(self,
                               prefix,
                               handler,
                               help_info,
                               prefix_aliases=None):
    """A wrapper around CommandHandlerRegistry.register_command_handler().

    In addition to calling the wrapped register_command_handler() method, this
    method also registers the top-level tab-completion context based on the
    command prefixes and their aliases.

    See the doc string of the wrapped method for more details on the args.

    Args:
      prefix: (str) command prefix.
      handler: (callable) command handler.
      help_info: (str) help information.
      prefix_aliases: (list of str) aliases of the command prefix.
    """

    self._command_handler_registry.register_command_handler(
        prefix, handler, help_info, prefix_aliases=prefix_aliases)

    self._tab_completion_registry.extend_comp_items("", [prefix])
    if prefix_aliases:
      self._tab_completion_registry.extend_comp_items("", prefix_aliases)

  def register_tab_comp_context(self, *args, **kwargs):
    """Wrapper around TabCompletionRegistry.register_tab_comp_context()."""

    self._tab_completion_registry.register_tab_comp_context(*args, **kwargs)

  def run_ui(self,
             init_command=None,
             title=None,
             title_color=None,
             enable_mouse_on_start=True):
    """Run the UI until user- or command- triggered exit.

    Args:
      init_command: (str) Optional command to run on CLI start up.
      title: (str) Optional title to display in the CLI.
      title_color: (str) Optional color of the title, e.g., "yellow".
      enable_mouse_on_start: (bool) Whether the mouse mode is to be enabled on
        start-up.

    Returns:
      An exit token of arbitrary type. Can be None.
    """

    raise NotImplementedError("run_ui() is not implemented in BaseUI")

  def _parse_command(self, command):
    """Parse a command string into prefix and arguments.

    Args:
      command: (str) Command string to be parsed.

    Returns:
      prefix: (str) The command prefix.
      args: (list of str) The command arguments (i.e., not including the
        prefix).
      output_file_path: (str or None) The path to save the screen output
        to (if any).
    """
    command = command.strip()
    if not command:
      return "", [], None

    command_items = command_parser.parse_command(command)
    command_items, output_file_path = command_parser.extract_output_file_path(
        command_items)

    return command_items[0], command_items[1:], output_file_path

  def _analyze_tab_complete_input(self, text):
    """Analyze raw input to tab-completer.

    Args:
      text: (str) the full, raw input text to be tab-completed.

    Returns:
      context: (str) the context str. For example,
        If text == "print_tensor softmax", returns "print_tensor".
        If text == "print", returns "".
        If text == "", returns "".
      prefix: (str) the prefix to be tab-completed, from the last word.
        For example, if text == "print_tensor softmax", returns "softmax".
        If text == "print", returns "print".
        If text == "", returns "".
      except_last_word: (str) the input text, except the last word.
        For example, if text == "print_tensor softmax", returns "print_tensor".
        If text == "print_tensor -a softmax", returns "print_tensor -a".
        If text == "print", returns "".
        If text == "", returns "".
    """
    text = text.lstrip()
    if not text:
      # Empty (top-level) context.
      context = ""
      prefix = ""
      except_last_word = ""
    else:
      items = text.split(" ")
      if len(items) == 1:
        # Single word: top-level context.
        context = ""
        prefix = items[0]
        except_last_word = ""
      else:
        # Multiple words.
        context = items[0]
        prefix = items[-1]
        except_last_word = " ".join(items[:-1]) + " "

    return context, prefix, except_last_word

  @property
  def config(self):
    """Obtain the CLIConfig of this `BaseUI` instance."""
    return self._config

  def _config_command_handler(self, args, screen_info=None):
    """Command handler for the "config" command."""
    del screen_info  # Currently unused.

    parsed = self._config_argparser.parse_args(args)
    if hasattr(parsed, "property_name") and hasattr(parsed, "property_value"):
      # set.
      self._config.set(parsed.property_name, parsed.property_value)
      return self._config.summarize(highlight=parsed.property_name)
    else:
      # show.
      return self._config.summarize()
