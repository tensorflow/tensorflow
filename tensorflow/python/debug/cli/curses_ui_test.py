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
"""Tests of the curses-based CLI."""
import argparse
import curses
import os
import queue
import tempfile
import threading

import numpy as np

from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_test_utils
from tensorflow.python.debug.cli import curses_ui
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import tensor_format
from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest


def string_to_codes(cmd):
  return [ord(c) for c in cmd]


def codes_to_string(cmd_code):
  # Omit non-ASCII key codes.
  return "".join(chr(code) for code in cmd_code if code < 256)


class MockCursesUI(curses_ui.CursesUI):
  """Mock subclass of CursesUI that bypasses actual terminal manipulations."""

  def __init__(self,
               height,
               width,
               command_sequence=None):
    self._height = height
    self._width = width

    self._command_sequence = command_sequence
    self._command_counter = 0

    # The mock class has no actual textbox. So use this variable to keep
    # track of what's entered in the textbox on creation.
    self._curr_existing_command = ""

    # Observers for test.
    # Observers of screen output.
    self.unwrapped_outputs = []
    self.wrapped_outputs = []
    self.scroll_messages = []
    self.output_array_pointer_indices = []

    self.output_pad_rows = []

    # Observers of command textbox.
    self.existing_commands = []

    # Observer for tab-completion candidates.
    self.candidates_lists = []

    # Observer for the main menu.
    self.main_menu_list = []

    # Observer for toast messages.
    self.toasts = []

    curses_ui.CursesUI.__init__(
        self,
        config=cli_config.CLIConfig(
            config_file_path=os.path.join(tempfile.mkdtemp(), ".tfdbg_config")))

    # Override the default path to the command history file to avoid test
    # concurrency issues.
    _, history_file_path = tempfile.mkstemp()  # safe to ignore fd
    self._command_history_store = debugger_cli_common.CommandHistory(
        history_file_path=history_file_path)

  # Below, override the _screen_ prefixed member methods that interact with the
  # actual terminal, so that the mock can run in a terminal-less environment.

  # TODO(cais): Search for a way to have a mock terminal object that behaves
  # like the actual terminal, so that we can test the terminal interaction
  # parts of the CursesUI class.

  def _screen_init(self):
    pass

  def _screen_refresh_size(self):
    self._max_y = self._height
    self._max_x = self._width

  def _screen_launch(self, enable_mouse_on_start):
    self._mouse_enabled = enable_mouse_on_start

  def _screen_terminate(self):
    pass

  def _screen_refresh(self):
    pass

  def _screen_create_command_window(self):
    pass

  def _screen_create_command_textbox(self, existing_command=None):
    """Override to insert observer of existing commands.

    Used in testing of history navigation and tab completion.

    Args:
      existing_command: Command string entered to the textbox at textbox
        creation time. Note that the textbox does not actually exist in this
        mock subclass. This method only keeps track of and records the state.
    """

    self.existing_commands.append(existing_command)
    self._curr_existing_command = existing_command

  def _screen_new_output_pad(self, rows, cols):
    return "mock_pad"

  def _screen_add_line_to_output_pad(self, pad, row, txt, color_segments=None):
    pass

  def _screen_draw_text_line(self, row, line, attr=curses.A_NORMAL, color=None):
    pass

  def _screen_scroll_output_pad(self, pad, viewport_top, viewport_left,
                                screen_location_top, screen_location_left,
                                screen_location_bottom, screen_location_right):
    pass

  def _screen_get_user_command(self):
    command = self._command_sequence[self._command_counter]

    self._command_key_counter = 0
    for c in command:
      if c == curses.KEY_RESIZE:
        # Special case for simulating a terminal resize event in curses.
        self._height = command[1]
        self._width = command[2]
        self._on_textbox_keypress(c)
        self._command_counter += 1
        return ""
      elif c == curses.KEY_MOUSE:
        mouse_x = command[1]
        mouse_y = command[2]
        self._command_counter += 1
        self._textbox_curr_terminator = c
        return self._fetch_hyperlink_command(mouse_x, mouse_y)
      else:
        y = self._on_textbox_keypress(c)

        self._command_key_counter += 1
        if y == curses_ui.CursesUI.CLI_TERMINATOR_KEY:
          break

    self._command_counter += 1

    # Take into account pre-existing string automatically entered on textbox
    # creation.
    return self._curr_existing_command + codes_to_string(command)

  def _screen_getmouse(self):
    output = (0, self._mouse_xy_sequence[self._mouse_counter][0],
              self._mouse_xy_sequence[self._mouse_counter][1], 0,
              curses.BUTTON1_CLICKED)
    self._mouse_counter += 1
    return output

  def _screen_gather_textbox_str(self):
    return codes_to_string(self._command_sequence[self._command_counter]
                           [:self._command_key_counter])

  def _scroll_output(self, direction, line_index=None):
    """Override to observe screen output.

    This method is invoked after every command that generates a new screen
    output and after every keyboard triggered screen scrolling. Therefore
    it is a good place to insert the observer.

    Args:
      direction: which direction to scroll.
      line_index: (int or None) Optional line index to scroll to. See doc string
        of the overridden method for more information.
    """

    curses_ui.CursesUI._scroll_output(self, direction, line_index=line_index)

    self.unwrapped_outputs.append(self._curr_unwrapped_output)
    self.wrapped_outputs.append(self._curr_wrapped_output)
    self.scroll_messages.append(self._scroll_info)
    self.output_array_pointer_indices.append(self._output_array_pointer_indices)
    self.output_pad_rows.append(self._output_pad_row)

  def _display_main_menu(self, output):
    curses_ui.CursesUI._display_main_menu(self, output)

    self.main_menu_list.append(self._main_menu)

  def _screen_render_nav_bar(self):
    pass

  def _screen_render_menu_pad(self):
    pass

  def _display_candidates(self, candidates):
    curses_ui.CursesUI._display_candidates(self, candidates)

    self.candidates_lists.append(candidates)

  def _toast(self, message, color=None, line_index=None):
    curses_ui.CursesUI._toast(self, message, color=color, line_index=line_index)

    self.toasts.append(message)


class CursesTest(test_util.TensorFlowTestCase):

  _EXIT = string_to_codes("exit\n")

  def _babble(self, args, screen_info=None):
    ap = argparse.ArgumentParser(
        description="Do babble.", usage=argparse.SUPPRESS)
    ap.add_argument(
        "-n",
        "--num_times",
        dest="num_times",
        type=int,
        default=60,
        help="How many times to babble")
    ap.add_argument(
        "-l",
        "--line",
        dest="line",
        type=str,
        default="bar",
        help="The content of each line")
    ap.add_argument(
        "-k",
        "--link",
        dest="link",
        action="store_true",
        help="Create a command link on each line")
    ap.add_argument(
        "-m",
        "--menu",
        dest="menu",
        action="store_true",
        help="Create a menu for testing")

    parsed = ap.parse_args(args)

    lines = [parsed.line] * parsed.num_times
    font_attr_segs = {}
    if parsed.link:
      for i in range(len(lines)):
        font_attr_segs[i] = [(
            0,
            len(lines[i]),
            debugger_cli_common.MenuItem("", "babble"),)]

    annotations = {}
    if parsed.menu:
      menu = debugger_cli_common.Menu()
      menu.append(
          debugger_cli_common.MenuItem("babble again", "babble"))
      menu.append(
          debugger_cli_common.MenuItem("ahoy", "ahoy", enabled=False))
      annotations[debugger_cli_common.MAIN_MENU_KEY] = menu

    output = debugger_cli_common.RichTextLines(
        lines, font_attr_segs=font_attr_segs, annotations=annotations)
    return output

  def _print_ones(self, args, screen_info=None):
    ap = argparse.ArgumentParser(
        description="Print all-one matrix.", usage=argparse.SUPPRESS)
    ap.add_argument(
        "-s",
        "--size",
        dest="size",
        type=int,
        default=3,
        help="Size of the matrix. For example, of the value is 3, "
        "the matrix will have shape (3, 3)")

    parsed = ap.parse_args(args)

    m = np.ones([parsed.size, parsed.size])

    return tensor_format.format_tensor(m, "m")

  def testInitialization(self):
    ui = MockCursesUI(40, 80)

    self.assertEqual(0, ui._command_pointer)
    self.assertEqual([], ui._active_command_history)
    self.assertEqual("", ui._pending_command)

  def testCursesUiInChildThreadStartsWithoutException(self):
    result = queue.Queue()
    def child_thread():
      try:
        MockCursesUI(40, 80)
      except ValueError as e:
        result.put(e)
    t = threading.Thread(target=child_thread)
    t.start()
    t.join()
    self.assertTrue(result.empty())

  def testRunUIExitImmediately(self):
    """Make sure that the UI can exit properly after launch."""

    ui = MockCursesUI(40, 80, command_sequence=[self._EXIT])
    ui.run_ui()

    # No screen output should have happened.
    self.assertEqual(0, len(ui.unwrapped_outputs))

  def testRunUIEmptyCommand(self):
    """Issue an empty command then exit."""

    ui = MockCursesUI(40, 80, command_sequence=[[], self._EXIT])
    ui.run_ui()

    # Empty command should not lead to any screen output.
    self.assertEqual(0, len(ui.unwrapped_outputs))

  def testRunUIInvalidCommandPrefix(self):
    """Handle an unregistered command prefix."""

    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("foo\n"), self._EXIT])
    ui.run_ui()

    # Screen output/scrolling should have happened exactly once.
    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual(1, len(ui.wrapped_outputs))
    self.assertEqual(1, len(ui.scroll_messages))

    self.assertEqual(["ERROR: Invalid command prefix \"foo\""],
                     ui.unwrapped_outputs[0].lines)
    # TODO(cais): Add explanation for the 35 extra lines.
    self.assertEqual(["ERROR: Invalid command prefix \"foo\""],
                     ui.wrapped_outputs[0].lines[:1])
    # A single line of output should not have caused scrolling.
    self.assertNotIn("Scroll", ui.scroll_messages[0])
    self.assertIn("Mouse:", ui.scroll_messages[0])

  def testRunUIInvalidCommandSyntax(self):
    """Handle a command with invalid syntax."""

    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("babble -z\n"), self._EXIT])

    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    # Screen output/scrolling should have happened exactly once.
    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual(1, len(ui.wrapped_outputs))
    self.assertEqual(1, len(ui.scroll_messages))
    self.assertIn("Mouse:", ui.scroll_messages[0])
    self.assertEqual(
        ["Syntax error for command: babble", "For help, do \"help babble\""],
        ui.unwrapped_outputs[0].lines)

  def testRunUIScrollTallOutputPageDownUp(self):
    """Scroll tall output with PageDown and PageUp."""

    # Use PageDown and PageUp to scroll back and forth a little before exiting.
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("babble\n"), [curses.KEY_NPAGE] * 2 +
                          [curses.KEY_PPAGE] + self._EXIT])

    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    # Screen output/scrolling should have happened exactly once.
    self.assertEqual(4, len(ui.unwrapped_outputs))
    self.assertEqual(4, len(ui.wrapped_outputs))
    self.assertEqual(4, len(ui.scroll_messages))

    # Before scrolling.
    self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 60, ui.wrapped_outputs[0].lines[:60])

    # Initial scroll: At the top.
    self.assertIn("Scroll (PgDn): 0.00%", ui.scroll_messages[0])
    self.assertIn("Mouse:", ui.scroll_messages[0])

    # After 1st scrolling (PageDown).
    # The screen output shouldn't have changed. Only the viewport should.
    self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 60, ui.wrapped_outputs[0].lines[:60])
    self.assertIn("Scroll (PgDn/PgUp): 1.69%", ui.scroll_messages[1])
    self.assertIn("Mouse:", ui.scroll_messages[1])

    # After 2nd scrolling (PageDown).
    self.assertIn("Scroll (PgDn/PgUp): 3.39%", ui.scroll_messages[2])
    self.assertIn("Mouse:", ui.scroll_messages[2])

    # After 3rd scrolling (PageUp).
    self.assertIn("Scroll (PgDn/PgUp): 1.69%", ui.scroll_messages[3])
    self.assertIn("Mouse:", ui.scroll_messages[3])

  def testCutOffTooManyOutputLines(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("babble -n 20\n"), self._EXIT])

    # Modify max_output_lines so that this test doesn't use too much time or
    # memory.
    ui.max_output_lines = 10

    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    self.assertEqual(["bar"] * 10 + ["Output cut off at 10 lines!"],
                     ui.wrapped_outputs[0].lines[:11])

  def testRunUIScrollTallOutputEndHome(self):
    """Scroll tall output with PageDown and PageUp."""

    # Use End and Home to scroll a little before exiting to test scrolling.
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble\n"),
            [curses.KEY_END] * 2 + [curses.KEY_HOME] + self._EXIT
        ])

    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    # Screen output/scrolling should have happened exactly once.
    self.assertEqual(4, len(ui.unwrapped_outputs))
    self.assertEqual(4, len(ui.wrapped_outputs))
    self.assertEqual(4, len(ui.scroll_messages))

    # Before scrolling.
    self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 60, ui.wrapped_outputs[0].lines[:60])

    # Initial scroll: At the top.
    self.assertIn("Scroll (PgDn): 0.00%", ui.scroll_messages[0])

    # After 1st scrolling (End).
    self.assertIn("Scroll (PgUp): 100.00%", ui.scroll_messages[1])

    # After 2nd scrolling (End).
    self.assertIn("Scroll (PgUp): 100.00%", ui.scroll_messages[2])

    # After 3rd scrolling (Hhome).
    self.assertIn("Scroll (PgDn): 0.00%", ui.scroll_messages[3])

  def testRunUIWithInitCmd(self):
    """Run UI with an initial command specified."""

    ui = MockCursesUI(40, 80, command_sequence=[self._EXIT])

    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui(init_command="babble")

    self.assertEqual(1, len(ui.unwrapped_outputs))

    self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 60, ui.wrapped_outputs[0].lines[:60])
    self.assertIn("Scroll (PgDn): 0.00%", ui.scroll_messages[0])

  def testCompileHelpWithoutHelpIntro(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("help\n"), self._EXIT])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.run_ui()

    self.assertEqual(["babble", "  Aliases: b", "", "  babble some"],
                     ui.unwrapped_outputs[0].lines[:4])

  def testCompileHelpWithHelpIntro(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("help\n"), self._EXIT])

    help_intro = debugger_cli_common.RichTextLines(
        ["This is a curses UI.", "All it can do is 'babble'.", ""])
    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.set_help_intro(help_intro)
    ui.run_ui()

    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual(
        help_intro.lines + ["babble", "  Aliases: b", "", "  babble some"],
        ui.unwrapped_outputs[0].lines[:7])

  def testCommandHistoryNavBackwardOnce(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("help\n"),
                          [curses.KEY_UP],  # Hit Up and Enter.
                          string_to_codes("\n"),
                          self._EXIT])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.run_ui()

    self.assertEqual(2, len(ui.unwrapped_outputs))

    for i in [0, 1]:
      self.assertEqual(["babble", "  Aliases: b", "", "  babble some"],
                       ui.unwrapped_outputs[i].lines[:4])

  def testCommandHistoryNavBackwardTwice(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("help\n"),
                          string_to_codes("babble\n"),
                          [curses.KEY_UP],
                          [curses.KEY_UP],  # Hit Up twice and Enter.
                          string_to_codes("\n"),
                          self._EXIT])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.run_ui()

    self.assertEqual(3, len(ui.unwrapped_outputs))

    # The 1st and 3rd outputs are for command "help".
    for i in [0, 2]:
      self.assertEqual(["babble", "  Aliases: b", "", "  babble some"],
                       ui.unwrapped_outputs[i].lines[:4])

    # The 2nd output is for command "babble".
    self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[1].lines)

  def testCommandHistoryNavBackwardOverLimit(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("help\n"),
                          string_to_codes("babble\n"),
                          [curses.KEY_UP],
                          [curses.KEY_UP],
                          [curses.KEY_UP],  # Hit Up three times and Enter.
                          string_to_codes("\n"),
                          self._EXIT])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.run_ui()

    self.assertEqual(3, len(ui.unwrapped_outputs))

    # The 1st and 3rd outputs are for command "help".
    for i in [0, 2]:
      self.assertEqual(["babble", "  Aliases: b", "", "  babble some"],
                       ui.unwrapped_outputs[i].lines[:4])

    # The 2nd output is for command "babble".
    self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[1].lines)

  def testCommandHistoryNavBackwardThenForward(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("help\n"),
                          string_to_codes("babble\n"),
                          [curses.KEY_UP],
                          [curses.KEY_UP],
                          [curses.KEY_DOWN],  # Hit Up twice and Down once.
                          string_to_codes("\n"),
                          self._EXIT])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.run_ui()

    self.assertEqual(3, len(ui.unwrapped_outputs))

    # The 1st output is for command "help".
    self.assertEqual(["babble", "  Aliases: b", "", "  babble some"],
                     ui.unwrapped_outputs[0].lines[:4])

    # The 2nd and 3rd outputs are for command "babble".
    for i in [1, 2]:
      self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[i].lines)

  def testCommandHistoryPrefixNavBackwardOnce(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 1\n"),
            string_to_codes("babble -n 10\n"),
            string_to_codes("help\n"),
            string_to_codes("b") + [curses.KEY_UP],  # Navigate with prefix.
            string_to_codes("\n"),
            self._EXIT
        ])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.run_ui()

    self.assertEqual(["bar"], ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 10, ui.unwrapped_outputs[1].lines)
    self.assertEqual(["babble", "  Aliases: b", "", "  babble some"],
                     ui.unwrapped_outputs[2].lines[:4])
    self.assertEqual(["bar"] * 10, ui.unwrapped_outputs[3].lines)

  def testTerminalResize(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("babble\n"),
                          [curses.KEY_RESIZE, 100, 85],  # Resize to [100, 85]
                          self._EXIT])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.run_ui()

    # The resize event should have caused a second screen output event.
    self.assertEqual(2, len(ui.unwrapped_outputs))
    self.assertEqual(2, len(ui.wrapped_outputs))
    self.assertEqual(2, len(ui.scroll_messages))

    # The 1st and 2nd screen outputs should be identical (unwrapped).
    self.assertEqual(ui.unwrapped_outputs[0], ui.unwrapped_outputs[1])

    # The 1st scroll info should contain scrolling, because the screen size
    # is less than the number of lines in the output.
    self.assertIn("Scroll (PgDn): 0.00%", ui.scroll_messages[0])

  def testTabCompletionWithCommonPrefix(self):
    # Type "b" and trigger tab completion.
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("b\t"), string_to_codes("\n"),
                          self._EXIT])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["ba"])
    ui.run_ui()

    # The automatically registered exit commands "exit" and "quit" should not
    # appear in the tab completion candidates because they don't start with
    # "b".
    self.assertEqual([["ba", "babble"]], ui.candidates_lists)

    # "ba" is a common prefix of the two candidates. So the "ba" command should
    # have been issued after the Enter.
    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual(1, len(ui.wrapped_outputs))
    self.assertEqual(1, len(ui.scroll_messages))
    self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 60, ui.wrapped_outputs[0].lines[:60])

  def testTabCompletionEmptyTriggerWithoutCommonPrefix(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("\t"),  # Trigger tab completion.
                          string_to_codes("\n"),
                          self._EXIT])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["a"])
    # Use a different alias "a" instead.
    ui.run_ui()

    # The manually registered command, along with the automatically registered
    # exit commands should appear in the candidates.
    self.assertEqual(
        [["a", "babble", "cfg", "config", "exit", "h", "help", "m", "mouse",
          "quit"]], ui.candidates_lists)

    # The two candidates have no common prefix. So no command should have been
    # issued.
    self.assertEqual(0, len(ui.unwrapped_outputs))
    self.assertEqual(0, len(ui.wrapped_outputs))
    self.assertEqual(0, len(ui.scroll_messages))

  def testTabCompletionNonemptyTriggerSingleCandidate(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("b\t"),  # Trigger tab completion.
                          string_to_codes("\n"),
                          self._EXIT])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["a"])
    ui.run_ui()

    # There is only one candidate, so no candidates should have been displayed.
    # Instead, the completion should have been automatically keyed in, leading
    # to the "babble" command being issue.
    self.assertEqual([[]], ui.candidates_lists)

    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual(1, len(ui.wrapped_outputs))
    self.assertEqual(1, len(ui.scroll_messages))
    self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 60, ui.wrapped_outputs[0].lines[:60])

  def testTabCompletionNoMatch(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[string_to_codes("c\t"),  # Trigger tab completion.
                          string_to_codes("\n"),
                          self._EXIT])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["a"])
    ui.run_ui()

    # Only the invalid command "c" should have been issued.
    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual(1, len(ui.wrapped_outputs))
    self.assertEqual(1, len(ui.scroll_messages))

    self.assertEqual(["ERROR: Invalid command prefix \"c\""],
                     ui.unwrapped_outputs[0].lines)
    self.assertEqual(["ERROR: Invalid command prefix \"c\""],
                     ui.wrapped_outputs[0].lines[:1])

  def testTabCompletionOneWordContext(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 3\t"),  # Trigger tab completion.
            string_to_codes("\n"),
            self._EXIT
        ])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.register_tab_comp_context(["babble", "b"], ["10", "20", "30", "300"])
    ui.run_ui()

    self.assertEqual([["30", "300"]], ui.candidates_lists)

    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual(1, len(ui.wrapped_outputs))
    self.assertEqual(1, len(ui.scroll_messages))
    self.assertEqual(["bar"] * 30, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 30, ui.wrapped_outputs[0].lines[:30])

  def testTabCompletionTwice(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 1\t"),  # Trigger tab completion.
            string_to_codes("2\t"),  # With more prefix, tab again.
            string_to_codes("3\n"),
            self._EXIT
        ])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.register_tab_comp_context(["babble", "b"], ["10", "120", "123"])
    ui.run_ui()

    # There should have been two different lists of candidates.
    self.assertEqual([["10", "120", "123"], ["120", "123"]],
                     ui.candidates_lists)

    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual(1, len(ui.wrapped_outputs))
    self.assertEqual(1, len(ui.scroll_messages))
    self.assertEqual(["bar"] * 123, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 123, ui.wrapped_outputs[0].lines[:123])

  def testRegexSearch(self):
    """Test regex search."""

    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 3\n"),
            string_to_codes("/(b|r)\n"),  # Regex search and highlight.
            string_to_codes("/a\n"),  # Regex search and highlight.
            self._EXIT
        ])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.run_ui()

    # The unwrapped (original) output should never have any highlighting.
    self.assertEqual(3, len(ui.unwrapped_outputs))
    for i in range(3):
      self.assertEqual(["bar"] * 3, ui.unwrapped_outputs[i].lines)
      self.assertEqual({}, ui.unwrapped_outputs[i].font_attr_segs)

    # The wrapped outputs should show highlighting depending on the regex.
    self.assertEqual(3, len(ui.wrapped_outputs))

    # The first output should have no highlighting.
    self.assertEqual(["bar"] * 3, ui.wrapped_outputs[0].lines[:3])
    self.assertEqual({}, ui.wrapped_outputs[0].font_attr_segs)

    # The second output should have highlighting for "b" and "r".
    self.assertEqual(["bar"] * 3, ui.wrapped_outputs[1].lines[:3])
    for i in range(3):
      self.assertEqual([(0, 1, "black_on_white"), (2, 3, "black_on_white")],
                       ui.wrapped_outputs[1].font_attr_segs[i])

    # The third output should have highlighting for "a" only.
    self.assertEqual(["bar"] * 3, ui.wrapped_outputs[1].lines[:3])
    for i in range(3):
      self.assertEqual([(1, 2, "black_on_white")],
                       ui.wrapped_outputs[2].font_attr_segs[i])

  def testRegexSearchContinuation(self):
    """Test continuing scrolling down to next regex match."""

    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 3\n"),
            string_to_codes("/(b|r)\n"),  # Regex search and highlight.
            string_to_codes("/\n"),  # Continue scrolling down: 1st time.
            string_to_codes("/\n"),  # Continue scrolling down: 2nd time.
            string_to_codes("/\n"),  # Continue scrolling down: 3rd time.
            string_to_codes("/\n"),  # Continue scrolling down: 4th time.
            self._EXIT
        ])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.run_ui()

    # The 1st output is for the non-searched output. The other three are for
    # the searched output. Even though continuation search "/" is performed
    # four times, there should be only three searched outputs, because the
    # last one has exceeded the end.
    self.assertEqual(4, len(ui.unwrapped_outputs))

    for i in range(4):
      self.assertEqual(["bar"] * 3, ui.unwrapped_outputs[i].lines)
      self.assertEqual({}, ui.unwrapped_outputs[i].font_attr_segs)

    self.assertEqual(["bar"] * 3, ui.wrapped_outputs[0].lines[:3])
    self.assertEqual({}, ui.wrapped_outputs[0].font_attr_segs)

    for j in range(1, 4):
      self.assertEqual(["bar"] * 3, ui.wrapped_outputs[j].lines[:3])
      self.assertEqual({
          0: [(0, 1, "black_on_white"), (2, 3, "black_on_white")],
          1: [(0, 1, "black_on_white"), (2, 3, "black_on_white")],
          2: [(0, 1, "black_on_white"), (2, 3, "black_on_white")]
      }, ui.wrapped_outputs[j].font_attr_segs)

    self.assertEqual([0, 0, 1, 2], ui.output_pad_rows)

  def testRegexSearchUnderLineWrapping(self):
    ui = MockCursesUI(
        40,
        6,  # Use a narrow window to trigger line wrapping
        command_sequence=[
            string_to_codes("babble -n 3 -l foo-bar-baz-qux\n"),
            string_to_codes("/foo\n"),  # Regex search and highlight.
            string_to_codes("/\n"),  # Continue scrolling down: 1st time.
            string_to_codes("/\n"),  # Continue scrolling down: 2nd time.
            string_to_codes("/\n"),  # Continue scrolling down: 3rd time.
            string_to_codes("/\n"),  # Continue scrolling down: 4th time.
            self._EXIT
        ])

    ui.register_command_handler(
        "babble", self._babble, "babble some")
    ui.run_ui()

    self.assertEqual(4, len(ui.wrapped_outputs))
    for wrapped_output in ui.wrapped_outputs:
      self.assertEqual(["foo-", "bar-", "baz-", "qux"] * 3,
                       wrapped_output.lines[0 : 12])

    # The scroll location should reflect the line wrapping.
    self.assertEqual([0, 0, 4, 8], ui.output_pad_rows)

  def testRegexSearchNoMatchContinuation(self):
    """Test continuing scrolling when there is no regex match."""

    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 3\n"),
            string_to_codes("/foo\n"),  # Regex search and highlight.
            string_to_codes("/\n"),  # Continue scrolling down.
            self._EXIT
        ])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.run_ui()

    # The regex search and continuation search in the 3rd command should not
    # have produced any output.
    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual([0], ui.output_pad_rows)

  def testRegexSearchContinuationWithoutSearch(self):
    """Test continuation scrolling when no regex search has been performed."""

    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 3\n"),
            string_to_codes("/\n"),  # Continue scrolling without search first.
            self._EXIT
        ])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.run_ui()

    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual([0], ui.output_pad_rows)

  def testRegexSearchWithInvalidRegex(self):
    """Test using invalid regex to search."""

    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 3\n"),
            string_to_codes("/[\n"),  # Continue scrolling without search first.
            self._EXIT
        ])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.run_ui()

    # Invalid regex should not have led to a new screen of output.
    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual([0], ui.output_pad_rows)

    # Invalid regex should have led to a toast error message.
    self.assertEqual(
        [MockCursesUI._UI_WAIT_MESSAGE,
         "ERROR: Invalid regular expression: \"[\"",
         MockCursesUI._UI_WAIT_MESSAGE],
        ui.toasts)

  def testRegexSearchFromCommandHistory(self):
    """Test regex search commands are recorded in command history."""

    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 3\n"),
            string_to_codes("/(b|r)\n"),  # Regex search and highlight.
            string_to_codes("babble -n 4\n"),
            [curses.KEY_UP],
            [curses.KEY_UP],
            string_to_codes("\n"),  # Hit Up twice and Enter.
            self._EXIT
        ])

    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.run_ui()

    self.assertEqual(4, len(ui.wrapped_outputs))

    self.assertEqual(["bar"] * 3, ui.wrapped_outputs[0].lines[:3])
    self.assertEqual({}, ui.wrapped_outputs[0].font_attr_segs)

    self.assertEqual(["bar"] * 3, ui.wrapped_outputs[1].lines[:3])
    for i in range(3):
      self.assertEqual([(0, 1, "black_on_white"), (2, 3, "black_on_white")],
                       ui.wrapped_outputs[1].font_attr_segs[i])

    self.assertEqual(["bar"] * 4, ui.wrapped_outputs[2].lines[:4])
    self.assertEqual({}, ui.wrapped_outputs[2].font_attr_segs)

    # The regex search command loaded from history should have worked on the
    # new screen output.
    self.assertEqual(["bar"] * 4, ui.wrapped_outputs[3].lines[:4])
    for i in range(4):
      self.assertEqual([(0, 1, "black_on_white"), (2, 3, "black_on_white")],
                       ui.wrapped_outputs[3].font_attr_segs[i])

  def testDisplayTensorWithIndices(self):
    """Test displaying tensor with indices."""

    ui = MockCursesUI(
        9,  # Use a small screen height to cause scrolling.
        80,
        command_sequence=[
            string_to_codes("print_ones --size 5\n"),
            [curses.KEY_NPAGE],
            [curses.KEY_NPAGE],
            [curses.KEY_NPAGE],
            [curses.KEY_END],
            [curses.KEY_NPAGE],  # This PageDown goes over the bottom limit.
            [curses.KEY_PPAGE],
            [curses.KEY_PPAGE],
            [curses.KEY_PPAGE],
            [curses.KEY_HOME],
            [curses.KEY_PPAGE],  # This PageDown goes over the top limit.
            self._EXIT
        ])

    ui.register_command_handler("print_ones", self._print_ones,
                                "print an all-one matrix of specified size")
    ui.run_ui()

    self.assertEqual(11, len(ui.unwrapped_outputs))
    self.assertEqual(11, len(ui.output_array_pointer_indices))
    self.assertEqual(11, len(ui.scroll_messages))

    for i in range(11):
      cli_test_utils.assert_lines_equal_ignoring_whitespace(
          self, ["Tensor \"m\":", ""], ui.unwrapped_outputs[i].lines[:2])
      self.assertEqual(
          repr(np.ones([5, 5])).split("\n"), ui.unwrapped_outputs[i].lines[2:])

    self.assertEqual({
        0: None,
        -1: [1, 0]
    }, ui.output_array_pointer_indices[0])
    self.assertIn(" Scroll (PgDn): 0.00% -[1,0] ", ui.scroll_messages[0])

    # Scrolled down one line.
    self.assertEqual({
        0: None,
        -1: [2, 0]
    }, ui.output_array_pointer_indices[1])
    self.assertIn(" Scroll (PgDn/PgUp): 16.67% -[2,0] ", ui.scroll_messages[1])

    # Scrolled down one line.
    self.assertEqual({
        0: [0, 0],
        -1: [3, 0]
    }, ui.output_array_pointer_indices[2])
    self.assertIn(" Scroll (PgDn/PgUp): 33.33% [0,0]-[3,0] ",
                  ui.scroll_messages[2])

    # Scrolled down one line.
    self.assertEqual({
        0: [1, 0],
        -1: [4, 0]
    }, ui.output_array_pointer_indices[3])
    self.assertIn(" Scroll (PgDn/PgUp): 50.00% [1,0]-[4,0] ",
                  ui.scroll_messages[3])

    # Scroll to the bottom.
    self.assertEqual({
        0: [4, 0],
        -1: None
    }, ui.output_array_pointer_indices[4])
    self.assertIn(" Scroll (PgUp): 100.00% [4,0]- ", ui.scroll_messages[4])

    # Attempt to scroll beyond the bottom should lead to no change.
    self.assertEqual({
        0: [4, 0],
        -1: None
    }, ui.output_array_pointer_indices[5])
    self.assertIn(" Scroll (PgUp): 100.00% [4,0]- ", ui.scroll_messages[5])

    # Scrolled up one line.
    self.assertEqual({
        0: [3, 0],
        -1: None
    }, ui.output_array_pointer_indices[6])
    self.assertIn(" Scroll (PgDn/PgUp): 83.33% [3,0]- ", ui.scroll_messages[6])

    # Scrolled up one line.
    self.assertEqual({
        0: [2, 0],
        -1: None
    }, ui.output_array_pointer_indices[7])
    self.assertIn(" Scroll (PgDn/PgUp): 66.67% [2,0]- ", ui.scroll_messages[7])

    # Scrolled up one line.
    self.assertEqual({
        0: [1, 0],
        -1: [4, 0]
    }, ui.output_array_pointer_indices[8])
    self.assertIn(" Scroll (PgDn/PgUp): 50.00% [1,0]-[4,0] ",
                  ui.scroll_messages[8])

    # Scroll to the top.
    self.assertEqual({
        0: None,
        -1: [1, 0]
    }, ui.output_array_pointer_indices[9])
    self.assertIn(" Scroll (PgDn): 0.00% -[1,0] ", ui.scroll_messages[9])

    # Attempt to scroll pass the top limit should lead to no change.
    self.assertEqual({
        0: None,
        -1: [1, 0]
    }, ui.output_array_pointer_indices[10])
    self.assertIn(" Scroll (PgDn): 0.00% -[1,0] ", ui.scroll_messages[10])

  def testScrollTensorByValidIndices(self):
    """Test scrolling to specified (valid) indices in a tensor."""

    ui = MockCursesUI(
        8,  # Use a small screen height to cause scrolling.
        80,
        command_sequence=[
            string_to_codes("print_ones --size 5\n"),
            string_to_codes("@[0, 0]\n"),  # Scroll to element [0, 0].
            string_to_codes("@1,0\n"),  # Scroll to element [3, 0].
            string_to_codes("@[0,2]\n"),  # Scroll back to line 0.
            self._EXIT
        ])

    ui.register_command_handler("print_ones", self._print_ones,
                                "print an all-one matrix of specified size")
    ui.run_ui()

    self.assertEqual(4, len(ui.unwrapped_outputs))
    self.assertEqual(4, len(ui.output_array_pointer_indices))

    for i in range(4):
      cli_test_utils.assert_lines_equal_ignoring_whitespace(
          self, ["Tensor \"m\":", ""], ui.unwrapped_outputs[i].lines[:2])
      self.assertEqual(
          repr(np.ones([5, 5])).split("\n"), ui.unwrapped_outputs[i].lines[2:])

    self.assertEqual({
        0: None,
        -1: [0, 0]
    }, ui.output_array_pointer_indices[0])
    self.assertEqual({
        0: [0, 0],
        -1: [2, 0]
    }, ui.output_array_pointer_indices[1])
    self.assertEqual({
        0: [1, 0],
        -1: [3, 0]
    }, ui.output_array_pointer_indices[2])
    self.assertEqual({
        0: [0, 0],
        -1: [2, 0]
    }, ui.output_array_pointer_indices[3])

  def testScrollTensorByInvalidIndices(self):
    """Test scrolling to specified invalid indices in a tensor."""

    ui = MockCursesUI(
        8,  # Use a small screen height to cause scrolling.
        80,
        command_sequence=[
            string_to_codes("print_ones --size 5\n"),
            string_to_codes("@[10, 0]\n"),  # Scroll to invalid indices.
            string_to_codes("@[]\n"),  # Scroll to invalid indices.
            string_to_codes("@\n"),  # Scroll to invalid indices.
            self._EXIT
        ])

    ui.register_command_handler("print_ones", self._print_ones,
                                "print an all-one matrix of specified size")
    ui.run_ui()

    # Because all scroll-by-indices commands are invalid, there should be only
    # one output event.
    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual(1, len(ui.output_array_pointer_indices))

    # Check error messages.
    self.assertEqual("ERROR: Indices exceed tensor dimensions.", ui.toasts[2])
    self.assertEqual("ERROR: invalid literal for int() with base 10: ''",
                     ui.toasts[4])
    self.assertEqual("ERROR: Empty indices.", ui.toasts[6])

  def testWriteScreenOutputToFileWorks(self):
    output_path = tempfile.mktemp()

    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 2>%s\n" % output_path),
            self._EXIT
        ])

    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    self.assertEqual(1, len(ui.unwrapped_outputs))

    with gfile.Open(output_path, "r") as f:
      self.assertEqual("bar\nbar\n", f.read())

    # Clean up output file.
    gfile.Remove(output_path)

  def testIncompleteRedirectErrors(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 2 >\n"),
            self._EXIT
        ])

    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    self.assertEqual(["ERROR: Redirect file path is empty"], ui.toasts)
    self.assertEqual(0, len(ui.unwrapped_outputs))

  def testAppendingRedirectErrors(self):
    output_path = tempfile.mktemp()

    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 2 >> %s\n" % output_path),
            self._EXIT
        ])

    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual(
        ["Syntax error for command: babble", "For help, do \"help babble\""],
        ui.unwrapped_outputs[0].lines)

    # Clean up output file.
    gfile.Remove(output_path)

  def testMouseOffTakesEffect(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("mouse off\n"), string_to_codes("babble\n"),
            self._EXIT
        ])
    ui.register_command_handler("babble", self._babble, "")

    ui.run_ui()
    self.assertFalse(ui._mouse_enabled)
    self.assertIn("Mouse: OFF", ui.scroll_messages[-1])

  def testMouseOffAndOnTakeEffect(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("mouse off\n"), string_to_codes("mouse on\n"),
            string_to_codes("babble\n"), self._EXIT
        ])
    ui.register_command_handler("babble", self._babble, "")

    ui.run_ui()
    self.assertTrue(ui._mouse_enabled)
    self.assertIn("Mouse: ON", ui.scroll_messages[-1])

  def testMouseClickOnLinkTriggersCommand(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 10 -k\n"),
            [curses.KEY_MOUSE, 1, 4],  # A click on a hyperlink.
            self._EXIT
        ])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    self.assertEqual(2, len(ui.unwrapped_outputs))
    self.assertEqual(["bar"] * 10, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[1].lines)

  def testMouseClickOnLinkWithExistingTextTriggersCommand(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 10 -k\n"),
            string_to_codes("foo"),  # Enter some existing code in the textbox.
            [curses.KEY_MOUSE, 1, 4],  # A click on a hyperlink.
            self._EXIT
        ])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    self.assertEqual(2, len(ui.unwrapped_outputs))
    self.assertEqual(["bar"] * 10, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[1].lines)

  def testMouseClickOffLinkDoesNotTriggersCommand(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 10 -k\n"),
            # A click off a hyperlink (too much to the right).
            [curses.KEY_MOUSE, 8, 4],
            self._EXIT
        ])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    # The mouse click event should not triggered no command.
    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual(["bar"] * 10, ui.unwrapped_outputs[0].lines)

    # This command should have generated no main menus.
    self.assertEqual([None], ui.main_menu_list)

  def testMouseClickOnEnabledMenuItemWorks(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 10 -m\n"),
            # A click on the enabled menu item.
            [curses.KEY_MOUSE, 3, 2],
            self._EXIT
        ])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    self.assertEqual(2, len(ui.unwrapped_outputs))
    self.assertEqual(["bar"] * 10, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[1].lines)

    # Check the content of the menu.
    self.assertEqual(["| babble again | ahoy | "], ui.main_menu_list[0].lines)
    self.assertEqual(1, len(ui.main_menu_list[0].font_attr_segs))
    self.assertEqual(1, len(ui.main_menu_list[0].font_attr_segs[0]))

    item_annot = ui.main_menu_list[0].font_attr_segs[0][0]
    self.assertEqual(2, item_annot[0])
    self.assertEqual(14, item_annot[1])
    self.assertEqual("babble", item_annot[2][0].content)
    self.assertEqual("underline", item_annot[2][1])

    # The output from the menu-triggered command does not have a menu.
    self.assertIsNone(ui.main_menu_list[1])

  def testMouseClickOnDisabledMenuItemTriggersNoCommand(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 10 -m\n"),
            # A click on the disabled menu item.
            [curses.KEY_MOUSE, 18, 1],
            self._EXIT
        ])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual(["bar"] * 10, ui.unwrapped_outputs[0].lines)

  def testNavigationUsingCommandLineWorks(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 2\n"),
            string_to_codes("babble -n 4\n"),
            string_to_codes("prev\n"),
            string_to_codes("next\n"),
            self._EXIT
        ])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    self.assertEqual(4, len(ui.unwrapped_outputs))
    self.assertEqual(["bar"] * 2, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 4, ui.unwrapped_outputs[1].lines)
    self.assertEqual(["bar"] * 2, ui.unwrapped_outputs[2].lines)
    self.assertEqual(["bar"] * 4, ui.unwrapped_outputs[3].lines)

  def testNavigationOverOldestLimitUsingCommandLineGivesCorrectWarning(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 2\n"),
            string_to_codes("babble -n 4\n"),
            string_to_codes("prev\n"),
            string_to_codes("prev\n"),  # Navigate over oldest limit.
            self._EXIT
        ])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    self.assertEqual(3, len(ui.unwrapped_outputs))
    self.assertEqual(["bar"] * 2, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 4, ui.unwrapped_outputs[1].lines)
    self.assertEqual(["bar"] * 2, ui.unwrapped_outputs[2].lines)

    self.assertEqual("At the OLDEST in navigation history!", ui.toasts[-2])

  def testNavigationOverLatestLimitUsingCommandLineGivesCorrectWarning(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 2\n"),
            string_to_codes("babble -n 4\n"),
            string_to_codes("prev\n"),
            string_to_codes("next\n"),
            string_to_codes("next\n"),  # Navigate over latest limit.
            self._EXIT
        ])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    self.assertEqual(4, len(ui.unwrapped_outputs))
    self.assertEqual(["bar"] * 2, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 4, ui.unwrapped_outputs[1].lines)
    self.assertEqual(["bar"] * 2, ui.unwrapped_outputs[2].lines)
    self.assertEqual(["bar"] * 4, ui.unwrapped_outputs[3].lines)

    self.assertEqual("At the LATEST in navigation history!", ui.toasts[-2])

  def testMouseClicksOnNavBarWorks(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 2\n"),
            string_to_codes("babble -n 4\n"),
            # A click on the back (prev) button of the nav bar.
            [curses.KEY_MOUSE, 3, 1],
            # A click on the forward (prev) button of the nav bar.
            [curses.KEY_MOUSE, 7, 1],
            self._EXIT
        ])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    self.assertEqual(4, len(ui.unwrapped_outputs))
    self.assertEqual(["bar"] * 2, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 4, ui.unwrapped_outputs[1].lines)
    self.assertEqual(["bar"] * 2, ui.unwrapped_outputs[2].lines)
    self.assertEqual(["bar"] * 4, ui.unwrapped_outputs[3].lines)

  def testMouseClicksOnNavBarAfterPreviousScrollingWorks(self):
    ui = MockCursesUI(
        40,
        80,
        command_sequence=[
            string_to_codes("babble -n 2\n"),
            [curses.KEY_NPAGE],   # Scroll down one line.
            string_to_codes("babble -n 4\n"),
            # A click on the back (prev) button of the nav bar.
            [curses.KEY_MOUSE, 3, 1],
            # A click on the forward (prev) button of the nav bar.
            [curses.KEY_MOUSE, 7, 1],
            self._EXIT
        ])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    self.assertEqual(6, len(ui.unwrapped_outputs))
    self.assertEqual(["bar"] * 2, ui.unwrapped_outputs[0].lines)
    # From manual scroll.
    self.assertEqual(["bar"] * 2, ui.unwrapped_outputs[1].lines)
    self.assertEqual(["bar"] * 4, ui.unwrapped_outputs[2].lines)
    # From history navigation.
    self.assertEqual(["bar"] * 2, ui.unwrapped_outputs[3].lines)
    # From history navigation's auto-scroll to history scroll position.
    self.assertEqual(["bar"] * 2, ui.unwrapped_outputs[4].lines)
    self.assertEqual(["bar"] * 4, ui.unwrapped_outputs[5].lines)

    self.assertEqual(6, len(ui.scroll_messages))
    self.assertIn("Scroll (PgDn): 0.00%", ui.scroll_messages[0])
    self.assertIn("Scroll (PgUp): 100.00%", ui.scroll_messages[1])
    self.assertIn("Scroll (PgDn): 0.00%", ui.scroll_messages[2])
    self.assertIn("Scroll (PgDn): 0.00%", ui.scroll_messages[3])
    self.assertIn("Scroll (PgUp): 100.00%", ui.scroll_messages[4])
    self.assertIn("Scroll (PgDn): 0.00%", ui.scroll_messages[5])


class ScrollBarTest(test_util.TensorFlowTestCase):

  def testConstructorRaisesExceptionForNotEnoughHeight(self):
    with self.assertRaisesRegex(ValueError,
                                r"Insufficient height for ScrollBar \(2\)"):
      curses_ui.ScrollBar(0, 0, 1, 1, 0, 0)

  def testLayoutIsEmptyForZeroRow(self):
    scroll_bar = curses_ui.ScrollBar(0, 0, 1, 7, 0, 0)
    layout = scroll_bar.layout()
    self.assertEqual(["  "] * 8, layout.lines)
    self.assertEqual({}, layout.font_attr_segs)

  def testLayoutIsEmptyFoOneRow(self):
    scroll_bar = curses_ui.ScrollBar(0, 0, 1, 7, 0, 1)
    layout = scroll_bar.layout()
    self.assertEqual(["  "] * 8, layout.lines)
    self.assertEqual({}, layout.font_attr_segs)

  def testClickCommandForOneRowIsNone(self):
    scroll_bar = curses_ui.ScrollBar(0, 0, 1, 7, 0, 1)
    self.assertIsNone(scroll_bar.get_click_command(0))
    self.assertIsNone(scroll_bar.get_click_command(3))
    self.assertIsNone(scroll_bar.get_click_command(7))
    self.assertIsNone(scroll_bar.get_click_command(8))

  def testLayoutIsCorrectForTopPosition(self):
    scroll_bar = curses_ui.ScrollBar(0, 0, 1, 7, 0, 20)
    layout = scroll_bar.layout()
    self.assertEqual(["UP"] + ["  "] * 6 + ["DN"], layout.lines)
    self.assertEqual(
        {0: [(0, 2, curses_ui.ScrollBar.BASE_ATTR)],
         1: [(0, 2, curses_ui.ScrollBar.BASE_ATTR)],
         7: [(0, 2, curses_ui.ScrollBar.BASE_ATTR)]},
        layout.font_attr_segs)

  def testWidth1LayoutIsCorrectForTopPosition(self):
    scroll_bar = curses_ui.ScrollBar(0, 0, 0, 7, 0, 20)
    layout = scroll_bar.layout()
    self.assertEqual(["U"] + [" "] * 6 + ["D"], layout.lines)
    self.assertEqual(
        {0: [(0, 1, curses_ui.ScrollBar.BASE_ATTR)],
         1: [(0, 1, curses_ui.ScrollBar.BASE_ATTR)],
         7: [(0, 1, curses_ui.ScrollBar.BASE_ATTR)]},
        layout.font_attr_segs)

  def testWidth3LayoutIsCorrectForTopPosition(self):
    scroll_bar = curses_ui.ScrollBar(0, 0, 2, 7, 0, 20)
    layout = scroll_bar.layout()
    self.assertEqual(["UP "] + ["   "] * 6 + ["DN "], layout.lines)
    self.assertEqual(
        {0: [(0, 3, curses_ui.ScrollBar.BASE_ATTR)],
         1: [(0, 3, curses_ui.ScrollBar.BASE_ATTR)],
         7: [(0, 3, curses_ui.ScrollBar.BASE_ATTR)]},
        layout.font_attr_segs)

  def testWidth4LayoutIsCorrectForTopPosition(self):
    scroll_bar = curses_ui.ScrollBar(0, 0, 3, 7, 0, 20)
    layout = scroll_bar.layout()
    self.assertEqual([" UP "] + ["    "] * 6 + ["DOWN"], layout.lines)
    self.assertEqual(
        {0: [(0, 4, curses_ui.ScrollBar.BASE_ATTR)],
         1: [(0, 4, curses_ui.ScrollBar.BASE_ATTR)],
         7: [(0, 4, curses_ui.ScrollBar.BASE_ATTR)]},
        layout.font_attr_segs)

  def testLayoutIsCorrectForBottomPosition(self):
    scroll_bar = curses_ui.ScrollBar(0, 0, 1, 7, 19, 20)
    layout = scroll_bar.layout()
    self.assertEqual(["UP"] + ["  "] * 6 + ["DN"], layout.lines)
    self.assertEqual(
        {0: [(0, 2, curses_ui.ScrollBar.BASE_ATTR)],
         6: [(0, 2, curses_ui.ScrollBar.BASE_ATTR)],
         7: [(0, 2, curses_ui.ScrollBar.BASE_ATTR)]},
        layout.font_attr_segs)

  def testLayoutIsCorrectForMiddlePosition(self):
    scroll_bar = curses_ui.ScrollBar(0, 0, 1, 7, 10, 20)
    layout = scroll_bar.layout()
    self.assertEqual(["UP"] + ["  "] * 6 + ["DN"], layout.lines)
    self.assertEqual(
        {0: [(0, 2, curses_ui.ScrollBar.BASE_ATTR)],
         3: [(0, 2, curses_ui.ScrollBar.BASE_ATTR)],
         7: [(0, 2, curses_ui.ScrollBar.BASE_ATTR)]},
        layout.font_attr_segs)

  def testClickCommandsAreCorrectForMiddlePosition(self):
    scroll_bar = curses_ui.ScrollBar(0, 0, 1, 7, 10, 20)
    self.assertIsNone(scroll_bar.get_click_command(-1))
    self.assertEqual(curses_ui._SCROLL_UP_A_LINE,
                     scroll_bar.get_click_command(0))
    self.assertEqual(curses_ui._SCROLL_UP,
                     scroll_bar.get_click_command(1))
    self.assertEqual(curses_ui._SCROLL_UP,
                     scroll_bar.get_click_command(2))
    self.assertIsNone(scroll_bar.get_click_command(3))
    self.assertEqual(curses_ui._SCROLL_DOWN,
                     scroll_bar.get_click_command(5))
    self.assertEqual(curses_ui._SCROLL_DOWN,
                     scroll_bar.get_click_command(6))
    self.assertEqual(curses_ui._SCROLL_DOWN_A_LINE,
                     scroll_bar.get_click_command(7))
    self.assertIsNone(scroll_bar.get_click_command(8))

  def testClickCommandsAreCorrectForBottomPosition(self):
    scroll_bar = curses_ui.ScrollBar(0, 0, 1, 7, 19, 20)
    self.assertIsNone(scroll_bar.get_click_command(-1))
    self.assertEqual(curses_ui._SCROLL_UP_A_LINE,
                     scroll_bar.get_click_command(0))
    for i in range(1, 6):
      self.assertEqual(curses_ui._SCROLL_UP,
                       scroll_bar.get_click_command(i))
    self.assertIsNone(scroll_bar.get_click_command(6))
    self.assertEqual(curses_ui._SCROLL_DOWN_A_LINE,
                     scroll_bar.get_click_command(7))
    self.assertIsNone(scroll_bar.get_click_command(8))

  def testClickCommandsAreCorrectForScrollBarNotAtZeroMinY(self):
    scroll_bar = curses_ui.ScrollBar(0, 5, 1, 12, 10, 20)
    self.assertIsNone(scroll_bar.get_click_command(0))
    self.assertIsNone(scroll_bar.get_click_command(4))
    self.assertEqual(curses_ui._SCROLL_UP_A_LINE,
                     scroll_bar.get_click_command(5))
    self.assertEqual(curses_ui._SCROLL_UP,
                     scroll_bar.get_click_command(6))
    self.assertEqual(curses_ui._SCROLL_UP,
                     scroll_bar.get_click_command(7))
    self.assertIsNone(scroll_bar.get_click_command(8))
    self.assertEqual(curses_ui._SCROLL_DOWN,
                     scroll_bar.get_click_command(10))
    self.assertEqual(curses_ui._SCROLL_DOWN,
                     scroll_bar.get_click_command(11))
    self.assertEqual(curses_ui._SCROLL_DOWN_A_LINE,
                     scroll_bar.get_click_command(12))
    self.assertIsNone(scroll_bar.get_click_command(13))


if __name__ == "__main__":
  googletest.main()
