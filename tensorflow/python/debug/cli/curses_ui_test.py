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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import curses

from tensorflow.python.debug.cli import curses_ui
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


def string_to_codes(cmd):
  return [ord(c) for c in cmd]


def codes_to_string(cmd_code):
  # Omit non-ASCII key codes.
  return "".join([chr(code) for code in cmd_code if code < 256])


class MockCursesUI(curses_ui.CursesUI):
  """Mock subclass of CursesUI that bypasses actual terminal manipulations."""

  def __init__(self, height, width, command_sequence=None):
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

    # Observers of command textbox.
    self.existing_commands = []

    # Observers for tab-completion candidates.
    self.candidates_lists = []

    curses_ui.CursesUI.__init__(self)

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

  def _screen_launch(self):
    pass

  def _screen_terminate(self):
    pass

  def _screen_refresh(self):
    pass

  def _screen_create_command_window(self):
    pass

  def _screen_create_command_textbox(self, existing_command):
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

      y = self._on_textbox_keypress(c)

      self._command_key_counter += 1
      if y == curses_ui.CursesUI.CLI_TERMINATOR_KEY:
        break

    self._command_counter += 1

    # Take into account pre-existing string automatically entered on textbox
    # creation.
    return self._curr_existing_command + codes_to_string(command)

  def _screen_gather_textbox_str(self):
    return codes_to_string(self._command_sequence[self._command_counter]
                           [:self._command_key_counter])

  def _scroll_output(self, direction):
    """Override to observe screen output.

    This method is invoked after every command that generates a new screen
    output and after every keyboard triggered screen scrolling. Therefore
    it is a good place to insert the observer.

    Args:
      direction: which direction to scroll.
    """

    curses_ui.CursesUI._scroll_output(self, direction)

    self.unwrapped_outputs.append(self._curr_unwrapped_output)
    self.wrapped_outputs.append(self._curr_wrapped_output)
    self.scroll_messages.append(self._scroll_info)

  def _display_candidates(self, candidates):
    curses_ui.CursesUI._display_candidates(self, candidates)

    self.candidates_lists.append(candidates)


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

    parsed = ap.parse_args(args)

    return debugger_cli_common.RichTextLines(["bar"] * parsed.num_times)

  def testInitialization(self):
    ui = MockCursesUI(40, 80)

    self.assertEqual(0, ui._command_pointer)
    self.assertEqual([], ui._active_command_history)
    self.assertEqual("", ui._pending_command)

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
    self.assertEqual(["ERROR: Invalid command prefix \"foo\""],
                     ui.wrapped_outputs[0].lines)
    # A single line of output should not have caused scrolling.
    self.assertEqual("-" * 80, ui.scroll_messages[0])

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
    self.assertEqual(["bar"] * 60, ui.wrapped_outputs[0].lines)

    # Initial scroll: At the top.
    self.assertIn("Scroll: 0.00%", ui.scroll_messages[0])

    # After 1st scrolling (PageDown).
    # The screen output shouldn't have changed. Only the viewport should.
    self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 60, ui.wrapped_outputs[0].lines)
    self.assertIn("Scroll: 4.17%", ui.scroll_messages[1])

    # After 2nd scrolling (PageDown).
    self.assertIn("Scroll: 8.33%", ui.scroll_messages[2])

    # After 3rd scrolling (PageUp).
    self.assertIn("Scroll: 4.17%", ui.scroll_messages[3])

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
    self.assertEqual(["bar"] * 60, ui.wrapped_outputs[0].lines)

    # Initial scroll: At the top.
    self.assertIn("Scroll: 0.00%", ui.scroll_messages[0])

    # After 1st scrolling (End).
    self.assertIn("Scroll: 100.00%", ui.scroll_messages[1])

    # After 2nd scrolling (End).
    self.assertIn("Scroll: 100.00%", ui.scroll_messages[2])

    # After 3rd scrolling (Hhome).
    self.assertIn("Scroll: 0.00%", ui.scroll_messages[3])

  def testRunUIWithInitCmd(self):
    """Run UI with an initial command specified."""

    ui = MockCursesUI(40, 80, command_sequence=[self._EXIT])

    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui(init_command="babble")

    self.assertEqual(1, len(ui.unwrapped_outputs))

    self.assertEqual(["bar"] * 60, ui.unwrapped_outputs[0].lines)
    self.assertEqual(["bar"] * 60, ui.wrapped_outputs[0].lines)
    self.assertIn("Scroll: 0.00%", ui.scroll_messages[0])

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

    help_intro = ["This is a curses UI.", "All it can do is 'babble'.", ""]
    ui.register_command_handler(
        "babble", self._babble, "babble some", prefix_aliases=["b"])
    ui.set_help_intro(help_intro)
    ui.run_ui()

    self.assertEqual(1, len(ui.unwrapped_outputs))
    self.assertEqual(
        help_intro + ["babble", "  Aliases: b", "", "  babble some"],
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
    self.assertIn("Scroll: 0.00%", ui.scroll_messages[0])

    # The 2nd scroll info should contain no scrolling, because the screen size
    # is now greater than the numberf lines in the output.
    self.assertEqual("-" * 85, ui.scroll_messages[1])

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
    self.assertEqual(["bar"] * 60, ui.wrapped_outputs[0].lines)

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
    self.assertEqual([["a", "babble", "exit", "h", "help", "quit"]],
                     ui.candidates_lists)

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
    self.assertEqual(["bar"] * 60, ui.wrapped_outputs[0].lines)

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
                     ui.wrapped_outputs[0].lines)

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
    self.assertEqual(["bar"] * 30, ui.wrapped_outputs[0].lines)

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
    self.assertEqual(["bar"] * 123, ui.wrapped_outputs[0].lines)

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

    # The unwrapped (original) output should never have any highglighting.
    self.assertEqual(3, len(ui.unwrapped_outputs))
    for i in range(3):
      self.assertEqual(["bar"] * 3, ui.unwrapped_outputs[i].lines)
      self.assertEqual({}, ui.unwrapped_outputs[i].font_attr_segs)

    # The wrapped outputs should show highlighting depending on the regex.
    self.assertEqual(3, len(ui.wrapped_outputs))

    # The first output should have no highlighting.
    self.assertEqual(["bar"] * 3, ui.wrapped_outputs[0].lines)
    self.assertEqual({}, ui.wrapped_outputs[0].font_attr_segs)

    # The second output should have highlighting for "b" and "r".
    self.assertEqual(["bar"] * 3, ui.wrapped_outputs[1].lines)
    for i in range(3):
      self.assertEqual([(0, 1, "bw_reversed"), (2, 3, "bw_reversed")],
                       ui.wrapped_outputs[1].font_attr_segs[i])

    # The third output should have highlighting for "a" only.
    self.assertEqual(["bar"] * 3, ui.wrapped_outputs[1].lines)
    for i in range(3):
      self.assertEqual([(1, 2, "bw_reversed")],
                       ui.wrapped_outputs[2].font_attr_segs[i])

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

    self.assertEqual(["bar"] * 3, ui.wrapped_outputs[0].lines)
    self.assertEqual({}, ui.wrapped_outputs[0].font_attr_segs)

    self.assertEqual(["bar"] * 3, ui.wrapped_outputs[1].lines)
    for i in range(3):
      self.assertEqual([(0, 1, "bw_reversed"), (2, 3, "bw_reversed")],
                       ui.wrapped_outputs[1].font_attr_segs[i])

    self.assertEqual(["bar"] * 4, ui.wrapped_outputs[2].lines)
    self.assertEqual({}, ui.wrapped_outputs[2].font_attr_segs)

    # The regex search command loaded from history should have worked on the
    # new screen output.
    self.assertEqual(["bar"] * 4, ui.wrapped_outputs[3].lines)
    for i in range(4):
      self.assertEqual([(0, 1, "bw_reversed"), (2, 3, "bw_reversed")],
                       ui.wrapped_outputs[3].font_attr_segs[i])


if __name__ == "__main__":
  googletest.main()
