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
"""Tests for Building Blocks of the TensorFlow Debugger CLI."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class RichTextLinesTest(test_util.TensorFlowTestCase):

  def testRichTextLinesConstructorComplete(self):
    # Test RichTextLines constructor.
    screen_output = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"],
        font_attr_segs={0: [(0, 5, "red")],
                        1: [(0, 7, "blue")]},
        annotations={0: "longer wavelength",
                     1: "shorter wavelength"})

    self.assertEqual(2, len(screen_output.lines))
    self.assertEqual(2, len(screen_output.font_attr_segs))
    self.assertEqual(1, len(screen_output.font_attr_segs[0]))
    self.assertEqual(1, len(screen_output.font_attr_segs[1]))
    self.assertEqual(2, len(screen_output.annotations))

  def testRichTextLinesConstructorWithInvalidType(self):
    with self.assertRaisesRegexp(ValueError, "Unexpected type in lines"):
      debugger_cli_common.RichTextLines(123)

  def testRichTextLinesConstructorWithString(self):
    # Test constructing a RichTextLines object with a string, instead of a list
    # of strings.
    screen_output = debugger_cli_common.RichTextLines(
        "Roses are red",
        font_attr_segs={0: [(0, 5, "red")]},
        annotations={0: "longer wavelength"})

    self.assertEqual(1, len(screen_output.lines))
    self.assertEqual(1, len(screen_output.font_attr_segs))
    self.assertEqual(1, len(screen_output.font_attr_segs[0]))
    self.assertEqual(1, len(screen_output.annotations))

  def testRichTextLinesConstructorIncomplete(self):
    # Test RichTextLines constructor, with incomplete keyword arguments.
    screen_output = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"],
        font_attr_segs={0: [(0, 5, "red")],
                        1: [(0, 7, "blue")]})

    self.assertEqual(2, len(screen_output.lines))
    self.assertEqual(2, len(screen_output.font_attr_segs))
    self.assertEqual(1, len(screen_output.font_attr_segs[0]))
    self.assertEqual(1, len(screen_output.font_attr_segs[1]))
    self.assertEqual({}, screen_output.annotations)

  def testModifyRichTextLinesObject(self):
    screen_output = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"])

    self.assertEqual(2, len(screen_output.lines))

    screen_output.lines.append("Sugar is sweet")
    self.assertEqual(3, len(screen_output.lines))


class CommandHandlerRegistryTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._intentional_error_msg = "Intentionally raised exception"

  def _noop_handler(self, argv, screen_info=None):
    # A handler that does nothing other than returning "Done."
    return debugger_cli_common.RichTextLines(["Done."])

  def _handler_raising_exception(self, argv, screen_info=None):
    # A handler that intentionally raises an exception.
    raise RuntimeError(self._intentional_error_msg)

  def _handler_returning_wrong_type(self, argv, screen_info=None):
    # A handler that returns a wrong type, instead of the correct type
    # (RichTextLines).
    return "Hello"

  def _echo_screen_cols(self, argv, screen_info=None):
    # A handler that uses screen_info.
    return debugger_cli_common.RichTextLines(
        ["cols = %d" % screen_info["cols"]])

  def testRegisterEmptyCommandPrefix(self):
    registry = debugger_cli_common.CommandHandlerRegistry()

    # Attempt to register an empty-string as a command prefix should trigger
    # an exception.
    with self.assertRaisesRegexp(ValueError, "Empty command prefix"):
      registry.register_command_handler("", self._noop_handler, "")

  def testRegisterAndInvokeHandler(self):
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler("noop", self._noop_handler, "")

    self.assertTrue(registry.is_registered("noop"))
    self.assertFalse(registry.is_registered("beep"))

    cmd_output = registry.dispatch_command("noop", [])
    self.assertEqual(["Done."], cmd_output.lines)

    # Attempt to invoke an unregistered command prefix should trigger an
    # exception.
    with self.assertRaisesRegexp(ValueError, "No handler is registered"):
      registry.dispatch_command("beep", [])

    # Empty command prefix should trigger an exception.
    with self.assertRaisesRegexp(ValueError, "Prefix is empty"):
      registry.dispatch_command("", [])

  def testInvokeHandlerWithScreenInfo(self):
    registry = debugger_cli_common.CommandHandlerRegistry()

    # Register and invoke a command handler that uses screen_info.
    registry.register_command_handler("cols", self._echo_screen_cols, "")

    cmd_output = registry.dispatch_command(
        "cols", [], screen_info={"cols": 100})
    self.assertEqual(["cols = 100"], cmd_output.lines)

  def testRegisterAndInvokeHandlerWithAliases(self):
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler(
        "noop", self._noop_handler, "", prefix_aliases=["n", "NOOP"])

    # is_registered() should work for full prefix and aliases.
    self.assertTrue(registry.is_registered("noop"))
    self.assertTrue(registry.is_registered("n"))
    self.assertTrue(registry.is_registered("NOOP"))

    cmd_output = registry.dispatch_command("n", [])
    self.assertEqual(["Done."], cmd_output.lines)

    cmd_output = registry.dispatch_command("NOOP", [])
    self.assertEqual(["Done."], cmd_output.lines)

  def testHandlerWithWrongReturnType(self):
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler("wrong_return",
                                      self._handler_returning_wrong_type, "")

    # If the command handler fails to return a RichTextLines instance, an error
    # should be triggered.
    with self.assertRaisesRegexp(
        ValueError,
        "Return value from command handler.*is not a RichTextLines instance"):
      registry.dispatch_command("wrong_return", [])

  def testRegisterDuplicateHandlers(self):
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler("noop", self._noop_handler, "")

    # Registering the same command prefix more than once should trigger an
    # exception.
    with self.assertRaisesRegexp(
        ValueError, "A handler is already registered for command prefix"):
      registry.register_command_handler("noop", self._noop_handler, "")

    cmd_output = registry.dispatch_command("noop", [])
    self.assertEqual(["Done."], cmd_output.lines)

  def testRegisterDuplicateAliases(self):
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler(
        "noop", self._noop_handler, "", prefix_aliases=["n"])

    # Clash with existing alias.
    with self.assertRaisesRegexp(ValueError,
                                 "clashes with existing prefixes or aliases"):
      registry.register_command_handler(
          "cols", self._echo_screen_cols, "", prefix_aliases=["n"])

    # The name clash should have prevent the handler from being registered.
    self.assertFalse(registry.is_registered("cols"))

    # Aliases can also clash with command prefixes.
    with self.assertRaisesRegexp(ValueError,
                                 "clashes with existing prefixes or aliases"):
      registry.register_command_handler(
          "cols", self._echo_screen_cols, "", prefix_aliases=["noop"])

    self.assertFalse(registry.is_registered("cols"))

  def testDispatchHandlerRaisingException(self):
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler("raise_exception",
                                      self._handler_raising_exception, "")

    # The registry should catch and wrap exceptions that occur during command
    # handling.
    cmd_output = registry.dispatch_command("raise_exception", [])
    self.assertEqual(2, len(cmd_output.lines))
    self.assertTrue(cmd_output.lines[0].startswith(
        "Error occurred during handling of command"))
    self.assertTrue(cmd_output.lines[1].endswith(self._intentional_error_msg))

  def testRegisterNonCallableHandler(self):
    registry = debugger_cli_common.CommandHandlerRegistry()

    # Attempt to register a non-callable handler should fail.
    with self.assertRaisesRegexp(ValueError, "handler is not callable"):
      registry.register_command_handler("non_callable", 1, "")

  def testRegisterHandlerWithInvalidHelpInfoType(self):
    registry = debugger_cli_common.CommandHandlerRegistry()

    with self.assertRaisesRegexp(ValueError, "help_info is not a str"):
      registry.register_command_handler("noop", self._noop_handler, ["foo"])

  def testGetHelpFull(self):
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler(
        "noop",
        self._noop_handler,
        "No operation.\nI.e., do nothing.",
        prefix_aliases=["n", "NOOP"])
    registry.register_command_handler(
        "cols",
        self._echo_screen_cols,
        "Show screen width in number of columns.",
        prefix_aliases=["c"])

    help_lines = registry.get_help().lines

    # The help info should list commands in alphabetically sorted order,
    # regardless of order in which the commands are reigstered.
    self.assertEqual("cols", help_lines[0])
    self.assertTrue(help_lines[1].endswith("Aliases: c"))
    self.assertFalse(help_lines[2])
    self.assertTrue(help_lines[3].endswith(
        "Show screen width in number of columns."))

    self.assertFalse(help_lines[4])
    self.assertFalse(help_lines[5])

    self.assertEqual("noop", help_lines[6])
    self.assertTrue(help_lines[7].endswith("Aliases: n, NOOP"))
    self.assertFalse(help_lines[8])
    self.assertTrue(help_lines[9].endswith("No operation."))
    self.assertTrue(help_lines[10].endswith("I.e., do nothing."))

  def testGetHelpSingleCommand(self):
    registry = debugger_cli_common.CommandHandlerRegistry()
    registry.register_command_handler(
        "noop",
        self._noop_handler,
        "No operation.\nI.e., do nothing.",
        prefix_aliases=["n", "NOOP"])
    registry.register_command_handler(
        "cols",
        self._echo_screen_cols,
        "Show screen width in number of columns.",
        prefix_aliases=["c"])

    # Get help info for one of the two commands, using full prefix.
    help_lines = registry.get_help("cols").lines

    self.assertTrue(help_lines[0].endswith("cols"))
    self.assertTrue(help_lines[1].endswith("Aliases: c"))
    self.assertFalse(help_lines[2])
    self.assertTrue(help_lines[3].endswith(
        "Show screen width in number of columns."))

    # Get help info for one of the two commands, using alias.
    help_lines = registry.get_help("c").lines

    self.assertTrue(help_lines[0].endswith("cols"))
    self.assertTrue(help_lines[1].endswith("Aliases: c"))
    self.assertFalse(help_lines[2])
    self.assertTrue(help_lines[3].endswith(
        "Show screen width in number of columns."))

    # Get help info for a nonexistent command.
    help_lines = registry.get_help("foo").lines

    self.assertEqual("Invalid command prefix: \"foo\"", help_lines[0])


class RegexFindTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._orig_screen_output = debugger_cli_common.RichTextLines(
        ["Roses are red", "Violets are blue"])

  def testRegexFindWithoutExistingFontAttrSegs(self):
    new_screen_output = debugger_cli_common.regex_find(self._orig_screen_output,
                                                       "are", "yellow")

    self.assertEqual(2, len(new_screen_output.font_attr_segs))
    self.assertEqual([(6, 9, "yellow")], new_screen_output.font_attr_segs[0])
    self.assertEqual([(8, 11, "yellow")], new_screen_output.font_attr_segs[1])

  def testRegexFindWithExistingFontAttrSegs(self):
    # Add a font attribute segment first.
    self._orig_screen_output.font_attr_segs[0] = [(9, 12, "red")]
    self.assertEqual(1, len(self._orig_screen_output.font_attr_segs))

    new_screen_output = debugger_cli_common.regex_find(self._orig_screen_output,
                                                       "are", "yellow")
    self.assertEqual(2, len(new_screen_output.font_attr_segs))

    self.assertEqual([(6, 9, "yellow"), (9, 12, "red")],
                     new_screen_output.font_attr_segs[0])


class WrapScreenOuptutTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._orig_screen_output = debugger_cli_common.RichTextLines(
        ["Folk song:", "Roses are red", "Violets are blue"],
        font_attr_segs={1: [(0, 5, "red"), (6, 9, "gray"), (10, 12, "red"),
                            (12, 13, "crimson")],
                        2: [(0, 7, "blue"), (8, 11, "gray"), (12, 14, "blue"),
                            (14, 16, "indigo")]},
        annotations={1: "longer wavelength",
                     2: "shorter wavelength"})

  def testNoActualWrapping(self):
    # Large column limit should lead to no actual wrapping.
    out = debugger_cli_common.wrap_rich_text_lines(self._orig_screen_output,
                                                   100)

    self.assertEqual(self._orig_screen_output.lines, out.lines)
    self.assertEqual(self._orig_screen_output.font_attr_segs,
                     out.font_attr_segs)
    self.assertEqual(self._orig_screen_output.annotations, out.annotations)

  def testWrappingWithAttrCutoff(self):
    out = debugger_cli_common.wrap_rich_text_lines(self._orig_screen_output, 11)

    # Check wrapped text.
    self.assertEqual(5, len(out.lines))
    self.assertEqual("Folk song:", out.lines[0])
    self.assertEqual("Roses are r", out.lines[1])
    self.assertEqual("ed", out.lines[2])
    self.assertEqual("Violets are", out.lines[3])
    self.assertEqual(" blue", out.lines[4])

    # Check wrapped font_attr_segs.
    self.assertFalse(0 in out.font_attr_segs)
    self.assertEqual([(0, 5, "red"), (6, 9, "gray"), (10, 11, "red")],
                     out.font_attr_segs[1])
    self.assertEqual([(0, 1, "red"), (1, 2, "crimson")], out.font_attr_segs[2])
    self.assertEqual([(0, 7, "blue"), (8, 11, "gray")], out.font_attr_segs[3])
    self.assertEqual([(1, 3, "blue"), (3, 5, "indigo")], out.font_attr_segs[4])

    # Check annotations.
    self.assertFalse(0 in out.annotations)
    self.assertEqual("longer wavelength", out.annotations[1])
    self.assertFalse(2 in out.annotations)
    self.assertEqual("shorter wavelength", out.annotations[3])
    self.assertFalse(4 in out.annotations)

  def testWrappingWithMultipleAttrCutoff(self):
    self._orig_screen_output = debugger_cli_common.RichTextLines(
        ["Folk song:", "Roses are red", "Violets are blue"],
        font_attr_segs={1: [(0, 12, "red")],
                        2: [(1, 16, "blue")]},
        annotations={1: "longer wavelength",
                     2: "shorter wavelength"})

    out = debugger_cli_common.wrap_rich_text_lines(self._orig_screen_output, 5)

    # Check wrapped text.
    self.assertEqual(9, len(out.lines))
    self.assertEqual("Folk ", out.lines[0])
    self.assertEqual("song:", out.lines[1])
    self.assertEqual("Roses", out.lines[2])
    self.assertEqual(" are ", out.lines[3])
    self.assertEqual("red", out.lines[4])
    self.assertEqual("Viole", out.lines[5])
    self.assertEqual("ts ar", out.lines[6])
    self.assertEqual("e blu", out.lines[7])
    self.assertEqual("e", out.lines[8])

    # Check wrapped font_attr_segs.
    self.assertFalse(0 in out.font_attr_segs)
    self.assertFalse(1 in out.font_attr_segs)
    self.assertEqual([(0, 5, "red")], out.font_attr_segs[2])
    self.assertEqual([(0, 5, "red")], out.font_attr_segs[3])
    self.assertEqual([(0, 2, "red")], out.font_attr_segs[4])
    self.assertEqual([(1, 5, "blue")], out.font_attr_segs[5])
    self.assertEqual([(0, 5, "blue")], out.font_attr_segs[6])
    self.assertEqual([(0, 5, "blue")], out.font_attr_segs[7])
    self.assertEqual([(0, 1, "blue")], out.font_attr_segs[8])

    # Check annotations.
    self.assertFalse(0 in out.annotations)
    self.assertFalse(1 in out.annotations)
    self.assertEqual("longer wavelength", out.annotations[2])
    self.assertFalse(3 in out.annotations)
    self.assertFalse(4 in out.annotations)
    self.assertEqual("shorter wavelength", out.annotations[5])
    self.assertFalse(6 in out.annotations)
    self.assertFalse(7 in out.annotations)
    self.assertFalse(8 in out.annotations)

  def testWrappingInvalidArguments(self):
    with self.assertRaisesRegexp(ValueError,
                                 "Invalid type of input screen_output"):
      debugger_cli_common.wrap_rich_text_lines("foo", 12)

    with self.assertRaisesRegexp(ValueError, "Invalid type of input cols"):
      debugger_cli_common.wrap_rich_text_lines(
          debugger_cli_common.RichTextLines(["foo", "bar"]), "12")


if __name__ == "__main__":
  googletest.main()
