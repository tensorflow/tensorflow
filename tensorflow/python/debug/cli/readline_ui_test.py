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
"""Tests of the readline-based CLI."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import tempfile

from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import readline_ui
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest


class MockReadlineUI(readline_ui.ReadlineUI):
  """Test subclass of ReadlineUI that bypasses terminal manipulations."""

  def __init__(self, on_ui_exit=None, command_sequence=None):
    readline_ui.ReadlineUI.__init__(
        self, on_ui_exit=on_ui_exit,
        config=cli_config.CLIConfig(config_file_path=tempfile.mktemp()))

    self._command_sequence = command_sequence
    self._command_counter = 0

    self.observers = {"screen_outputs": []}

  def _get_user_command(self):
    command = self._command_sequence[self._command_counter]
    self._command_counter += 1
    return command

  def _display_output(self, screen_output):
    self.observers["screen_outputs"].append(screen_output)


class CursesTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    self._tmp_config_path = os.path.join(self._tmp_dir, ".tfdbg_config")
    self.assertFalse(gfile.Exists(self._tmp_config_path))
    super(CursesTest, self).setUp()

  def tearDown(self):
    file_io.delete_recursively(self._tmp_dir)
    super(CursesTest, self).tearDown()

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

    lines = ["bar"] * parsed.num_times
    return debugger_cli_common.RichTextLines(lines)

  def testUIFactoryCreatesReadlineUI(self):
    ui = ui_factory.get_ui(
        "readline",
        config=cli_config.CLIConfig(config_file_path=self._tmp_config_path))
    self.assertIsInstance(ui, readline_ui.ReadlineUI)

  def testUIFactoryRaisesExceptionOnInvalidUIType(self):
    with self.assertRaisesRegexp(ValueError, "Invalid ui_type: 'foobar'"):
      ui_factory.get_ui(
          "foobar",
          config=cli_config.CLIConfig(config_file_path=self._tmp_config_path))

  def testUIFactoryRaisesExceptionOnInvalidUITypeGivenAvailable(self):
    with self.assertRaisesRegexp(ValueError, "Invalid ui_type: 'readline'"):
      ui_factory.get_ui(
          "readline",
          available_ui_types=["curses"],
          config=cli_config.CLIConfig(config_file_path=self._tmp_config_path))

  def testRunUIExitImmediately(self):
    """Make sure that the UI can exit properly after launch."""

    ui = MockReadlineUI(command_sequence=["exit"])
    ui.run_ui()

    # No screen output should have happened.
    self.assertEqual(0, len(ui.observers["screen_outputs"]))

  def testRunUIEmptyCommand(self):
    """Issue an empty command then exit."""

    ui = MockReadlineUI(command_sequence=["", "exit"])
    ui.run_ui()
    self.assertEqual(1, len(ui.observers["screen_outputs"]))

  def testRunUIWithInitCmd(self):
    """Run UI with an initial command specified."""

    ui = MockReadlineUI(command_sequence=["exit"])

    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui(init_command="babble")

    screen_outputs = ui.observers["screen_outputs"]
    self.assertEqual(1, len(screen_outputs))
    self.assertEqual(["bar"] * 60, screen_outputs[0].lines)

  def testRunUIWithValidUsersCommands(self):
    """Run UI with an initial command specified."""

    ui = MockReadlineUI(command_sequence=["babble -n 3", "babble -n 6", "exit"])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    screen_outputs = ui.observers["screen_outputs"]
    self.assertEqual(2, len(screen_outputs))
    self.assertEqual(["bar"] * 3, screen_outputs[0].lines)
    self.assertEqual(["bar"] * 6, screen_outputs[1].lines)

  def testRunUIWithInvalidUsersCommands(self):
    """Run UI with an initial command specified."""

    ui = MockReadlineUI(command_sequence=["babble -n 3", "wobble", "exit"])
    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    screen_outputs = ui.observers["screen_outputs"]
    self.assertEqual(2, len(screen_outputs))
    self.assertEqual(["bar"] * 3, screen_outputs[0].lines)
    self.assertEqual(["ERROR: Invalid command prefix \"wobble\""],
                     screen_outputs[1].lines)

  def testRunUIWithOnUIExitCallback(self):
    observer = {"callback_invoked": False}

    def callback_for_test():
      observer["callback_invoked"] = True

    ui = MockReadlineUI(on_ui_exit=callback_for_test, command_sequence=["exit"])

    self.assertFalse(observer["callback_invoked"])
    ui.run_ui()

    self.assertEqual(0, len(ui.observers["screen_outputs"]))
    self.assertTrue(observer["callback_invoked"])

  def testIncompleteRedirectWorks(self):
    output_path = tempfile.mktemp()

    ui = MockReadlineUI(
        command_sequence=["babble -n 2 > %s" % output_path, "exit"])

    ui.register_command_handler("babble", self._babble, "")
    ui.run_ui()

    screen_outputs = ui.observers["screen_outputs"]
    self.assertEqual(1, len(screen_outputs))
    self.assertEqual(["bar"] * 2, screen_outputs[0].lines)

    with gfile.Open(output_path, "r") as f:
      self.assertEqual("bar\nbar\n", f.read())

  def testConfigSetAndShow(self):
    """Run UI with an initial command specified."""

    ui = MockReadlineUI(command_sequence=[
        "config set graph_recursion_depth 5", "config show", "exit"])
    ui.run_ui()
    outputs = ui.observers["screen_outputs"]
    self.assertEqual(
        ["Command-line configuration:",
         "",
         "  graph_recursion_depth: 5"], outputs[1].lines[:3])


if __name__ == "__main__":
  googletest.main()
