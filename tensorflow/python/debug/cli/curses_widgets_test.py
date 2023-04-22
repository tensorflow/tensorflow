# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for curses-based CLI widgets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.debug.cli import curses_widgets
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

RTL = debugger_cli_common.RichTextLines
CNH = curses_widgets.CursesNavigationHistory


class CNHTest(test_util.TensorFlowTestCase):

  def testConstructorWorks(self):
    CNH(10)

  def testConstructorWithInvalidCapacityErrors(self):
    with self.assertRaises(ValueError):
      CNH(0)
    with self.assertRaises(ValueError):
      CNH(-1)

  def testInitialStateIsCorrect(self):
    nav_history = CNH(10)
    self.assertEqual(0, nav_history.size())
    self.assertFalse(nav_history.can_go_forward())
    self.assertFalse(nav_history.can_go_back())

    with self.assertRaisesRegex(ValueError, "Empty navigation history"):
      nav_history.go_back()
    with self.assertRaisesRegex(ValueError, "Empty navigation history"):
      nav_history.go_forward()
    with self.assertRaisesRegex(ValueError, "Empty navigation history"):
      nav_history.update_scroll_position(3)

  def testAddOneItemWorks(self):
    nav_history = CNH(10)
    nav_history.add_item("foo", RTL(["bar"]), 0)

    self.assertEqual(1, nav_history.size())
    self.assertEqual(0, nav_history.pointer())

    self.assertFalse(nav_history.can_go_forward())
    self.assertFalse(nav_history.can_go_back())

    output = nav_history.go_back()
    self.assertEqual("foo", output.command)
    self.assertEqual(["bar"], output.screen_output.lines)
    self.assertEqual(0, output.scroll_position)

  def testAddItemsBeyondCapacityWorks(self):
    nav_history = CNH(2)
    nav_history.add_item("foo", RTL(["foo_output"]), 0)
    nav_history.add_item("bar", RTL(["bar_output"]), 0)

    self.assertEqual(2, nav_history.size())
    self.assertEqual(1, nav_history.pointer())
    self.assertTrue(nav_history.can_go_back())
    self.assertFalse(nav_history.can_go_forward())

    nav_history.add_item("baz", RTL(["baz_output"]), 0)

    self.assertEqual(2, nav_history.size())
    self.assertEqual(1, nav_history.pointer())
    self.assertTrue(nav_history.can_go_back())
    self.assertFalse(nav_history.can_go_forward())

    item = nav_history.go_back()
    self.assertEqual("bar", item.command)
    self.assertFalse(nav_history.can_go_back())
    self.assertTrue(nav_history.can_go_forward())

    item = nav_history.go_forward()
    self.assertEqual("baz", item.command)
    self.assertTrue(nav_history.can_go_back())
    self.assertFalse(nav_history.can_go_forward())

  def testAddItemFromNonLatestPointerPositionWorks(self):
    nav_history = CNH(2)
    nav_history.add_item("foo", RTL(["foo_output"]), 0)
    nav_history.add_item("bar", RTL(["bar_output"]), 0)

    nav_history.go_back()
    nav_history.add_item("baz", RTL(["baz_output"]), 0)

    self.assertEqual(2, nav_history.size())
    self.assertEqual(1, nav_history.pointer())
    self.assertTrue(nav_history.can_go_back())
    self.assertFalse(nav_history.can_go_forward())

    item = nav_history.go_back()
    self.assertEqual("foo", item.command)
    item = nav_history.go_forward()
    self.assertEqual("baz", item.command)

  def testUpdateScrollPositionOnLatestItemWorks(self):
    nav_history = CNH(2)
    nav_history.add_item("foo", RTL(["foo_out", "more_foo_out"]), 0)
    nav_history.add_item("bar", RTL(["bar_out", "more_bar_out"]), 0)

    nav_history.update_scroll_position(1)
    nav_history.go_back()
    item = nav_history.go_forward()
    self.assertEqual("bar", item.command)
    self.assertEqual(1, item.scroll_position)

  def testUpdateScrollPositionOnOldItemWorks(self):
    nav_history = CNH(2)
    nav_history.add_item("foo", RTL(["foo_out", "more_foo_out"]), 0)
    nav_history.add_item("bar", RTL(["bar_out", "more_bar_out"]), 0)

    item = nav_history.go_back()
    self.assertEqual("foo", item.command)
    self.assertEqual(0, item.scroll_position)

    nav_history.update_scroll_position(1)
    nav_history.go_forward()
    item = nav_history.go_back()
    self.assertEqual("foo", item.command)
    self.assertEqual(1, item.scroll_position)

    item = nav_history.go_forward()
    self.assertEqual("bar", item.command)
    self.assertEqual(0, item.scroll_position)

  def testRenderWithEmptyHistoryWorks(self):
    nav_history = CNH(2)

    output = nav_history.render(40, "prev", "next")
    self.assertEqual(1, len(output.lines))
    self.assertEqual(
        "| " + CNH.BACK_ARROW_TEXT + " " + CNH.FORWARD_ARROW_TEXT,
        output.lines[0])
    self.assertEqual({}, output.font_attr_segs)

  def testRenderLatestWithSufficientLengthWorks(self):
    nav_history = CNH(2)
    nav_history.add_item("foo", RTL(["foo_out", "more_foo_out"]), 0)
    nav_history.add_item("bar", RTL(["bar_out", "more_bar_out"]), 0)

    output = nav_history.render(
        40,
        "prev",
        "next",
        latest_command_attribute="green",
        old_command_attribute="yellow")
    self.assertEqual(1, len(output.lines))
    self.assertEqual(
        "| " + CNH.BACK_ARROW_TEXT + " " + CNH.FORWARD_ARROW_TEXT +
        " | bar",
        output.lines[0])
    self.assertEqual(2, output.font_attr_segs[0][0][0])
    self.assertEqual(5, output.font_attr_segs[0][0][1])
    self.assertEqual("prev", output.font_attr_segs[0][0][2].content)

    self.assertEqual(12, output.font_attr_segs[0][1][0])
    self.assertEqual(15, output.font_attr_segs[0][1][1])
    self.assertEqual("green", output.font_attr_segs[0][1][2])

  def testRenderOldButNotOldestWithSufficientLengthWorks(self):
    nav_history = CNH(3)
    nav_history.add_item("foo", RTL(["foo_out", "more_foo_out"]), 0)
    nav_history.add_item("bar", RTL(["bar_out", "more_bar_out"]), 0)
    nav_history.add_item("baz", RTL(["baz_out", "more_baz_out"]), 0)

    nav_history.go_back()

    output = nav_history.render(
        40,
        "prev",
        "next",
        latest_command_attribute="green",
        old_command_attribute="yellow")
    self.assertEqual(1, len(output.lines))
    self.assertEqual(
        "| " + CNH.BACK_ARROW_TEXT + " " + CNH.FORWARD_ARROW_TEXT +
        " | (-1) bar",
        output.lines[0])
    self.assertEqual(2, output.font_attr_segs[0][0][0])
    self.assertEqual(5, output.font_attr_segs[0][0][1])
    self.assertEqual("prev", output.font_attr_segs[0][0][2].content)

    self.assertEqual(6, output.font_attr_segs[0][1][0])
    self.assertEqual(9, output.font_attr_segs[0][1][1])
    self.assertEqual("next", output.font_attr_segs[0][1][2].content)

    self.assertEqual(12, output.font_attr_segs[0][2][0])
    self.assertEqual(17, output.font_attr_segs[0][2][1])
    self.assertEqual("yellow", output.font_attr_segs[0][2][2])

    self.assertEqual(17, output.font_attr_segs[0][3][0])
    self.assertEqual(20, output.font_attr_segs[0][3][1])
    self.assertEqual("yellow", output.font_attr_segs[0][3][2])

  def testRenderOldestWithSufficientLengthWorks(self):
    nav_history = CNH(3)
    nav_history.add_item("foo", RTL(["foo_out", "more_foo_out"]), 0)
    nav_history.add_item("bar", RTL(["bar_out", "more_bar_out"]), 0)
    nav_history.add_item("baz", RTL(["baz_out", "more_baz_out"]), 0)

    nav_history.go_back()
    nav_history.go_back()

    output = nav_history.render(
        40,
        "prev",
        "next",
        latest_command_attribute="green",
        old_command_attribute="yellow")
    self.assertEqual(1, len(output.lines))
    self.assertEqual(
        "| " + CNH.BACK_ARROW_TEXT + " " + CNH.FORWARD_ARROW_TEXT +
        " | (-2) foo",
        output.lines[0])
    self.assertEqual(6, output.font_attr_segs[0][0][0])
    self.assertEqual(9, output.font_attr_segs[0][0][1])
    self.assertEqual("next", output.font_attr_segs[0][0][2].content)

    self.assertEqual(12, output.font_attr_segs[0][1][0])
    self.assertEqual(17, output.font_attr_segs[0][1][1])
    self.assertEqual("yellow", output.font_attr_segs[0][1][2])

    self.assertEqual(17, output.font_attr_segs[0][2][0])
    self.assertEqual(20, output.font_attr_segs[0][2][1])
    self.assertEqual("yellow", output.font_attr_segs[0][2][2])

  def testRenderWithInsufficientLengthWorks(self):
    nav_history = CNH(2)
    nav_history.add_item("long_command", RTL(["output"]), 0)

    output = nav_history.render(
        15,
        "prev",
        "next",
        latest_command_attribute="green",
        old_command_attribute="yellow")
    self.assertEqual(1, len(output.lines))
    self.assertEqual(
        "| " + CNH.BACK_ARROW_TEXT + " " + CNH.FORWARD_ARROW_TEXT +
        " | lon",
        output.lines[0])

    self.assertEqual(12, output.font_attr_segs[0][0][0])
    self.assertEqual(15, output.font_attr_segs[0][0][1])
    self.assertEqual("green", output.font_attr_segs[0][0][2])


if __name__ == "__main__":
  googletest.main()
