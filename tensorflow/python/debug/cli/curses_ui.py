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
"""Curses-Based Command-Line Interface of TensorFlow Debugger (tfdbg)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
from curses import textpad

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.debug.cli import debugger_cli_common


class CursesUI(object):
  """Curses-based Command-line UI.

  In this class, the methods with the prefix "_screen_" are the methods that
  interact with the actual terminal using the curses library.
  """

  CLI_PROMPT = "tfdbg> "
  CLI_EXIT_CMDS = ["exit", "quit"]
  CLI_TERMINATOR_KEY = 7  # Terminator key for input text box.
  CLI_TAB_KEY = ord("\t")

  # Possible Enter keys. 343 is curses key code for the num-pad Enter key when
  # num lock is off.
  CLI_CR_KEYS = [ord("\n"), ord("\r"), 343]

  _SCROLL_UP = "up"
  _SCROLL_DOWN = "down"
  _SCROLL_HOME = "home"
  _SCROLL_END = "end"

  def __init__(self):
    self._screen_init()
    self._screen_refresh_size()
    # TODO(cais): Error out if the size of the screen is too small.

    # Initialize some UI component size and locations.
    self._init_layout()

    self._command_handler_registry = (
        debugger_cli_common.CommandHandlerRegistry())

    self._command_history_store = debugger_cli_common.CommandHistory()

    # Active list of command history, used in history navigation.
    # _command_handler_registry holds all the history commands the CLI has
    # received, up to a size limit. _active_command_history is the history
    # currently being navigated in, e.g., using the Up/Down keys. The latter
    # can be different from the former during prefixed or regex-based history
    # navigation, e.g., when user enter the beginning of a command and hit Up.
    self._active_command_history = []

    # Pointer to the current position in the history sequence.
    # 0 means it is a new command being keyed in.
    self._command_pointer = 0

    self._command_history_limit = 100

    self._pending_command = ""

    # State related to screen output.
    self._curr_unwrapped_output = None
    self._curr_wrapped_output = None

  def _init_layout(self):
    """Initialize the layout of UI components.

    Initialize the location and size of UI components such as command textbox
    and output region according to the terminal size.
    """

    # Height of command text box
    self._command_textbox_height = 2

    self._title_row = 0

    # Top row index of the output pad.
    self._output_top_row = 1

    # Row index of scroll information line: Taking into account the zero-based
    # row indexing and the command textbox area under the scroll information
    # row.
    self._output_scroll_row = self._max_y - 1 - self._command_textbox_height

  def _screen_init(self):
    """Screen initialization.

    Creates curses stdscr and initialize the color pairs for display.
    """

    self._stdscr = curses.initscr()
    self._command_window = None

    # Prepare color pairs.
    curses.start_color()

    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_BLUE, curses.COLOR_BLACK)

    self._color_pairs = {}
    self._color_pairs["white"] = curses.color_pair(1)
    self._color_pairs["red"] = curses.color_pair(2)
    self._color_pairs["green"] = curses.color_pair(3)
    self._color_pairs["yellow"] = curses.color_pair(4)
    self._color_pairs["blue"] = curses.color_pair(5)

  def _screen_launch(self):
    """Launch the curses screen."""

    curses.noecho()
    curses.cbreak()
    self._stdscr.keypad(1)

    self._screen_create_command_window()

  def _screen_create_command_window(self):
    """Create command window according to screen size."""
    if self._command_window:
      del self._command_window

    self._command_window = curses.newwin(
        self._command_textbox_height, self._max_x - len(self.CLI_PROMPT),
        self._max_y - self._command_textbox_height, len(self.CLI_PROMPT))

  def _screen_refresh(self):
    self._stdscr.refresh()

  def _screen_terminate(self):
    """Terminate the curses screen."""

    self._stdscr.keypad(0)
    curses.nocbreak()
    curses.echo()
    curses.endwin()

  def run_ui(self, init_command=None, title=None, title_color=None):
    """Run the Curses CLI.

    Args:
      init_command: (str) Optional command to run on CLI start up.
      title: (str) Optional title to display in the CLI.
      title_color: (str) Optional color of the title, e.g., "yellow".

    Returns:
      An exit token of arbitrary type. Can be None.
    """

    self._screen_launch()

    # Optional initial command.
    if init_command is not None:
      self._dispatch_command(init_command)

    if title is not None:
      self._title(title, title_color=title_color)

    # CLI main loop.
    exit_token = self._ui_loop()

    self._screen_terminate()

    return exit_token

  def register_command_handler(self, *args, **kwargs):
    self._command_handler_registry.register_command_handler(*args, **kwargs)

  def set_help_intro(self, help_intro):
    """Set an introductory message to the help output of the command registry.

    Args:
      help_intro: (list of str) Text lines appended to the beginning of the
        the output of the command "help", as introductory information.
    """

    self._command_handler_registry.set_help_intro(help_intro=help_intro)

  def get_help(self):
    return self._command_handler_registry.get_help()

  def _screen_create_command_textbox(self, existing_command):
    """Create command textbox on screen.

    Args:
      existing_command: (str) A command string to put in the textbox right
        after its creation.
    """

    # Display the tfdbg prompt.
    self._stdscr.addstr(self._max_y - self._command_textbox_height, 0,
                        self.CLI_PROMPT, curses.A_BOLD)
    self._stdscr.refresh()

    self._command_window.clear()

    # Command text box.
    self._command_textbox = textpad.Textbox(
        self._command_window, insert_mode=True)

    # Enter existing command.
    self._auto_key_in(existing_command)

  def _ui_loop(self):
    """Command-line UI loop.

    Returns:
      An exit token of arbitrary type. The token can be None.
    """

    while True:
      # Enter history command if pointer is in history (> 0):
      if self._command_pointer > 0:
        existing_command = self._active_command_history[-self._command_pointer]
      else:
        existing_command = self._pending_command
      self._screen_create_command_textbox(existing_command)

      command, terminator, pending_command_changed = self._get_user_command()

      if terminator in self.CLI_CR_KEYS:
        exit_token = self._dispatch_command(command)
        if exit_token is not None:
          return exit_token
      elif terminator == self.CLI_TAB_KEY:
        pass  # TODO(cais): Implement tab completion.
      elif pending_command_changed:
        self._pending_command = command

    return

  def _get_user_command(self):
    """Get user command from UI.

    Returns:
      command: (str) The user-entered command.
      terminator: (str) Terminator type for the command.
        If command is a normal command entered with the Enter key, the value
        will be the key itself. If this is a tab completion call (using the
        Tab key), the value will reflect that as well.
      pending_command_changed:  (bool) If the pending command has changed.
        Used during command history navigation.
    """

    # First, reset textbox state variables.
    self._textbox_curr_terminator = None
    self._textbox_pending_command_changed = False

    command = self._screen_get_user_command()
    command = self._strip_terminator(command)
    return (command, self._textbox_curr_terminator,
            self._textbox_pending_command_changed)

  def _screen_get_user_command(self):
    return self._command_textbox.edit(validate=self._on_textbox_keypress)

  def _strip_terminator(self, command):
    for v in self.CLI_CR_KEYS:
      if v < 256:
        command = command.replace(chr(v), "")

    return command.strip()

  def _screen_refresh_size(self):
    self._max_y, self._max_x = self._stdscr.getmaxyx()

  def _dispatch_command(self, command):
    """Dispatch user command.

    Args:
      command: (str) Command to dispatch.

    Returns:
      An exit token object. None value means that the UI loop should not exit.
      A non-None value means the UI loop should exit.
    """
    if command in self.CLI_EXIT_CMDS:
      # Explicit user command-triggered exit: EXPLICIT_USER_EXIT as the exit
      # token.
      return debugger_cli_common.EXPLICIT_USER_EXIT

    prefix, args = self._parse_command(command)

    if not prefix:
      # Empty command: take no action. Should not exit.
      return

    self._command_history_store.add_command(command)

    screen_info = {"cols": self._max_x}
    if self._command_handler_registry.is_registered(prefix):
      screen_output = self._command_handler_registry.dispatch_command(
          prefix, args, screen_info=screen_info)

    else:
      screen_output = debugger_cli_common.RichTextLines([
          "ERROR: Invalid command prefix \"%s\"" % prefix
      ])

    exit_token = None
    if debugger_cli_common.EXIT_TOKEN_KEY in screen_output.annotations:
      exit_token = screen_output.annotations[
          debugger_cli_common.EXIT_TOKEN_KEY]

    # Clear active command history. Until next up/down history navigation
    # occurs, it will stay empty.
    self._active_command_history = []

    if exit_token is not None:
      return exit_token

    self._display_lines(screen_output)

    self._command_pointer = 0
    self._pending_command = ""

  def _parse_command(self, command):
    """Parse a command string into prefix and arguments.

    Args:
      command: (str) Command string to be parsed.

    Returns:
      prefix: (str) The command prefix.
      args: (list of str) The command arguments (i.e., not including the
        prefix).
    """

    # TODO(cais): Support parsing of arguments surrounded by pairs of quotes
    #   and with spaces in them.

    command = command.strip()
    if not command:
      return "", []

    # Split and remove extra spaces.
    command_items = command.split(" ")
    command_items = [item for item in command_items if item]

    return command_items[0], command_items[1:]

  def _screen_gather_textbox_str(self):
    """Gather the text string in the command text box.

    Returns:
      (str) the current text string in the command textbox, excluding any
      return keys.
    """

    txt = self._command_textbox.gather()
    return txt.strip()

  def _on_textbox_keypress(self, x):
    """Text box key validator: Callback of key strokes.

    Handles a user's keypress in the input text box. Translates certain keys to
    terminator keys for the textbox to allow its edit() method to return.
    Also handles special key-triggered events such as PgUp/PgDown scrolling of
    the screen output.

    Args:
      x: (int) Key code.

    Returns:
      (int) A translated key code. In most cases, this is identical to the
        input x. However, if x is a Return key, the return value will be
        CLI_TERMINATOR_KEY, so that the text box's edit() method can return.

    Raises:
      TypeError: If the input x is not of type int.
    """
    if not isinstance(x, int):
      raise TypeError("Key validator expected type int, received type %s" %
                      type(x))

    if x in self.CLI_CR_KEYS:
      # Make Enter key the terminator
      self._textbox_curr_terminator = x
      return self.CLI_TERMINATOR_KEY
    elif x == curses.KEY_PPAGE:
      self._scroll_output(self._SCROLL_UP)
      return x
    elif x == curses.KEY_NPAGE:
      self._scroll_output(self._SCROLL_DOWN)
      return x
    elif x == curses.KEY_HOME:
      self._scroll_output(self._SCROLL_HOME)
      return x
    elif x == curses.KEY_END:
      self._scroll_output(self._SCROLL_END)
      return x
    elif x in [curses.KEY_UP, curses.KEY_DOWN]:
      # Command history navigation.
      if not self._active_command_history:
        hist_prefix = self._screen_gather_textbox_str()
        self._active_command_history = (
            self._command_history_store.lookup_prefix(
                hist_prefix, self._command_history_limit))

      if self._active_command_history:
        if x == curses.KEY_UP:
          if self._command_pointer < len(self._active_command_history):
            self._command_pointer += 1
        elif x == curses.KEY_DOWN:
          if self._command_pointer > 0:
            self._command_pointer -= 1
      else:
        self._command_pointer = 0

      self._textbox_curr_terminator = x

      # Force return from the textbox edit(), so that the textbox can be
      # redrawn with a history command entered.
      return self.CLI_TERMINATOR_KEY
    elif x == curses.KEY_RESIZE:
      # Respond to terminal resize.
      self._screen_refresh_size()
      self._init_layout()
      self._screen_create_command_window()
      if self._curr_unwrapped_output is not None:
        # Force render screen output again, under new screen size.
        self._display_lines(self._curr_unwrapped_output)

      # Force return from the textbox edit(), so that the textbox can be
      # redrawn.
      return self.CLI_TERMINATOR_KEY
    else:
      # Mark the pending command as modified.
      self._textbox_pending_command_changed = True
      # Invalidate active command history.
      self._command_pointer = 0
      self._active_command_history = []
      return x

  def _title(self, title, title_color=None):
    """Display title.

    Args:
      title: (str) The title to display.
      title_color: (str) Color of the title, e.g., "yellow".
    """

    # Pad input title str with "-" and space characters to make it pretty.
    self._title_line = "--- %s " % title
    if len(self._title_line) < self._max_x:
      self._title_line += "-" * (self._max_x - len(self._title_line))

    self._screen_draw_text_line(
        self._title_row, self._title_line, color=title_color)

  def _auto_key_in(self, command):
    """Automatically key in a command to the command Textbox.

    Args:
      command: The command, as a string.
    """
    for c in command:
      self._command_textbox.do_command(ord(c))

  def _screen_draw_text_line(self, row, line, attr=curses.A_NORMAL, color=None):
    """Render a line of text on the screen.

    Args:
      row: (int) Row index.
      line: (str) The line content.
      attr: curses font attribute.
      color: (str) font foreground color name.

    Raises:
      TypeError: If row is not of type int.
    """

    if not isinstance(row, int):
      raise TypeError("Invalid type in row")

    if len(line) > self._max_x:
      line = line[:self._max_x]

    if color is None:
      self._stdscr.addstr(row, 0, line, attr)
    else:
      self._stdscr.addstr(row, 0, line, self._color_pairs[color])
    self._screen_refresh()

  def _screen_new_output_pad(self, rows, cols):
    self._output_pad = curses.newpad(rows, cols)

  def _display_lines(self, output):
    """Display RichTextLines object on screen.

    Args:
      output: A RichTextLines object.

    Raises:
      ValueError: If input argument "output" is invalid.
    """

    if not isinstance(output, debugger_cli_common.RichTextLines):
      raise ValueError(
          "Output is required to be an instance of RichTextLines, but is not.")

    self._curr_unwrapped_output = output

    # TODO(cais): Cut off output with too many lines to prevent overflow issues
    # in curses.

    cols = self._max_x
    self._curr_wrapped_output = debugger_cli_common.wrap_rich_text_lines(
        output, cols - 1)

    self._screen_refresh()

    # Minimum number of rows that the output area has to have: Screen height
    # space above the output region, the height of the command textbox and
    # the single scroll information row.
    min_rows = (
        self._max_y - self._output_top_row - self._command_textbox_height - 1)

    rows = max(min_rows, len(self._curr_wrapped_output.lines))

    # Size of the output pad, which may exceed screen size and require
    # scrolling.
    self._output_pad_height = rows
    self._output_pad_width = cols

    # Size of view port on screen, which is always smaller or equal to the
    # screen size.
    self._output_pad_scr_height = min_rows - 1
    self._output_pad_scr_width = cols

    # Create new output pad.
    self._screen_new_output_pad(rows, cols)

    for i in xrange(len(self._curr_wrapped_output.lines)):
      if i in self._curr_wrapped_output.font_attr_segs:
        self._screen_add_line_to_output_pad(
            i,
            self._curr_wrapped_output.lines[i],
            color_segments=self._curr_wrapped_output.font_attr_segs[i])
      else:
        self._screen_add_line_to_output_pad(i,
                                            self._curr_wrapped_output.lines[i])

    # 1st row of the output pad to be displayed: Scroll to top first.
    self._output_pad_row = 0

    # The location of the rectangular viewport on the screen.
    self._output_pad_scr_loc = [
        self._output_top_row, 0, self._output_top_row + min_rows, cols
    ]
    self._scroll_output("home")

  def _screen_add_line_to_output_pad(self, row, txt, color_segments=None):
    """Render a line in screen output pad.

    Assumes: segments in color_segments are sorted in ascending order of the
    beginning index.
    Note: Gaps between the segments are allowed and will be fixed in with a
    default color.

    Args:
      row: Row index, as an int.
      txt: The text to be displayed on the specified row, as a str.
      color_segments: A list of 3-tuples. Each tuple represents the beginning
        and the end of a color segment, in the form of a right-open interval:
        [start, end). The last element of the tuple is a color string, e.g.,
        "red".

    Raisee:
      TypeError: If color_segments is not of type list.
    """

    default_color_pair = self._color_pairs["white"]

    if not color_segments:
      self._output_pad.addstr(row, 0, txt, default_color_pair)
      return

    if not isinstance(color_segments, list):
      raise TypeError("Input color_segments needs to be a list, but is not.")

    all_segments = []
    all_color_pairs = []

    # Process the beginning.
    if color_segments[0][0] == 0:
      pass
    else:
      all_segments.append((0, color_segments[0][0]))
      all_color_pairs.append(default_color_pair)

    for (curr_start, curr_end, curr_color), (next_start, _, _) in zip(
        color_segments, color_segments[1:] + [(len(txt), None, None)]):
      all_segments.append((curr_start, curr_end))

      # TODO(cais): Deal with the case in which the color pair is unavailable.
      all_color_pairs.append(self._color_pairs[curr_color])

      if curr_end < next_start:
        # Fill in the gap with the default color.
        all_segments.append((curr_end, next_start))
        all_color_pairs.append(default_color_pair)

    # Finally, draw all the segments.
    for segment, color_pair in zip(all_segments, all_color_pairs):
      self._output_pad.addstr(row, segment[0], txt[segment[0]:segment[1]],
                              color_pair)

  def _screen_scroll_output_pad(self):
    self._output_pad.refresh(self._output_pad_row, 0,
                             self._output_pad_scr_loc[0],
                             self._output_pad_scr_loc[1],
                             self._output_pad_scr_loc[2],
                             self._output_pad_scr_loc[3])

  def _scroll_output(self, direction):
    """Scroll the output pad.

    Args:
      direction: _SCROLL_UP, _SCROLL_DOWN, _SCROLL_HOME or _SCROLL_END

    Raises:
      ValueError: On invalid scroll direction.
    """

    if not self._output_pad:
      # No output pad is present. Do nothing.
      return

    if direction == self._SCROLL_UP:
      # Scroll up
      if self._output_pad_row - 1 >= 0:
        self._output_pad_row -= 1
    elif direction == self._SCROLL_DOWN:
      # Scroll down
      if self._output_pad_row + 1 < (
          self._output_pad_height - self._output_pad_scr_height):
        self._output_pad_row += 1
    elif direction == self._SCROLL_HOME:
      # Scroll to top
      self._output_pad_row = 0
    elif direction == self._SCROLL_END:
      # Scroll to bottom
      self._output_pad_row = (
          self._output_pad_height - self._output_pad_scr_height - 1)
    else:
      raise ValueError("Unsupported scroll mode: %s" % direction)

    # Actually scroll the output pad: refresh with new location.
    self._screen_scroll_output_pad()

    if self._output_pad_height > self._output_pad_scr_height + 1:
      # Display information about the scrolling of tall screen output.
      self._scroll_info = "--- Scroll: %.2f%% " % (100.0 * (
          float(self._output_pad_row) /
          (self._output_pad_height - self._output_pad_scr_height - 1)))
      if len(self._scroll_info) < self._max_x:
        self._scroll_info += "-" * (self._max_x - len(self._scroll_info))
      self._screen_draw_text_line(
          self._output_scroll_row, self._scroll_info, color="green")
    else:
      # Screen output is not tall enough to cause scrolling.
      self._scroll_info = "-" * self._max_x
      self._screen_draw_text_line(
          self._output_scroll_row, self._scroll_info, color="green")
