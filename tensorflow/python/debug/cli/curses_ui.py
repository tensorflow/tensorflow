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

import collections
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
  CLI_EXIT_COMMANDS = ["exit", "quit"]
  CLI_TERMINATOR_KEY = 7  # Terminator key for input text box.
  CLI_TAB_KEY = ord("\t")
  REGEX_SEARCH_PREFIX = "/"

  # Possible Enter keys. 343 is curses key code for the num-pad Enter key when
  # num lock is off.
  CLI_CR_KEYS = [ord("\n"), ord("\r"), 343]

  _SCROLL_REFRESH = "refresh"
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

    # Create tab completion registry and register the empty-str (top-level)
    # tab-completion context with it.
    self._tab_completion_registry = debugger_cli_common.TabCompletionRegistry()

    # Create top-level tab-completion context and register the exit and help
    # commands.
    self._tab_completion_registry.register_tab_comp_context(
        [""], self.CLI_EXIT_COMMANDS +
        [debugger_cli_common.CommandHandlerRegistry.HELP_COMMAND] +
        debugger_cli_common.CommandHandlerRegistry.HELP_COMMAND_ALIASES)

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
    self._output_pad = None
    self._curr_unwrapped_output = None
    self._curr_wrapped_output = None

    # NamedTuple for rectangular locations on screen
    self.rectangle = collections.namedtuple("rectangle",
                                            "top left bottom right")

  def _init_layout(self):
    """Initialize the layout of UI components.

    Initialize the location and size of UI components such as command textbox
    and output region according to the terminal size.
    """

    # Height of command text box
    self._command_textbox_height = 2

    self._title_row = 0

    # Top row index of the output pad.
    # A "pad" is a curses object that holds lines of text and not limited to
    # screen size. It can be rendered on the screen partially with scroll
    # parameters specified.
    self._output_top_row = 1

    # Number of rows that the output pad has.
    self._output_num_rows = (
        self._max_y - self._output_top_row - self._command_textbox_height - 1)

    # Row index of scroll information line: Taking into account the zero-based
    # row indexing and the command textbox area under the scroll information
    # row.
    self._output_scroll_row = self._max_y - 1 - self._command_textbox_height

    # Tab completion bottom row.
    self._candidates_top_row = self._output_scroll_row - 4
    self._candidates_bottom_row = self._output_scroll_row - 1

    # Maximum number of lines the candidates display can have.
    self._candidates_max_lines = int(self._output_num_rows / 2)

    # Font attribute for search and highlighting.
    self._search_highlight_font_attr = "bw_reversed"

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
    curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_WHITE)

    self._color_pairs = {}
    self._color_pairs["white"] = curses.color_pair(1)
    self._color_pairs["red"] = curses.color_pair(2)
    self._color_pairs["green"] = curses.color_pair(3)
    self._color_pairs["yellow"] = curses.color_pair(4)
    self._color_pairs["blue"] = curses.color_pair(5)

    # Black-white reversed
    self._color_pairs["bw_reversed"] = curses.color_pair(6)

    # A_BOLD is not really a "color". But place it here for convenience.
    self._color_pairs["bold"] = curses.A_BOLD

    # Default color pair to use when a specified color pair does not exist.
    self._default_color_pair = self._color_pairs["white"]

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
        tab_completed = self._tab_complete(command)
        self._pending_command = tab_completed
        self._cmd_ptr = 0
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

    if command in self.CLI_EXIT_COMMANDS:
      # Explicit user command-triggered exit: EXPLICIT_USER_EXIT as the exit
      # token.
      return debugger_cli_common.EXPLICIT_USER_EXIT

    if command:
      self._command_history_store.add_command(command)

    if (len(command) > len(self.REGEX_SEARCH_PREFIX) and
        command.startswith(self.REGEX_SEARCH_PREFIX) and
        self._curr_unwrapped_output):
      # Regex search and highlighting in screen output.
      regex = command[len(self.REGEX_SEARCH_PREFIX):]

      # TODO(cais): Support scrolling to matches.
      # TODO(cais): Display warning message on screen if no match.
      self._display_output(self._curr_unwrapped_output, highlight_regex=regex)
      self._command_pointer = 0
      self._pending_command = ""
      return

    prefix, args = self._parse_command(command)

    if not prefix:
      # Empty command: take no action. Should not exit.
      return

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

    self._display_output(screen_output)
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
    elif x == self.CLI_TAB_KEY:
      self._textbox_curr_terminator = self.CLI_TAB_KEY
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
        self._output_pad = self._display_output(
            self._curr_unwrapped_output, is_refresh=True)

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
    """Generate a new pad on the screen.

    Args:
      rows: (int) Number of rows the pad will have: not limited to screen size.
      cols: (int) Number of columns the pad will have: not limited to screen
        size.

    Returns:
      A curses textpad object.
    """

    return curses.newpad(rows, cols)

  def _display_output(self, output, is_refresh=False, highlight_regex=None):
    """Display text output in a scrollable text pad.

    Args:
      output: A RichTextLines object that is the screen output text.
      is_refresh: (bool) Is this a refreshing display with existing output.
      highlight_regex: (str) Optional string representing the regex used to
        search and highlight in the current screen output.
    """

    if highlight_regex:
      output = debugger_cli_common.regex_find(
          output, highlight_regex, font_attr=self._search_highlight_font_attr)
    else:
      self._curr_unwrapped_output = output

    self._curr_wrapped_output = debugger_cli_common.wrap_rich_text_lines(
        output, self._max_x - 1)

    (self._output_pad, self._output_pad_height,
     self._output_pad_width) = self._display_lines(self._curr_wrapped_output,
                                                   self._output_num_rows)

    # Size of view port on screen, which is always smaller or equal to the
    # screen size.
    self._output_pad_screen_height = self._output_num_rows - 1
    self._output_pad_screen_width = self._max_x - 1
    self._output_pad_screen_location = self.rectangle(
        top=self._output_top_row,
        left=0,
        bottom=self._output_top_row + self._output_num_rows,
        right=self._output_pad_screen_width)

    if is_refresh:
      self._scroll_output(self._SCROLL_REFRESH)
    else:
      self._output_pad_row = 0
      self._scroll_output(self._SCROLL_HOME)

  def _display_lines(self, output, min_num_rows):
    """Display RichTextLines object on screen.

    Args:
      output: A RichTextLines object.
      min_num_rows: (int) Minimum number of output rows.

    Returns:
      1) The text pad object used to display the text.
      2) (int) number of rows of the text pad, which may exceed screen size.
      3) (int) number of columns of the text pad.

    Raises:
      ValueError: If input argument "output" is invalid.
    """

    if not isinstance(output, debugger_cli_common.RichTextLines):
      raise ValueError(
          "Output is required to be an instance of RichTextLines, but is not.")

    # TODO(cais): Cut off output with too many lines to prevent overflow issues
    # in curses.
    self._screen_refresh()

    # Number of rows the output area will have.
    rows = max(min_num_rows, len(output.lines))

    # Size of the output pad, which may exceed screen size and require
    # scrolling.
    cols = self._max_x - 1

    # Create new output pad.
    pad = self._screen_new_output_pad(rows, cols)

    for i in xrange(len(output.lines)):
      if i in output.font_attr_segs:
        self._screen_add_line_to_output_pad(
            pad, i, output.lines[i], color_segments=output.font_attr_segs[i])
      else:
        self._screen_add_line_to_output_pad(pad, i, output.lines[i])

    return pad, rows, cols

  def _screen_add_line_to_output_pad(self, pad, row, txt, color_segments=None):
    """Render a line in a text pad.

    Assumes: segments in color_segments are sorted in ascending order of the
    beginning index.
    Note: Gaps between the segments are allowed and will be fixed in with a
    default color.

    Args:
      pad: The text pad to render the line in.
      row: Row index, as an int.
      txt: The text to be displayed on the specified row, as a str.
      color_segments: A list of 3-tuples. Each tuple represents the beginning
        and the end of a color segment, in the form of a right-open interval:
        [start, end). The last element of the tuple is a color string, e.g.,
        "red".

    Raisee:
      TypeError: If color_segments is not of type list.
    """

    if not color_segments:
      pad.addstr(row, 0, txt, self._default_color_pair)
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
      all_color_pairs.append(self._default_color_pair)

    for (curr_start, curr_end, curr_color), (next_start, _, _) in zip(
        color_segments, color_segments[1:] + [(len(txt), None, None)]):
      all_segments.append((curr_start, curr_end))

      all_color_pairs.append(
          self._color_pairs.get(curr_color, self._default_color_pair))

      if curr_end < next_start:
        # Fill in the gap with the default color.
        all_segments.append((curr_end, next_start))
        all_color_pairs.append(self._default_color_pair)

    # Finally, draw all the segments.
    for segment, color_pair in zip(all_segments, all_color_pairs):
      pad.addstr(row, segment[0], txt[segment[0]:segment[1]], color_pair)

  def _screen_scroll_output_pad(self, pad, viewport_top, viewport_left,
                                screen_location_top, screen_location_left,
                                screen_location_bottom, screen_location_right):
    pad.refresh(viewport_top, viewport_left, screen_location_top,
                screen_location_left, screen_location_bottom,
                screen_location_right)

  def _scroll_output(self, direction):
    """Scroll the output pad.

    Args:
      direction: _SCROLL_REFRESH, _SCROLL_UP, _SCROLL_DOWN, _SCROLL_HOME or
        _SCROLL_END

    Raises:
      ValueError: On invalid scroll direction.
    """

    if not self._output_pad:
      # No output pad is present. Do nothing.
      return

    if direction == self._SCROLL_REFRESH:
      pass
    elif direction == self._SCROLL_UP:
      # Scroll up
      if self._output_pad_row - 1 >= 0:
        self._output_pad_row -= 1
    elif direction == self._SCROLL_DOWN:
      # Scroll down
      if self._output_pad_row + 1 < (
          self._output_pad_height - self._output_pad_screen_height):
        self._output_pad_row += 1
    elif direction == self._SCROLL_HOME:
      # Scroll to top
      self._output_pad_row = 0
    elif direction == self._SCROLL_END:
      # Scroll to bottom
      self._output_pad_row = (
          self._output_pad_height - self._output_pad_screen_height - 1)
    else:
      raise ValueError("Unsupported scroll mode: %s" % direction)

    # Actually scroll the output pad: refresh with new location.
    self._screen_scroll_output_pad(self._output_pad, self._output_pad_row, 0,
                                   self._output_pad_screen_location.top,
                                   self._output_pad_screen_location.left,
                                   self._output_pad_screen_location.bottom,
                                   self._output_pad_screen_location.right)

    if self._output_pad_height > self._output_pad_screen_height + 1:
      # Display information about the scrolling of tall screen output.
      self._scroll_info = "--- Scroll: %.2f%% " % (100.0 * (
          float(self._output_pad_row) /
          (self._output_pad_height - self._output_pad_screen_height - 1)))
      if len(self._scroll_info) < self._max_x:
        self._scroll_info += "-" * (self._max_x - len(self._scroll_info))
      self._screen_draw_text_line(
          self._output_scroll_row, self._scroll_info, color="green")
    else:
      # Screen output is not tall enough to cause scrolling.
      self._scroll_info = "-" * self._max_x
      self._screen_draw_text_line(
          self._output_scroll_row, self._scroll_info, color="green")

  def _tab_complete(self, command_str):
    """Perform tab completion.

    Obtains tab completion candidates.
    If there are no candidates, return command_str and take no other actions.
    If there are candidates, display the candidates on screen and return
    command_str + (common prefix of the candidates).

    Args:
      command_str: (str) The str in the command input textbox when Tab key is
        hit.

    Returns:
      (str) Completed string. Could be the same as command_str if no completion
      candidate is available. If candidate(s) are available, return command_str
      appended by the common prefix of the candidates.
    """

    command_str = command_str.lstrip()

    if not command_str:
      # Empty (top-level) context.
      context = ""
      prefix = ""
      items = []
    else:
      items = command_str.split(" ")
      if len(items) == 1:
        # Single word: top-level context.
        context = ""
        prefix = items[0]
      else:
        # Multiple words.
        context = items[0]
        prefix = items[-1]

    candidates, common_prefix = self._tab_completion_registry.get_completions(
        context, prefix)

    if candidates and len(candidates) > 1:
      self._display_candidates(candidates)
    else:
      # In the case of len(candidates) == 1, the single completion will be
      # entered to the textbox automatically. So there is no need to show any
      # candidates.
      self._display_candidates([])

    if common_prefix:
      # Common prefix is not None and non-empty. The completed string will
      # incorporate the common prefix.
      return " ".join(items[:-1] + [common_prefix])
    else:
      return " ".join(items)

  def _display_candidates(self, candidates):
    """Show candidates (e.g., tab-completion candidates) on multiple lines.

    Args:
      candidates: (list of str) candidates.
    """

    if self._curr_unwrapped_output:
      # Force refresh screen output.
      self._scroll_output(self._SCROLL_REFRESH)

    if not candidates:
      return

    candidates_prefix = "Candidates: "
    candidates_line = candidates_prefix + " ".join(candidates)
    candidates_output = debugger_cli_common.RichTextLines(
        candidates_line,
        font_attr_segs={
            0: [(len(candidates_prefix), len(candidates_line), "yellow")]
        })

    candidates_output = debugger_cli_common.wrap_rich_text_lines(
        candidates_output, self._max_x - 1)

    # Calculate how many lines the candidate text should occupy. Limit it to
    # a maximum value.
    candidates_num_rows = min(
        len(candidates_output.lines), self._candidates_max_lines)
    self._candidates_top_row = (
        self._candidates_bottom_row - candidates_num_rows + 1)

    # Render the candidate text on screen.
    pad, _, _ = self._display_lines(candidates_output, 0)
    self._screen_scroll_output_pad(
        pad, 0, 0, self._candidates_top_row, 0,
        self._candidates_top_row + candidates_num_rows - 1, self._max_x - 1)
