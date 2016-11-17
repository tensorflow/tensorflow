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
"""Building Blocks of TensorFlow Debugger Command-Line Interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import re
import sre_constants
import traceback

from six.moves import xrange  # pylint: disable=redefined-builtin

HELP_INDENT = "  "

EXPLICIT_USER_EXIT = "explicit_user_exit"
REGEX_MATCH_LINES_KEY = "regex_match_lines"


class CommandLineExit(Exception):

  def __init__(self, exit_token=None):
    Exception.__init__(self)
    self._exit_token = exit_token

  @property
  def exit_token(self):
    return self._exit_token


class RichTextLines(object):
  """Rich multi-line text.

  Line-by-line text output, with font attributes (e.g., color) and annotations
  (e.g., indices in a multi-dimensional tensor). Used as the text output of CLI
  commands. Can be rendered on terminal environments such as curses.

  This is not to be confused with Rich Text Format (RTF). This class is for text
  lines only.
  """

  def __init__(self, lines, font_attr_segs=None, annotations=None):
    """Constructor of RichTextLines.

    Args:
      lines: A list of str or a single str, representing text output to
        screen. The latter case is for convenience when the text output is
        single-line.
      font_attr_segs: A map from 0-based row index to a list of 3-tuples.
        It lists segments in each row that have special font attributes, such
        as colors, that are not the default attribute. For example:
        {1: [(0, 3, "red"), (4, 7, "green")], 2: [(10, 20, "yellow")]}

        In each tuple, the 1st element is the start index of the segment. The
        2nd element is the end index, in an "open interval" fashion. The 3rd
        element is a string that represents the font attribute.
      annotations: A map from 0-based row index to any object for annotating
        the row. A typical use example is annotating rows of the output as
        indices in a multi-dimensional tensor. For example, consider the
        following text representation of a 3x2x2 tensor:
          [[[0, 0], [0, 0]],
           [[0, 0], [0, 0]],
           [[0, 0], [0, 0]]]
        The annotation can indicate the indices of the first element shown in
        each row, i.e.,
          {0: [0, 0, 0], 1: [1, 0, 0], 2: [2, 0, 0]}
        This information can make display of tensors on screen clearer and can
        help the user navigate (scroll) to the desired location in a large
        tensor.

    Raises:
      ValueError: If lines is of invalid type.
    """
    if isinstance(lines, list):
      self._lines = lines
    elif isinstance(lines, str):
      self._lines = [lines]
    else:
      raise ValueError("Unexpected type in lines: %s" % type(lines))

    self._font_attr_segs = font_attr_segs
    if not self._font_attr_segs:
      self._font_attr_segs = {}
      # TODO(cais): Refactor to collections.defaultdict(list) to simplify code.

    self._annotations = annotations
    if not self._annotations:
      self._annotations = {}
      # TODO(cais): Refactor to collections.defaultdict(list) to simplify code.

  @property
  def lines(self):
    return self._lines

  @property
  def font_attr_segs(self):
    return self._font_attr_segs

  @property
  def annotations(self):
    return self._annotations

  def num_lines(self):
    return len(self._lines)

  def slice(self, begin, end):
    """Slice a RichTextLines object.

    The object itself is not changed. A sliced instance is returned.

    Args:
      begin: (int) Beginning line index (inclusive). Must be >= 0.
      end: (int) Ending line index (exclusive). Must be >= 0.

    Returns:
      (RichTextLines) Sliced output instance of RichTextLines.

    Raises:
      ValueError: If begin or end is negative.
    """

    if begin < 0 or end < 0:
      raise ValueError("Encountered negative index.")

    # Copy lines.
    lines = self.lines[begin:end]

    # Slice font attribute segments.
    font_attr_segs = {}
    for key in self.font_attr_segs:
      if key >= begin and key < end:
        font_attr_segs[key - begin] = self.font_attr_segs[key]

    # Slice annotations.
    annotations = {}
    for key in self.annotations:
      if not isinstance(key, int):
        # Annotations can contain keys that are not line numbers.
        annotations[key] = self.annotations[key]
      elif key >= begin and key < end:
        annotations[key - begin] = self.annotations[key]

    return RichTextLines(
        lines, font_attr_segs=font_attr_segs, annotations=annotations)

  def extend(self, other):
    """Extend this instance of RichTextLines with another instance.

    The extension takes effect on the text lines, the font attribute segments,
    as well as the annotations. The line indices in the font attribute
    segments and the annotations are adjusted to account for the existing
    lines. If there are duplicate, non-line-index fields in the annotations,
    the value from the input argument "other" will override that in this
    instance.

    Args:
      other: (RichTextLines) The other RichTextLines instance to be appended at
        the end of this instance.
    """

    orig_num_lines = self.num_lines()  # Record original number of lines.

    # Merge the lines.
    self._lines.extend(other.lines)

    # Merge the font_attr_segs.
    for line_index in other.font_attr_segs:
      self._font_attr_segs[orig_num_lines + line_index] = (
          other.font_attr_segs[line_index])

    # Merge the annotations.
    for key in other.annotations:
      if isinstance(key, int):
        self._annotations[orig_num_lines + key] = (other.annotations[key])
      else:
        self._annotations[key] = other.annotations[key]


def regex_find(orig_screen_output, regex, font_attr):
  """Perform regex match in rich text lines.

  Produces a new RichTextLines object with font_attr_segs containing highlighted
  regex matches.

  Example use cases include:
  1) search for specific nodes in a large list of nodes, and
  2) search for specific numerical values in a large tensor.

  Args:
    orig_screen_output: The original RichTextLines, in which the regex find
      is to be performed.
    regex: The regex used for matching.
    font_attr: Font attribute used for highlighting the found result.

  Returns:
    A modified copy of orig_screen_output.

  Raises:
    ValueError: If input str regex is not a valid regular expression.
  """
  new_screen_output = RichTextLines(
      orig_screen_output.lines,
      font_attr_segs=copy.deepcopy(orig_screen_output.font_attr_segs),
      annotations=orig_screen_output.annotations)

  try:
    re_prog = re.compile(regex)
  except sre_constants.error:
    raise ValueError("Invalid regular expression: \"%s\"" % regex)

  regex_match_lines = []
  for i in xrange(len(new_screen_output.lines)):
    line = new_screen_output.lines[i]
    find_it = re_prog.finditer(line)

    match_segs = []
    for match in find_it:
      match_segs.append((match.start(), match.end(), font_attr))

    if match_segs:
      if i not in new_screen_output.font_attr_segs:
        new_screen_output.font_attr_segs[i] = match_segs
      else:
        new_screen_output.font_attr_segs[i].extend(match_segs)
        new_screen_output.font_attr_segs[i] = sorted(
            new_screen_output.font_attr_segs[i], key=lambda x: x[0])
      regex_match_lines.append(i)

  new_screen_output.annotations[REGEX_MATCH_LINES_KEY] = regex_match_lines
  return new_screen_output


def wrap_rich_text_lines(inp, cols):
  """Wrap RichTextLines according to maximum number of columns.

  Produces a new RichTextLines object with the text lines, font_attr_segs and
  annotations properly wrapped. This ought to be used sparingly, as in most
  cases, command handlers producing RichTextLines outputs should know the
  screen/panel width via the screen_info kwarg and should produce properly
  length-limited lines in the output accordingly.

  Args:
    inp: Input RichTextLines object.
    cols: Number of columns, as an int.

  Returns:
    1) A new instance of RichTextLines, with line lengths limited to cols.
    2) A list of new (wrapped) line index. For example, if the original input
      consists of three lines and only the second line is wrapped, and it's
      wrapped into two lines, this return value will be: [0, 1, 3].
  Raises:
    ValueError: If inputs have invalid types.
  """

  new_line_indices = []

  if not isinstance(inp, RichTextLines):
    raise ValueError("Invalid type of input screen_output")

  if not isinstance(cols, int):
    raise ValueError("Invalid type of input cols")

  out = RichTextLines([])

  row_counter = 0  # Counter for new row index
  for i in xrange(len(inp.lines)):
    new_line_indices.append(out.num_lines())

    line = inp.lines[i]

    if i in inp.annotations:
      out.annotations[row_counter] = inp.annotations[i]

    if len(line) <= cols:
      # No wrapping.
      out.lines.append(line)
      if i in inp.font_attr_segs:
        out.font_attr_segs[row_counter] = inp.font_attr_segs[i]

      row_counter += 1
    else:
      # Wrap.
      wlines = []  # Wrapped lines.

      osegs = []
      if i in inp.font_attr_segs:
        osegs = inp.font_attr_segs[i]

      idx = 0
      while idx < len(line):
        if idx + cols > len(line):
          rlim = len(line)
        else:
          rlim = idx + cols

        wlines.append(line[idx:rlim])
        for seg in osegs:
          if (seg[0] < rlim) and (seg[1] >= idx):
            # Calculate left bound within wrapped line.
            if seg[0] >= idx:
              lb = seg[0] - idx
            else:
              lb = 0

            # Calculate right bound within wrapped line.
            if seg[1] < rlim:
              rb = seg[1] - idx
            else:
              rb = rlim - idx

            if rb > lb:  # Omit zero-length segments.
              wseg = (lb, rb, seg[2])
              if row_counter not in out.font_attr_segs:
                out.font_attr_segs[row_counter] = [wseg]
              else:
                out.font_attr_segs[row_counter].append(wseg)

        idx += cols
        row_counter += 1

      out.lines.extend(wlines)

  # Copy over keys of annotation that are not row indices.
  for key in inp.annotations:
    if not isinstance(key, int):
      out.annotations[key] = inp.annotations[key]

  return out, new_line_indices


class CommandHandlerRegistry(object):
  """Registry of command handlers for CLI.

  Handler methods (callables) for user commands can be registered with this
  class, which then is able to dispatch commands to the correct handlers and
  retrieve the RichTextLines output.

  For example, suppose you have the following handler defined:
    def echo(argv, screen_info=None):
      return RichTextLines(["arguments = %s" % " ".join(argv),
                            "screen_info = " + repr(screen_info)])

  you can register the handler with the command prefix "echo" and alias "e":
    registry = CommandHandlerRegistry()
    registry.register_command_handler("echo", echo,
        "Echo arguments, along with screen info", prefix_aliases=["e"])

  then to invoke this command handler with some arguments and screen_info, do:
    registry.dispatch_command("echo", ["foo", "bar"], screen_info={"cols": 80})

  or with the prefix alias:
    registry.dispatch_command("e", ["foo", "bar"], screen_info={"cols": 80})

  The call will return a RichTextLines object which can be rendered by a CLI.
  """

  HELP_COMMAND = "help"
  HELP_COMMAND_ALIASES = ["h"]

  def __init__(self):
    # A dictionary from command prefix to handler.
    self._handlers = {}

    # A dictionary from prefix alias to prefix.
    self._alias_to_prefix = {}

    # A dictionary from prefix to aliases.
    self._prefix_to_aliases = {}

    # A dictionary from command prefix to help string.
    self._prefix_to_help = {}

    # Introductory text to help information.
    self._help_intro = None

    # Register a default handler for the command "help".
    self.register_command_handler(
        self.HELP_COMMAND,
        self._help_handler,
        "Print this help message.",
        prefix_aliases=self.HELP_COMMAND_ALIASES)

  def register_command_handler(self,
                               prefix,
                               handler,
                               help_info,
                               prefix_aliases=None):
    """Register a callable as a command handler.

    Args:
      prefix: Command prefix, i.e., the first word in a command, e.g.,
        "print" as in "print tensor_1".
      handler: A callable of the following signature:
          foo_handler(argv, screen_info=None),
        where argv is the argument vector (excluding the command prefix) and
          screen_info is a dictionary containing information about the screen,
          such as number of columns, e.g., {"cols": 100}.
        The callable should return:
          1) a RichTextLines object representing the screen output.

        The callable can also raise an exception of the type CommandLineExit,
        which if caught by the command-line interface, will lead to its exit.
        The exception can optionally carry an exit token of arbitrary type.
      help_info: A help string.
      prefix_aliases: Aliases for the command prefix, as a list of str. E.g.,
        shorthands for the command prefix: ["p", "pr"]

    Raises:
      ValueError: If
        1) the prefix is empty, or
        2) handler is not callable, or
        3) a handler is already registered for the prefix, or
        4) elements in prefix_aliases clash with existing aliases.
        5) help_info is not a str.
    """

    if not prefix:
      raise ValueError("Empty command prefix")

    if prefix in self._handlers:
      raise ValueError(
          "A handler is already registered for command prefix \"%s\"" % prefix)

    # Make sure handler is callable.
    if not callable(handler):
      raise ValueError("handler is not callable")

    # Make sure that help info is a string.
    if not isinstance(help_info, str):
      raise ValueError("help_info is not a str")

    # Process prefix aliases.
    if prefix_aliases:
      for alias in prefix_aliases:
        if self._resolve_prefix(alias):
          raise ValueError(
              "The prefix alias \"%s\" clashes with existing prefixes or "
              "aliases." % alias)
        self._alias_to_prefix[alias] = prefix

      self._prefix_to_aliases[prefix] = prefix_aliases

    # Store handler.
    self._handlers[prefix] = handler

    # Store help info.
    self._prefix_to_help[prefix] = help_info

  def dispatch_command(self, prefix, argv, screen_info=None):
    """Handles a command by dispatching it to a registered command handler.

    Args:
      prefix: Command prefix, as a str, e.g., "print".
      argv: Command argument vector, excluding the command prefix, represented
        as a list of str, e.g.,
        ["tensor_1"]
      screen_info: A dictionary containing screen info, e.g., {"cols": 100}.

    Returns:
      An instance of RichTextLines. If any exception is caught during the
      invocation of the command handler, the RichTextLines will wrap the error
      type and message.

    Raises:
      ValueError: If
        1) prefix is empty, or
        2) no command handler is registered for the command prefix, or
        3) the handler is found for the prefix, but it fails to return a
          RichTextLines or raise any exception.
      CommandLineExit:
        If the command handler raises this type of exception, tihs method will
        simply pass it along.
    """
    if not prefix:
      raise ValueError("Prefix is empty")

    resolved_prefix = self._resolve_prefix(prefix)
    if not resolved_prefix:
      raise ValueError("No handler is registered for command prefix \"%s\"" %
                       prefix)

    handler = self._handlers[resolved_prefix]
    try:
      output = handler(argv, screen_info=screen_info)
    except CommandLineExit as e:
      raise e
    except SystemExit as e:
      # Special case for syntax errors caught by argparse.
      lines = ["Syntax error for command: %s" % prefix,
               "For help, do \"help %s\"" % prefix]
      output = RichTextLines(lines)

    except BaseException as e:  # pylint: disable=broad-except
      lines = ["Error occurred during handling of command: %s %s:" %
               (resolved_prefix, " ".join(argv)), "%s: %s" % (type(e), str(e))]

      # Include traceback of the exception.
      lines.append("")
      lines.extend(traceback.format_exc().split("\n"))

      output = RichTextLines(lines)

    if not isinstance(output, RichTextLines):
      raise ValueError(
          "Return value from command handler %s is not a RichTextLines instance"
          % str(handler))

    return output

  def is_registered(self, prefix):
    """Test if a command prefix or its alias is has a registered handler.

    Args:
      prefix: A prefix or its alias, as a str.

    Returns:
      True iff a handler is registered for prefix.
    """
    return self._resolve_prefix(prefix) is not None

  def get_help(self, cmd_prefix=None):
    """Compile help information into a RichTextLines object.

    Args:
      cmd_prefix: Optional command prefix. As the prefix itself or one of its
        aliases.

    Returns:
      A RichTextLines object containing the help information. If cmd_prefix
      is None, the return value will be the full command-line help. Otherwise,
      it will be the help information for the specified command.
    """
    if not cmd_prefix:
      # Print full help information, in sorted order of the command prefixes.
      lines = []
      if self._help_intro:
        # If help intro is available, show it at the beginning.
        lines.extend(self._help_intro)

      sorted_prefixes = sorted(self._handlers)
      for cmd_prefix in sorted_prefixes:
        lines.extend(self._get_help_for_command_prefix(cmd_prefix))
        lines.append("")
        lines.append("")

      return RichTextLines(lines)
    else:
      return RichTextLines(self._get_help_for_command_prefix(cmd_prefix))

  def set_help_intro(self, help_intro):
    """Set an introductory message to help output.

    Args:
      help_intro: (list of str) Text lines appended to the beginning of the
        beginning of the output of the command "help", as introductory
        information.
    """
    self._help_intro = help_intro

  def _help_handler(self, args, screen_info=None):
    """Command handler for "help".

    "help" is a common command that merits built-in support from this class.

    Args:
      args: Command line arguments to "help" (not including "help" itself).
      screen_info: (dict) Information regarding the screen, e.g., the screen
        width in characters: {"cols": 80}

    Returns:
      (RichTextLines) Screen text output.
    """

    _ = screen_info  # Unused currently.

    if not args:
      return self.get_help()
    elif len(args) == 1:
      return self.get_help(args[0])
    else:
      return RichTextLines(["ERROR: help takes only 0 or 1 input argument."])

  def _resolve_prefix(self, token):
    """Resolve command prefix from the prefix itself or its alias.

    Args:
      token: a str to be resolved.

    Returns:
      If resolvable, the resolved command prefix.
      If not resolvable, None.
    """
    if token in self._handlers:
      return token
    elif token in self._alias_to_prefix:
      return self._alias_to_prefix[token]
    else:
      return None

  def _get_help_for_command_prefix(self, cmd_prefix):
    """Compile the help information for a given command prefix.

    Args:
      cmd_prefix: Command prefix, as the prefix itself or one of its
        aliases.

    Returns:
      A list of str as the help information fo cmd_prefix. If the cmd_prefix
        does not exist, the returned list of str will indicate that.
    """
    lines = []

    resolved_prefix = self._resolve_prefix(cmd_prefix)
    if not resolved_prefix:
      lines.append("Invalid command prefix: \"%s\"" % cmd_prefix)
      return lines

    lines.append(resolved_prefix)

    if resolved_prefix in self._prefix_to_aliases:
      lines.append(HELP_INDENT + "Aliases: " + ", ".join(
          self._prefix_to_aliases[resolved_prefix]))

    lines.append("")
    help_lines = self._prefix_to_help[resolved_prefix].split("\n")
    for line in help_lines:
      lines.append(HELP_INDENT + line)

    return lines


class TabCompletionRegistry(object):
  """Registry for tab completion responses."""

  def __init__(self):
    self._comp_dict = {}

  # TODO(cais): Rename method names with "comp" to "*completion*" to avoid
  # confusion.

  def register_tab_comp_context(self, context_words, comp_items):
    """Register a tab-completion context.

    Register that, for each word in context_words, the potential tab-completions
    are the words in comp_items.

    A context word is a pre-existing, completed word in the command line that
    determines how tab-completion works for another, incomplete word in the same
    command line.
    Completion items consist of potential candidates for the incomplete word.

    To give a general example, a context word can be "drink", and the completion
    items can be ["coffee", "tea", "water"]

    Note: A context word can be empty, in which case the context is for the
     top-level commands.

    Args:
      context_words: A list of context words belonging to the context being
        registerd. It is a list of str, instead of a single string, to support
        synonym words triggering the same tab-completion context, e.g.,
        both "drink" and the short-hand "dr" can trigger the same context.
      comp_items: A list of completion items, as a list of str.

    Raises:
      TypeError: if the input arguments are not all of the correct types.
    """

    if not isinstance(context_words, list):
      raise TypeError("Incorrect type in context_list: Expected list, got %s" %
                      type(context_words))

    if not isinstance(comp_items, list):
      raise TypeError("Incorrect type in comp_items: Expected list, got %s" %
                      type(comp_items))

    # Sort the completion items on registration, so that later during
    # get_completions calls, no sorting will be necessary.
    sorted_comp_items = sorted(comp_items)

    for context_word in context_words:
      self._comp_dict[context_word] = sorted_comp_items

  def deregister_context(self, context_words):
    """Deregister a list of context words.

    Args:
      context_words: A list of context words to deregister, as a list of str.

    Raises:
      KeyError: if there are word(s) in context_words that do not correspond
        to any registered contexts.
    """

    for context_word in context_words:
      if context_word not in self._comp_dict:
        raise KeyError("Cannot deregister unregistered context word \"%s\"" %
                       context_word)

    for context_word in context_words:
      del self._comp_dict[context_word]

  def extend_comp_items(self, context_word, new_comp_items):
    """Add a list of completion items to a completion context.

    Args:
      context_word: A single completion word as a string. The extension will
        also apply to all other context words of the same context.
      new_comp_items: (list of str) New completion items to add.

    Raises:
      KeyError: if the context word has not been registered.
    """

    if context_word not in self._comp_dict:
      raise KeyError("Context word \"%s\" has not been registered" %
                     context_word)

    self._comp_dict[context_word].extend(new_comp_items)
    self._comp_dict[context_word] = sorted(self._comp_dict[context_word])

  def remove_comp_items(self, context_word, comp_items):
    """Remove a list of completion items from a completion context.

    Args:
      context_word: A single completion word as a string. The removal will
        also apply to all other context words of the same context.
      comp_items: Completion items to remove.

    Raises:
      KeyError: if the context word has not been registered.
    """

    if context_word not in self._comp_dict:
      raise KeyError("Context word \"%s\" has not been registered" %
                     context_word)

    for item in comp_items:
      self._comp_dict[context_word].remove(item)

  def get_completions(self, context_word, prefix):
    """Get the tab completions given a context word and a prefix.

    Args:
      context_word: The context word.
      prefix: The prefix of the incomplete word.

    Returns:
      (1) None if no registered context matches the context_word.
          A list of str for the matching completion items. Can be an empty list
          of a matching context exists, but no completion item matches the
          prefix.
      (2) Common prefix of all the words in the first return value. If the
          first return value is None, this return value will be None, too. If
          the first return value is not None, i.e., a list, this return value
          will be a str, which can be an empty str if there is no common
          prefix among the items of the list.
    """

    if context_word not in self._comp_dict:
      return None, None

    comp_items = self._comp_dict[context_word]
    comp_items = sorted(
        [item for item in comp_items if item.startswith(prefix)])

    return comp_items, self._common_prefix(comp_items)

  def _common_prefix(self, m):
    """Given a list of str, returns the longest common prefix.

    Args:
      m: (list of str) A list of strings.

    Returns:
      (str) The longest common prefix.
    """
    if not m:
      return ""

    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
      if c != s2[i]:
        return s1[:i]

    return s1


class CommandHistory(object):
  """Keeps command history and supports lookup."""

  def __init__(self, limit=100):
    """CommandHistory constructor.

    Args:
      limit: Maximum number of the most recent commands that this instance
        keeps track of, as an int.
    """

    self._commands = []
    self._limit = limit

  def add_command(self, command):
    """Add a command to the command history.

    Args:
      command: The history command, as a str.

    Raises:
      TypeError: if command is not a str.
    """

    if not isinstance(command, str):
      raise TypeError("Attempt to enter non-str entry to command history")

    self._commands.append(command)

    if len(self._commands) > self._limit:
      self._commands = self._commands[-self._limit:]

  def most_recent_n(self, n):
    """Look up the n most recent commands.

    Args:
      n: Number of most recent commands to look up.

    Returns:
      A list of n most recent commands, or all available most recent commands,
      if n exceeds size of the command history, in chronological order.
    """

    return self._commands[-n:]

  def lookup_prefix(self, prefix, n):
    """Look up the n most recent commands that starts with prefix.

    Args:
      prefix: The prefix to lookup.
      n: Number of most recent commands to look up.

    Returns:
      A list of n most recent commands that have the specified prefix, or all
      available most recent commands that have the prefix, if n exceeds the
      number of history commands with the prefix.
    """

    commands = [cmd for cmd in self._commands if cmd.startswith(prefix)]

    return commands[-n:]

  # TODO(cais): Lookup by regex.
