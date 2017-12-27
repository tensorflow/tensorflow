# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Generic entry point script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno as _errno
import sys as _sys

from tensorflow.python.platform import flags
from tensorflow.python.util.all_util import remove_undocumented


def _usage(shorthelp):
  """Writes __main__'s docstring to stdout with some help text.

  Args:
    shorthelp: bool, if True, prints only flags from the main module,
        rather than all flags.
  """
  doc = _sys.modules['__main__'].__doc__
  if not doc:
    doc = '\nUSAGE: %s [flags]\n' % _sys.argv[0]
    doc = flags.text_wrap(doc, indent='       ', firstline_indent='')
  else:
    # Replace all '%s' with sys.argv[0], and all '%%' with '%'.
    num_specifiers = doc.count('%') - 2 * doc.count('%%')
    try:
      doc %= (_sys.argv[0],) * num_specifiers
    except (OverflowError, TypeError, ValueError):
      # Just display the docstring as-is.
      pass
  if shorthelp:
    flag_str = flags.FLAGS.main_module_help()
  else:
    flag_str = str(flags.FLAGS)
  try:
    _sys.stdout.write(doc)
    if flag_str:
      _sys.stdout.write('\nflags:\n')
      _sys.stdout.write(flag_str)
    _sys.stdout.write('\n')
  except IOError as e:
    # We avoid printing a huge backtrace if we get EPIPE, because
    # "foo.par --help | less" is a frequent use case.
    if e.errno != _errno.EPIPE:
      raise


class _HelpFlag(flags.BooleanFlag):
  """Special boolean flag that displays usage and raises SystemExit."""
  NAME = 'help'
  SHORT_NAME = 'h'

  def __init__(self):
    super(_HelpFlag, self).__init__(
        self.NAME, False, 'show this help', short_name=self.SHORT_NAME)

  def parse(self, arg):
    if arg:
      _usage(shorthelp=True)
      print()
      print('Try --helpfull to get a list of all flags.')
      _sys.exit(1)


class _HelpshortFlag(_HelpFlag):
  """--helpshort is an alias for --help."""
  NAME = 'helpshort'
  SHORT_NAME = None


class _HelpfullFlag(flags.BooleanFlag):
  """Display help for flags in main module and all dependent modules."""

  def __init__(self):
    super(_HelpfullFlag, self).__init__('helpfull', False, 'show full help')

  def parse(self, arg):
    if arg:
      _usage(shorthelp=False)
      _sys.exit(1)


_define_help_flags_called = False


def _define_help_flags():
  global _define_help_flags_called
  if not _define_help_flags_called:
    flags.DEFINE_flag(_HelpFlag())
    flags.DEFINE_flag(_HelpfullFlag())
    flags.DEFINE_flag(_HelpshortFlag())
    _define_help_flags_called = True


def run(main=None, argv=None):
  """Runs the program with an optional 'main' function and 'argv' list."""

  # Define help flags.
  _define_help_flags()

  # Parse known flags.
  argv = flags.FLAGS(_sys.argv if argv is None else argv, known_only=True)

  main = main or _sys.modules['__main__'].main

  # Call the main function, passing through any arguments
  # to the final program.
  _sys.exit(main(argv))


_allowed_symbols = [
    'run',
    # Allowed submodule.
    'flags',
]

remove_undocumented(__name__, _allowed_symbols)
