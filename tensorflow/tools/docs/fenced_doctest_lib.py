# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Run doctests for tensorflow."""

import ast
import doctest
import os
import re
import textwrap
from typing import Any, Callable, Dict, Iterable, Optional

import astor

from tensorflow.tools.docs import tf_doctest_lib


def load_from_files(
    files,
    globs: Optional[Dict[str, Any]] = None,
    set_up: Optional[Callable[[Any], None]] = None,
    tear_down: Optional[Callable[[Any], None]] = None) -> doctest.DocFileSuite:
  """Creates a doctest suite from the the files list.

  Args:
    files: A list of file paths to test.
    globs: The global namespace the tests are run in.
    set_up: Run before each test, recieves the test as argument.
    tear_down: Run after each test, recieves the test as argument.

  Returns:
    A DocFileSuite containing the tests.
  """
  if globs is None:
    globs = {}

  # __fspath__ isn't respected everywhere in doctest so convert paths to
  # strings.
  files = [os.fspath(f) for f in files]

  globs['_print_if_not_none'] = _print_if_not_none
  # Ref: https://docs.python.org/3/library/doctest.html#doctest.DocFileSuite
  return doctest.DocFileSuite(
      *files,
      module_relative=False,
      parser=FencedCellParser(fence_label='python'),
      globs=globs,
      setUp=set_up,
      tearDown=tear_down,
      checker=FencedCellOutputChecker(),
      optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
                   | doctest.IGNORE_EXCEPTION_DETAIL
                   | doctest.DONT_ACCEPT_BLANKLINE),
  )


class FencedCellOutputChecker(tf_doctest_lib.TfDoctestOutputChecker):
  """TfDoctestChecker with a different warning message."""
  MESSAGE = textwrap.dedent("""\n
        ##############################################################
        # Check the documentation (go/g3doctest) on how to write
        # testable g3docs.
        ##############################################################
        """)


class FencedCellParser(doctest.DocTestParser):
  """Implements test parsing for ``` fenced cells.

  https://docs.python.org/3/library/doctest.html#doctestparser-objects

  The `get_examples` method recieves a string and returns an
  iterable of `doctest.Example` objects.
  """
  patched = False

  def __init__(self, fence_label='python'):
    super().__init__()

    if not self.patched:
      # The default doctest compiles in "single" mode. The fenced block may
      # contain multiple statements. The `_patch_compile` function fixes the
      # compile mode.
      doctest.compile = _patch_compile
      print(
          textwrap.dedent("""
          *********************************************************************
          * Caution: `fenced_doctest` patches `doctest.compile` don't use this
          *   in the same binary as any other doctests.
          *********************************************************************
          """))
      type(self).patched = True

    # Match anything, except if the look-behind sees a closing fence.
    no_fence = '(.(?<!```))*?'
    self.fence_cell_re = re.compile(
        rf"""
        ^(                             # After a newline
            \s*```\s*({fence_label})\n   # Open a labeled ``` fence
            (?P<doctest>{no_fence})      # Match anything except a closing fence
            \n\s*```\s*(\n|$)            # Close the fence.
        )
        (                              # Optional!
            [\s\n]*                      # Any number of blank lines.
            ```\s*\n                     # Open ```
            (?P<output>{no_fence})       # Anything except a closing fence
            \n\s*```                     # Close the fence.
        )?
        """,
        # Multiline so ^ matches after a newline
        re.MULTILINE |
        # Dotall so `.` matches newlines.
        re.DOTALL |
        # Verbose to allow comments/ignore-whitespace.
        re.VERBOSE)

  def get_examples(self,
                   string: str,
                   name: str = '<string>') -> Iterable[doctest.Example]:
    # Check for a file-level skip comment.
    if re.search('<!--.*?doctest.*?skip.*?all.*?-->', string, re.IGNORECASE):
      return

    for match in self.fence_cell_re.finditer(string):
      if re.search('doctest.*skip', match.group(0), re.IGNORECASE):
        continue

      groups = match.groupdict()

      source = textwrap.dedent(groups['doctest'])
      want = groups['output']
      if want is not None:
        want = textwrap.dedent(want)

      yield doctest.Example(
          lineno=string[:match.start()].count('\n') + 1,
          source=source,
          want=want)


def _print_if_not_none(obj):
  """Print like a notebook: Show the repr if the object is not None.

  `_patch_compile` Uses this on the final expression in each cell.

  This way the outputs feel like notebooks.

  Args:
    obj: the object to print.
  """
  if obj is not None:
    print(repr(obj))


def _patch_compile(source,
                   filename,
                   mode,
                   flags=0,
                   dont_inherit=False,
                   optimize=-1):
  """Patch `doctest.compile` to make doctest to behave like a notebook.

  Default settings for doctest are configured to run like a repl: one statement
  at a time. The doctest source uses `compile(..., mode="single")`

  So to let doctest act like a notebook:

  1. We need `mode="exec"` (easy)
  2. We need the last expression to be printed (harder).

  To print the last expression, just wrap the last expression in
  `_print_if_not_none(expr)`. To detect the last expression use `AST`.
  if the last node is an expression modify the ast to to call
  `_print_if_not_none` on it, convert the ast back to source and compile that.

  https://docs.python.org/3/library/functions.html#compile

  Args:
    source: Can either be a normal string, a byte string, or an AST object.
    filename: Argument should give the file from which the code was read; pass
      some recognizable value if it wasn’t read from a file ('<string>' is
      commonly used).
    mode: [Ignored] always use exec.
    flags: Compiler options.
    dont_inherit: Compiler options.
    optimize: Compiler options.

  Returns:
    The resulting code object.
  """
  # doctest passes some dummy string as the file name, AFAICT
  # but tf.function freaks-out if this doesn't look like a
  # python file name.
  del filename
  # Doctest always passes "single" here, you need exec for multiple lines.
  del mode

  source_ast = ast.parse(source)

  final = source_ast.body[-1]
  if isinstance(final, ast.Expr):
    # Wrap the final expression as `_print_if_not_none(expr)`
    print_it = ast.Expr(
        lineno=-1,
        col_offset=-1,
        value=ast.Call(
            func=ast.Name(
                id='_print_if_not_none',
                ctx=ast.Load(),
                lineno=-1,
                col_offset=-1),
            lineno=-1,
            col_offset=-1,
            args=[final],  # wrap the final Expression
            keywords=[]))
    source_ast.body[-1] = print_it

    # It's not clear why this step is necessary. `compile` is supposed to handle
    # AST directly.
    source = astor.to_source(source_ast)

  return compile(
      source,
      filename='dummy.py',
      mode='exec',
      flags=flags,
      dont_inherit=dont_inherit,
      optimize=optimize)
