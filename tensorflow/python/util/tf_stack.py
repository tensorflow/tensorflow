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
"""Functions used to extract and analyze stacks.  Faster than Python libs."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import inspect
import linecache
import sys
import threading

# Names for indices into TF traceback tuples.
TB_FILENAME = 0
TB_LINENO = 1
TB_FUNCNAME = 2
TB_CODEDICT = 3  # Dictionary of Python interpreter state.


stacks = threading.local()


def _source_mappers():
  if not hasattr(stacks, 'source_mapper'):
    stacks.source_mapper = []
  return stacks.source_mapper


def _source_filters():
  if not hasattr(stacks, 'source_filter'):
    stacks.source_filter = []
  return stacks.source_filter


class StackTraceMapper(object):
  """Allows remapping traceback information to different source code."""

  def __enter__(self):
    _source_mappers().append(self)
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    assert _source_mappers()[-1] is self, 'Concurrent access?'
    _source_mappers().pop()

  def map(self, filename, lineno, name):
    raise NotImplementedError('subclasses need to override this')


class StackTraceFilter(object):
  """Allows filtering traceback information by removing superfluous frames."""

  def __enter__(self):
    _source_filters().append(self)
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    assert _source_filters()[-1] is self, 'Concurrent access?'
    _source_filters().pop()

  def filter(self, filename, lineno, name):
    raise NotImplementedError('subclasses need to override this')


class CurrentModuleFilter(StackTraceFilter):
  """Filters stack frames from the module where this is used (best effort)."""

  def __init__(self):
    filter_filename = None
    outer_f = None
    f = inspect.currentframe()
    try:
      if f is not None:
        # The current frame is __init__. The first outer frame should be the
        # caller.
        outer_f = f.f_back
        if outer_f is not None:
          filter_filename = inspect.getsourcefile(outer_f)
      self._filename = filter_filename
    finally:
      # Avoid reference cycles, see:
      # https://docs.python.org/3.7/library/inspect.html#the-interpreter-stack
      del f
      del outer_f

  def should_remove(self, filename, lineno, name):
    del lineno, name
    return filename == self._filename


def extract_stack(limit=None):
  """A lightweight, extensible re-implementation of traceback.extract_stack.

  NOTE(mrry): traceback.extract_stack eagerly retrieves the line of code for
      each stack frame using linecache, which results in an abundance of stat()
      calls. This implementation does not retrieve the code, and any consumer
      should apply _convert_stack to the result to obtain a traceback that can
      be formatted etc. using traceback methods.

  Args:
    limit: A limit on the number of frames to return.

  Returns:
    A list of 5-tuples
        (filename, lineno, name, frame_globals, func_start_lineno)
    corresponding to the call stack of the current thread.  The returned tuples
    have the innermost stack frame at the end, unlike the Python inspect
    module's stack() function.
  """
  try:
    raise ZeroDivisionError
  except ZeroDivisionError:
    f = sys.exc_info()[2].tb_frame.f_back
  ret = []
  length = 0
  while f is not None and (limit is None or length < limit):
    lineno = f.f_lineno
    co = f.f_code
    filename = co.co_filename
    name = co.co_name
    frame_globals = f.f_globals
    func_start_lineno = co.co_firstlineno

    for mapper in _source_mappers():
      # TODO(mdan): Show some indication that the frame was translated.
      filename, lineno, name = mapper.map(filename, lineno, name)

    keep = True
    if ret:  # Never filter the innermost frame.
      keep = not any(
          f.should_remove(filename, lineno, name) for f in _source_filters())
    if keep:
      ret.append((filename, lineno, name, frame_globals, func_start_lineno))
      length += 1

    f = f.f_back

  # TODO(mdan): Also add a truncation mechanism.

  ret.reverse()
  return ret


FileAndLine = collections.namedtuple('FileAndLine', ['file', 'line'])


def extract_stack_file_and_line(max_length=1000):
  """A version of extract_stack that only returns filenames and line numbers.

  Callers often only require filenames and line numbers, and do not need the
  additional information gathered by extract_stack, as they never call
  convert_stack.

  As a further optimisation, we allow users to specify a limit on the number of
  frames examined.

  Args:
    max_length: The maximum length of stack to extract.

  Returns:
    A list of FileAndLine objects corresponding to the call stack of the current
    thread.
  """
  try:
    raise ZeroDivisionError
  except ZeroDivisionError:
    frame = sys.exc_info()[2].tb_frame.f_back
  ret = []
  length = 0
  while frame is not None and length < max_length:
    ret.append(FileAndLine(frame.f_code.co_filename, frame.f_lineno))
    length += 1
    frame = frame.f_back
  ret.reverse()
  return ret


def convert_stack(stack, include_func_start_lineno=False):
  """Converts a stack extracted using extract_stack() to a traceback stack.

  Args:
    stack: A list of n 5-tuples,
      (filename, lineno, name, frame_globals, func_start_lineno).
    include_func_start_lineno: True if function start line number should be
      included as the 5th entry in return tuples.

  Returns:
    A list of n 4-tuples or 5-tuples
    (filename, lineno, name, code, [optional: func_start_lineno]), where the
    code tuple element is calculated from the corresponding elements of the
    input tuple.
  """
  ret = []
  for (filename, lineno, name, frame_globals, func_start_lineno) in stack:
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, frame_globals)
    if line:
      line = line.strip()
    else:
      line = None
    if include_func_start_lineno:
      ret.append((filename, lineno, name, line, func_start_lineno))
    else:
      ret.append((filename, lineno, name, line))
  return ret
