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

import six

# Generally such lookups should be done using `threading.local()`. See
# https://blogs.gnome.org/jamesh/2008/06/11/tls-python/ for a detailed
# explanation of why. However the transform stacks are expected to be empty
# when a thread is joined, so reusing the key does not introduce a correctness
# issue. Moreover, get_ident is faster than storing and retrieving a unique
# key in a thread local store.
if six.PY3:
  _get_thread_key = threading.get_ident
else:
  import thread  # pylint: disable=g-import-not-at-top
  _get_thread_key = thread.get_ident


# Names for indices into TF traceback tuples.
TB_FILENAME = 0
TB_LINENO = 1
TB_FUNCNAME = 2
TB_CODEDICT = 3  # Dictionary of Python interpreter state.


_source_mapper_stacks = collections.defaultdict(list)
_source_filter_stacks = collections.defaultdict(list)


class StackTraceTransform(object):
  """Base class for stack trace transformation functions."""

  _stack_dict = None  # Subclasses should override
  _thread_key = None

  def __enter__(self):
    self.reset()

    # Any given instance is assumed to be used by a single thread, which reduces
    # expensive thread local lookups.
    if self._thread_key is None:
      self._thread_key = _get_thread_key()
    else:
      assert self._thread_key == _get_thread_key(), 'Shared across threads?'

    stack = self._stack_dict[self._thread_key]
    if stack:
      self.parent = stack[-1]
    else:
      self.parent = None
    stack.append(self)
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    top = self._stack_dict[self._thread_key].pop()
    assert top is self, 'Concurrent access?'

  def reset(self):
    pass


class StackTraceMapper(StackTraceTransform):
  """Allows remapping traceback information to different source code."""
  _stack_dict = _source_mapper_stacks

  def reset(self):
    self._effective_source_map = None

  def get_effective_source_map(self):
    """Returns a map (filename, lineno) -> (filename, lineno, function_name)."""
    raise NotImplementedError('subclasses need to override this')


class StackTraceFilter(StackTraceTransform):
  """Allows filtering traceback information by removing superfluous frames."""
  _stack_dict = _source_filter_stacks

  def reset(self):
    self._filtered_filenames = None

  def get_filtered_filenames(self):
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

  def get_filtered_filenames(self):
    if self._filtered_filenames is None:
      self._filtered_filenames = frozenset((self._filename,))
      if self.parent is not None:
        self._filtered_filenames |= self.parent.get_filtered_filenames()
    return self._filtered_filenames


EMPTY_FROZEN_MAP = {}
EMPTY_FROZEN_SET = frozenset()


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

  thread_key = _get_thread_key()
  source_mappers = _source_mapper_stacks[thread_key]
  # TODO(mdan): Use sentinels instead.
  if source_mappers:
    source_map = source_mappers[-1].get_effective_source_map()
  else:
    source_map = EMPTY_FROZEN_MAP

  source_filters = _source_filter_stacks[thread_key]
  if source_filters:
    filtered_filenames = source_filters[-1].get_filtered_filenames()
  else:
    filtered_filenames = EMPTY_FROZEN_SET

  while f is not None and (limit is None or length < limit):
    lineno = f.f_lineno
    co = f.f_code
    filename = co.co_filename
    name = co.co_name
    frame_globals = f.f_globals
    func_start_lineno = co.co_firstlineno

    # TODO(mdan): Show some indication that the frame was translated.
    filename, lineno, name = source_map.get(
        (filename, lineno), (filename, lineno, name))

    # Note: we never filter the innermost frame.
    if not (ret and filename in filtered_filenames):
      ret.append((filename, lineno, name, frame_globals, func_start_lineno))
      length += 1

    f = f.f_back

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
    A tuple of n 4-tuples or 5-tuples
    (filename, lineno, name, code, [optional: func_start_lineno]), where the
    code tuple element is calculated from the corresponding elements of the
    input tuple.
  """
  def _tuple_generator():  # pylint: disable=missing-docstring
    for (filename, lineno, name, frame_globals, func_start_lineno) in stack:
      linecache.checkcache(filename)
      line = linecache.getline(filename, lineno, frame_globals)
      if line:
        line = line.strip()
      else:
        line = None
      if include_func_start_lineno:
        yield (filename, lineno, name, line, func_start_lineno)
      else:
        yield (filename, lineno, name, line)

  return tuple(_tuple_generator())
