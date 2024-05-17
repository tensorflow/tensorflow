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
import collections
import inspect
import threading

from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack

# Generally such lookups should be done using `threading.local()`. See
# https://blogs.gnome.org/jamesh/2008/06/11/tls-python/ for a detailed
# explanation of why. However the transform stacks are expected to be empty
# when a thread is joined, so reusing the key does not introduce a correctness
# issue. Moreover, get_ident is faster than storing and retrieving a unique
# key in a thread local store.
_get_thread_key = threading.get_ident


# TODO(mdan): Move these to C++ as well.
# Moving to C++ can further avoid extra copies made by get_effective_map.
_source_mapper_stacks = collections.defaultdict(lambda: [SentinelMapper()])
_source_filter_stacks = collections.defaultdict(lambda: [SentinelFilter()])


class StackTraceTransform(object):
  """Base class for stack trace transformation functions."""

  _stack_dict = None  # Subclasses should override
  _thread_key = None

  def __enter__(self):
    # Any given instance is assumed to be used by a single thread, which reduces
    # expensive thread local lookups.
    if self._thread_key is None:
      self._thread_key = _get_thread_key()
    else:
      assert self._thread_key == _get_thread_key(), 'Shared across threads?'

    stack = self._stack_dict[self._thread_key]
    self.parent = stack[-1]
    stack.append(self)
    self.update()
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    top = self._stack_dict[self._thread_key].pop()
    assert top is self, 'Concurrent access?'

  def update(self):
    raise NotImplementedError('subclasses need to override this')


class StackTraceMapper(StackTraceTransform):
  """Allows remapping traceback information to different source code."""
  _stack_dict = _source_mapper_stacks

  def __init__(self):
    self.internal_map = _tf_stack.PyBindSourceMap()

  def update(self):
    self.internal_map.update_to(tuple(self.get_effective_source_map().items()))

  def get_effective_source_map(self):
    """Returns a map (filename, lineno) -> (filename, lineno, function_name)."""
    raise NotImplementedError('subclasses need to override this')


EMPTY_DICT = {}


class SentinelMapper(StackTraceMapper):

  def get_effective_source_map(self):
    return EMPTY_DICT


class StackTraceFilter(StackTraceTransform):
  """Allows filtering traceback information by removing superfluous frames."""
  _stack_dict = _source_filter_stacks

  def __init__(self):
    self.internal_set = _tf_stack.PyBindFileSet()

  def update(self):
    self.internal_set.update_to(set(self.get_filtered_filenames()))

  def get_filtered_filenames(self):
    raise NotImplementedError('subclasses need to override this')


EMPTY_SET = frozenset()


class SentinelFilter(StackTraceFilter):

  def get_filtered_filenames(self):
    return EMPTY_SET


class CurrentModuleFilter(StackTraceFilter):
  """Filters stack frames from the module where this is used (best effort)."""

  def __init__(self):
    super().__init__()
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
      # This may be called repeatedly: once on entry by the superclass, then by
      # each child context manager.
      self._cached_set = None
    finally:
      # Avoid reference cycles, see:
      # https://docs.python.org/3.7/library/inspect.html#the-interpreter-stack
      del f
      del outer_f

  def get_filtered_filenames(self):
    if self._cached_set is not None:
      return self._cached_set

    filtered_filenames = frozenset((self._filename,))
    if self.parent is not None:
      filtered_filenames |= self.parent.get_filtered_filenames()
    self._cached_set = filtered_filenames
    return filtered_filenames


def extract_stack(stacklevel=1):
  """An eager-friendly alternative to traceback.extract_stack.

  Args:
    stacklevel: number of initial frames to skip when producing the stack.

  Returns:
    A list-like FrameSummary containing StackFrame-like objects, which are
    namedtuple-like objects with the following fields: filename, lineno, name,
    line, meant to masquerade as traceback.FrameSummary objects.
  """
  thread_key = _get_thread_key()
  return _tf_stack.extract_stack(
      _source_mapper_stacks[thread_key][-1].internal_map,
      _source_filter_stacks[thread_key][-1].internal_set,
      stacklevel,
  )


def LoadTracesFromDebugInfo(debug_info):
  return _tf_stack.LoadTracesFromDebugInfo(debug_info.SerializeToString())


class GraphDebugInfoBuilder(_tf_stack.GraphDebugInfoBuilder):

  def AppendGraphDebugInfo(self, fn_name, fn_debug_info):
    debug_info_str = fn_debug_info.SerializeToString()
    super().AppendGraphDebugInfo(fn_name, debug_info_str)

  def Build(self):
    debug_info_str = super().Build()
    debug_info = graph_debug_info_pb2.GraphDebugInfo()
    debug_info.ParseFromString(debug_info_str)
    return debug_info


StackSummary = _tf_stack.StackTrace
FrameSummary = _tf_stack.StackFrame
