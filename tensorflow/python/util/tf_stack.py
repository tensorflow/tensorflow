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
import threading

import six

# TODO(b/138203821): change to from ...util import ... once the bug is fixed.
from tensorflow.python import _tf_stack

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


def extract_stack(limit=-1):
  """A lightweight, extensible re-implementation of traceback.extract_stack.

  NOTE(mrry): traceback.extract_stack eagerly retrieves the line of code for
      each stack frame using linecache, which results in an abundance of stat()
      calls. This implementation does not retrieve the code, and any consumer
      should apply _convert_stack to the result to obtain a traceback that can
      be formatted etc. using traceback methods.

  Args:
    limit: A limit on the number of frames to return.

  Returns:
    A sequence of FrameSummary objects (filename, lineno, name, line)
    corresponding to the call stack of the current thread.
  """
  # N.B ExtractStack in tf_stack.cc will drop this frame prior to
  # traversing the stack.
  thread_key = _get_thread_key()
  return _tf_stack.extract_stack(
      limit,
      _source_mapper_stacks[thread_key],
      _source_filter_stacks[thread_key])

StackSummary = _tf_stack.StackSummary
FrameSummary = _tf_stack.FrameSummary
