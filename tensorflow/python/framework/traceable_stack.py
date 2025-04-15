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
"""A simple stack that associates filename and line numbers with each object."""

from collections.abc import Iterator
import inspect
import types

from typing import cast, Generic, Optional, TypeVar

T = TypeVar("T")


class TraceableObject(Generic[T]):
  """Wrap an object together with its the code definition location."""

  # Return codes for the set_filename_and_line_from_caller() method.
  SUCCESS, HEURISTIC_USED, FAILURE = (0, 1, 2)

  def __init__(
      self,
      obj: T,
      filename: Optional[str] = None,
      lineno: Optional[int] = None,
  ):
    self.obj = obj
    self.filename = filename
    self.lineno = lineno

  def set_filename_and_line_from_caller(self, offset: int = 0) -> int:
    """Set filename and line using the caller's stack frame.

    If the requested stack information is not available, a heuristic may
    be applied and self.HEURISTIC USED will be returned.  If the heuristic
    fails then no change will be made to the filename and lineno members
    (None by default) and self.FAILURE will be returned.

    Args:
      offset: Integer.  If 0, the caller's stack frame is used.  If 1,
          the caller's caller's stack frame is used.  Larger values are
          permissible but if out-of-range (larger than the number of stack
          frames available) the outermost stack frame will be used.

    Returns:
      TraceableObject.SUCCESS if appropriate stack information was found,
      TraceableObject.HEURISTIC_USED if the offset was larger than the stack,
      and TraceableObject.FAILURE if the stack was empty.
    """
    retcode = self.SUCCESS
    frame = inspect.currentframe()
    if not frame:
      return self.FAILURE
    frame = cast(types.FrameType, frame)
    # Offset is defined in "Args" as relative to the caller. We are one frame
    # beyond the caller.
    for _ in range(offset + 1):
      parent = frame.f_back
      if parent is None:
        # If the offset is too large then we use the largest offset possible.
        retcode = self.HEURISTIC_USED
        break
      parent = cast(types.FrameType, parent)
      frame = parent
    self.filename = frame.f_code.co_filename
    self.lineno = cast(int, frame.f_lineno)
    return retcode

  def copy_metadata(self):
    """Return a TraceableObject like this one, but without the object."""
    return self.__class__(None, filename=self.filename, lineno=self.lineno)


class TraceableStack(Generic[T]):
  """A stack of TraceableObjects."""

  def __init__(
      self, existing_stack: Optional[list[TraceableObject[T]]] = None,
  ):
    """Constructor.

    Args:
      existing_stack: [TraceableObject, ...] If provided, this object will
        set its new stack to a SHALLOW COPY of existing_stack.
    """
    self._stack: list[TraceableObject[T]] = (existing_stack[:] if existing_stack
                                             else [])

  def push_obj(self, obj: T, offset: int = 0):
    """Add object to the stack and record its filename and line information.

    Args:
      obj: An object to store on the stack.
      offset: Integer.  If 0, the caller's stack frame is used.  If 1,
          the caller's caller's stack frame is used.

    Returns:
      TraceableObject.SUCCESS if appropriate stack information was found,
      TraceableObject.HEURISTIC_USED if the stack was smaller than expected,
      and TraceableObject.FAILURE if the stack was empty.
    """
    traceable_obj = TraceableObject(obj)
    self._stack.append(traceable_obj)
    # Offset is defined in "Args" as relative to the caller.  We are 1 frame
    # beyond the caller and need to compensate.
    return traceable_obj.set_filename_and_line_from_caller(offset + 1)

  def pop_obj(self) -> T:
    """Remove last-inserted object and return it, without filename/line info."""
    return self._stack.pop().obj

  def peek_top_obj(self) -> T:
    """Return the most recent stored object."""
    return self._stack[-1].obj

  def peek_objs(self) -> Iterator[T]:
    """Return iterator over stored objects ordered newest to oldest."""
    return (t_obj.obj for t_obj in reversed(self._stack))

  def peek_traceable_objs(self) -> Iterator[TraceableObject[T]]:
    """Return iterator over stored TraceableObjects ordered newest to oldest."""
    return reversed(self._stack)

  def __len__(self) -> int:
    """Return number of items on the stack, and used for truth-value testing."""
    return len(self._stack)

  def copy(self) -> "TraceableStack[T]":
    """Return a copy of self referencing the same objects but in a new list.

    This method is implemented to support thread-local stacks.

    Returns:
      TraceableStack with a new list that holds existing objects.
    """
    return TraceableStack(self._stack)
