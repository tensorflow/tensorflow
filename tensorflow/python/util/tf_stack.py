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

import linecache
import sys


def extract_stack(extract_frame_info_fn=None):
  """A lightweight, extensible re-implementation of traceback.extract_stack.

  NOTE(mrry): traceback.extract_stack eagerly retrieves the line of code for
      each stack frame using linecache, which results in an abundance of stat()
      calls. This implementation does not retrieve the code, and any consumer
      should apply _convert_stack to the result to obtain a traceback that can
      be formatted etc. using traceback methods.

  Args:
    extract_frame_info_fn: Optional callable fn(stack_frame) applied to each
        stack frame.  This callable's return value is stored as the sixth (last)
        element of the returned tuples.  If not provided, the returned tuples
        will have None as their sixth value.

  Returns:
    A list of 6-tuples
        (filename, lineno, name, frame_globals, func_start_lineno, custom_info)
    corresponding to the call stack of the current thread.  The returned tuples
    have the innermost stack frame at the end, unlike the Python inspect
    module's stack() function.
  """
  default_fn = lambda f: None
  extract_frame_info_fn = extract_frame_info_fn or default_fn
  try:
    raise ZeroDivisionError
  except ZeroDivisionError:
    f = sys.exc_info()[2].tb_frame.f_back
  ret = []
  while f is not None:
    lineno = f.f_lineno
    co = f.f_code
    filename = co.co_filename
    name = co.co_name
    frame_globals = f.f_globals
    func_start_lineno = co.co_firstlineno
    frame_info = extract_frame_info_fn(f)
    ret.append((filename, lineno, name, frame_globals, func_start_lineno,
                frame_info))
    f = f.f_back
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
  for (filename, lineno, name, frame_globals, func_start_lineno,
       unused_frame_info) in stack:
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
