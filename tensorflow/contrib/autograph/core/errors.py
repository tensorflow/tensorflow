# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Error rewriting logic.

Contains the functions responsible for rewriting tracebacks of errors raised
in AutoGraph (AG) code to refer to user written code, so that errors only refer
to the original user code.

When 'user code' is used in comments it refers to the original source code that
the user wrote and is converting using AutoGraph.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import logging
import sys
import traceback

from tensorflow.contrib.autograph.pyct import origin_info
from tensorflow.python.framework import errors_impl

# TODO(mdan): Add a superclass common to all errors.


class GraphConstructionError(Exception):
  """Error for graph construction errors from AutoGraph generated code."""

  def __init__(self, original_error, custom_traceback):
    self.original_error = original_error
    self.custom_traceback = custom_traceback
    super(GraphConstructionError, self).__init__()

  def __str__(self):
    traceback_str = ''.join(traceback.format_list(self.custom_traceback))
    return ('Traceback (most recent call last):\n' + traceback_str + '\n' + str(
        self.original_error) + '\n')


class TfRuntimeError(Exception):
  """Error wrapper for runtime errors raised by AutoGraph generated code."""

  def __init__(self, op_name, op_message, custom_traceback):
    self.op_name = op_name
    self.op_message = op_message
    self.custom_traceback = custom_traceback
    super(TfRuntimeError, self).__init__()

  def __str__(self):
    message = '%s\n\nCaused by op %r, defined at:\n' % (self.op_message,
                                                        self.op_name)
    return message + ''.join(traceback.format_list(self.custom_traceback))


def _rewrite_tb(source_map, tb):
  """Rewrites code references in a traceback.

  Args:
    source_map: Dict[origin_info.LineLocation, origin_info.OriginInfo], mapping
        locations to their origin
    tb: List[Tuple[Text, Text, Text, Text]], consistent with
        traceback.extract_tb.
  Returns:
    List[Tuple[Text, Text, Text, Text]], the rewritten traceback
  """
  new_tb = []
  for frame in tb:
    filename, lineno, _, _ = frame
    loc = origin_info.LineLocation(filename, lineno)
    origin = source_map.get(loc)
    if origin is not None:
      new_tb.append(origin.as_frame())
    else:
      new_tb.append(frame)
  return new_tb


# TODO(mdan): rename to raise_*
def rewrite_graph_construction_error(source_map):
  """Rewrites errors raised by non-AG APIs inside AG generated code.

  This is called from the except handler inside an AutoGraph generated function
  (that is, during exception handling). Only rewrites the frames corresponding
  to the function that this is called from, so each function is responsible
  to call this to have its own frames rewritten.

  This function always raises an error.

  Args:
    source_map: Dict[origin_info.Location, origin_info.OriginInfo], the source
        map belonging to the calling function

  Raises:
    GraphConstructionError: The rewritten underlying error.
    Exception: The underlying error, if it could not be rewritten.
  """
  error_info = sys.exc_info()
  _, original_error, e_traceback = error_info
  assert original_error is not None
  try:
    current_traceback = _cut_traceback_loops(source_map,
                                             traceback.extract_tb(e_traceback))
    if isinstance(original_error, GraphConstructionError):
      # TODO(mdan): This is incomplete.
      # The error might have bubbled through a non-converted function.
      previous_traceback = original_error.custom_traceback
      cleaned_traceback = [current_traceback[0]] + previous_traceback
    else:
      cleaned_traceback = current_traceback

    cleaned_traceback = _rewrite_tb(source_map, cleaned_traceback)

    if isinstance(original_error, GraphConstructionError):
      original_error.custom_traceback = cleaned_traceback
      new_error = original_error
    else:
      new_error = GraphConstructionError(original_error, cleaned_traceback)
  except Exception:
    logging.exception('Error while rewriting AutoGraph error:')
    # TODO(mdan): Should reraise here, removing the top frame as well.
    raise original_error
  else:
    raise new_error
  finally:
    # Addresses warning https://docs.python.org/2/library/sys.html#sys.exc_info.
    del e_traceback


def _cut_traceback_loops(source_map, original_traceback):
  """Check for cases where we leave a user method and re-enter it.

  This is done by looking at the function names when the filenames are from any
  files the user code is in.  If we find a case where we return to a user method
  after leaving it then we cut out the frames in between because we assume this
  means these in between frames are from internal AutoGraph code that shouldn't
  be included.

  An example of this is:

   File "file1.py", line 57, in my_func
     ...
   File "control_flow_ops.py", line 231, in cond
     ...
   File "control_flow_ops.py", line 1039, in inner_cond
     ...
   File "file1.py", line 68, in my_func
     ...

  Where we would remove the control_flow_ops.py frames because we re-enter
  my_func in file1.py.

  The source map keys are (file_path, line_number) so get the set of all user
  file_paths.

  Args:
    source_map: Dict[origin_info.LineLocation, origin_info.OriginInfo], mapping
      locations to their origin
    original_traceback: List[Tuple[Text, Text, Text, Text]], consistent with
      traceback.extract_tb.

  Returns:
    List[Tuple[Text, Text, Text, Text]], the traceback with any loops removed.
  """
  all_user_files = set(loc.filename for loc in source_map)
  cleaned_traceback = []
  last_user_frame_index = None
  last_user_user_file_path = None
  # TODO(mdan): Simplify this logic.
  for fi, frame in enumerate(original_traceback):
    frame_file_path, lineno, _, _ = frame
    src_map_key = origin_info.LineLocation(frame_file_path, lineno)
    if frame_file_path in all_user_files:
      if src_map_key in source_map:
        if (last_user_frame_index is not None and
            last_user_user_file_path == frame_file_path):
          cleaned_traceback = cleaned_traceback[:last_user_frame_index]
      last_user_frame_index = fi
      last_user_user_file_path = frame_file_path
    cleaned_traceback.append(frame)
  return cleaned_traceback


# TODO(mdan): This should be consistent with rewrite_graph_construction_error
# Both should either raise or return.
def rewrite_tf_runtime_error(error, source_map):
  """Rewrites TensorFlow runtime errors raised by ops created in AG code.

  Args:
    error: tf.OpError
    source_map: Dict[origin_info.LineLocation, origin_info.OriginInfo]

  Returns:
    TfRuntimeError, the rewritten underlying error.
  """
  try:
    cleaned_traceback = _cut_traceback_loops(source_map, error.op.traceback)
    # cleaned_traceback = error.op.traceback
    cleaned_traceback = _rewrite_tb(source_map, cleaned_traceback)

    op_name = error.op.name
    op_message = error.message
    rewritten_error = TfRuntimeError(op_name, op_message, cleaned_traceback)
    return rewritten_error
  except Exception:  # pylint: disable=broad-except
    logging.exception('Error while rewriting AutoGraph error:')
    return error


# TODO(znado): Add arg to enable different levels of error rewriting.
@contextlib.contextmanager
def improved_errors(converted_function):
  """Context manager that rewrites runtime errors.

  This context manager will rewrite runtime errors so that their traceback
  is relative to the original code before conversion.

  Use with the output of to_graph, and wrap the execution of respective ops.
  Example:

    converted_my_func = ag.to_graph(my_func)
    ops = converted_my_func(...)

    with ag.improved_errors(converted_my_func):
      sess.run(ops)

  Args:
    converted_function: Callable[..., Any], the output of a to_graph call

  Yields:
    None

  Raises:
    TfRuntimeError: if any OpError originates in the converted code, it will
        be wrapped into a TfRuntimeError
    ValueError: If converted_function is not generated by AutoGraph
  """
  if (getattr(converted_function, 'ag_source_map', None) is None or
      not isinstance(converted_function.ag_source_map, dict)):
    raise ValueError(
        'converted_function must be the result of an autograph.to_graph call')
  try:
    yield
  except errors_impl.OpError as e:
    raise rewrite_tf_runtime_error(e, converted_function.ag_source_map)
