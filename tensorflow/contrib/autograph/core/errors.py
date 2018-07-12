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

from tensorflow.contrib.autograph.pyct.origin_info import CodeLocation
from tensorflow.python.framework import errors_impl
from tensorflow.python.util import tf_inspect


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


def _rewrite_frame(source_map, cleaned_traceback, stack_frame_indices):
  """Rewrites the stack frames at the given indices using the given source map.

  Args:
    source_map: Dict[CodeLocation, OriginInfo], a mapping between the user and
        AG generated code.
    cleaned_traceback: List[Tuple[text, text, text, text]], the current
        traceback.
    stack_frame_indices: Iterable[Int], frame indices to possibly rewrite if
        there are matching source mapping keys.

  Returns:
    None
  """
  for frame_index in stack_frame_indices:
    # (file_path, line number, function name, code)
    file_path, line_number, _, _ = cleaned_traceback[frame_index]
    source_map_key = CodeLocation(file_path=file_path, line_number=line_number)
    found_mapping = source_map_key in source_map
    if found_mapping:
      cleaned_traceback[frame_index] = source_map[source_map_key].as_frame()


# TODO(znado): Make more robust to name changes in the rewriting logic.
def _remove_rewrite_frames(tb):
  """Remove stack frames containing the error rewriting logic."""
  cleaned_tb = []
  for f in tb:
    if 'ag__.rewrite_graph_construction_error' not in f[3]:
      cleaned_tb.append(f)
  return cleaned_tb


def rewrite_graph_construction_error(source_map):
  """Rewrites errors raised by non-AG APIs inside AG generated code.

  Meant to be called from the try/except block inside each AutoGraph generated
  function.  Only rewrites the traceback frames corresponding to the function
  that this is called from.  When we raise a GraphConstructionError at the end
  it is then caught by calling functions, where they can be responsible for
  rewriting their own frames.

  Args:
    source_map: Dict[CodeLocation, OriginInfo], a mapping between the user and
        AG generated code.

  Raises:
    GraphConstructionError: The rewritten underlying error.
    Exception: The underlying error, if it could not be rewritten.
  """
  error_info = sys.exc_info()
  _, original_error, e_traceback = error_info
  assert original_error is not None
  try:
    _, _, _, func_name, _, _ = tf_inspect.stack()[1]
    # The latest function call is added to the beginning of a traceback, but
    # when rewriting the traceback of multiple function calls the previous
    # functions' except blocks may have already rewritten their own frames so
    # we want to copy over all of the previous frames. We may have rewritten
    # previous frames only if the error is a GraphConstructionError.
    if isinstance(original_error, GraphConstructionError):
      cleaned_traceback = traceback.extract_tb(e_traceback)
      previous_traceback = original_error.custom_traceback
      cleaned_traceback = [cleaned_traceback[0]] + previous_traceback
    else:
      cleaned_traceback = traceback.extract_tb(e_traceback)
    cleaned_traceback = _remove_rewrite_frames(cleaned_traceback)

    current_frame_indices = []
    # This code is meant to be called from the try/except block that wraps a
    # function body.  Here we look for all frames that came from the function
    # that this wraps, look for any matching line numbers in the source
    # mapping, and then rewrite them if matches are found.
    for fi, frame in enumerate(cleaned_traceback):
      _, _, frame_func_name, _ = frame
      if frame_func_name == func_name:
        current_frame_indices.append(fi)
        break
    if current_frame_indices:
      _rewrite_frame(source_map, cleaned_traceback, current_frame_indices)

    if isinstance(original_error, GraphConstructionError):
      original_error.custom_traceback = cleaned_traceback
      new_error = original_error
    else:
      new_error = GraphConstructionError(original_error, cleaned_traceback)
  except Exception:
    logging.exception('Error while rewriting AutoGraph error:')
    raise original_error
  else:
    raise new_error
  finally:
    # Addresses warning https://docs.python.org/2/library/sys.html#sys.exc_info.
    del e_traceback


def rewrite_tf_runtime_error(error, source_map):
  """Rewrites TensorFlow runtime errors raised by ops created in AG code.

  Args:
    error: error_impl.OpError, an TensorFlow error that will have its traceback
        rewritten.
    source_map: Dict[CodeLocation, OriginInfo], a mapping between the user and
        AG generated code.

  Returns:
    A TfRuntimeError with a traceback rewritten according to the given
    source mapping.
  """
  # Check for cases where we leave a user method and re-enter it in the
  # traceback.  This is done by looking at the function names when the
  # filenames are from any files the user code is in.  If we find a case where
  # we return to a user method after leaving it then we cut out the frames in
  # between because we assume this means these in between frames are from
  # internal AutoGraph code that shouldn't be included.
  #
  # An example of this is:
  #
  #  File "file1.py", line 57, in my_func
  #    ...
  #  File "control_flow_ops.py", line 231, in cond
  #    ...
  #  File "control_flow_ops.py", line 1039, in inner_cond
  #    ...
  #  File "file1.py", line 68, in my_func
  #    ...
  #
  # Where we would remove the control_flow_ops.py frames because we re-enter
  # my_func in file1.py.
  #
  # The source map keys are (file_path, line_number) so get the set of all user
  # file_paths.
  try:
    all_user_files = set(k.file_path for k in source_map)
    cleaned_traceback = []
    last_user_frame_index = None
    last_user_user_file_path = None
    last_user_user_fn_name = None
    for fi, frame in enumerate(error.op.traceback):
      frame_file_path, frame_line_number, _, _ = frame
      src_map_key = CodeLocation(
          file_path=frame_file_path, line_number=frame_line_number)
      if frame_file_path in all_user_files:
        if src_map_key in source_map:
          original_fn_name = source_map[src_map_key].function_name
          if (last_user_frame_index is not None and
              last_user_user_file_path == frame_file_path):
            if last_user_user_fn_name == original_fn_name:
              cleaned_traceback = cleaned_traceback[:last_user_frame_index]
            else:
              cleaned_traceback = cleaned_traceback[:last_user_frame_index + 1]
          last_user_user_fn_name = original_fn_name
        else:
          last_user_user_fn_name = None
        last_user_frame_index = fi
        last_user_user_file_path = frame_file_path
      cleaned_traceback.append(frame)

    for fi in range(len(cleaned_traceback)):
      _rewrite_frame(source_map, cleaned_traceback, [fi])
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
      not converted_function.ag_source_map):
    raise ValueError(
        'converted_function must be the result of an autograph.to_graph call')
  try:
    yield
  except errors_impl.OpError as e:
    raise rewrite_tf_runtime_error(e, converted_function.ag_source_map)
