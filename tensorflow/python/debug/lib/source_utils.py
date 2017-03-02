# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Classes and functions that help to inspect Python source w.r.t. TF graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def _convert_watch_key_to_tensor_name(watch_key):
  return watch_key[:watch_key.rfind(":")]


def annotate_source(dump,
                    source_file_path,
                    do_dumped_tensors=False,
                    file_stack_top=False,
                    min_line=None,
                    max_line=None):
  """Annotate a Python source file with a list of ops created at each line.

  (The annotation doesn't change the source file itself.)

  Args:
    dump: (`DebugDumpDir`) A `DebugDumpDir` object of which the Python graph
      has been loaded.
    source_file_path: (`str`) Path to the source file being annotated.
    do_dumped_tensors: (`str`) Whether dumped Tensors, instead of ops are to be
      used to annotate the source file.
    file_stack_top: (`bool`) Whether only the top stack trace in the
      specified source file is to be annotated.
    min_line: (`None` or `int`) The 1-based line to start annotate the source
      file from (inclusive).
    max_line: (`None` or `int`) The 1-based line number to end the annotation
      at (exclusive).

  Returns:
    A `dict` mapping 1-based line number to a list of op name(s) created at
      that line, or tensor names if `do_dumped_tensors` is True.

  Raises:
    ValueError: If the dump object does not have a Python graph set.
  """

  py_graph = dump.python_graph
  if not py_graph:
    raise ValueError("Cannot perform source annotation due to a lack of set "
                     "Python graph in the dump object")

  source_file_path = os.path.normpath(source_file_path)

  line_to_op_names = {}
  for op in py_graph.get_operations():
    try:
      traceback = dump.node_traceback(op.name)
    except KeyError:
      pass

    for file_path, line_number, _, _ in reversed(traceback):
      if (min_line is not None and line_number < min_line or
          max_line is not None and line_number >= max_line):
        continue

      if os.path.normpath(file_path) != source_file_path:
        continue

      if do_dumped_tensors:
        watch_keys = dump.debug_watch_keys(op.name)
        # Convert watch keys to unique Tensor names.
        items_to_append = list(
            set(map(_convert_watch_key_to_tensor_name, watch_keys)))
      else:
        items_to_append = [op.name]

      if line_number in line_to_op_names:
        line_to_op_names[line_number].extend(items_to_append)
      else:
        line_to_op_names[line_number] = items_to_append

      if file_stack_top:
        break

  return line_to_op_names
