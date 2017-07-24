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
"""Stores debugging information regarding TensorFlow model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import parser
import re
import token

from google.protobuf import json_format

from tensorflow.contrib.tensorboard.plugins.trace.trace_info_pb2 import TraceInfo
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile

# List of regex patterns that match files in the core tensorflow library.
TF_LIB_REGEX_FPATHS = [os.sep + os.path.join('tensorflow', 'python')]

LEFT_TOKENS = [token.LPAR, token.LSQB, token.LBRACE]
RIGHT_TOKENS = [token.RPAR, token.RSQB, token.RBRACE]
TOKENS = LEFT_TOKENS + RIGHT_TOKENS


def store_trace_info(output_file_path,
                     graph=ops.get_default_graph(),
                     ignore_regex_fpaths=None):
  """Collects and stores trace information for a TensorFlow model.

  The output proto is stored in json format.

  Args:
    output_file_path: The path where to store the output proto.
    graph: Optional. The data flow graph. Defaults to `tf.get_default_graph()`.
    ignore_regex_fpaths: Optional. Files whose path matches any of the regexes
        in this list will be ignored. Defaults to patterns that match the core
        tensorflow python library.
  """
  if not ignore_regex_fpaths:
    ignore_regex_fpaths = TF_LIB_REGEX_FPATHS

  trace_info = TraceInfo()
  # Extract trace information for every op in the graph.
  source_fpaths = set()
  for op in graph.get_operations():
    op_info = trace_info.ops.add()
    op_info.name = op.name
    op_info.op_type = op.type
    op_info.device = op.device
    for trace in op.traceback:
      fname, lineno, _, _ = trace
      # Ignore traces in specified file paths.
      if os.path.isabs(fname) and not _ignore_file_path(fname,
                                                        ignore_regex_fpaths):
        line_trace = op_info.traceback.add()
        line_trace.file_path = fname
        line_trace.line_number = lineno
        source_fpaths.add(fname)
    _add_data_from_tensors(op.inputs, op_info.inputs)
    _add_data_from_tensors(op.outputs, op_info.outputs)

  # Read the source files involved in the graph construction.
  for fpath in source_fpaths:
    file_info = trace_info.files.add()

    with gfile.Open(fpath, 'r') as f:
      source = f.read()

    file_info.file_path = fpath
    file_info.source_code = source

    line2start = find_multiline_statements(source)

    for key, value in line2start.items():
      file_info.multiline_statements[key] = value

  # Make sure the directory for the output file exists.
  output_file_path = os.path.expanduser(output_file_path)
  output_dir = os.path.dirname(output_file_path)
  if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)

  # Store the debug information.
  with gfile.Open(output_file_path, 'w') as f:
    f.write(json_format.MessageToJson(trace_info))


def find_multiline_statements(source):
  """Parses the python source and finds multiline statements.

  Based on counting the number of open and closed parenthesis on each line.

  Args:
    source: The source code string.

  Returns:
    A dict that maps a line index A to a line index B, where A is the end of a
    multiline statement and B is the start. Line indexing is 0-based.
  """
  # Get the AST.
  tree = parser.suite(source)
  line2paren_count = [0] * (source.count('\n') + 1)
  _count_brackets_braces_parenthesis(tree.totuple(True), line2paren_count)

  line2start = {}
  for end in range(len(line2paren_count)):
    if line2paren_count[end] >= 0:
      # This is not the end of a multiline statement.
      continue
    cumulative_paren_count = 0
    for start in range(end, -1, -1):
      cumulative_paren_count += line2paren_count[start]
      if cumulative_paren_count == 0:
        line2start[end] = start
        break
  return line2start


def _add_data_from_tensors(tensors, info):
  for t in tensors:
    tensor_info = info.add()

    shape = t.get_shape()
    if shape.ndims:
      shape = [(-1 if s is None else s) for s in shape.as_list()]
      tensor_info.shape.extend(shape)
    tensor_info.dtype = t.dtype.name
    tensor_info.num_bytes_per_elem = t.dtype.size

    for c in t.consumers():
      tensor_info.consumers.append(c.name)


def _ignore_file_path(fname, ignore_regex_fpaths):
  for regex_pattern in ignore_regex_fpaths:
    if re.search(regex_pattern, fname):
      return True
  return False


def _count_brackets_braces_parenthesis(node, line2par):
  if isinstance(node[1], tuple):
    for child in node[1:]:
      _count_brackets_braces_parenthesis(child, line2par)
  else:
    tok = node[0]
    if tok in TOKENS:
      lineno = node[2]
      line2par[lineno - 1] += (1 if tok in LEFT_TOKENS else -1)
  return line2par
