# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Utilities for using the TensorFlow C API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib


class ScopedTFStatus(object):
  """Wrapper around TF_Status that handles deletion."""

  def __init__(self):
    self.status = c_api.TF_NewStatus()

  def __del__(self):
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api.TF_DeleteStatus is not None:
      c_api.TF_DeleteStatus(self.status)


class ScopedTFGraph(object):
  """Wrapper around TF_Graph that handles deletion."""

  def __init__(self):
    self.graph = c_api.TF_NewGraph()

  def __del__(self):
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api.TF_DeleteGraph is not None:
      c_api.TF_DeleteGraph(self.graph)


class ScopedTFImportGraphDefOptions(object):
  """Wrapper around TF_ImportGraphDefOptions that handles deletion."""

  def __init__(self):
    self.options = c_api.TF_NewImportGraphDefOptions()

  def __del__(self):
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api.TF_DeleteImportGraphDefOptions is not None:
      c_api.TF_DeleteImportGraphDefOptions(self.options)


@tf_contextlib.contextmanager
def tf_buffer(data=None):
  """Context manager that creates and deletes TF_Buffer.

  Example usage:
    with tf_buffer() as buf:
      # get serialized graph def into buf
      ...
      proto_data = c_api.TF_GetBuffer(buf)
      graph_def.ParseFromString(compat.as_bytes(proto_data))
    # buf has been deleted

    with tf_buffer(some_string) as buf:
      c_api.TF_SomeFunction(buf)
    # buf has been deleted

  Args:
    data: An optional `bytes`, `str`, or `unicode` object. If not None, the
      yielded buffer will contain this data.

  Yields:
    Created TF_Buffer
  """
  if data:
    buf = c_api.TF_NewBufferFromString(compat.as_bytes(data))
  else:
    buf = c_api.TF_NewBuffer()
  try:
    yield buf
  finally:
    c_api.TF_DeleteBuffer(buf)


def tf_output(c_op, index):
  """Returns a wrapped TF_Output with specified operation and index.

  Args:
    c_op: wrapped TF_Operation
    index: integer

  Returns:
    Wrapped TF_Output
  """
  ret = c_api.TF_Output()
  ret.oper = c_op
  ret.index = index
  return ret


def tf_operations(graph):
  """Generator that yields every TF_Operation in `graph`.

  Args:
    graph: Graph

  Yields:
    wrapped TF_Operation
  """
  # pylint: disable=protected-access
  pos = 0
  c_op, pos = c_api.TF_GraphNextOperation(graph._c_graph, pos)
  while c_op is not None:
    yield c_op
    c_op, pos = c_api.TF_GraphNextOperation(graph._c_graph, pos)
  # pylint: enable=protected-access


def new_tf_operations(graph):
  """Generator that yields newly-added TF_Operations in `graph`.

  Specifically, yields TF_Operations that don't have associated Operations in
  `graph`. This is useful for processing nodes added by the C API.

  Args:
    graph: Graph

  Yields:
    wrapped TF_Operation
  """
  # TODO(b/69679162): do this more efficiently
  for c_op in tf_operations(graph):
    try:
      graph._get_operation_by_tf_operation(c_op)  # pylint: disable=protected-access
    except KeyError:
      yield c_op
