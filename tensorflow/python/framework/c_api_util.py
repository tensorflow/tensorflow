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

from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib


class ScopedTFStatus(object):
  """Wrapper around TF_Status that handles deletion."""

  __slots__ = ["status"]

  def __init__(self):
    self.status = c_api.TF_NewStatus()

  def __del__(self):
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api is not None and c_api.TF_DeleteStatus is not None:
      c_api.TF_DeleteStatus(self.status)


class ScopedTFGraph(object):
  """Wrapper around TF_Graph that handles deletion."""

  __slots__ = ["graph", "deleter"]

  def __init__(self):
    self.graph = c_api.TF_NewGraph()
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we may have already deleted other modules. By capturing the
    # DeleteGraph function here, we retain the ability to cleanly destroy the
    # graph at shutdown, which satisfies leak checkers.
    self.deleter = c_api.TF_DeleteGraph

  def __del__(self):
    self.deleter(self.graph)


class ScopedTFImportGraphDefOptions(object):
  """Wrapper around TF_ImportGraphDefOptions that handles deletion."""

  __slots__ = ["options"]

  def __init__(self):
    self.options = c_api.TF_NewImportGraphDefOptions()

  def __del__(self):
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api is not None and c_api.TF_DeleteImportGraphDefOptions is not None:
      c_api.TF_DeleteImportGraphDefOptions(self.options)


class ScopedTFImportGraphDefResults(object):
  """Wrapper around TF_ImportGraphDefOptions that handles deletion."""

  __slots__ = ["results"]

  def __init__(self, results):
    self.results = results

  def __del__(self):
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api is not None and c_api.TF_DeleteImportGraphDefResults is not None:
      c_api.TF_DeleteImportGraphDefResults(self.results)


class ScopedTFFunction(object):
  """Wrapper around TF_Function that handles deletion."""

  __slots__ = ["func", "deleter"]

  def __init__(self, func):
    self.func = func
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we may have already deleted other modules. By capturing the
    # DeleteFunction function here, we retain the ability to cleanly destroy the
    # Function at shutdown, which satisfies leak checkers.
    self.deleter = c_api.TF_DeleteFunction

  @property
  def has_been_garbage_collected(self):
    return self.func is None

  def __del__(self):
    if not self.has_been_garbage_collected:
      self.deleter(self.func)
      self.func = None


class ScopedTFBuffer(object):
  """An internal class to help manage the TF_Buffer lifetime."""

  __slots__ = ["buffer"]

  def __init__(self, buf_string):
    self.buffer = c_api.TF_NewBufferFromString(compat.as_bytes(buf_string))

  def __del__(self):
    c_api.TF_DeleteBuffer(self.buffer)


class ApiDefMap(object):
  """Wrapper around Tf_ApiDefMap that handles querying and deletion.

  The OpDef protos are also stored in this class so that they could
  be queried by op name.
  """

  __slots__ = ["_api_def_map", "_op_per_name"]

  def __init__(self):
    op_def_proto = op_def_pb2.OpList()
    buf = c_api.TF_GetAllOpList()
    try:
      op_def_proto.ParseFromString(c_api.TF_GetBuffer(buf))
      self._api_def_map = c_api.TF_NewApiDefMap(buf)
    finally:
      c_api.TF_DeleteBuffer(buf)

    self._op_per_name = {}
    for op in op_def_proto.op:
      self._op_per_name[op.name] = op

  def __del__(self):
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api is not None and c_api.TF_DeleteApiDefMap is not None:
      c_api.TF_DeleteApiDefMap(self._api_def_map)

  def put_api_def(self, text):
    c_api.TF_ApiDefMapPut(self._api_def_map, text, len(text))

  def get_api_def(self, op_name):
    api_def_proto = api_def_pb2.ApiDef()
    buf = c_api.TF_ApiDefMapGet(self._api_def_map, op_name, len(op_name))
    try:
      api_def_proto.ParseFromString(c_api.TF_GetBuffer(buf))
    finally:
      c_api.TF_DeleteBuffer(buf)
    return api_def_proto

  def get_op_def(self, op_name):
    if op_name in self._op_per_name:
      return self._op_per_name[op_name]
    raise ValueError(f"No op_def found for op name {op_name}.")

  def op_names(self):
    return self._op_per_name.keys()


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
