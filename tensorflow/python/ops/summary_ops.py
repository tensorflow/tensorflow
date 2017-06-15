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
"""Summary Operations."""
# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf import json_format
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import summary_op_util
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_logging_ops import *
# pylint: enable=wildcard-import


# TODO(dandelion): As currently implemented, this op has several problems.
# The 'summary_description' field is passed but not used by the kernel.
# The 'name' field is used to creat a scope and passed down via name=scope,
# but gen_logging_ops._tensor_summary ignores this parameter and uses the
# kernel's op name as the name. This is ok because scope and the op name
# are identical, but it's probably worthwhile to fix.
# Finally, because of the complications above, this currently does not
# support the family= attribute added to other summaries in cl/156791589.
def tensor_summary(  # pylint: disable=invalid-name
    name,
    tensor,
    summary_description=None,
    collections=None):
  # pylint: disable=line-too-long
  """Outputs a `Summary` protocol buffer with a serialized tensor.proto.

  The generated
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  has one summary value containing the input tensor.

  Args:
    name: A name for the generated node. Will also serve as the series name in
      TensorBoard.
    tensor: A tensor of any type and shape to serialize.
    summary_description: Optional summary_pb2.SummaryDescription()
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  # pylint: enable=line-too-long

  if summary_description is None:
    summary_description = summary_pb2.SummaryDescription()

  description = json_format.MessageToJson(summary_description)
  with ops.name_scope(name, None, [tensor]) as scope:
    val = gen_logging_ops._tensor_summary(
        tensor=tensor,
        description=description,
        name=scope)
    summary_op_util.collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val

ops.NotDifferentiable("TensorSummary")


def _tensor_summary_v2(  # pylint: disable=invalid-name
    name,
    tensor,
    summary_description=None,
    collections=None,
    summary_metadata=None,
    family=None):
  # pylint: disable=line-too-long
  """Outputs a `Summary` protocol buffer with a serialized tensor.proto.

  NOTE(chizeng): This method is temporary. It should never make it into
  TensorFlow 1.3, and nothing should depend on it. This method should be deleted
  before August 2017 (ideally, earlier). This method exists to unblock the
  TensorBoard plugin refactoring effort. We will later modify the tensor_summary
  method to directly make use of the TensorSummaryV2 op. There must be a 3-week
  difference between adding a new op (C++) and changing a python interface to
  use it.

  The generated
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  has one summary value containing the input tensor.

  Args:
    name: A name for the generated node. Will also serve as the series name in
      TensorBoard.
    tensor: A tensor of any type and shape to serialize.
    summary_description: This is currently un-used but must be kept for
      backwards compatibility.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    summary_metadata: Optional SummaryMetadata proto (which describes which
      plugins may use the summary value).
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  # pylint: enable=line-too-long

  # The summary description is unused now.
  del summary_description

  serialized_summary_metadata = ""
  if summary_metadata:
    serialized_summary_metadata = summary_metadata.SerializeToString()

  with summary_op_util.summary_scope(
      name, family, values=[tensor]) as (tag, scope):
    val = gen_logging_ops._tensor_summary_v2(
        tensor=tensor,
        tag=tag,
        description="",
        name=scope,
        serialized_summary_metadata=serialized_summary_metadata)
    summary_op_util.collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val
