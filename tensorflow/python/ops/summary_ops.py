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
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_logging_ops import *
# pylint: enable=wildcard-import


def _Collect(val, collections, default_collections):
  if collections is None:
    collections = default_collections
  for key in collections:
    ops.add_to_collection(key, val)


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
    _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val


ops.NotDifferentiable("TensorSummary")
