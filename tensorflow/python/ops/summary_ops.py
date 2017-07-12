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

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import summary_op_util
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_logging_ops import *
# pylint: enable=wildcard-import


def tensor_summary(
    name,
    tensor,
    summary_description=None,
    collections=None,
    summary_metadata=None,
    family=None):
  """Outputs a `Summary` protocol buffer with a serialized tensor.proto.

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
        name=scope,
        serialized_summary_metadata=serialized_summary_metadata)
    summary_op_util.collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val

ops.NotDifferentiable("TensorSummary")
