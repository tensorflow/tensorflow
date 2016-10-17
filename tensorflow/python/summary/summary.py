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

# pylint: disable=line-too-long
"""This module contains ops for generating summaries.

## Summary Ops
@@tensor_summary
@@scalar

"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework.dtypes import as_dtype
from tensorflow.python.ops.summary_ops import tensor_summary
from tensorflow.python.util.all_util import make_all


def scalar(name, tensor, summary_description=None, collections=None):
  """Outputs a `Summary` protocol buffer containing a single scalar value.

  The generated Summary has a Tensor.proto containing the input Tensor.

  Args:
    name: A name for the generated node. Will also serve as the series name in
      TensorBoard.
    tensor: A tensor containing a single floating point or integer value.
    summary_description: Optional summary_description_pb2.SummaryDescription
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.

  Returns:
    A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.

  Raises:
    ValueError: If tensor has the wrong shape or type.
  """
  dtype = as_dtype(tensor.dtype)
  if dtype.is_quantized or not (dtype.is_integer or dtype.is_floating):
    raise ValueError("Can't create scalar summary for type %s." % dtype)

  shape = tensor.get_shape()
  if not shape.is_compatible_with(tensor_shape.scalar()):
    raise ValueError("Can't create scalar summary for shape %s." % shape)

  if summary_description is None:
    summary_description = summary_pb2.SummaryDescription()
  summary_description.type_hint = "scalar"

  return tensor_summary(name, tensor, summary_description, collections)


__all__ = make_all(__name__)
