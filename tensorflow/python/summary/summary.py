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

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework.dtypes import as_dtype
from tensorflow.python.ops.summary_ops import tensor_summary
from tensorflow.python.util.all_util import make_all

SCALAR_SUMMARY_LABEL = "tf_summary_type:scalar"


def scalar(display_name,
           tensor,
           description="",
           labels=None,
           collections=None,
           name=None):
  """Outputs a `Summary` protocol buffer containing a single scalar value.

  The generated Summary has a Tensor.proto containing the input Tensor.

  Args:
    display_name: A name to associate with the data series. Will be used to
      organize output data and as a name in visualizers.
    tensor: A tensor containing a single floating point or integer value.
    description: An optional long description of the data being output.
    labels: a list of strings used to attach metadata.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    name: An optional name for the generated node (optional).

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

  if labels is None:
    labels = []
  else:
    labels = labels[:]  # Otherwise we would mutate the input argument

  labels.append(SCALAR_SUMMARY_LABEL)

  with ops.name_scope(name, "ScalarSummary", [tensor]):
    tensor = ops.convert_to_tensor(tensor)
    return tensor_summary(display_name, tensor, description, labels,
                          collections, name)


__all__ = make_all(__name__)
