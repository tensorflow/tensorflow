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
"""SavedModel utility functions.

Utility functions to assist with setup and construction of the SavedModel proto.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import dtypes


# TensorInfo helpers.


def build_tensor_info(tensor):
  """Utility function to build TensorInfo proto.

  Args:
    tensor: Tensor whose name, dtype and shape are used to build the TensorInfo.

  Returns:
    A TensorInfo protocol buffer constructed based on the supplied argument.
  """
  dtype_enum = dtypes.as_dtype(tensor.dtype).as_datatype_enum
  return meta_graph_pb2.TensorInfo(
      name=tensor.name,
      dtype=dtype_enum,
      tensor_shape=tensor.get_shape().as_proto())
