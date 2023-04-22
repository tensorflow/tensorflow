# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Functions for querying registered kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import kernel_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat


def get_all_registered_kernels():
  """Returns a KernelList proto of all registered kernels.
  """
  buf = c_api.TF_GetAllRegisteredKernels()
  data = c_api.TF_GetBuffer(buf)
  kernel_list = kernel_def_pb2.KernelList()
  kernel_list.ParseFromString(compat.as_bytes(data))
  return kernel_list


def get_registered_kernels_for_op(name):
  """Returns a KernelList proto of registered kernels for a given op.

  Args:
    name: A string representing the name of the op whose kernels to retrieve.
  """
  buf = c_api.TF_GetRegisteredKernelsForOp(name)
  data = c_api.TF_GetBuffer(buf)
  kernel_list = kernel_def_pb2.KernelList()
  kernel_list.ParseFromString(compat.as_bytes(data))
  return kernel_list
