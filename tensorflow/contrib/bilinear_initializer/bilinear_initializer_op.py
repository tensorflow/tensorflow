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
# =============================================================================
"""Initialize with bilinear interpolation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.bilinear_initializer.ops import gen_bilinear_initializer_op_wrapper
from tensorflow.contrib.util import loader
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

_bilinear_initializer_so = loader.load_op_library(
    resource_loader.get_path_to_datafile('bilinear_initializer.so'))


def bilinear_initializer(shape, dtype=dtypes.float32):
  """Bilinear Initializer Operation.

  Computes batch filters with bilinear weights. Result is a
  4D tensor of dimension [H, W, C, N] where:
  (1) H = height of kernel
  (2) W = width of kernel
  (3) C = number of input channels
  (4) N = number of output channels (number of filters)
  Typical use case is weight initialization for deconvolution layer.
  
  Args:
    shape: The shape of the tensor to be initialized.
    dtype: The data type. Only floating point types are supported.
    
  Returns:
    A 2D tensor initialized with bilinear interpolation.
  """
  return gen_bilinear_initializer_op_wrapper.bilinear_initializer(shape)


ops.NotDifferentiable('BilinearInitializer')
