# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Wrappers for primitive Neural Net (NN) Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.quantization.ops import gen_nn_ops
from tensorflow.contrib.quantization.ops.gen_nn_ops import *
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops


ops.RegisterShape("QuantizedAvgPool")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("QuantizedBiasAdd")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("QuantizedConv2D")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("QuantizedMaxPool")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("QuantizedRelu")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("QuantizedRelu6")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("QuantizedReluX")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("QuantizeDownAndShrinkRange")(common_shapes.call_cpp_shape_fn)
