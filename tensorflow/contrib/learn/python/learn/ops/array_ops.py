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

"""TensorFlow ops for array / tensor manipulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework import deprecated
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops as array_ops_
from tensorflow.python.ops import math_ops


@deprecated('2016-12-01', 'Use `tf.one_hot` instead.')
def one_hot_matrix(tensor_in, num_classes, on_value=1.0, off_value=0.0,
                   name=None):
  """Encodes indices from given tensor as one-hot tensor.

  TODO(ilblackdragon): Ideally implementation should be
  part of TensorFlow with Eigen-native operation.

  Args:
    tensor_in: Input `Tensor` of shape [N1, N2].
    num_classes: Number of classes to expand index into.
    on_value: `Tensor` or float, value to fill-in given index.
    off_value: `Tensor` or float, value to fill-in everything else.
    name: Name of the op.
  Returns:
    `Tensor` of shape `[N1, N2, num_classes]` with 1.0 for each id in original
    tensor.
  """
  with ops.name_scope(
      name, 'one_hot_matrix',
      [tensor_in, num_classes, on_value, off_value]) as name_scope:
    return array_ops_.one_hot(
        math_ops.cast(tensor_in, dtypes.int64), num_classes, on_value,
        off_value, name=name_scope)
