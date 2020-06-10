# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tensorflow numpy API."""
# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.numpy_ops import np_random as random

# pylint: disable=wildcard-import

from tensorflow.python.ops.numpy_ops.np_array_ops import *
# TODO(wangpeng): Move ShardedNdArray, convert_to_tensor, tensor_to_ndarray out
# of here.
from tensorflow.python.ops.numpy_ops.np_arrays import convert_to_tensor
from tensorflow.python.ops.numpy_ops.np_arrays import ndarray
from tensorflow.python.ops.numpy_ops.np_arrays import ShardedNdArray
from tensorflow.python.ops.numpy_ops.np_arrays import tensor_to_ndarray
from tensorflow.python.ops.numpy_ops.np_dtypes import *
from tensorflow.python.ops.numpy_ops.np_math_ops import *
from tensorflow.python.ops.numpy_ops.np_utils import finfo
from tensorflow.python.ops.numpy_ops.np_utils import promote_types
from tensorflow.python.ops.numpy_ops.np_utils import result_type
# pylint: enable=wildcard-import

# pylint: disable=redefined-builtin,undefined-variable
max = amax
min = amin
round = around
# pylint: enable=redefined-builtin,undefined-variable

from tensorflow.python.ops.array_ops import newaxis
