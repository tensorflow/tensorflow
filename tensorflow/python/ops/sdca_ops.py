# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed unde_sdca_opsr the Apache License, Version 2.0 (the "License");
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
"""## Script Language Operators.

A Dual Cordinate Ascent optimizer for TensorFlow for training fast linear
models.

@@sdca_optimizer
@@sdca_fprint
@@sdca_shrink_l1
"""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_sdca_ops import *
# pylint: enable=wildcard-import

# pylint: disable=anomalous-backslash-in-string,protected-access
ops.NotDifferentiable("SdcaFprint")
ops.NotDifferentiable("SdcaOptimizer")
ops.NotDifferentiable("SdcaShrinkL1")
ops.RegisterShape("SdcaFprint")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SdcaOptimizer")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SdcaShrinkL1")(common_shapes.call_cpp_shape_fn)
