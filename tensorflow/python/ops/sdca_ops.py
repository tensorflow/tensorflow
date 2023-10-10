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
"""A Dual Coordinate Ascent optimizer library for training fast linear models.
"""

# pylint: disable=g-bad-name
from tensorflow.python.framework import ops

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_sdca_ops import *
# pylint: enable=wildcard-import

ops.NotDifferentiable("SdcaFprint")
ops.NotDifferentiable("SdcaOptimizer")
ops.NotDifferentiable("SdcaOptimizerV2")
ops.NotDifferentiable("SdcaShrinkL1")
