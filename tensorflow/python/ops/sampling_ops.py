# Copyright 2016 Google Inc. All Rights Reserved.
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
"""## Random Sampling Operators.

TensorFlow provides you functions to sample from distributions.

"""

import sys
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_sampling_ops import *


@ops.RegisterShape("BernoulliSample")
def _BernoulliSampleShape(op):
  a_shape = op.inputs[1].get_shape().with_rank(1)
  b_shape = op.inputs[2].get_shape().with_rank(1)

  assert a_shape == b_shape

  return [a_shape]

ops.NoGradient("BernoulliSample")

@ops.RegisterShape("SampleDistributionIndex")
def _SampleDistributionIndexShape(op):
  batch_size = op.inputs[0].get_shape().with_rank(2)[0].value
  return [tensor_shape.TensorShape([batch_size])]

ops.NoGradient("SampleDistributionIndex")
