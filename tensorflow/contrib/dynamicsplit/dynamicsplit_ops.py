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

"""Dynamic Split Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util

from tensorflow.python.ops import gen_dynamicsplit_ops
from tensorflow.python.ops import math_ops

from tensorflow.python.ops.gen_dynamicsplit_ops import *


ops.NoGradient("DynamicSplit")


@ops.RegisterShape("DynamicSplit")
def _DecodeCSVShape(op):  # pylint: disable=invalid-name
  """Shape function for the DynamicSplit op."""
  input_shape = op.inputs[0].get_shape()
  # Optionally check that all of other inputs are scalar or empty.
  for default_input in op.inputs[1:]:
    default_input_shape = default_input.get_shape().with_rank(1)
    if default_input_shape[0] > 1:
      raise ValueError(
          "Shape of a default must be a length-0 or length-1 vector.")
  return [input_shape] * len(op.outputs)

def dynamicsplit(record, record_default):
  split_res = gen_dynamicsplit_ops._dynamic_split(
    record,
    record_default,field_delim=',')
  
  return split_res
