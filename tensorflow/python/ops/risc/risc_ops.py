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
"""RISC Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.ops import gen_risc_ops

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.risc_ops_gen import *
# pylint: enable=wildcard-import


def risc_add(
    input_lhs,
    input_rhs,
    name='RISC_ADD'):
  return gen_risc_ops.risc_add(input_lhs, input_rhs, name=name)


def risc_conv(x,
              kernel,
              strides,
              data_format='NHWC',
              dilations=None,
              name='RISC_CONV'):
  return gen_risc_ops.risc_conv(
      x,
      kernel,
      strides,
      data_format=data_format,
      dilations=dilations,
      name=name)


def risc_max(input_lhs, input_rhs, name='RISC_MAX'):
  return gen_risc_ops.risc_max(input_lhs, input_rhs, name=name)
