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


def risc_add(
    input_lhs,
    input_rhs,
    name='RISC_ADD'):
  return gen_risc_ops.risc_add(input_lhs, input_rhs, name=name)


def risc_broadcast(x, shape, name='RISC_BROADCAST'):
  return gen_risc_ops.risc_broadcast(x, shape, name=name)


def risc_concat(x, axis, name='RISC_CONCAT'):
  return gen_risc_ops.risc_concat(x, axis, name=name)


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


def risc_dot(input_lhs,
             input_rhs,
             transpose_a=False,
             transpose_b=False,
             name='RISC_DOT'):
  return gen_risc_ops.risc_dot(
      input_lhs,
      input_rhs,
      transpose_a=transpose_a,
      transpose_b=transpose_b,
      name=name)


def risc_max(input_lhs, input_rhs, name='RISC_MAX'):
  return gen_risc_ops.risc_max(input_lhs, input_rhs, name=name)


def risc_pad(x, padding, constant_values, name='RISC_PAD'):
  return gen_risc_ops.risc_pad(x, padding, constant_values, name=name)


def risc_pool(x, ksize, strides, pooling_type='MAX', name='RISC_POOL'):
  return gen_risc_ops.risc_pool(
      x, ksize, strides, pooling_type=pooling_type, name=name)


def risc_reshape(x, shape, name='RISC_RESHAPE'):
  return gen_risc_ops.risc_reshape(x, shape, name=name)


def risc_shape(x, name='RISC_SHAPE'):
  return gen_risc_ops.risc_shape(x, name=name)


def risc_slice(x, begin, size, name='RISC_SLICE'):
  return gen_risc_ops.risc_slice(x, begin, size, name=name)
