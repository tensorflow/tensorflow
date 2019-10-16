# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for helping test ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range


def ConvertBetweenDataFormats(x, data_format_src, data_format_dst):
  """Converts 4D tensor between data formats."""

  valid_data_formats = ["NHWC", "NCHW", "HWNC", "HWCN"]
  if data_format_src not in valid_data_formats:
    raise ValueError("data_format_src must be of %s, got %s." %
                     (valid_data_formats, data_format_src))
  if data_format_dst not in valid_data_formats:
    raise ValueError("data_format_dst must be of %s, got %s." %
                     (valid_data_formats, data_format_dst))
  if len(x.shape) != 4:
    raise ValueError("x must be 4D, got shape %s." % x.shape)

  if data_format_src == data_format_dst:
    return x

  dim_map = {d: i for i, d in enumerate(data_format_src)}
  transpose_dims = [dim_map[d] for d in data_format_dst]
  return np.transpose(x, transpose_dims)


def PermuteDimsBetweenDataFormats(dims, data_format_src, data_format_dst):
  """Get new shape for converting between data formats."""

  valid_data_formats = ["NHWC", "NCHW", "HWNC", "HWCN"]
  if data_format_src not in valid_data_formats:
    raise ValueError("data_format_src must be of %s, got %s." %
                     (valid_data_formats, data_format_src))
  if data_format_dst not in valid_data_formats:
    raise ValueError("data_format_dst must be of %s, got %s." %
                     (valid_data_formats, data_format_dst))
  if len(dims) != 4:
    raise ValueError("dims must be of length 4, got %s." % dims)

  if data_format_src == data_format_dst:
    return dims

  dim_map = {d: i for i, d in enumerate(data_format_src)}
  permuted_dims = [dims[dim_map[d]] for d in data_format_dst]
  return permuted_dims


_JIT_WARMUP_ITERATIONS = 10


def RunWithWarmup(sess, op_to_run, feed_dict, options=None, run_metadata=None):
  """Runs a graph a few times to ensure that its clusters are compiled."""
  for _ in range(0, _JIT_WARMUP_ITERATIONS):
    sess.run(op_to_run, feed_dict, options=options)
  return sess.run(
      op_to_run, feed_dict, options=options, run_metadata=run_metadata)
