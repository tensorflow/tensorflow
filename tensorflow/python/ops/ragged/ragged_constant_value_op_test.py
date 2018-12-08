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
"""Tests for ragged.constant_value."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import ragged
from tensorflow.python.ops.ragged import ragged_test_util
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedConstantValueOpTest(ragged_test_util.RaggedTensorTestCase,
                                parameterized.TestCase):

  @parameterized.parameters(
      #=========================================================================
      # 0-dimensional tensors.
      dict(pylist='x', expected_shape=()),

      #=========================================================================
      # 1-dimensional tensors.
      dict(pylist=[1, 2, 3], expected_shape=(3,)),

      #=========================================================================
      # 2-dimensional tensors.
      dict(pylist=[[1, 2, 3], [4], [5, 6]], expected_shape=(3, None)),
      dict(pylist=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], expected_shape=(3, None)),

      #=========================================================================
      # 3-dimensional tensors.
      dict(
          pylist=[[[1, 2], [3, 4]], [], [[5, 6], [7, 8], [9, 0]]],
          expected_shape=(3, None, None)),
      dict(
          pylist=[[[1, 2], [3, 4]], [], [[5, 6], [7, 8], [9, 0]]],
          ragged_rank=1,
          expected_shape=(3, None, 2)),
      dict(
          pylist=[[[1, 2], [3, 4]], [], [[5, 6], [7, 8], [9, 0]]],
          inner_shape=(2,),
          expected_shape=(3, None, 2)),
      dict(
          pylist=[[[1, 2], [3, 4]], [], [[5, 6], [7, 8], [9, 0]]],
          ragged_rank=1,
          inner_shape=(2,),
          expected_shape=(3, None, 2)),
      #=========================================================================
      # 4-dimensional tensors.
      dict(
          pylist=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                  [[[2, 4], [6, 8]], [[1, 5], [7, 9]]]],
          expected_shape=(2, None, None, None)),
      dict(
          pylist=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                  [[[2, 4], [6, 8]], [[1, 5], [7, 9]]]],
          ragged_rank=1,
          expected_shape=(2, None, 2, 2)),
      dict(
          pylist=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                  [[[2, 4], [6, 8]], [[1, 5], [7, 9]]]],
          inner_shape=(2,),
          expected_shape=(2, None, None, 2)),
      dict(
          pylist=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                  [[[2, 4], [6, 8]], [[1, 5], [7, 9]]]],
          inner_shape=(2, 2),
          expected_shape=(2, None, 2, 2)),

      #=========================================================================
      # Empty tensors (no scalar values) w/ default ragged_rank and inner_shape
      dict(pylist=[], expected_shape=(0,)),
      dict(pylist=[[], [], []], expected_shape=(3, None)),
      dict(
          pylist=[[[], []], [], [[], [[]]]],
          expected_shape=(3, None, None, None)),

      #=========================================================================
      # Empty tensors (no scalar values) w/ explicit ragged_rank or inner_shape
      dict(pylist=[], ragged_rank=1, expected_shape=(0, None)),
      dict(pylist=[], ragged_rank=2, expected_shape=(0, None, None)),
      dict(pylist=[], inner_shape=(0, 100, 20), expected_shape=(0, 100, 20)),
      dict(
          pylist=[],
          ragged_rank=1,
          inner_shape=(100, 20),
          expected_shape=(0, None, 100, 20)),
      dict(
          pylist=[],
          ragged_rank=2,
          inner_shape=(100, 20),
          expected_shape=(0, None, None, 100, 20)),
      dict(pylist=[[], [], []], ragged_rank=2, expected_shape=(3, None, None)),
      dict(pylist=[], inner_shape=(0,), expected_shape=(0,)),
      dict(pylist=[[]], inner_shape=(1, 0), expected_shape=(1, 0)),

      #=========================================================================
      # default/inferred dtypes.
      #
      # Note: numpy has different default/inferred types than tensorflow.
      # Since we are using values, not tensors, we get the default numpy types
      # here.
      dict(pylist=[], expected_dtype=np.float64),
      dict(pylist=[[[], [[[]], []]]], expected_dtype=np.float64),
      dict(pylist=[[1, 2], [3], [4, 5, 6]], expected_dtype=np.int64),
      dict(pylist=[[1., 2.], [], [4., 5., 6.]], expected_dtype=np.float64),
      dict(pylist=[[1, 2], [3.], [4, 5, 6]], expected_dtype=np.float64),
      dict(pylist=[[b'a', b'b'], [b'c']], expected_dtype=np.dtype('S1')),
      dict(pylist=[[True]], expected_dtype=np.bool),

      #=========================================================================
      # explicit dtypes
      dict(pylist=[], dtype=np.float32),
      dict(pylist=[], dtype=np.dtype('S1')),
      dict(pylist=[[1, 2], [3], [4, 5, 6]], dtype=np.int64),
      dict(pylist=[[1, 2], [3], [4, 5, 6]], dtype=np.int32),
      dict(pylist=[[1, 2], [3], [4, 5, 6]], dtype=np.float32),
      dict(pylist=[[1., 2.], [3.], [4., 5., 6.]], dtype=np.float16),
      dict(pylist=[[1., 2.], [3.], [4., 5., 6.]], dtype=np.float32),
      dict(
          pylist=[[b'a', b'b'], [b'c'], [b'd', b'e', b'f']],
          dtype=np.dtype('S1')),
  )
  def testRaggedValues(self,
                       pylist,
                       dtype=None,
                       ragged_rank=None,
                       inner_shape=None,
                       expected_shape=None,
                       expected_dtype=None):
    """Tests that `ragged_value(pylist).to_list() == pylist`."""
    rt = ragged.constant_value(
        pylist, dtype=dtype, ragged_rank=ragged_rank, inner_shape=inner_shape)

    # If dtype was explicitly specified, check it.
    if dtype is not None:
      self.assertEqual(rt.dtype, dtype)
    if expected_dtype is not None:
      self.assertEqual(rt.dtype, expected_dtype)

    # If ragged_rank was explicitly specified, check it.
    if ragged_rank is not None:
      if isinstance(rt, ragged.RaggedTensorValue):
        self.assertEqual(rt.ragged_rank, ragged_rank)
      else:
        self.assertEqual(0, ragged_rank)

    # If inner_shape was explicitly specified, check it.
    if inner_shape is not None:
      if isinstance(rt, ragged.RaggedTensorValue):
        self.assertEqual(rt.flat_values.shape[1:], inner_shape)
      else:
        self.assertEqual(rt.shape, inner_shape)

    if expected_shape is not None:
      self.assertEqual(tuple(rt.shape), expected_shape)

    if rt.shape:
      if isinstance(rt, ragged.RaggedTensorValue):
        self.assertEqual(rt.to_list(), pylist)
      else:
        self.assertEqual(rt.tolist(), pylist)
      if expected_shape is not None:
        self.assertEqual(rt.shape, expected_shape)
    else:
      self.assertEqual(rt, pylist)
      if expected_shape is not None:
        self.assertEqual((), expected_shape)

  @parameterized.parameters(
      dict(
          pylist=12,
          ragged_rank=1,
          exception=ValueError,
          message='Invalid pylist=12: incompatible with ragged_rank=1'),
      dict(
          pylist=12,
          inner_shape=(1,),
          exception=ValueError,
          message='Invalid pylist=12: incompatible with '
          'dim\\(inner_shape\\)=1'),
      dict(
          pylist=[[[1], [2]]],
          ragged_rank=-1,
          exception=ValueError,
          message='Invalid ragged_rank=-1: must be nonnegative'),
      dict(
          pylist=[[1, [2]]],
          exception=ValueError,
          message='all scalar values must have the same nesting depth'),
      dict(
          pylist=[[[1]], [[[2]]]],
          exception=ValueError,
          message='all scalar values must have the same nesting depth'),
      dict(
          pylist=[[1], [[]]],
          exception=ValueError,
          message='Invalid pylist=.*: empty list nesting is greater '
          'than scalar value nesting'),
      dict(
          pylist=[1, 2, 3],
          ragged_rank=1,
          exception=ValueError,
          message='pylist has scalar values depth 1, but ragged_rank=1 '
          'requires scalar value depth greater than 1'),
      dict(
          pylist=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          ragged_rank=2,
          exception=ValueError,
          message='pylist has scalar values depth 2, but ragged_rank=2 '
          'requires scalar value depth greater than 2'),
      dict(
          pylist=[1, 2, 3],
          inner_shape=(1, 1),
          exception=ValueError,
          message='cannot reshape array'),
      dict(
          pylist=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
          inner_shape=(2, 2),
          ragged_rank=1,
          exception=ValueError,
          message='Invalid pylist=.*: incompatible with ragged_rank=1 and '
          'dim\\(inner_shape\\)=2'),
      dict(
          pylist=[[[1, 2], [3, 4]], [[5, 6], [7, 8, 9]]],
          ragged_rank=1,
          exception=ValueError,
          message='inner values have inconsistent shape'),
      dict(
          pylist=[[[], [[]]]],
          ragged_rank=1,
          exception=ValueError,
          message='inner values have inconsistent shape'),
  )
  def testRaggedValuesError(self,
                            pylist,
                            dtype=None,
                            ragged_rank=None,
                            inner_shape=None,
                            exception=None,
                            message=None):
    """Tests that `ragged.constant_value()` raises an expected exception."""
    self.assertRaisesRegexp(
        exception,
        message,
        ragged.constant_value,
        pylist,
        dtype=dtype,
        ragged_rank=ragged_rank,
        inner_shape=inner_shape)


if __name__ == '__main__':
  googletest.main()
