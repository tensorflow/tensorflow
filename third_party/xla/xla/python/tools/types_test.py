# Copyright 2024 The OpenXLA Authors.
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

import itertools
import math
import re
from typing import List, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

# NOTE: These protos are not part of the public API, therefore we cannot
# abide by [g-direct-tensorflow-import].
# pylint: disable=g-direct-tensorflow-import
from local_xla.xla import xla_data_pb2
from xla.python.tools import types
# pylint: enable=g-direct-tensorflow-import


class MakeNdarrayInvalidTest(absltest.TestCase):
  """Tests for invalid/unsupported arguments to `make_ndarray`."""

  def setUp(self):
    super().setUp()
    self.assert_cannot_create_from_proto = self.assertRaisesRegex(
        ValueError, re.escape('Cannot `xla::Literal::CreateFromProto`')
    )

  # NOTE: The `Literal(const Shape&, bool, ArrayValueState)` ctor does
  # a CHECK forbidding `element_size_in_bits` from being specified;
  # so we can't test anything about custom sizes here.

  def testMissingLayout(self):
    # NOTE: `CreateFromProto` requires explicit `shape.layout.minor_to_major`.
    # Though in principle it could use a default ctor instead, like we
    # do in `make_named_parameter` below`.
    pb = xla_data_pb2.LiteralProto(
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.PrimitiveType.F64,
            dimensions=[1, 2, 3],
        )
    )
    with self.assert_cannot_create_from_proto:
      types.make_ndarray(pb)

  def testMissingMinorToMajor(self):
    # NOTE: `CreateFromProto` requires explicit `shape.layout.minor_to_major`.
    # Though in principle it could use a default ctor instead, like we
    # do in `make_named_parameter` below`.
    pb = xla_data_pb2.LiteralProto(
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.PrimitiveType.F64,
            dimensions=[1, 2, 3],
            layout=xla_data_pb2.LayoutProto(),
        )
    )
    with self.assert_cannot_create_from_proto:
      types.make_ndarray(pb)

  def testInvalidPrimitiveType(self):
    # NOTE: The `is_dynamic_dimension` field isn't required by
    # `CreateFromProto`; however, the `Shape(const ShapeProto&)` ctor
    # will log warnings if we leave it unspecified.
    pb = xla_data_pb2.LiteralProto(
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.PrimitiveType.PRIMITIVE_TYPE_INVALID,
            dimensions=[1, 2, 3],
            is_dynamic_dimension=[False, False, False],
            layout=xla_data_pb2.LayoutProto(
                minor_to_major=[0, 1, 2],
            ),
        )
    )
    with self.assert_cannot_create_from_proto:
      types.make_ndarray(pb)

  def testHasDimLevelTypes(self):
    # NOTE: `CreateFromProto` forbids `dim_level_types` (even if all-dense).
    pb = xla_data_pb2.LiteralProto(
        shape=xla_data_pb2.ShapeProto(
            element_type=xla_data_pb2.PrimitiveType.F64,
            dimensions=[1, 2, 3],
            is_dynamic_dimension=[False, False, False],
            layout=xla_data_pb2.LayoutProto(
                dim_level_types=[
                    xla_data_pb2.DimLevelType.DIM_DENSE,
                    xla_data_pb2.DimLevelType.DIM_DENSE,
                    xla_data_pb2.DimLevelType.DIM_DENSE,
                ],
                minor_to_major=[0, 1, 2],
            ),
        )
    )
    with self.assert_cannot_create_from_proto:
      types.make_ndarray(pb)


class MakeNdarrayValidTestParameter(NamedTuple):
  testcase_name: str
  proto: xla_data_pb2.LiteralProto
  arr: np.ndarray


def make_named_parameter(
    testcase_name: str,
    dimensions: List[int],
    data: List[float],
) -> MakeNdarrayValidTestParameter:
  """Helper function to construct parameters for `MakeNdarrayValidTest`."""
  assert math.prod(dimensions) == len(data)
  nd = len(dimensions)
  proto = xla_data_pb2.LiteralProto(
      shape=xla_data_pb2.ShapeProto(
          element_type=xla_data_pb2.PrimitiveType.F64,
          dimensions=dimensions,
          is_dynamic_dimension=itertools.repeat(False, nd),
          layout=xla_data_pb2.LayoutProto(
              minor_to_major=range(nd),
          ),
      ),
      f64s=data,
  )
  arr = types.make_ndarray(proto)
  return MakeNdarrayValidTestParameter(testcase_name, proto, arr)


@parameterized.named_parameters(
    make_named_parameter('A', [2, 3], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    make_named_parameter('B', [1, 2, 3], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    make_named_parameter('C', [2, 3], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
    make_named_parameter('D', [3, 2], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
)
class MakeNdarrayValidTest(parameterized.TestCase):
  """Correctness tests for valid arguments to `make_ndarray`."""

  def testHasCorrectDtype(self, proto, arr):
    """Test that the result has the right dtype."""
    e = proto.shape.element_type
    d = arr.dtype
    with self.subTest(msg='etype_to_dtype'):
      self.assertEqual(types.etype_to_dtype(e), d)
    with self.subTest(msg='dtype_to_etype'):
      self.assertEqual(e, types.dtype_to_etype(d))

  def testHasCorrectRank(self, proto, arr):
    """Test that the result has the right rank."""
    self.assertLen(proto.shape.dimensions, arr.ndim)

  def testHasCorrectShape(self, proto, arr):
    """Test that the result has the same/right shape."""
    self.assertTupleEqual(tuple(proto.shape.dimensions), arr.shape)

  def testHasCorrectData(self, proto, arr):
    """Test that the result has the same/right data."""
    # TODO(wrengr): Figure out a way to abstract away the name of the
    # proto field containing the data; so that we can test multiple types.
    self.assertSequenceAlmostEqual(proto.f64s, list(np.nditer(arr)))

  # TODO(wrengr): Add tests for:
  # * dynamic dimension sizes.
  # * non-trivial `minor_to_major`.
  # * problematic types {PRED,F16,C64,C128} are all handled correctly.
  # * BF16 is handled correctly.
  # * tuples are handled correctly


if __name__ == '__main__':
  absltest.main()
