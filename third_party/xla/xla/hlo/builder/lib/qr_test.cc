/* Copyright 2018 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/builder/lib/qr.h"

#include <algorithm>
#include <cstdint>

#include <gtest/gtest.h>
#include "xla/array.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using QrTest = ClientLibraryTestRunnerMixin<
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>>;

TEST_F(QrTest, Simple) {
  Array2D<float> data({
      {4, 6, 8, 10},
      {6, 45, 54, 63},
      {8, 54, 146, 166},
      {10, 63, 166, 310},
  });

  for (bool full_matrices : {false, true}) {
    for (int64_t m : {3, 4}) {
      for (int64_t n : {3, 4}) {
        XlaBuilder builder(TestName());
        XlaOp a, q, r;
        Array<float> a_vals = data.Slice({0, 0}, {m, n});
        const Literal expected = LiteralUtil::CreateFromArray(a_vals);
        const Literal a_data =
            CreateParameterAndTransferLiteral(0, expected, "a", &builder, &a);
        QrExplicit(a, full_matrices, q, r);

        // Verifies that the decomposition composes back to the original matrix.
        //
        // This isn't a terribly demanding test, (e.g., we should verify that Q
        // is orthonormal and R is upper-triangular) but it's awkward to write
        // such tests without more linear algebra libraries. It's easier to test
        // the numerics from Python, anyway, where we have access to numpy and
        // scipy.
        BatchDot(q, r, PrecisionConfig::HIGHEST);
        ASSERT_OK_AND_ASSIGN(Shape q_shape, builder.GetShape(q));
        ASSERT_OK_AND_ASSIGN(Shape r_shape, builder.GetShape(r));
        EXPECT_EQ(q_shape, ShapeUtil::MakeShape(
                               F32, {m, full_matrices ? m : std::min(m, n)}));
        EXPECT_EQ(r_shape, ShapeUtil::MakeShape(
                               F32, {full_matrices ? m : std::min(m, n), n}));
        ComputeAndCompareLiteral(&builder, expected, {&a_data},
                                 ErrorSpec(1e-4, 1e-4));
      }
    }
  }
}

TEST_F(QrTest, ZeroDiagonal) {
  XlaBuilder builder(TestName());

  Array2D<float> a_vals({
      {0, 1, 1},
      {1, 0, 1},
      {1, 1, 0},
  });

  XlaOp a, q, r;
  const Literal a_data = CreateR2Parameter<float>(a_vals, 0, "a", &builder, &a);
  QrExplicit(a, /*full_matrices=*/true, q, r);

  // Verifies that the decomposition composes back to the original matrix.
  //
  // This isn't a terribly demanding test, (e.g., we should verify that Q is
  // orthonormal and R is upper-triangular) but it's awkward to write such tests
  // without more linear algebra libraries. It's easier to test the numerics
  // from Python, anyway, where we have access to numpy and scipy.
  BatchDot(q, r, PrecisionConfig::HIGHEST);

  ComputeAndCompareR2<float>(&builder, a_vals, {&a_data},
                             ErrorSpec(1e-4, 1e-4));
}

TEST_F(QrTest, SimpleBatched) {
  XlaBuilder builder(TestName());

  Array3D<float> a_vals({
      {
          {4, 6, 8, 10},
          {6, 45, 54, 63},
          {8, 54, 146, 166},
          {10, 63, 166, 310},
      },
      {
          {16, 24, 8, 12},
          {24, 61, 82, 48},
          {8, 82, 456, 106},
          {12, 48, 106, 62},
      },
  });

  XlaOp a, q, r;
  const Literal a_data = CreateR3Parameter<float>(a_vals, 0, "a", &builder, &a);
  QrExplicit(a, /*full_matrices=*/true, q, r);

  BatchDot(q, r, PrecisionConfig::HIGHEST);

  ComputeAndCompareR3<float>(&builder, a_vals, {&a_data},
                             ErrorSpec(1e-4, 1e-4));
}

TEST_F(QrTest, SubnormalComplex) {
  // Verifies that we don't get NaNs in the case that the norm of a complex
  // number would be denormal but its imaginary value is not exactly 0.
  Array2D<complex64> a_vals({
      {complex64(4e-20, 5e-23), 6, 80},
      {0, 45, 54},
      {0, 54, 146},
  });

  XlaBuilder builder(TestName());
  XlaOp a, q, r;
  const Literal expected = LiteralUtil::CreateFromArray(a_vals);
  const Literal a_data =
      CreateParameterAndTransferLiteral(0, expected, "a", &builder, &a);
  QrExplicit(a, /*full_matrices=*/true, q, r);
  BatchDot(q, r, PrecisionConfig::HIGHEST);
  ComputeAndCompareLiteral(&builder, expected, {&a_data},
                           ErrorSpec(1e-4, 1e-4));
}

TEST_F(QrTest, DuplicateHouseholderExpansion) {
  XlaBuilder builder(TestName());

  Array2D<float> a0_vals({
      {0, 1, 1},
      {1, 0, 1},
      {1, 1, 0},
  });
  Array2D<float> a1_vals({
      {1, 0},
      {0, 1},
      {1, 0},
  });

  // Verifies that different computations are created to generate HouseHolder
  // transformations with identical QR shapes, but different tau shapes.
  // The first QR decomposition should generate a ([3,3], [3]) computation,
  // the second should generate a ([3,3], [2]) computation. Mismatch will result
  // in compilation failure.

  XlaOp a0, q0, r0;
  const Literal a0_data =
      CreateR2Parameter<float>(a0_vals, 0, "a0", &builder, &a0);
  QrExplicit(a0, /*full_matrices=*/true, q0, r0);

  XlaOp a1, q1, r1;
  const Literal a1_data =
      CreateR2Parameter<float>(a1_vals, 1, "a1", &builder, &a1);
  QrExplicit(a1, /*full_matrices=*/true, q1, r1);

  // Verifies that the decomposition composes back to the original matrix.
  BatchDot(q1, r1, PrecisionConfig::HIGHEST);

  ComputeAndCompareR2<float>(&builder, a1_vals, {&a0_data, &a1_data},
                             ErrorSpec(1e-4, 1e-4));
}

}  // namespace
}  // namespace xla
