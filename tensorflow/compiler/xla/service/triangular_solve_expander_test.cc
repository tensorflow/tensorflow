/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class TriangularExpanderTest : public HloTestBase,
                               public ::testing::WithParamInterface<int32> {};

TEST_P(TriangularExpanderTest, TestBlockSize) {
  auto block_size = GetParam();
  std::string hlo_string = R"(
    HloModule TensorFlowTriangularSolve

    ENTRY main {
      a = f32[256,256]{1,0} parameter(0)
      b = f32[256,192]{1,0} parameter(1)
      ROOT triangular-solve = f32[256,192]{1,0} triangular-solve(a, b),
                                    left_side=true, unit_diagonal=true,
                                    lower=true, transpose_a=NO_TRANSPOSE
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  {
    TriangularSolveExpander triangular_solve_expander(block_size);

    TF_ASSERT_OK_AND_ASSIGN(
        bool result, RunHloPass(&triangular_solve_expander, module.get()));
    EXPECT_TRUE(result);
  }

  // To test triangular solver expander we generate simple bidiagonal matrix:
  // Solve a * x = b.
  // Check that shape is still valid.
  // Use reference matrix multiplication to test validity of result.

  Array2D<float> a(256, 256);
  for (int64 row = 0; row < a.dim(0); ++row) {
    a(row, row) = 1;
    if (row > 0) {
      a(row, row - 1) = 0.01;
    }
  }

  Array2D<float> b(256, 192);
  const float kMax = static_cast<float>(b.dim(0) * b.dim(1) + 1);
  for (int64 row = 0; row < b.dim(0); ++row) {
    for (int64 col = 0; col < b.dim(1); ++col) {
      b(row, col) = static_cast<float>(row + col + 1) / kMax;
    }
  }
  auto la = LiteralUtil::CreateR2FromArray2D(a);
  auto lb = LiteralUtil::CreateR2FromArray2D(b);

  TF_ASSERT_OK_AND_ASSIGN(Literal lx, Execute(std::move(module), {&la, &lb}));

  auto x_shape = lx.shape();
  EXPECT_EQ(x_shape.dimensions_size(), 2);
  EXPECT_EQ(x_shape.dimensions(0), b.dim(0));
  EXPECT_EQ(x_shape.dimensions(1), b.dim(1));

  Array2D<float> x(x_shape.dimensions(0), x_shape.dimensions(1));
  x.SetValues(lx.data<float>());

  auto ref_b = ReferenceUtil::MatmulArray2D(a, x);
  auto ref_lb = LiteralUtil::CreateR2FromArray2D(*ref_b);

  EXPECT_TRUE(
      LiteralTestUtil::NearOrEqual(ref_lb, lb, ErrorSpec{0.001, 0.001}));
}

// block_size test limits based on the following considerations:
// - test at least twice the range of original value
// - try to test odd values unaligned with matrix dims
// - full 1-256 range test takes too long to run

INSTANTIATE_TEST_CASE_P(TriangularExpanderTestInstances, TriangularExpanderTest,
                        ::testing::Range(2, 256, 7));

}  // namespace
}  // namespace xla
