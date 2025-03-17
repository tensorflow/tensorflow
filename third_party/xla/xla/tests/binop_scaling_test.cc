/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/array2d.h"
#include "xla/array4d.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/reference_util.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class BinopScalingTest : public ClientLibraryTestBase {};

TEST_F(BinopScalingTest, MatrixPlusPseudoMatrixRowVector_32x4) {
  auto alhs = MakeLinspaceArray2D(0.0, 1.0, 32, 4);
  auto arhs = MakeLinspaceArray2D(0.0, 1.0, 1, 4);

  XlaBuilder builder(TestName());
  auto lhs = ConstantR2FromArray2D<float>(&builder, *alhs);
  auto rhs = ConstantR2FromArray2D<float>(&builder, *arhs);
  Add(lhs, rhs);

  auto aexpected = ReferenceUtil::MapWithIndexArray2D(
      *alhs, [&](float lhs_value, int64_t row, int64_t col) {
        return lhs_value + (*arhs)(0, col);
      });
  ComputeAndCompareR2<float>(&builder, *aexpected, {}, ErrorSpec(0.0001));
}

TEST_F(BinopScalingTest, MatrixPlusPseudoMatrixRowVector_129x129) {
  auto alhs = MakeLinspaceArray2D(0.0, 1.0, 129, 129);
  auto arhs = MakeLinspaceArray2D(0.0, 1.0, 1, 129);

  XlaBuilder builder(TestName());
  auto lhs = ConstantR2FromArray2D<float>(&builder, *alhs);
  auto rhs = ConstantR2FromArray2D<float>(&builder, *arhs);
  Add(lhs, rhs);

  auto aexpected = ReferenceUtil::MapWithIndexArray2D(
      *alhs, [&](float lhs_value, int64_t row, int64_t col) {
        return lhs_value + (*arhs)(0, col);
      });
  ComputeAndCompareR2<float>(&builder, *aexpected, {}, ErrorSpec(0.0001));
}

TEST_F(BinopScalingTest, MatrixPlusPseudoMatrixColVector_9x5) {
  auto alhs = MakeLinspaceArray2D(0.0, 1.0, 9, 5);
  auto arhs = MakeLinspaceArray2D(0.0, 1.0, 9, 1);

  XlaBuilder builder(TestName());
  auto lhs = ConstantR2FromArray2D<float>(&builder, *alhs);
  auto rhs = ConstantR2FromArray2D<float>(&builder, *arhs);
  Add(lhs, rhs);

  auto aexpected = ReferenceUtil::MapWithIndexArray2D(
      *alhs, [&](float lhs_value, int64_t row, int64_t col) {
        return lhs_value + (*arhs)(row, 0);
      });
  ComputeAndCompareR2<float>(&builder, *aexpected, {}, ErrorSpec(0.0001));
}

TEST_F(BinopScalingTest, MatrixPlusPseudoMatrixColVector_129x257) {
  auto alhs = MakeLinspaceArray2D(0.0, 1.0, 129, 257);
  auto arhs = MakeLinspaceArray2D(0.0, 1.0, 129, 1);

  XlaBuilder builder(TestName());
  auto lhs = ConstantR2FromArray2D<float>(&builder, *alhs);
  auto rhs = ConstantR2FromArray2D<float>(&builder, *arhs);
  Add(lhs, rhs);

  auto aexpected = ReferenceUtil::MapWithIndexArray2D(
      *alhs, [&](float lhs_value, int64_t row, int64_t col) {
        return lhs_value + (*arhs)(row, 0);
      });
  ComputeAndCompareR2<float>(&builder, *aexpected, {}, ErrorSpec(0.0001));
}

TEST_F(BinopScalingTest, R0PlusR2F32) {
  XlaBuilder builder(TestName());
  auto lhs = ConstantR0<float>(&builder, 42.0);
  auto rhs = ConstantR2<float>(&builder, {
                                             {1.0, 2.0},
                                             {3.0, 4.0},
                                         });
  Add(lhs, rhs);

  Array2D<float> expected(2, 2);
  expected(0, 0) = 42.0 + 1.0;
  expected(0, 1) = 42.0 + 2.0;
  expected(1, 0) = 42.0 + 3.0;
  expected(1, 1) = 42.0 + 4.0;
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(BinopScalingTest, R4PlusR0S32) {
  XlaBuilder builder(TestName());
  // clang-format off
  Array4D<int> lhs_array({
    {{{1, 2},
      {3, 4},
      {5, 6}}},
    {{{7, 8},
      {9, 10},
      {11, 12}}},
  });
  Array4D<int> expected({
    {{{43, 44},
      {45, 46},
      {47, 48}}},
    {{{49, 50},
      {51, 52},
      {53, 54}}},
  });
  // clang-format on

  auto lhs = ConstantR4FromArray4D(&builder, lhs_array);
  auto rhs = ConstantR0<int>(&builder, 42);
  Add(lhs, rhs);
  ComputeAndCompareR4<int>(&builder, expected, {});
}

}  // namespace
}  // namespace xla
