/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class BinopScalingTest : public ClientLibraryTestBase {};

TEST_F(BinopScalingTest, MatrixPlusPseudoMatrixRowVector_32x4) {
  auto alhs = MakeLinspaceArray2D(0.0, 1.0, 32, 4);
  auto arhs = MakeLinspaceArray2D(0.0, 1.0, 1, 4);

  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR2FromArray2D<float>(*alhs);
  auto rhs = builder.ConstantR2FromArray2D<float>(*arhs);
  builder.Add(lhs, rhs);

  auto aexpected = ReferenceUtil::MapWithIndexArray2D(
      *alhs, [&](float lhs_value, int64 row, int64 col) {
        return lhs_value + (*arhs)(0, col);
      });
  ComputeAndCompareR2<float>(&builder, *aexpected, {}, ErrorSpec(0.0001));
}

TEST_F(BinopScalingTest, MatrixPlusPseudoMatrixRowVector_129x129) {
  auto alhs = MakeLinspaceArray2D(0.0, 1.0, 129, 129);
  auto arhs = MakeLinspaceArray2D(0.0, 1.0, 1, 129);

  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR2FromArray2D<float>(*alhs);
  auto rhs = builder.ConstantR2FromArray2D<float>(*arhs);
  builder.Add(lhs, rhs);

  auto aexpected = ReferenceUtil::MapWithIndexArray2D(
      *alhs, [&](float lhs_value, int64 row, int64 col) {
        return lhs_value + (*arhs)(0, col);
      });
  ComputeAndCompareR2<float>(&builder, *aexpected, {}, ErrorSpec(0.0001));
}

TEST_F(BinopScalingTest, MatrixPlusPseudoMatrixColVector_9x5) {
  auto alhs = MakeLinspaceArray2D(0.0, 1.0, 9, 5);
  auto arhs = MakeLinspaceArray2D(0.0, 1.0, 9, 1);

  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR2FromArray2D<float>(*alhs);
  auto rhs = builder.ConstantR2FromArray2D<float>(*arhs);
  builder.Add(lhs, rhs);

  auto aexpected = ReferenceUtil::MapWithIndexArray2D(
      *alhs, [&](float lhs_value, int64 row, int64 col) {
        return lhs_value + (*arhs)(row, 0);
      });
  ComputeAndCompareR2<float>(&builder, *aexpected, {}, ErrorSpec(0.0001));
}

TEST_F(BinopScalingTest, MatrixPlusPseudoMatrixColVector_129x257) {
  auto alhs = MakeLinspaceArray2D(0.0, 1.0, 129, 257);
  auto arhs = MakeLinspaceArray2D(0.0, 1.0, 129, 1);

  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR2FromArray2D<float>(*alhs);
  auto rhs = builder.ConstantR2FromArray2D<float>(*arhs);
  builder.Add(lhs, rhs);

  auto aexpected = ReferenceUtil::MapWithIndexArray2D(
      *alhs, [&](float lhs_value, int64 row, int64 col) {
        return lhs_value + (*arhs)(row, 0);
      });
  ComputeAndCompareR2<float>(&builder, *aexpected, {}, ErrorSpec(0.0001));
}

TEST_F(BinopScalingTest, R0PlusR2F32) {
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR0<float>(42.0);
  auto rhs = builder.ConstantR2<float>({
      {1.0, 2.0}, {3.0, 4.0},
  });
  builder.Add(lhs, rhs);

  Array2D<float> expected(2, 2);
  expected(0, 0) = 42.0 + 1.0;
  expected(0, 1) = 42.0 + 2.0;
  expected(1, 0) = 42.0 + 3.0;
  expected(1, 1) = 42.0 + 4.0;
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(BinopScalingTest, R4PlusR0S32) {
  ComputationBuilder builder(client_, TestName());
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

  auto lhs = builder.ConstantR4FromArray4D(lhs_array);
  auto rhs = builder.ConstantR0<int>(42);
  builder.Add(lhs, rhs);
  ComputeAndCompareR4<int>(&builder, expected, {});
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
