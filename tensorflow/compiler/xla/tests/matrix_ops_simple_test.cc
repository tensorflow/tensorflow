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

#include <algorithm>
#include <memory>
#include <string>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class MatOpsSimpleTest : public ClientLibraryTestBase {
 protected:
  Computation BuildSum() {
    // sum(x, y) = x + y
    ComputationBuilder builder(client_, "sum");
    auto x_value =
        builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x_value");
    auto y_value =
        builder.Parameter(1, ShapeUtil::MakeShape(F32, {}), "y_value");
    builder.Add(x_value, y_value);
    auto computation_status = builder.Build();
    TF_CHECK_OK(computation_status.status());
    return computation_status.ConsumeValueOrDie();
  }

  void TestLinspaceMax(int64 rows, int64 cols) {
    float from = -128.0, to = 256.0;
    std::unique_ptr<Array2D<float>> alhs =
        MakeLinspaceArray2D(from, to, rows, cols);
    auto arhs = MakeUnique<Array2D<float>>(rows, cols, 1.0);

    ComputationBuilder builder(
        client_,
        tensorflow::strings::Printf("max_%lldx%lld_linspace", rows, cols));
    auto lhs = builder.ConstantR2FromArray2D<float>(*alhs);
    auto rhs = builder.ConstantR2FromArray2D<float>(*arhs);
    auto max = builder.Max(lhs, rhs);

    Array2D<float> aexpected(rows, cols);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        aexpected(row, col) = std::max((*alhs)(row, col), (*arhs)(row, col));
      }
    }

    ComputeAndCompareR2<float>(&builder, aexpected, {}, ErrorSpec(1e-6));
  }
};

TEST_F(MatOpsSimpleTest, ExpTwoByTwoValues) {
  ComputationBuilder builder(client_, "exp_2x2");
  auto data = builder.ConstantR2<float>({
      {1.0, 0.0},   // row 0
      {-1.0, 0.5},  // row 1
  });
  builder.Exp(data);

  std::unique_ptr<Literal> expected =
      LiteralUtil::CreateR2<float>({{2.71828, 1.00000},    // row 0
                                    {0.36788, 1.64872}});  // row 1

  ComputeAndCompareLiteral(&builder, *expected, {}, ErrorSpec(1e-5));
}

TEST_F(MatOpsSimpleTest, MapTwoByTwo) {
  Computation add_half;
  {
    // add_half(x) = x + 0.5
    ComputationBuilder builder(client_, "add_half");
    auto x_value =
        builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x_value");
    auto half = builder.ConstantR0<float>(0.5);
    builder.Add(x_value, half);
    auto computation_status = builder.Build();
    ASSERT_IS_OK(computation_status.status());
    add_half = computation_status.ConsumeValueOrDie();
  }

  ComputationBuilder builder(client_, "map_2x2");
  auto data = builder.ConstantR2<float>({
      {1.0, 0.0},   // row 0
      {-1.0, 0.5},  // row 1
  });
  auto map = builder.Map({data}, add_half);

  std::unique_ptr<Literal> expected =
      LiteralUtil::CreateR2<float>({{1.5, 0.5},     // row 0
                                    {-0.5, 1.0}});  // row 1
  ComputeAndCompareLiteral(&builder, *expected, {}, ErrorSpec(1e-5));
}

TEST_F(MatOpsSimpleTest, MaxTwoByTwoValues) {
  ComputationBuilder builder(client_, "max_2x2");
  auto lhs = builder.ConstantR2<float>({
      {7.0, 2.0},   // row 0
      {3.0, -4.0},  // row 1
  });
  auto rhs = builder.ConstantR2<float>({
      {5.0, 6.0},   // row 0
      {1.0, -8.0},  // row 1
  });
  auto max = builder.Max(lhs, rhs);

  std::unique_ptr<Literal> expected =
      LiteralUtil::CreateR2<float>({{7.0, 6.0},     // row 0
                                    {3.0, -4.0}});  // row 1
  ComputeAndCompareLiteral(&builder, *expected, {}, ErrorSpec(1e-6));
}

TEST_F(MatOpsSimpleTest, Max1x1Linspace) { TestLinspaceMax(1, 1); }

TEST_F(MatOpsSimpleTest, Max2x2Linspace) { TestLinspaceMax(2, 2); }

TEST_F(MatOpsSimpleTest, Max3x3Linspace) { TestLinspaceMax(3, 3); }

TEST_F(MatOpsSimpleTest, Max4x4Linspace) { TestLinspaceMax(4, 4); }

TEST_F(MatOpsSimpleTest, Max6x6Linspace) { TestLinspaceMax(6, 6); }

TEST_F(MatOpsSimpleTest, Max8x8Linspace) { TestLinspaceMax(8, 8); }

TEST_F(MatOpsSimpleTest, Max12x12Linspace) { TestLinspaceMax(12, 12); }

TEST_F(MatOpsSimpleTest, Max16x16Linspace) { TestLinspaceMax(16, 16); }

TEST_F(MatOpsSimpleTest, Max32x8Linspace) { TestLinspaceMax(32, 8); }

TEST_F(MatOpsSimpleTest, Max64x8Linspace) { TestLinspaceMax(64, 8); }

class MatOpsDotAddTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<std::tuple<bool, bool>> {};

TEST_P(MatOpsDotAddTest, Dot_Add_2x2_2x2) {
  bool row_major = std::get<0>(GetParam());
  bool add_lhs = std::get<1>(GetParam());
  Array2D<float> lhs({{1.0, 2.0}, {3.0, 4.0}});
  Array2D<float> rhs({{10.0, 11.0}, {12.0, 13.0}});

  auto minor_to_major = [](bool row_major) -> std::vector<int64> {
    return {row_major ? 1 : 0, row_major ? 0 : 1};
  };

  auto prim_type = primitive_util::NativeToPrimitiveType<float>();
  Shape lhs_shape =
      ShapeUtil::MakeShape(prim_type, {lhs.height(), lhs.width()});
  Shape rhs_shape =
      ShapeUtil::MakeShape(prim_type, {rhs.height(), rhs.width()});

  TF_ASSIGN_OR_ASSERT_OK(
      auto lhs_handle,
      client_->TransferToServer(
          *LiteralUtil::CreateR2FromArray2DWithLayout<float>(
              lhs, LayoutUtil::MakeLayout(minor_to_major(row_major)))));
  TF_ASSIGN_OR_ASSERT_OK(
      auto rhs_handle,
      client_->TransferToServer(
          *LiteralUtil::CreateR2FromArray2DWithLayout<float>(
              rhs, LayoutUtil::MakeLayout(minor_to_major(row_major)))));

  ComputationBuilder builder(client_, TestName());
  auto lhs_arg = builder.Parameter(0, lhs_shape, "lhs");
  auto rhs_arg = builder.Parameter(1, rhs_shape, "rhs");
  auto result = builder.Dot(lhs_arg, rhs_arg);
  Array2D<float> expected;
  if (add_lhs) {
    result = builder.Add(result, lhs_arg);
    expected = Array2D<float>({{35, 39}, {81, 89}});
  } else {
    result = builder.Add(result, rhs_arg);
    expected = Array2D<float>({{44, 48}, {90, 98}});
  }

  ComputeAndCompareR2<float>(&builder, expected,
                             {lhs_handle.get(), rhs_handle.get()},
                             ErrorSpec(1e-6));
}

INSTANTIATE_TEST_CASE_P(MatOpsDotAddTestInstances, MatOpsDotAddTest,
                        ::testing::Combine(::testing::Bool(),
                                           ::testing::Bool()));

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendDebugOptionsFlags(&flag_list);
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
