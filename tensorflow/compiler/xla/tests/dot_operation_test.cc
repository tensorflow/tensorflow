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

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_runtime_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/layout_util_flags.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace {

// TODO(b/34468543): use GUnit typed tests when we can do all tests on all
// backends.
class DotOperationTest : public ClientLibraryTestBase {
 public:
  ErrorSpec error_spec_{0.0001, 1e-5};

 protected:
  template <typename Element>
  void TestOneElementVectorDot();
  template <typename Element>
  void TestVectorDot();
  template <typename Element>
  void TestSquareMatrixDot(bool lhs_row_major = false,
                           bool rhs_row_major = false);
  template <typename Element>
  void TestNonsquareMatrixDot(bool lhs_row_major = false,
                              bool rhs_row_major = false);
  void TestMatrixDot(int M, int K, int N, bool lhs_row_major = false,
                     bool rhs_row_major = false);
};

XLA_TEST_F(DotOperationTest, ZeroElementVectorDotF32) {
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<float>({});
  auto rhs = builder.ConstantR1<float>({});
  auto result = builder.Dot(lhs, rhs);

  ComputeAndCompareR0<float>(&builder, 0.0, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, TrivialMatrixVectorDotF32) {
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR2<float>({{3.0, 4.0}});
  auto rhs = builder.ConstantR1<float>({3.0, 4.0});
  auto result = builder.Dot(lhs, rhs);

  ComputeAndCompareR1<float>(&builder, {25.0}, {}, error_spec_);
}

template <typename Element>
void DotOperationTest::TestOneElementVectorDot() {
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<Element>({2.0});
  auto rhs = builder.ConstantR1<Element>({3.0});
  auto result = builder.Dot(lhs, rhs);

  ComputeAndCompareR0<Element>(&builder, 6.0, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, OneElementVectorDotF32) {
  TestOneElementVectorDot<float>();
}

XLA_TEST_F(DotOperationTest, OneElementVectorDotF64) {
  TestOneElementVectorDot<double>();
}

template <typename Element>
void DotOperationTest::TestVectorDot() {
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR1<Element>({1.0, 2.5, 42.0});
  auto rhs = builder.ConstantR1<Element>({11.0, -1.0, 0.5});
  auto result = builder.Dot(lhs, rhs);

  ComputeAndCompareR0<Element>(&builder, 29.5, {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, VectorDotF32) { TestVectorDot<float>(); }

XLA_TEST_F(DotOperationTest, VectorDotF64) { TestVectorDot<double>(); }

namespace {

std::vector<int64> MinorToMajorForIsRowMajor(bool row_major) {
  return {row_major ? 1 : 0, row_major ? 0 : 1};
}

}  // namespace

XLA_TEST_F(DotOperationTest, Dot_0x2_2x0) {
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR2FromArray2D<float>(Array2D<float>(0, 2));
  auto rhs = builder.ConstantR2FromArray2D<float>(Array2D<float>(2, 0));
  auto result = builder.Dot(lhs, rhs);

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 0), {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, Dot_0x2_2x3) {
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR2FromArray2D<float>(Array2D<float>(0, 2));
  auto rhs = builder.ConstantR2<float>({{7.0, 8.0, 9.0}, {42.0, 77.0, 101.0}});
  auto result = builder.Dot(lhs, rhs);

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 3), {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, Dot_3x2_2x0) {
  ComputationBuilder builder(client_, TestName());
  auto lhs =
      builder.ConstantR2<float>({{7.0, 8.0}, {9.0, 42.0}, {77.0, 101.0}});
  auto rhs = builder.ConstantR2FromArray2D<float>(Array2D<float>(2, 0));
  auto result = builder.Dot(lhs, rhs);

  ComputeAndCompareR2<float>(&builder, Array2D<float>(3, 0), {}, error_spec_);
}

XLA_TEST_F(DotOperationTest, Dot_2x0_0x2) {
  ComputationBuilder builder(client_, TestName());
  auto lhs = builder.ConstantR2FromArray2D<float>(Array2D<float>(2, 0));
  auto rhs = builder.ConstantR2FromArray2D<float>(Array2D<float>(0, 2));
  auto result = builder.Dot(lhs, rhs);

  ComputeAndCompareR2<float>(&builder, Array2D<float>(2, 2, 0.0f), {},
                             error_spec_);
}

template <typename Element>
void DotOperationTest::TestSquareMatrixDot(bool lhs_row_major,
                                           bool rhs_row_major) {
  auto lhs_handle =
      client_
          ->TransferToServer(*test_utils::CreateR2LiteralWithLayout<Element>(
              {{1.0, 2.0}, {3.0, -4.0}},
              MinorToMajorForIsRowMajor(lhs_row_major)))
          .ConsumeValueOrDie();
  auto rhs_handle =
      client_
          ->TransferToServer(*test_utils::CreateR2LiteralWithLayout<Element>(
              {{1.0, 6.0}, {7.0, -4.0}},
              MinorToMajorForIsRowMajor(rhs_row_major)))
          .ConsumeValueOrDie();

  ComputationBuilder builder(client_, TestName());
  auto prim_type = primitive_util::NativeToPrimitiveType<Element>();
  auto result = builder.Dot(
      builder.Parameter(0, ShapeUtil::MakeShape(prim_type, {2, 2}), "lhs"),
      builder.Parameter(1, ShapeUtil::MakeShape(prim_type, {2, 2}), "rhs"));

  Array2D<Element> expected({{15.0, -2.0}, {-25.0, 34.0}});
  ComputeAndCompareR2<Element>(
      &builder, expected, {lhs_handle.get(), rhs_handle.get()}, error_spec_);
}

void DotOperationTest::TestMatrixDot(int M, int K, int N, bool lhs_row_major,
                                     bool rhs_row_major) {
  std::unique_ptr<Array2D<float>> lhs_data =
      MakeLinspaceArray2D(0.0, 1.0, M, K);
  std::unique_ptr<Literal> lhs_lit = LiteralUtil::CreateR2FromArray2DWithLayout(
      *lhs_data,
      LayoutUtil::MakeLayout(MinorToMajorForIsRowMajor(lhs_row_major)));
  auto lhs_handle = client_->TransferToServer(*lhs_lit).ConsumeValueOrDie();

  std::unique_ptr<Array2D<float>> rhs_data =
      MakeLinspaceArray2D(0.0, 1.0, K, N);
  std::unique_ptr<Literal> rhs_lit = LiteralUtil::CreateR2FromArray2DWithLayout(
      *rhs_data,
      LayoutUtil::MakeLayout(MinorToMajorForIsRowMajor(rhs_row_major)));
  auto rhs_handle = client_->TransferToServer(*rhs_lit).ConsumeValueOrDie();

  ComputationBuilder builder(client_, TestName());
  auto prim_type = primitive_util::NativeToPrimitiveType<float>();
  auto result = builder.Dot(
      builder.Parameter(0, ShapeUtil::MakeShape(prim_type, {M, K}), "lhs"),
      builder.Parameter(1, ShapeUtil::MakeShape(prim_type, {K, N}), "rhs"));

  std::unique_ptr<Array2D<float>> expected =
      ReferenceUtil::MatmulArray2D(*lhs_data, *rhs_data);

  ComputeAndCompareR2<float>(&builder, *expected,
                             {lhs_handle.get(), rhs_handle.get()},
                             ErrorSpec(0.3, 3e-3));
}

XLA_TEST_F(DotOperationTest, MatrixDotF32_12_117_7_MinorToMajorTF) {
  TestMatrixDot(12, 117, 7, true, false);
}

XLA_TEST_F(DotOperationTest, MatrixDotF32_12_117_7_MinorToMajorFT) {
  TestMatrixDot(12, 117, 7, false, true);
}

XLA_TEST_F(DotOperationTest, MatrixDotF32_12_117_7_MinorToMajorTT) {
  TestMatrixDot(12, 117, 7, true, true);
}

XLA_TEST_F(DotOperationTest, MatrixDotF32_12_117_7_MinorToMajorFF) {
  TestMatrixDot(12, 117, 7, false, false);
}

XLA_TEST_F(DotOperationTest, MatrixDotF32_270_270_520_MinorToMajorTT) {
  TestMatrixDot(270, 270, 520, true, true);
}

XLA_TEST_F(DotOperationTest, MatrixDotF32_270_270_520_MinorToMajorTF) {
  TestMatrixDot(270, 270, 520, true, false);
}

XLA_TEST_F(DotOperationTest, MatrixDotF32_270_270_520_MinorToMajorFT) {
  TestMatrixDot(270, 270, 520, false, true);
}

XLA_TEST_F(DotOperationTest, MatrixDotF32_270_270_520_MinorToMajorFF) {
  TestMatrixDot(270, 270, 520, false, false);
}

XLA_TEST_F(DotOperationTest, MatrixDotF32_260_3_520_MinorToMajorTT) {
  TestMatrixDot(269, 3, 520, true, true);
}

XLA_TEST_F(DotOperationTest, MatrixDotF32_260_3_520_MinorToMajorTF) {
  TestMatrixDot(260, 3, 520, true, false);
}

XLA_TEST_F(DotOperationTest, MatrixDotF32_260_3_520_MinorToMajorFT) {
  TestMatrixDot(260, 3, 520, false, true);
}

XLA_TEST_F(DotOperationTest, MatrixDotF32_260_3_520_MinorToMajorFF) {
  TestMatrixDot(260, 3, 520, false, false);
}

XLA_TEST_F(DotOperationTest, SquareMatrixDotF32MinorToMajorFF) {
  constexpr bool kLhsRowMajor = false;
  constexpr bool kRhsRowMajor = false;
  TestSquareMatrixDot<float>(kLhsRowMajor, kRhsRowMajor);
}

XLA_TEST_F(DotOperationTest, SquareMatrixDotF32MinorToMajorFT) {
  TestSquareMatrixDot<float>(false, true);
}

XLA_TEST_F(DotOperationTest, SquareMatrixDotF32MinorToMajorTF) {
  TestSquareMatrixDot<float>(true, false);
}

TEST_F(DotOperationTest, SquareMatrixDotF32MinorToMajorTT) {
  constexpr bool kLhsRowMajor = true;
  constexpr bool kRhsRowMajor = true;
  TestSquareMatrixDot<float>(kLhsRowMajor, kRhsRowMajor);
}

XLA_TEST_F(DotOperationTest, SquareMatrixDotF64) {
  TestSquareMatrixDot<double>();
}

template <typename Element>
void DotOperationTest::TestNonsquareMatrixDot(bool lhs_row_major,
                                              bool rhs_row_major) {
  auto lhs_handle =
      client_
          ->TransferToServer(*test_utils::CreateR2LiteralWithLayout<Element>(
              {{1.0, 2.0, 3.0}, {3.0, -4.0, -1.0}},
              MinorToMajorForIsRowMajor(lhs_row_major)))
          .ConsumeValueOrDie();
  auto rhs_handle =
      client_
          ->TransferToServer(*test_utils::CreateR2LiteralWithLayout<Element>(
              {{1.0, 6.0}, {2.0, 3.0}, {7.0, -4.0}},
              MinorToMajorForIsRowMajor(rhs_row_major)))
          .ConsumeValueOrDie();

  ComputationBuilder builder(client_, TestName());
  auto prim_type = primitive_util::NativeToPrimitiveType<Element>();
  auto result = builder.Dot(
      builder.Parameter(0, ShapeUtil::MakeShape(prim_type, {2, 3}), "lhs"),
      builder.Parameter(1, ShapeUtil::MakeShape(prim_type, {3, 2}), "rhs"));

  Array2D<Element> expected({{26.0, 0.0}, {-12.0, 10.0}});

  ComputeAndCompareR2<Element>(
      &builder, expected, {lhs_handle.get(), rhs_handle.get()}, error_spec_);
}

XLA_TEST_F(DotOperationTest, NonsquareMatrixDotF32MajorToMinorFF) {
  constexpr bool kLhsRowMajor = false;
  constexpr bool kRhsRowMajor = false;
  TestNonsquareMatrixDot<float>(kLhsRowMajor, kRhsRowMajor);
}

XLA_TEST_F(DotOperationTest, NonsquareMatrixDotF32MajorToMinorFT) {
  constexpr bool kLhsRowMajor = false;
  constexpr bool kRhsRowMajor = true;
  TestNonsquareMatrixDot<float>(kLhsRowMajor, kRhsRowMajor);
}

XLA_TEST_F(DotOperationTest, NonsquareMatrixDotF32MajorToMinorTF) {
  constexpr bool kLhsRowMajor = true;
  constexpr bool kRhsRowMajor = false;
  TestNonsquareMatrixDot<float>(kLhsRowMajor, kRhsRowMajor);
}

TEST_F(DotOperationTest, NonsquareMatrixDotF32MajorToMinorTT) {
  constexpr bool kLhsRowMajor = true;
  constexpr bool kRhsRowMajor = true;
  TestNonsquareMatrixDot<float>(kLhsRowMajor, kRhsRowMajor);
}

XLA_TEST_F(DotOperationTest, NonsquareMatrixDotF64) {
  TestNonsquareMatrixDot<double>();
}

TEST_F(DotOperationTest, ConcurrentMatMul) {
  ComputationBuilder builder(client_, TestName());
  auto matrix1 = builder.ConstantR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto matrix2 = builder.ConstantR2<float>({{5.0, 6.0}, {7.0, 8.0}});
  auto matrix12 = builder.Dot(matrix1, matrix2);
  auto matrix21 = builder.Dot(matrix2, matrix1);
  builder.Add(matrix12, matrix21);

  Array2D<float> expected({{42.0, 56.0}, {74.0, 96.0}});
  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

// Regression test for b/32055648. The root of the graph is a kFusion of 4
// bitcasts. Although bitcasts don't map to thunks, the root should still be
// sync-dependent on bitcasts' operands.
XLA_TEST_F(DotOperationTest, BatchMatMul) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {2, 2, 2, 2}), "x");
  auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {2, 2, 2, 2}), "y");

  auto x_flat = builder.Reshape(x, {0, 1, 2, 3}, {4, 2, 2});
  auto y_flat = builder.Reshape(y, {0, 1, 2, 3}, {4, 2, 2});

  // Slice batches into individual matrices and multiply them.
  std::vector<xla::ComputationDataHandle> out_slices;
  for (int i = 0; i < 4; ++i) {
    // Slice off individual matrices and reshape to 2D tensors.
    auto x_slice = builder.Slice(x_flat, {i, 0, 0}, {i + 1, 2, 2});
    x_slice = builder.Reshape(x_slice, {0, 1, 2}, {2, 2});
    auto y_slice = builder.Slice(y_flat, {i, 0, 0}, {i + 1, 2, 2});
    y_slice = builder.Reshape(y_slice, {0, 1, 2}, {2, 2});

    auto out = builder.Dot(x_slice, y_slice);
    out = builder.Reshape(out, {0, 1}, {1, 2, 2});
    out_slices.push_back(out);
  }
  auto out_flat = builder.ConcatInDim(out_slices, 0);
  builder.Reshape(out_flat, {0, 1, 2}, {2, 2, 2, 2});

  auto x_data = client_
                    ->TransferToServer(*LiteralUtil::CreateR4<float>(
                        {{{{1000, 100}, {10, 1}}, {{2000, 200}, {20, 2}}},
                         {{{3000, 300}, {30, 3}}, {{4000, 400}, {40, 4}}}}))
                    .ConsumeValueOrDie();
  auto y_data = client_
                    ->TransferToServer(*LiteralUtil::CreateR4<float>(
                        {{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}},
                         {{{11, 22}, {33, 44}}, {{55, 66}, {77, 88}}}}))
                    .ConsumeValueOrDie();

  ComputeAndCompareR4<float>(
      &builder,
      /*expected=*/{{{{1300, 2400}, {13, 24}}, {{11400, 13600}, {114, 136}}},
                    {{{42900, 79200}, {429, 792}},
                     {{250800, 299200}, {2508, 2992}}}},
      {x_data.get(), y_data.get()}, error_spec_);
}

TEST_F(DotOperationTest, TransposeFolding) {
  for (bool transpose_lhs : {false, true}) {
    for (bool transpose_rhs : {false, true}) {
      for (bool row_major : {false, true}) {
        std::unique_ptr<Array2D<float>> lhs(
            new Array2D<float>({{1.0, 2.0, 3.0}, {3.0, -4.0, -1.0}}));
        std::unique_ptr<Array2D<float>> rhs(
            new Array2D<float>({{1.0, 6.0}, {2.0, 3.0}, {7.0, -4.0}}));

        if (transpose_lhs) {
          lhs = ReferenceUtil::TransposeArray2D(*lhs);
        }
        if (transpose_rhs) {
          rhs = ReferenceUtil::TransposeArray2D(*rhs);
        }
        auto lhs_handle =
            client_
                ->TransferToServer(
                    *LiteralUtil::CreateR2FromArray2DWithLayout<float>(
                        *lhs, LayoutUtil::MakeLayout(
                                  MinorToMajorForIsRowMajor(row_major))))
                .ConsumeValueOrDie();
        auto rhs_handle =
            client_
                ->TransferToServer(
                    *LiteralUtil::CreateR2FromArray2DWithLayout<float>(
                        *rhs, LayoutUtil::MakeLayout(
                                  MinorToMajorForIsRowMajor(row_major))))
                .ConsumeValueOrDie();

        ComputationBuilder builder(client_, TestName());
        auto prim_type = primitive_util::NativeToPrimitiveType<float>();
        auto lhs_arg = builder.Parameter(
            0, ShapeUtil::MakeShape(prim_type, {lhs->height(), lhs->width()}),
            "lhs");
        auto rhs_arg = builder.Parameter(
            1, ShapeUtil::MakeShape(prim_type, {rhs->height(), rhs->width()}),
            "rhs");
        if (transpose_lhs) {
          lhs_arg = builder.Transpose(lhs_arg, {1, 0});
        }
        if (transpose_rhs) {
          rhs_arg = builder.Transpose(rhs_arg, {1, 0});
        }
        auto result = builder.Dot(lhs_arg, rhs_arg);

        Array2D<float> expected({{26.0, 0.0}, {-12.0, 10.0}});
        VLOG(1) << "TestTransposeFolding " << transpose_lhs << " "
                << transpose_rhs << " " << row_major;
        ComputeAndCompareR2<float>(&builder, expected,
                                   {lhs_handle.get(), rhs_handle.get()},
                                   error_spec_);
      }
    }
  }
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendLayoutUtilFlags(&flag_list);
  xla::legacy_flags::AppendCpuRuntimeFlags(&flag_list);
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
