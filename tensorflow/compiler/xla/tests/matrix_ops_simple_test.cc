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
#include <tuple>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

#ifdef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
using TypesF16F32 = ::testing::Types<float>;
#else
using TypesF16F32 = ::testing::Types<Eigen::half, float>;
#endif

class MatOpsSimpleTest : public ClientLibraryTestBase {};

template <typename T>
class MatOpsSimpleTest_F16F32 : public MatOpsSimpleTest {};

TYPED_TEST_CASE(MatOpsSimpleTest_F16F32, TypesF16F32);

XLA_TYPED_TEST(MatOpsSimpleTest_F16F32, ExpTwoByTwoValues) {
  using T = TypeParam;
  XlaBuilder builder("exp_2x2");
  auto data = ConstantR2FromArray2D<T>(&builder, {
                                                     {1.0f, 0.0f},   // row 0
                                                     {-1.0f, 0.5f},  // row 1
                                                 });
  Exp(data);

  Literal expected =
      LiteralUtil::CreateR2FromArray2D<T>({{2.71828f, 1.00000f},    // row 0
                                           {0.36788f, 1.64872f}});  // row 1

  this->ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-5));
}

XLA_TYPED_TEST(MatOpsSimpleTest_F16F32, MapTwoByTwo) {
  using T = TypeParam;
  XlaComputation add_half;
  {
    // add_half(x) = x + 0.5
    XlaBuilder builder("add_half");
    auto x_value =
        Parameter(&builder, 0, ShapeUtil::MakeShapeWithType<T>({}), "x_value");
    auto half = ConstantR0<T>(&builder, static_cast<T>(0.5));
    Add(x_value, half);
    auto computation_status = builder.Build();
    ASSERT_IS_OK(computation_status.status());
    add_half = std::move(computation_status).value();
  }

  XlaBuilder builder("map_2x2");
  auto data = ConstantR2FromArray2D<T>(&builder, {
                                                     {1.0f, 0.0f},   // row 0
                                                     {-1.0f, 0.5f},  // row 1
                                                 });
  Map(&builder, {data}, add_half, {0, 1});

  Literal expected =
      LiteralUtil::CreateR2FromArray2D<T>({{1.5f, 0.5f},     // row 0
                                           {-0.5f, 1.0f}});  // row 1
  this->ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-5));
}

XLA_TYPED_TEST(MatOpsSimpleTest_F16F32, MaxTwoByTwoValues) {
  using T = TypeParam;
  XlaBuilder builder("max_2x2");
  auto lhs = ConstantR2FromArray2D<T>(&builder, {
                                                    {7.0f, 2.0f},   // row 0
                                                    {3.0f, -4.0f},  // row 1
                                                });
  auto rhs = ConstantR2FromArray2D<T>(&builder, {
                                                    {5.0f, 6.0f},   // row 0
                                                    {1.0f, -8.0f},  // row 1
                                                });
  Max(lhs, rhs);

  Literal expected =
      LiteralUtil::CreateR2FromArray2D<T>({{7.0f, 6.0f},     // row 0
                                           {3.0f, -4.0f}});  // row 1
  this->ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-6));
}

struct TestLinspaceMaxParam {
  int64_t rows;
  int64_t cols;
};

class TestLinspaceMaxParametric
    : public MatOpsSimpleTest,
      public ::testing::WithParamInterface<TestLinspaceMaxParam> {
 public:
  template <typename T>
  void TestImpl() {
    TestLinspaceMaxParam param = GetParam();
    int64_t rows = param.rows;
    int64_t cols = param.cols;
    float from = -128.0, to = 256.0;
    std::unique_ptr<Array2D<T>> alhs =
        MakeLinspaceArray2D<T>(from, to, rows, cols);
    auto arhs = std::make_unique<Array2D<T>>(rows, cols, static_cast<T>(1.0f));

    XlaBuilder builder(absl::StrFormat("max_%dx%d_linspace", rows, cols));
    auto lhs = ConstantR2FromArray2D<T>(&builder, *alhs);
    auto rhs = ConstantR2FromArray2D<T>(&builder, *arhs);
    Max(lhs, rhs);

    Array2D<T> expected(rows, cols);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        expected(row, col) = std::max<T>((*alhs)(row, col), (*arhs)(row, col));
      }
    }
    ErrorSpec error_spec(1e-6);
    if (std::is_same<Eigen::half, T>::value) {
      error_spec = ErrorSpec(1e-6, 2e-4);
    }
    ComputeAndCompareR2<T>(&builder, expected, {}, error_spec);
  }
};

std::string PrintTestLinspaceMaxParam(
    const ::testing::TestParamInfo<TestLinspaceMaxParam>& test_param) {
  const TestLinspaceMaxParam& param = test_param.param;
  return absl::StrCat(param.rows, "r", param.cols, "c");
}

#ifndef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
XLA_TEST_P(TestLinspaceMaxParametric, TestF16) { TestImpl<Eigen::half>(); }
#endif
XLA_TEST_P(TestLinspaceMaxParametric, TestF32) { TestImpl<float>(); }

INSTANTIATE_TEST_CASE_P(
    TestLinspaceMax, TestLinspaceMaxParametric,
    ::testing::Values(TestLinspaceMaxParam{1, 1}, TestLinspaceMaxParam{2, 2},
                      TestLinspaceMaxParam{3, 3}, TestLinspaceMaxParam{4, 4},
                      TestLinspaceMaxParam{6, 6}, TestLinspaceMaxParam{8, 8},
                      TestLinspaceMaxParam{12, 12},
                      TestLinspaceMaxParam{16, 16}, TestLinspaceMaxParam{32, 8},
                      TestLinspaceMaxParam{64, 8}),
    PrintTestLinspaceMaxParam);

class MatOpsDotAddTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<std::tuple<bool, bool, bool, bool>> {
 public:
  template <typename T>
  void TestImpl() {
    bool row_major = std::get<0>(GetParam());
    bool add_lhs = std::get<1>(GetParam());
    bool transpose = std::get<2>(GetParam());
    bool use_cublaslt = std::get<3>(GetParam());
#ifndef GOOGLE_CUDA
    use_cublaslt = false;
#endif
    execution_options_.mutable_debug_options()->set_xla_gpu_enable_cublaslt(
        use_cublaslt);
    Array2D<T> lhs({{1.0f, 2.0f}, {3.0f, 4.0f}});
    Array2D<T> rhs({{10.0f, 11.0f}, {12.0f, 13.0f}});

    auto minor_to_major = [](bool row_major) -> std::vector<int64_t> {
      return {row_major ? 1 : 0, row_major ? 0 : 1};
    };

    auto prim_type = primitive_util::NativeToPrimitiveType<T>();
    Shape lhs_shape =
        ShapeUtil::MakeShape(prim_type, {lhs.height(), lhs.width()});
    Shape rhs_shape =
        ShapeUtil::MakeShape(prim_type, {rhs.height(), rhs.width()});

    TF_ASSERT_OK_AND_ASSIGN(
        auto lhs_handle,
        client_->TransferToServer(LiteralUtil::CreateR2FromArray2DWithLayout<T>(
            lhs, LayoutUtil::MakeLayout(minor_to_major(row_major)))));
    TF_ASSERT_OK_AND_ASSIGN(
        auto rhs_handle,
        client_->TransferToServer(LiteralUtil::CreateR2FromArray2DWithLayout<T>(
            rhs, LayoutUtil::MakeLayout(minor_to_major(row_major)))));

    XlaBuilder builder(TestName());
    auto lhs_arg = Parameter(&builder, 0, lhs_shape, "lhs");
    auto lhs_mat_arg = lhs_arg;
    if (transpose) {
      lhs_mat_arg = Transpose(lhs_mat_arg, {1, 0});
    }
    auto rhs_arg = Parameter(&builder, 1, rhs_shape, "rhs");
    auto result = Dot(lhs_mat_arg, rhs_arg);
    Array2D<T> expected;
    if (add_lhs) {
      result = Add(result, lhs_arg);
      if (transpose) {
        expected = Array2D<T>({{47.0f, 52.0f}, {71.0f, 78.0f}});
      } else {
        expected = Array2D<T>({{35.0f, 39.0f}, {81.0f, 89.0f}});
      }
    } else {
      result = Add(result, rhs_arg);
      if (transpose) {
        expected = Array2D<T>({{56.0f, 61.0f}, {80.0f, 87.0f}});
      } else {
        expected = Array2D<T>({{44.0f, 48.0f}, {90.0f, 98.0f}});
      }
    }

    ComputeAndCompareR2<T>(&builder, expected,
                           {lhs_handle.get(), rhs_handle.get()},
                           ErrorSpec(1e-6));
  }

  template <typename T>
  void iota_int_init_value(std::vector<T>& values, int init_value) {
    absl::c_for_each(values,
                     [&](T& value) { value = static_cast<T>(init_value++); });
  }

  template <typename T>
  XlaOp generate_dot(XlaBuilder& builder, Array2D<T>& lhs, Array2D<T>& rhs,
                     bool transpose) {
    auto prim_type = primitive_util::NativeToPrimitiveType<T>();
    Shape lhs_shape =
        ShapeUtil::MakeShape(prim_type, {lhs.height(), lhs.width()});
    Shape rhs_shape =
        ShapeUtil::MakeShape(prim_type, {rhs.height(), rhs.width()});

    auto lhs_arg = Parameter(&builder, 0, lhs_shape, "lhs");
    auto lhs_mat_arg = lhs_arg;
    if (transpose) {
      lhs_mat_arg = Transpose(lhs_mat_arg, {1, 0});
    }
    auto rhs_arg = Parameter(&builder, 1, rhs_shape, "rhs");
    auto dot = Dot(lhs_mat_arg, rhs_arg);
    return dot;
  }

  template <typename T>
  XlaOp generate_dot_add(XlaBuilder& builder, Array2D<T>& lhs, Array2D<T>& rhs,
                         bool transpose, int n) {
    auto dot = generate_dot(builder, lhs, rhs, transpose);
    auto prim_type = primitive_util::NativeToPrimitiveType<T>();
    Shape bias_shape = ShapeUtil::MakeShape(prim_type, {n});
    auto bias_arg = Parameter(&builder, 2, bias_shape, "bias");
    auto bias_bcast = BroadcastInDim(bias_arg, {n, n}, {1});
    auto add_result = Add(dot, bias_bcast);
    return add_result;
  }

  template <typename T>
  void TestImplBiasAddEpilogueFusion() {
    bool row_major = std::get<0>(GetParam());
    bool transpose = std::get<2>(GetParam());
    bool use_cublaslt = std::get<3>(GetParam());
#ifndef GOOGLE_CUDA
    use_cublaslt = false;
#endif
    execution_options_.mutable_debug_options()->set_xla_gpu_enable_cublaslt(
        use_cublaslt);
    Array2D<T> lhs({{1.0f, 2.0f}, {3.0f, 4.0f}});
    Array2D<T> rhs({{10.0f, 11.0f}, {12.0f, 13.0f}});
    auto minor_to_major = [](bool row_major) -> std::vector<int64_t> {
      return {row_major ? 1 : 0, row_major ? 0 : 1};
    };

    XlaBuilder builder(TestName());
    auto result = generate_dot_add(builder, lhs, rhs, transpose, 2);
    (void)result;
    TF_ASSERT_OK_AND_ASSIGN(
        auto lhs_handle,
        client_->TransferToServer(LiteralUtil::CreateR2FromArray2DWithLayout<T>(
            lhs, LayoutUtil::MakeLayout(minor_to_major(row_major)))));
    TF_ASSERT_OK_AND_ASSIGN(
        auto rhs_handle,
        client_->TransferToServer(LiteralUtil::CreateR2FromArray2DWithLayout<T>(
            rhs, LayoutUtil::MakeLayout(minor_to_major(row_major)))));

    auto prim_type = primitive_util::NativeToPrimitiveType<T>();
    Shape bias_shape = ShapeUtil::MakeShape(prim_type, {2});
    std::vector<T> bias_elems(ShapeUtil::ElementsIn(bias_shape));
    iota_int_init_value(bias_elems, 1);
    auto bias_literal = LiteralUtil::CreateR1<T>(bias_elems);
    TF_ASSERT_OK_AND_ASSIGN(auto bias_handle,
                            client_->TransferToServer(bias_literal));
    Array2D<T> expected;

    if (transpose) {
      expected = Array2D<T>({{47.0f, 52.0f}, {69.0f, 76.0f}});
    } else {
      expected = Array2D<T>({{35.0f, 39.0f}, {79.0f, 87.0f}});
    }

    ComputeAndCompareR2<T>(
        &builder, expected,
        {lhs_handle.get(), rhs_handle.get(), bias_handle.get()},
        ErrorSpec(1e-6));
  }

  template <typename T>
  void TestImplReluActivationEpilogueFusion() {
    bool row_major = std::get<0>(GetParam());
    bool transpose = std::get<2>(GetParam());
    bool use_cublaslt = std::get<3>(GetParam());
#if GOOGLE_CUDA
#else
    use_cublaslt = false;
#endif
    execution_options_.mutable_debug_options()->set_xla_gpu_enable_cublaslt(
        use_cublaslt);
    Array2D<T> lhs({{-1.0f, 2.0f}, {3.0f, 4.0f}});
    Array2D<T> rhs({{10.0f, 11.0f}, {-12.0f, 13.0f}});
    auto minor_to_major = [](bool row_major) -> std::vector<int64_t> {
      return {row_major ? 1 : 0, row_major ? 0 : 1};
    };

    XlaBuilder builder(TestName());
    auto result = generate_dot(builder, lhs, rhs, transpose);
    auto constant_zero = ConstantR0<T>(&builder, T(0.0f));
    auto constant_zero_bcast = BroadcastInDim(constant_zero, {2, 2}, {});
    result = Max(result, constant_zero_bcast);

    TF_ASSERT_OK_AND_ASSIGN(
        auto lhs_handle,
        client_->TransferToServer(LiteralUtil::CreateR2FromArray2DWithLayout<T>(
            lhs, LayoutUtil::MakeLayout(minor_to_major(row_major)))));
    TF_ASSERT_OK_AND_ASSIGN(
        auto rhs_handle,
        client_->TransferToServer(LiteralUtil::CreateR2FromArray2DWithLayout<T>(
            rhs, LayoutUtil::MakeLayout(minor_to_major(row_major)))));

    Array2D<T> expected;

    if (transpose) {
      expected = Array2D<T>({{0.0f, 28.0f}, {0.0f, 74.0f}});
    } else {
      expected = Array2D<T>({{0.0f, 15.0f}, {0.0f, 85.0f}});
    }

    ComputeAndCompareR2<T>(&builder, expected,
                           {lhs_handle.get(), rhs_handle.get()},
                           ErrorSpec(1e-6));
  }

  template <typename T>
  void TestImplBiasAddReluActivationEpilogueFusion() {
    bool row_major = std::get<0>(GetParam());
    bool transpose = std::get<2>(GetParam());
    bool use_cublaslt = std::get<3>(GetParam());
#ifndef GOOGLE_CUDA
    use_cublaslt = false;
#endif
    execution_options_.mutable_debug_options()->set_xla_gpu_enable_cublaslt(
        use_cublaslt);
    Array2D<T> lhs({{-1.0f, 2.0f}, {3.0f, 4.0f}});
    Array2D<T> rhs({{10.0f, 11.0f}, {-12.0f, 13.0f}});
    auto minor_to_major = [](bool row_major) -> std::vector<int64_t> {
      return {row_major ? 1 : 0, row_major ? 0 : 1};
    };

    XlaBuilder builder(TestName());
    auto result = generate_dot_add(builder, lhs, rhs, transpose, 2);
    auto constant = ConstantR0<T>(&builder, T(0.0f));
    auto constant_bcast = BroadcastInDim(constant, {2, 2}, {});
    result = Max(result, constant_bcast);

    TF_ASSERT_OK_AND_ASSIGN(
        auto lhs_handle,
        client_->TransferToServer(LiteralUtil::CreateR2FromArray2DWithLayout<T>(
            lhs, LayoutUtil::MakeLayout(minor_to_major(row_major)))));
    TF_ASSERT_OK_AND_ASSIGN(
        auto rhs_handle,
        client_->TransferToServer(LiteralUtil::CreateR2FromArray2DWithLayout<T>(
            rhs, LayoutUtil::MakeLayout(minor_to_major(row_major)))));

    auto prim_type = primitive_util::NativeToPrimitiveType<T>();
    Shape bias_shape = ShapeUtil::MakeShape(prim_type, {2});
    std::vector<T> bias_elems(ShapeUtil::ElementsIn(bias_shape));
    iota_int_init_value(bias_elems, 1);
    auto bias_literal = LiteralUtil::CreateR1<T>(bias_elems);
    TF_ASSERT_OK_AND_ASSIGN(auto bias_handle,
                            client_->TransferToServer(bias_literal));
    Array2D<T> expected;

    if (transpose) {
      expected = Array2D<T>({{0.0f, 30.0f}, {0.0f, 76.0f}});
    } else {
      expected = Array2D<T>({{0.0f, 17.0f}, {0.0f, 87.0f}});
    }

    ComputeAndCompareR2<T>(
        &builder, expected,
        {lhs_handle.get(), rhs_handle.get(), bias_handle.get()},
        ErrorSpec(1e-6));
  }
};

XLA_TEST_P(MatOpsDotAddTest, Dot_Add_2x2_2x2BF16) { TestImpl<bfloat16>(); }
#ifndef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
XLA_TEST_P(MatOpsDotAddTest, Dot_Add_2x2_2x2F16) { TestImpl<Eigen::half>(); }
XLA_TEST_P(MatOpsDotAddTest, Dot_BiasAdd_2x2_2x2F16) {
  TestImplBiasAddEpilogueFusion<Eigen::half>();
}
XLA_TEST_P(MatOpsDotAddTest, Dot_ReluActivation_2x2_2x2F16) {
  TestImplReluActivationEpilogueFusion<Eigen::half>();
}
XLA_TEST_P(MatOpsDotAddTest, Dot_BiasAddReluActivation_2x2_2x2F16) {
  TestImplBiasAddReluActivationEpilogueFusion<Eigen::half>();
}
#endif

XLA_TEST_P(MatOpsDotAddTest, Dot_Add_2x2_2x2F32) { TestImpl<float>(); }
XLA_TEST_P(MatOpsDotAddTest, Dot_BiasAdd_2x2_2x2F32) {
  TestImplBiasAddEpilogueFusion<float>();
}
XLA_TEST_P(MatOpsDotAddTest, Dot_ReluActivation_2x2_2x2F32) {
  TestImplReluActivationEpilogueFusion<float>();
}
XLA_TEST_P(MatOpsDotAddTest, Dot_BiasAddReluActivation_2x2_2x2F32) {
  TestImplBiasAddReluActivationEpilogueFusion<float>();
}

INSTANTIATE_TEST_CASE_P(MatOpsDotAddTestInstances, MatOpsDotAddTest,
                        ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                                           ::testing::Bool(),
                                           ::testing::Bool()));

}  // namespace
}  // namespace xla
