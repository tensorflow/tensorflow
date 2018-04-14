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

class DotOperationTest : public ClientLibraryTestBase {
 public:
  ErrorSpec error_spec_{0.0001, 1e-5};
};

#if defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16) && \
    defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)
using TypesF16F32 = ::testing::Types<float>;
using TypesF16F32F64 = ::testing::Types<float>;
using TypesF16F32F64CF64 = ::testing::Types<float>;
#elif !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16) && \
    !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)
using TypesF16F32 = ::testing::Types<Eigen::half, float>;
using TypesF16F32F64 = ::testing::Types<Eigen::half, float, double>;
using TypesF16F32F64CF64 =
    ::testing::Types<Eigen::half, float, double, complex64>;
#elif !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16) && \
    defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64) && \
    defined(XLA_BACKEND_DOES_NOT_SUPPORT_COMPLEX)
using TypesF16F32 = ::testing::Types<Eigen::half, float>;
using TypesF16F32F64 = ::testing::Types<Eigen::half, float>;
using TypesF16F32F64CF64 =
    ::testing::Types<Eigen::half, float>;
#else
#error "Situation not handled yet"
#endif

// Check that we can safely pass an input tuple's elements to a dot operation.
TEST_F(DotOperationTest, DotOfInputTupleElem) {
  ComputationBuilder builder(client_, TestName());

  ComputationDataHandle param;
  auto param_data = CreateParameterAndTransferLiteral(
      0,
      *Literal::MakeTuple({Literal::CreateR2<float>({{1, 2}, {3, 4}}).get(),
                           Literal::CreateR2<float>({{5, 6}, {7, 8}}).get()}),
      "arg0", &builder, &param);
  auto lhs = builder.GetTupleElement(param, 0);
  auto rhs = builder.GetTupleElement(param, 1);
  builder.Dot(lhs, rhs);

  ComputeAndCompareLiteral(&builder,
                           *Literal::CreateR2<float>({{19, 22}, {43, 50}}),
                           {param_data.get()});
}

template <typename T>
class DotOperationTest_F16F32F64CF64 : public DotOperationTest {};
TYPED_TEST_CASE(DotOperationTest_F16F32F64CF64, TypesF16F32F64CF64);

XLA_TYPED_TEST(DotOperationTest_F16F32F64CF64, ZeroElementVectorDot) {
  using T = TypeParam;
  ComputationBuilder builder(this->client_, this->TestName());

  auto lhs = builder.ConstantR1<T>({});
  auto rhs = builder.ConstantR1<T>({});
  auto result = builder.Dot(lhs, rhs);

  this->template ComputeAndCompareR0<T>(&builder, static_cast<T>(0.0), {},
                                        this->error_spec_);
}

template <typename T>
class DotOperationTest_F16F32F64 : public DotOperationTest {};
TYPED_TEST_CASE(DotOperationTest_F16F32F64, TypesF16F32F64);

XLA_TYPED_TEST(DotOperationTest_F16F32F64, TrivialMatrixVectorDot) {
  using T = TypeParam;
  ComputationBuilder builder(this->client_, this->TestName());
  auto lhs = builder.ConstantR2FromArray2D<T>({{3.0f, 4.0f}});
  auto rhs = builder.ConstantFromArray<T>({3.0f, 4.0f});
  auto result = builder.Dot(lhs, rhs);

  this->template ComputeAndCompareR1<T>(&builder, {static_cast<T>(25.0f)}, {},
                                        this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64, OneElementVectorDot) {
  using T = TypeParam;
  ComputationBuilder builder(this->client_, this->TestName());
  auto lhs = builder.ConstantR1<T>({static_cast<T>(2.0f)});
  auto rhs = builder.ConstantR1<T>({static_cast<T>(3.0f)});
  auto result = builder.Dot(lhs, rhs);

  this->template ComputeAndCompareR0<T>(&builder, static_cast<T>(6.0f), {},
                                        this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64, VectorDot) {
  using T = TypeParam;
  ComputationBuilder builder(this->client_, this->TestName());
  auto lhs = builder.ConstantFromArray<T>({1.0f, 2.5f, 42.0f});
  auto rhs = builder.ConstantFromArray<T>({11.0f, -1.0f, 0.5f});
  auto result = builder.Dot(lhs, rhs);

  this->template ComputeAndCompareR0<T>(&builder, static_cast<T>(29.5f), {},
                                        this->error_spec_);
}

std::vector<int64> MinorToMajorForIsRowMajor(bool row_major) {
  return {row_major ? 1 : 0, row_major ? 0 : 1};
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64, Dot_0x2_2x0) {
  using T = TypeParam;
  ComputationBuilder builder(this->client_, this->TestName());
  auto lhs = builder.ConstantR2FromArray2D<T>(Array2D<T>(0, 2));
  auto rhs = builder.ConstantR2FromArray2D<T>(Array2D<T>(2, 0));
  auto result = builder.Dot(lhs, rhs);

  this->template ComputeAndCompareR2<T>(&builder, Array2D<T>(0, 0), {},
                                        this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64, Dot_0x2_2x3) {
  using T = TypeParam;
  ComputationBuilder builder(this->client_, this->TestName());
  auto lhs = builder.ConstantR2FromArray2D<T>(Array2D<T>(0, 2));
  auto rhs = builder.ConstantR2FromArray2D<T>(
      {{7.0f, 8.0f, 9.0f}, {42.0f, 77.0f, 101.0f}});
  auto result = builder.Dot(lhs, rhs);

  this->template ComputeAndCompareR2<T>(&builder, Array2D<T>(0, 3), {},
                                        this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64, Dot_3x2_2x0) {
  using T = TypeParam;
  ComputationBuilder builder(this->client_, this->TestName());
  auto lhs = builder.ConstantR2FromArray2D<T>(
      {{7.0f, 8.0f}, {9.0f, 42.0f}, {77.0f, 101.0f}});
  auto rhs = builder.ConstantR2FromArray2D<T>(Array2D<T>(2, 0));
  auto result = builder.Dot(lhs, rhs);

  this->template ComputeAndCompareR2<T>(&builder, Array2D<T>(3, 0), {},
                                        this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64, Dot_2x0_0x2) {
  using T = TypeParam;
  ComputationBuilder builder(this->client_, this->TestName());
  auto lhs = builder.ConstantR2FromArray2D<T>(Array2D<T>(2, 0));
  auto rhs = builder.ConstantR2FromArray2D<T>(Array2D<T>(0, 2));
  auto result = builder.Dot(lhs, rhs);

  this->template ComputeAndCompareR2<T>(
      &builder, Array2D<T>(2, 2, static_cast<T>(0.0f)), {}, this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64, FusedDot) {
  using T = TypeParam;
  ComputationBuilder builder(this->client_, this->TestName());
  auto param0 =
      builder.Parameter(0, ShapeUtil::MakeShapeWithType<T>({2, 4}), "arg0");
  auto param1 =
      builder.Parameter(1, ShapeUtil::MakeShapeWithType<T>({4, 1}), "arg1");
  auto exp0 = builder.Exp(param0);
  auto result = builder.Dot(exp0, param1);

  auto lhs_handle =
      this->client_
          ->TransferToServer(*Literal::CreateR2FromArray2D<T>(
              {{1.0f, 2.0f, 3.0f, 4.0f}, {-1.0f, -2.0f, -3.0f, -4.0f}}))
          .ConsumeValueOrDie();
  auto rhs_handle = this->client_
                        ->TransferToServer(*Literal::CreateR2FromArray2D<T>(
                            {{1.0f}, {2.0f}, {3.0f}, {4.0f}}))
                        .ConsumeValueOrDie();

  if (std::is_same<Eigen::half, T>::value) {
    this->error_spec_ = ErrorSpec{0.0001, 1e-3};
  }

  this->template ComputeAndCompareR2<T>(
      &builder, Array2D<T>({{296.14560492846033f}, {0.8611737683031964f}}),
      {lhs_handle.get(), rhs_handle.get()}, this->error_spec_);
}

template <typename T>
class SquareMatrixDot : public DotOperationTest {
 public:
  void TestImpl(bool lhs_row_major, bool rhs_row_major) {
    auto lhs_handle =
        client_
            ->TransferToServer(*Literal::CreateFromArrayWithLayout<T>(
                {{1.0f, 2.0f}, {3.0f, -4.0f}},
                LayoutUtil::MakeLayout(
                    MinorToMajorForIsRowMajor(lhs_row_major))))
            .ConsumeValueOrDie();
    auto rhs_handle =
        client_
            ->TransferToServer(*Literal::CreateFromArrayWithLayout<T>(
                {{1.0f, 6.0f}, {7.0f, -4.0f}},
                LayoutUtil::MakeLayout(
                    MinorToMajorForIsRowMajor(rhs_row_major))))
            .ConsumeValueOrDie();
    ComputationBuilder builder(client_, TestName());
    auto prim_type = primitive_util::NativeToPrimitiveType<T>();
    auto result = builder.Dot(
        builder.Parameter(0, ShapeUtil::MakeShape(prim_type, {2, 2}), "lhs"),
        builder.Parameter(1, ShapeUtil::MakeShape(prim_type, {2, 2}), "rhs"));

    Array2D<T> expected({{15.0f, -2.0f}, {-25.0f, 34.0f}});
    ComputeAndCompareR2<T>(&builder, expected,
                           {lhs_handle.get(), rhs_handle.get()}, error_spec_);
  }
};

TYPED_TEST_CASE(SquareMatrixDot, TypesF16F32F64CF64);
XLA_TYPED_TEST(SquareMatrixDot, TypesFF) { this->TestImpl(false, false); }
XLA_TYPED_TEST(SquareMatrixDot, TypesFT) { this->TestImpl(false, true); }
XLA_TYPED_TEST(SquareMatrixDot, TypesTF) { this->TestImpl(true, false); }
XLA_TYPED_TEST(SquareMatrixDot, TypesTT) { this->TestImpl(true, true); }

struct DotTestParam {
  int m;
  int k;
  int n;
  bool dot_lhs_row_major;
  bool dot_rhs_row_major;
  bool has_addend;
  bool addend_row_major;
};

string PrintDotTestParam(
    const ::testing::TestParamInfo<DotTestParam>& test_param) {
  const DotTestParam& param = test_param.param;
  if (param.has_addend) {
    return tensorflow::strings::StrCat(param.m, "x", param.k, "x", param.n,
                                       "_MajorToMinor",
                                       param.dot_lhs_row_major ? "T" : "F",
                                       param.dot_rhs_row_major ? "T" : "F",
                                       param.addend_row_major ? "T" : "F");
  } else {
    return tensorflow::strings::StrCat(param.m, "x", param.k, "x", param.n,
                                       "_MajorToMinor",
                                       param.dot_lhs_row_major ? "T" : "F",
                                       param.dot_rhs_row_major ? "T" : "F");
  }
}

class ParametricDotTest : public DotOperationTest,
                          public ::testing::WithParamInterface<DotTestParam> {
 protected:
  template <typename NativeT>
  void TestImpl();
};

template <typename NativeT>
void ParametricDotTest::TestImpl() {
  DotTestParam param = GetParam();

  std::unique_ptr<Array2D<NativeT>> dot_lhs_data =
      MakeLinspaceArray2D<NativeT>(0.0, 1.0, param.m, param.k);
  std::unique_ptr<Literal> dot_lhs_lit = Literal::CreateR2FromArray2DWithLayout(
      *dot_lhs_data, LayoutUtil::MakeLayout(
                         MinorToMajorForIsRowMajor(param.dot_lhs_row_major)));
  std::unique_ptr<GlobalData> dot_lhs_handle =
      client_->TransferToServer(*dot_lhs_lit).ConsumeValueOrDie();

  std::unique_ptr<Array2D<NativeT>> dot_rhs_data =
      MakeLinspaceArray2D<NativeT>(0.0, 1.0, param.k, param.n);
  Layout rhs_layout = LayoutUtil::MakeLayout(
      MinorToMajorForIsRowMajor(param.dot_rhs_row_major));
  std::unique_ptr<Literal> dot_rhs_lit =
      Literal::CreateR2FromArray2DWithLayout(*dot_rhs_data, rhs_layout);
  std::unique_ptr<GlobalData> dot_rhs_handle =
      client_->TransferToServer(*dot_rhs_lit).ConsumeValueOrDie();

  std::unique_ptr<Array2D<NativeT>> addend_data;
  std::unique_ptr<Literal> addend_lit;
  std::unique_ptr<GlobalData> addend_handle;

  if (param.has_addend) {
    addend_data = MakeLinspaceArray2D<NativeT>(0.0, 1.0, param.m, param.n);
    addend_lit = Literal::CreateR2FromArray2DWithLayout(
        *addend_data, LayoutUtil::MakeLayout(
                          MinorToMajorForIsRowMajor(param.addend_row_major)));
    addend_handle = client_->TransferToServer(*addend_lit).ConsumeValueOrDie();
  }

  ComputationBuilder builder(client_, TestName());
  auto prim_type = primitive_util::NativeToPrimitiveType<NativeT>();
  auto result = builder.Dot(
      builder.Parameter(0,
                        ShapeUtil::MakeShapeWithLayout(
                            prim_type, {param.m, param.k},
                            MinorToMajorForIsRowMajor(param.dot_lhs_row_major)),
                        "dot_lhs"),
      builder.Parameter(1,
                        ShapeUtil::MakeShapeWithLayout(
                            prim_type, {param.k, param.n},
                            MinorToMajorForIsRowMajor(param.dot_rhs_row_major)),
                        "dot_rhs"));

  if (param.has_addend) {
    result = builder.Add(
        result, builder.Parameter(
                    2,
                    ShapeUtil::MakeShapeWithLayout(
                        prim_type, {param.m, param.n},
                        MinorToMajorForIsRowMajor(param.addend_row_major)),
                    "addend"));
  }

  std::unique_ptr<Array2D<NativeT>> expected;
  if (param.has_addend) {
    expected = ReferenceUtil::ApplyElementwise2D(
        std::plus<NativeT>(),
        *ReferenceUtil::MatmulArray2D(*dot_lhs_data, *dot_rhs_data),
        *addend_data);
  } else {
    expected = ReferenceUtil::MatmulArray2D(*dot_lhs_data, *dot_rhs_data);
  }

  std::vector<GlobalData*> args = {dot_lhs_handle.get(), dot_rhs_handle.get()};
  if (param.has_addend) {
    args.push_back(addend_handle.get());
  }
  ErrorSpec error_spec(0.3, 3e-3);
  if (std::is_same<Eigen::half, NativeT>::value) {
    error_spec = ErrorSpec(0.3, 5e-3);
  }
  ComputeAndCompareR2<NativeT>(&builder, *expected, args, error_spec);
}

std::vector<DotTestParam> CreateDotTestParameters() {
  std::vector<DotTestParam> params;

  auto add_matrix_matrix_dot_test = [&](int m, int k, int n) {
    for (bool lhs_row_major : {true, false}) {
      for (bool rhs_row_major : {true, false}) {
        params.push_back({/*m=*/m, /*k=*/k, /*n=*/n,
                          /*dot_lhs_row_major=*/lhs_row_major,
                          /*dot_rhs_row_major=*/rhs_row_major,
                          /*has_addend=*/false, /*addend_row_major=*/true});
      }
    }
  };

  add_matrix_matrix_dot_test(/*m=*/12, /*k=*/117, /*n=*/7);
  add_matrix_matrix_dot_test(/*m=*/270, /*k=*/270, /*n=*/520);
  add_matrix_matrix_dot_test(/*m=*/260, /*k=*/3, /*n=*/520);

  return params;
}

#ifndef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
XLA_TEST_P(ParametricDotTest, TestF16) { TestImpl<Eigen::half>(); }
#endif
XLA_TEST_P(ParametricDotTest, TestF32) { TestImpl<float>(); }
XLA_TEST_P(ParametricDotTest, TestF64) { TestImpl<double>(); }

INSTANTIATE_TEST_CASE_P(DotTests, ParametricDotTest,
                        ::testing::ValuesIn(CreateDotTestParameters()),
                        PrintDotTestParam);

class ParametricDotTestWithoutLayoutAssignment : public ParametricDotTest {
 public:
  ParametricDotTestWithoutLayoutAssignment() {
    execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
        "layout-assignment");
  }
};

std::vector<DotTestParam> CreateNoLayoutAssignmentDotTestParameters() {
  std::vector<DotTestParam> params;

  auto add_matrix_vector_dot_test = [&](int k, int n) {
    for (bool lhs_row_major : {true, false}) {
      for (bool rhs_row_major : {true, false}) {
        for (bool has_addend : {true, false}) {
          params.push_back({/*m=*/1, /*k=*/k, /*n=*/n,
                            /*dot_lhs_row_major=*/lhs_row_major,
                            /*dot_rhs_row_major=*/rhs_row_major,
                            /*has_addend=*/has_addend,
                            /*addend_row_major=*/true});
          if (has_addend) {
            params.push_back({/*m=*/1, /*k=*/k, /*n=*/n,
                              /*dot_lhs_row_major=*/lhs_row_major,
                              /*dot_rhs_row_major=*/rhs_row_major,
                              /*has_addend=*/has_addend,
                              /*addend_row_major=*/false});
          }
          if (n != 1) {
            params.push_back({/*m=*/n, /*k=*/k, /*n=*/1,
                              /*dot_lhs_row_major=*/lhs_row_major,
                              /*dot_rhs_row_major=*/rhs_row_major,
                              /*has_addend=*/has_addend,
                              /*addend_row_major=*/true});
            if (has_addend) {
              params.push_back({/*m=*/n, /*k=*/k, /*n=*/1,
                                /*dot_lhs_row_major=*/lhs_row_major,
                                /*dot_rhs_row_major=*/rhs_row_major,
                                /*has_addend=*/has_addend,
                                /*addend_row_major=*/false});
            }
          }
        }
      }
    }
  };

  add_matrix_vector_dot_test(/*k=*/8, /*n=*/8);
  add_matrix_vector_dot_test(/*k=*/130, /*n=*/8);
  add_matrix_vector_dot_test(/*k=*/8, /*n=*/130);
  add_matrix_vector_dot_test(/*k=*/290, /*n=*/130);
  add_matrix_vector_dot_test(/*k=*/1, /*n=*/1);
  add_matrix_vector_dot_test(/*k=*/1, /*n=*/16);
  add_matrix_vector_dot_test(/*k=*/1, /*n=*/4);
  add_matrix_vector_dot_test(/*k=*/1, /*n=*/3);
  add_matrix_vector_dot_test(/*k=*/3, /*n=*/16);
  add_matrix_vector_dot_test(/*k=*/3, /*n=*/3);
  add_matrix_vector_dot_test(/*k=*/29, /*n=*/29);
  add_matrix_vector_dot_test(/*k=*/8, /*n=*/2);
  add_matrix_vector_dot_test(/*k=*/2, /*n=*/8);
  add_matrix_vector_dot_test(/*k=*/259, /*n=*/258);

  return params;
}

#ifndef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
XLA_TEST_P(ParametricDotTestWithoutLayoutAssignment, TestF16) {
  TestImpl<Eigen::half>();
}
#endif
XLA_TEST_P(ParametricDotTestWithoutLayoutAssignment, TestF32) {
  TestImpl<float>();
}
XLA_TEST_P(ParametricDotTestWithoutLayoutAssignment, TestF64) {
  TestImpl<double>();
}

INSTANTIATE_TEST_CASE_P(
    DotTests, ParametricDotTestWithoutLayoutAssignment,
    ::testing::ValuesIn(CreateNoLayoutAssignmentDotTestParameters()),
    PrintDotTestParam);

template <typename T>
class NonsquareMatrixDot : public DotOperationTest {
 public:
  void TestImpl(bool lhs_row_major, bool rhs_row_major) {
    auto lhs_handle =
        client_
            ->TransferToServer(*Literal::CreateFromArrayWithLayout<T>(
                {{1.0f, 2.0f, 3.0f}, {3.0f, -4.0f, -1.0f}},
                LayoutUtil::MakeLayout(
                    MinorToMajorForIsRowMajor(lhs_row_major))))
            .ConsumeValueOrDie();
    auto rhs_handle =
        client_
            ->TransferToServer(*Literal::CreateFromArrayWithLayout<T>(
                {{1.0f, 6.0f}, {2.0f, 3.0f}, {7.0f, -4.0f}},
                LayoutUtil::MakeLayout(
                    MinorToMajorForIsRowMajor(rhs_row_major))))
            .ConsumeValueOrDie();

    ComputationBuilder builder(client_, TestName());
    auto prim_type = primitive_util::NativeToPrimitiveType<T>();
    auto result = builder.Dot(
        builder.Parameter(0, ShapeUtil::MakeShape(prim_type, {2, 3}), "lhs"),
        builder.Parameter(1, ShapeUtil::MakeShape(prim_type, {3, 2}), "rhs"));

    Array2D<T> expected({{26.0f, 0.0f}, {-12.0f, 10.0f}});

    ComputeAndCompareR2<T>(&builder, expected,
                           {lhs_handle.get(), rhs_handle.get()}, error_spec_);
  }
};

TYPED_TEST_CASE(NonsquareMatrixDot, TypesF16F32F64CF64);
XLA_TYPED_TEST(NonsquareMatrixDot, TestFF) { this->TestImpl(false, false); }
XLA_TYPED_TEST(NonsquareMatrixDot, TestFT) { this->TestImpl(false, true); }
XLA_TYPED_TEST(NonsquareMatrixDot, TestTF) { this->TestImpl(true, false); }
XLA_TYPED_TEST(NonsquareMatrixDot, TestTT) { this->TestImpl(true, true); }

XLA_TEST_F(DotOperationTest, MatrixVectorC64) {
  auto lhs_handle =
      client_
          ->TransferToServer(*Literal::CreateR2WithLayout<complex64>(
              {{1.0, 2.0, 3.0, -4.0}}, LayoutUtil::MakeLayout({1, 0})))
          .ConsumeValueOrDie();
  auto rhs_handle =
      client_
          ->TransferToServer(*Literal::CreateR2WithLayout<complex64>(
              {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {-4.0, 4.0}},
              LayoutUtil::MakeLayout({1, 0})))
          .ConsumeValueOrDie();

  ComputationBuilder builder(client_, TestName());
  auto prim_type = primitive_util::NativeToPrimitiveType<complex64>();
  auto result = builder.Dot(
      builder.Parameter(0, ShapeUtil::MakeShape(prim_type, {1, 4}), "lhs"),
      builder.Parameter(1, ShapeUtil::MakeShape(prim_type, {4, 2}), "rhs"));

  Array2D<complex64> expected({{30.0, -2.0}});

  ComputeAndCompareR2<complex64>(
      &builder, expected, {lhs_handle.get(), rhs_handle.get()}, error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64, ConcurrentMatMult) {
  using T = TypeParam;

  ComputationBuilder builder(this->client_, this->TestName());
  auto matrix1 = builder.ConstantR2FromArray2D<T>({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto matrix2 = builder.ConstantR2FromArray2D<T>({{5.0f, 6.0f}, {7.0f, 8.0f}});
  auto matrix12 = builder.Dot(matrix1, matrix2);
  auto matrix21 = builder.Dot(matrix2, matrix1);
  builder.Add(matrix12, matrix21);

  Array2D<T> expected({{42.0f, 56.0f}, {74.0f, 96.0f}});
  this->template ComputeAndCompareR2<T>(&builder, expected, {},
                                        this->error_spec_);
}

template <typename T>
class DotOperationTestForBatchMatMul : public DotOperationTest {};
TYPED_TEST_CASE(DotOperationTestForBatchMatMul, TypesF16F32F64);

// Regression test for b/32055648. The root of the graph is a kFusion of 4
// bitcasts. Although bitcasts don't map to thunks, the root should still be
// sync-dependent on bitcasts' operands.
XLA_TYPED_TEST(DotOperationTestForBatchMatMul, Types) {
  using T = TypeParam;
  ComputationBuilder builder(this->client_, this->TestName());
  auto x =
      builder.Parameter(0, ShapeUtil::MakeShapeWithType<T>({2, 2, 2, 2}), "x");
  auto y =
      builder.Parameter(1, ShapeUtil::MakeShapeWithType<T>({2, 2, 2, 2}), "y");

  auto x_flat = builder.Reshape(x, {0, 1, 2, 3}, {4, 2, 2});
  auto y_flat = builder.Reshape(y, {0, 1, 2, 3}, {4, 2, 2});

  // Slice batches into individual matrices and multiply them.
  std::vector<xla::ComputationDataHandle> out_slices;
  for (int i = 0; i < 4; ++i) {
    // Slice off individual matrices and reshape to 2D tensors.
    auto x_slice = builder.Slice(x_flat, {i, 0, 0}, {i + 1, 2, 2}, {1, 1, 1});
    x_slice = builder.Reshape(x_slice, {0, 1, 2}, {2, 2});
    auto y_slice = builder.Slice(y_flat, {i, 0, 0}, {i + 1, 2, 2}, {1, 1, 1});
    y_slice = builder.Reshape(y_slice, {0, 1, 2}, {2, 2});

    auto out = builder.Dot(x_slice, y_slice);
    out = builder.Reshape(out, {0, 1}, {1, 2, 2});
    out_slices.push_back(out);
  }
  auto out_flat = builder.ConcatInDim(out_slices, 0);
  builder.Reshape(out_flat, {0, 1, 2}, {2, 2, 2, 2});

  auto x_data = this->client_
                    ->TransferToServer(*Literal::CreateR4FromArray4D<T>(
                        {{{{1000.0f, 100.0f}, {10.0f, 1.0f}},
                          {{2000.0f, 200.0f}, {20.0f, 2.0f}}},
                         {{{3000.0f, 300.0f}, {30.0f, 3.0f}},
                          {{4000.0f, 400.0f}, {40.0f, 4.0f}}}}))
                    .ConsumeValueOrDie();
  auto y_data =
      this->client_
          ->TransferToServer(*Literal::CreateR4FromArray4D<T>(
              {{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}},
               {{{11.0f, 22.0f}, {33.0f, 44.0f}},
                {{55.0f, 66.0f}, {77.0f, 88.0f}}}}))
          .ConsumeValueOrDie();

  if (std::is_same<Eigen::half, T>::value) {
    this->error_spec_ = ErrorSpec{0.0001, 1e-3};
  }
  this->template ComputeAndCompareR4<T>(
      &builder,
      /*expected=*/
      {{{{1300.0f, 2400.0f}, {13.0f, 24.0f}},
        {{11400.0f, 13600.0f}, {114.0f, 136.0f}}},
       {{{42900.0f, 79200.0f}, {429.0f, 792.0f}},
        {{250800.0f, 299200.0f}, {2508.0f, 2992.0f}}}},
      {x_data.get(), y_data.get()}, this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64, GeneralMatMul) {
  using T = TypeParam;

  ComputationBuilder builder(this->client_, this->TestName());
  auto x =
      builder.Parameter(0, ShapeUtil::MakeShapeWithType<T>({2, 2, 2}), "x");
  auto y =
      builder.Parameter(1, ShapeUtil::MakeShapeWithType<T>({2, 2, 2}), "y");

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(1);
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);

  auto out = builder.DotGeneral(x, y, dnums);

  auto x_data =
      this->client_
          ->TransferToServer(*Literal::CreateR3FromArray3D<T>(
              {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}}))
          .ConsumeValueOrDie();

  auto y_data =
      this->client_
          ->TransferToServer(*Literal::CreateR3FromArray3D<T>(
              {{{1.0f, 0.0f}, {0.0f, 1.0f}}, {{1.0f, 0.0f}, {0.0f, 1.0f}}}))
          .ConsumeValueOrDie();

  this->template ComputeAndCompareR3<T>(
      &builder,
      /*expected=*/
      {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}},
      {x_data.get(), y_data.get()}, this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64, TransposeFolding) {
  using T = TypeParam;
  for (bool transpose_lhs : {false, true}) {
    for (bool transpose_rhs : {false, true}) {
      for (bool row_major : {false, true}) {
        std::unique_ptr<Array2D<T>> lhs(
            new Array2D<T>({{1.0f, 2.0f, 3.0f}, {3.0f, -4.0f, -1.0f}}));
        std::unique_ptr<Array2D<T>> rhs(
            new Array2D<T>({{1.0f, 6.0f}, {2.0f, 3.0f}, {7.0f, -4.0f}}));

        if (transpose_lhs) {
          lhs = ReferenceUtil::TransposeArray2D(*lhs);
        }
        if (transpose_rhs) {
          rhs = ReferenceUtil::TransposeArray2D(*rhs);
        }
        auto lhs_handle =
            this->client_
                ->TransferToServer(*Literal::CreateR2FromArray2DWithLayout<T>(
                    *lhs, LayoutUtil::MakeLayout(
                              MinorToMajorForIsRowMajor(row_major))))
                .ConsumeValueOrDie();
        auto rhs_handle =
            this->client_
                ->TransferToServer(*Literal::CreateR2FromArray2DWithLayout<T>(
                    *rhs, LayoutUtil::MakeLayout(
                              MinorToMajorForIsRowMajor(row_major))))
                .ConsumeValueOrDie();

        ComputationBuilder builder(this->client_, this->TestName());
        auto prim_type = primitive_util::NativeToPrimitiveType<T>();
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

        Array2D<T> expected({{26.0f, 0.0f}, {-12.0f, 10.0f}});
        VLOG(1) << "TestTransposeFolding " << transpose_lhs << " "
                << transpose_rhs << " " << row_major;
        this->template ComputeAndCompareR2<T>(
            &builder, expected, {lhs_handle.get(), rhs_handle.get()},
            this->error_spec_);
      }
    }
  }
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64,
               DotOfConcatOptimizationWithConstLHS) {
  using T = TypeParam;
  auto prim_type = primitive_util::NativeToPrimitiveType<T>();

  std::unique_ptr<Array2D<T>> constant_lhs_array(
      new Array2D<T>({{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                      {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}}));

  ComputationBuilder builder(this->client_, this->TestName());
  auto lhs_constant = builder.ConstantR2FromArray2D(*constant_lhs_array);
  auto rhs_arg_0 = builder.Parameter(0, ShapeUtil::MakeShape(prim_type, {2, 2}),
                                     "rhs_arg_0");
  auto rhs_arg_1 = builder.Parameter(1, ShapeUtil::MakeShape(prim_type, {3, 2}),
                                     "rhs_arg_1");
  auto rhs_arg_2 = builder.Parameter(2, ShapeUtil::MakeShape(prim_type, {1, 2}),
                                     "rhs_arg_2");
  auto result = builder.Dot(
      lhs_constant, builder.ConcatInDim({rhs_arg_0, rhs_arg_1, rhs_arg_2}, 0));

  std::unique_ptr<Array2D<T>> arg_0_value_array(
      new Array2D<T>({{1.0f, 2.0f}, {3.0f, 4.0f}}));
  std::unique_ptr<Array2D<T>> arg_1_value_array(
      new Array2D<T>({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}));
  std::unique_ptr<Array2D<T>> arg_2_value_array(new Array2D<T>({{1.0f, 2.0f}}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_0_value,
      this->client_->TransferToServer(
          *Literal::CreateR2FromArray2D<T>(*arg_0_value_array)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_1_value,
      this->client_->TransferToServer(
          *Literal::CreateR2FromArray2D<T>(*arg_1_value_array)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_2_value,
      this->client_->TransferToServer(
          *Literal::CreateR2FromArray2D<T>(*arg_2_value_array)));

  Array2D<T> expected({{53.0f, 74.0f}, {45.0f, 66.0f}});
  this->template ComputeAndCompareR2<T>(
      &builder, expected,
      {arg_0_value.get(), arg_1_value.get(), arg_2_value.get()},
      this->error_spec_);
}

XLA_TYPED_TEST(DotOperationTest_F16F32F64,
               DotOfConcatOptimizationWithConstRHS) {
  using T = TypeParam;
  std::unique_ptr<Array2D<T>> constant_rhs_array(
      new Array2D<T>({{1.0f, 2.0f},
                      {3.0f, 4.0f},
                      {5.0f, 6.0f},
                      {6.0f, 5.0f},
                      {4.0f, 3.0f},
                      {2.0f, 1.0f}}));

  ComputationBuilder builder(this->client_, this->TestName());
  auto rhs_constant = builder.ConstantR2FromArray2D(*constant_rhs_array);
  auto lhs_arg_0 = builder.Parameter(0, ShapeUtil::MakeShapeWithType<T>({2, 2}),
                                     "lhs_arg_0");
  auto lhs_arg_1 = builder.Parameter(1, ShapeUtil::MakeShapeWithType<T>({2, 3}),
                                     "lhs_arg_1");
  auto lhs_arg_2 = builder.Parameter(2, ShapeUtil::MakeShapeWithType<T>({2, 1}),
                                     "lhs_arg_2");
  auto result = builder.Dot(
      builder.ConcatInDim({lhs_arg_0, lhs_arg_1, lhs_arg_2}, 1), rhs_constant);

  std::unique_ptr<Array2D<T>> arg_0_value_array(
      new Array2D<T>({{1.0f, 2.0f}, {3.0f, 4.0f}}));
  std::unique_ptr<Array2D<T>> arg_1_value_array(
      new Array2D<T>({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}));
  std::unique_ptr<Array2D<T>> arg_2_value_array(
      new Array2D<T>({{1.0f}, {2.0f}}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_0_value,
      this->client_->TransferToServer(
          *Literal::CreateR2FromArray2D<T>(*arg_0_value_array)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_1_value,
      this->client_->TransferToServer(
          *Literal::CreateR2FromArray2D<T>(*arg_1_value_array)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_2_value,
      this->client_->TransferToServer(
          *Literal::CreateR2FromArray2D<T>(*arg_2_value_array)));

  Array2D<T> expected({{38.0f, 36.0f}, {93.0f, 91.0f}});
  this->template ComputeAndCompareR2<T>(
      &builder, expected,
      {arg_0_value.get(), arg_1_value.get(), arg_2_value.get()},
      this->error_spec_);
}

}  // namespace
}  // namespace xla
