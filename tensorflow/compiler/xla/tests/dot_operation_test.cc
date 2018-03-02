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

XLA_TEST_F(DotOperationTest, FusedDot) {
  ComputationBuilder builder(client_, TestName());
  auto param0 = builder.Parameter(0, ShapeUtil::MakeShape(F32, {2, 4}), "arg0");
  auto param1 = builder.Parameter(1, ShapeUtil::MakeShape(F32, {4, 1}), "arg1");
  auto exp0 = builder.Exp(param0);
  auto result = builder.Dot(exp0, param1);

  auto lhs_handle = client_
                        ->TransferToServer(*Literal::CreateR2<float>(
                            {{1.0, 2.0, 3.0, 4.0}, {-1.0, -2.0, -3.0, -4.0}}))
                        .ConsumeValueOrDie();
  auto rhs_handle = client_
                        ->TransferToServer(*Literal::CreateR2<float>(
                            {{1.0}, {2.0}, {3.0}, {4.0}}))
                        .ConsumeValueOrDie();

  ComputeAndCompareR2<float>(
      &builder, Array2D<float>({{296.14560492846033}, {0.8611737683031964}}),
      {lhs_handle.get(), rhs_handle.get()}, error_spec_);
}

template <typename Element>
void DotOperationTest::TestSquareMatrixDot(bool lhs_row_major,
                                           bool rhs_row_major) {
  auto lhs_handle =
      client_
          ->TransferToServer(*Literal::CreateR2WithLayout<Element>(
              {{1.0, 2.0}, {3.0, -4.0}},
              LayoutUtil::MakeLayout(MinorToMajorForIsRowMajor(lhs_row_major))))
          .ConsumeValueOrDie();
  auto rhs_handle =
      client_
          ->TransferToServer(*Literal::CreateR2WithLayout<Element>(
              {{1.0, 6.0}, {7.0, -4.0}},
              LayoutUtil::MakeLayout(MinorToMajorForIsRowMajor(rhs_row_major))))
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

  ComputeAndCompareR2<NativeT>(&builder, *expected, args, ErrorSpec(0.3, 3e-3));
}

XLA_TEST_P(ParametricDotTest, TestF32) { TestImpl<float>(); }

XLA_TEST_P(ParametricDotTest, TestF64) { TestImpl<double>(); }

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

XLA_TEST_P(ParametricDotTestWithoutLayoutAssignment, TestF32) {
  TestImpl<float>();
}

XLA_TEST_P(ParametricDotTestWithoutLayoutAssignment, TestF64) {
  TestImpl<double>();
}

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

INSTANTIATE_TEST_CASE_P(
    DotTests, ParametricDotTestWithoutLayoutAssignment,
    ::testing::ValuesIn(CreateNoLayoutAssignmentDotTestParameters()),
    PrintDotTestParam);

XLA_TEST_F(DotOperationTest, SquareMatrixDotF32MinorToMajorFF) {
  TestSquareMatrixDot<float>(false, false);
}

XLA_TEST_F(DotOperationTest, SquareMatrixDotF32MinorToMajorFT) {
  TestSquareMatrixDot<float>(false, true);
}

XLA_TEST_F(DotOperationTest, SquareMatrixDotF32MinorToMajorTF) {
  TestSquareMatrixDot<float>(true, false);
}

XLA_TEST_F(DotOperationTest, SquareMatrixDotF32MinorToMajorTT) {
  TestSquareMatrixDot<float>(true, true);
}

XLA_TEST_F(DotOperationTest, SquareMatrixDotC64MinorToMajorFF) {
  TestSquareMatrixDot<complex64>(false, false);
}

XLA_TEST_F(DotOperationTest, SquareMatrixDotC64MinorToMajorFT) {
  TestSquareMatrixDot<complex64>(false, true);
}

XLA_TEST_F(DotOperationTest, SquareMatrixDotC64MinorToMajorTF) {
  TestSquareMatrixDot<complex64>(true, false);
}

XLA_TEST_F(DotOperationTest, SquareMatrixDotC64MinorToMajorTT) {
  TestSquareMatrixDot<complex64>(true, true);
}

XLA_TEST_F(DotOperationTest, SquareMatrixDotF64) {
  TestSquareMatrixDot<double>();
}

template <typename Element>
void DotOperationTest::TestNonsquareMatrixDot(bool lhs_row_major,
                                              bool rhs_row_major) {
  auto lhs_handle =
      client_
          ->TransferToServer(*Literal::CreateR2WithLayout<Element>(
              {{1.0, 2.0, 3.0}, {3.0, -4.0, -1.0}},
              LayoutUtil::MakeLayout(MinorToMajorForIsRowMajor(lhs_row_major))))
          .ConsumeValueOrDie();
  auto rhs_handle =
      client_
          ->TransferToServer(*Literal::CreateR2WithLayout<Element>(
              {{1.0, 6.0}, {2.0, 3.0}, {7.0, -4.0}},
              LayoutUtil::MakeLayout(MinorToMajorForIsRowMajor(rhs_row_major))))
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
  TestNonsquareMatrixDot<float>(false, false);
}

XLA_TEST_F(DotOperationTest, NonsquareMatrixDotF32MajorToMinorFT) {
  TestNonsquareMatrixDot<float>(false, true);
}

XLA_TEST_F(DotOperationTest, NonsquareMatrixDotF32MajorToMinorTF) {
  TestNonsquareMatrixDot<float>(true, false);
}

XLA_TEST_F(DotOperationTest, NonsquareMatrixDotF32MajorToMinorTT) {
  TestNonsquareMatrixDot<float>(true, true);
}

XLA_TEST_F(DotOperationTest, NonsquareMatrixDotF64) {
  TestNonsquareMatrixDot<double>();
}

XLA_TEST_F(DotOperationTest, NonsquareMatrixDotC64MajorToMinorFF) {
  TestNonsquareMatrixDot<complex64>(false, false);
}

XLA_TEST_F(DotOperationTest, NonsquareMatrixDotC64MajorToMinorFT) {
  TestNonsquareMatrixDot<complex64>(false, true);
}

XLA_TEST_F(DotOperationTest, NonsquareMatrixDotC64MajorToMinorTF) {
  TestNonsquareMatrixDot<complex64>(true, false);
}

XLA_TEST_F(DotOperationTest, NonsquareMatrixDotC64MajorToMinorTT) {
  TestNonsquareMatrixDot<complex64>(true, true);
}

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

XLA_TEST_F(DotOperationTest, ConcurrentMatMul) {
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

  auto x_data = client_
                    ->TransferToServer(*Literal::CreateR4<float>(
                        {{{{1000, 100}, {10, 1}}, {{2000, 200}, {20, 2}}},
                         {{{3000, 300}, {30, 3}}, {{4000, 400}, {40, 4}}}}))
                    .ConsumeValueOrDie();
  auto y_data = client_
                    ->TransferToServer(*Literal::CreateR4<float>(
                        {{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}},
                         {{{11, 22}, {33, 44}}, {{55, 66}, {77, 88}}}}))
                    .ConsumeValueOrDie();

  ComputeAndCompareR4<float>(
      &builder,
      /*expected=*/
      {{{{1300, 2400}, {13, 24}}, {{11400, 13600}, {114, 136}}},
       {{{42900, 79200}, {429, 792}}, {{250800, 299200}, {2508, 2992}}}},
      {x_data.get(), y_data.get()}, error_spec_);
}

XLA_TEST_F(DotOperationTest, GeneralMatMul) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {2, 2, 2}), "x");
  auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {2, 2, 2}), "y");

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(1);
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);

  auto out = builder.DotGeneral(x, y, dnums);

  auto x_data = client_
                    ->TransferToServer(*Literal::CreateR3<float>(
                        {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}))
                    .ConsumeValueOrDie();

  auto y_data = client_
                    ->TransferToServer(*Literal::CreateR3<float>(
                        {{{1.0, 0.0}, {0.0, 1.0}}, {{1.0, 0.0}, {0.0, 1.0}}}))
                    .ConsumeValueOrDie();

  ComputeAndCompareR3<float>(
      &builder,
      /*expected=*/
      {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}},
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
                    *Literal::CreateR2FromArray2DWithLayout<float>(
                        *lhs, LayoutUtil::MakeLayout(
                                  MinorToMajorForIsRowMajor(row_major))))
                .ConsumeValueOrDie();
        auto rhs_handle =
            client_
                ->TransferToServer(
                    *Literal::CreateR2FromArray2DWithLayout<float>(
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

TEST_F(DotOperationTest, DotOfConcatOptimizationWithConstLHS) {
  auto prim_type = primitive_util::NativeToPrimitiveType<float>();

  std::unique_ptr<Array2D<float>> constant_lhs_array(new Array2D<float>(
      {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0}}));

  ComputationBuilder builder(client_, TestName());
  auto lhs_constant = builder.ConstantR2FromArray2D(*constant_lhs_array);
  auto rhs_arg_0 = builder.Parameter(0, ShapeUtil::MakeShape(prim_type, {2, 2}),
                                     "rhs_arg_0");
  auto rhs_arg_1 = builder.Parameter(1, ShapeUtil::MakeShape(prim_type, {3, 2}),
                                     "rhs_arg_1");
  auto rhs_arg_2 = builder.Parameter(2, ShapeUtil::MakeShape(prim_type, {1, 2}),
                                     "rhs_arg_2");
  auto result = builder.Dot(
      lhs_constant, builder.ConcatInDim({rhs_arg_0, rhs_arg_1, rhs_arg_2}, 0));

  std::unique_ptr<Array2D<float>> arg_0_value_array(
      new Array2D<float>({{1.0, 2.0}, {3.0, 4.0}}));
  std::unique_ptr<Array2D<float>> arg_1_value_array(
      new Array2D<float>({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}));
  std::unique_ptr<Array2D<float>> arg_2_value_array(
      new Array2D<float>({{1.0, 2.0}}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_0_value,
      client_->TransferToServer(
          *Literal::CreateR2FromArray2D<float>(*arg_0_value_array)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_1_value,
      client_->TransferToServer(
          *Literal::CreateR2FromArray2D<float>(*arg_1_value_array)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_2_value,
      client_->TransferToServer(
          *Literal::CreateR2FromArray2D<float>(*arg_2_value_array)));

  Array2D<float> expected({{53.0, 74.0}, {45.0, 66.0}});
  ComputeAndCompareR2<float>(
      &builder, expected,
      {arg_0_value.get(), arg_1_value.get(), arg_2_value.get()}, error_spec_);
}

TEST_F(DotOperationTest, DotOfConcatOptimizationWithConstRHS) {
  auto prim_type = primitive_util::NativeToPrimitiveType<float>();

  std::unique_ptr<Array2D<float>> constant_rhs_array(
      new Array2D<float>({{1.0, 2.0},
                          {3.0, 4.0},
                          {5.0, 6.0},
                          {6.0, 5.0},
                          {4.0, 3.0},
                          {2.0, 1.0}}));

  ComputationBuilder builder(client_, TestName());
  auto rhs_constant = builder.ConstantR2FromArray2D(*constant_rhs_array);
  auto lhs_arg_0 = builder.Parameter(0, ShapeUtil::MakeShape(prim_type, {2, 2}),
                                     "lhs_arg_0");
  auto lhs_arg_1 = builder.Parameter(1, ShapeUtil::MakeShape(prim_type, {2, 3}),
                                     "lhs_arg_1");
  auto lhs_arg_2 = builder.Parameter(2, ShapeUtil::MakeShape(prim_type, {2, 1}),
                                     "lhs_arg_2");
  auto result = builder.Dot(
      builder.ConcatInDim({lhs_arg_0, lhs_arg_1, lhs_arg_2}, 1), rhs_constant);

  std::unique_ptr<Array2D<float>> arg_0_value_array(
      new Array2D<float>({{1.0, 2.0}, {3.0, 4.0}}));
  std::unique_ptr<Array2D<float>> arg_1_value_array(
      new Array2D<float>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
  std::unique_ptr<Array2D<float>> arg_2_value_array(
      new Array2D<float>({{1.0}, {2.0}}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_0_value,
      client_->TransferToServer(
          *Literal::CreateR2FromArray2D<float>(*arg_0_value_array)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_1_value,
      client_->TransferToServer(
          *Literal::CreateR2FromArray2D<float>(*arg_1_value_array)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_2_value,
      client_->TransferToServer(
          *Literal::CreateR2FromArray2D<float>(*arg_2_value_array)));

  Array2D<float> expected({{38.0, 36.0}, {93.0, 91.0}});
  ComputeAndCompareR2<float>(
      &builder, expected,
      {arg_0_value.get(), arg_1_value.get(), arg_2_value.get()}, error_spec_);
}
}  // namespace
}  // namespace xla
