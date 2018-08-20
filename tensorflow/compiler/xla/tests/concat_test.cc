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
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

using ConcatTest = ClientLibraryTestBase;
using ::testing::HasSubstr;

// Concatenate expects at least one argument.
XLA_TEST_F(ConcatTest, Concat_Nothing) {
  XlaBuilder builder(TestName());
  ConcatInDim(&builder, {}, 0);
  StatusOr<XlaComputation> computation_status = builder.Build();
  ASSERT_FALSE(computation_status.ok());
  EXPECT_THAT(computation_status.status().ToString(),
              HasSubstr("Concatenate expects at least one argument"));
}

// Concatenate with one argument works.
XLA_TEST_F(ConcatTest, Concat_R1_With_Nothing) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.0, 64.0});
  ConcatInDim(&builder, {a}, 0);

  std::vector<float> expected = {42, 64};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_R1_L0_With_Nothing) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {});
  ConcatInDim(&builder, {a}, 0);

  std::vector<float> expected = {};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// Show that we can't concatenate R0 with R0 because we can't name the dimension
// to concatenate on.
XLA_TEST_F(ConcatTest, CannotConcatR0WithR0) {
  XlaBuilder builder(TestName());
  auto a = ConstantR0<float>(&builder, 42.0);
  auto b = ConstantR0<float>(&builder, 64.0);
  ConcatInDim(&builder, {a, b}, 0);
  StatusOr<XlaComputation> computation_status = builder.Build();
  ASSERT_FALSE(computation_status.ok());
  EXPECT_THAT(computation_status.status().ToString(),
              HasSubstr("out of bounds: 0"));
}

XLA_TEST_F(ConcatTest, Concat_R1_L0_With_R1_L0) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {});
  auto b = ConstantR1<float>(&builder, {});
  ConcatInDim(&builder, {a, b}, 0);

  std::vector<float> expected = {};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_R1_L0_With_R1_L1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {});
  auto b = ConstantR1<float>(&builder, {256.0});
  ConcatInDim(&builder, {a, b}, 0);

  std::vector<float> expected = {256};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_R1_L2_With_R1_L0) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.0, 64.0});
  auto b = ConstantR1<float>(&builder, {});
  ConcatInDim(&builder, {a, b}, 0);

  std::vector<float> expected = {42, 64};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_R1_L2_With_R1_L1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.0, 64.0});
  auto b = ConstantR1<float>(&builder, {256.0});
  ConcatInDim(&builder, {a, b}, 0);

  std::vector<float> expected = {42, 64, 256};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_R1_L253_With_R1_L7) {
  std::vector<float> lhs(253);
  std::vector<float> rhs(7);
  std::vector<float> expected(253 + 7);
  for (int i = 0; i < 253; ++i) {
    expected[i] = lhs[i] = i + 1;
  }
  for (int i = 0; i < 7; ++i) {
    expected[253 + i] = rhs[i] = 253 + i + 1;
  }

  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, lhs);
  auto b = ConstantR1<float>(&builder, rhs);
  ConcatInDim(&builder, {a, b}, 0);

  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_0x0_With_0x0) {
  for (int dim : {0, 1}) {
    XlaBuilder builder(TestName());
    auto a = ConstantR2FromArray2D(&builder, Array2D<float>(0, 0));
    auto b = ConstantR2FromArray2D(&builder, Array2D<float>(0, 0));
    ConcatInDim(&builder, {a, b}, dim);

    ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 0), {},
                               ErrorSpec(0.0001));
  }
}

XLA_TEST_F(ConcatTest, Concat_1x1_With_1x1_InDim0) {
  XlaBuilder builder(TestName());
  auto a_array = CreatePatternedMatrix(1, 1);
  auto b_array = CreatePatternedMatrix(1, 1, /*offset=*/64.0);
  auto a = ConstantR2FromArray2D(&builder, *a_array);
  auto b = ConstantR2FromArray2D(&builder, *b_array);
  ConcatInDim(&builder, {a, b}, 0);

  Array2D<float> expected({
      {0},
      {64},
  });
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_1x1_With_1x1_InDim1) {
  XlaBuilder builder(TestName());
  auto a_array = CreatePatternedMatrix(1, 1);
  auto b_array = CreatePatternedMatrix(1, 1, /*offset=*/64.0);
  auto a = ConstantR2FromArray2D(&builder, *a_array);
  auto b = ConstantR2FromArray2D(&builder, *b_array);
  ConcatInDim(&builder, {a, b}, 1);

  Array2D<float> expected({
      {0, 64},
  });
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat2x0With2x5) {
  XlaBuilder builder(TestName());
  auto b_array = CreatePatternedMatrix(2, 5, /*offset=*/64.0);
  auto a = ConstantR2FromArray2D(&builder, Array2D<float>(2, 0));
  auto b = ConstantR2FromArray2D(&builder, *b_array);
  ConcatInDim(&builder, {a, b}, 1);

  ComputeAndCompareR2<float>(&builder, *b_array, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat2x3With2x5) {
  XlaBuilder builder(TestName());
  auto a_array = CreatePatternedMatrix(2, 3);
  auto b_array = CreatePatternedMatrix(2, 5, /*offset=*/64.0);
  auto a = ConstantR2FromArray2D(&builder, *a_array);
  auto b = ConstantR2FromArray2D(&builder, *b_array);
  ConcatInDim(&builder, {a, b}, 1);

  Array2D<float> expected({
      {0, 1, 2, 64, 65, 66, 67, 68},
      {1000, 1001, 1002, 1064, 1065, 1066, 1067, 1068},
  });
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat3x2With0x2) {
  XlaBuilder builder(TestName());
  auto a_array = CreatePatternedMatrix(3, 2);
  auto a = ConstantR2FromArray2D(&builder, *a_array);
  auto b = ConstantR2FromArray2D(&builder, Array2D<float>(0, 2));
  ConcatInDim(&builder, {a, b}, 0);

  ComputeAndCompareR2<float>(&builder, *a_array, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat3x2With5x2) {
  XlaBuilder builder(TestName());
  auto a_array = CreatePatternedMatrix(3, 2);
  auto b_array = CreatePatternedMatrix(5, 2, /*offset=*/64.0);
  auto a = ConstantR2FromArray2D(&builder, *a_array);
  auto b = ConstantR2FromArray2D(&builder, *b_array);
  ConcatInDim(&builder, {a, b}, 0);

  Array2D<float> expected({
      {0, 1},
      {1000, 1001},
      {2000, 2001},
      {64, 65},
      {1064, 1065},
      {2064, 2065},
      {3064, 3065},
      {4064, 4065},
  });
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_R3_3x0x2_3x0x1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR3FromArray3D(&builder, Array3D<float>(3, 0, 2));
  auto b = ConstantR3FromArray3D(&builder, Array3D<float>(3, 0, 1));
  ConcatInDim(&builder, {a, b}, 2);
  ComputeAndCompareR3<float>(&builder, Array3D<float>(3, 0, 3), {},
                             ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_R3_3x1x2_3x1x1) {
  XlaBuilder builder(TestName());
  Array3D<float> a_array({
      // 3x1x2
      {{0, 1}},
      {{2, 3}},
      {{4, 5}},
  });
  Array3D<float> b_array({
      // 3x1x1
      {{6}},
      {{7}},
      {{8}},
  });
  auto a = ConstantR3FromArray3D(&builder, a_array);
  auto b = ConstantR3FromArray3D(&builder, b_array);
  ConcatInDim(&builder, {a, b}, 2);

  Array3D<float> expected({
      {{0, 1, 6}},
      {{2, 3, 7}},
      {{4, 5, 8}},
  });
  ComputeAndCompareR3<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_R1_1x1_1x1_1x1) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.0});
  auto b = ConstantR1<float>(&builder, {64.0});
  auto c = ConstantR1<float>(&builder, {256.0});
  ConcatInDim(&builder, {a, b, c}, 0);

  std::vector<float> expected = {42, 64, 256};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_R3_3x1x2_3x1x1_3x1x1) {
  XlaBuilder builder(TestName());
  Array3D<float> a_array({
      // 3x1x2
      {{0, 1}},
      {{4, 5}},
      {{8, 9}},
  });
  Array3D<float> b_array({
      // 3x1x1
      {{2}},
      {{6}},
      {{10}},
  });
  Array3D<float> c_array({
      // 3x1x1
      {{3}},
      {{7}},
      {{11}},
  });
  auto a = ConstantR3FromArray3D(&builder, a_array);
  auto b = ConstantR3FromArray3D(&builder, b_array);
  auto c = ConstantR3FromArray3D(&builder, c_array);
  ConcatInDim(&builder, {a, b, c}, 2);

  Array3D<float> expected({
      {{0, 1, 2, 3}},
      {{4, 5, 6, 7}},
      {{8, 9, 10, 11}},
  });
  ComputeAndCompareR3<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, DoubleConcatLeftAssociative) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.0});
  auto b = ConstantR1<float>(&builder, {64.0});
  auto c = ConstantR1<float>(&builder, {256.0});
  // concatenated = (a concat b) concat c
  ConcatInDim(&builder, {ConcatInDim(&builder, {a, b}, 0), c}, 0);

  std::vector<float> expected = {42, 64, 256};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, DoubleConcatRightAssociative) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.0});
  auto b = ConstantR1<float>(&builder, {64.0});
  auto c = ConstantR1<float>(&builder, {256.0});
  // concatenated = a concat (b concat c)
  ConcatInDim(&builder, {a, ConcatInDim(&builder, {b, c}, 0)}, 0);

  std::vector<float> expected = {42, 64, 256};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_1x1024_With_1x1024_InDim0) {
  Array2D<float> lhs(1, 1024);
  Array2D<float> rhs(1, 1024);
  for (int i = 0; i < 1024; ++i) {
    lhs(0, i) = i;
    rhs(0, i) = i + 1024;
  }

  XlaBuilder builder(TestName());
  auto a = ConstantR2FromArray2D<float>(&builder, lhs);
  auto b = ConstantR2FromArray2D<float>(&builder, rhs);
  ConcatInDim(&builder, {a, b}, 0);

  Array2D<float> expected(2, 1024);
  for (int i = 0; i < 1024; ++i) {
    expected(0, i) = i;
    expected(1, i) = i + 1024;
  }
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_1x1024_With_1x1024_InDim1) {
  Array2D<float> lhs(1, 1024);
  Array2D<float> rhs(1, 1024);
  for (int i = 0; i < 1024; ++i) {
    lhs(0, i) = i;
    rhs(0, i) = i + 1024;
  }

  XlaBuilder builder(TestName());
  auto a = ConstantR2FromArray2D<float>(&builder, lhs);
  auto b = ConstantR2FromArray2D<float>(&builder, rhs);
  ConcatInDim(&builder, {a, b}, 1);

  Array2D<float> expected(1, 2048);
  for (int i = 0; i < 1024; ++i) {
    expected(0, i) = i;
    expected(0, i + 1024) = i + 1024;
  }
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConcatTest, Concat_64x64_With_64x2) {
  Array2D<float> lhs(64, 64);
  Array2D<float> rhs(64, 2);
  for (int i0 = 0; i0 < 64; ++i0) {
    for (int i1 = 0; i1 < 64; ++i1) {
      lhs(i0, i1) = (i0 << 10) | i1;
    }
    for (int i1 = 0; i1 < 2; ++i1) {
      rhs(i0, i1) = (i0 << 10) | (i1 + 64);
    }
  }

  XlaBuilder builder(TestName());
  auto a = ConstantR2FromArray2D<float>(&builder, lhs);
  auto b = ConstantR2FromArray2D<float>(&builder, rhs);
  ConcatInDim(&builder, {a, b}, 1);

  Array2D<float> expected(64, 66);
  for (int i0 = 0; i0 < 64; ++i0) {
    for (int i1 = 0; i1 < 66; ++i1) {
      expected(i0, i1) = (i0 << 10) | i1;
    }
  }
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// Show that we can't concatenate with an opaques.
XLA_TEST_F(ConcatTest, CannotConcatOpaques) {
  XlaBuilder builder(TestName());
  auto opaque_shape = ShapeUtil::MakeOpaqueShape();
  auto r1f32 = xla::ShapeUtil::MakeShape(xla::F32, {1});
  auto x = Parameter(&builder, 0, r1f32, "x");
  auto y = Parameter(&builder, 1, opaque_shape, "y");
  ConcatInDim(&builder, {x, y}, 0);
  StatusOr<XlaComputation> computation_status = builder.Build();
  ASSERT_FALSE(computation_status.ok());
  EXPECT_THAT(
      computation_status.status().ToString(),
      HasSubstr("Expected array argument for operand of concatenation"));
}

// Show that we can't concatenate with tokens.
XLA_TEST_F(ConcatTest, CannotConcatTokens) {
  XlaBuilder builder(TestName());
  auto token_shape = ShapeUtil::MakeTokenShape();
  auto r1f32 = xla::ShapeUtil::MakeShape(xla::F32, {1});
  auto x = Parameter(&builder, 0, r1f32, "x");
  auto y = Parameter(&builder, 1, token_shape, "y");
  ConcatInDim(&builder, {x, y}, 0);
  StatusOr<XlaComputation> computation_status = builder.Build();
  ASSERT_FALSE(computation_status.ok());
  EXPECT_THAT(
      computation_status.status().ToString(),
      HasSubstr("Expected array argument for operand of concatenation"));
}

XLA_TEST_F(ConcatTest, ConcatSeveralBoxedPredicates) {
  XlaBuilder builder(TestName());
  auto p0 = ConstantR1<bool>(&builder, {true});
  auto p1 = ConstantR1<bool>(&builder, {false});
  auto p2 = ConstantR1<bool>(&builder, {true});
  ConcatInDim(&builder, {p0, p1, p2}, 0);

  bool expected[] = {true, false, true};
  ComputeAndCompareR1<bool>(&builder, expected, {});
}

XLA_TEST_F(ConcatTest, ConcatSeveralR1S32s) {
  XlaBuilder builder(TestName());
  auto a0 = ConstantR1<int32>(&builder, {1});
  auto a1 = ConstantR1<int32>(&builder, {2, 3});
  auto a2 = ConstantR1<int32>(&builder, {4, 5, 6});
  auto a3 = ConstantR1<int32>(&builder, {7, 8, 9, 10});
  ConcatInDim(&builder, {a0, a1, a2, a3}, 0);

  std::vector<int32> expected(10);
  std::iota(expected.begin(), expected.end(), 1);
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

XLA_TEST_F(ConcatTest, ConcatR3WeirdDims) {
  XlaBuilder builder(TestName());

  Array3D<float> arr0(9, 17, 1);
  arr0.Fill(1);

  Array3D<float> arr1(9, 17, 256);
  arr1.Fill(2);

  Array3D<float> expected(9, 17, arr0.n3() + arr1.n3());
  for (int64 i = 0; i < expected.n1(); ++i) {
    for (int64 j = 0; j < expected.n2(); ++j) {
      int64 kk = 0;
      for (const Array3D<float>& arr : {arr0, arr1}) {
        for (int64 k = 0; k < arr.n3(); ++k, ++kk) {
          expected(i, j, kk) = arr(i, j, k);
        }
      }
    }
  }

  XlaOp h0;
  auto p0 = CreateR3Parameter<float>(arr0, /*parameter_number=*/0, "p0",
                                     &builder, &h0);
  XlaOp h1;
  auto p1 = CreateR3Parameter<float>(arr1, /*parameter_number=*/1, "p1",
                                     &builder, &h1);

  ConcatInDim(&builder, {h0, h1}, 2);

  ComputeAndCompareR3<float>(&builder, expected, {p0.get(), p1.get()});
}

// Describes a binary rank-2 concatenation test.
struct R2BinarySpec {
  int64 lhs_dim0;
  int64 lhs_dim1;
  int64 rhs_dim0;
  int64 rhs_dim1;
  int64 concat_dimension;
};

// TEST_P harness for binary rank-2 concatenation.
class ConcatR2BinaryTest : public ClientLibraryTestBase,
                           public ::testing::WithParamInterface<R2BinarySpec> {
};

TEST_P(ConcatR2BinaryTest, DoIt) {
  const R2BinarySpec& spec = GetParam();
  Array2D<int32> lhs(spec.lhs_dim0, spec.lhs_dim1);
  lhs.FillUnique();
  Array2D<int32> rhs(spec.rhs_dim0, spec.rhs_dim1);
  rhs.FillUnique(1000);

  XlaBuilder builder(TestName());
  auto a0 = ConstantR2FromArray2D<int32>(&builder, lhs);
  auto a1 = ConstantR2FromArray2D<int32>(&builder, rhs);
  ConcatInDim(&builder, {a0, a1}, spec.concat_dimension);

  std::unique_ptr<Array2D<int32>> expected =
      ReferenceUtil::Concat2D(lhs, rhs, spec.concat_dimension);
  ComputeAndCompareR2<int32>(&builder, *expected, {});
}

// Regression test for b/31944287. x*y is used (at the same index) by all
// operands of the concat. We should emit x*y in three incoming basic blocks of
// the concat because these basic blocks are not control-equivalent.
//
//      x*y
//    /  |   \
// add1 add2 add3
//    \  |   /
//     concat
XLA_TEST_F(ConcatTest, ConcatOperandsOfSameOperand) {
  auto f32_scalar = ShapeUtil::MakeShape(xla::F32, {});
  auto x_literal = LiteralUtil::CreateR0<float>(2.f);
  auto y_literal = LiteralUtil::CreateR0<float>(3.f);
  auto x_data = client_->TransferToServer(*x_literal).ConsumeValueOrDie();
  auto y_data = client_->TransferToServer(*y_literal).ConsumeValueOrDie();

  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, f32_scalar, "x");
  auto y = Parameter(&builder, 1, f32_scalar, "y");
  auto mul = Mul(x, y);
  auto add1 = Add(mul, ConstantR1<float>(&builder, {1.f, 2.f}));
  auto add2 = Add(mul, ConstantR1<float>(&builder, {3.f, 4.f}));
  auto add3 = Add(mul, ConstantR1<float>(&builder, {5.f, 6.f}));
  ConcatInDim(&builder, {add1, add2, add3}, /*dimension=*/0);

  ComputeAndCompareR1<float>(&builder, {7., 8., 9., 10., 11., 12.},
                             {x_data.get(), y_data.get()}, ErrorSpec(1e-4));
}

// Test that the HLO optimization to replace a concat of a bradcasted scalar
// produces the correct result in rank 1.
XLA_TEST_F(ConcatTest, ConcatBroadcastArgument) {
  auto f32_scalar = ShapeUtil::MakeShape(xla::F32, {});
  auto x_literal = LiteralUtil::CreateR1<float>({2.0f, 3.0f, 5.0f, 6.0f});
  auto y_literal = LiteralUtil::CreateR0<float>(1.5f);
  auto z_literal = LiteralUtil::CreateR0<float>(5.5f);
  auto x_data = client_->TransferToServer(*x_literal).ConsumeValueOrDie();
  auto y_data = client_->TransferToServer(*y_literal).ConsumeValueOrDie();
  auto z_data = client_->TransferToServer(*z_literal).ConsumeValueOrDie();

  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, x_literal->shape(), "x");
  auto y = Parameter(&builder, 1, f32_scalar, "y");
  auto z = Parameter(&builder, 2, f32_scalar, "z");
  auto bcast = Broadcast(y, {5});
  auto bcast2 = Broadcast(z, {3});
  auto concat = ConcatInDim(&builder, {bcast, x}, /*dimension=*/0);
  ConcatInDim(&builder, {concat, bcast2}, /*dimension=*/0);

  ComputeAndCompareR1<float>(
      &builder,
      {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 2.0f, 3.0f, 5.0f, 6.0f, 5.5f, 5.5f, 5.5f},
      {x_data.get(), y_data.get(), z_data.get()}, ErrorSpec(1e-4));
}

// Test that the HLO optimization to replace a concat of a bradcasted scalar
// produces the correct result in rank 3 with both high and low padding in
// different dimensions.
XLA_TEST_F(ConcatTest, ConcatBroadcastArgumentR3) {
  auto f32_scalar = ShapeUtil::MakeShape(xla::F32, {});
  Array3D<float> x3d(3, 5, 7, 3.14f);
  auto x_literal = LiteralUtil::CreateR3FromArray3D<float>(x3d);
  auto y_literal = LiteralUtil::CreateR0<float>(1.5f);
  auto z_literal = LiteralUtil::CreateR0<float>(5.5f);
  auto x_data = client_->TransferToServer(*x_literal).ConsumeValueOrDie();
  auto y_data = client_->TransferToServer(*y_literal).ConsumeValueOrDie();
  auto z_data = client_->TransferToServer(*z_literal).ConsumeValueOrDie();

  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, x_literal->shape(), "x");
  auto y = Parameter(&builder, 1, f32_scalar, "y");
  auto z = Parameter(&builder, 2, f32_scalar, "y");
  auto y_bcast = Broadcast(y, {1, 5, 7});
  auto z_bcast = Broadcast(z, {4, 1, 7});
  auto concat = ConcatInDim(&builder, {y_bcast, x}, /*dimension=*/0);
  ConcatInDim(&builder, {concat, z_bcast}, /*dimension=*/1);
  Array3D<float> y_bcast3d(1, 5, 7, 1.5f);
  Array3D<float> z_bcast3d(4, 1, 7, 5.5f);
  auto concat0 = ReferenceUtil::Concat3D(y_bcast3d, x3d, 0);
  auto concat1 = ReferenceUtil::Concat3D(*concat0, z_bcast3d, 1);

  ComputeAndCompareR3<float>(&builder, *concat1,
                             {x_data.get(), y_data.get(), z_data.get()},
                             ErrorSpec(1e-4));
}

INSTANTIATE_TEST_CASE_P(ConcatR2BinaryTestInstantiation, ConcatR2BinaryTest,
                        ::testing::Values(R2BinarySpec{1, 1, 1, 1, 0},
                                          R2BinarySpec{1, 1, 1, 1, 1},
                                          R2BinarySpec{4, 3, 4, 3, 0},
                                          R2BinarySpec{4, 3, 4, 3, 1},
                                          R2BinarySpec{7, 128, 1, 128, 0},
                                          R2BinarySpec{8, 127, 8, 1, 1}));

}  // namespace
}  // namespace xla
