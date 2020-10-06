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
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

using ConcatTest = ClientLibraryTestBase;
using ConcatTestHlo = HloTestBase;
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

XLA_TEST_F(ConcatTest, ConcatDeeplyNested) {
  XlaBuilder builder(TestName());
  auto a_literal = LiteralUtil::CreateR1<float>({256.0});
  auto a = Parameter(&builder, 0, a_literal.shape(), "x");
  auto b = ConcatInDim(&builder, {a, a}, 0);
  auto c = ConcatInDim(&builder, {b, b}, 0);
  auto d = ConcatInDim(&builder, {c, c}, 0);
  auto e = ConcatInDim(&builder, {d, d}, 0);
  auto f = ConcatInDim(&builder, {e, e}, 0);
  auto g = ConcatInDim(&builder, {f, f}, 0);
  auto h = ConcatInDim(&builder, {g, g}, 0);
  auto i = ConcatInDim(&builder, {h, h}, 0);
  auto j = ConcatInDim(&builder, {i, i}, 0);
  auto k = ConcatInDim(&builder, {j, j}, 0);
  auto l = ConcatInDim(&builder, {k, k}, 0);
  auto m = ConcatInDim(&builder, {l, l}, 0);
  auto n = ConcatInDim(&builder, {m, m}, 0);
  auto o = ConcatInDim(&builder, {n, n}, 0);
  auto p = ConcatInDim(&builder, {o, o}, 0);
  auto q = ConcatInDim(&builder, {p, p}, 0);
  ConcatInDim(&builder, {q, q}, 0);
  std::vector<float> expected(131072, 256.0);
  auto a_data = client_->TransferToServer(a_literal).ConsumeValueOrDie();
  ComputeAndCompareR1<float>(&builder, expected, {a_data.get()});
}

XLA_TEST_F(ConcatTestHlo, ConcatWithBitcast) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule jit_broken.874

primitive_computation_add.866 {
  parameter.867 = f32[] parameter(0)
  parameter.868 = f32[] parameter(1)
  ROOT add.869 = f32[] add(parameter.867, parameter.868)
}

ENTRY jit_broken.874 {
  constant.39 = pred[] constant(false)
  parameter.38 = f32[4,2]{1,0} parameter(0)
  broadcast.40 = f32[4,2,1]{2,1,0} broadcast(parameter.38), dimensions={0,1}
  broadcast.41 = f32[4,2,1]{2,1,0} broadcast(parameter.38), dimensions={0,1}
  concatenate.42 = f32[4,2,2]{2,1,0} concatenate(broadcast.40, broadcast.41), dimensions={2}
  constant.1 = s32[0]{0} constant({})
  constant.43 = s32[] constant(0)
  constant.44 = s32[] constant(0)
  compare.45 = pred[] compare(constant.43, constant.44), direction=LT, type=UNSIGNED
  constant.46 = s32[] constant(0)
  constant.47 = s32[] constant(2)
  add.48 = s32[] add(constant.46, constant.47)
  constant.49 = s32[] constant(0)
  select.50 = s32[] select(compare.45, add.48, constant.49)
  broadcast.51 = s32[1]{0} broadcast(select.50), dimensions={}
  concatenate.52 = s32[1]{0} concatenate(constant.1, broadcast.51), dimensions={0}
  gather.53 = f32[4,2]{1,0} gather(concatenate.42, concatenate.52), offset_dims={0,1}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1,2}
  broadcast.54 = f32[4,2]{1,0} broadcast(gather.53), dimensions={0,1}
  constant.4 = s32[0]{0} constant({})
  constant.90 = s32[] constant(1)
  constant.91 = s32[] constant(0)
  compare.92 = pred[] compare(constant.90, constant.91), direction=LT, type=UNSIGNED
  constant.93 = s32[] constant(1)
  constant.94 = s32[] constant(2)
  add.95 = s32[] add(constant.93, constant.94)
  constant.96 = s32[] constant(1)
  select.97 = s32[] select(compare.92, add.95, constant.96)
  broadcast.98 = s32[1]{0} broadcast(select.97), dimensions={}
  concatenate.99 = s32[1]{0} concatenate(constant.4, broadcast.98), dimensions={0}
  gather.100 = f32[4]{0} gather(broadcast.54, concatenate.99), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.101 = f32[4]{0} broadcast(gather.100), dimensions={0}
  constant.5 = s32[0]{0} constant({})
  constant.102 = s32[] constant(0)
  constant.103 = s32[] constant(0)
  compare.104 = pred[] compare(constant.102, constant.103), direction=LT, type=UNSIGNED
  constant.105 = s32[] constant(0)
  constant.106 = s32[] constant(2)
  add.107 = s32[] add(constant.105, constant.106)
  constant.108 = s32[] constant(0)
  select.109 = s32[] select(compare.104, add.107, constant.108)
  broadcast.110 = s32[1]{0} broadcast(select.109), dimensions={}
  concatenate.111 = s32[1]{0} concatenate(constant.5, broadcast.110), dimensions={0}
  constant.112 = s32[] constant(1)
  constant.113 = s32[] constant(0)
  compare.114 = pred[] compare(constant.112, constant.113), direction=LT, type=UNSIGNED
  constant.115 = s32[] constant(1)
  constant.116 = s32[] constant(2)
  add.117 = s32[] add(constant.115, constant.116)
  constant.118 = s32[] constant(1)
  select.119 = s32[] select(compare.114, add.117, constant.118)
  broadcast.120 = s32[1]{0} broadcast(select.119), dimensions={}
  concatenate.121 = s32[2]{0} concatenate(concatenate.111, broadcast.120), dimensions={0}
  gather.122 = f32[4]{0} gather(concatenate.42, concatenate.121), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.123 = f32[4]{0} broadcast(gather.122), dimensions={0}
  add.124 = f32[4]{0} add(broadcast.101, broadcast.123)
  constant.2 = s32[0]{0} constant({})
  constant.55 = s32[] constant(0)
  constant.56 = s32[] constant(0)
  compare.57 = pred[] compare(constant.55, constant.56), direction=LT, type=UNSIGNED
  constant.58 = s32[] constant(0)
  constant.59 = s32[] constant(2)
  add.60 = s32[] add(constant.58, constant.59)
  constant.61 = s32[] constant(0)
  select.62 = s32[] select(compare.57, add.60, constant.61)
  broadcast.63 = s32[1]{0} broadcast(select.62), dimensions={}
  concatenate.64 = s32[1]{0} concatenate(constant.2, broadcast.63), dimensions={0}
  gather.65 = f32[4]{0} gather(broadcast.54, concatenate.64), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.66 = f32[4]{0} broadcast(gather.65), dimensions={0}
  constant.3 = s32[0]{0} constant({})
  constant.67 = s32[] constant(0)
  constant.68 = s32[] constant(0)
  compare.69 = pred[] compare(constant.67, constant.68), direction=LT, type=UNSIGNED
  constant.70 = s32[] constant(0)
  constant.71 = s32[] constant(2)
  add.72 = s32[] add(constant.70, constant.71)
  constant.73 = s32[] constant(0)
  select.74 = s32[] select(compare.69, add.72, constant.73)
  broadcast.75 = s32[1]{0} broadcast(select.74), dimensions={}
  concatenate.76 = s32[1]{0} concatenate(constant.3, broadcast.75), dimensions={0}
  constant.77 = s32[] constant(0)
  constant.78 = s32[] constant(0)
  compare.79 = pred[] compare(constant.77, constant.78), direction=LT, type=UNSIGNED
  constant.80 = s32[] constant(0)
  constant.81 = s32[] constant(2)
  add.82 = s32[] add(constant.80, constant.81)
  constant.83 = s32[] constant(0)
  select.84 = s32[] select(compare.79, add.82, constant.83)
  broadcast.85 = s32[1]{0} broadcast(select.84), dimensions={}
  concatenate.86 = s32[2]{0} concatenate(concatenate.76, broadcast.85), dimensions={0}
  gather.87 = f32[4]{0} gather(concatenate.42, concatenate.86), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.88 = f32[4]{0} broadcast(gather.87), dimensions={0}
  add.89 = f32[4]{0} add(broadcast.66, broadcast.88)
  subtract.126 = f32[4]{0} subtract(add.124, add.89)
  is-finite.127 = pred[4]{0} is-finite(subtract.126)
  not.128 = pred[4]{0} not(is-finite.127)
  abs.129 = f32[4]{0} abs(subtract.126)
  constant.130 = f32[] constant(inf)
  broadcast.131 = f32[4]{0} broadcast(constant.130), dimensions={}
  compare.132 = pred[4]{0} compare(abs.129, broadcast.131), direction=EQ, type=UNSIGNED
  not.133 = pred[4]{0} not(compare.132)
  and.134 = pred[4]{0} and(not.128, not.133)
  add.135 = f32[4]{0} add(add.124, add.89)
  maximum.125 = f32[4]{0} maximum(add.124, add.89)
  abs.136 = f32[4]{0} abs(subtract.126)
  negate.137 = f32[4]{0} negate(abs.136)
  exponential.138 = f32[4]{0} exponential(negate.137)
  log-plus-one.139 = f32[4]{0} log-plus-one(exponential.138)
  add.140 = f32[4]{0} add(maximum.125, log-plus-one.139)
  select.141 = f32[4]{0} select(and.134, add.135, add.140)
  broadcast.142 = f32[4,1]{1,0} broadcast(select.141), dimensions={0}
  broadcast.143 = f32[4,1]{1,0} broadcast(select.141), dimensions={0}
  concatenate.144 = f32[4,2]{1,0} concatenate(broadcast.142, broadcast.143), dimensions={1}
  constant.8 = s32[0]{0} constant({})
  constant.180 = s32[] constant(1)
  constant.181 = s32[] constant(0)
  compare.182 = pred[] compare(constant.180, constant.181), direction=LT, type=UNSIGNED
  constant.183 = s32[] constant(1)
  constant.184 = s32[] constant(2)
  add.185 = s32[] add(constant.183, constant.184)
  constant.186 = s32[] constant(1)
  select.187 = s32[] select(compare.182, add.185, constant.186)
  broadcast.188 = s32[1]{0} broadcast(select.187), dimensions={}
  concatenate.189 = s32[1]{0} concatenate(constant.8, broadcast.188), dimensions={0}
  gather.190 = f32[4]{0} gather(concatenate.144, concatenate.189), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.191 = f32[4]{0} broadcast(gather.190), dimensions={0}
  constant.9 = s32[0]{0} constant({})
  constant.192 = s32[] constant(0)
  constant.193 = s32[] constant(0)
  compare.194 = pred[] compare(constant.192, constant.193), direction=LT, type=UNSIGNED
  constant.195 = s32[] constant(0)
  constant.196 = s32[] constant(2)
  add.197 = s32[] add(constant.195, constant.196)
  constant.198 = s32[] constant(0)
  select.199 = s32[] select(compare.194, add.197, constant.198)
  broadcast.200 = s32[1]{0} broadcast(select.199), dimensions={}
  concatenate.201 = s32[1]{0} concatenate(constant.9, broadcast.200), dimensions={0}
  constant.202 = s32[] constant(1)
  constant.203 = s32[] constant(0)
  compare.204 = pred[] compare(constant.202, constant.203), direction=LT, type=UNSIGNED
  constant.205 = s32[] constant(1)
  constant.206 = s32[] constant(2)
  add.207 = s32[] add(constant.205, constant.206)
  constant.208 = s32[] constant(1)
  select.209 = s32[] select(compare.204, add.207, constant.208)
  broadcast.210 = s32[1]{0} broadcast(select.209), dimensions={}
  concatenate.211 = s32[2]{0} concatenate(concatenate.201, broadcast.210), dimensions={0}
  gather.212 = f32[4]{0} gather(concatenate.42, concatenate.211), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.213 = f32[4]{0} broadcast(gather.212), dimensions={0}
  add.214 = f32[4]{0} add(broadcast.191, broadcast.213)
  constant.6 = s32[0]{0} constant({})
  constant.145 = s32[] constant(0)
  constant.146 = s32[] constant(0)
  compare.147 = pred[] compare(constant.145, constant.146), direction=LT, type=UNSIGNED
  constant.148 = s32[] constant(0)
  constant.149 = s32[] constant(2)
  add.150 = s32[] add(constant.148, constant.149)
  constant.151 = s32[] constant(0)
  select.152 = s32[] select(compare.147, add.150, constant.151)
  broadcast.153 = s32[1]{0} broadcast(select.152), dimensions={}
  concatenate.154 = s32[1]{0} concatenate(constant.6, broadcast.153), dimensions={0}
  gather.155 = f32[4]{0} gather(concatenate.144, concatenate.154), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.156 = f32[4]{0} broadcast(gather.155), dimensions={0}
  constant.7 = s32[0]{0} constant({})
  constant.157 = s32[] constant(0)
  constant.158 = s32[] constant(0)
  compare.159 = pred[] compare(constant.157, constant.158), direction=LT, type=UNSIGNED
  constant.160 = s32[] constant(0)
  constant.161 = s32[] constant(2)
  add.162 = s32[] add(constant.160, constant.161)
  constant.163 = s32[] constant(0)
  select.164 = s32[] select(compare.159, add.162, constant.163)
  broadcast.165 = s32[1]{0} broadcast(select.164), dimensions={}
  concatenate.166 = s32[1]{0} concatenate(constant.7, broadcast.165), dimensions={0}
  constant.167 = s32[] constant(0)
  constant.168 = s32[] constant(0)
  compare.169 = pred[] compare(constant.167, constant.168), direction=LT, type=UNSIGNED
  constant.170 = s32[] constant(0)
  constant.171 = s32[] constant(2)
  add.172 = s32[] add(constant.170, constant.171)
  constant.173 = s32[] constant(0)
  select.174 = s32[] select(compare.169, add.172, constant.173)
  broadcast.175 = s32[1]{0} broadcast(select.174), dimensions={}
  concatenate.176 = s32[2]{0} concatenate(concatenate.166, broadcast.175), dimensions={0}
  gather.177 = f32[4]{0} gather(concatenate.42, concatenate.176), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.178 = f32[4]{0} broadcast(gather.177), dimensions={0}
  add.179 = f32[4]{0} add(broadcast.156, broadcast.178)
  subtract.216 = f32[4]{0} subtract(add.214, add.179)
  is-finite.217 = pred[4]{0} is-finite(subtract.216)
  not.218 = pred[4]{0} not(is-finite.217)
  abs.219 = f32[4]{0} abs(subtract.216)
  constant.220 = f32[] constant(inf)
  broadcast.221 = f32[4]{0} broadcast(constant.220), dimensions={}
  compare.222 = pred[4]{0} compare(abs.219, broadcast.221), direction=EQ, type=UNSIGNED
  not.223 = pred[4]{0} not(compare.222)
  and.224 = pred[4]{0} and(not.218, not.223)
  add.225 = f32[4]{0} add(add.214, add.179)
  maximum.215 = f32[4]{0} maximum(add.214, add.179)
  abs.226 = f32[4]{0} abs(subtract.216)
  negate.227 = f32[4]{0} negate(abs.226)
  exponential.228 = f32[4]{0} exponential(negate.227)
  log-plus-one.229 = f32[4]{0} log-plus-one(exponential.228)
  add.230 = f32[4]{0} add(maximum.215, log-plus-one.229)
  select.231 = f32[4]{0} select(and.224, add.225, add.230)
  broadcast.232 = f32[4,1]{1,0} broadcast(select.231), dimensions={0}
  broadcast.233 = f32[4,1]{1,0} broadcast(select.231), dimensions={0}
  concatenate.234 = f32[4,2]{1,0} concatenate(broadcast.232, broadcast.233), dimensions={1}
  constant.12 = s32[0]{0} constant({})
  constant.270 = s32[] constant(1)
  constant.271 = s32[] constant(0)
  compare.272 = pred[] compare(constant.270, constant.271), direction=LT, type=UNSIGNED
  constant.273 = s32[] constant(1)
  constant.274 = s32[] constant(2)
  add.275 = s32[] add(constant.273, constant.274)
  constant.276 = s32[] constant(1)
  select.277 = s32[] select(compare.272, add.275, constant.276)
  broadcast.278 = s32[1]{0} broadcast(select.277), dimensions={}
  concatenate.279 = s32[1]{0} concatenate(constant.12, broadcast.278), dimensions={0}
  gather.280 = f32[4]{0} gather(concatenate.234, concatenate.279), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.281 = f32[4]{0} broadcast(gather.280), dimensions={0}
  constant.13 = s32[0]{0} constant({})
  constant.282 = s32[] constant(0)
  constant.283 = s32[] constant(0)
  compare.284 = pred[] compare(constant.282, constant.283), direction=LT, type=UNSIGNED
  constant.285 = s32[] constant(0)
  constant.286 = s32[] constant(2)
  add.287 = s32[] add(constant.285, constant.286)
  constant.288 = s32[] constant(0)
  select.289 = s32[] select(compare.284, add.287, constant.288)
  broadcast.290 = s32[1]{0} broadcast(select.289), dimensions={}
  concatenate.291 = s32[1]{0} concatenate(constant.13, broadcast.290), dimensions={0}
  constant.292 = s32[] constant(1)
  constant.293 = s32[] constant(0)
  compare.294 = pred[] compare(constant.292, constant.293), direction=LT, type=UNSIGNED
  constant.295 = s32[] constant(1)
  constant.296 = s32[] constant(2)
  add.297 = s32[] add(constant.295, constant.296)
  constant.298 = s32[] constant(1)
  select.299 = s32[] select(compare.294, add.297, constant.298)
  broadcast.300 = s32[1]{0} broadcast(select.299), dimensions={}
  concatenate.301 = s32[2]{0} concatenate(concatenate.291, broadcast.300), dimensions={0}
  gather.302 = f32[4]{0} gather(concatenate.42, concatenate.301), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.303 = f32[4]{0} broadcast(gather.302), dimensions={0}
  add.304 = f32[4]{0} add(broadcast.281, broadcast.303)
  constant.10 = s32[0]{0} constant({})
  constant.235 = s32[] constant(0)
  constant.236 = s32[] constant(0)
  compare.237 = pred[] compare(constant.235, constant.236), direction=LT, type=UNSIGNED
  constant.238 = s32[] constant(0)
  constant.239 = s32[] constant(2)
  add.240 = s32[] add(constant.238, constant.239)
  constant.241 = s32[] constant(0)
  select.242 = s32[] select(compare.237, add.240, constant.241)
  broadcast.243 = s32[1]{0} broadcast(select.242), dimensions={}
  concatenate.244 = s32[1]{0} concatenate(constant.10, broadcast.243), dimensions={0}
  gather.245 = f32[4]{0} gather(concatenate.234, concatenate.244), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.246 = f32[4]{0} broadcast(gather.245), dimensions={0}
  constant.11 = s32[0]{0} constant({})
  constant.247 = s32[] constant(0)
  constant.248 = s32[] constant(0)
  compare.249 = pred[] compare(constant.247, constant.248), direction=LT, type=UNSIGNED
  constant.250 = s32[] constant(0)
  constant.251 = s32[] constant(2)
  add.252 = s32[] add(constant.250, constant.251)
  constant.253 = s32[] constant(0)
  select.254 = s32[] select(compare.249, add.252, constant.253)
  broadcast.255 = s32[1]{0} broadcast(select.254), dimensions={}
  concatenate.256 = s32[1]{0} concatenate(constant.11, broadcast.255), dimensions={0}
  constant.257 = s32[] constant(0)
  constant.258 = s32[] constant(0)
  compare.259 = pred[] compare(constant.257, constant.258), direction=LT, type=UNSIGNED
  constant.260 = s32[] constant(0)
  constant.261 = s32[] constant(2)
  add.262 = s32[] add(constant.260, constant.261)
  constant.263 = s32[] constant(0)
  select.264 = s32[] select(compare.259, add.262, constant.263)
  broadcast.265 = s32[1]{0} broadcast(select.264), dimensions={}
  concatenate.266 = s32[2]{0} concatenate(concatenate.256, broadcast.265), dimensions={0}
  gather.267 = f32[4]{0} gather(concatenate.42, concatenate.266), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.268 = f32[4]{0} broadcast(gather.267), dimensions={0}
  add.269 = f32[4]{0} add(broadcast.246, broadcast.268)
  subtract.306 = f32[4]{0} subtract(add.304, add.269)
  is-finite.307 = pred[4]{0} is-finite(subtract.306)
  not.308 = pred[4]{0} not(is-finite.307)
  abs.309 = f32[4]{0} abs(subtract.306)
  constant.310 = f32[] constant(inf)
  broadcast.311 = f32[4]{0} broadcast(constant.310), dimensions={}
  compare.312 = pred[4]{0} compare(abs.309, broadcast.311), direction=EQ, type=UNSIGNED
  not.313 = pred[4]{0} not(compare.312)
  and.314 = pred[4]{0} and(not.308, not.313)
  add.315 = f32[4]{0} add(add.304, add.269)
  maximum.305 = f32[4]{0} maximum(add.304, add.269)
  abs.316 = f32[4]{0} abs(subtract.306)
  negate.317 = f32[4]{0} negate(abs.316)
  exponential.318 = f32[4]{0} exponential(negate.317)
  log-plus-one.319 = f32[4]{0} log-plus-one(exponential.318)
  add.320 = f32[4]{0} add(maximum.305, log-plus-one.319)
  select.321 = f32[4]{0} select(and.314, add.315, add.320)
  broadcast.322 = f32[4,1]{1,0} broadcast(select.321), dimensions={0}
  broadcast.323 = f32[4,1]{1,0} broadcast(select.321), dimensions={0}
  concatenate.324 = f32[4,2]{1,0} concatenate(broadcast.322, broadcast.323), dimensions={1}
  constant.16 = s32[0]{0} constant({})
  constant.360 = s32[] constant(1)
  constant.361 = s32[] constant(0)
  compare.362 = pred[] compare(constant.360, constant.361), direction=LT, type=UNSIGNED
  constant.363 = s32[] constant(1)
  constant.364 = s32[] constant(2)
  add.365 = s32[] add(constant.363, constant.364)
  constant.366 = s32[] constant(1)
  select.367 = s32[] select(compare.362, add.365, constant.366)
  broadcast.368 = s32[1]{0} broadcast(select.367), dimensions={}
  concatenate.369 = s32[1]{0} concatenate(constant.16, broadcast.368), dimensions={0}
  gather.370 = f32[4]{0} gather(concatenate.324, concatenate.369), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.371 = f32[4]{0} broadcast(gather.370), dimensions={0}
  constant.17 = s32[0]{0} constant({})
  constant.372 = s32[] constant(0)
  constant.373 = s32[] constant(0)
  compare.374 = pred[] compare(constant.372, constant.373), direction=LT, type=UNSIGNED
  constant.375 = s32[] constant(0)
  constant.376 = s32[] constant(2)
  add.377 = s32[] add(constant.375, constant.376)
  constant.378 = s32[] constant(0)
  select.379 = s32[] select(compare.374, add.377, constant.378)
  broadcast.380 = s32[1]{0} broadcast(select.379), dimensions={}
  concatenate.381 = s32[1]{0} concatenate(constant.17, broadcast.380), dimensions={0}
  constant.382 = s32[] constant(1)
  constant.383 = s32[] constant(0)
  compare.384 = pred[] compare(constant.382, constant.383), direction=LT, type=UNSIGNED
  constant.385 = s32[] constant(1)
  constant.386 = s32[] constant(2)
  add.387 = s32[] add(constant.385, constant.386)
  constant.388 = s32[] constant(1)
  select.389 = s32[] select(compare.384, add.387, constant.388)
  broadcast.390 = s32[1]{0} broadcast(select.389), dimensions={}
  concatenate.391 = s32[2]{0} concatenate(concatenate.381, broadcast.390), dimensions={0}
  gather.392 = f32[4]{0} gather(concatenate.42, concatenate.391), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.393 = f32[4]{0} broadcast(gather.392), dimensions={0}
  add.394 = f32[4]{0} add(broadcast.371, broadcast.393)
  constant.14 = s32[0]{0} constant({})
  constant.325 = s32[] constant(0)
  constant.326 = s32[] constant(0)
  compare.327 = pred[] compare(constant.325, constant.326), direction=LT, type=UNSIGNED
  constant.328 = s32[] constant(0)
  constant.329 = s32[] constant(2)
  add.330 = s32[] add(constant.328, constant.329)
  constant.331 = s32[] constant(0)
  select.332 = s32[] select(compare.327, add.330, constant.331)
  broadcast.333 = s32[1]{0} broadcast(select.332), dimensions={}
  concatenate.334 = s32[1]{0} concatenate(constant.14, broadcast.333), dimensions={0}
  gather.335 = f32[4]{0} gather(concatenate.324, concatenate.334), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.336 = f32[4]{0} broadcast(gather.335), dimensions={0}
  constant.15 = s32[0]{0} constant({})
  constant.337 = s32[] constant(0)
  constant.338 = s32[] constant(0)
  compare.339 = pred[] compare(constant.337, constant.338), direction=LT, type=UNSIGNED
  constant.340 = s32[] constant(0)
  constant.341 = s32[] constant(2)
  add.342 = s32[] add(constant.340, constant.341)
  constant.343 = s32[] constant(0)
  select.344 = s32[] select(compare.339, add.342, constant.343)
  broadcast.345 = s32[1]{0} broadcast(select.344), dimensions={}
  concatenate.346 = s32[1]{0} concatenate(constant.15, broadcast.345), dimensions={0}
  constant.347 = s32[] constant(0)
  constant.348 = s32[] constant(0)
  compare.349 = pred[] compare(constant.347, constant.348), direction=LT, type=UNSIGNED
  constant.350 = s32[] constant(0)
  constant.351 = s32[] constant(2)
  add.352 = s32[] add(constant.350, constant.351)
  constant.353 = s32[] constant(0)
  select.354 = s32[] select(compare.349, add.352, constant.353)
  broadcast.355 = s32[1]{0} broadcast(select.354), dimensions={}
  concatenate.356 = s32[2]{0} concatenate(concatenate.346, broadcast.355), dimensions={0}
  gather.357 = f32[4]{0} gather(concatenate.42, concatenate.356), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.358 = f32[4]{0} broadcast(gather.357), dimensions={0}
  add.359 = f32[4]{0} add(broadcast.336, broadcast.358)
  subtract.396 = f32[4]{0} subtract(add.394, add.359)
  is-finite.397 = pred[4]{0} is-finite(subtract.396)
  not.398 = pred[4]{0} not(is-finite.397)
  abs.399 = f32[4]{0} abs(subtract.396)
  constant.400 = f32[] constant(inf)
  broadcast.401 = f32[4]{0} broadcast(constant.400), dimensions={}
  compare.402 = pred[4]{0} compare(abs.399, broadcast.401), direction=EQ, type=UNSIGNED
  not.403 = pred[4]{0} not(compare.402)
  and.404 = pred[4]{0} and(not.398, not.403)
  add.405 = f32[4]{0} add(add.394, add.359)
  maximum.395 = f32[4]{0} maximum(add.394, add.359)
  abs.406 = f32[4]{0} abs(subtract.396)
  negate.407 = f32[4]{0} negate(abs.406)
  exponential.408 = f32[4]{0} exponential(negate.407)
  log-plus-one.409 = f32[4]{0} log-plus-one(exponential.408)
  add.410 = f32[4]{0} add(maximum.395, log-plus-one.409)
  select.411 = f32[4]{0} select(and.404, add.405, add.410)
  broadcast.412 = f32[4,1]{1,0} broadcast(select.411), dimensions={0}
  broadcast.413 = f32[4,1]{1,0} broadcast(select.411), dimensions={0}
  concatenate.414 = f32[4,2]{1,0} concatenate(broadcast.412, broadcast.413), dimensions={1}
  constant.20 = s32[0]{0} constant({})
  constant.450 = s32[] constant(1)
  constant.451 = s32[] constant(0)
  compare.452 = pred[] compare(constant.450, constant.451), direction=LT, type=UNSIGNED
  constant.453 = s32[] constant(1)
  constant.454 = s32[] constant(2)
  add.455 = s32[] add(constant.453, constant.454)
  constant.456 = s32[] constant(1)
  select.457 = s32[] select(compare.452, add.455, constant.456)
  broadcast.458 = s32[1]{0} broadcast(select.457), dimensions={}
  concatenate.459 = s32[1]{0} concatenate(constant.20, broadcast.458), dimensions={0}
  gather.460 = f32[4]{0} gather(concatenate.414, concatenate.459), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.461 = f32[4]{0} broadcast(gather.460), dimensions={0}
  constant.21 = s32[0]{0} constant({})
  constant.462 = s32[] constant(0)
  constant.463 = s32[] constant(0)
  compare.464 = pred[] compare(constant.462, constant.463), direction=LT, type=UNSIGNED
  constant.465 = s32[] constant(0)
  constant.466 = s32[] constant(2)
  add.467 = s32[] add(constant.465, constant.466)
  constant.468 = s32[] constant(0)
  select.469 = s32[] select(compare.464, add.467, constant.468)
  broadcast.470 = s32[1]{0} broadcast(select.469), dimensions={}
  concatenate.471 = s32[1]{0} concatenate(constant.21, broadcast.470), dimensions={0}
  constant.472 = s32[] constant(1)
  constant.473 = s32[] constant(0)
  compare.474 = pred[] compare(constant.472, constant.473), direction=LT, type=UNSIGNED
  constant.475 = s32[] constant(1)
  constant.476 = s32[] constant(2)
  add.477 = s32[] add(constant.475, constant.476)
  constant.478 = s32[] constant(1)
  select.479 = s32[] select(compare.474, add.477, constant.478)
  broadcast.480 = s32[1]{0} broadcast(select.479), dimensions={}
  concatenate.481 = s32[2]{0} concatenate(concatenate.471, broadcast.480), dimensions={0}
  gather.482 = f32[4]{0} gather(concatenate.42, concatenate.481), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.483 = f32[4]{0} broadcast(gather.482), dimensions={0}
  add.484 = f32[4]{0} add(broadcast.461, broadcast.483)
  constant.18 = s32[0]{0} constant({})
  constant.415 = s32[] constant(0)
  constant.416 = s32[] constant(0)
  compare.417 = pred[] compare(constant.415, constant.416), direction=LT, type=UNSIGNED
  constant.418 = s32[] constant(0)
  constant.419 = s32[] constant(2)
  add.420 = s32[] add(constant.418, constant.419)
  constant.421 = s32[] constant(0)
  select.422 = s32[] select(compare.417, add.420, constant.421)
  broadcast.423 = s32[1]{0} broadcast(select.422), dimensions={}
  concatenate.424 = s32[1]{0} concatenate(constant.18, broadcast.423), dimensions={0}
  gather.425 = f32[4]{0} gather(concatenate.414, concatenate.424), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.426 = f32[4]{0} broadcast(gather.425), dimensions={0}
  constant.19 = s32[0]{0} constant({})
  constant.427 = s32[] constant(0)
  constant.428 = s32[] constant(0)
  compare.429 = pred[] compare(constant.427, constant.428), direction=LT, type=UNSIGNED
  constant.430 = s32[] constant(0)
  constant.431 = s32[] constant(2)
  add.432 = s32[] add(constant.430, constant.431)
  constant.433 = s32[] constant(0)
  select.434 = s32[] select(compare.429, add.432, constant.433)
  broadcast.435 = s32[1]{0} broadcast(select.434), dimensions={}
  concatenate.436 = s32[1]{0} concatenate(constant.19, broadcast.435), dimensions={0}
  constant.437 = s32[] constant(0)
  constant.438 = s32[] constant(0)
  compare.439 = pred[] compare(constant.437, constant.438), direction=LT, type=UNSIGNED
  constant.440 = s32[] constant(0)
  constant.441 = s32[] constant(2)
  add.442 = s32[] add(constant.440, constant.441)
  constant.443 = s32[] constant(0)
  select.444 = s32[] select(compare.439, add.442, constant.443)
  broadcast.445 = s32[1]{0} broadcast(select.444), dimensions={}
  concatenate.446 = s32[2]{0} concatenate(concatenate.436, broadcast.445), dimensions={0}
  gather.447 = f32[4]{0} gather(concatenate.42, concatenate.446), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.448 = f32[4]{0} broadcast(gather.447), dimensions={0}
  add.449 = f32[4]{0} add(broadcast.426, broadcast.448)
  subtract.486 = f32[4]{0} subtract(add.484, add.449)
  is-finite.487 = pred[4]{0} is-finite(subtract.486)
  not.488 = pred[4]{0} not(is-finite.487)
  abs.489 = f32[4]{0} abs(subtract.486)
  constant.490 = f32[] constant(inf)
  broadcast.491 = f32[4]{0} broadcast(constant.490), dimensions={}
  compare.492 = pred[4]{0} compare(abs.489, broadcast.491), direction=EQ, type=UNSIGNED
  not.493 = pred[4]{0} not(compare.492)
  and.494 = pred[4]{0} and(not.488, not.493)
  add.495 = f32[4]{0} add(add.484, add.449)
  maximum.485 = f32[4]{0} maximum(add.484, add.449)
  abs.496 = f32[4]{0} abs(subtract.486)
  negate.497 = f32[4]{0} negate(abs.496)
  exponential.498 = f32[4]{0} exponential(negate.497)
  log-plus-one.499 = f32[4]{0} log-plus-one(exponential.498)
  add.500 = f32[4]{0} add(maximum.485, log-plus-one.499)
  select.501 = f32[4]{0} select(and.494, add.495, add.500)
  broadcast.502 = f32[4,1]{1,0} broadcast(select.501), dimensions={0}
  broadcast.503 = f32[4,1]{1,0} broadcast(select.501), dimensions={0}
  concatenate.504 = f32[4,2]{1,0} concatenate(broadcast.502, broadcast.503), dimensions={1}
  constant.24 = s32[0]{0} constant({})
  constant.540 = s32[] constant(1)
  constant.541 = s32[] constant(0)
  compare.542 = pred[] compare(constant.540, constant.541), direction=LT, type=UNSIGNED
  constant.543 = s32[] constant(1)
  constant.544 = s32[] constant(2)
  add.545 = s32[] add(constant.543, constant.544)
  constant.546 = s32[] constant(1)
  select.547 = s32[] select(compare.542, add.545, constant.546)
  broadcast.548 = s32[1]{0} broadcast(select.547), dimensions={}
  concatenate.549 = s32[1]{0} concatenate(constant.24, broadcast.548), dimensions={0}
  gather.550 = f32[4]{0} gather(concatenate.504, concatenate.549), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.551 = f32[4]{0} broadcast(gather.550), dimensions={0}
  constant.25 = s32[0]{0} constant({})
  constant.552 = s32[] constant(0)
  constant.553 = s32[] constant(0)
  compare.554 = pred[] compare(constant.552, constant.553), direction=LT, type=UNSIGNED
  constant.555 = s32[] constant(0)
  constant.556 = s32[] constant(2)
  add.557 = s32[] add(constant.555, constant.556)
  constant.558 = s32[] constant(0)
  select.559 = s32[] select(compare.554, add.557, constant.558)
  broadcast.560 = s32[1]{0} broadcast(select.559), dimensions={}
  concatenate.561 = s32[1]{0} concatenate(constant.25, broadcast.560), dimensions={0}
  constant.562 = s32[] constant(1)
  constant.563 = s32[] constant(0)
  compare.564 = pred[] compare(constant.562, constant.563), direction=LT, type=UNSIGNED
  constant.565 = s32[] constant(1)
  constant.566 = s32[] constant(2)
  add.567 = s32[] add(constant.565, constant.566)
  constant.568 = s32[] constant(1)
  select.569 = s32[] select(compare.564, add.567, constant.568)
  broadcast.570 = s32[1]{0} broadcast(select.569), dimensions={}
  concatenate.571 = s32[2]{0} concatenate(concatenate.561, broadcast.570), dimensions={0}
  gather.572 = f32[4]{0} gather(concatenate.42, concatenate.571), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.573 = f32[4]{0} broadcast(gather.572), dimensions={0}
  add.574 = f32[4]{0} add(broadcast.551, broadcast.573)
  constant.22 = s32[0]{0} constant({})
  constant.505 = s32[] constant(0)
  constant.506 = s32[] constant(0)
  compare.507 = pred[] compare(constant.505, constant.506), direction=LT, type=UNSIGNED
  constant.508 = s32[] constant(0)
  constant.509 = s32[] constant(2)
  add.510 = s32[] add(constant.508, constant.509)
  constant.511 = s32[] constant(0)
  select.512 = s32[] select(compare.507, add.510, constant.511)
  broadcast.513 = s32[1]{0} broadcast(select.512), dimensions={}
  concatenate.514 = s32[1]{0} concatenate(constant.22, broadcast.513), dimensions={0}
  gather.515 = f32[4]{0} gather(concatenate.504, concatenate.514), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.516 = f32[4]{0} broadcast(gather.515), dimensions={0}
  constant.23 = s32[0]{0} constant({})
  constant.517 = s32[] constant(0)
  constant.518 = s32[] constant(0)
  compare.519 = pred[] compare(constant.517, constant.518), direction=LT, type=UNSIGNED
  constant.520 = s32[] constant(0)
  constant.521 = s32[] constant(2)
  add.522 = s32[] add(constant.520, constant.521)
  constant.523 = s32[] constant(0)
  select.524 = s32[] select(compare.519, add.522, constant.523)
  broadcast.525 = s32[1]{0} broadcast(select.524), dimensions={}
  concatenate.526 = s32[1]{0} concatenate(constant.23, broadcast.525), dimensions={0}
  constant.527 = s32[] constant(0)
  constant.528 = s32[] constant(0)
  compare.529 = pred[] compare(constant.527, constant.528), direction=LT, type=UNSIGNED
  constant.530 = s32[] constant(0)
  constant.531 = s32[] constant(2)
  add.532 = s32[] add(constant.530, constant.531)
  constant.533 = s32[] constant(0)
  select.534 = s32[] select(compare.529, add.532, constant.533)
  broadcast.535 = s32[1]{0} broadcast(select.534), dimensions={}
  concatenate.536 = s32[2]{0} concatenate(concatenate.526, broadcast.535), dimensions={0}
  gather.537 = f32[4]{0} gather(concatenate.42, concatenate.536), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.538 = f32[4]{0} broadcast(gather.537), dimensions={0}
  add.539 = f32[4]{0} add(broadcast.516, broadcast.538)
  subtract.576 = f32[4]{0} subtract(add.574, add.539)
  is-finite.577 = pred[4]{0} is-finite(subtract.576)
  not.578 = pred[4]{0} not(is-finite.577)
  abs.579 = f32[4]{0} abs(subtract.576)
  constant.580 = f32[] constant(inf)
  broadcast.581 = f32[4]{0} broadcast(constant.580), dimensions={}
  compare.582 = pred[4]{0} compare(abs.579, broadcast.581), direction=EQ, type=UNSIGNED
  not.583 = pred[4]{0} not(compare.582)
  and.584 = pred[4]{0} and(not.578, not.583)
  add.585 = f32[4]{0} add(add.574, add.539)
  maximum.575 = f32[4]{0} maximum(add.574, add.539)
  abs.586 = f32[4]{0} abs(subtract.576)
  negate.587 = f32[4]{0} negate(abs.586)
  exponential.588 = f32[4]{0} exponential(negate.587)
  log-plus-one.589 = f32[4]{0} log-plus-one(exponential.588)
  add.590 = f32[4]{0} add(maximum.575, log-plus-one.589)
  select.591 = f32[4]{0} select(and.584, add.585, add.590)
  broadcast.592 = f32[4,1]{1,0} broadcast(select.591), dimensions={0}
  broadcast.593 = f32[4,1]{1,0} broadcast(select.591), dimensions={0}
  concatenate.594 = f32[4,2]{1,0} concatenate(broadcast.592, broadcast.593), dimensions={1}
  constant.28 = s32[0]{0} constant({})
  constant.630 = s32[] constant(1)
  constant.631 = s32[] constant(0)
  compare.632 = pred[] compare(constant.630, constant.631), direction=LT, type=UNSIGNED
  constant.633 = s32[] constant(1)
  constant.634 = s32[] constant(2)
  add.635 = s32[] add(constant.633, constant.634)
  constant.636 = s32[] constant(1)
  select.637 = s32[] select(compare.632, add.635, constant.636)
  broadcast.638 = s32[1]{0} broadcast(select.637), dimensions={}
  concatenate.639 = s32[1]{0} concatenate(constant.28, broadcast.638), dimensions={0}
  gather.640 = f32[4]{0} gather(concatenate.594, concatenate.639), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.641 = f32[4]{0} broadcast(gather.640), dimensions={0}
  constant.29 = s32[0]{0} constant({})
  constant.642 = s32[] constant(0)
  constant.643 = s32[] constant(0)
  compare.644 = pred[] compare(constant.642, constant.643), direction=LT, type=UNSIGNED
  constant.645 = s32[] constant(0)
  constant.646 = s32[] constant(2)
  add.647 = s32[] add(constant.645, constant.646)
  constant.648 = s32[] constant(0)
  select.649 = s32[] select(compare.644, add.647, constant.648)
  broadcast.650 = s32[1]{0} broadcast(select.649), dimensions={}
  concatenate.651 = s32[1]{0} concatenate(constant.29, broadcast.650), dimensions={0}
  constant.652 = s32[] constant(1)
  constant.653 = s32[] constant(0)
  compare.654 = pred[] compare(constant.652, constant.653), direction=LT, type=UNSIGNED
  constant.655 = s32[] constant(1)
  constant.656 = s32[] constant(2)
  add.657 = s32[] add(constant.655, constant.656)
  constant.658 = s32[] constant(1)
  select.659 = s32[] select(compare.654, add.657, constant.658)
  broadcast.660 = s32[1]{0} broadcast(select.659), dimensions={}
  concatenate.661 = s32[2]{0} concatenate(concatenate.651, broadcast.660), dimensions={0}
  gather.662 = f32[4]{0} gather(concatenate.42, concatenate.661), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.663 = f32[4]{0} broadcast(gather.662), dimensions={0}
  add.664 = f32[4]{0} add(broadcast.641, broadcast.663)
  constant.26 = s32[0]{0} constant({})
  constant.595 = s32[] constant(0)
  constant.596 = s32[] constant(0)
  compare.597 = pred[] compare(constant.595, constant.596), direction=LT, type=UNSIGNED
  constant.598 = s32[] constant(0)
  constant.599 = s32[] constant(2)
  add.600 = s32[] add(constant.598, constant.599)
  constant.601 = s32[] constant(0)
  select.602 = s32[] select(compare.597, add.600, constant.601)
  broadcast.603 = s32[1]{0} broadcast(select.602), dimensions={}
  concatenate.604 = s32[1]{0} concatenate(constant.26, broadcast.603), dimensions={0}
  gather.605 = f32[4]{0} gather(concatenate.594, concatenate.604), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.606 = f32[4]{0} broadcast(gather.605), dimensions={0}
  constant.27 = s32[0]{0} constant({})
  constant.607 = s32[] constant(0)
  constant.608 = s32[] constant(0)
  compare.609 = pred[] compare(constant.607, constant.608), direction=LT, type=UNSIGNED
  constant.610 = s32[] constant(0)
  constant.611 = s32[] constant(2)
  add.612 = s32[] add(constant.610, constant.611)
  constant.613 = s32[] constant(0)
  select.614 = s32[] select(compare.609, add.612, constant.613)
  broadcast.615 = s32[1]{0} broadcast(select.614), dimensions={}
  concatenate.616 = s32[1]{0} concatenate(constant.27, broadcast.615), dimensions={0}
  constant.617 = s32[] constant(0)
  constant.618 = s32[] constant(0)
  compare.619 = pred[] compare(constant.617, constant.618), direction=LT, type=UNSIGNED
  constant.620 = s32[] constant(0)
  constant.621 = s32[] constant(2)
  add.622 = s32[] add(constant.620, constant.621)
  constant.623 = s32[] constant(0)
  select.624 = s32[] select(compare.619, add.622, constant.623)
  broadcast.625 = s32[1]{0} broadcast(select.624), dimensions={}
  concatenate.626 = s32[2]{0} concatenate(concatenate.616, broadcast.625), dimensions={0}
  gather.627 = f32[4]{0} gather(concatenate.42, concatenate.626), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.628 = f32[4]{0} broadcast(gather.627), dimensions={0}
  add.629 = f32[4]{0} add(broadcast.606, broadcast.628)
  subtract.666 = f32[4]{0} subtract(add.664, add.629)
  is-finite.667 = pred[4]{0} is-finite(subtract.666)
  not.668 = pred[4]{0} not(is-finite.667)
  abs.669 = f32[4]{0} abs(subtract.666)
  constant.670 = f32[] constant(inf)
  broadcast.671 = f32[4]{0} broadcast(constant.670), dimensions={}
  compare.672 = pred[4]{0} compare(abs.669, broadcast.671), direction=EQ, type=UNSIGNED
  not.673 = pred[4]{0} not(compare.672)
  and.674 = pred[4]{0} and(not.668, not.673)
  add.675 = f32[4]{0} add(add.664, add.629)
  maximum.665 = f32[4]{0} maximum(add.664, add.629)
  abs.676 = f32[4]{0} abs(subtract.666)
  negate.677 = f32[4]{0} negate(abs.676)
  exponential.678 = f32[4]{0} exponential(negate.677)
  log-plus-one.679 = f32[4]{0} log-plus-one(exponential.678)
  add.680 = f32[4]{0} add(maximum.665, log-plus-one.679)
  select.681 = f32[4]{0} select(and.674, add.675, add.680)
  broadcast.682 = f32[4,1]{1,0} broadcast(select.681), dimensions={0}
  broadcast.683 = f32[4,1]{1,0} broadcast(select.681), dimensions={0}
  concatenate.684 = f32[4,2]{1,0} concatenate(broadcast.682, broadcast.683), dimensions={1}
  constant.32 = s32[0]{0} constant({})
  constant.720 = s32[] constant(1)
  constant.721 = s32[] constant(0)
  compare.722 = pred[] compare(constant.720, constant.721), direction=LT, type=UNSIGNED
  constant.723 = s32[] constant(1)
  constant.724 = s32[] constant(2)
  add.725 = s32[] add(constant.723, constant.724)
  constant.726 = s32[] constant(1)
  select.727 = s32[] select(compare.722, add.725, constant.726)
  broadcast.728 = s32[1]{0} broadcast(select.727), dimensions={}
  concatenate.729 = s32[1]{0} concatenate(constant.32, broadcast.728), dimensions={0}
  gather.730 = f32[4]{0} gather(concatenate.684, concatenate.729), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.731 = f32[4]{0} broadcast(gather.730), dimensions={0}
  constant.33 = s32[0]{0} constant({})
  constant.732 = s32[] constant(0)
  constant.733 = s32[] constant(0)
  compare.734 = pred[] compare(constant.732, constant.733), direction=LT, type=UNSIGNED
  constant.735 = s32[] constant(0)
  constant.736 = s32[] constant(2)
  add.737 = s32[] add(constant.735, constant.736)
  constant.738 = s32[] constant(0)
  select.739 = s32[] select(compare.734, add.737, constant.738)
  broadcast.740 = s32[1]{0} broadcast(select.739), dimensions={}
  concatenate.741 = s32[1]{0} concatenate(constant.33, broadcast.740), dimensions={0}
  constant.742 = s32[] constant(1)
  constant.743 = s32[] constant(0)
  compare.744 = pred[] compare(constant.742, constant.743), direction=LT, type=UNSIGNED
  constant.745 = s32[] constant(1)
  constant.746 = s32[] constant(2)
  add.747 = s32[] add(constant.745, constant.746)
  constant.748 = s32[] constant(1)
  select.749 = s32[] select(compare.744, add.747, constant.748)
  broadcast.750 = s32[1]{0} broadcast(select.749), dimensions={}
  concatenate.751 = s32[2]{0} concatenate(concatenate.741, broadcast.750), dimensions={0}
  gather.752 = f32[4]{0} gather(concatenate.42, concatenate.751), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.753 = f32[4]{0} broadcast(gather.752), dimensions={0}
  add.754 = f32[4]{0} add(broadcast.731, broadcast.753)
  constant.30 = s32[0]{0} constant({})
  constant.685 = s32[] constant(0)
  constant.686 = s32[] constant(0)
  compare.687 = pred[] compare(constant.685, constant.686), direction=LT, type=UNSIGNED
  constant.688 = s32[] constant(0)
  constant.689 = s32[] constant(2)
  add.690 = s32[] add(constant.688, constant.689)
  constant.691 = s32[] constant(0)
  select.692 = s32[] select(compare.687, add.690, constant.691)
  broadcast.693 = s32[1]{0} broadcast(select.692), dimensions={}
  concatenate.694 = s32[1]{0} concatenate(constant.30, broadcast.693), dimensions={0}
  gather.695 = f32[4]{0} gather(concatenate.684, concatenate.694), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.696 = f32[4]{0} broadcast(gather.695), dimensions={0}
  constant.31 = s32[0]{0} constant({})
  constant.697 = s32[] constant(0)
  constant.698 = s32[] constant(0)
  compare.699 = pred[] compare(constant.697, constant.698), direction=LT, type=UNSIGNED
  constant.700 = s32[] constant(0)
  constant.701 = s32[] constant(2)
  add.702 = s32[] add(constant.700, constant.701)
  constant.703 = s32[] constant(0)
  select.704 = s32[] select(compare.699, add.702, constant.703)
  broadcast.705 = s32[1]{0} broadcast(select.704), dimensions={}
  concatenate.706 = s32[1]{0} concatenate(constant.31, broadcast.705), dimensions={0}
  constant.707 = s32[] constant(0)
  constant.708 = s32[] constant(0)
  compare.709 = pred[] compare(constant.707, constant.708), direction=LT, type=UNSIGNED
  constant.710 = s32[] constant(0)
  constant.711 = s32[] constant(2)
  add.712 = s32[] add(constant.710, constant.711)
  constant.713 = s32[] constant(0)
  select.714 = s32[] select(compare.709, add.712, constant.713)
  broadcast.715 = s32[1]{0} broadcast(select.714), dimensions={}
  concatenate.716 = s32[2]{0} concatenate(concatenate.706, broadcast.715), dimensions={0}
  gather.717 = f32[4]{0} gather(concatenate.42, concatenate.716), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.718 = f32[4]{0} broadcast(gather.717), dimensions={0}
  add.719 = f32[4]{0} add(broadcast.696, broadcast.718)
  subtract.756 = f32[4]{0} subtract(add.754, add.719)
  is-finite.757 = pred[4]{0} is-finite(subtract.756)
  not.758 = pred[4]{0} not(is-finite.757)
  abs.759 = f32[4]{0} abs(subtract.756)
  constant.760 = f32[] constant(inf)
  broadcast.761 = f32[4]{0} broadcast(constant.760), dimensions={}
  compare.762 = pred[4]{0} compare(abs.759, broadcast.761), direction=EQ, type=UNSIGNED
  not.763 = pred[4]{0} not(compare.762)
  and.764 = pred[4]{0} and(not.758, not.763)
  add.765 = f32[4]{0} add(add.754, add.719)
  maximum.755 = f32[4]{0} maximum(add.754, add.719)
  abs.766 = f32[4]{0} abs(subtract.756)
  negate.767 = f32[4]{0} negate(abs.766)
  exponential.768 = f32[4]{0} exponential(negate.767)
  log-plus-one.769 = f32[4]{0} log-plus-one(exponential.768)
  add.770 = f32[4]{0} add(maximum.755, log-plus-one.769)
  select.771 = f32[4]{0} select(and.764, add.765, add.770)
  broadcast.772 = f32[4,1]{1,0} broadcast(select.771), dimensions={0}
  broadcast.773 = f32[4,1]{1,0} broadcast(select.771), dimensions={0}
  concatenate.774 = f32[4,2]{1,0} concatenate(broadcast.772, broadcast.773), dimensions={1}
  constant.36 = s32[0]{0} constant({})
  constant.810 = s32[] constant(1)
  constant.811 = s32[] constant(0)
  compare.812 = pred[] compare(constant.810, constant.811), direction=LT, type=UNSIGNED
  constant.813 = s32[] constant(1)
  constant.814 = s32[] constant(2)
  add.815 = s32[] add(constant.813, constant.814)
  constant.816 = s32[] constant(1)
  select.817 = s32[] select(compare.812, add.815, constant.816)
  broadcast.818 = s32[1]{0} broadcast(select.817), dimensions={}
  concatenate.819 = s32[1]{0} concatenate(constant.36, broadcast.818), dimensions={0}
  gather.820 = f32[4]{0} gather(concatenate.774, concatenate.819), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.821 = f32[4]{0} broadcast(gather.820), dimensions={0}
  constant.37 = s32[0]{0} constant({})
  constant.822 = s32[] constant(0)
  constant.823 = s32[] constant(0)
  compare.824 = pred[] compare(constant.822, constant.823), direction=LT, type=UNSIGNED
  constant.825 = s32[] constant(0)
  constant.826 = s32[] constant(2)
  add.827 = s32[] add(constant.825, constant.826)
  constant.828 = s32[] constant(0)
  select.829 = s32[] select(compare.824, add.827, constant.828)
  broadcast.830 = s32[1]{0} broadcast(select.829), dimensions={}
  concatenate.831 = s32[1]{0} concatenate(constant.37, broadcast.830), dimensions={0}
  constant.832 = s32[] constant(1)
  constant.833 = s32[] constant(0)
  compare.834 = pred[] compare(constant.832, constant.833), direction=LT, type=UNSIGNED
  constant.835 = s32[] constant(1)
  constant.836 = s32[] constant(2)
  add.837 = s32[] add(constant.835, constant.836)
  constant.838 = s32[] constant(1)
  select.839 = s32[] select(compare.834, add.837, constant.838)
  broadcast.840 = s32[1]{0} broadcast(select.839), dimensions={}
  concatenate.841 = s32[2]{0} concatenate(concatenate.831, broadcast.840), dimensions={0}
  gather.842 = f32[4]{0} gather(concatenate.42, concatenate.841), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.843 = f32[4]{0} broadcast(gather.842), dimensions={0}
  add.844 = f32[4]{0} add(broadcast.821, broadcast.843)
  constant.34 = s32[0]{0} constant({})
  constant.775 = s32[] constant(0)
  constant.776 = s32[] constant(0)
  compare.777 = pred[] compare(constant.775, constant.776), direction=LT, type=UNSIGNED
  constant.778 = s32[] constant(0)
  constant.779 = s32[] constant(2)
  add.780 = s32[] add(constant.778, constant.779)
  constant.781 = s32[] constant(0)
  select.782 = s32[] select(compare.777, add.780, constant.781)
  broadcast.783 = s32[1]{0} broadcast(select.782), dimensions={}
  concatenate.784 = s32[1]{0} concatenate(constant.34, broadcast.783), dimensions={0}
  gather.785 = f32[4]{0} gather(concatenate.774, concatenate.784), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=0, slice_sizes={4,1}
  broadcast.786 = f32[4]{0} broadcast(gather.785), dimensions={0}
  constant.35 = s32[0]{0} constant({})
  constant.787 = s32[] constant(0)
  constant.788 = s32[] constant(0)
  compare.789 = pred[] compare(constant.787, constant.788), direction=LT, type=UNSIGNED
  constant.790 = s32[] constant(0)
  constant.791 = s32[] constant(2)
  add.792 = s32[] add(constant.790, constant.791)
  constant.793 = s32[] constant(0)
  select.794 = s32[] select(compare.789, add.792, constant.793)
  broadcast.795 = s32[1]{0} broadcast(select.794), dimensions={}
  concatenate.796 = s32[1]{0} concatenate(constant.35, broadcast.795), dimensions={0}
  constant.797 = s32[] constant(0)
  constant.798 = s32[] constant(0)
  compare.799 = pred[] compare(constant.797, constant.798), direction=LT, type=UNSIGNED
  constant.800 = s32[] constant(0)
  constant.801 = s32[] constant(2)
  add.802 = s32[] add(constant.800, constant.801)
  constant.803 = s32[] constant(0)
  select.804 = s32[] select(compare.799, add.802, constant.803)
  broadcast.805 = s32[1]{0} broadcast(select.804), dimensions={}
  concatenate.806 = s32[2]{0} concatenate(concatenate.796, broadcast.805), dimensions={0}
  gather.807 = f32[4]{0} gather(concatenate.42, concatenate.806), offset_dims={0}, collapsed_slice_dims={1,2}, start_index_map={1,2}, index_vector_dim=0, slice_sizes={4,1,1}
  broadcast.808 = f32[4]{0} broadcast(gather.807), dimensions={0}
  add.809 = f32[4]{0} add(broadcast.786, broadcast.808)
  subtract.846 = f32[4]{0} subtract(add.844, add.809)
  is-finite.847 = pred[4]{0} is-finite(subtract.846)
  not.848 = pred[4]{0} not(is-finite.847)
  abs.849 = f32[4]{0} abs(subtract.846)
  constant.850 = f32[] constant(inf)
  broadcast.851 = f32[4]{0} broadcast(constant.850), dimensions={}
  compare.852 = pred[4]{0} compare(abs.849, broadcast.851), direction=EQ, type=UNSIGNED
  not.853 = pred[4]{0} not(compare.852)
  and.854 = pred[4]{0} and(not.848, not.853)
  add.855 = f32[4]{0} add(add.844, add.809)
  maximum.845 = f32[4]{0} maximum(add.844, add.809)
  abs.856 = f32[4]{0} abs(subtract.846)
  negate.857 = f32[4]{0} negate(abs.856)
  exponential.858 = f32[4]{0} exponential(negate.857)
  log-plus-one.859 = f32[4]{0} log-plus-one(exponential.858)
  add.860 = f32[4]{0} add(maximum.845, log-plus-one.859)
  select.861 = f32[4]{0} select(and.854, add.855, add.860)
  broadcast.862 = f32[4,1]{1,0} broadcast(select.861), dimensions={0}
  broadcast.863 = f32[4,1]{1,0} broadcast(select.861), dimensions={0}
  concatenate.864 = f32[4,2]{1,0} concatenate(broadcast.862, broadcast.863), dimensions={1}
  constant.865 = f32[] constant(0)
  reduce.870 = f32[] reduce(concatenate.864, constant.865), dimensions={0,1}, to_apply=primitive_computation_add.866
  constant.871 = f32[] constant(8)
  divide.872 = f32[] divide(reduce.870, constant.871)
  ROOT tuple.873 = (f32[]) tuple(divide.872)
}
  )")
                    .ConsumeValueOrDie();
  auto input_array = absl::make_unique<Array2D<float>>(4, 2);
  input_array->FillUnique(1.0f);
  auto input = LiteralUtil::CreateR2FromArray2D<float>(*input_array);
  EXPECT_TRUE(RunAndCompare(std::move(module), {&input}, absl::nullopt));
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
  auto x_data = client_->TransferToServer(x_literal).ConsumeValueOrDie();
  auto y_data = client_->TransferToServer(y_literal).ConsumeValueOrDie();

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

// Test that the HLO optimization to replace a concat of a broadcasted scalar
// produces the correct result in rank 1.
XLA_TEST_F(ConcatTest, ConcatBroadcastArgument) {
  auto f32_scalar = ShapeUtil::MakeShape(xla::F32, {});
  auto x_literal = LiteralUtil::CreateR1<float>({2.0f, 3.0f, 5.0f, 6.0f});
  auto y_literal = LiteralUtil::CreateR0<float>(1.5f);
  auto z_literal = LiteralUtil::CreateR0<float>(5.5f);
  auto x_data = client_->TransferToServer(x_literal).ConsumeValueOrDie();
  auto y_data = client_->TransferToServer(y_literal).ConsumeValueOrDie();
  auto z_data = client_->TransferToServer(z_literal).ConsumeValueOrDie();

  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, x_literal.shape(), "x");
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

// Test that the HLO optimization to replace a concat of a broadcasted scalar
// produces the correct result in rank 3 with both high and low padding in
// different dimensions.
XLA_TEST_F(ConcatTest, ConcatBroadcastArgumentR3) {
  auto f32_scalar = ShapeUtil::MakeShape(xla::F32, {});
  Array3D<float> x3d(3, 5, 7, 3.14f);
  auto x_literal = LiteralUtil::CreateR3FromArray3D<float>(x3d);
  auto y_literal = LiteralUtil::CreateR0<float>(1.5f);
  auto z_literal = LiteralUtil::CreateR0<float>(5.5f);
  auto x_data = client_->TransferToServer(x_literal).ConsumeValueOrDie();
  auto y_data = client_->TransferToServer(y_literal).ConsumeValueOrDie();
  auto z_data = client_->TransferToServer(z_literal).ConsumeValueOrDie();

  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, x_literal.shape(), "x");
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
