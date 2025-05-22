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

#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/reference_util.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/client_library_test_runner_utils.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace {

using ConcatTest = ClientLibraryTestRunnerMixin<
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>>;
using ConcatTestHlo = HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;
using ::testing::HasSubstr;

// Concatenate expects at least one argument.
XLA_TEST_F(ConcatTest, Concat_Nothing) {
  XlaBuilder builder(TestName());
  ConcatInDim(&builder, {}, 0);
  absl::StatusOr<XlaComputation> computation_status = builder.Build();
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
  absl::StatusOr<XlaComputation> computation_status = builder.Build();
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
  absl::StatusOr<XlaComputation> computation_status = builder.Build();
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
  absl::StatusOr<XlaComputation> computation_status = builder.Build();
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
  auto a0 = ConstantR1<int32_t>(&builder, {1});
  auto a1 = ConstantR1<int32_t>(&builder, {2, 3});
  auto a2 = ConstantR1<int32_t>(&builder, {4, 5, 6});
  auto a3 = ConstantR1<int32_t>(&builder, {7, 8, 9, 10});
  ConcatInDim(&builder, {a0, a1, a2, a3}, 0);

  std::vector<int32_t> expected(10);
  std::iota(expected.begin(), expected.end(), 1);
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(ConcatTest, ConcatR3WeirdDims) {
  XlaBuilder builder(TestName());

  Array3D<float> arr0(9, 17, 1);
  arr0.Fill(1);

  Array3D<float> arr1(9, 17, 256);
  arr1.Fill(2);

  Array3D<float> expected(9, 17, arr0.n3() + arr1.n3());
  for (int64_t i = 0; i < expected.n1(); ++i) {
    for (int64_t j = 0; j < expected.n2(); ++j) {
      int64_t kk = 0;
      for (const Array3D<float>& arr : {arr0, arr1}) {
        for (int64_t k = 0; k < arr.n3(); ++k, ++kk) {
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

  ComputeAndCompareR3<float>(&builder, expected, {&p0, &p1});
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
  ComputeAndCompareR1<float>(&builder, expected, {&a_literal});
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
  parameter.38 = f32[4,2]{1,0} parameter(0)
  reshape.723 = f32[4,2,1]{2,1,0} reshape(parameter.38)
  reshape.724 = f32[4,2,1]{2,1,0} reshape(parameter.38)
  concatenate.42 = f32[4,2,2]{2,1,0} concatenate(reshape.723, reshape.724), dimensions={2}
  slice.351 = f32[4,1,2]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [0:2]}
  reshape.1058 = f32[4,2]{1,0} reshape(slice.351)
  slice.352 = f32[4,1]{1,0} slice(reshape.1058), slice={[0:4], [1:2]}
  reshape.1059 = f32[4]{0} reshape(slice.352)
  slice.353 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [1:2]}
  reshape.1060 = f32[4]{0} reshape(slice.353)
  add.124 = f32[4]{0} add(reshape.1059, reshape.1060)
  slice.354 = f32[4,1]{1,0} slice(reshape.1058), slice={[0:4], [0:1]}
  reshape.1061 = f32[4]{0} reshape(slice.354)
  slice.379 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [0:1]}
  reshape.1062 = f32[4]{0} reshape(slice.379)
  add.89 = f32[4]{0} add(reshape.1061, reshape.1062)
  subtract.126 = f32[4]{0} subtract(add.124, add.89)
  is-finite.127 = pred[4]{0} is-finite(subtract.126)
  not.128 = pred[4]{0} not(is-finite.127)
  abs.129 = f32[4]{0} abs(subtract.126)
  constant.130 = f32[] constant(inf)
  broadcast.131 = f32[4]{0} broadcast(constant.130), dimensions={}
  compare.132 = pred[4]{0} compare(abs.129, broadcast.131), direction=EQ
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
  slice.356 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [1:2]}
  reshape.1064 = f32[4]{0} reshape(slice.356)
  add.214 = f32[4]{0} add(select.141, reshape.1064)
  slice.380 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [0:1]}
  reshape.1066 = f32[4]{0} reshape(slice.380)
  add.179 = f32[4]{0} add(select.141, reshape.1066)
  subtract.216 = f32[4]{0} subtract(add.214, add.179)
  is-finite.217 = pred[4]{0} is-finite(subtract.216)
  not.218 = pred[4]{0} not(is-finite.217)
  abs.219 = f32[4]{0} abs(subtract.216)
  constant.220 = f32[] constant(inf)
  broadcast.221 = f32[4]{0} broadcast(constant.220), dimensions={}
  compare.222 = pred[4]{0} compare(abs.219, broadcast.221), direction=EQ
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
  slice.359 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [1:2]}
  reshape.1068 = f32[4]{0} reshape(slice.359)
  add.304 = f32[4]{0} add(select.231, reshape.1068)
  slice.381 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [0:1]}
  reshape.1070 = f32[4]{0} reshape(slice.381)
  add.269 = f32[4]{0} add(select.231, reshape.1070)
  subtract.306 = f32[4]{0} subtract(add.304, add.269)
  is-finite.307 = pred[4]{0} is-finite(subtract.306)
  not.308 = pred[4]{0} not(is-finite.307)
  abs.309 = f32[4]{0} abs(subtract.306)
  constant.310 = f32[] constant(inf)
  broadcast.311 = f32[4]{0} broadcast(constant.310), dimensions={}
  compare.312 = pred[4]{0} compare(abs.309, broadcast.311), direction=EQ
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
  slice.362 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [1:2]}
  reshape.1072 = f32[4]{0} reshape(slice.362)
  add.394 = f32[4]{0} add(select.321, reshape.1072)
  slice.382 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [0:1]}
  reshape.1074 = f32[4]{0} reshape(slice.382)
  add.359 = f32[4]{0} add(select.321, reshape.1074)
  subtract.396 = f32[4]{0} subtract(add.394, add.359)
  is-finite.397 = pred[4]{0} is-finite(subtract.396)
  not.398 = pred[4]{0} not(is-finite.397)
  abs.399 = f32[4]{0} abs(subtract.396)
  constant.400 = f32[] constant(inf)
  broadcast.401 = f32[4]{0} broadcast(constant.400), dimensions={}
  compare.402 = pred[4]{0} compare(abs.399, broadcast.401), direction=EQ
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
  slice.365 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [1:2]}
  reshape.1076 = f32[4]{0} reshape(slice.365)
  add.484 = f32[4]{0} add(select.411, reshape.1076)
  slice.383 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [0:1]}
  reshape.1078 = f32[4]{0} reshape(slice.383)
  add.449 = f32[4]{0} add(select.411, reshape.1078)
  subtract.486 = f32[4]{0} subtract(add.484, add.449)
  is-finite.487 = pred[4]{0} is-finite(subtract.486)
  not.488 = pred[4]{0} not(is-finite.487)
  abs.489 = f32[4]{0} abs(subtract.486)
  constant.490 = f32[] constant(inf)
  broadcast.491 = f32[4]{0} broadcast(constant.490), dimensions={}
  compare.492 = pred[4]{0} compare(abs.489, broadcast.491), direction=EQ
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
  slice.368 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [1:2]}
  reshape.1080 = f32[4]{0} reshape(slice.368)
  add.574 = f32[4]{0} add(select.501, reshape.1080)
  slice.384 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [0:1]}
  reshape.1082 = f32[4]{0} reshape(slice.384)
  add.539 = f32[4]{0} add(select.501, reshape.1082)
  subtract.576 = f32[4]{0} subtract(add.574, add.539)
  is-finite.577 = pred[4]{0} is-finite(subtract.576)
  not.578 = pred[4]{0} not(is-finite.577)
  abs.579 = f32[4]{0} abs(subtract.576)
  constant.580 = f32[] constant(inf)
  broadcast.581 = f32[4]{0} broadcast(constant.580), dimensions={}
  compare.582 = pred[4]{0} compare(abs.579, broadcast.581), direction=EQ
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
  slice.371 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [1:2]}
  reshape.1084 = f32[4]{0} reshape(slice.371)
  add.664 = f32[4]{0} add(select.591, reshape.1084)
  slice.385 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [0:1]}
  reshape.1086 = f32[4]{0} reshape(slice.385)
  add.629 = f32[4]{0} add(select.591, reshape.1086)
  subtract.666 = f32[4]{0} subtract(add.664, add.629)
  is-finite.667 = pred[4]{0} is-finite(subtract.666)
  not.668 = pred[4]{0} not(is-finite.667)
  abs.669 = f32[4]{0} abs(subtract.666)
  constant.670 = f32[] constant(inf)
  broadcast.671 = f32[4]{0} broadcast(constant.670), dimensions={}
  compare.672 = pred[4]{0} compare(abs.669, broadcast.671), direction=EQ
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
  slice.374 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [1:2]}
  reshape.1088 = f32[4]{0} reshape(slice.374)
  add.754 = f32[4]{0} add(select.681, reshape.1088)
  slice.386 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [0:1]}
  reshape.1090 = f32[4]{0} reshape(slice.386)
  add.719 = f32[4]{0} add(select.681, reshape.1090)
  subtract.756 = f32[4]{0} subtract(add.754, add.719)
  is-finite.757 = pred[4]{0} is-finite(subtract.756)
  not.758 = pred[4]{0} not(is-finite.757)
  abs.759 = f32[4]{0} abs(subtract.756)
  constant.760 = f32[] constant(inf)
  broadcast.761 = f32[4]{0} broadcast(constant.760), dimensions={}
  compare.762 = pred[4]{0} compare(abs.759, broadcast.761), direction=EQ
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
  slice.377 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [1:2]}
  reshape.1092 = f32[4]{0} reshape(slice.377)
  add.844 = f32[4]{0} add(select.771, reshape.1092)
  slice.387 = f32[4,1,1]{2,1,0} slice(concatenate.42), slice={[0:4], [0:1], [0:1]}
  reshape.1094 = f32[4]{0} reshape(slice.387)
  add.809 = f32[4]{0} add(select.771, reshape.1094)
  subtract.846 = f32[4]{0} subtract(add.844, add.809)
  is-finite.847 = pred[4]{0} is-finite(subtract.846)
  not.848 = pred[4]{0} not(is-finite.847)
  abs.849 = f32[4]{0} abs(subtract.846)
  constant.850 = f32[] constant(inf)
  broadcast.851 = f32[4]{0} broadcast(constant.850), dimensions={}
  compare.852 = pred[4]{0} compare(abs.849, broadcast.851), direction=EQ
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
  constant.865 = f32[] constant(0)
  reduce.2 = f32[] reduce(select.861, constant.865), dimensions={0}, to_apply=primitive_computation_add.866
  reduce.3 = f32[] reduce(select.861, constant.865), dimensions={0}, to_apply=primitive_computation_add.866
  add.77 = f32[] add(reduce.2, reduce.3)
  constant.719 = f32[] constant(0.125)
  multiply = f32[] multiply(add.77, constant.719)
  ROOT tuple.873 = (f32[]) tuple(multiply)
})")
                    .value();
  auto input_array = std::make_unique<Array2D<float>>(4, 2);
  input_array->FillUnique(1.0f);
  auto input = LiteralUtil::CreateR2FromArray2D<float>(*input_array);
  EXPECT_TRUE(RunAndCompare(std::move(module), {&input}, kDefaultErrorSpec));
}

// Describes a binary rank-2 concatenation test.
struct R2BinarySpec {
  int64_t lhs_dim0;
  int64_t lhs_dim1;
  int64_t rhs_dim0;
  int64_t rhs_dim1;
  int64_t concat_dimension;
};

// TEST_P harness for binary rank-2 concatenation.
class ConcatR2BinaryTest : public ConcatTest,
                           public ::testing::WithParamInterface<R2BinarySpec> {
};

TEST_P(ConcatR2BinaryTest, DoIt) {
  const R2BinarySpec& spec = GetParam();
  Array2D<int32_t> lhs(spec.lhs_dim0, spec.lhs_dim1);
  lhs.FillUnique();
  Array2D<int32_t> rhs(spec.rhs_dim0, spec.rhs_dim1);
  rhs.FillUnique(1000);

  XlaBuilder builder(TestName());
  auto a0 = ConstantR2FromArray2D<int32_t>(&builder, lhs);
  auto a1 = ConstantR2FromArray2D<int32_t>(&builder, rhs);
  ConcatInDim(&builder, {a0, a1}, spec.concat_dimension);

  std::unique_ptr<Array2D<int32_t>> expected =
      ReferenceUtil::Concat2D(lhs, rhs, spec.concat_dimension);
  ComputeAndCompareR2<int32_t>(&builder, *expected, {});
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

  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, f32_scalar, "x");
  auto y = Parameter(&builder, 1, f32_scalar, "y");
  auto mul = Mul(x, y);
  auto add1 = Add(mul, ConstantR1<float>(&builder, {1.f, 2.f}));
  auto add2 = Add(mul, ConstantR1<float>(&builder, {3.f, 4.f}));
  auto add3 = Add(mul, ConstantR1<float>(&builder, {5.f, 6.f}));
  ConcatInDim(&builder, {add1, add2, add3}, /*dimension=*/0);

  ComputeAndCompareR1<float>(&builder, {7., 8., 9., 10., 11., 12.},
                             {&x_literal, &y_literal}, ErrorSpec(1e-4));
}

// Test that the HLO optimization to replace a concat of a broadcasted scalar
// produces the correct result in rank 1.
XLA_TEST_F(ConcatTest, ConcatBroadcastArgument) {
  auto f32_scalar = ShapeUtil::MakeShape(xla::F32, {});
  auto x_literal = LiteralUtil::CreateR1<float>({2.0f, 3.0f, 5.0f, 6.0f});
  auto y_literal = LiteralUtil::CreateR0<float>(1.5f);
  auto z_literal = LiteralUtil::CreateR0<float>(5.5f);

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
      {&x_literal, &y_literal, &z_literal}, ErrorSpec(1e-4));
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
                             {&x_literal, &y_literal, &z_literal},
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
