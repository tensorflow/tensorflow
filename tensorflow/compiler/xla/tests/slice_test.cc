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

// Tests that slice operations can be performed.

#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class SliceTest : public ClientLibraryTestBase {};

TEST_F(SliceTest, Slice3x3x3_To_3x3x1_F32) {
  Array3D<float> values(3, 3, 3);
  values.FillIota(0);

  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR3FromArray3D<float>(values);
  builder.Slice(original, {0, 0, 0}, {3, 3, 1}, {1, 1, 1});

  Array3D<float> expected{
      {{0.0}, {3.0}, {6.0}}, {{9.0}, {12.0}, {15.0}}, {{18.0}, {21.0}, {24.0}}};
  ComputeAndCompareR3<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

TEST_F(SliceTest, Slice3x3x3_To_3x1x3_F32) {
  Array3D<float> values(3, 3, 3);
  values.FillIota(0);

  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR3FromArray3D<float>(values);
  builder.Slice(original, {0, 0, 0}, {3, 1, 3}, {1, 1, 1});

  Array3D<float> expected{
      {{0.0, 1.0, 2.0}}, {{9.0, 10.0, 11.0}}, {{18.0, 19.0, 20.0}}};
  ComputeAndCompareR3<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

TEST_F(SliceTest, Slice3x3x3_To_1x3x3_F32) {
  Array3D<float> values(3, 3, 3);
  values.FillIota(0);

  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR3FromArray3D<float>(values);
  builder.Slice(original, {0, 0, 0}, {1, 3, 3}, {1, 1, 1});

  Array3D<float> expected{
      {{{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}}}};
  ComputeAndCompareR3<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

XLA_TEST_F(SliceTest, Slice0x0to0x0F32) {
  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR2FromArray2D<float>(Array2D<float>(0, 0));
  builder.Slice(original, {0, 0}, {0, 0}, {1, 1});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 0), {});
}

XLA_TEST_F(SliceTest, Slice0x20to0x5F32) {
  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR2FromArray2D<float>(Array2D<float>(0, 20));
  builder.Slice(original, {0, 15}, {0, 20}, {1, 1});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 5), {});
}

XLA_TEST_F(SliceTest, Slice3x0to2x0F32) {
  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR2FromArray2D<float>(Array2D<float>(3, 0));
  builder.Slice(original, {1, 0}, {3, 0}, {1, 1});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(2, 0), {});
}

XLA_TEST_F(SliceTest, SliceQuadrantOf256x256) {
  Array2D<float> values(256, 256);
  for (int row = 0; row < 256; ++row) {
    for (int col = 0; col < 256; ++col) {
      values(row, col) = (row << 10) | col;
    }
  }

  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR2FromArray2D<float>(values);
  builder.Slice(original, {128, 128}, {256, 256}, {1, 1});

  Array2D<float> expected(128, 128);
  for (int row = 0; row < 128; ++row) {
    for (int col = 0; col < 128; ++col) {
      expected(row, col) = ((row + 128) << 10) | (col + 128);
    }
  }
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

// Tests: (f32[1,4096], starts={0, 3072}, limits={1, 4096}) -> f32[1,1024])
TEST_F(SliceTest, Slice_1x4096_To_1x1024) {
  Array2D<float> values(1, 4096);
  std::iota(values.data(), values.data() + 4096, 0.0);

  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR2FromArray2D<float>(values);
  builder.Slice(original, {0, 3072}, {1, 4096}, {1, 1});

  Array2D<float> expected(1, 1024);
  std::iota(expected.data(), expected.data() + 1024, 3072.0);
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

// Tests slice: (f32[16,4], starts={0, 0}, limits={16, 2}) -> f32[16,2]
TEST_F(SliceTest, Slice_16x4_To_16x2) {
  Array2D<float> values(16, 4);
  Array2D<float> expected(16, 2);
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 4; ++col) {
      values(row, col) = (row << 10) | col;
      if (col < 2) {
        expected(row, col) = (row << 10) | col;
      }
    }
  }
  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR2FromArray2D<float>(values);
  builder.Slice(original, {0, 0}, {16, 2}, {1, 1});
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

// Tests: (f32[2, 2, 24, 256], starts = {1, 0, 8, 0}, ends = {2, 2, 16, 128}
TEST_F(SliceTest, SliceR4ThreeDimsMiddleMinor) {
  Array4D<float> values(2, 2, 24, 256);
  values.FillRandom(3.14f);
  auto expected = ReferenceUtil::Slice4D(
      values, {{1, 0, 8, 0}}, {{2, 2, 16, 128}}, /*strides=*/{{1, 1, 1, 1}});
  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR4FromArray4D(values);
  builder.Slice(original, {1, 0, 8, 0}, {2, 2, 16, 128}, {1, 1, 1, 1});
  ComputeAndCompareR4(&builder, *expected, {}, ErrorSpec(0.000001));
}

struct R1Spec {
  int64 input_dim0;
  int64 slice_start;
  int64 slice_limit;
  int64 slice_stride;
};

// Parameterized test that generates R1 values, slices them according
// to the R1Spec, and compares the result with a computed version.
class SliceR1Test : public ClientLibraryTestBase,
                    public ::testing::WithParamInterface<R1Spec> {
 protected:
  template <typename NativeT>
  void Run(const R1Spec& spec) {
    std::vector<NativeT> input(spec.input_dim0);
    std::iota(input.begin(), input.end(), NativeT());

    ComputationBuilder builder(client_, TestName());
    auto original = builder.ConstantR1<NativeT>(input);
    builder.Slice(original, {spec.slice_start}, {spec.slice_limit},
                  {spec.slice_stride});

    std::vector<NativeT> expected;
    for (int i = spec.slice_start; i < spec.slice_limit;
         i += spec.slice_stride) {
      expected.push_back(i);
    }

    ComputeAndCompareR1<NativeT>(&builder, expected, {});
  }
};

XLA_TEST_P(SliceR1Test, DoIt_F32) {
  Run<float>(GetParam());
}

XLA_TEST_P(SliceR1Test, DoIt_F64) {
  Run<double>(GetParam());
}

XLA_TEST_P(SliceR1Test, DoIt_U32) {
  Run<uint32>(GetParam());
}

XLA_TEST_P(SliceR1Test, DoIt_S32) {
  Run<int32>(GetParam());
}

XLA_TEST_P(SliceR1Test, DoIt_U64) {
  Run<uint64>(GetParam());
}

XLA_TEST_P(SliceR1Test, DoIt_S64) {
  Run<int64>(GetParam());
}

INSTANTIATE_TEST_CASE_P(                  //
    SliceR1TestInstantiation,             //
    SliceR1Test,                          //
    ::testing::Values(                    //
        R1Spec{10, 0, 0, 1},              //
        R1Spec{10, 7, 7, 1},              //
        R1Spec{10, 2, 4, 1},              //
        R1Spec{10, 2, 4, 1},              //
        R1Spec{10, 2, 4, 1},              //
        R1Spec{10, 2, 4, 1},              //
        R1Spec{10, 2, 4, 1},              //
        R1Spec{10, 2, 4, 1},              //
        R1Spec{10, 0, 10, 1},             //
        R1Spec{1024, 1024 - 4, 1024, 1},  //
        R1Spec{4096, 7, 7 + 1024, 1},     //
        R1Spec{10, 0, 10, 2},             //
        R1Spec{10, 0, 10, 3},             //
        R1Spec{10, 0, 10, 4},             //
        R1Spec{10, 0, 10, 5},             //
        R1Spec{10, 0, 10, 10}             //
        )                                 //
);

struct R2Spec {
  int64 input_dim0;
  int64 input_dim1;
  std::array<int64, 2> slice_starts;
  std::array<int64, 2> slice_limits;
  std::array<int64, 2> slice_strides;
  Layout layout;
};

// Parameterized test that generates patterned R2 values, slices them according
// to the R2Spec, and compares the results with the ReferenceUtil version.
class SliceR2Test : public ClientLibraryTestBase,
                    public ::testing::WithParamInterface<R2Spec> {};

XLA_TEST_P(SliceR2Test, DoIt) {
  const R2Spec& spec = GetParam();
  Array2D<int32> input(spec.input_dim0, spec.input_dim1);
  input.FillUnique();

  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR2FromArray2DWithLayout<int32>(input, spec.layout);
  builder.Slice(a, spec.slice_starts, spec.slice_limits, spec.slice_strides);

  std::unique_ptr<Array2D<int32>> expected = ReferenceUtil::Slice2D(
      input, spec.slice_starts, spec.slice_limits, spec.slice_strides);
  ComputeAndCompareR2<int32>(&builder, *expected, {});
}

// clang-format off
INSTANTIATE_TEST_CASE_P(
    SliceR2TestInstantiation, SliceR2Test,
    ::testing::Values(
        R2Spec {4, 12, {{0, 3}}, {{4, 6}}, {{1, 1}},
          LayoutUtil::MakeLayout({0, 1})},
        R2Spec {4, 12, {{0, 3}}, {{4, 6}}, {{1, 1}},
          LayoutUtil::MakeLayout({1, 0})},
        R2Spec {16, 4, {{0, 2}}, {{16, 4}}, {{1, 1}},
          LayoutUtil::MakeLayout({0, 1})},
        R2Spec {16, 4, {{0, 2}}, {{16, 4}}, {{1, 1}},
          LayoutUtil::MakeLayout({1, 0})},
        R2Spec {256, 400, {{0, 300}}, {{256, 400}}, {{1, 1}},
          LayoutUtil::MakeLayout({1, 0})},
        R2Spec {500, 400, {{111, 123}}, {{300, 257}}, {{1, 1}},
          LayoutUtil::MakeLayout({1, 0})},
        R2Spec {500, 400, {{111, 123}}, {{300, 400}}, {{1, 1}},
          LayoutUtil::MakeLayout({1, 0})},
        R2Spec {384, 512, {{128, 256}}, {{256, 384}}, {{1, 1}},
          LayoutUtil::MakeLayout({1, 0})},
        R2Spec {357, 512, {{111, 256}}, {{301, 384}}, {{1, 1}},
          LayoutUtil::MakeLayout({1, 0})},
        R2Spec {10, 10, {{0, 0}}, {{10, 10}}, {{1, 2}},
          LayoutUtil::MakeLayout({0, 1})},
        R2Spec {10, 10, {{0, 0}}, {{10, 10}}, {{1, 2}},
          LayoutUtil::MakeLayout({1, 0})},
        R2Spec {10, 10, {{0, 0}}, {{10, 10}}, {{2, 1}},
          LayoutUtil::MakeLayout({0, 1})},
        R2Spec {10, 10, {{0, 0}}, {{10, 10}}, {{2, 1}},
          LayoutUtil::MakeLayout({1, 0})},
        R2Spec {10, 10, {{0, 0}}, {{10, 10}}, {{2, 2}},
          LayoutUtil::MakeLayout({0, 1})},
        R2Spec {10, 10, {{0, 0}}, {{10, 10}}, {{2, 2}},
          LayoutUtil::MakeLayout({1, 0})}
    )
);
// clang-format on

}  // namespace
}  // namespace xla
