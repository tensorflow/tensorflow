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
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class SliceTest : public ClientLibraryTestBase {
 protected:
  template <typename NativeT>
  void RunSliceTenToTwo() {
    std::vector<NativeT> constant;
    constant.reserve(10);
    for (int i = 0; i < 10; ++i) {
      constant.push_back(static_cast<NativeT>(i));
    }

    ComputationBuilder builder(client_, TestName());
    auto original = builder.ConstantR1<NativeT>(constant);
    builder.Slice(original, {2}, {4});

    const std::vector<NativeT> expected = {static_cast<NativeT>(2),
                                           static_cast<NativeT>(3)};
    ComputeAndCompareR1<NativeT>(&builder, expected, {});
  }
};

XLA_TEST_F(SliceTest, SliceZeroToZeroF32) {
  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR1<float>({});
  builder.Slice(original, {0}, {0});

  ComputeAndCompareR1<float>(&builder, {}, {});
}

XLA_TEST_F(SliceTest, SliceTenToZeroF32) {
  ComputationBuilder builder(client_, TestName());
  std::vector<float> constant(10, 0.3);
  auto original = builder.ConstantR1<float>(constant);
  builder.Slice(original, {7}, {7});

  ComputeAndCompareR1<float>(&builder, {}, {});
}

TEST_F(SliceTest, SliceTenToTwoF32) { RunSliceTenToTwo<float>(); }

XLA_TEST_F(SliceTest, SliceTenToTwoF64) { RunSliceTenToTwo<double>(); }

TEST_F(SliceTest, SliceTenToTwoU32) { RunSliceTenToTwo<uint32>(); }

TEST_F(SliceTest, SliceTenToTwoS32) { RunSliceTenToTwo<int32>(); }

XLA_TEST_F(SliceTest, SliceTenToTwoU64) { RunSliceTenToTwo<uint64>(); }

XLA_TEST_F(SliceTest, SliceTenToTwoS64) { RunSliceTenToTwo<int64>(); }

TEST_F(SliceTest, SliceTenToTen) {
  const std::vector<float> values = {0.0, 1.0, 2.0, 3.0, 4.0,
                                     5.0, 6.0, 7.0, 8.0, 9.0};

  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR1<float>(values);
  builder.Slice(original, {0}, {10});

  ComputeAndCompareR1<float>(&builder, values, {}, ErrorSpec(0.000001));
}

TEST_F(SliceTest, SliceLastFourOf1024) {
  std::vector<float> values(1024);
  std::iota(values.begin(), values.end(), 0.0);

  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR1<float>(values);
  builder.Slice(original, {1024 - 4}, {1024});

  const std::vector<float> expected = {1020, 1021, 1022, 1023};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

// TODO(b/28491443): Fix wrong result on CPU and GPU. Failed on
// 2016-05-01. Also b/28508652
TEST_F(SliceTest, DISABLED_SliceUnaligned1024In4096Values) {
  std::vector<float> values(4096);
  std::iota(values.begin(), values.end(), 0.0);

  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR1<float>(values);
  builder.Slice(original, {7}, {7 + 1024});

  std::vector<float> expected(1024);
  std::iota(values.begin(), values.end(), 7.0);
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

XLA_TEST_F(SliceTest, Slice0x0to0x0F32) {
  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR2FromArray2D<float>(Array2D<float>(0, 0));
  builder.Slice(original, {0, 0}, {0, 0});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 0), {});
}

XLA_TEST_F(SliceTest, Slice0x20to0x5F32) {
  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR2FromArray2D<float>(Array2D<float>(0, 20));
  builder.Slice(original, {0, 15}, {0, 20});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 5), {});
}

XLA_TEST_F(SliceTest, Slice3x0to2x0F32) {
  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR2FromArray2D<float>(Array2D<float>(3, 0));
  builder.Slice(original, {1, 0}, {3, 0});

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
  builder.Slice(original, {128, 128}, {256, 256});

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
  builder.Slice(original, {0, 3072}, {1, 4096});

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
  builder.Slice(original, {0, 0}, {16, 2});
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

// Tests: (f32[2, 2, 24, 256], starts = {1, 0, 8, 0}, ends = {2, 2, 16, 128}
TEST_F(SliceTest, SliceR4ThreeDimsMiddleMinor) {
  Array4D<float> values(2, 2, 24, 256);
  values.FillRandom(3.14f);
  auto expected =
      ReferenceUtil::Slice4D(values, {{1, 0, 8, 0}}, {{2, 2, 16, 128}});
  ComputationBuilder builder(client_, TestName());
  auto original = builder.ConstantR4FromArray4D(values);
  builder.Slice(original, {1, 0, 8, 0}, {2, 2, 16, 128});
  ComputeAndCompareR4(&builder, *expected, {}, ErrorSpec(0.000001));
}

struct R2Spec {
  int64 input_dim0;
  int64 input_dim1;
  std::array<int64, 2> slice_starts;
  std::array<int64, 2> slice_limits;
  Layout layout;
};

// Parameterized test that generates patterned R2 values, slices them according
// to the R2Spec, and compares the results with the ReferenceUtil version.
class SliceR2Test : public ClientLibraryTestBase,
                    public ::testing::WithParamInterface<R2Spec> {};

TEST_P(SliceR2Test, DoIt) {
  const R2Spec& spec = GetParam();
  Array2D<int32> input(spec.input_dim0, spec.input_dim1);
  input.FillUnique();

  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR2FromArray2D<int32>(input);
  builder.Slice(a, spec.slice_starts, spec.slice_limits);

  std::unique_ptr<Array2D<int32>> expected =
      ReferenceUtil::Slice2D(input, spec.slice_starts, spec.slice_limits);
  ComputeAndCompareR2<int32>(&builder, *expected, {});
}

// clang-format off
INSTANTIATE_TEST_CASE_P(
    SliceR2TestInstantiation, SliceR2Test,
    ::testing::Values(
        R2Spec {4, 12, {{0, 3}}, {{4, 6}}, LayoutUtil::MakeLayout({0, 1})},
        R2Spec {4, 12, {{0, 3}}, {{4, 6}}, LayoutUtil::MakeLayout({1, 0})},
        R2Spec {16, 4, {{0, 2}}, {{16, 4}}, LayoutUtil::MakeLayout({0, 1})},
        R2Spec {16, 4, {{0, 2}}, {{16, 4}}, LayoutUtil::MakeLayout({1, 0})},
        R2Spec {256, 400, {{0, 300}}, {{256, 400}},
          LayoutUtil::MakeLayout({1, 0})},
        R2Spec {500, 400, {{111, 123}}, {{300, 257}},
          LayoutUtil::MakeLayout({1, 0})},
        R2Spec {500, 400, {{111, 123}}, {{300, 400}},
          LayoutUtil::MakeLayout({1, 0})},
        R2Spec {384, 512, {{128, 256}}, {{256, 384}},
          LayoutUtil::MakeLayout({1, 0})},
        R2Spec {357, 512, {{111, 256}}, {{301, 384}},
          LayoutUtil::MakeLayout({1, 0})}
    )
);
// clang-format on

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
