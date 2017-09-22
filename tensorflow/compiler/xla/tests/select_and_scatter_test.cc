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

// Tests the select-and-scatter XLA operation.

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

struct SelectAndScatterTestParam {
  Array4D<float> operand_shape;
  Array4D<float> source_shape;
  Padding padding_type;
  tensorflow::gtl::ArraySlice<int64> window_dimensions;
  tensorflow::gtl::ArraySlice<int64> window_strides;
};

class SelectAndScatterTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<SelectAndScatterTestParam> {
 public:
  SelectAndScatterTest() : builder_(client_, TestName()) {
    // Create S32 GE and ADD computations for select and scatter respectively.
    ge_s32_ = CreateScalarGeComputation(S32, &builder_);
    add_s32_ = CreateScalarAddComputation(S32, &builder_);
    ge_f32_ = CreateScalarGeComputation(F32, &builder_);
    add_f32_ = CreateScalarAddComputation(F32, &builder_);
    max_f32_ = CreateScalarMaxComputation(F32, &builder_);
    min_f32_ = CreateScalarMinComputation(F32, &builder_);
  }

  ComputationBuilder builder_;
  Computation ge_s32_;
  Computation add_s32_;
  Computation ge_f32_;
  Computation add_f32_;
  Computation max_f32_;
  Computation min_f32_;
};

XLA_TEST_P(SelectAndScatterTest, R4Randomized) {
  Array4D<float> o(GetParam().operand_shape);
  o.FillRandom(1.5f);
  auto operand = builder_.ConstantR4FromArray4D(o);

  Array4D<float> s(GetParam().source_shape);
  s.FillRandom(12.0f);
  auto source = builder_.ConstantR4FromArray4D(s);

  builder_.SelectAndScatter(operand, ge_f32_, GetParam().window_dimensions,
                            GetParam().window_strides, GetParam().padding_type,
                            source, builder_.ConstantR0<float>(0.0f), add_f32_);

  auto e = ReferenceUtil::SelectAndScatter4DGePlus(
      o, s, 0.0f, GetParam().window_dimensions, GetParam().window_strides,
      GetParam().padding_type == Padding::kSame);

  ComputeAndCompareR4<float>(&builder_, *e, {}, ErrorSpec(1e-5));
}

INSTANTIATE_TEST_CASE_P(
    SelectAndScatterTest_Instantiation, SelectAndScatterTest,
    ::testing::Values(SelectAndScatterTestParam{{6, 6, 256, 128},
                                                {3, 3, 256, 128},
                                                Padding::kSame,
                                                {3, 3, 1, 1},
                                                {2, 2, 1, 1}},
                      SelectAndScatterTestParam{{7, 7, 256, 128},
                                                {3, 3, 256, 128},
                                                Padding::kValid,
                                                {3, 3, 1, 1},
                                                {2, 2, 1, 1}},
                      SelectAndScatterTestParam{{6, 7, 256, 128},
                                                {3, 3, 256, 128},
                                                Padding::kValid,
                                                {2, 3, 1, 1},
                                                {2, 2, 1, 1}},
                      SelectAndScatterTestParam{{6, 7, 256, 128},
                                                {2, 3, 256, 128},
                                                Padding::kValid,
                                                {2, 3, 1, 1},
                                                {3, 2, 1, 1}},
                      SelectAndScatterTestParam{{9, 9, 16, 128},
                                                {3, 3, 16, 128},
                                                Padding::kValid,
                                                {3, 3, 1, 1},
                                                {3, 3, 1, 1}},
                      SelectAndScatterTestParam{{3, 3, 4, 4},
                                                {1, 1, 4, 4},
                                                Padding::kValid,
                                                {3, 3, 1, 1},
                                                {3, 3, 1, 1}},
                      SelectAndScatterTestParam{{3, 3, 4, 4},
                                                {1, 1, 4, 4},
                                                Padding::kValid,
                                                {3, 3, 1, 1},
                                                {3, 3, 1, 1}},
                      SelectAndScatterTestParam{{9, 3, 4, 4},
                                                {3, 1, 4, 4},
                                                Padding::kValid,
                                                {3, 3, 1, 1},
                                                {3, 3, 1, 1}},
                      SelectAndScatterTestParam{{7, 3, 4, 4},
                                                {3, 1, 4, 4},
                                                Padding::kValid,
                                                {3, 3, 1, 1},
                                                {2, 3, 1, 1}},
                      SelectAndScatterTestParam{{1, 1, 5, 5},
                                                {1, 1, 5, 5},
                                                Padding::kSame,
                                                {3, 3, 1, 1},
                                                {3, 3, 1, 1}},
                      SelectAndScatterTestParam{{7, 7, 8, 256},
                                                {4, 4, 8, 256},
                                                Padding::kSame,
                                                {2, 2, 1, 1},
                                                {2, 2, 1, 1}}));

// Test for F32 1D array, with a zero-element input.
XLA_TEST_F(SelectAndScatterTest, R1S0F32) {
  const auto operand = builder_.ConstantR1<float>({});
  const auto source = builder_.ConstantR1<float>({});
  builder_.SelectAndScatter(operand, ge_f32_, /*window_dimensions=*/{3},
                            /*window_strides=*/{3}, Padding::kValid, source,
                            builder_.ConstantR0<float>(0.0f), add_f32_);
  ComputeAndCompareR1<float>(&builder_, {}, {}, ErrorSpec(1e-7));
}

// Test for F32 1D array, when windows do not overlap.
XLA_TEST_F(SelectAndScatterTest, R1F32) {
  const auto operand =
      builder_.ConstantR1<float>({1.f, 9.f, 3.f, 7.f, 5.f, 6.f});
  const auto source = builder_.ConstantR1<float>({34.f, 42.f});
  const std::vector<float> expected = {0.f, 34.f, 0.f, 42.f, 0.f, 0.f};
  builder_.SelectAndScatter(operand, ge_f32_, /*window_dimensions=*/{3},
                            /*window_strides=*/{3}, Padding::kValid, source,
                            builder_.ConstantR0<float>(0.0f), add_f32_);
  ComputeAndCompareR1<float>(&builder_, expected, {}, ErrorSpec(1e-7));
}

// Test for S32 1D array, when windows do not overlap and the init value is 1.
XLA_TEST_F(SelectAndScatterTest, R1S32) {
  const auto operand = builder_.ConstantR1<int32>({-1, 0, 6, 4, -4, 10});
  const auto source = builder_.ConstantR1<int32>({-10, 20});
  const std::vector<int32> expected = {1, 1, -9, 1, 1, 21};
  builder_.SelectAndScatter(operand, ge_s32_, /*window_dimensions=*/{3},
                            /*window_strides=*/{3}, Padding::kValid, source,
                            builder_.ConstantR0<int32>(1), add_s32_);
  ComputeAndCompareR1<int32>(&builder_, expected, {});
}

// Test for S32 1D array, when windows overlap with each other.
XLA_TEST_F(SelectAndScatterTest, R1S32OverlappingWindow) {
  const auto operand = builder_.ConstantR1<int32>({1, 9, 3, 7, 5, 6});
  const auto source = builder_.ConstantR1<int32>({34, 42, 53, 19});
  const std::vector<int32> expected = {0, 76, 0, 72, 0, 0};
  builder_.SelectAndScatter(operand, ge_s32_, /*window_dimensions=*/{3},
                            /*window_strides=*/{1}, Padding::kValid, source,
                            builder_.ConstantR0<int32>(0), add_s32_);
  ComputeAndCompareR1<int32>(&builder_, expected, {});
}

// Test for S32 2D array, when windows do not overlap.
XLA_TEST_F(SelectAndScatterTest, R2S32) {
  const auto operand =
      builder_.ConstantR2<int32>({{7, 2, 5, 3, 10, 2}, {3, 8, 9, 3, 4, 2}});
  const auto source = builder_.ConstantR2<int32>({{2, 6}});
  Array2D<int32> expected({{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}});
  builder_.SelectAndScatter(operand, ge_s32_, /*window_dimensions=*/{2, 3},
                            /*window_strides=*/{2, 3}, Padding::kValid, source,
                            builder_.ConstantR0<int32>(0), add_s32_);
  ComputeAndCompareR2<int32>(&builder_, expected, {});
}

// Similar to SelectAndScatterTest.R2S32 but the input is transposed.
XLA_TEST_F(SelectAndScatterTest, ReshapeR2S32) {
  const auto operand = builder_.ConstantR2<int32>(
      {{7, 3}, {2, 8}, {5, 9}, {3, 3}, {10, 4}, {2, 2}});
  const auto reshape =
      builder_.Reshape(operand, /*dimensions=*/{1, 0}, /*new_sizes=*/{2, 6});
  const auto source = builder_.ConstantR2<int32>({{2, 6}});
  Array2D<int32> expected({{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}});
  builder_.SelectAndScatter(reshape, ge_s32_, /*window_dimensions=*/{2, 3},
                            /*window_strides=*/{2, 3}, Padding::kValid, source,
                            builder_.ConstantR0<int32>(0), add_s32_);
  ComputeAndCompareR2<int32>(&builder_, expected, {});
}

// Test for S32 2D array, when windows overlap with each other.
XLA_TEST_F(SelectAndScatterTest, R2S32OverlappingWindow) {
  const auto operand =
      builder_.ConstantR2<int32>({{7, 2, 5, 3, 8}, {3, 8, 9, 3, 4}});
  const auto source = builder_.ConstantR2<int32>({{2, 6, 4}});
  Array2D<int32> expected({{0, 0, 0, 0, 0}, {0, 0, 12, 0, 0}});
  builder_.SelectAndScatter(operand, ge_s32_, /*window_dimensions=*/{2, 3},
                            /*window_strides=*/{1, 1}, Padding::kValid, source,
                            builder_.ConstantR0<int32>(0), add_s32_);
  ComputeAndCompareR2<int32>(&builder_, expected, {});
}

// Test for S32 2D array, when the padding is Padding::kSAME.
XLA_TEST_F(SelectAndScatterTest, R2S32SamePadding) {
  const auto operand =
      builder_.ConstantR2<int32>({{7, 2, 5, 3, 8}, {3, 8, 9, 3, 4}});
  const auto source = builder_.ConstantR2<int32>({{2, 6, 4}});
  Array2D<int32> expected({{0, 0, 0, 0, 4}, {0, 2, 6, 0, 0}});
  builder_.SelectAndScatter(operand, ge_s32_, /*window_dimensions=*/{2, 2},
                            /*window_strides=*/{2, 2}, Padding::kSame, source,
                            builder_.ConstantR0<int32>(0), add_s32_);
  ComputeAndCompareR2<int32>(&builder_, expected, {});
}

// Test for S32 2D array, when the padding is Padding::kSAME and windows overlap
// with each other.
XLA_TEST_F(SelectAndScatterTest, R2S32SamePaddingOverlappingWindow) {
  const auto operand =
      builder_.ConstantR2<int32>({{7, 2, 5, 3, 8}, {3, 8, 9, 3, 4}});
  const auto source =
      builder_.ConstantR2<int32>({{2, 6, 4, 7, 1}, {3, 5, 8, 9, 10}});
  Array2D<int32> expected({{0, 0, 0, 0, 8}, {0, 5, 23, 0, 19}});
  builder_.SelectAndScatter(operand, ge_s32_, /*window_dimensions=*/{2, 2},
                            /*window_strides=*/{1, 1}, Padding::kSame, source,
                            builder_.ConstantR0<int32>(0), add_s32_);
  ComputeAndCompareR2<int32>(&builder_, expected, {});
}

XLA_TEST_F(SelectAndScatterTest, R2F32OverlappingR2Source) {
  const auto operand = builder_.ConstantR2<float>(
      {{1.5f, 2.5f, 1.5f}, {3.5f, 1.5f, 3.5f}, {4.5f, 2.5f, 4.5f}});
  const auto source = builder_.ConstantR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}});
  Array2D<float> expected(
      {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 2.0f}, {3.0f, 0.0f, 4.0f}});
  builder_.SelectAndScatter(operand, ge_f32_, /*window_dimensions=*/{2, 2},
                            /*window_strides=*/{1, 1}, Padding::kValid, source,
                            builder_.ConstantR0<float>(0.0f), add_f32_);
  ComputeAndCompareR2<float>(&builder_, expected, {}, ErrorSpec(1e-7));
}

TEST_F(SelectAndScatterTest, R4F32Valid) {
  Array2D<float> pzo = {{7.0f, 2.0f, 5.0f, 3.0f, 10.0f, 2.0f},
                        {3.0f, 8.0f, 9.0f, 3.0f, 4.00f, 2.0f},
                        {1.0f, 5.0f, 7.0f, 5.0f, 6.00f, 1.0f},
                        {0.0f, 6.0f, 2.0f, 7.0f, 2.00f, 8.0f}};
  Array2D<float> pzs = {{2.0f, 6.0f}, {3.0f, 1.0f}};
  Array2D<float> pze = {{0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 0.0f},
                        {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}};
  Array4D<float> o(4, 6, 15, 220);
  o.FillWithPZ(pzo);
  auto operand = builder_.ConstantR4FromArray4D(o);
  Array4D<float> e(4, 6, 15, 220);
  e.FillWithPZ(pze);
  Array4D<float> s(2, 2, 15, 220);
  s.FillWithPZ(pzs);
  auto source = builder_.ConstantR4FromArray4D(s);
  s.FillWithPZ(pzs);
  builder_.SelectAndScatter(operand, ge_f32_, {2, 3, 1, 1}, {2, 3, 1, 1},
                            Padding::kValid, source,
                            builder_.ConstantR0<float>(0.0f), add_f32_);
  ComputeAndCompareR4<float>(&builder_, e, {}, ErrorSpec(1e-7));
}

TEST_F(SelectAndScatterTest, R4F32Overlap) {
  Array2D<float> pzo = {{7.0f, 2.0f, 5.0f, 3.0f, 8.0f},
                        {3.0f, 8.0f, 9.0f, 3.0f, 4.0f},
                        {1.0f, 5.0f, 7.0f, 5.0f, 6.0f},
                        {0.0f, 6.0f, 2.0f, 10.0f, 2.0f}};
  Array2D<float> pzs = {{2.0f, 6.0f}, {3.0f, 1.0f}};
  Array2D<float> pze = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 8.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 3.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 0.0f, 1.0f, 0.0f}};
  Array4D<float> o(4, 5, 17, 128);
  o.FillWithPZ(pzo);
  auto operand = builder_.ConstantR4FromArray4D(o);
  Array4D<float> e(4, 5, 17, 128);
  e.FillWithPZ(pze);
  Array4D<float> s(2, 2, 17, 128);
  s.FillWithPZ(pzs);
  auto source = builder_.ConstantR4FromArray4D(s);
  s.FillWithPZ(pzs);
  builder_.SelectAndScatter(operand, ge_f32_, {2, 3, 1, 1}, {2, 2, 1, 1},
                            Padding::kValid, source,
                            builder_.ConstantR0<float>(0.0f), add_f32_);
  ComputeAndCompareR4<float>(&builder_, e, {}, ErrorSpec(1e-7));
}

TEST_F(SelectAndScatterTest, R4F32OverlapSmall) {
  Array2D<float> pzo = {{7.0f, 2.0f, 5.0f, 3.0f, 8.0f},
                        {3.0f, 8.0f, 9.0f, 3.0f, 4.0f},
                        {1.0f, 5.0f, 7.0f, 5.0f, 6.0f},
                        {0.0f, 6.0f, 2.0f, 10.0f, 2.0f}};
  Array2D<float> pzs = {{2.0f, 6.0f}, {3.0f, 1.0f}};
  Array2D<float> pze = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 8.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 3.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 0.0f, 1.0f, 0.0f}};
  Array4D<float> o(4, 5, 1, 1);
  o.FillWithPZ(pzo);
  auto operand = builder_.ConstantR4FromArray4D(o);
  Array4D<float> e(4, 5, 1, 1);
  e.FillWithPZ(pze);
  Array4D<float> s(2, 2, 1, 1);
  s.FillWithPZ(pzs);
  auto source = builder_.ConstantR4FromArray4D(s);
  s.FillWithPZ(pzs);
  builder_.SelectAndScatter(operand, ge_f32_, {2, 3, 1, 1}, {2, 2, 1, 1},
                            Padding::kValid, source,
                            builder_.ConstantR0<float>(0.0f), add_f32_);
  ComputeAndCompareR4<float>(&builder_, e, {}, ErrorSpec(1e-7));
}

TEST_F(SelectAndScatterTest, R4F32RefValidFixedSmall) {
  // This test is testing the Reference Util
  Array2D<float> pzo = {{7.0f, 2.0f, 5.0f, 3.0f, 10.0f, 2.0f},
                        {3.0f, 8.0f, 9.0f, 3.0f, 4.00f, 2.0f},
                        {1.0f, 5.0f, 7.0f, 5.0f, 6.00f, 1.0f},
                        {0.0f, 6.0f, 2.0f, 7.0f, 2.00f, 8.0f}};
  Array2D<float> pzs = {{2.0f, 6.0f}, {3.0f, 1.0f}};
  Array4D<float> o(4, 6, 4, 4);
  o.FillWithPZ(pzo);
  auto operand = builder_.ConstantR4FromArray4D(o);
  Array4D<float> s(2, 2, 4, 4);
  s.FillWithPZ(pzs);

  auto source = builder_.ConstantR4FromArray4D(s);
  s.FillWithPZ(pzs);
  builder_.SelectAndScatter(operand, ge_f32_, {2, 3, 1, 1}, {2, 3, 1, 1},
                            Padding::kValid, source,
                            builder_.ConstantR0<float>(0.0f), add_f32_);
  auto e = ReferenceUtil::SelectAndScatter4DGePlus(o, s, 0.0f, {2, 3, 1, 1},
                                                   {2, 3, 1, 1}, false);
  ComputeAndCompareR4<float>(&builder_, *e, {}, ErrorSpec(1e-7));
}

XLA_TEST_F(SelectAndScatterTest, R1F32OverlappingWindowMaxScatter) {
  const auto operand = builder_.ConstantR1<float>({1, 2, 3, 100, 3, 2, 1});
  const auto source = builder_.ConstantR1<float>({34, 42, 53, 19});
  const std::vector<float> expected = {0, 0, 0, 53, 0, 0, 0};
  builder_.SelectAndScatter(operand, ge_f32_, /*window_dimensions=*/{4},
                            /*window_strides=*/{1}, Padding::kValid, source,
                            builder_.ConstantR0<float>(0), max_f32_);
  ComputeAndCompareR1<float>(&builder_, expected, {}, ErrorSpec(1e-7));
}

XLA_TEST_F(SelectAndScatterTest, R1F32OverlappingWindowMinScatter) {
  const auto operand = builder_.ConstantR1<float>({1, 2, 3, 100, 3, 2, 1});
  const auto source = builder_.ConstantR1<float>({34, 42, 53, 19});
  const float max_float = std::numeric_limits<float>::max();
  const std::vector<float> expected = {max_float, max_float, max_float, 19,
                                       max_float, max_float, max_float};
  builder_.SelectAndScatter(operand, ge_f32_, /*window_dimensions=*/{4},
                            /*window_strides=*/{1}, Padding::kValid, source,
                            builder_.ConstantR0<float>(max_float), min_f32_);
  ComputeAndCompareR1<float>(&builder_, expected, {}, ErrorSpec(1e-7));
}

}  // namespace
}  // namespace xla
