/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/shlo/ops/is_finite.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
#include "tensorflow/lite/experimental/shlo/tensor_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor_with_data.h"

namespace shlo_ref {
namespace {

using ::shlo_ref::testing::TensorEq;

struct Params {
  TensorWithData operand;
  TensorWithData expected;
};

class IsFiniteTest : public ::testing::TestWithParam<Params> {};

TEST_P(IsFiniteTest, IsFinite) {
  const auto& params = GetParam();

  IsFiniteOp op = Create(IsFiniteOp::Attributes{});
  Tensor result{.type = params.expected.tensor().type};

  ASSERT_OK(Prepare(op, params.operand.tensor(), result));

  std::vector<std::byte> result_data(result.SizeInBytes());
  result.data = result_data.data();

  EXPECT_OK(Evaluate(op, params.operand.tensor(), result));
  EXPECT_THAT(result, TensorEq(params.expected.tensor()));
}

INSTANTIATE_TEST_SUITE_P(
    Unquantized, IsFiniteTest,
    ::testing::Values(
        Params{TensorWithData::Create<DataType::kBF16>(
                   Shape{{7}},
                   {BF16{+NAN}, BF16{-NAN}, BF16{-INFINITY}, BF16{+INFINITY},
                    BF16{-1.0f}, BF16{0.0f}, BF16{1.0f}}),
               TensorWithData::Create<DataType::kI1>(
                   Shape{{7}}, {false, false, false, false, true, true, true})},
        Params{
            TensorWithData::Create<DataType::kF16>(
                Shape{{7}}, {F16{+NAN}, F16{-NAN}, F16{-INFINITY},
                             F16{+INFINITY}, F16{-1.0f}, F16{0.0f}, F16{1.0f}}),
            TensorWithData::Create<DataType::kI1>(
                Shape{{7}}, {false, false, false, false, true, true, true})},
        Params{
            TensorWithData::Create<DataType::kF32>(
                Shape{{7}},
                {+NAN, -NAN, -INFINITY, +INFINITY, -1.0f, 0.0f, 1.0f}),
            TensorWithData::Create<DataType::kI1>(
                Shape{{7}}, {false, false, false, false, true, true, true})}));

INSTANTIATE_TEST_SUITE_P(
    Quantized, IsFiniteTest,
    ::testing::Values(Params{
        .operand = TensorWithData::Create<DataType::kSI16, DataType::kF32>(
            Shape{{7}}, {0.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f}, 0.1f, 0),
        .expected = TensorWithData::Create<DataType::kI1>(
            Shape{{7}}, {true, true, true, true, true, true, true})}));
}  // namespace
}  // namespace shlo_ref
