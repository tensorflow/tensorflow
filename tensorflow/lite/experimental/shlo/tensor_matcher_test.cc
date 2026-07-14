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

#include "tensorflow/lite/experimental/shlo/tensor_matcher.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor_with_data.h"

namespace shlo_ref {
namespace {

using ::shlo_ref::testing::TensorEq;
using ::testing::Not;

TEST(TensorMatcherTest, Eq) {
  auto lhs = TensorWithData::Create<DataType::kSI8>(Shape{{1, 3}}, {5, 7, 9});
  auto rhs = TensorWithData::Create<DataType::kSI8>(Shape{{1, 3}}, {5, 7, 9});

  EXPECT_THAT(lhs.tensor(), TensorEq(rhs.tensor()));
}

TEST(TensorMatcherTest, NotEqQuantized) {
  auto lhs = TensorWithData::Create<DataType::kSI8>(Shape{{1, 3}}, {5, 7, 9});
  auto rhs = TensorWithData::Create<DataType::kSI8, DataType::kF32>(
      Shape{{1, 3}}, {.5f, 1.0f, 1.5f}, 0.1, 0);

  EXPECT_THAT(lhs.tensor(), Not(TensorEq(rhs.tensor())));
}

TEST(TensorMatcherTest, NotEqType) {
  auto lhs = TensorWithData::Create<DataType::kSI8>(Shape{{1, 3}}, {5, 7, 9});
  auto rhs = TensorWithData::Create<DataType::kSI32>(Shape{{1, 3}}, {5, 7, 9});

  EXPECT_THAT(lhs.tensor(), Not(TensorEq(rhs.tensor())));
}

TEST(TensorMatcherTest, NotEqShape) {
  auto lhs = TensorWithData::Create<DataType::kSI8>(Shape{{1, 3}}, {5, 7, 9});
  auto rhs = TensorWithData::Create<DataType::kSI8>(Shape{{3, 1}}, {5, 7, 9});

  EXPECT_THAT(lhs.tensor(), Not(TensorEq(rhs.tensor())));
}

TEST(TensorMatcherTest, NotEqData) {
  auto lhs = TensorWithData::Create<DataType::kSI8>(Shape{{1, 3}}, {5, 7, 9});
  auto rhs = TensorWithData::Create<DataType::kSI8>(Shape{{1, 3}}, {5, 11, 9});

  EXPECT_THAT(lhs.tensor(), Not(TensorEq(rhs.tensor())));
}

}  // namespace
}  // namespace shlo_ref
