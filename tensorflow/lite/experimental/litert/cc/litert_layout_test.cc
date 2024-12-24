// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/cc/litert_layout.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert {
namespace {

using ::testing::ElementsAreArray;

static constexpr int32_t kStaticDims[] = {2, 2};
static constexpr int32_t kDynDims[] = {-1, 2};
static constexpr uint32_t kStrides[] = {1, 1};

TEST(LayoutTest, BuildFromDims) {
  auto layout = BuildLayout(kStaticDims);
  EXPECT_EQ(layout.rank, 2);
  EXPECT_THAT(DimsSpan(layout), ElementsAreArray(kStaticDims));
  EXPECT_EQ(layout.strides, nullptr);
  EXPECT_FALSE(StridesSpan(layout).has_value());
}

TEST(LayoutTest, BuildFromDimsWithStrides) {
  auto layout = BuildLayout(kStaticDims, kStrides);
  EXPECT_EQ(layout.rank, 2);
  EXPECT_THAT(DimsSpan(layout), ElementsAreArray(kStaticDims));
  auto strides = StridesSpan(layout);
  ASSERT_TRUE(strides.has_value());
  EXPECT_THAT(*strides, ElementsAreArray(kStrides));
}

TEST(LayoutTest, NumElements) {
  auto layout = BuildLayout(kStaticDims);
  auto num_elements = NumElements(layout);
  ASSERT_TRUE(num_elements.has_value());
  EXPECT_EQ(*num_elements, 4);
}

TEST(LayoutTest, NumElementsDynamic) {
  auto layout = BuildLayout(kDynDims);
  auto num_elements = NumElements(layout);
  ASSERT_FALSE(num_elements.has_value());
}

}  // namespace
}  // namespace litert
