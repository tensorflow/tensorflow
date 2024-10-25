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

#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/cc/litert_model.h"

namespace {

constexpr const int32_t kTensorDimensions[] = {1, 2, 3};
constexpr const auto kRank =
    sizeof(kTensorDimensions) / sizeof(kTensorDimensions[0]);
constexpr const uint32_t kTensorStrides[] = {6, 3, 1};

}  // namespace

TEST(Layout, NoStrides) {
  constexpr const LiteRtLayout kLayout = {
      /*.rank=*/kRank,
      /*.dimensions=*/kTensorDimensions,
      /*.strides=*/nullptr,
  };

  litert::Layout layout(kLayout);

  ASSERT_EQ(layout.Rank(), kLayout.rank);
  for (auto i = 0; i < layout.Rank(); ++i) {
    ASSERT_EQ(layout.Dimensions()[i], kLayout.dimensions[i]);
  }
  ASSERT_FALSE(layout.HasStrides());
}

TEST(Layout, WithStrides) {
  constexpr const LiteRtLayout kLayout = {
      /*.rank=*/kRank,
      /*.dimensions=*/kTensorDimensions,
      /*.strides=*/kTensorStrides,
  };

  litert::Layout layout(kLayout);

  ASSERT_EQ(layout.Rank(), kLayout.rank);
  for (auto i = 0; i < layout.Rank(); ++i) {
    ASSERT_EQ(layout.Dimensions()[i], kLayout.dimensions[i]);
  }
  ASSERT_TRUE(layout.HasStrides());
  for (auto i = 0; i < layout.Rank(); ++i) {
    ASSERT_EQ(layout.Strides()[i], kLayout.strides[i]);
  }
}

TEST(Layout, Equal) {
  litert::Layout layout1({
      /*.rank=*/kRank,
      /*.dimensions=*/kTensorDimensions,
      /*.strides=*/kTensorStrides,
  });
  litert::Layout layout2({
      /*.rank=*/kRank,
      /*.dimensions=*/kTensorDimensions,
      /*.strides=*/kTensorStrides,
  });
  ASSERT_TRUE(layout1 == layout2);
}

TEST(Layout, NotEqual) {
  litert::Layout layout1({
      /*.rank=*/kRank,
      /*.dimensions=*/kTensorDimensions,
      /*.strides=*/nullptr,
  });
  litert::Layout layout2({
      /*.rank=*/kRank,
      /*.dimensions=*/kTensorDimensions,
      /*.strides=*/kTensorStrides,
  });
  ASSERT_FALSE(layout1 == layout2);
}
