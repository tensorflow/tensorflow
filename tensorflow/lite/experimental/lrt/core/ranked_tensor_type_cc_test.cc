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
constexpr const size_t kRank =
    sizeof(kTensorDimensions) / sizeof(kTensorDimensions[0]);

constexpr const LiteRtLayout kLayout = {
    /*.rank=*/kRank,
    /*.dimensions=*/kTensorDimensions,
    /*.strides=*/nullptr,
};

constexpr const LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    /*.layout=*/kLayout,
};

}  // namespace

TEST(RankedTensorType, Accessors) {
  litert::Layout layout(kLayout);
  litert::RankedTensorType tensor_type(kTensorType);
  ASSERT_EQ(tensor_type.ElementType(),
            static_cast<litert::ElementType>(kTensorType.element_type));
  ASSERT_TRUE(tensor_type.Layout() == layout);
}

TEST(Layout, Equal) {
  litert::RankedTensorType tensor_type1(kTensorType);
  litert::RankedTensorType tensor_type2({
      /*.element_type=*/kLiteRtElementTypeFloat32,
      /*.layout=*/kLayout,
  });
  ASSERT_TRUE(tensor_type1 == tensor_type2);
}

TEST(Layout, NotEqual) {
  litert::RankedTensorType tensor_type1(kTensorType);
  litert::RankedTensorType tensor_type2({
      /*.element_type=*/kLiteRtElementTypeFloat16,
      /*.layout=*/kLayout,
  });
  ASSERT_TRUE(tensor_type1 != tensor_type2);
}
