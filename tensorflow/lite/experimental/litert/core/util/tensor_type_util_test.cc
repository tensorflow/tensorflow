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

#include "tensorflow/lite/experimental/litert/core/util/tensor_type_util.h"

#include <array>
#include <cstdint>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"

using litert::internal::GetNumBytes;
using litert::internal::GetNumElements;
using litert::internal::GetNumPackedBytes;

TEST(TensorTypeUtil, GetNumElements) {
  constexpr std::array<int, 3> dimensions = {3, 2, 1};
  auto num_elements = GetNumElements(absl::MakeSpan(dimensions));
  EXPECT_TRUE(num_elements);
  EXPECT_EQ(*num_elements, 6);
}

TEST(TensorTypeUtil, GetNumElementsWithUnknownDimension) {
  constexpr std::array<int, 3> dimensions = {3, -1, 1};
  auto num_elements = GetNumElements(absl::MakeSpan(dimensions));
  EXPECT_FALSE(num_elements);
}

TEST(TensorTypeUtil, GetNumElementsWithZeroDimension) {
  constexpr std::array<int, 3> dimensions = {3, 0, 1};
  auto num_elements = GetNumElements(absl::MakeSpan(dimensions));
  EXPECT_FALSE(num_elements);
}

TEST(TensorTypeUtil, GetNumPackedBytes) {
  LiteRtElementType element_type = kLiteRtElementTypeInt32;
  constexpr std::array<int, 3> dimensions = {3, 2, 1};
  auto num_bytes = GetNumPackedBytes(element_type, absl::MakeSpan(dimensions));
  EXPECT_TRUE(num_bytes);
  EXPECT_EQ(*num_bytes, sizeof(int32_t) * 6);
}

TEST(TensorTypeUtil, GetNumBytes) {
  LiteRtElementType element_type = kLiteRtElementTypeInt32;
  constexpr std::array<int, 3> dimensions = {3, 2, 1};
  constexpr std::array<int, 3> strides = {1, 4, 8};
  // The data should be allocated as follows (where 'X' is a used cell and 'o'
  // is an unused/padding cell):
  //
  //     XXXo XXX
  //
  // The total is 4 + 3 = 7 cells
  auto num_bytes = GetNumBytes(element_type, absl::MakeSpan(dimensions),
                               absl::MakeSpan(strides));
  EXPECT_TRUE(num_bytes);
  EXPECT_EQ(*num_bytes, sizeof(int32_t) * 7);
}
