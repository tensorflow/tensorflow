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

#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"

#include <array>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"

namespace {

constexpr const LiteRtTensorBufferType kSupportedTensorBufferTypes[] = {
    kLiteRtTensorBufferTypeHostMemory,
    kLiteRtTensorBufferTypeAhwb,
    kLiteRtTensorBufferTypeIon,
    kLiteRtTensorBufferTypeFastRpc,
};

constexpr const size_t kNumSupportedTensorBufferTypes =
    sizeof(kSupportedTensorBufferTypes) /
    sizeof(kSupportedTensorBufferTypes[0]);

constexpr const size_t kBufferSize = 1234;

}  // namespace

TEST(TensorBufferRequirements, NoStrides) {
  LiteRtTensorBufferRequirements requirements;
  ASSERT_EQ(LiteRtCreateTensorBufferRequirements(
                kNumSupportedTensorBufferTypes, kSupportedTensorBufferTypes,
                kBufferSize,
                /*num_strides=*/0, /*strides=*/nullptr, &requirements),
            kLiteRtStatusOk);

  int num_types;
  ASSERT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                requirements, &num_types),
            kLiteRtStatusOk);
  ASSERT_EQ(num_types, kNumSupportedTensorBufferTypes);

  for (auto i = 0; i < num_types; ++i) {
    LiteRtTensorBufferType type;
    ASSERT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                  requirements, i, &type),
              kLiteRtStatusOk);
    ASSERT_EQ(type, kSupportedTensorBufferTypes[i]);
  }

  size_t size;
  ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(requirements, &size),
            kLiteRtStatusOk);
  ASSERT_EQ(size, kBufferSize);

  LiteRtDestroyTensorBufferRequirements(requirements);
}

TEST(TensorBufferRequirements, WithStrides) {
  constexpr std::array<uint32_t, 3> kStrides = {1, 2, 3};

  LiteRtTensorBufferRequirements requirements;
  ASSERT_EQ(LiteRtCreateTensorBufferRequirements(
                kNumSupportedTensorBufferTypes, kSupportedTensorBufferTypes,
                kBufferSize, kStrides.size(), kStrides.data(), &requirements),
            kLiteRtStatusOk);

  int num_strides;
  const uint32_t* strides;
  ASSERT_EQ(LiteRtGetTensorBufferRequirementsStrides(requirements, &num_strides,
                                                     &strides),
            kLiteRtStatusOk);
  ASSERT_EQ(num_strides, kStrides.size());
  for (auto i = 0; i < kStrides.size(); ++i) {
    ASSERT_EQ(strides[i], kStrides[i]);
  }

  LiteRtDestroyTensorBufferRequirements(requirements);
}
