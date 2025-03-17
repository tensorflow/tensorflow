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

#include <array>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"

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

TEST(TensorBufferRequirements, Owned) {
  auto requirements = litert::TensorBufferRequirements::Create(
      absl::MakeSpan(kSupportedTensorBufferTypes,
                     kNumSupportedTensorBufferTypes),
      kBufferSize);
  ASSERT_TRUE(requirements);

  auto supported_types = requirements->SupportedTypes();
  ASSERT_TRUE(supported_types);
  ASSERT_EQ(supported_types->size(), kNumSupportedTensorBufferTypes);
  for (auto i = 0; i < supported_types->size(); ++i) {
    ASSERT_EQ((*supported_types)[i], kSupportedTensorBufferTypes[i]);
  }

  auto size = requirements->BufferSize();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, kBufferSize);
}

TEST(TensorBufferRequirements, NotOwned) {
  LiteRtTensorBufferRequirements litert_requirements;
  ASSERT_EQ(LiteRtCreateTensorBufferRequirements(
                kNumSupportedTensorBufferTypes, kSupportedTensorBufferTypes,
                kBufferSize, /*num_strides=*/0, /*strides=*/nullptr,
                &litert_requirements),
            kLiteRtStatusOk);

  litert::TensorBufferRequirements requirements(litert_requirements,
                                                /*owned=*/false);

  auto supported_types = requirements.SupportedTypes();
  ASSERT_TRUE(supported_types);
  ASSERT_EQ(supported_types->size(), kNumSupportedTensorBufferTypes);
  for (auto i = 0; i < supported_types->size(); ++i) {
    ASSERT_EQ((*supported_types)[i], kSupportedTensorBufferTypes[i]);
  }

  auto size = requirements.BufferSize();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, kBufferSize);

  ASSERT_EQ(requirements.Get(), litert_requirements);

  LiteRtDestroyTensorBufferRequirements(litert_requirements);
}

TEST(TensorBufferRequirements, WithStrides) {
  constexpr std::array<uint32_t, 3> kStrides = {1, 2, 3};

  auto requirements = litert::TensorBufferRequirements::Create(
      absl::MakeSpan(kSupportedTensorBufferTypes,
                     kNumSupportedTensorBufferTypes),
      kBufferSize, absl::MakeSpan(kStrides.data(), kStrides.size()));
  ASSERT_TRUE(requirements);

  auto strides = requirements->Strides();
  ASSERT_TRUE(strides);
  ASSERT_EQ(strides->size(), kStrides.size());
  for (auto i = 0; i < kStrides.size(); ++i) {
    ASSERT_EQ((*strides)[i], kStrides[i]);
  }
}
