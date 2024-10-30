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

TEST(TensorBufferRequirements, SimpleTest) {
  LiteRtTensorBufferRequirements requirements;
  ASSERT_EQ(LiteRtCreateTensorBufferRequirements(kNumSupportedTensorBufferTypes,
                                                 kSupportedTensorBufferTypes,
                                                 kBufferSize, &requirements),
            kLiteRtStatusOk);

  int num_types;
  ASSERT_EQ(LiteRtGetTensorBufferRequirementsNumSupportedTensorBufferTypes(
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
