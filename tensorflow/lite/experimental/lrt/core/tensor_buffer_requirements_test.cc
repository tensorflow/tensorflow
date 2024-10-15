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

#include <cstring>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer_requirements.h"

namespace {

constexpr const LrtTensorBufferType kSupportedTensorBufferTypes[] = {
    kLrtTensorBufferTypeHostMemory,
    kLrtTensorBufferTypeAhwb,
    kLrtTensorBufferTypeIon,
    kLrtTensorBufferTypeFastRpc,
};

constexpr const size_t kNumSupportedTensorBufferTypes =
    sizeof(kSupportedTensorBufferTypes) /
    sizeof(kSupportedTensorBufferTypes[0]);

constexpr const size_t kBufferSize = 1234;

}  // namespace

TEST(TensorBufferRequirements, SimpleTest) {
  LrtTensorBufferRequirements requirements;
  ASSERT_EQ(LrtCreateTensorBufferRequirements(kNumSupportedTensorBufferTypes,
                                              kSupportedTensorBufferTypes,
                                              kBufferSize, &requirements),
            kLrtStatusOk);

  int num_types;
  ASSERT_EQ(LrtGetTensorBufferRequirementsNumSupportedTensorBufferTypes(
                requirements, &num_types),
            kLrtStatusOk);
  ASSERT_EQ(num_types, kNumSupportedTensorBufferTypes);

  for (auto i = 0; i < num_types; ++i) {
    LrtTensorBufferType type;
    ASSERT_EQ(LrtGetTensorBufferRequirementsSupportedTensorBufferType(
                  requirements, i, &type),
              kLrtStatusOk);
    ASSERT_EQ(type, kSupportedTensorBufferTypes[i]);
  }

  size_t size;
  ASSERT_EQ(LrtGetTensorBufferRequirementsBufferSize(requirements, &size),
            kLrtStatusOk);
  ASSERT_EQ(size, kBufferSize);

  LrtDestroyTensorBufferRequirements(requirements);
}
