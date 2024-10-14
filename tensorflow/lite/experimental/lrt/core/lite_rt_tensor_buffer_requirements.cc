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

#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer_requirements.h"

#include <cstddef>
#include <vector>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"

class LrtTensorBufferRequirementsT {
 public:
  LrtTensorBufferRequirementsT(
      int num_supported_tensor_buffer_types,
      const LrtTensorBufferType* supported_tensor_buffer_types,
      size_t buffer_size)
      : supported_buffer_types_(
            supported_tensor_buffer_types,
            supported_tensor_buffer_types + num_supported_tensor_buffer_types),
        buffer_size_(buffer_size) {}
  std::vector<LrtTensorBufferType> supported_buffer_types() const {
    return supported_buffer_types_;
  }
  size_t buffer_size() const { return buffer_size_; }

 private:
  std::vector<LrtTensorBufferType> supported_buffer_types_;
  size_t buffer_size_;
};

LrtStatus LrtCreateTensorBufferRequirements(
    int num_supported_tensor_buffer_types,
    const LrtTensorBufferType* supported_tensor_buffer_types,
    size_t buffer_size, LrtTensorBufferRequirements* requirements) {
  if (num_supported_tensor_buffer_types < 1 || !supported_tensor_buffer_types ||
      !requirements) {
    return kLrtStatusErrorInvalidArgument;
  }
  *requirements = new LrtTensorBufferRequirementsT(
      num_supported_tensor_buffer_types, supported_tensor_buffer_types,
      buffer_size);
  return kLrtStatusOk;
}

LrtStatus LrtGetTensorBufferRequirementsNumSupportedTensorBufferTypes(
    LrtTensorBufferRequirements requirements, int* num_types) {
  if (!requirements || !num_types) {
    return kLrtStatusErrorInvalidArgument;
  }
  *num_types = requirements->supported_buffer_types().size();
  return kLrtStatusOk;
}

LrtStatus LrtGetTensorBufferRequirementsSupportedTensorBufferType(
    LrtTensorBufferRequirements requirements, int type_index,
    LrtTensorBufferType* type) {
  if (!requirements || type_index < 0 ||
      type_index >= requirements->supported_buffer_types().size()) {
    return kLrtStatusErrorInvalidArgument;
  }
  *type = requirements->supported_buffer_types()[type_index];
  return kLrtStatusOk;
}

LrtStatus LrtGetTensorBufferRequirementsBufferSize(
    LrtTensorBufferRequirements requirements, size_t* buffer_size) {
  if (!requirements || !buffer_size) {
    return kLrtStatusErrorInvalidArgument;
  }
  *buffer_size = requirements->buffer_size();
  return kLrtStatusOk;
}

void LrtDestroyTensorBufferRequirements(
    LrtTensorBufferRequirements requirements) {
  delete requirements;
}
