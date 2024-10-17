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

#include "tensorflow/lite/experimental/lrt/c/litert_tensor_buffer_requirements.h"

#include <cstddef>
#include <vector>

#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_tensor_buffer.h"

class LiteRtTensorBufferRequirementsT {
 public:
  LiteRtTensorBufferRequirementsT(
      int num_supported_tensor_buffer_types,
      const LiteRtTensorBufferType* supported_tensor_buffer_types,
      size_t buffer_size)
      : supported_buffer_types_(
            supported_tensor_buffer_types,
            supported_tensor_buffer_types + num_supported_tensor_buffer_types),
        buffer_size_(buffer_size) {}
  std::vector<LiteRtTensorBufferType> supported_buffer_types() const {
    return supported_buffer_types_;
  }
  size_t buffer_size() const { return buffer_size_; }

 private:
  std::vector<LiteRtTensorBufferType> supported_buffer_types_;
  size_t buffer_size_;
};

LiteRtStatus LiteRtCreateTensorBufferRequirements(
    int num_supported_tensor_buffer_types,
    const LiteRtTensorBufferType* supported_tensor_buffer_types,
    size_t buffer_size, LiteRtTensorBufferRequirements* requirements) {
  if (num_supported_tensor_buffer_types < 1 || !supported_tensor_buffer_types ||
      !requirements) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *requirements = new LiteRtTensorBufferRequirementsT(
      num_supported_tensor_buffer_types, supported_tensor_buffer_types,
      buffer_size);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferRequirementsNumSupportedTensorBufferTypes(
    LiteRtTensorBufferRequirements requirements, int* num_types) {
  if (!requirements || !num_types) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_types = requirements->supported_buffer_types().size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
    LiteRtTensorBufferRequirements requirements, int type_index,
    LiteRtTensorBufferType* type) {
  if (!requirements || type_index < 0 ||
      type_index >= requirements->supported_buffer_types().size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *type = requirements->supported_buffer_types()[type_index];
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferRequirementsBufferSize(
    LiteRtTensorBufferRequirements requirements, size_t* buffer_size) {
  if (!requirements || !buffer_size) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *buffer_size = requirements->buffer_size();
  return kLiteRtStatusOk;
}

void LiteRtDestroyTensorBufferRequirements(
    LiteRtTensorBufferRequirements requirements) {
  delete requirements;
}
