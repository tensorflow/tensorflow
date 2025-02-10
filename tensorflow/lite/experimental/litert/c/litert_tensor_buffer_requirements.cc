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

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"

class LiteRtTensorBufferRequirementsT {
 public:
  LiteRtTensorBufferRequirementsT(
      int num_supported_tensor_buffer_types,
      const LiteRtTensorBufferType* supported_tensor_buffer_types,
      size_t buffer_size, std::vector<uint32_t>&& strides)
      : supported_buffer_types_(
            supported_tensor_buffer_types,
            supported_tensor_buffer_types + num_supported_tensor_buffer_types),
        buffer_size_(buffer_size),
        strides_(std::move(strides)) {}
  const std::vector<LiteRtTensorBufferType>& SupportedBufferTypes() const {
    return supported_buffer_types_;
  }
  size_t BufferSize() const { return buffer_size_; }
  const std::vector<uint32_t>& Strides() const { return strides_; }

 private:
  std::vector<LiteRtTensorBufferType> supported_buffer_types_;
  size_t buffer_size_;
  // Stride per each dimension.
  std::vector<uint32_t> strides_;
};

LiteRtStatus LiteRtCreateTensorBufferRequirements(
    int num_supported_tensor_buffer_types,
    const LiteRtTensorBufferType* supported_tensor_buffer_types,
    size_t buffer_size, int num_strides, const uint32_t* strides,
    LiteRtTensorBufferRequirements* requirements) {
  if (num_supported_tensor_buffer_types < 1 || !supported_tensor_buffer_types ||
      !requirements) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *requirements = new LiteRtTensorBufferRequirementsT(
      num_supported_tensor_buffer_types, supported_tensor_buffer_types,
      buffer_size, std::vector<uint32_t>(strides, strides + num_strides));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
    LiteRtTensorBufferRequirements requirements, int* num_types) {
  if (!requirements || !num_types) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_types = requirements->SupportedBufferTypes().size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
    LiteRtTensorBufferRequirements requirements, int type_index,
    LiteRtTensorBufferType* type) {
  if (!requirements || type_index < 0 ||
      type_index >= requirements->SupportedBufferTypes().size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *type = requirements->SupportedBufferTypes()[type_index];
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferRequirementsBufferSize(
    LiteRtTensorBufferRequirements requirements, size_t* buffer_size) {
  if (!requirements || !buffer_size) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *buffer_size = requirements->BufferSize();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferRequirementsStrides(
    LiteRtTensorBufferRequirements requirements, int* num_strides,
    const uint32_t** strides) {
  if (!requirements || !num_strides || !strides) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& s = requirements->Strides();
  *num_strides = s.size();
  *strides = s.data();
  return kLiteRtStatusOk;
}

void LiteRtDestroyTensorBufferRequirements(
    LiteRtTensorBufferRequirements requirements) {
  delete requirements;
}
