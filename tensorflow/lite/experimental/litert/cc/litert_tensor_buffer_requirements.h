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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_TENSOR_BUFFER_REQUIREMENTS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_TENSOR_BUFFER_REQUIREMENTS_H_

#include <cstddef>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_handle.h"

namespace litert {

// Requirements for allocating a TensorBuffer, typically specified by a HW
// accelerator for a given I/O tensor. C++ equivalent to
// LiteRtTensorBufferRequirements.
class TensorBufferRequirements
    : public internal::Handle<LiteRtTensorBufferRequirements,
                              LiteRtDestroyTensorBufferRequirements> {
 public:
  TensorBufferRequirements() = default;

  // Parameter `owned` indicates if the created TensorBufferRequirements object
  // should take ownership of the provided `requirements` handle.
  explicit TensorBufferRequirements(LiteRtTensorBufferRequirements requirements,
                                    bool owned = true)
      : internal::Handle<LiteRtTensorBufferRequirements,
                         LiteRtDestroyTensorBufferRequirements>(requirements,
                                                                owned) {}

  static absl::StatusOr<TensorBufferRequirements> Create(
      absl::Span<const LiteRtTensorBufferType> buffer_types,
      size_t buffer_size) {
    LiteRtTensorBufferRequirements tensor_buffer_requirements;
    if (auto status = LiteRtCreateTensorBufferRequirements(
            buffer_types.size(), buffer_types.data(), buffer_size,
            &tensor_buffer_requirements);
        status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to create tensor buffer requirements");
    }
    return TensorBufferRequirements(tensor_buffer_requirements);
  }

  absl::StatusOr<std::vector<LiteRtTensorBufferType>> SupportedTypes() const {
    int num_types;
    if (auto status =
            LiteRtGetTensorBufferRequirementsNumSupportedTensorBufferTypes(
                Get(), &num_types);
        status != kLiteRtStatusOk) {
      return absl::InternalError(
          "Failed to get the number of supported tensor types");
    }
    std::vector<LiteRtTensorBufferType> types(num_types);
    for (auto i = 0; i < num_types; ++i) {
      if (auto status =
              LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                  Get(), i, &types[i]);
          status != kLiteRtStatusOk) {
        return absl::InternalError("Failed to get supported tensor type");
      }
    }
    return types;
  }

  absl::StatusOr<size_t> BufferSize() const {
    size_t buffer_size;
    if (auto status =
            LiteRtGetTensorBufferRequirementsBufferSize(Get(), &buffer_size);
        status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to get tensor buffer size");
    }
    return buffer_size;
  }
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_TENSOR_BUFFER_REQUIREMENTS_H_
