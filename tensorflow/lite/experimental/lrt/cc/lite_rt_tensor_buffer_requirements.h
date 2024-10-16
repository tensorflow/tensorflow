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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_TENSOR_BUFFER_REQUIREMENTS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_TENSOR_BUFFER_REQUIREMENTS_H_

#include <cstddef>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_handle.h"

namespace lrt {

// Requirements for allocating a TensorBuffer, typically specified by a HW
// accelerator for a given I/O tensor. C++ equivalent to
// LrtTensorBufferRequirements.
class TensorBufferRequirements {
 public:
  TensorBufferRequirements() = default;

  // Parameter `owned` indicates if the created TensorBufferRequirements object
  // should take ownership of the provided `requirements` handle.
  explicit TensorBufferRequirements(LrtTensorBufferRequirements requirements,
                                    bool owned = true)
      : handle_(requirements,
                owned ? LrtDestroyTensorBufferRequirements : nullptr) {}

  static absl::StatusOr<TensorBufferRequirements> Create(
      absl::Span<const LrtTensorBufferType> buffer_types, size_t buffer_size) {
    LrtTensorBufferRequirements tensor_buffer_requirements;
    if (auto status = LrtCreateTensorBufferRequirements(
            buffer_types.size(), buffer_types.data(), buffer_size,
            &tensor_buffer_requirements);
        status != kLrtStatusOk) {
      return absl::InternalError("Failed to create tensor buffer requirements");
    }
    return TensorBufferRequirements(tensor_buffer_requirements);
  }

  // Return true if the underlying LrtTensorBufferRequirements handle is valid.
  bool IsValid() const { return handle_.IsValid(); }

  // Return the underlying LrtTensorBufferRequirements handle.
  explicit operator LrtTensorBufferRequirements() { return handle_.Get(); }

  absl::StatusOr<std::vector<LrtTensorBufferType>> SupportedTypes() const {
    int num_types;
    if (auto status =
            LrtGetTensorBufferRequirementsNumSupportedTensorBufferTypes(
                handle_.Get(), &num_types);
        status != kLrtStatusOk) {
      return absl::InternalError(
          "Failed to get the number of supported tensor types");
    }
    std::vector<LrtTensorBufferType> types(num_types);
    for (auto i = 0; i < num_types; ++i) {
      if (auto status = LrtGetTensorBufferRequirementsSupportedTensorBufferType(
              handle_.Get(), i, &types[i]);
          status != kLrtStatusOk) {
        return absl::InternalError("Failed to get supported tensor type");
      }
    }
    return types;
  }

  absl::StatusOr<size_t> BufferSize() const {
    size_t buffer_size;
    if (auto status = LrtGetTensorBufferRequirementsBufferSize(handle_.Get(),
                                                               &buffer_size);
        status != kLrtStatusOk) {
      return absl::InternalError("Failed to get tensor buffer size");
    }
    return buffer_size;
  }

 private:
  internal::Handle<LrtTensorBufferRequirements> handle_;
};

}  // namespace lrt

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_TENSOR_BUFFER_REQUIREMENTS_H_
