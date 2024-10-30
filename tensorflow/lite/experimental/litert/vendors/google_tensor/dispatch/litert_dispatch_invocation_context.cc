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

#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/litert_dispatch_invocation_context.h"

#include <cstddef>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/core/utils.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"

namespace {

constexpr const size_t kEdgeTpuPadding = 64;

inline constexpr auto Pad(auto x, auto align) {
  return ((x + align - 1) / align) * align;
}

absl::StatusOr<LiteRtTensorBufferRequirements> GetTensorBufferRequirements(
    const LiteRtRankedTensorType& tensor_type) {
  auto* tensor_strides = tensor_type.layout.strides;
  if (tensor_strides != nullptr) {
    return absl::InternalError(
        "Tensor strides are not supported on GoogleTensor");
  }

  LiteRtTensorBufferType supported_tensor_buffer_types[] = {
      kLiteRtTensorBufferTypeAhwb,
  };
  int num_supported_tensor_buffer_types =
      sizeof(supported_tensor_buffer_types) /
      sizeof(supported_tensor_buffer_types[0]);

  auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size.ok()) {
    return buffer_size.status();
  }

  size_t padded_buffer_size = Pad(*buffer_size, kEdgeTpuPadding);

  LiteRtTensorBufferRequirements requirements;
  if (auto status = LiteRtCreateTensorBufferRequirements(
          num_supported_tensor_buffer_types, supported_tensor_buffer_types,
          padded_buffer_size, /*num_strides=*/0, /*strides=*/nullptr,
          &requirements);
      status != kLiteRtStatusOk) {
    return absl::InternalError("Not implemented");
  }

  return requirements;
}
}  // namespace

absl::StatusOr<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

absl::StatusOr<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}
