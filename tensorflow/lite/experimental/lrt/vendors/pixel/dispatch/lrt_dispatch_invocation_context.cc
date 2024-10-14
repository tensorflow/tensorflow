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

#include "tensorflow/lite/experimental/lrt/vendors/pixel/dispatch/lrt_dispatch_invocation_context.h"

#include <cstddef>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/lrt/core/utils.h"

namespace {

constexpr const size_t kEdgeTpuPadding = 64;

inline constexpr auto Pad(auto x, auto align) {
  return ((x + align - 1) / align) * align;
}

absl::StatusOr<LrtTensorBufferRequirements> GetTensorBufferRequirements(
    const LrtRankedTensorType& tensor_type) {
  auto* tensor_strides = tensor_type.layout.strides;
  if (tensor_strides != nullptr) {
    return absl::InternalError("Tensor strides are not supported on Pixel");
  }

  LrtTensorBufferType supported_tensor_buffer_types[] = {
      kLrtTensorBufferTypeAhwb,
  };
  int num_supported_tensor_buffer_types =
      sizeof(supported_tensor_buffer_types) /
      sizeof(supported_tensor_buffer_types[0]);

  auto buffer_size = lrt::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size.ok()) {
    return buffer_size.status();
  }

  size_t padded_buffer_size = Pad(*buffer_size, kEdgeTpuPadding);

  LrtTensorBufferRequirements requirements;
  if (auto status = LrtCreateTensorBufferRequirements(
          num_supported_tensor_buffer_types, supported_tensor_buffer_types,
          padded_buffer_size, &requirements);
      status != kLrtStatusOk) {
    return absl::InternalError("Not implemented");
  }

  return requirements;
}
}  // namespace

absl::StatusOr<LrtTensorBufferRequirements>
LrtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LrtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

absl::StatusOr<LrtTensorBufferRequirements>
LrtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LrtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}
