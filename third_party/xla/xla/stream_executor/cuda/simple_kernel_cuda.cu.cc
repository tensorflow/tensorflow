/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/stream_executor/cuda/simple_kernel_cuda.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor::cuda {
namespace {

__global__ void Write42Kernel(int32_t* out, int num_elements) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < num_elements) {
    out[index] = 42;
  }
}

}  // namespace

absl::Status LaunchWrite42Kernel(Stream* stream,
                                 stream_executor::DeviceAddress<int32_t> out,
                                 int num_elements) {
  auto cuda_stream =
      static_cast<cudaStream_t>(stream->platform_specific_handle().stream);
  Write42Kernel<<<1, num_elements, 0, cuda_stream>>>(out.base(), num_elements);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return absl::InternalError(
        absl::StrCat("Kernel launch failed: ", cudaGetErrorString(err)));
  }
  return absl::OkStatus();
}

}  // namespace stream_executor::cuda
