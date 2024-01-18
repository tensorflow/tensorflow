/* Copyright 2023 The OpenXLA Authors.

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

#include <cstdint>

#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::kernel::topk {

// Fallback implementation of creating a CustomKernel for TopK operation.
absl::StatusOr<CustomKernel> GetTopKKernel(std::string name,
                                           PrimitiveType dtype,
                                           size_t num_elements, size_t k,
                                           size_t batch_size) {
  return absl::InternalError("XLA compiled without CUDA support");
}

}  // namespace xla::gpu::kernel::topk
