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

#ifndef XLA_BACKENDS_GPU_RUNTIME_TOPK_H_
#define XLA_BACKENDS_GPU_RUNTIME_TOPK_H_

#include <cstddef>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::kernel::topk {

// Creates a CustomKernel for TopK operation.
absl::StatusOr<CustomKernel> GetTopKKernel(
    std::string name, PrimitiveType dtype, size_t num_elements, size_t k,
    size_t batch_size, absl::string_view platform_name, size_t wavefront_size);

}  // namespace xla::gpu::kernel::topk

#endif  // XLA_BACKENDS_GPU_RUNTIME_TOPK_H_
