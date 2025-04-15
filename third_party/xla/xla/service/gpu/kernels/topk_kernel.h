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

#ifndef XLA_SERVICE_GPU_KERNELS_TOPK_KERNEL_H_
#define XLA_SERVICE_GPU_KERNELS_TOPK_KERNEL_H_

#include <stddef.h>

#include "absl/status/status.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Input: [batch_size, num_elements]dtype
// Output:
//  - top_elements: [batch_size, k] dtype
//  - top_indices: [batch_size, k] u32
// Where `top_elements` contains the largest elements of the input, and
// `top_indices` their original indices.
absl::Status RunTopk(se::Stream* stream, PrimitiveType dtype,
                     se::DeviceMemoryBase data, size_t num_elements,
                     se::DeviceMemoryBase top_elements,
                     se::DeviceMemoryBase top_indices, size_t k,
                     size_t batch_size);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_KERNELS_TOPK_KERNEL_H_
