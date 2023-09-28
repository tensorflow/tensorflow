/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_TOPK_KERNEL_H_
#define XLA_SERVICE_GPU_RUNTIME_TOPK_KERNEL_H_

#include <stddef.h>
#include <stdint.h>

#include "absl/status/status.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/platform.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Input: [batch_size, num_elements]dtype
// Output:
//  - top_elements: [batch_size, k] dtype
//  - top_indices: [batch_size, k] u32
// Where `top_elements` contains the largest elements of the input, and
// `top_indices` their original indices.
absl::Status RunTopk(::tensorflow::se::gpu::GpuStreamHandle stream,
                     PrimitiveType dtype, void* data, size_t num_elements,
                     void* top_elements, uint32_t* top_indices, size_t k,
                     size_t batch_size);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_TOPK_KERNEL_H_
