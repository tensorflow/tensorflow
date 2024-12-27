/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_KERNELS_PTX_CUSTOM_KERNEL_H_
#define XLA_SERVICE_GPU_KERNELS_PTX_CUSTOM_KERNEL_H_

#include <cstddef>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::kernel {

absl::StatusOr<CustomKernel> GetPtxCustomKernel(std::string kernel_name,
                                                absl::string_view ptx,
                                                int num_args,
                                                se::BlockDim block_dim,
                                                se::ThreadDim thread_dim,
                                                size_t shared_memory_bytes = 0);
}

#endif  // XLA_SERVICE_GPU_KERNELS_PTX_CUSTOM_KERNEL_H_
