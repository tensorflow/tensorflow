/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/make_batch_pointers.h"

#include <cstddef>

#include "absl/status/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/make_batch_pointers_kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {

absl::Status MakeBatchPointers(se::Stream* stream,
                               se::DeviceMemoryBase base_ptr,
                               size_t stride_bytes, size_t n,
                               se::DeviceMemoryBase ptrs_out) {
  se::StreamExecutor* executor = stream->parent();
  size_t threads_per_block = [&] {
    if (executor->GetPlatform()->id() ==
        stream_executor::rocm::kROCmPlatformId) {
      return 256;
    } else {
      return 128;
    }
  }();

  TF_ASSIGN_OR_RETURN(
      auto kernel,
      stream_executor::gpu::GpuKernelRegistry::GetGlobalRegistry()
          .LoadKernel<stream_executor::gpu::MakeBatchPointersKernel>(executor));

  return kernel.Launch(se::ThreadDim(threads_per_block, 1, 1),
                       se::BlockDim(CeilOfRatio(n, threads_per_block), 1, 1),
                       stream, base_ptr, stride_bytes, n, ptrs_out);
}

}  // namespace xla::gpu
