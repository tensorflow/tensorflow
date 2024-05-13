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

#include "xla/service/gpu/make_batch_pointers.h"

#include <cstddef>

#include "absl/status/status.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

#if TENSORFLOW_USE_ROCM
#include "xla/stream_executor/gpu/gpu_stream.h"
namespace stream_executor::gpu {

extern void rocm_MakeBatchPointers(void* stream, char* base, int stride, int n,
                                   void** ptrs_out);

}  // namespace stream_executor::gpu
#endif

namespace xla::gpu {

namespace make_batch_pointers {
void* kernel();  // returns a pointer to a CUDA C++ device function
}  // namespace make_batch_pointers

absl::Status MakeBatchPointers(se::Stream* stream,
                               se::DeviceMemoryBase base_ptr,
                               size_t stride_bytes, size_t n,
                               se::DeviceMemoryBase ptrs_out) {
  static constexpr size_t kThreads = 128;

  se::StreamExecutor* executor = stream->parent();

#if TENSORFLOW_USE_ROCM
  stream_executor::gpu::rocm_MakeBatchPointers(
      se::gpu::AsGpuStreamValue(stream),
      reinterpret_cast<char*>(base_ptr.opaque()), stride_bytes, n,
      reinterpret_cast<void**>(ptrs_out.opaque()));
#else

  TF_ASSIGN_OR_RETURN(
      auto kernel,
      (se::TypedKernel<
          se::DeviceMemoryBase, size_t, size_t,
          se::DeviceMemoryBase>::Create(executor, "make_batch_pointers",
                                        make_batch_pointers::kernel())));

  TF_RETURN_IF_ERROR(
      stream->ThenLaunch(se::ThreadDim(kThreads, 1, 1),
                         se::BlockDim(CeilOfRatio(n, kThreads), 1, 1), kernel,
                         base_ptr, stride_bytes, n, ptrs_out));
#endif
  return absl::OkStatus();
}

}  // namespace xla::gpu
