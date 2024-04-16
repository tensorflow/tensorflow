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

#include "xla/stream_executor/host/host_kernel.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/launch_dim.h"

namespace stream_executor::host {

HostKernel::HostKernel(unsigned arity, SE_HOST_Kernel* kernel)
    : arity_(arity), kernel_(kernel) {}

absl::Status HostKernel::Launch(const ThreadDim& thread_dims,
                                absl::Span<const DeviceMemoryBase> buffers) {
  SE_HOST_KernelThreadDim kernel_thread_dims = {thread_dims.x, thread_dims.y,
                                                thread_dims.z};

  // Convert buffers to kernel arguments.
  std::vector<SE_HOST_KernelArg> args(buffers.size());
  for (int32_t i = 0; i < buffers.size(); ++i) {
    args[i].data = const_cast<void*>(buffers[i].opaque());
    args[i].size = buffers[i].size();
  }

  // TODO(b/331430625): We should be using thread pool to call kernel function
  // for different threads (blocks) concurrently. For now it's the most trivial
  // implementation that runs tasks sequentially.

  for (uint64_t z = 0; z < thread_dims.z; ++z) {
    for (uint64_t y = 0; y < thread_dims.y; ++y) {
      for (uint64_t x = 0; x < thread_dims.x; ++x) {
        SE_HOST_KernelThread kernel_thread = {x, y, z};

        SE_HOST_KernelCallFrame call_frame = {
            &kernel_thread_dims, &kernel_thread, args.size(), args.data()};

        SE_HOST_KernelError* error = (*kernel_)(&call_frame);

        if (error != nullptr) {
          return absl::InternalError("Failed to call host kernel");
        }
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace stream_executor::host
