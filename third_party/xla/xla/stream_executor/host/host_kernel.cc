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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/launch_dim.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/threadpool.h"

namespace stream_executor::host {

HostKernel::HostKernel(std::shared_ptr<tsl::thread::ThreadPool> thread_pool)
    : thread_pool_(thread_pool) {
  // Kernel and arity will be set separately
}

HostKernel::HostKernel(unsigned arity, SE_HOST_Kernel* kernel,
                       std::shared_ptr<tsl::thread::ThreadPool> thread_pool)
    : function_(std::make_unique<KernelFunctionPtr>(kernel)),
      arity_(arity),
      thread_pool_(thread_pool) {}

// This function prepares call frame and does actual kernel execution.
SE_HOST_KernelError* HostKernel::worker(
    SE_HOST_Kernel* kernel, uint64_t block_size, uint64_t starting_i,
    SE_HOST_KernelThreadDim& max_dims,
    std::vector<SE_HOST_KernelArg>& args) const {
  // Compute starting coordinates
  const uint64_t sx = starting_i % max_dims.x;
  const uint64_t sy = (starting_i / max_dims.x) % max_dims.y;
  const uint64_t sz = starting_i / (max_dims.x * max_dims.y);

  // Count down to break all nested loops
  uint64_t c = block_size;

  VLOG(3) << "  Started thread {" << sx << ", " << sy << ", " << sz
          << "}, batch size is " << block_size << ".";

  SE_HOST_KernelError* res = nullptr;
  uint64_t z = sz;
  uint64_t y = sy;
  uint64_t x = sx;
  for (; z < max_dims.z; ++z) {
    for (; y < max_dims.y; ++y) {
      for (; x < max_dims.x; ++x) {
        SE_HOST_KernelThread kernel_thread = {x, y, z};
        SE_HOST_KernelCallFrame call_frame = {&max_dims, &kernel_thread,
                                              args.size(), args.data()};
        res = (*kernel)(&call_frame);
        --c;
        if (c == 0 || res != nullptr) {
          return res;
        }
      }
      x = 0;
    }
    y = 0;
  }
  return res;
};

absl::Status HostKernel::Launch(
    const ThreadDim& thread_dims,
    absl::Span<const DeviceMemoryBase> buffers) const {
  // TODO: use cost model instead of hardcoded block_size
  const uint64_t block_size = 1024;

  SE_HOST_KernelThreadDim kernel_thread_dims = {thread_dims.x, thread_dims.y,
                                                thread_dims.z};

  // Convert buffers to kernel arguments.
  std::vector<SE_HOST_KernelArg> args(buffers.size());
  for (int32_t i = 0; i < buffers.size(); ++i) {
    args[i].data = const_cast<void*>(buffers[i].opaque());
    args[i].size = buffers[i].size();
  }

  const uint64_t workload =
      kernel_thread_dims.z * kernel_thread_dims.y * kernel_thread_dims.x;

  const uint64_t num_partitions = thread_pool_ ? (workload / block_size) : 0;
  const uint64_t remainder = thread_pool_ ? workload % block_size : workload;

  SE_HOST_Kernel* kernel = function_->kernel();

  std::vector<const SE_HOST_KernelError*> statuses(num_partitions);

  VLOG(2) << "HostKernel::Launch start" << " num_partitions: " << num_partitions
          << " dims: {" << thread_dims.x << ", " << thread_dims.y << ", "
          << thread_dims.z << "}";

  // Dispatch 'num_partitions' compute functions to run in parallel.
  tsl::BlockingCounter bc(num_partitions);
  for (uint64_t i = 0; i < num_partitions; ++i) {
    const SE_HOST_KernelError** status = &statuses[i];
    const uint64_t starting_i = i * block_size;
    auto func = [this, kernel, starting_i, &kernel_thread_dims, &args, &bc,
                 status]() {
      *status =
          worker(kernel, block_size, starting_i, kernel_thread_dims, args);
      bc.DecrementCount();
    };

    thread_pool_->Schedule(func);
  }

  SE_HOST_KernelError* reminder_status = nullptr;
  if (remainder != 0) {
    const uint64_t starting_i = num_partitions * block_size;
    reminder_status =
        worker(kernel, remainder, starting_i, kernel_thread_dims, args);
  }

  bc.Wait();
  VLOG(2) << "HostKernel::Launch task execution done.";

  if (std::any_of(statuses.cbegin(), statuses.cend(),
                  [](const SE_HOST_KernelError* e) { return e != nullptr; }) ||
      reminder_status != nullptr) {
    return absl::InternalError("Failed to call host kernel");
  } else {
    return absl::OkStatus();
  }
}

}  // namespace stream_executor::host
