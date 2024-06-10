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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/logging.h"
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

absl::Status HostKernel::Launch(
    const ThreadDim& thread_dims,
    absl::Span<const DeviceMemoryBase> buffers) const {
  SE_HOST_KernelThreadDim kernel_thread_dims = {thread_dims.x, thread_dims.y,
                                                thread_dims.z};

  // Convert buffers to kernel arguments.
  std::vector<SE_HOST_KernelArg> args(buffers.size());
  for (int32_t i = 0; i < buffers.size(); ++i) {
    args[i].data = const_cast<void*>(buffers[i].opaque());
    args[i].size = buffers[i].size();
  }

  SE_HOST_Kernel* kernel = function_->kernel();

  for (uint64_t z = 0; z < thread_dims.z; ++z) {
    for (uint64_t y = 0; y < thread_dims.y; ++y) {
      for (uint64_t x = 0; x < thread_dims.x; ++x) {
        SE_HOST_KernelThread kernel_thread = {x, y, z};

        SE_HOST_KernelCallFrame call_frame = {
            &kernel_thread_dims, &kernel_thread, args.size(), args.data()};

        SE_HOST_KernelError* error = (*kernel)(&call_frame);

        if (error != nullptr) {
          return absl::InternalError("Failed to call host kernel");
        }
      }
    }
  }

  return absl::OkStatus();
}

namespace {

using CompletionEvent = HostKernel::CompletionEvent;
using TaskRunner = HostKernel::TaskRunner;

// Keep a state of an in-flight kernel execution on a heap to keep it alive
// until the last task is done.
struct HostKernelExecuteState {
  void NotifyTaskCompleted() {
    if (counter.load(std::memory_order_relaxed) == 1 ||
        counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
      event.SetStateConcrete();
    }
  }

  HostKernel::TaskRunner task_runner;

  SE_HOST_Kernel* kernel = nullptr;
  SE_HOST_KernelThreadDim thread_dims = {};
  absl::InlinedVector<SE_HOST_KernelArg, 8> args;

  std::atomic<int64_t> counter;
  tsl::AsyncValueRef<CompletionEvent> event;
};
}  // namespace

tsl::AsyncValueRef<CompletionEvent> HostKernel::Launch(
    const ThreadDim& thread_dims, absl::Span<const DeviceMemoryBase> buffers,
    TaskRunner task_runner) const {
  auto state = std::make_shared<HostKernelExecuteState>();
  state->task_runner = std::move(task_runner);
  state->kernel = function_->kernel();
  state->thread_dims = {thread_dims.x, thread_dims.y, thread_dims.z};

  // Convert buffers to kernel arguments.
  state->args.resize(buffers.size());
  for (int32_t i = 0; i < buffers.size(); ++i) {
    state->args[i].data = const_cast<void*>(buffers[i].opaque());
    state->args[i].size = buffers[i].size();
  }

  // Each logical thread will be executed as a separate task.
  size_t num_tasks = thread_dims.x * thread_dims.y * thread_dims.z;
  state->counter.store(num_tasks, std::memory_order_relaxed);
  state->event = tsl::MakeConstructedAsyncValueRef<CompletionEvent>(num_tasks);

  // Calls the kernel function for a single thread.
  auto call = [state](uint64_t x, uint64_t y, uint64_t z) {
    SE_HOST_KernelThread kernel_thread = {x, y, z};

    SE_HOST_KernelCallFrame call_frame = {&state->thread_dims, &kernel_thread,
                                          state->args.size(),
                                          state->args.data()};

    SE_HOST_KernelError* error = (*state->kernel)(&call_frame);
    CHECK(error == nullptr) << "Failed to call host kernel";

    state->NotifyTaskCompleted();
  };

  // First pass tasks to the task runner.
  for (uint64_t z = 0; z < thread_dims.z; ++z) {
    for (uint64_t y = 0; y < thread_dims.y; ++y) {
      for (uint64_t x = 0; x < thread_dims.x; ++x) {
        if (x == 0 && y == 0 && z == 0) continue;  // execute first task inline
        state->task_runner([=] { call(x, y, z); });
      }
    }
  }

  // Then call the first task in the caller thread.
  call(0, 0, 0);

  return state->event;
}

}  // namespace stream_executor::host
