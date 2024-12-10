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

#include "xla/backends/cpu/runtime/kernel.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/threadpool.h"

namespace xla::cpu {

using LaunchEvent = Kernel::LaunchEvent;

// Non-reference-counted async value ref for host kernels executed inline.
static tsl::AsyncValueRef<LaunchEvent> OkLaunchEvent() {
  static tsl::AsyncValueOwningRef<LaunchEvent>* event = [] {
    auto* storage = new tsl::internal::AsyncValueStorage<LaunchEvent>();
    return new tsl::AsyncValueOwningRef<LaunchEvent>(
        tsl::MakeAvailableAsyncValueRef<LaunchEvent>(*storage));
  }();
  return event->AsRef();
}

static absl::InlinedVector<XLA_CPU_KernelArg, 8> ConvertBuffersToKernelArgs(
    absl::Span<const Kernel::DeviceMemoryBase> buffers) {
  absl::InlinedVector<XLA_CPU_KernelArg, 8> args(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    args[i].data = const_cast<void*>(buffers[i].opaque());
    args[i].size = buffers[i].size();
  }
  return args;
}

namespace {
// Keep a state of an in-flight asynchronous kernel execution on a heap to keep
// it alive until the last task is done.
class KernelExecuteState {
 public:
  KernelExecuteState(Kernel::TaskRunner task_runner, XLA_CPU_Kernel* kernel,
                     Kernel::ThreadDim thread_dims,
                     absl::Span<const XLA_CPU_KernelArg> args);
  ~KernelExecuteState();

  // Calls a task with index `task_index` synchronously.
  void CallSync(uint64_t task_index);

  // Calls tasks in the [start_index, end_index) range asynchronously using task
  // runner to schedule work. Executes a single task in the caller thread.
  void CallAsync(uint64_t start_index, uint64_t end_index);

  tsl::AsyncValueRef<LaunchEvent> event() const { return event_.AsRef(); }

 private:
  // Converts linear task index in [0, num_tasks) to (x, y, z) coordinate. We
  // assume that `x` is the fastest iterating dimension.
  XLA_CPU_KernelThread Delinearize(uint64_t task_index);

  Kernel::TaskRunner task_runner_;
  size_t num_tasks_;

  XLA_CPU_Kernel* kernel_;
  XLA_CPU_KernelThreadDim thread_dims_;
  absl::InlinedVector<XLA_CPU_KernelArg, 8> args_;

  tsl::CountDownAsyncValueRef<LaunchEvent> event_;
};
}  // namespace

Kernel::Kernel(unsigned arity, XLA_CPU_Kernel* kernel)
    : function_(std::make_unique<KernelFunctionPtr>(kernel)),
      kernel_(function_->kernel()),
      arity_(arity) {}

absl::Status Kernel::Launch(const ThreadDim& thread_dims,
                            absl::Span<const DeviceMemoryBase> buffers) const {
  return Launch(thread_dims, ConvertBuffersToKernelArgs(buffers));
}

absl::Status Kernel::Launch(const ThreadDim& thread_dims,
                            absl::Span<const XLA_CPU_KernelArg> args) const {
  XLA_CPU_KernelThreadDim kernel_thread_dims = {
      thread_dims.x,
      thread_dims.y,
      thread_dims.z,
  };

  for (uint64_t z = 0; z < thread_dims.z; ++z) {
    for (uint64_t y = 0; y < thread_dims.y; ++y) {
      for (uint64_t x = 0; x < thread_dims.x; ++x) {
        XLA_CPU_KernelThread kernel_thread = {x, y, z};

        XLA_CPU_KernelCallFrame call_frame = {
            &kernel_thread_dims, &kernel_thread, args.size(), args.data()};

        XLA_CPU_KernelError* error = (*kernel_)(&call_frame);

        if (ABSL_PREDICT_FALSE(error != nullptr)) {
          return absl::InternalError("Failed to call host kernel");
        }
      }
    }
  }

  return absl::OkStatus();
}

tsl::AsyncValueRef<LaunchEvent> Kernel::Launch(
    const ThreadDim& thread_dims, absl::Span<const DeviceMemoryBase> buffers,
    TaskRunner task_runner) const {
  return Launch(thread_dims, ConvertBuffersToKernelArgs(buffers),
                std::move(task_runner));
}

tsl::AsyncValueRef<LaunchEvent> Kernel::Launch(
    const ThreadDim& thread_dims, absl::Span<const XLA_CPU_KernelArg> args,
    TaskRunner task_runner) const {
  size_t num_tasks = thread_dims.x * thread_dims.y * thread_dims.z;
  CHECK_GT(num_tasks, 0) << "Number of tasks must be positive";  // Crash Ok

  // Short-circuit launch with a single task and run it in the caller thread.
  if (ABSL_PREDICT_TRUE(num_tasks == 1)) {
    absl::Status launched = Launch(thread_dims, args);
    return ABSL_PREDICT_TRUE(launched.ok())
               ? OkLaunchEvent()
               : tsl::MakeErrorAsyncValueRef(std::move(launched));
  }

  // Create host kernel execute state on heap and kick-off execution.
  auto state = std::make_unique<KernelExecuteState>(std::move(task_runner),
                                                    kernel_, thread_dims, args);
  state->CallAsync(/*start_index=*/0, /*end_index=*/num_tasks);

  // Move execute state to the execute event callback to ensure that it is kept
  // alive while host kernel has pending tasks.
  auto execute_event = state->event();
  execute_event.AndThen([state = std::move(state)] {});

  return execute_event;
}

KernelExecuteState::KernelExecuteState(Kernel::TaskRunner task_runner,
                                       XLA_CPU_Kernel kernel,
                                       Kernel::ThreadDim thread_dims,
                                       absl::Span<const XLA_CPU_KernelArg> args)
    : task_runner_(std::move(task_runner)),
      num_tasks_(thread_dims.x * thread_dims.y * thread_dims.z),
      kernel_(kernel),
      thread_dims_({thread_dims.x, thread_dims.y, thread_dims.z}),
      args_(args.begin(), args.end()),
      event_(num_tasks_) {}

KernelExecuteState::~KernelExecuteState() {
  auto cnt = event_.count();
  DCHECK_EQ(cnt, 0) << "Host kernel execute state is destroyed before all "
                       "tasks are completed";
}

void KernelExecuteState::CallSync(uint64_t task_index) {
  CHECK_LT(task_index, num_tasks_) << "Task index out of range";  // Crash OK

  // Do not execute the task if the kernel execution has already failed.
  if (ABSL_PREDICT_FALSE(event_.is_error())) {
    event_.CountDown(absl::OkStatus());
    return;
  }

  XLA_CPU_KernelThread kernel_thread = Delinearize(task_index);
  XLA_CPU_KernelCallFrame call_frame = {&thread_dims_, &kernel_thread,
                                        args_.size(), args_.data()};

  XLA_CPU_KernelError* error = (*kernel_)(&call_frame);

  if (ABSL_PREDICT_TRUE(error == nullptr)) {
    event_.CountDown(absl::OkStatus());
  } else {
    event_.CountDown(absl::InternalError(
        absl::StrFormat("Failed to call host kernel: x=%d, y=%d, z=%d",
                        kernel_thread.x, kernel_thread.y, kernel_thread.z)));
  }
}

void KernelExecuteState::CallAsync(uint64_t start_index, uint64_t end_index) {
  CHECK_LT(start_index, end_index) << "Invalid task index range";  // Crash OK
  while (end_index - start_index > 1) {
    uint64_t mid_index = (start_index + end_index) / 2;
    task_runner_([self = this, mid_index, end_index] {
      self->CallAsync(mid_index, end_index);
    });
    end_index = mid_index;
  }
  CallSync(start_index);
}

XLA_CPU_KernelThread KernelExecuteState::Delinearize(uint64_t task_index) {
  uint64_t stride_z = thread_dims_.y * thread_dims_.x;
  uint64_t stride_y = thread_dims_.x;

  uint64_t z = task_index / stride_z;
  task_index = task_index % stride_z;

  uint64_t y = task_index / stride_y;
  task_index = task_index % stride_y;

  uint64_t x = task_index;

  return XLA_CPU_KernelThread{x, y, z};
}

}  // namespace xla::cpu
