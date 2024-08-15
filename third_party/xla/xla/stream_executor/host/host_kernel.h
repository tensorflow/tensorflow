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

#ifndef XLA_STREAM_EXECUTOR_HOST_HOST_KERNEL_H_
#define XLA_STREAM_EXECUTOR_HOST_HOST_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "tsl/platform/threadpool.h"

namespace stream_executor::host {

class HostExecutor;

class HostKernel : public Kernel {
 public:
  using Task = std::function<void()>;
  using TaskRunner = absl::AnyInvocable<void(Task)>;

  // A struct to report completion of the kernel execution.
  using LaunchEvent = tsl::Chain;

  // Virtual base class that owns the function behind the host kernel. It can be
  // a function in a jit-compiled LLVM module or simply a pointer to the
  // in-process function written in C++. HostKernel is responsible for launching
  // the kernel function owned by the KernelFunction with given user-provided
  // arguments potentially on a thread pool.
  class KernelFunction {
   public:
    virtual ~KernelFunction() = default;
    virtual SE_HOST_Kernel* kernel() const = 0;
  };

  // A wrapper around function pointer that implements SE_HOST_Kernel API.
  class KernelFunctionPtr final : public KernelFunction {
   public:
    explicit KernelFunctionPtr(SE_HOST_Kernel* ptr) : ptr_(ptr) {}
    SE_HOST_Kernel* kernel() const override { return ptr_; }

   private:
    SE_HOST_Kernel* ptr_;  // not owned
  };

  // TODO(ezhulenev): Remove this constructor as we prefer to rely on task
  // runner as it gives us more flexibility.
  explicit HostKernel(std::shared_ptr<tsl::thread::ThreadPool> thread_pool);

  // TODO(tsilytskyi): make this implementation detail private
  HostKernel(unsigned arity, SE_HOST_Kernel* kernel,
             std::shared_ptr<tsl::thread::ThreadPool> thread_pool = nullptr);

  // Calls the kernel once in the caller thread for a thread dim (0,0,0).
  // This is a fast path for small host kernels that have just one thread.
  absl::Status CallOnce(absl::Span<const SE_HOST_KernelArg> args) const;

  // Launches the kernel on the current thread by iterating over all threads in
  // `thread_dims` and calling the kernel function.
  absl::Status Launch(const ThreadDim& thread_dims,
                      absl::Span<const DeviceMemoryBase> buffers) const;
  absl::Status Launch(const ThreadDim& thread_dims,
                      absl::Span<const SE_HOST_KernelArg> args) const;

  // Launches the kernel by iterating over all threads in `thread_dims` and
  // calling `task_runner` to run individual task (implementation might decide
  // to run some of the tasks in the caller thread to save on scheduling
  // overheads). It's up to the caller to define where task runner will execute
  // the task, i.e., a common case is to launch them on a thread pool.
  //
  // The returned async value becomes available after all tasks are completed.
  // Async value returned in constructed state and the caller can access it to
  // get the number of tasks that are expected to be completed.
  tsl::AsyncValueRef<LaunchEvent> Launch(
      const ThreadDim& thread_dims, absl::Span<const DeviceMemoryBase> buffers,
      TaskRunner task_runner) const;
  tsl::AsyncValueRef<LaunchEvent> Launch(
      const ThreadDim& thread_dims, absl::Span<const SE_HOST_KernelArg> args,
      TaskRunner task_runner) const;

  // For host platform, we assume that a core is a thread, and we can run at
  // most one instance of a kernel on a given thread.
  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(ThreadDim,
                                                      size_t) const override {
    return 1;
  };

  void SetArity(unsigned arity) { arity_ = arity; };
  unsigned Arity() const override { return arity_; };

  template <typename T,
            std::enable_if_t<std::is_base_of_v<KernelFunction, T>>* = nullptr>
  void SetKernelFunction(std::unique_ptr<T> function) {
    function_ = std::move(function);
    kernel_ = function_->kernel();
  }

 private:
  std::unique_ptr<KernelFunction> function_;
  SE_HOST_Kernel* kernel_;  // pointer to the kernel owned by `function_`

  unsigned arity_;
  std::shared_ptr<tsl::thread::ThreadPool> thread_pool_;
};

inline ABSL_ATTRIBUTE_ALWAYS_INLINE absl::Status HostKernel::CallOnce(
    absl::Span<const SE_HOST_KernelArg> args) const {
  constexpr SE_HOST_KernelThreadDim kernel_thread_dims = {1, 1, 1};
  constexpr SE_HOST_KernelThread kernel_thread = {1, 1, 1};

  SE_HOST_KernelCallFrame call_frame = {&kernel_thread_dims, &kernel_thread,
                                        args.size(), args.data()};

  SE_HOST_KernelError* error = (*kernel_)(&call_frame);

  if (ABSL_PREDICT_FALSE(error != nullptr)) {
    return absl::InternalError("Failed to call host kernel");
  }

  return absl::OkStatus();
}

inline const HostKernel* AsHostKernel(const Kernel* kernel) {
  return static_cast<const HostKernel*>(kernel);
}

inline HostKernel* AsHostKernel(Kernel* kernel) {
  return static_cast<HostKernel*>(kernel);
}

}  // namespace stream_executor::host

#endif  // XLA_STREAM_EXECUTOR_HOST_HOST_KERNEL_H_
