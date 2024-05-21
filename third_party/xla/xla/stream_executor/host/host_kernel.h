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
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"

namespace stream_executor::host {

class HostExecutor;

class HostKernel : public Kernel {
 public:
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

  explicit HostKernel() = default;

  // TODO(tsilytskyi): make this implementation detail private
  explicit HostKernel(unsigned arity, SE_HOST_Kernel* kernel);

  // TODO(b/331430625): Connect this API to Launch API defined at
  // StreamExecutor level, which requires refactoring how arguments passed to
  // kernels, as current KernelArgs structure tied to the GPU kernel ABI.
  absl::Status Launch(const ThreadDim& thread_dims,
                      absl::Span<const DeviceMemoryBase> buffers) const;

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
  void SetExecutionEngine(std::unique_ptr<T> execution_engine) {
    function_ = std::move(execution_engine);
  }

 private:
  std::unique_ptr<KernelFunction> function_;

  unsigned arity_;
};

inline const HostKernel* AsHostKernel(const Kernel* kernel) {
  return static_cast<const HostKernel*>(kernel);
}

inline HostKernel* AsHostKernel(Kernel* kernel) {
  return static_cast<HostKernel*>(kernel);
}

}  // namespace stream_executor::host

#endif  // XLA_STREAM_EXECUTOR_HOST_HOST_KERNEL_H_
