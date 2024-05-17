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
#include "xla/stream_executor/host/host_execution_engine.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"

namespace stream_executor::host {

class HostExecutor;

class HostKernel : public Kernel {
 public:
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

  template <typename T>
  void SetExecutionEngine(std::unique_ptr<T> execution_engine) {
    static_assert(std::is_base_of<HostExecutionEngine, T>::value,
                  "T is not derived from HostExecutionEngine");
    execution_engine_ = std::move(execution_engine);
  }

 private:
  std::unique_ptr<HostExecutionEngine> execution_engine_;

  unsigned arity_;
  SE_HOST_Kernel* kernel_ = nullptr;
};

inline const HostKernel* AsHostKernel(const Kernel* kernel) {
  return static_cast<const HostKernel*>(kernel);
}

inline HostKernel* AsHostKernel(Kernel* kernel) {
  return static_cast<HostKernel*>(kernel);
}

}  // namespace stream_executor::host

#endif  // XLA_STREAM_EXECUTOR_HOST_HOST_KERNEL_H_
