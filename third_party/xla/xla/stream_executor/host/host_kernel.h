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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"

namespace stream_executor::host {

class HostKernel : public Kernel {
 public:
  HostKernel(unsigned arity, SE_HOST_Kernel* kernel);

  // TODO(b/331430625): Connect this API to Launch API defined at StreamExecutor
  // level, which requires refactoring how arguments passed to kernels, as
  // current KernelArgs structure tied to the GPU kernel ABI.
  absl::Status Launch(const ThreadDim& thread_dims,
                      absl::Span<const DeviceMemoryBase> buffers);

  // For host platform, we assume that a core is a thread, and we can run at
  // most one instance of a kernel on a given thread.
  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(ThreadDim,
                                                      size_t) const override {
    return 1;
  };

  unsigned Arity() const override { return arity_; };

 private:
  unsigned arity_;
  SE_HOST_Kernel* kernel_ = nullptr;
};

}  // namespace stream_executor::host

#endif  // XLA_STREAM_EXECUTOR_HOST_HOST_KERNEL_H_
