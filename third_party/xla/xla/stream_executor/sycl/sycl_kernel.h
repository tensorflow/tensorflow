/* Copyright 2025 The OpenXLA Authors.
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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_KERNEL_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <sycl/sycl.hpp>  // NOLINT

#include "absl/status/statusor.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::sycl {

class SyclKernel : public Kernel {
 public:
  explicit SyclKernel(StreamExecutor* executor) : executor_(executor) {}

  // Note that the function is unloaded when the module is unloaded, and the
  // module that the function is contained in is owned by the StreamExecutor.
  ~SyclKernel() override { executor_->UnloadKernel(this); }

  void set_arity(unsigned arity) { arity_ = arity; }
  unsigned Arity() const override { return arity_; }

  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(
      ThreadDim threads, size_t dynamic_shared_memory_bytes) const override;

  // Simple accessor methods.
  ::sycl::kernel* gpu_function() const { return sycl_function_; }
  void set_gpu_function(::sycl::kernel* sycl_function) {
    sycl_function_ = sycl_function;
  }

  // Collects metadata for the specified kernel.
  absl::StatusOr<KernelMetadata> GetKernelMetadata();

 private:
  absl::Status Launch(const ThreadDim& thread_dims, const BlockDim& block_dims,
                      const std::optional<ClusterDim>& cluster_dims,
                      Stream* stream, const KernelArgs& args) override;

  StreamExecutor* executor_ = nullptr;

  ::sycl::kernel* sycl_function_ = nullptr;  // wrapped SYCL kernel handle
  unsigned arity_ = 0;  // number of formal parameters the kernel takes
};

}  // namespace stream_executor::sycl

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_KERNEL_H_
