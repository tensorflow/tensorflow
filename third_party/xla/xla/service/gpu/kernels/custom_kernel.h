/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_KERNELS_CUSTOM_KERNEL_H_
#define XLA_SERVICE_GPU_KERNELS_CUSTOM_KERNEL_H_

#include <cstddef>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"

// WARNING: This header (and a build target) should have minimal dependencies as
// it's included into all device kernel implementations, and we want to minimize
// the number of (very expensive!) recompilations.

namespace xla::gpu {
namespace se = ::stream_executor;  // NOLINT

// Custom kernel is a mechanism for plugging pre-compiled device kernels into
// XLA GPU runtime. Custom kernel defines how to load the kernel on an executor
// and what are the grid size requirements for running it.
//
// We use this API to hide kernel implementation details from XLA (e.g. we can
// export CUTLASS gemm kernels as custom kernels to XLA), so that XLA can only
// implement generic interfaces, e.g. command buffer implementation for CUDA
// can automatically add a kernel node to a graph for arbitrary custom kernel.
//
// TODO(ezhulenev): Add custom kernel signature to track number and types of
// buffer arguments, and a way to mark one of an arguments as a workspace and
// define if it has to be zeroed first.
class CustomKernel {
 public:
  CustomKernel(std::string name, se::KernelLoaderSpec kernel_spec,
               se::BlockDim block_dims, se::ThreadDim thread_dims,
               size_t shared_memory_bytes);

  CustomKernel(std::string name, se::KernelLoaderSpec kernel_spec,
               se::BlockDim block_dims, se::ThreadDim thread_dims,
               se::ClusterDim cluster_dims, size_t shared_memory_bytes);

  absl::string_view name() const;

  const se::KernelLoaderSpec& kernel_spec() const;

  se::BlockDim block_dims() const;

  se::ThreadDim thread_dims() const;

  std::optional<se::ClusterDim> cluster_dims() const;

  size_t shared_memory_bytes() const;

  std::string ToString() const;

 private:
  std::string name_;
  se::KernelLoaderSpec kernel_spec_;
  se::BlockDim block_dims_;
  se::ThreadDim thread_dims_;
  std::optional<se::ClusterDim> cluster_dims_;
  size_t shared_memory_bytes_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_KERNELS_CUSTOM_KERNEL_H_
