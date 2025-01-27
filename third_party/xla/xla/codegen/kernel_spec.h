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

#ifndef XLA_CODEGEN_KERNEL_SPEC_H_
#define XLA_CODEGEN_KERNEL_SPEC_H_

#include <cstddef>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "xla/runtime/buffer_use.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla {

// KernelSpec is a specification of an XLA kernel produced by the XLA codegen.
// At XLA compilation time, backends instantiates kernel specification into run
// time instances that can be executed on the device, i.e. on GPU XLA runtime
// will load kernel PTX on device and instantiate a KernelThunk.
class KernelSpec {
 public:
  using BufferUses = absl::InlinedVector<BufferUse, 8>;

  KernelSpec(absl::string_view name, se::ThreadDim thread_dim,
             BufferUses buffer_uses,
             std::optional<size_t> scratch_bytes = std::nullopt);

  KernelSpec(absl::string_view name, se::ClusterDim cluster_dim,
             se::BlockDim block_dim, se::ThreadDim thread_dim,
             BufferUses buffer_uses,
             std::optional<size_t> scratch_bytes = std::nullopt);

  // Get the backend specific name of the kernel.
  // Thus may be used to identify the kernel in the backend specific runtime.
  const std::string& name() const { return name_; }

  // Kernel launch dimensions define how the kernel execution must be
  // parallelized. The meaning of these dimensions is backend specific, i.e.
  // on GPU these are CUDA block and thread dimensions, and on CPU these
  // dimensions mapped to tasks submitted to a thread pool.
  //
  // At a high level kernel codegen can rely on these dimensions to define
  // spatial partitioning of the computation problem and optimize for data
  // locality. However it's up to the backend codegen and runtime to agree
  // on the exact meaning of these dimensions and how they are mapped to the
  // underlying hardware, and how to use them for perfrormance optimization.
  se::ClusterDim cluster_dim() const { return cluster_dim_; }
  se::BlockDim block_dim() const { return block_dim_; }
  se::ThreadDim thread_dim() const { return thread_dim_; }

  // Requested amount of scratch bytes for the kernel (backed by backend
  // specific memory, i.e. on GPU this is shared memory, on CPU it can runtime
  // managed buffer that is likely to be in L1/L2 cache).
  std::optional<size_t> scratch_bytes() const { return scratch_bytes_; }

  // Buffers (buffer allocation slices) used by the kernel.
  const BufferUses& buffer_uses() const { return buffer_uses_; }

 private:
  std::string name_;
  se::ClusterDim cluster_dim_;
  se::BlockDim block_dim_;
  se::ThreadDim thread_dim_;
  BufferUses buffer_uses_;
  std::optional<size_t> scratch_bytes_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_KERNEL_SPEC_H_
