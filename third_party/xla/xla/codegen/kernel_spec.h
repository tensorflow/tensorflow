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
#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "xla/runtime/work_cluster.h"
#include "xla/runtime/work_group.h"
#include "xla/runtime/work_item.h"
#include "xla/service/buffer_assignment.h"

namespace xla {

// KernelSpec is a specification of an XLA kernel produced by the XLA codegen.
// At XLA compilation time, backends instantiates kernel specification into run
// time instances that can be executed on the device, i.e. on GPU XLA runtime
// will load kernel PTX on device and instantiate a KernelThunk.
class KernelSpec {
 public:
  using Buffers = absl::InlinedVector<BufferAllocation::Slice, 8>;

  KernelSpec(absl::string_view name, NumWorkGroups num_workgroups,
             Buffers argument_buffers, Buffers result_buffers,
             absl::flat_hash_set<int64_t> invariant_arguments,
             std::optional<size_t> scratch_bytes = std::nullopt);

  KernelSpec(absl::string_view name, NumWorkClusters num_workclusters,
             NumWorkGroups num_workgroups, NumWorkItems num_workitems,
             Buffers argument_buffers, Buffers result_buffers,
             absl::flat_hash_set<int64_t> invariant_arguments,
             std::optional<size_t> scratch_bytes = std::nullopt);

  // Get the backend specific name of the kernel.
  // This may be used to identify the kernel in the backend specific runtime.
  const std::string& name() const { return name_; }

  // Kernel work dimensions define how the kernel execution must be
  // parallelized. The meaning of these dimensions is backend specific, i.e.
  // on GPU these dimensions are mapped to CUDA cluster, block and thread
  // dimensions, and on CPU these dimensions mapped to tasks submitted to a
  // thread pool.
  //
  // At a high level kernel codegen can rely on these dimensions to define
  // spatial partitioning of the computation problem and optimize for data
  // locality. However it's up to the backend codegen and runtime to agree
  // on the exact meaning of these dimensions and how they are mapped to the
  // underlying hardware, and how to use them for perfrormance optimization.
  NumWorkClusters num_workclusters() const { return num_workclusters_; }
  NumWorkGroups num_workgroups() const { return num_workgroups_; }
  NumWorkItems num_workitems() const { return num_workitems_; }

  // Requested amount of scratch bytes for the kernel (backed by backend
  // specific memory, i.e. on GPU this is shared memory, on CPU it can runtime
  // managed buffer that is likely to be in L1/L2 cache).
  std::optional<size_t> scratch_bytes() const { return scratch_bytes_; }

  // Argument buffers read by the kernel.
  const Buffers& argument_buffers() const { return argument_buffers_; }
  // Result buffers written to by the kernel.
  const Buffers& result_buffers() const { return result_buffers_; }

  // Returns a set of invariant arguments (corresponding to the indices in the
  // argument buffers list).
  const absl::flat_hash_set<int64_t>& invariant_arguments() const {
    return invariant_arguments_;
  }

 private:
  std::string name_;

  NumWorkClusters num_workclusters_;
  NumWorkGroups num_workgroups_;
  NumWorkItems num_workitems_;

  Buffers argument_buffers_;
  Buffers result_buffers_;

  absl::flat_hash_set<int64_t> invariant_arguments_;

  std::optional<size_t> scratch_bytes_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_KERNEL_SPEC_H_
