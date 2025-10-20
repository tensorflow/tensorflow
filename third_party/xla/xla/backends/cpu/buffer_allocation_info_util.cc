/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/backends/cpu/buffer_allocation_info_util.h"

#include <cassert>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/buffer_allocation_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::cpu {

std::vector<BufferAllocationInfo> CreateBufferAllocationInfos(
    const HloModule& module, const BufferAssignment& buffer_assignment) {
  std::vector<BufferAllocationInfo> allocations;

  // A mapping from a buffer allocation index to the result parameter number.
  absl::flat_hash_map<BufferAllocation::Index, int64_t> result_allocations;
  const HloInstruction* root = module.entry_computation()->root_instruction();
  ShapeUtil::ForEachLeafShape(
      root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        BufferAllocation::Index allocation_index =
            buffer_assignment.GetUniqueSlice(root, index)->index();
        int64_t result_index = result_allocations.size();
        result_allocations[allocation_index] = result_index;
      });

  for (const BufferAllocation& allocation : buffer_assignment.Allocations()) {
    // Check that the allocations index is contiguous in [0, num_allocations).
    DCHECK_EQ(allocation.index(), allocations.size());

    if (allocation.is_thread_local()) {
      allocations.push_back(
          BufferAllocationInfo::ThreadLocal(allocation.size()));

    } else if (allocation.is_constant()) {
      allocations.push_back(BufferAllocationInfo::Constant(allocation.size()));

    } else if (allocation.is_entry_computation_parameter() &&
               allocation.maybe_live_out()) {
      // Entry computation parameter that is aliased with one of the results.
      allocations.push_back(BufferAllocationInfo::InOutParameter(
          allocation.size(), allocation.parameter_number(),
          result_allocations.at(allocation.index())));

    } else if (allocation.is_entry_computation_parameter()) {
      // A read-only entry computation parameter.
      allocations.push_back(BufferAllocationInfo::EntryParameter(
          allocation.size(), allocation.parameter_number()));

    } else if (allocation.maybe_live_out() &&
               result_allocations.contains(allocation.index())) {
      // This is a result buffer that corresponds to a flatten result index.
      allocations.push_back(BufferAllocationInfo::Result(
          allocation.size(), result_allocations[allocation.index()]));

    } else if (allocation.maybe_live_out()) {
      // This is a result buffer that holds the tuple. It doesn't correspond to
      // a flatten result index, and it's never used by XLA:CPU at run time, but
      // we still record it as we want to know about all allocations.
      allocations.push_back(
          BufferAllocationInfo::Result(allocation.size(), -1));

    } else {
      // A temporary allocation that holds intermediate buffers.
      DCHECK(allocation.IsPreallocatedTempBuffer());
      allocations.push_back(BufferAllocationInfo::Temp(allocation.size()));
    }
  }

  return allocations;
}

std::vector<int32_t> CreateArgIndexTable(
    absl::Span<const BufferAllocationInfo> allocations) {
  std::vector<int32_t> ret;
  for (int64_t i = 0; i < allocations.size(); i++) {
    if (allocations[i].is_entry_parameter()) {
      int32_t parameter_number = allocations[i].entry_parameter_number();
      if (parameter_number >= ret.size()) {
        ret.resize(parameter_number + 1);
      }
      ret[parameter_number] = i;
    }
  }
  return ret;
}

std::vector<int32_t> CreateResultIndexTable(
    absl::Span<const BufferAllocationInfo> allocations) {
  std::vector<int32_t> ret;
  for (int64_t i = 0; i < allocations.size(); i++) {
    if (allocations[i].is_result()) {
      int32_t result_number = allocations[i].result_number();
      if (result_number >= ret.size()) {
        ret.resize(result_number + 1);
      }
      ret[result_number] = i;
    }
  }
  return ret;
}

}  // namespace xla::cpu
