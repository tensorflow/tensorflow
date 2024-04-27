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

#include "xla/service/cpu/buffer_info_util.h"

#include "xla/cpu_function_runtime.h"

namespace xla {
namespace cpu {

using BufferInfo = cpu_function_runtime::BufferInfo;

std::vector<BufferInfo> CreateBufferInfosFromBufferAssignment(
    const HloModule& module, const BufferAssignment& buffer_assignment) {
  std::vector<BufferInfo> buffer_infos;
  for (const BufferAllocation& allocation : buffer_assignment.Allocations()) {
    if (allocation.is_thread_local()) {
      buffer_infos.push_back(BufferInfo::MakeOnStackBuffer(allocation.size()));
    } else if (allocation.is_constant()) {
      buffer_infos.push_back(BufferInfo::MakeConstant(allocation.size()));
    } else if (allocation.is_entry_computation_parameter()) {
      buffer_infos.push_back(BufferInfo::MakeEntryParameter(
          /*size=*/allocation.size(),
          /*param_number=*/allocation.parameter_number()));
    } else {
      buffer_infos.push_back(BufferInfo::MakeTempBuffer(allocation.size()));
    }
  }

  // Fill in the result parameters' indices, expanding all tuples.
  auto root_instr = module.entry_computation()->root_instruction();
  auto output_allocation = buffer_assignment.GetUniqueTopLevelOutputSlice();
  if (output_allocation->allocation()->is_tuple()) {
    int out_index = 0;
    ShapeUtil::ForEachSubshape(
        root_instr->shape(),
        [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsTuple()) {
            return;
          }
          int64_t result_index =
              buffer_assignment.GetUniqueSlice(root_instr, index)->index();
          assert(result_index < buffer_infos.size());
          buffer_infos[result_index].set_result_parameter_number(out_index++);
        });
  }

  return buffer_infos;
}

std::vector<int32_t> CreateArgIndexTableFromBufferInfos(
    absl::Span<const BufferInfo> buffer_infos) {
  std::vector<int32_t> ret;
  for (int64_t i = 0; i < buffer_infos.size(); i++) {
    if (!buffer_infos[i].is_entry_parameter()) {
      continue;
    }
    uint64_t param_index = buffer_infos[i].entry_parameter_number();
    if (param_index >= ret.size()) {
      ret.resize(param_index + 1);
    }
    ret[param_index] = i;
  }
  return ret;
}

std::vector<int32_t> CreateResultIndexTableFromBufferInfos(
    absl::Span<const BufferInfo> buffer_infos) {
  std::vector<int32_t> ret;
  for (int64_t i = 0; i < buffer_infos.size(); i++) {
    if (!buffer_infos[i].is_result_parameter()) {
      continue;
    }
    uint64_t result_index = buffer_infos[i].result_parameter_number();
    if (result_index >= ret.size()) {
      ret.resize(result_index + 1);
    }
    ret[result_index] = i;
  }
  return ret;
}

}  // namespace cpu
}  // namespace xla
