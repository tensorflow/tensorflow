/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/buffer_info_util.h"

#include "tensorflow/compiler/xla/cpu_function_runtime.h"

namespace xla {
namespace cpu {

using BufferInfo = cpu_function_runtime::BufferInfo;

std::vector<BufferInfo> CreateBufferInfosFromBufferAssignment(
    const BufferAssignment& buffer_assignment) {
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
  return buffer_infos;
}

std::vector<int32_t> CreateArgIndexTableFromBufferInfos(
    absl::Span<const BufferInfo> buffer_infos) {
  std::vector<int32_t> result;
  for (int64_t i = 0; i < buffer_infos.size(); i++) {
    if (buffer_infos[i].is_entry_parameter()) {
      if (buffer_infos[i].entry_parameter_number() >= result.size()) {
        result.resize(buffer_infos[i].entry_parameter_number() + 1);
      }
      result[buffer_infos[i].entry_parameter_number()] = i;
    }
  }
  return result;
}

}  // namespace cpu
}  // namespace xla
