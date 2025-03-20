/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/hlo_cost_analysis_wrapper.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/xla_data.pb.h"

namespace tensorflow {
namespace profiler {

int64_t HloCostAnalysisWrapper::GetModelFlops(
    const xla::HloInstruction& hlo) const {
  const xla::HloCostAnalysis* cost_analysis = GetXlaCostAnalysis();
  DCHECK(cost_analysis != nullptr);
  return GetCostPerCoreFunction(hlo)(cost_analysis->flop_count(hlo));
}

int64_t HloCostAnalysisWrapper::GetDeviceFlops(
    const xla::HloInstruction& hlo) const {
  return GetModelFlops(hlo) - GetDeviceFlopsAdjustment(hlo);
}

std::unique_ptr<PerformanceInfo>
HloCostAnalysisWrapper::GeneratePerformanceInfo(
    const xla::HloInstruction& hlo) const {
  const xla::HloCostAnalysis* cost_analysis = GetXlaCostAnalysis();
  DCHECK(cost_analysis != nullptr);

  auto cost_per_core = GetCostPerCoreFunction(hlo);

  auto performance_info = std::make_unique<PerformanceInfo>();
  performance_info->set_flops(GetModelFlops(hlo));

  performance_info->set_bytes_accessed(
      cost_per_core(cost_analysis->bytes_accessed(hlo)));

  if (performance_info->bytes_accessed() > 0) {
    // Calculate per-memory-space bytes accessed and export it to
    // PerformanceInfo.
    auto add_memory_space_bytes_accessed =
        [&](int64_t bytes_accessed, bool is_read,
            PerformanceInfo::MemoryAccessed::MemorySpace memory_space) {
          bytes_accessed = cost_per_core(bytes_accessed);
          // Only export non-zero bytes.
          if (bytes_accessed == 0) {
            return;
          }
          auto* memory_accessed =
              performance_info->add_memory_accessed_breakdown();
          memory_accessed->set_is_read(is_read);
          memory_accessed->set_memory_space(memory_space);
          memory_accessed->set_bytes_accessed(bytes_accessed);
        };
    for (const auto& [memory_space_xprof, memory_space_xla] :
         GetMemorySpaceMapping()) {
      add_memory_space_bytes_accessed(
          cost_analysis->GetBytesRead(hlo, memory_space_xla),
          /*is_read=*/true, memory_space_xprof);
      add_memory_space_bytes_accessed(
          cost_analysis->GetBytesWritten(hlo, memory_space_xla),
          /*is_read=*/false, memory_space_xprof);
    }
  }
  return performance_info;
}

std::vector<uint32_t> GetInputBitwidths(const xla::HloInstruction& hlo) {
  std::vector<uint32_t> input_bitwidths;
  for (const auto& operand : hlo.operands()) {
    switch (operand->shape().element_type()) {
      case xla::PRIMITIVE_TYPE_INVALID:
      case xla::TUPLE:
      case xla::OPAQUE_TYPE:
      case xla::TOKEN:
        break;
      default:
        input_bitwidths.push_back(
            xla::primitive_util::BitWidth(operand->shape().element_type()));
    }
  }
  return input_bitwidths;
}

}  // namespace profiler
}  // namespace tensorflow
