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

#include "tensorflow/core/profiler/utils/performance_info_wrapper.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/hlo_cost_analysis_wrapper.h"

namespace tensorflow {
namespace profiler {

MemoryAccessBreakdown PerformanceInfoWrapper::GetMemmoryAccessBreakdown()
    const {
  MemoryAccessBreakdown breakdown;
  for (const auto& m : performance_info_->memory_accessed_breakdown()) {
    auto* memory_access = breakdown.add_memory_accessed();
    memory_access->set_operation_type(m.is_read()
                                          ? OpMetrics::MemoryAccessed::READ
                                          : OpMetrics::MemoryAccessed::WRITE);
    memory_access->set_memory_space(m.memory_space());
    memory_access->set_bytes_accessed(m.bytes_accessed());
  }
  return breakdown;
}

std::unique_ptr<PerformanceInfoWrapper> PerformanceInfoWrapper::Create(
    const HloCostAnalysisWrapper* hlo_cost_analysis,
    const xla::HloInstruction* hlo_instruction) {
  std::unique_ptr<PerfInfoType> performance_info =
      hlo_cost_analysis->GeneratePerformanceInfo(*hlo_instruction);
  if (!performance_info) {
    return nullptr;
  }

  // Using `new` to access non public constructor
  return absl::WrapUnique(new PerformanceInfoWrapper(
      std::move(performance_info), hlo_instruction->opcode(),
      hlo_cost_analysis->GetDeviceFlops(*hlo_instruction),
      GetInputBitwidths(*hlo_instruction)));
}

std::unique_ptr<PerformanceInfoWrapper> PerformanceInfoWrapper::Create(
    std::unique_ptr<PerfInfoType> performance_info) {
  if (!performance_info) {
    return nullptr;
  }

  auto flops = performance_info->flops();
  std::optional<xla::HloOpcode> opcode;
  return absl::WrapUnique(new PerformanceInfoWrapper(
      std::move(performance_info), opcode, flops, /*input_bitwidths=*/{}));
}

int64_t PerformanceInfoWrapper::DeviceFlops() const { return device_flops_; }

std::vector<uint32_t> PerformanceInfoWrapper::InputBitwidths() const {
  return input_bitwidths_;
}

int64_t PerformanceInfoWrapper::ComputationalPrimitiveBitwidth() const {
  auto inputs = InputBitwidths();
  if (inputs.empty()) return 0;

  if (opcode_.has_value() && opcode_ == xla::HloOpcode::kConvolution &&
      inputs.size() == 2) {
    return *std::max_element(inputs.begin(), inputs.end());
  }
  return 0;
}
PerformanceInfoWrapper::PerformanceInfoWrapper(
    std::unique_ptr<PerfInfoType> performance_info,
    std::optional<xla::HloOpcode> opcode, int64_t device_flops,
    std::vector<uint32_t> input_bitwidths)
    : performance_info_(std::move(performance_info)),
      opcode_(opcode),
      device_flops_(device_flops),
      input_bitwidths_(input_bitwidths) {}

}  // namespace profiler
}  // namespace tensorflow
