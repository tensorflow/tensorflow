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

#include "xla/service/gpu/legacy_gpu_aot_compilation_result.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::unique_ptr<LegacyGpuAotCompilationResult>>
LegacyGpuAotCompilationResult::FromModule(
    const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
    absl::string_view asm_text, absl::Span<const uint8_t> binary,
    const BinaryMap& dnn_compiled_graphs, int pointer_size,
    Compiler* compiler) {
  tsl::profiler::TraceMe traceme("ResultFromModule");
  GpuExecutableProto proto;
  *proto.mutable_hlo_module_with_config() = hlo_module->ToProtoWithConfig();
  *proto.mutable_buffer_assignment() = buffer_assignment->ToProto();
  proto.set_asm_text(asm_text);
  proto.set_binary(binary.data(), binary.size());
  proto.mutable_dnn_compiled_graphs()->insert(dnn_compiled_graphs.cbegin(),
                                              dnn_compiled_graphs.cend());
  return std::unique_ptr<LegacyGpuAotCompilationResult>(
      new LegacyGpuAotCompilationResult(hlo_module->Clone(), std::move(proto),
                                        pointer_size, compiler));
}

absl::StatusOr<std::unique_ptr<LegacyGpuAotCompilationResult>>
LegacyGpuAotCompilationResult::FromString(const std::string& serialized,
                                          int pointer_size,
                                          Compiler* compiler) {
  tsl::profiler::TraceMe traceme("ResultFromString");
  GpuExecutableProto proto;
  if (!proto.ParseFromString(serialized)) {
    return Internal(
        "Failed to parse serialized LegacyGpuAotCompilationResult.");
  }

  return FromProto(proto, pointer_size, compiler);
}

absl::StatusOr<std::unique_ptr<LegacyGpuAotCompilationResult>>
LegacyGpuAotCompilationResult::FromProto(const GpuExecutableProto& proto,
                                         int pointer_size, Compiler* compiler) {
  tsl::profiler::TraceMe traceme("ResultFromProto");
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      HloModule::CreateFromProtoWithConfig(proto.hlo_module_with_config()));
  return std::unique_ptr<LegacyGpuAotCompilationResult>(
      new LegacyGpuAotCompilationResult(std::move(module), std::move(proto),
                                        pointer_size, compiler));
}

absl::StatusOr<std::string> LegacyGpuAotCompilationResult::SerializeAsString()
    const {
  std::string serialized;
  if (!tsl::SerializeToStringDeterministic(proto_, &serialized)) {
    return Internal(
        "Failed to serialize LegacyGpuAotCompilationResult deterministically.");
  }
  return serialized;
}

absl::StatusOr<std::unique_ptr<Executable>>
LegacyGpuAotCompilationResult::LoadExecutable(
    se::Platform::Id platform_id,
    const se::DeviceDescription& device_description,
    const DebugOptions& debug_options) && {
  return compiler_->LoadExecutableFromAotResult(*this, device_description);
}

absl::StatusOr<CompiledMemoryStats>
LegacyGpuAotCompilationResult::GetCompiledMemoryStats() const {
  CompiledMemoryStats memory_stats;
  memory_stats.serialized_buffer_assignment =
      proto_.buffer_assignment().SerializeAsString();

  std::vector<BufferAllocation> allocations;
  allocations.reserve(proto_.buffer_assignment().buffer_allocations_size());
  for (const BufferAllocationProto& allocation :
       proto_.buffer_assignment().buffer_allocations()) {
    allocations.push_back(BufferAllocation::FromProto(allocation));
  }
  std::vector<const BufferAllocation*> alloc_ptrs;
  alloc_ptrs.reserve(allocations.size());
  for (const BufferAllocation& alloc : allocations) {
    alloc_ptrs.push_back(&alloc);
  }
  memory_stats.PopulateBufferStatsFromAllocations(alloc_ptrs);
  ASSIGN_OR_RETURN(
      auto peak_memories,
      ComputePeakMemorySizes(proto_.buffer_assignment(),
                             proto_.hlo_module_with_config().hlo_module()));
  memory_stats.peak_memory_in_bytes = peak_memories.padded;
  memory_stats.peak_unpadded_heap_bytes = peak_memories.unpadded;
  memory_stats.total_allocation_bytes = ComputeTotalAllocationBytes(
      proto_.buffer_assignment(), /*memory_color=*/0);
  memory_stats.indefinite_allocations = ComputeIndefiniteAllocationsInBytes(
      proto_.buffer_assignment(), /*memory_color=*/0);
  return memory_stats;
}

}  // namespace gpu
}  // namespace xla
