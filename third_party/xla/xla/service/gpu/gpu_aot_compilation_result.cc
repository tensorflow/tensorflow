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

#include "xla/service/gpu/gpu_aot_compilation_result.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/overload.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "google/protobuf/arena.h"
#include "riegeli/bytes/string_writer.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/printer.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_symbol_registry.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util/split_proto/split_gpu_executable_writer.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla::gpu {

static absl::StatusOr<std::pair<std::unique_ptr<HloModule>, tsl::Fprint128>>
ParseHloModuleAndFingerprint(const HloModuleProtoWithConfig& proto) {
  ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                   HloModule::CreateFromProtoWithConfig(proto));
  HighwayHashPrinter printer;
  module->Print(&printer, HloPrintOptions::Canonical()
                              .set_print_backend_config(true)
                              .set_sort_backend_config(true));
  return std::make_pair(std::move(module), printer.ToFingerprint128());
}

absl::StatusOr<std::unique_ptr<GpuAotCompilationResult>>
GpuAotCompilationResult::FromProto(GpuExecutableProto executable_proto) {
  tsl::Fprint128 executable_fingerprint = {
      tsl::DeterministicProtoHash64(executable_proto),
      tsl::DeterministicProtoHash64(executable_proto, /*seed=*/1)};
  ASSIGN_OR_RETURN(
      auto module_and_fingerprint,
      ParseHloModuleAndFingerprint(executable_proto.hlo_module_with_config()));
  auto& [module, hlo_fingerprint] = module_and_fingerprint;
  return absl::WrapUnique(new GpuAotCompilationResult(
      std::move(executable_proto), std::move(module), hlo_fingerprint,
      executable_fingerprint));
}

absl::StatusOr<std::unique_ptr<GpuAotCompilationResult>>
GpuAotCompilationResult::FromSerialized(
    std::unique_ptr<riegeli::Reader> reader) {
  auto arena = std::make_unique<google::protobuf::Arena>();
  GpuExecutableProto* executable_proto =
      google::protobuf::Arena::Create<GpuExecutableProto>(arena.get());

  RETURN_IF_ERROR(ReadSplitProto(std::move(reader), *executable_proto));

  tsl::Fprint128 executable_fingerprint = {
      tsl::DeterministicProtoHash64(*executable_proto),
      tsl::DeterministicProtoHash64(*executable_proto, /*seed=*/1)};
  ASSIGN_OR_RETURN(
      auto module_and_fingerprint,
      ParseHloModuleAndFingerprint(executable_proto->hlo_module_with_config()));
  auto& [module, hlo_fingerprint] = module_and_fingerprint;
  return absl::WrapUnique(new GpuAotCompilationResult(
      internal::ArenaAllocatedGpuExecutableProto(std::move(arena),
                                                 executable_proto),
      std::move(module), hlo_fingerprint, executable_fingerprint));
}

absl::StatusOr<std::string> GpuAotCompilationResult::SerializeAsString() const {
  std::string serialized;
  RETURN_IF_ERROR(WriteSplitGpuExecutable(
      GetExecutableProto(),
      std::make_unique<riegeli::StringWriter<>>(&serialized)));
  return serialized;
}

absl::StatusOr<std::unique_ptr<Executable>>
GpuAotCompilationResult::LoadExecutable(
    se::Platform::Id platform_id,
    const se::DeviceDescription& device_description,
    const DebugOptions& debug_options) && {
  const auto symbol_resolver = [&](absl::string_view symbol_name) {
    stream_executor::KernelSymbolRegistry& registry =
        stream_executor::KernelSymbolRegistry::GetGlobalInstance();
    return registry.FindSymbol(symbol_name, platform_id);
  };

  VLOG(1) << absl::StrFormat(
      "GpuAotCompilationResult::LoadExecutable: module=%s "
      "num_instructions=%d hlo_fingerprint=%016x%016x "
      "executable_fingerprint=%016x%016x",
      hlo_module_->name(), hlo_module_->instruction_count(),
      hlo_fingerprint_.low64, hlo_fingerprint_.high64,
      executable_fingerprint_.low64, executable_fingerprint_.high64);

  return GpuExecutable::FromProto(GetExecutableProto(), device_description,
                                  platform_id->ToName(), debug_options,
                                  symbol_resolver);
}

const GpuExecutableProto& GpuAotCompilationResult::GetExecutableProto() const {
  return std::visit(
      absl::Overload(
          [](const internal::ArenaAllocatedGpuExecutableProto& arena_proto)
              -> const GpuExecutableProto& { return *arena_proto.proto; },
          [](const GpuExecutableProto& stack_proto)
              -> const GpuExecutableProto& { return stack_proto; }),
      gpu_executable_proto_);
}

absl::StatusOr<CompiledMemoryStats>
GpuAotCompilationResult::GetCompiledMemoryStats() const {
  CompiledMemoryStats memory_stats;
  memory_stats.serialized_buffer_assignment =
      GetExecutableProto().buffer_assignment().SerializeAsString();

  std::vector<BufferAllocation> allocations;
  allocations.reserve(
      GetExecutableProto().buffer_assignment().buffer_allocations_size());
  for (const BufferAllocationProto& allocation :
       GetExecutableProto().buffer_assignment().buffer_allocations()) {
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
      ComputePeakMemorySizes(
          GetExecutableProto().buffer_assignment(),
          GetExecutableProto().hlo_module_with_config().hlo_module()));
  memory_stats.peak_memory_in_bytes = peak_memories.padded;
  memory_stats.peak_unpadded_heap_bytes = peak_memories.unpadded;
  memory_stats.total_allocation_bytes = ComputeTotalAllocationBytes(
      GetExecutableProto().buffer_assignment(), /*memory_color=*/0);
  memory_stats.indefinite_allocations = ComputeIndefiniteAllocationsInBytes(
      GetExecutableProto().buffer_assignment(), /*memory_color=*/0);
  return memory_stats;
}

}  // namespace xla::gpu
