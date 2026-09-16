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
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "riegeli/bytes/string_writer.h"
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

// Fingerprint of the canonical HLO module text.
static tsl::Fprint128 HloModuleFingerprint(const HloModule& module) {
  HighwayHashPrinter printer;
  module.Print(&printer, HloPrintOptions::Canonical()
                             .set_print_backend_config(true)
                             .set_sort_backend_config(true));
  return printer.ToFingerprint128();
}

// Fingerprint of the deterministic serialization of the executable proto.
//
// NOTE: This is expensive as it re-serializes the whole proto (including all
// constants) twice, so it should only be called when its result is actually
// needed (e.g. for logging).
static tsl::Fprint128 ExecutableFingerprint(const GpuExecutableProto& proto) {
  // Fingerprint twice with differents seeds to catch non-deterministic
  // serializations.
  return {tsl::DeterministicProtoHash64(proto),
          tsl::DeterministicProtoHash64(proto, /*seed=*/1)};
}

// Formats a fingerprint as 32 hex digits, low64 first.
static std::string FprintToHex(const tsl::Fprint128& fp) {
  return absl::StrFormat("%016x%016x", fp.low64, fp.high64);
}

absl::StatusOr<std::unique_ptr<GpuAotCompilationResult>>
GpuAotCompilationResult::FromProto(GpuExecutableProto executable_proto) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                   HloModule::CreateFromProtoWithConfig(
                       executable_proto.hlo_module_with_config()));
  return absl::WrapUnique(new GpuAotCompilationResult(
      std::move(executable_proto), std::move(module)));
}

absl::StatusOr<std::unique_ptr<GpuAotCompilationResult>>
GpuAotCompilationResult::FromSerialized(
    std::unique_ptr<riegeli::Reader> reader) {
  auto arena = std::make_unique<google::protobuf::Arena>();
  GpuExecutableProto* executable_proto =
      google::protobuf::Arena::Create<GpuExecutableProto>(arena.get());

  ABSL_RETURN_IF_ERROR(ReadSplitProto(std::move(reader), *executable_proto));

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                   HloModule::CreateFromProtoWithConfig(
                       executable_proto->hlo_module_with_config()));
  return absl::WrapUnique(
      new GpuAotCompilationResult(internal::ArenaAllocatedGpuExecutableProto(
                                      std::move(arena), executable_proto),
                                  std::move(module)));
}

absl::StatusOr<std::string> GpuAotCompilationResult::SerializeAsString() const {
  std::string serialized;
  ABSL_RETURN_IF_ERROR(WriteSplitGpuExecutable(
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
      "num_instructions=%d hlo_fingerprint=%s executable_fingerprint=%s",
      hlo_module_->name(), hlo_module_->instruction_count(),
      FprintToHex(HloModuleFingerprint(*hlo_module_)),
      FprintToHex(ExecutableFingerprint(GetExecutableProto())));

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
  ABSL_ASSIGN_OR_RETURN(
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
