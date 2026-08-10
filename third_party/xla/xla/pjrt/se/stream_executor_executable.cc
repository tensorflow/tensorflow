/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/se/stream_executor_executable.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "riegeli/base/maker.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/messages/parse_message.h"
#include "xla/client/local_client.h"
#include "xla/layout.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/se/stream_executor_executable.pb.h"
#include "xla/pjrt/se/stream_executor_pjrt_abi_version.h"
#include "xla/pjrt/utils.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_layout.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/util/split_proto/split_executable_and_options_writer.h"
#include "xla/util/split_proto/split_proto_reader.h"

namespace xla {
namespace {
constexpr absl::string_view kPjRtStreamExecutorClientName =
    "PjRtStreamExecutorClient";
}  // namespace

absl::StatusOr<std::string> StreamExecutorExecutable::SerializeExecutable()
    const {
  absl::MutexLock lock(&mu_);
  if (IsEarlyExitCompilation(compile_options_)) {
    ExecutableAndOptionsProto proto;
    ABSL_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                     compile_options_.ToProto());
    *proto.mutable_pjrt_client_name() = kPjRtStreamExecutorClientName;
    std::string result;
    ABSL_RETURN_IF_ERROR(WriteSplitExecutableAndOptions(
        proto, riegeli::Maker<riegeli::StringWriter>(&result)));
    return result;
  }
  std::string serialized;
  if (std::holds_alternative<std::unique_ptr<CompiledModule>>(executables_)) {
    const auto& aot_executable =
        std::get<std::unique_ptr<CompiledModule>>(executables_);
    if (aot_executable == nullptr) {
      return absl::InternalError("No local executable");
    }
    ABSL_ASSIGN_OR_RETURN(serialized, aot_executable->SerializeAsString());
  } else {
    const auto& local_executable =
        std::get<std::shared_ptr<LocalExecutable>>(executables_);
    if (local_executable == nullptr) {
      return absl::InternalError("No local executable");
    }
    Executable* built_executable = local_executable->executable();
    CHECK(local_client_ != nullptr);
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<CompiledModule> aot_result,
        local_client_->backend().compiler()->Export(built_executable));

    ABSL_ASSIGN_OR_RETURN(serialized, aot_result->SerializeAsString());
  }
  if (serialized.empty()) {
    return absl::InternalError(
        "PjRtStreamExecutorClient::SerializeExecutable proto serialization "
        "failed");
  }
  ExecutableAndOptionsProto proto;
  *proto.mutable_serialized_executable() = std::move(serialized);
  ABSL_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                   compile_options_.ToProto());
  *proto.mutable_pjrt_client_name() = kPjRtStreamExecutorClientName;
  std::string result;
  ABSL_RETURN_IF_ERROR(WriteSplitExecutableAndOptions(
      proto, riegeli::Maker<riegeli::StringWriter>(&result)));
  return result;
}

absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
StreamExecutorExecutable::GetCostAnalysis() const {
  if (local_client_ == nullptr) {
    return absl::UnimplementedError("GetCostAnalysis is not supported.");
  }
  HloCostAnalysis cost_analysis(
      local_client_->backend().compiler()->ShapeSizeBytesFunction());
  return PjRtExecutableUtil::RunHloCostAnalysis(*this, &cost_analysis);
}

namespace {

std::unique_ptr<CompiledModule> ExtractSingleModule(
    std::vector<std::unique_ptr<CompiledModule>> executables) {
  DCHECK_LE(executables.size(), 1);
  return executables.empty() ? nullptr : std::move(executables[0]);
}

}  // namespace

StreamExecutorExecutable::StreamExecutorExecutable(
    PjRtPlatformId platform_id, const CompileOptions& compile_options,
    std::vector<std::unique_ptr<CompiledModule>> executables, int num_replicas,
    int num_partitions, absl::string_view name, absl::string_view fingerprint,
    absl::string_view default_memory_kind)
    : StreamExecutorExecutable(platform_id, compile_options,
                               ExtractSingleModule(std::move(executables)),
                               num_replicas, num_partitions, name, fingerprint,
                               default_memory_kind) {}

StreamExecutorExecutable::StreamExecutorExecutable(
    PjRtPlatformId platform_id, const CompileOptions& compile_options,
    std::unique_ptr<CompiledModule> executable, int num_replicas,
    int num_partitions, absl::string_view name, absl::string_view fingerprint,
    absl::string_view default_memory_kind)
    : platform_id_(platform_id),
      compile_options_(compile_options),
      executables_(std::move(executable)),
      num_replicas_(num_replicas),
      num_partitions_(num_partitions),
      name_(name),
      fingerprint_(fingerprint),
      default_memory_kind_(default_memory_kind) {
  const auto& mod = std::get<std::unique_ptr<CompiledModule>>(executables_);
  if (mod != nullptr) {
    hlo_module_ = mod->shared_optimized_module();
  }
}

StreamExecutorExecutable::StreamExecutorExecutable(
    PjRtPlatformId platform_id, const CompileOptions& compile_options,
    std::optional<HloModuleProto> unoptimized_hlo_module_proto,
    std::shared_ptr<LocalExecutable> local_executable,
    LocalClient* local_client, int num_replicas, int num_partitions,
    absl::string_view name, absl::string_view fingerprint,
    absl::string_view default_memory_kind)
    : platform_id_(platform_id),
      compile_options_(compile_options),
      unoptimized_hlo_module_proto_(std::move(unoptimized_hlo_module_proto)),
      executables_(std::move(local_executable)),
      local_client_(local_client),
      num_replicas_(num_replicas),
      num_partitions_(num_partitions),
      name_(name),
      fingerprint_(fingerprint),
      default_memory_kind_(default_memory_kind) {
  const auto& local_exec =
      std::get<std::shared_ptr<LocalExecutable>>(executables_);
  if (local_exec != nullptr) {
    hlo_module_ = local_exec->executable()->shared_module();
  }
}

absl::StatusOr<CompiledMemoryStats>
StreamExecutorExecutable::GetCompiledMemoryStats() const {
  absl::MutexLock lock(&mu_);
  CompiledMemoryStats memory_stats = CompiledMemoryStats();
  if (auto* aot_executable =
          std::get_if<std::unique_ptr<CompiledModule>>(&executables_)) {
    if (*aot_executable == nullptr) {
      return absl::InternalError("No compiled module.");
    }
    return (*aot_executable)->GetCompiledMemoryStats();
  }

  const auto& local_exec =
      std::get<std::shared_ptr<LocalExecutable>>(executables_);
  if (local_exec == nullptr) {
    return absl::InternalError("No local executable.");
  }
  const BufferAssignmentProto* proto =
      local_exec->executable()->buffer_assignment_proto();
  if (proto != nullptr) {
    memory_stats.serialized_buffer_assignment = proto->SerializeAsString();
    HloModuleProto hlo_module_proto =
        local_exec->executable()->module().ToProto();
    ABSL_ASSIGN_OR_RETURN(auto peak_memories,
                     ComputePeakMemorySizes(*proto, hlo_module_proto));
    memory_stats.peak_memory_in_bytes = peak_memories.padded;
    memory_stats.peak_unpadded_heap_bytes = peak_memories.unpadded;
    memory_stats.total_allocation_bytes =
        ComputeTotalAllocationBytes(*proto, /*memory_color=*/0);
    memory_stats.indefinite_allocations =
        ComputeIndefiniteAllocationsInBytes(*proto, /*memory_color=*/0);
  }
  memory_stats.PopulateBufferStatsFromAllocations(
      local_exec->executable()->GetAllocations());
  memory_stats.generated_code_size_in_bytes =
      SizeOfGeneratedCodeInBytesLocked();
  return memory_stats;
}

int64_t StreamExecutorExecutable::SizeOfGeneratedCodeInBytes() const {
  absl::MutexLock lock(&mu_);
  return SizeOfGeneratedCodeInBytesLocked();
}

int64_t StreamExecutorExecutable::SizeOfGeneratedCodeInBytesLocked() const {
  if (std::holds_alternative<std::unique_ptr<CompiledModule>>(executables_)) {
    return 0;
  }
  const auto& local_exec =
      std::get<std::shared_ptr<LocalExecutable>>(executables_);
  if (local_exec == nullptr) {
    return 0;
  }
  return local_exec->executable()->SizeOfGeneratedCodeInBytes();
}

namespace {

absl::StatusOr<absl::string_view> MemoryKindFromLayout(
    const Layout& layout, absl::string_view default_memory_kind) {
  switch (layout.memory_space()) {
    case Layout::kHostMemorySpace:
      return PinnedHostMemorySpace::kKind;
    case Layout::kGenericFastMemorySpace:
    case Layout::kDefaultMemorySpace:
      return default_memory_kind;
    default:
      return InvalidArgument("Unexpected memory space %d in output layout",
                             layout.memory_space());
  }
}

absl::StatusOr<absl::string_view> MemoryKindFromSimpleShape(
    const Shape& shape, absl::string_view default_memory_kind) {
  if (!shape.has_layout()) {
    return default_memory_kind;
  }
  return MemoryKindFromLayout(shape.layout(), default_memory_kind);
}

absl::StatusOr<std::vector<absl::string_view>> MemoryKindsFromShape(
    const Shape& shape, absl::string_view default_memory_kind) {
  if (!shape.IsTuple()) {
    ABSL_ASSIGN_OR_RETURN(absl::string_view memory_kind,
                     MemoryKindFromSimpleShape(shape, default_memory_kind));
    return {{memory_kind}};
  }
  std::vector<absl::string_view> result;
  result.reserve(shape.tuple_shapes().size());
  for (const auto& element_shape : shape.tuple_shapes()) {
    ABSL_ASSIGN_OR_RETURN(
        absl::string_view element_memory_kind,
        MemoryKindFromSimpleShape(element_shape, default_memory_kind));
    result.push_back(element_memory_kind);
  }
  return result;
}

}  // namespace

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
StreamExecutorExecutable::GetParameterMemoryKinds() const {
  ABSL_ASSIGN_OR_RETURN(auto modules, GetHloModules());
  // If no modules are available, we cannot determine memory kinds. Returning
  // Unimplemented here triggers a safe fallback in IFRT (executable.cc) to
  // avoid a crash when memory kinds are not available (e.g., when annotations
  // are stripped).
  if (modules.empty()) {
    return absl::UnimplementedError(
        "GetParameterMemoryKinds is not supported when no modules are "
        "available.");
  }

  std::vector<std::vector<absl::string_view>> out;
  out.reserve(modules.size());
  for (const auto& module : modules) {
    const ComputationLayout& comp_layout = module->entry_computation_layout();
    ABSL_ASSIGN_OR_RETURN(std::vector<Layout> layouts,
                     xla::FlattenedParameterLayouts(comp_layout));
    std::vector<absl::string_view>& memory_kinds = out.emplace_back();
    memory_kinds.reserve(layouts.size());
    for (const xla::Layout& layout : layouts) {
      ABSL_ASSIGN_OR_RETURN(absl::string_view memory_kind,
                       MemoryKindFromLayout(layout, default_memory_kind_));
      memory_kinds.push_back(memory_kind);
    }
  }
  return out;
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
StreamExecutorExecutable::GetOutputMemoryKinds() const {
  ABSL_ASSIGN_OR_RETURN(auto shapes, GetOutputShapes());
  std::vector<std::vector<absl::string_view>> out;
  out.reserve(shapes.size());
  for (const auto& shape : shapes) {
    ABSL_ASSIGN_OR_RETURN(std::vector<absl::string_view> memory_kind,
                     MemoryKindsFromShape(shape, default_memory_kind_));
    out.push_back(memory_kind);
  }
  return out;
}

absl::StatusOr<std::shared_ptr<LocalExecutable>>
StreamExecutorExecutable::GetOrLoadExecutable(LocalClient* client) {
  absl::MutexLock lock(&mu_);
  if (std::holds_alternative<std::shared_ptr<LocalExecutable>>(executables_)) {
    const auto& tmp = std::get<std::shared_ptr<LocalExecutable>>(executables_);
    if (tmp == nullptr) {
      return absl::InternalError("No local executable");
    }
    return tmp;
  } else if (std::holds_alternative<std::unique_ptr<CompiledModule>>(
                 executables_)) {
    auto aot_executable =
        std::get<std::unique_ptr<CompiledModule>>(std::move(executables_));
    if (aot_executable == nullptr) {
      return absl::InternalError("No local executable");
    }
    ABSL_ASSIGN_OR_RETURN(auto local_executable,
                     client->Load(std::move(aot_executable),
                                  compile_options_.executable_build_options));
    std::shared_ptr<LocalExecutable> shared_exec = std::move(local_executable);
    executables_ = shared_exec;
    local_client_ = client;
    return shared_exec;
  }
  return absl::UnimplementedError("Unsupported executable type.");
}

absl::StatusOr<stream_executor::ExecutableAbiVersion>
StreamExecutorExecutable::ExtractExecutableAbiVersion() const {
  absl::MutexLock lock(&mu_);
  if (executables_.index() == 0) {
    const std::unique_ptr<CompiledModule>& compiled_module =
        std::get<0>(executables_);
    if (compiled_module == nullptr) {
      return absl::InternalError("No compiled module");
    }
    return compiled_module->GetExecutableAbiVersion();
  }
  const std::shared_ptr<LocalExecutable>& local_executable =
      std::get<1>(executables_);
  if (local_executable == nullptr) {
    return absl::InternalError("No local executable");
  }

  return local_executable->executable()->GetExecutableAbiVersion();
}

absl::StatusOr<std::unique_ptr<PjRtExecutableAbiVersion>>
StreamExecutorExecutable::GetAbiVersion() const {
  ABSL_ASSIGN_OR_RETURN(stream_executor::ExecutableAbiVersion executable_abi_version,
                   ExtractExecutableAbiVersion());
  return std::make_unique<StreamExecutorPjRtExecutableAbiVersion>(
      platform_id_, std::move(executable_abi_version));
}

absl::StatusOr<ExecutableAndOptionsProto> SerializedGpuExecutableFromReader(
    std::unique_ptr<riegeli::Reader> reader) {
  ExecutableAndOptionsProto proto;
  // The serialized string may be of the new SplitProto format (which allows
  // executables larger than 2GB) or the legacy format which is just a regular
  // proto.
  ABSL_ASSIGN_OR_RETURN(bool is_split_proto, IsSplitProto(*reader));
  if (is_split_proto) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        ReadSplitProto(std::move(reader), proto),
        "Failed to read serialized StreamExecutorExecutable");
    return proto;
  }

  ABSL_RETURN_IF_ERROR(riegeli::ParseMessage(std::move(reader), proto));
  return proto;
}

absl::StatusOr<ExecutableAndOptionsProto> SerializedGpuExecutableFromString(
    absl::string_view serialized) {
  return SerializedGpuExecutableFromReader(
      std::make_unique<riegeli::StringReader<>>(serialized));
}

absl::StatusOr<ExecutableAndOptionsProto> SerializedGpuExecutableFromString(
    const absl::Cord& serialized) {
  return SerializedGpuExecutableFromReader(
      std::make_unique<riegeli::CordReader<>>(&serialized));
}

}  // namespace xla
