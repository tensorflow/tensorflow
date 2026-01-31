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

#include "xla/pjrt/stream_executor_executable.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/stream_executor_executable.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

absl::StatusOr<std::string> StreamExecutorExecutable::SerializeExecutable()
    const {
  if (IsEarlyExitCompilation(compile_options_)) {
    ExecutableAndOptionsProto proto;
    TF_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                        compile_options_.ToProto());
    return proto.SerializeAsString();
  }
  std::string serialized;
  if (std::holds_alternative<std::vector<std::unique_ptr<CompiledModule>>>(
          executables_)) {
    const auto& aot_executables =
        std::get<std::vector<std::unique_ptr<CompiledModule>>>(executables_);
    if (aot_executables.empty()) {
      return absl::InternalError("No local executable");
    }
    if (aot_executables.size() != 1) {
      return absl::UnimplementedError(
          "PjRtStreamExecutorClient::SerializeExecutable unimplemented for "
          "MPMD executables");
    }
    TF_ASSIGN_OR_RETURN(serialized, aot_executables[0]->SerializeAsString());
  } else {
    const auto& local_executables =
        std::get<std::vector<std::unique_ptr<LocalExecutable>>>(executables_);
    Executable* built_executable = local_executables[0]->executable();
    CHECK(local_client_ != nullptr);
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<CompiledModule> aot_result,
        local_client_->backend().compiler()->Export(built_executable));

    TF_ASSIGN_OR_RETURN(serialized, aot_result->SerializeAsString());
  }

  if (serialized.empty()) {
    return absl::InternalError(
        "PjRtStreamExecutorClient::SerializeExecutable proto serialization "
        "failed");
  }
  ExecutableAndOptionsProto proto;
  *proto.mutable_serialized_executable() = std::move(serialized);
  TF_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                      compile_options_.ToProto());
  return proto.SerializeAsString();
}

StreamExecutorExecutable::StreamExecutorExecutable(
    const CompileOptions& compile_options,
    std::vector<std::unique_ptr<CompiledModule>> executables, int num_replicas,
    int num_partitions, absl::string_view name, absl::string_view fingerprint,
    absl::string_view default_memory_kind)
    : compile_options_(compile_options),
      executables_(std::move(executables)),
      num_replicas_(num_replicas),
      num_partitions_(num_partitions),
      name_(name),
      fingerprint_(fingerprint),
      default_memory_kind_(default_memory_kind) {
  std::vector<std::shared_ptr<HloModule>> hlo_modules;
  for (const auto& executable :
       std::get<std::vector<std::unique_ptr<CompiledModule>>>(executables_)) {
    hlo_modules.push_back(executable->shared_optimized_module());
  }
  hlo_modules_ = std::move(hlo_modules);
}

StreamExecutorExecutable::StreamExecutorExecutable(
    const CompileOptions& compile_options,
    std::optional<HloModuleProto> unoptimized_hlo_module_proto,
    std::vector<std::unique_ptr<LocalExecutable>> local_executables,
    LocalClient* local_client, int num_replicas, int num_partitions,
    absl::string_view name, absl::string_view fingerprint,
    absl::string_view default_memory_kind)
    : compile_options_(compile_options),
      unoptimized_hlo_module_proto_(std::move(unoptimized_hlo_module_proto)),
      executables_(std::move(local_executables)),
      local_client_(local_client),
      num_replicas_(num_replicas),
      num_partitions_(num_partitions),
      name_(name),
      fingerprint_(fingerprint),
      default_memory_kind_(default_memory_kind) {
  std::vector<std::shared_ptr<HloModule>> hlo_modules;
  for (const auto& local_executable :
       std::get<std::vector<std::unique_ptr<LocalExecutable>>>(executables_)) {
    hlo_modules.push_back(local_executable->executable()->shared_module());
  }
  hlo_modules_ = std::move(hlo_modules);
}

absl::StatusOr<CompiledMemoryStats>
StreamExecutorExecutable::GetCompiledMemoryStats() const {
  CompiledMemoryStats memory_stats = CompiledMemoryStats();
  if (auto* aot_executables =
          std::get_if<std::vector<std::unique_ptr<CompiledModule>>>(
              &executables_)) {
    if (aot_executables->size() != 1) {
      return Unimplemented(
          "Retrieving CompiledMemoryStats is not supported for multiple "
          "executables.");
    }
    const auto& aot_executable = (*aot_executables)[0];
    TF_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> buffers,
                        aot_executable->buffer_assignment());

    BufferAssignmentProto proto = buffers->ToProto();
    memory_stats.serialized_buffer_assignment = proto.SerializeAsString();
    std::vector<const BufferAllocation*> alloc_ptrs;
    alloc_ptrs.reserve(buffers->Allocations().size());
    for (const BufferAllocation& alloc : buffers->Allocations()) {
      alloc_ptrs.push_back(&alloc);
    }
    memory_stats.PopulateBufferStatsFromAllocations(alloc_ptrs);
    TF_ASSIGN_OR_RETURN(memory_stats.peak_memory_in_bytes,
                        ComputePeakMemory(proto));
    return memory_stats;
  }

  const auto& local_executables =
      std::get<std::vector<std::unique_ptr<LocalExecutable>>>(executables_);
  if (local_executables.size() != 1) {
    return absl::UnimplementedError(
        "Retrieving CompiledMemoryStats is not supported for multiple "
        "executables.");
  }
  const BufferAssignmentProto* proto =
      local_executables[0]->executable()->buffer_assignment_proto();
  if (proto != nullptr) {
    memory_stats.serialized_buffer_assignment = proto->SerializeAsString();
    TF_ASSIGN_OR_RETURN(memory_stats.peak_memory_in_bytes,
                        ComputePeakMemory(*proto));
  }
  memory_stats.PopulateBufferStatsFromAllocations(
      local_executables[0]->executable()->GetAllocations());
  memory_stats.generated_code_size_in_bytes = SizeOfGeneratedCodeInBytes();
  return memory_stats;
}

int64_t StreamExecutorExecutable::SizeOfGeneratedCodeInBytes() const {
  if (std::holds_alternative<std::vector<std::unique_ptr<CompiledModule>>>(
          executables_)) {
    return 0;
  }
  int64_t size = 0;
  for (auto& executable :
       std::get<std::vector<std::unique_ptr<LocalExecutable>>>(executables_)) {
    size += executable->executable()->SizeOfGeneratedCodeInBytes();
  }
  return size;
}

namespace {

absl::StatusOr<absl::string_view> MemoryKindFromSimpleShape(
    const Shape& shape, absl::string_view default_memory_kind) {
  if (!shape.has_layout()) {
    return default_memory_kind;
  }
  switch (shape.layout().memory_space()) {
    case Layout::kHostMemorySpace:
      return PinnedHostMemorySpace::kKind;
    case Layout::kGenericFastMemorySpace:
    case Layout::kDefaultMemorySpace:
      return default_memory_kind;
    default:
      return InvalidArgument("Unexpected memory space %d in output layout",
                             shape.layout().memory_space());
  }
}

absl::StatusOr<std::vector<absl::string_view>> MemoryKindsFromShape(
    const Shape& shape, absl::string_view default_memory_kind) {
  if (!shape.IsTuple()) {
    TF_ASSIGN_OR_RETURN(absl::string_view memory_kind,
                        MemoryKindFromSimpleShape(shape, default_memory_kind));
    return {{memory_kind}};
  }
  std::vector<absl::string_view> result;
  result.reserve(shape.tuple_shapes().size());
  for (const auto& element_shape : shape.tuple_shapes()) {
    TF_ASSIGN_OR_RETURN(
        absl::string_view element_memory_kind,
        MemoryKindFromSimpleShape(element_shape, default_memory_kind));
    result.push_back(element_memory_kind);
  }
  return result;
}

}  // namespace

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
StreamExecutorExecutable::GetOutputMemoryKinds() const {
  TF_ASSIGN_OR_RETURN(auto shapes, GetOutputShapes());
  std::vector<std::vector<absl::string_view>> out;
  out.reserve(shapes.size());
  for (const auto& shape : shapes) {
    TF_ASSIGN_OR_RETURN(std::vector<absl::string_view> memory_kind,
                        MemoryKindsFromShape(shape, default_memory_kind_));
    out.push_back(memory_kind);
  }
  return out;
}

absl::StatusOr<std::unique_ptr<LocalExecutable>>
StreamExecutorExecutable::ConsumeExecutable(
    LocalClient* client, const CompileOptions& compile_options) {
  if (std::holds_alternative<std::vector<std::unique_ptr<LocalExecutable>>>(
          executables_)) {
    auto tmp = std::get<std::vector<std::unique_ptr<LocalExecutable>>>(
        std::move(executables_));
    if (tmp.size() == 0) {
      return absl::InternalError("No local executable");
    }
    if (tmp.size() > 1) {
      return absl::InternalError(
          "ConsumeExecutable is not supported for more than one executable.");
    }
    return std::move(tmp[0]);
  } else if (std::holds_alternative<
                 std::vector<std::unique_ptr<CompiledModule>>>(executables_)) {
    auto aot_executables =
        std::get<std::vector<std::unique_ptr<CompiledModule>>>(
            std::move(executables_));
    if (aot_executables.size() == 0) {
      return absl::InternalError("No local executable");
    }
    if (aot_executables.size() > 1) {
      return absl::InternalError(
          "ConsumeExecutable is not supported for more than one executable.");
    }
    return client->Load(std::move(aot_executables[0]),
                        compile_options.executable_build_options);
  }
  return absl::UnimplementedError("Unsupported executable type.");
}

}  // namespace xla
