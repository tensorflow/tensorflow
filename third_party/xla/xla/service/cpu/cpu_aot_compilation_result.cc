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

#include "xla/service/cpu/cpu_aot_compilation_result.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/buffer_allocation_info.h"
#include "xla/backends/cpu/buffer_allocation_info_util.h"
#include "xla/backends/cpu/constant_allocation.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

CpuAotCompilationOptions::CpuAotCompilationOptions(
    std::string triple, std::string cpu_name, std::string features,
    std::string entry_point_name, RelocationModel relocation_model,
    bool compile_copy_as_llvm_kernel)
    : triple_(std::move(triple)),
      cpu_name_(std::move(cpu_name)),
      features_(std::move(features)),
      entry_point_name_(std::move(entry_point_name)),
      relocation_model_(relocation_model),
      compile_copy_as_llvm_kernel_(compile_copy_as_llvm_kernel) {}

CpuAotCompilationOptions::~CpuAotCompilationOptions() = default;

se::Platform::Id CpuAotCompilationOptions::PlatformId() const {
  return se::host::kHostPlatformId;
}

absl::StatusOr<std::unique_ptr<CpuAotCompilationResult>>
CpuAotCompilationResult::Create(
    const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
    absl::string_view function_name, std::vector<ObjFileProto> obj_files,
    std::vector<SymbolProto> symbols, const ThunkSequence& thunks,
    std::unique_ptr<FunctionLibrary> function_library,
    TargetMachineOptionsProto target_machine_options) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes(
      hlo_module, &buffer_assignment->Allocations());
  TF_ASSIGN_OR_RETURN(ThunkSequenceProto thunk_proto,
                      thunk_sequence_serdes.ToProto(thunks));

  std::vector<cpu::BufferAllocationInfo> buffer_allocation_infos;
  std::optional<size_t> temp_allocation_index;

  if (buffer_assignment) {
    buffer_allocation_infos =
        CreateBufferAllocationInfos(*hlo_module, *buffer_assignment);

    // Find temp allocation index if it exists
    for (const BufferAllocation& allocation :
         buffer_assignment->Allocations()) {
      if (allocation.IsPreallocatedTempBuffer()) {
        if (temp_allocation_index.has_value()) {
          return Internal("Multiple temp buffer allocations found");
        }
        temp_allocation_index = allocation.index();
      }
    }
  }

  return absl::WrapUnique(new CpuAotCompilationResult(
      hlo_module, buffer_assignment, function_name, std::move(obj_files),
      std::move(symbols), thunk_proto, std::move(temp_allocation_index),
      std::move(buffer_allocation_infos), std::move(function_library),
      std::move(target_machine_options)));
}

CpuAotCompilationResult::CpuAotCompilationResult(
    const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
    absl::string_view function_name, std::vector<ObjFileProto> obj_files,
    std::vector<SymbolProto> symbols, const ThunkSequenceProto& thunks,
    std::optional<size_t> temp_allocation_index,
    std::vector<BufferAllocationInfo> buffer_allocation_infos,
    std::unique_ptr<FunctionLibrary> function_library,
    TargetMachineOptionsProto target_machine_options)
    : temp_allocation_index_(temp_allocation_index),
      buffer_allocation_infos_(std::move(buffer_allocation_infos)),
      function_library_(std::move(function_library)) {
  *proto_.mutable_hlo_module()->mutable_hlo_module() = hlo_module->ToProto();
  *proto_.mutable_hlo_module()->mutable_config() =
      hlo_module->config().ToProto();
  *proto_.mutable_buffer_assignment() = buffer_assignment->ToProto();
  proto_.set_entry_function_name(function_name);
  *proto_.mutable_target_machine_options() = std::move(target_machine_options);
  for (ObjFileProto& obj_file : obj_files) {
    *proto_.add_object_files() = std::move(obj_file);
  }

  for (const auto& symbol : symbols) {
    auto* symbol_proto = proto_.add_compiled_symbols();
    *symbol_proto = symbol;
  }
  proto_.set_obj_files_kind(CompilationResultProto::KERNELS);
  module_ = hlo_module->Clone();

  ThunkSequenceSerDesProtobuf thunk_sequence_serdes(
      hlo_module, &buffer_assignment->Allocations());
  *proto_.mutable_thunk_sequence() = thunks;
}

absl::StatusOr<std::unique_ptr<Executable>>
CpuAotCompilationResult::LoadExecutable() && {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      HloModule::CreateFromProtoWithConfig(proto_.hlo_module()));

  VLOG(2) << "Load XLA:CPU executable for module: " << module->name();

  // Copied from cpu_compiler.cc in order to avoid dependency on cpu_compiler.
  std::function<int64_t(const BufferValue&)> buffer_size_bytes_function_getter =
      [](const BufferValue& buffer) {
        return CpuExecutable::ShapeSizeBytes(buffer.shape());
      };

  // Recreate BufferAssignment from proto.
  AliasInfo alias_info;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> buffer_assignment,
                      BufferAssignment::FromProto(
                          proto_.buffer_assignment(), module.get(),
                          buffer_size_bytes_function_getter, &alias_info));

  std::unique_ptr<CpuExecutable> cpu_executable;

  if (proto_.obj_files_kind() != CompilationResultProto::KERNELS) {
    return Internal("AOT compilation result does not have thunks.");
  }

  ThunkSequenceSerDesProtobuf thunk_sequence_serdes(
      module.get(), &buffer_assignment->Allocations());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<ThunkSequence> thunks,
                      thunk_sequence_serdes.FromProto(proto_.thunk_sequence()));

  VLOG(3) << "Loaded " << thunks->size() << " thunks.";

  // Create constant allocations from the buffer assignment.
  TF_ASSIGN_OR_RETURN(std::vector<ConstantAllocation> constants,
                      CreateConstantAllocations(*buffer_assignment));

  TF_ASSIGN_OR_RETURN(
      TargetMachineOptions target_machine_options,
      TargetMachineOptions::FromProto(proto_.target_machine_options()));

  TF_ASSIGN_OR_RETURN(
      cpu_executable,
      CpuExecutable::Create(std::move(function_library_),
                            std::move(buffer_assignment), std::move(module),
                            std::move(*thunks), std::move(constants),
                            target_machine_options));

  // Dump computation proto state and buffer assignment for
  // GetCompiledMemoryStats results.
  auto hlo_proto = std::make_unique<HloProto>();
  *hlo_proto->mutable_hlo_module() = cpu_executable->module().ToProto();
  *hlo_proto->mutable_buffer_assignment() =
      cpu_executable->buffer_assignment().ToProto();
  cpu_executable->set_hlo_proto(std::move(hlo_proto));

  return cpu_executable;
}

absl::StatusOr<std::unique_ptr<Executable>>
CpuAotCompilationResult::LoadExecutable(const se::StreamExecutor* executor) && {
  return std::move((*this)).LoadExecutable();
}

absl::StatusOr<std::unique_ptr<Executable>>
CpuAotCompilationResult::LoadExecutable(
    se::Platform::Id platform_id,
    const se::DeviceDescription& device_description) && {
  return std::move((*this)).LoadExecutable();
}

}  // namespace xla::cpu
