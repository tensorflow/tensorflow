/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/ifrt_ir_loaded_executable.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/OperationSupport.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ir_executable_version.h"
#include "xla/python/ifrt/ir/program_memory_tracer.h"
#include "xla/python/ifrt/ir/serialization_utils.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/python/ifrt/ir/version.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/support/sharding_conversions.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace ifrt {

using ::tsl::profiler::TraceMe;

char IfrtIrLoadedExecutable::ID = 0;

namespace {

using DeviceIdToLogicalDeviceIdMap =
    absl::flat_hash_map<xla::ifrt::DeviceId, IfrtIrLogicalDeviceId>;

// Returns a DeviceList for the given device ids.
absl::StatusOr<xla::ifrt::DeviceListRef> LookUpDevices(
    xla::ifrt::Client* client, absl::Span<const xla::ifrt::DeviceId> ids) {
  absl::InlinedVector<xla::ifrt::Device*, 1> devices;
  devices.reserve(ids.size());
  for (xla::ifrt::DeviceId id : ids) {
    TF_ASSIGN_OR_RETURN(xla::ifrt::Device * device, client->LookupDevice(id));
    devices.push_back(device);
  }
  return client->MakeDeviceList(devices);
}

// Create a map from runtime device id to logical device id.
absl::StatusOr<DeviceIdToLogicalDeviceIdMap> CreateDeviceIdToLogicalDeviceIdMap(
    std::shared_ptr<CompiledIfrtIrProgram> program) {
  DeviceIdToLogicalDeviceIdMap device_id_to_logical_device_id;
  for (int i = 0; i < program->device_assignments.size(); ++i) {
    const xla::ifrt::DeviceId device_id = program->device_assignments[i];
    auto [_, inserted] = device_id_to_logical_device_id.insert(
        {device_id, IfrtIrLogicalDeviceId(i)});
    if (!inserted) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate device id ", device_id.value(),
                       " found in device assignments"));
    }
  }
  return device_id_to_logical_device_id;
}

absl::StatusOr<IfrtIrExecutableVersion::AtomExecutableVersion>
CreateAtomExecutableVersion(
    std::shared_ptr<const xla::ifrt::LoadedExecutable> executable,
    absl::flat_hash_map<xla::ifrt::DeviceId, IfrtIrLogicalDeviceId>&
        device_id_to_logical_device_id) {
  std::optional<xla::ifrt::DeviceListRef> device_list = executable->devices();
  if (!device_list.has_value()) {
    return absl::UnimplementedError("Portable executables are not supported.");
  }
  absl::Span<xla::ifrt::Device* const> devices = (*device_list)->devices();

  IfrtIrExecutableVersion::AtomExecutableVersion atom_executable_version;
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const xla::ifrt::ExecutableVersion> version,
      executable->executable_version());
  atom_executable_version.runtime_abi_version = std::move(version);
  atom_executable_version.logical_device_ids.reserve(devices.size());
  for (xla::ifrt::Device* device : devices) {
    atom_executable_version.logical_device_ids.push_back(
        device_id_to_logical_device_id[device->Id()]);
  }
  return atom_executable_version;
}

}  // namespace

absl::StatusOr<std::optional<std::string>> IfrtIrLoadedExecutable::Fingerprint()
    const {
  // Do not return a fingerprint because IFRT IR executables do not support
  // JAX-style (de)serialization which requires the IFRT IR program and the
  // SPMD executables to be deserialized on the client.
  return std::optional<std::string>();
}

absl::StatusOr<std::shared_ptr<const xla::ifrt::ExecutableVersion>>
IfrtIrLoadedExecutable::executable_version() const {
  absl::call_once(version_once_, [&]() {
    version_ = [&]()
        -> absl::StatusOr<std::shared_ptr<const xla::ifrt::ExecutableVersion>> {
      // Create list of runtime ABI versions for the IFRT IR atom executables.
      std::vector<IfrtIrExecutableVersion::AtomExecutableVersion>
          runtime_abi_versions;
      runtime_abi_versions.reserve(program_->atom_program_executables->size());
      TF_ASSIGN_OR_RETURN(
          DeviceIdToLogicalDeviceIdMap device_id_to_logical_device_id,
          CreateDeviceIdToLogicalDeviceIdMap(program_));

      for (const auto& [name, executable] :
           *program_->atom_program_executables) {
        TF_ASSIGN_OR_RETURN(IfrtIrExecutableVersion::AtomExecutableVersion
                                atom_executable_version,
                            CreateAtomExecutableVersion(
                                executable, device_id_to_logical_device_id));
        runtime_abi_versions.push_back(std::move(atom_executable_version));
      }

      return std::make_unique<IfrtIrExecutableVersion>(
          Version::getCurrentVersion(), program_->device_assignments,
          std::move(runtime_abi_versions));
    }();
  });
  return version_;
}

absl::StatusOr<std::string> IfrtIrLoadedExecutable::Serialize() const {
  return SerializeIfrtIrExecutable(program_);
}

absl::StatusOr<std::string>
IfrtIrLoadedExecutable::GetHumanReadableProgramText() const {
  std::string result =
      OperationToString(program_->program->mlir_module,
                        mlir::OpPrintingFlags().enableDebugInfo());
  for (const auto& [name, executable] : *program_->atom_program_executables) {
    TF_ASSIGN_OR_RETURN(auto t, executable->GetHumanReadableProgramText());
    absl::StrAppend(&result, "\n\n", name, " -> ", t);
  }
  return result;
}

int IfrtIrLoadedExecutable::num_devices() const {
  DCHECK(this);
  return devices_->size();
}

int64_t IfrtIrLoadedExecutable::SizeOfGeneratedCodeInBytes() const {
  // TODO(b/261226026): Implement this API or remove it from IFRT.
  return -1;
}

absl::StatusOr<xla::CompiledMemoryStats>
IfrtIrLoadedExecutable::GetCompiledMemoryStats() const {
  return xla::Unimplemented(
      "IfrtIrLoadedExecutable does not support GetCompiledMemoryStats()");
}

absl::StatusOr<absl::flat_hash_map<std::string, xla::CompiledMemoryStats>>
IfrtIrLoadedExecutable::GetMpmdCompiledMemoryStats() const {
  absl::flat_hash_map<std::string, xla::CompiledMemoryStats>
      mpmd_compiled_memory_stats;
  for (const auto& [name, executable] : *program_->atom_program_executables) {
    TF_ASSIGN_OR_RETURN(auto compiled_memory_stats,
                        executable->GetCompiledMemoryStats());
    mpmd_compiled_memory_stats.insert({name, std::move(compiled_memory_stats)});
  }
  return mpmd_compiled_memory_stats;
}

absl::StatusOr<IfrtIrProgramMemoryStats>
IfrtIrLoadedExecutable::GetIfrtIrProgramMemoryStats() const {
  return memory_tracer_->GetMemoryStats();
}

absl::StatusOr<std::string> IfrtIrLoadedExecutable::GetIfrtIrProgramXprofUrl()
    const {
  return memory_tracer_->GetXprofUrl();
}

std::optional<std::vector<xla::OpSharding>>
IfrtIrLoadedExecutable::GetParameterShardings() const {
  DCHECK(this);
  TraceMe traceme(
      []() { return "IfrtIrLoadedExecutable::GetParameterShardings"; });
  std::vector<xla::OpSharding> parameter_shardings;
  for (const auto& [idx, spec] : llvm::enumerate(program_->in_specs)) {
    auto sharding = xla::ifrt::support::ToOpSharding(*spec.sharding);
    if (sharding.ok()) {
      parameter_shardings.push_back(*sharding);
    } else {
      LOG(ERROR) << "Failed to convert parameter sharding #" << idx
                 << sharding.status().message();
      return std::nullopt;
    }
  }
  return parameter_shardings;
}

std::optional<std::vector<xla::OpSharding>>
IfrtIrLoadedExecutable::GetOutputShardings() const {
  DCHECK(this);
  TraceMe traceme(
      []() { return "IfrtIrLoadedExecutable::GetOutputShardings"; });
  std::vector<xla::OpSharding> output_shardings;
  for (const auto& [idx, spec] : llvm::enumerate(program_->out_specs)) {
    auto sharding = xla::ifrt::support::ToOpSharding(*spec.sharding);
    if (sharding.ok()) {
      output_shardings.push_back(*sharding);
    } else {
      LOG(ERROR) << "Failed to convert output sharding #" << idx
                 << sharding.status().message();
      return std::nullopt;
    }
  }
  return output_shardings;
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
IfrtIrLoadedExecutable::GetParameterLayouts() const {
  DCHECK(this);
  TraceMe traceme(
      []() { return "IfrtIrLoadedExecutable::GetParameterLayouts"; });
  TF_RETURN_IF_ERROR(program_->layout_status);
  std::vector<std::shared_ptr<const xla::PjRtLayout>> parameter_layouts;
  parameter_layouts.reserve(program_->in_specs.size());
  for (const auto& spec : program_->in_specs) {
    parameter_layouts.push_back(spec.layout);
  }
  return parameter_layouts;
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
IfrtIrLoadedExecutable::GetOutputLayouts() const {
  DCHECK(this);
  TraceMe traceme([]() { return "IfrtIrLoadedExecutable::GetOutputLayouts"; });
  TF_RETURN_IF_ERROR(program_->layout_status);
  std::vector<std::shared_ptr<const xla::PjRtLayout>> output_layouts;
  output_layouts.reserve(program_->out_specs.size());
  for (const auto& spec : program_->out_specs) {
    output_layouts.push_back(spec.layout);
  }
  return output_layouts;
}

absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>>
IfrtIrLoadedExecutable::GetHloModules() const {
  // LoadedExecutable::GetHloModules should return an HLO module per partition.
  // However, in the case of IFRT IR executables it returns a vector of HLO
  // modules per partition per SPMD program, which is used to print useful debug
  // information.
  //
  // Keys to `atom_program_executables` are lost, but usually the size of value
  // vector is one because modules are rarely partitioned and the keys are the
  // same as HloModule::name()
  std::vector<std::shared_ptr<xla::HloModule>> all_modules;
  for (const auto& [name, executable] : *program_->atom_program_executables) {
    TF_ASSIGN_OR_RETURN(
        std::vector<std::shared_ptr<xla::HloModule>> this_modules,
        executable->GetHloModules());
    all_modules.insert(all_modules.end(), this_modules.begin(),
                       this_modules.end());
  }
  return all_modules;
}

absl::StatusOr<absl::flat_hash_map<
    std::string, std::vector<std::shared_ptr<xla::HloModule>>>>
IfrtIrLoadedExecutable::GetMpmdHloModules() const {
  absl::flat_hash_map<std::string, std::vector<std::shared_ptr<xla::HloModule>>>
      mpmd_hlo_modules;
  for (const auto& [name, executable] : *program_->atom_program_executables) {
    TF_ASSIGN_OR_RETURN(auto hlo_modules, executable->GetHloModules());
    mpmd_hlo_modules.insert({name, std::move(hlo_modules)});
  }
  return mpmd_hlo_modules;
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
IfrtIrLoadedExecutable::GetOutputMemoryKinds() const {
  DCHECK(this);
  TraceMe traceme(
      []() { return "IfrtIrLoadedExecutable::GetOutputMemoryKinds"; });
  std::vector<absl::string_view> output_memory_kinds;
  for (const auto& [idx, spec] : llvm::enumerate(program_->out_specs)) {
    if (!spec.sharding->memory_kind().memory_kind().has_value()) {
      return absl::FailedPreconditionError(
          absl::StrFormat("IfrtIrLoadedExecutable %s does not have memory kind "
                          "set for its output # %d.",
                          name(), idx));
    }
    output_memory_kinds.push_back(
        spec.sharding->memory_kind().memory_kind().value());
  }
  // Pretend that the MPMD executable is a SPMD executable and return a
  // single array. We do not return per-SPMD program memory kinds because the
  // order in the vector would not be well-defined.
  std::vector<std::vector<absl::string_view>> output_memory_kinds_vector = {
      output_memory_kinds};
  return output_memory_kinds_vector;
}

absl::StatusOr<xla::ifrt::AttributeMap>
IfrtIrLoadedExecutable::GetCostAnalysis() const {
  return xla::Unimplemented(
      "IfrtIrLoadedExecutable does not support GetCostAnalysis()");
}

absl::StatusOr<absl::flat_hash_map<std::string, xla::ifrt::AttributeMap>>
IfrtIrLoadedExecutable::GetMpmdCostAnalysis() const {
  absl::flat_hash_map<std::string, xla::ifrt::AttributeMap> mpmd_cost_analysis;
  for (const auto& [name, executable] : *program_->atom_program_executables) {
    TF_ASSIGN_OR_RETURN(auto atom_program_analysis,
                        executable->GetCostAnalysis());
    mpmd_cost_analysis.insert({name, std::move(atom_program_analysis)});
  }
  return mpmd_cost_analysis;
}

absl::Span<xla::ifrt::Device* const>
IfrtIrLoadedExecutable::addressable_devices() const {
  DCHECK(this);
  return devices_->devices();
}

std::optional<DeviceListRef> IfrtIrLoadedExecutable::devices() const {
  return devices_;
}

absl::StatusOr<
    absl::flat_hash_map<std::string, absl::Span<xla::ifrt::Device* const>>>
IfrtIrLoadedExecutable::GetMpmdAddressableDevices() const {
  absl::flat_hash_map<std::string, absl::Span<xla::ifrt::Device* const>>
      mpmd_addressable_devices;
  mpmd_addressable_devices.reserve(program_->atom_program_executables->size());
  for (const auto& [name, executable] : *program_->atom_program_executables) {
    mpmd_addressable_devices.insert({name, executable->addressable_devices()});
  }
  return mpmd_addressable_devices;
}

absl::StatusOr<xla::ifrt::LoadedExecutableRef> IfrtIrLoadedExecutable::Create(
    xla::ifrt::Client* client, std::shared_ptr<CompiledIfrtIrProgram> program) {
  tsl::profiler::TraceMe traceme("IfrtIrLoadedExecutable::Create");

  TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                      LookUpDevices(client, program->device_assignments));
  TF_ASSIGN_OR_RETURN(auto memory_tracer, ProgramMemoryTracer::Create(
                                              program, client, device_list));
  return std::unique_ptr<IfrtIrLoadedExecutable>(new IfrtIrLoadedExecutable(
      client, std::move(program), std::move(device_list),
      std::move(memory_tracer)));
}

IfrtIrLoadedExecutable::IfrtIrLoadedExecutable(
    xla::ifrt::Client* client, std::shared_ptr<CompiledIfrtIrProgram> program,
    xla::ifrt::DeviceListRef devices,
    std::unique_ptr<ProgramMemoryTracer> memory_tracer)
    : client_(client),
      program_(std::move(program)),
      devices_(std::move(devices)),
      memory_tracer_(std::move(memory_tracer)),
      user_context_(xla::ifrt::UserContextScope::current()) {}

absl::StatusOr<xla::ifrt::LoadedExecutable::ExecuteResult>
IfrtIrLoadedExecutable::Execute(
    absl::Span<xla::ifrt::ArrayRef> args, const ExecuteOptions& options,
    std::optional<xla::ifrt::DeviceListRef> devices) {
  return program_->execute_fn(args, options, std::move(devices));
}

absl::StatusOr<absl::Span<const int>>
IfrtIrLoadedExecutable::GetDonatableInputIndices() const {
  return absl::MakeConstSpan(program_->donatable_input_indices);
}
}  // namespace ifrt
}  // namespace xla
