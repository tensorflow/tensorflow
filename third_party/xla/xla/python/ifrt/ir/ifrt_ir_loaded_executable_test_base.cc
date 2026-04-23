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

#include "xla/python/ifrt/ir/ifrt_ir_loaded_executable_test_base.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Version.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/support/module_parsing.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/version.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace test_util {

IfrtIrLoadedExecutableTestBase::IfrtIrLoadedExecutableTestBase() {
  mlir::registerMLIRContextCLOptions();
  xla::ifrt::support::RegisterMlirDialects(mlir_context_);
}

void IfrtIrLoadedExecutableTestBase::SetUp() {
  TF_ASSERT_OK_AND_ASSIGN(client_, GetClient());
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
IfrtIrLoadedExecutableTestBase::LoadFromSource(absl::string_view source) {
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(&mlir_context_);
  auto op_ref = mlir::parseSourceString<mlir::ModuleOp>(source, &mlir_context_);
  if (!op_ref) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Failed to parse IFRT IR module string: %s",
                        diagnostic_handler.ConsumeStatus().message()));
  }
  return op_ref;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
IfrtIrLoadedExecutableTestBase::LoadFromFile(absl::string_view file_path) {
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(&mlir_context_);
  auto op_ref =
      mlir::parseSourceFile<mlir::ModuleOp>(file_path, &mlir_context_);
  if (!op_ref) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Failed to parse IFRT IR module file: %s",
                        diagnostic_handler.ConsumeStatus().message()));
  }
  return op_ref;
}

absl::StatusOr<std::unique_ptr<IfrtIRProgram>>
IfrtIrLoadedExecutableTestBase::SerDeRoundTrip(
    std::unique_ptr<IfrtIRProgram> program,
    Version::CompatibilityRequirement compatibility_requirement) {
  // Ensure the atom programs are outlined to modules. If the atom programs are
  // already outlined, this pipeline will do nothing.
  mlir::PassManager pm(program->mlir_module.getContext());
  createIfrtToOutlinedAtomProgramsPipeline(pm);
  mlir::BaseScopedDiagnosticHandler diag_handler(
      program->mlir_module.getContext());
  if (mlir::failed(pm.run(program->mlir_module))) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to outline IFRT IR program: ",
                     diag_handler.ConsumeStatus().message()));
  }

  // Serialize IFRT IR program with the given compatibility requirement, and the
  // atom programs at the current VHLO version.
  TF_ASSIGN_OR_RETURN(
      auto serialized,
      Serialize(
          *program,
          std::make_unique<SerializeIfrtIRProgramOptions>(
              Version::fromCompatibilityRequirement(compatibility_requirement)
                  .toString(),
              mlir::vhlo::Version::getCurrentVersion().toString())));

  // Deserialize the versioned IFRT IR program.
  TF_ASSIGN_OR_RETURN(
      program, Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr));
  return program;
}

absl::StatusOr<ArrayRef> IfrtIrLoadedExecutableTestBase::CreateArray(
    absl::Span<void* const> per_shard_data, Shape shape, Shape shard_shape,
    DType dtype, DeviceListRef device_list,
    std::optional<MemoryKind> memory_kind) {
  TF_RET_CHECK(per_shard_data.size() == device_list->devices().size())
      << "Inconsistent sizes. per_shard_data " << per_shard_data.size()
      << " vs device_list " << device_list->devices().size();
  ShardingRef sharding = ConcreteEvenSharding::Create(
      device_list, memory_kind.value_or(MemoryKind()), shape, shard_shape,
      /*is_fully_replicated=*/shape == shard_shape);
  TF_ASSIGN_OR_RETURN(
      auto per_shard,
      sharding->Disassemble(shape, SingleDeviceShardSemantics::kAllShards));
  // All shards have the same shape. Just pick 0.
  Shape per_shard_shape = per_shard[0].first;
  std::vector<ArrayRef> per_shard_arrays;
  per_shard_arrays.reserve(per_shard_data.size());
  for (int i = 0; i < per_shard_data.size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        ArrayRef per_shard_array,
        client_->MakeArrayFromHostBuffer(
            per_shard_data[i], dtype, per_shard_shape,
            /*byte_strides=*/std::nullopt,
            SingleDeviceSharding::Create(device_list->devices()[i],
                                         memory_kind.value_or(MemoryKind())),
            /*layout=*/nullptr,
            Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr));
    per_shard_arrays.push_back(per_shard_array);
  }
  return client_->AssembleArrayFromSingleDeviceArrays(
      dtype, shape, sharding, absl::MakeSpan(per_shard_arrays),
      ArrayCopySemantics::kAlwaysCopy,
      SingleDeviceShardSemantics::kAddressableShards);
}

int32_t IfrtIrLoadedExecutableTestBase::GetNumDevices() {
  return client_->devices().size();
}

absl::StatusOr<DeviceListRef> IfrtIrLoadedExecutableTestBase::PickDevices(
    int count, std::optional<absl::string_view> platform_name) {
  if (!platform_name.has_value()) {
    absl::Span<Device* const> devices = client_->devices();
    TF_RET_CHECK(count <= devices.size())
        << "Requested " << count << " devices. Only have " << devices.size();
    return client_->MakeDeviceList(devices.first(count));
  }

  if (*platform_name == "cpu") {
    absl::InlinedVector<Device*, 1> cpu_devices;
    cpu_devices.reserve(count);
    for (Device* const device : client_->GetAllDevices()) {
      auto platform_name =
          device->Attributes().Get<std::string>("platform_name");
      if (!platform_name.ok()) {
        continue;
      }
      if (*platform_name == "cpu") {
        cpu_devices.push_back(device);
        if (cpu_devices.size() == count) {
          break;
        }
      }
    }
    TF_RET_CHECK(count <= cpu_devices.size())
        << "Requested " << count << " devices. Only have "
        << cpu_devices.size();
    return client_->MakeDeviceList(cpu_devices);
  }

  return absl::InvalidArgumentError(
      absl::Substitute("Unsupported device type $0.", *platform_name));
}

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla
