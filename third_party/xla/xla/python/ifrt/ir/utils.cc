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

#include "xla/python/ifrt/ir/utils.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/memory.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

// Returns a DeviceList for the given device ids.
absl::StatusOr<DeviceListRef> LookUpDevices(Client* client,
                                            absl::Span<const DeviceId> ids) {
  std::vector<Device*> devices;
  devices.reserve(ids.size());
  for (DeviceId id : ids) {
    TF_ASSIGN_OR_RETURN(devices.emplace_back(), client->LookupDevice(id));
  }
  return client->MakeDeviceList(devices);
}

absl::StatusOr<std::unique_ptr<HloProgram>> XlaComputationToHloProgram(
    const xla::XlaComputation& xla_computation,
    absl::Span<const int64_t> donated_input_indices,
    absl::Span<const MemoryKind> arg_memory_kinds,
    absl::Span<const MemoryKind> result_memory_kinds) {
  const xla::HloModuleProto& hlo_module_proto = xla_computation.proto();
  TF_ASSIGN_OR_RETURN(
      auto host_program_shape,
      xla::ProgramShape::FromProto(hlo_module_proto.host_program_shape()));
  const xla::HloModuleConfig hlo_module_config(host_program_shape,
                                               /*ignore_layouts=*/false);
  TF_ASSIGN_OR_RETURN(
      const auto hlo_module,
      xla::HloModule::CreateFromProto(hlo_module_proto, hlo_module_config));

  auto mlir_context = std::make_unique<mlir::MLIRContext>();
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      xla::ConvertHloToStablehlo(*mlir_context, hlo_module.get()));
  auto program = std::make_unique<HloProgram>(std::move(mlir_context),
                                              std::move(mlir_module));

  mlir::func::FuncOp main =
      program->mlir_module().lookupSymbol<mlir::func::FuncOp>("main");
  if (main == nullptr) {
    return absl::InvalidArgumentError("module has no 'main' function");
  }
  // TODO(b/390732473): We should always donate inputs. If an input cannot be
  // donated at runtime, the execution can use `non_donatable_input_indices` to
  // create a copy of device buffers on the fly. However, several runtimes
  // ignore `non_donatable_input_indices`, so we need to be conservative at
  // input donation.
  for (int64_t idx : donated_input_indices) {
    main.setArgAttr(
        idx, kBufferDonationAttrName,
        mlir::BoolAttr::get(program->mlir_module()->getContext(), true));
  }
  for (int64_t idx = 0; idx < arg_memory_kinds.size(); ++idx) {
    if (arg_memory_kinds[idx].memory_kind().has_value()) {
      main.setArgAttr(
          idx, kHloMemoryKindAttrName,
          mlir::StringAttr::get(program->mlir_module()->getContext(),
                                *arg_memory_kinds[idx].memory_kind()));
    }
  }
  for (int64_t idx = 0; idx < result_memory_kinds.size(); ++idx) {
    if (result_memory_kinds[idx].memory_kind().has_value()) {
      main.setResultAttr(
          idx, kHloMemoryKindAttrName,
          mlir::StringAttr::get(program->mlir_module()->getContext(),
                                *result_memory_kinds[idx].memory_kind()));
    }
  }
  return program;
}

}  // namespace ifrt
}  // namespace xla
