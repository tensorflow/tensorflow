/* Copyright 2024 The OpenXLA Authors.

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

#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/support/sharding_conversions.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTLOWERATOMPROGRAMMETADATATOXLAPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

mlir::LogicalResult SetArgHloSharding(mlir::func::FuncOp func_op,
                                      mlir::UnitAttr local_view_attr,
                                      int num_devices, mlir::OpBuilder& builder,
                                      int arg_idx) {
  const auto sharding_attr =
      func_op.getArgAttrOfType<IfrtShardingAttrInterface>(
          arg_idx, kIfrtShardingAttrName);
  if (sharding_attr == nullptr) {
    // The op has already been visited, and the IFRT attributes have been
    // removed. Verify that kHloShardingAttrName has been set if the module is
    // not sdy partitioned.
    if (func_op.getArgAttr(arg_idx, kHloShardingAttrName) == nullptr) {
      func_op.emitOpError()
          << "can't find `" << kIfrtShardingAttrName << "` attribute of input #"
          << arg_idx << " to set `" << kHloShardingAttrName << "` attribute";
      return mlir::failure();
    }
    return mlir::success();
  }

  if (IsUnspecifiedSharding(sharding_attr)) {
    // Sharding is not specified so we cannot lower to kHloShardingAttrName.
    return mlir::success();
  }

  // Verify that the input is sharded over the same number of devices as
  // the computation.
  if (int attr_num_devices = sharding_attr.NumDevices();
      attr_num_devices != num_devices) {
    func_op.emitOpError() << "can't lower sharding of input #" << arg_idx
                          << ". Sharding: " << sharding_attr << " uses "
                          << attr_num_devices
                          << " devices while computation uses " << num_devices
                          << " devices";
    return mlir::failure();
  }

  if (local_view_attr != nullptr) {
    // The arguments to the function are already sharded, so we do not
    // need to shard them again.
    func_op.setArgAttr(
        arg_idx, kHloShardingAttrName,
        builder.getStringAttr(xla::HloSharding::Replicate().ToString()));
  } else {
    const auto sharding_param_attr =
        func_op.getArgAttrOfType<IfrtShardingParamAttr>(arg_idx,
                                                        kIfrtShardingAttrName);
    auto hlo_sharding =
        xla::ifrt::support::ToHloSharding(sharding_param_attr.getSharding());
    if (!hlo_sharding.ok()) {
      func_op.emitOpError() << "can't lower sharding of input #" << arg_idx
                            << ". Sharding: " << sharding_param_attr << ". "
                            << hlo_sharding.status().message();
      return mlir::failure();
    }
    func_op.setArgAttr(arg_idx, kHloShardingAttrName,
                       builder.getStringAttr(hlo_sharding.value().ToString()));
  }
  return mlir::success();
}

mlir::LogicalResult SetResultHloSharding(mlir::func::FuncOp func_op,
                                         mlir::UnitAttr local_view_attr,
                                         int num_devices,
                                         mlir::OpBuilder& builder,
                                         int res_idx) {
  const auto sharding_attr =
      func_op.getResultAttrOfType<IfrtShardingAttrInterface>(
          res_idx, kIfrtShardingAttrName);

  if (sharding_attr == nullptr) {
    // The op has already been visited, and the IFRT attributes have been
    // removed. Verify that kHloShardingAttrName has been set.
    if (func_op.getResultAttr(res_idx, kHloShardingAttrName) == nullptr) {
      func_op.emitOpError()
          << "can't find `" << kIfrtShardingAttrName
          << "` attribute of output #" << res_idx << " to set `"
          << kHloShardingAttrName << "` attribute";
      return mlir::failure();
    }
    return mlir::success();
  }

  if (IsUnspecifiedSharding(sharding_attr)) {
    // Sharding is not specified so we cannot lower to kHloShardingAttrName.
    return mlir::success();
  }

  // Verify that the output is sharded over the same number of devices as
  // the computation.
  if (int attr_num_devices = sharding_attr.NumDevices();
      attr_num_devices != num_devices) {
    func_op.emitOpError() << "can't lower sharding of output #" << res_idx
                          << ". Sharding: " << sharding_attr << " uses "
                          << attr_num_devices
                          << " devices while computation uses " << num_devices
                          << " devices";
    return mlir::failure();
  }

  if (local_view_attr != nullptr) {
    // The results of the function are already sharded, so we do not need
    // to shard them again.
    func_op.setResultAttr(
        res_idx, kHloShardingAttrName,
        builder.getStringAttr(xla::HloSharding::Replicate().ToString()));
  } else {
    const auto sharding_param_attr =
        func_op.getResultAttrOfType<IfrtShardingParamAttr>(
            res_idx, kIfrtShardingAttrName);
    auto hlo_sharding =
        xla::ifrt::support::ToHloSharding(sharding_param_attr.getSharding());
    if (!hlo_sharding.ok()) {
      func_op.emitOpError() << "can't lower sharding of output #" << res_idx
                            << ". Sharding: " << sharding_param_attr << ". "
                            << hlo_sharding.status().message();
      return mlir::failure();
    }
    func_op.setResultAttr(
        res_idx, kHloShardingAttrName,
        builder.getStringAttr(hlo_sharding.value().ToString()));
  }
  return mlir::success();
}

// Pass that does the following:
// 1) sets kHloMemoryKindAttrName attribute to the corresponding
// memory kind if kIfrtMemoryKindAttrName is set.
// 2) transforms kIfrtShardingAttrName attribute on the main FuncOp inputs and
// outputs to HloSharding and sets it to kHloShardingAttrName attribute. This
// is only done if the module is not sdy partitioned.

class IfrtLowerAtomProgramMetadataToXlaPass
    : public impl::IfrtLowerAtomProgramMetadataToXlaPassBase<
          IfrtLowerAtomProgramMetadataToXlaPass> {
 public:
  using impl::IfrtLowerAtomProgramMetadataToXlaPassBase<
      IfrtLowerAtomProgramMetadataToXlaPass>::
      IfrtLowerAtomProgramMetadataToXlaPassBase;

  void runOnOperation() override;
};

void IfrtLowerAtomProgramMetadataToXlaPass::runOnOperation() {
  mlir::OpBuilder builder(&getContext());
  mlir::ModuleOp module_op = getOperation();
  const auto num_devices_attr =
      module_op->getAttrOfType<mlir::IntegerAttr>(kIfrtNumDevicesAttrName);
  if (num_devices_attr == nullptr) {
    module_op.emitOpError()
        << "module `" << module_op.getSymName().value_or("unknown").str()
        << "` must have `" << kIfrtNumDevicesAttrName.str() << "` attribute";
    signalPassFailure();
    return;
  }

  // If the ModuleOp has a compile options key, then try to use the provided
  // compile options.
  auto compile_options_key =
      module_op->getAttrOfType<mlir::StringAttr>(kIfrtCompileOptionsKey);
  absl::StatusOr<std::optional<xla::CompileOptions>>
      compile_options_override_or = GetModuleXlaCompileOverrides(
          compile_options_key, compile_options->compile_options_overrides);

  if (!compile_options_override_or.ok()) {
    module_op.emitOpError()
        << "Unexpected error getting compile options override: "
        << compile_options_override_or.status().message();
    signalPassFailure();
    return;
  }

  bool is_sdy = false;
  if (compile_options_override_or->has_value()) {
    is_sdy = compile_options_override_or->value()
                 .executable_build_options.use_shardy_partitioner();
  }
  // TODO(icgog): If Shardy is used, then verify that the sdy shardings are set.

  int num_devices = num_devices_attr.getInt();
  mlir::func::FuncOp func_op = GetMainFunction(module_op);
  mlir::UnitAttr local_view_attr =
      module_op->getAttrOfType<mlir::UnitAttr>(kIfrtLocalViewAttrName);

  // Lower input shardings.
  for (int arg_idx = 0; arg_idx < func_op.getNumArguments(); ++arg_idx) {
    const auto memory_kind_attr = func_op.getArgAttrOfType<mlir::StringAttr>(
        arg_idx, kIfrtMemoryKindAttrName);
    if (memory_kind_attr) {
      func_op.setArgAttr(arg_idx, kHloMemoryKindAttrName, memory_kind_attr);
    }
    // Only set the HLO shardings if Shardy is not used.
    if (!is_sdy &&
        mlir::failed(SetArgHloSharding(func_op, local_view_attr, num_devices,
                                       builder, arg_idx))) {
      signalPassFailure();
      return;
    }
  }

  // Lower output shardings.
  for (int res_idx = 0; res_idx < func_op.getNumResults(); ++res_idx) {
    if (const auto memory_kind_attr =
            func_op.getResultAttrOfType<mlir::StringAttr>(
                res_idx, kIfrtMemoryKindAttrName);
        memory_kind_attr) {
      func_op.setResultAttr(res_idx, kHloMemoryKindAttrName, memory_kind_attr);
    }
    // Only set the HLO shardings if Shardy is not used.
    if (!is_sdy &&
        mlir::failed(SetResultHloSharding(func_op, local_view_attr, num_devices,
                                          builder, res_idx))) {
      signalPassFailure();
      return;
    }
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
