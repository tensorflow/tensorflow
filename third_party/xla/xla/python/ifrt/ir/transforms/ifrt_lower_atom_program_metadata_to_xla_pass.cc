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
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/python/ifrt/support/sharding_conversions.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTLOWERATOMPROGRAMMETADATATOXLAPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

// Pass that does the following:
// 1) transforms kIfrtShardingAttrName attribute on the main FuncOp inputs and
// outputs to HloSharding.
// 2) sets FuncOps input/outputs kHloShardingAttrName attribute to the
// corresponding computed HloSharding.
// 3) sents kHloMemoryKindAttrName attribute to the corresponding
// memory kind if kIfrtMemoryKindAttrName is set.

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

  // TODO(b/433244129) - Do not use kIsSdyPartitioned attr and default to true
  // after 6 month bwd compatibility window.
  bool is_sdy = module_op->hasAttr(kIsSdyPartitioned);
  if (is_sdy) {
    mlir::emitWarning(module_op->getLoc())
        << "`" << kIsSdyPartitioned
        << "` attribute is deprecated and will be removed. See b/433244129."
           " Please use `compile_options_override` to specify sharding.";
  }
  if (compile_options_override_or->has_value()) {
    is_sdy = compile_options_override_or->value()
                 .executable_build_options.use_shardy_partitioner();
  }

  int num_devices = num_devices_attr.getInt();
  mlir::func::FuncOp func_op = GetMainFunction(module_op);
  auto local_view_attr =
      module_op->getAttrOfType<mlir::UnitAttr>(kIfrtLocalViewAttrName);
  // Lower input shardings.
  for (int i = 0; i < func_op.getNumArguments(); ++i) {
    const auto sharding_attr =
        func_op.getArgAttrOfType<IfrtShardingAttrInterface>(
            i, kIfrtShardingAttrName);
    if (sharding_attr == nullptr) {
      // The op has already been visited, and the IFRT attributes have been
      // removed. Verify that kHloShardingAttrName has been set if the module is
      // not sdy partitioned.
      if (!is_sdy && func_op.getArgAttr(i, kHloShardingAttrName) == nullptr) {
        func_op.emitOpError() << "can't find `" << kIfrtShardingAttrName
                              << "` attribute of input #" << i << " to set `"
                              << kHloShardingAttrName << "` attribute";
        signalPassFailure();
        return;
      }
      continue;
    }
    if (llvm::isa<IfrtUnspecifiedShardingAttr>(sharding_attr)) {
      // Sharding is not specified so we cannot lower to kHloShardingAttrName.
      continue;
    }

    // Verify that the input is sharded over the same number of devices as
    // the computation.
    auto attr_num_devices = sharding_attr.NumDevices();
    if (attr_num_devices != num_devices) {
      func_op.emitOpError()
          << "can't lower sharding of input #" << i
          << ". Sharding: " << sharding_attr << " uses " << attr_num_devices
          << " devices while computation uses " << num_devices << " devices";
      signalPassFailure();
      return;
    }

    // Shardy doesn't use HLO shardings, but the old propagation system GSPMD
    // does, so we don't need to set HLO shardings for sdy.
    if (!is_sdy) {
      if (local_view_attr != nullptr) {
        // The arguments to the function are already sharded, so we do not
        // need to shard them again.
        func_op.setArgAttr(
            i, kHloShardingAttrName,
            builder.getStringAttr(xla::HloSharding::Replicate().ToString()));
      } else {
        const auto sharding_param_attr =
            func_op.getArgAttrOfType<IfrtShardingParamAttr>(
                i, kIfrtShardingAttrName);
        auto hlo_sharding = xla::ifrt::support::ToHloSharding(
            sharding_param_attr.getSharding());
        if (!hlo_sharding.ok()) {
          func_op.emitOpError() << "can't lower sharding of input #" << i
                                << ". Sharding: " << sharding_param_attr << ". "
                                << hlo_sharding.status().message();
          signalPassFailure();
          return;
        }
        func_op.setArgAttr(
            i, kHloShardingAttrName,
            builder.getStringAttr(hlo_sharding.value().ToString()));
      }
    }
    const auto memory_kind_attr =
        func_op.getArgAttrOfType<mlir::StringAttr>(i, kIfrtMemoryKindAttrName);
    if (memory_kind_attr) {
      func_op.setArgAttr(i, kHloMemoryKindAttrName, memory_kind_attr);
    }
  }

  // Lower output shardings.
  for (int i = 0; i < func_op.getNumResults(); ++i) {
    const auto sharding_attr =
        func_op.getResultAttrOfType<IfrtShardingAttrInterface>(
            i, kIfrtShardingAttrName);
    if (sharding_attr == nullptr) {
      // The op has already been visited, and the IFRT attributes have been
      // removed. Verify that kHloShardingAttrName has been set if the module is
      // not sdy partitioned.
      if (!is_sdy &&
          func_op.getResultAttr(i, kHloShardingAttrName) == nullptr) {
        func_op.emitOpError() << "can't find `" << kIfrtShardingAttrName
                              << "` attribute of output #" << i << " to set `"
                              << kHloShardingAttrName << "` attribute";
        signalPassFailure();
        return;
      }
      continue;
    } else if (llvm::isa<IfrtUnspecifiedShardingAttr>(sharding_attr)) {
      // Sharding is not specified so we cannot lower to kHloShardingAttrName.
      continue;
    }

    // Verify that the output is sharded over the same number of devices as
    // the computation.
    auto attr_num_devices = sharding_attr.NumDevices();
    if (attr_num_devices != num_devices) {
      func_op.emitOpError()
          << "can't lower sharding of output #" << i
          << ". Sharding: " << sharding_attr << " uses " << attr_num_devices
          << " devices while computation uses " << num_devices << " devices";
      signalPassFailure();
      return;
    }

    if (!is_sdy) {
      if (local_view_attr != nullptr) {
        // The results of the function are already sharded, so we do not need
        // to shard them again.
        func_op.setResultAttr(
            i, kHloShardingAttrName,
            builder.getStringAttr(xla::HloSharding::Replicate().ToString()));
      } else {
        const auto sharding_param_attr =
            func_op.getResultAttrOfType<IfrtShardingParamAttr>(
                i, kIfrtShardingAttrName);
        auto hlo_sharding = xla::ifrt::support::ToHloSharding(
            sharding_param_attr.getSharding());
        if (!hlo_sharding.ok()) {
          func_op.emitOpError() << "can't lower sharding of output #" << i
                                << ". Sharding: " << sharding_param_attr << ". "
                                << hlo_sharding.status().message();
          signalPassFailure();
          return;
        }
        func_op.setResultAttr(
            i, kHloShardingAttrName,
            builder.getStringAttr(hlo_sharding.value().ToString()));
      }
    }
    const auto memory_kind_attr = func_op.getResultAttrOfType<mlir::StringAttr>(
        i, kIfrtMemoryKindAttrName);
    if (memory_kind_attr) {
      func_op.setResultAttr(i, kHloMemoryKindAttrName, memory_kind_attr);
    }
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
