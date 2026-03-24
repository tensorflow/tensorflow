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

#include "xla/hlo/experimental/auto_sharding/auto_sharding_stablehlo_pass.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/auto_partitioner_registry.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_option.h"
#include "xla/hlo/experimental/auto_sharding/stablehlo_utils.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/service/spmd/shardy/sdy_round_trip/dedup_meshes.h"

namespace xla {
namespace spmd {
namespace sdy = ::mlir::sdy;

void RegisterDialectDependencies(mlir::DialectRegistry& registry) {
  // We are loading all dialects here, otherwise the pass will fail in
  // multi-threaded pass manager execution.
  registry.insert<mlir::func::FuncDialect, mlir::mhlo::MhloDialect,
                  mlir::chlo::ChloDialect, sdy::SdyDialect,
                  mlir::stablehlo::StablehloDialect, mlir::shape::ShapeDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                  mlir::ub::UBDialect>();
  mlir::func::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  registry.insert<mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                  mlir::shape::ShapeDialect>();
}

namespace {

using ::mlir::ModuleOp;
using ::mlir::StringRef;

class AutoShardingWrapperPass
    : public mlir::PassWrapper<AutoShardingWrapperPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutoShardingWrapperPass);

  void runOnOperation() final {
    ModuleOp module_op = getOperation();
    mlir::MLIRContext* context = &getContext();
    // Get mesh from Shardy module.
    auto mesh_ops = module_op.getOps<sdy::MeshOp>();
    if (mesh_ops.empty()) {
      signalPassFailure();
    }
    sdy::MeshOp mesh_op_handle;
    if (llvm::hasSingleElement(mesh_ops)) {
      mesh_op_handle = *mesh_ops.begin();
    } else {
      // In case of multiple mesh ops, use the one with the most axes.
      // TODO(b/426573047): Need to support maximal mesh axes.
      LOG(WARNING)
          << "Multiple mesh ops found, using the one with the most axes."
          << "The other mesh ops will be dropped";
      mesh_op_handle = *std::max_element(
          mesh_ops.begin(), mesh_ops.end(), [](sdy::MeshOp a, sdy::MeshOp b) {
            return a.getMesh().getAxes().size() < b.getMesh().getAxes().size();
          });
    }
    sdy::MeshAttr sdy_mesh = mesh_op_handle.getMesh();

    // Shardy -> HLO
    absl::StatusOr<std::unique_ptr<xla::HloModule>> hlo_module_or_status =
        ConvertShardyToHlo(getOperation());
    if (!hlo_module_or_status.ok()) {
      signalPassFailure();
    }

    std::unique_ptr<xla::HloModule> hlo_module =
        std::move(hlo_module_or_status.value());

    // Invoke AutoSharding on the HLO module.
    mlir::ArrayRef<sdy::MeshAxisAttr> axes = sdy_mesh.getAxes();
    std::vector<int64_t> device_mesh_shape;
    for (sdy::MeshAxisAttr axis : axes) {
      device_mesh_shape.push_back(axis.getSize());
    }
    AutoShardingOption option;
    option.enable = true;
    option.device_mesh_shape = device_mesh_shape;
    // Keep the mesh shape unchanged.
    option.allow_mixed_mesh_shape = false;
    option.replace_sharding_with_copy = false;
    // TODO(hanruobing): Add an option to control whether to keep the original
    // sharding or not. The current behavior is to keep the original sharding.
    // TODO(b/424109294): Figure out whether we need to pass backend-specific
    // AliasInfo here.
    AliasInfo alias_info;
    if (!AutoSharding(option, &alias_info).Run(hlo_module.get()).ok()) {
      signalPassFailure();
    }

    // HLO -> Shardy
    absl::StatusOr<mlir::OwningOpRef<ModuleOp>>
        shardy_stablehlo_module_or_status =
            ConvertHloToShardyStablehlo(*hlo_module, module_op.getContext());
    if (!shardy_stablehlo_module_or_status.ok()) {
      signalPassFailure();
    }

    // Replace to the AutoSharding transformed module.
    module_op.getBodyRegion().takeBody(
        shardy_stablehlo_module_or_status.value().get().getBodyRegion());
    // Re-insert the mesh to keep the axes name consistent.
    mlir::SymbolTable symbol_table(module_op);
    mlir::OpBuilder builder(context);

    auto original_mesh_op =
        sdy::MeshOp::create(builder, module_op.getLoc(), "mesh", sdy_mesh);
    symbol_table.insert(original_mesh_op, module_op.getBody()->begin());
    mlir::PassManager dedup_pm(context);
    dedup_pm.addPass(xla::sdy::createSdyRoundTripDedupMeshesPass());
    if (mlir::failed(dedup_pm.run(module_op))) {
      return signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "auto-sharding-automatic-partition";
  }

  StringRef getDescription() const override {
    return "Invokes AutoSharding on the module.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    RegisterDialectDependencies(registry);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateAutoShardingWrapperPass() {
  return std::make_unique<AutoShardingWrapperPass>();
}

void AddAutoShardingToPipeline(mlir::OpPassManager& pm) {
  pm.addPass(CreateAutoShardingWrapperPass());
}

void RegisterAutoSharding() {
  sdy::AutoPartitionerRegistry::setCallback(&AddAutoShardingToPipeline);
}

void RegisterAutoShardingIfRegistryEmpty() {
  if (!sdy::AutoPartitionerRegistry::isRegistered()) {
    RegisterAutoSharding();
  }
}

}  // namespace spmd
}  // namespace xla
