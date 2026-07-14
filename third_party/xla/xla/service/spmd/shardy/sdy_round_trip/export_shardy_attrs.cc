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

#include "xla/service/spmd/shardy/sdy_round_trip/export_shardy_attrs.h"

#include <cstdint>
#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::DictionaryAttr;
using ::mlir::LogicalResult;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::Pass;
using ::mlir::PassWrapper;
using ::mlir::SmallVector;
using ::mlir::StringRef;
using ::mlir::Value;
using ::mlir::func::FuncOp;

using ::mlir::stablehlo::CustomCallOp;

using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::kShardingRuleAttr;
using ::mlir::sdy::MeshOp;
using ::mlir::sdy::OpShardingRuleAttr;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;

// Saves `shardingPerValueAttr` including any existing `frontendAttributes` on
// the `op`.
void saveOpShardingPerValueAttr(
    Operation* op, TensorShardingPerValueAttr shardingPerValueAttr) {
  setFrontendAttribute(op,
                       xla::ToStringRef(HloSharding::kShardingFrontendAttrName),
                       shardingPerValueAttr);
}

// Exports sharding rules from `kShardingRuleAttr` to
// `kShardingRuleRoundTripAttr` as a frontend attribute.
void exportShardingRules(FuncOp funcOp) {
  funcOp.front().walk([&](Operation* op) {
    if (auto oldShardingRule =
            op->getAttrOfType<OpShardingRuleAttr>(kShardingRuleAttr)) {
      setFrontendAttribute(op, kShardingRuleRoundTripAttr, oldShardingRule);
      op->removeAttr(kShardingRuleAttr);
    }
  });
}

// Converts the shardings from `kShardingAttr` into
// `HloSharding::kShardingFrontendAttrName`.
LogicalResult exportFunc(FuncOp funcOp, OpBuilder& builder) {
  for (int64_t argNum = 0; argNum < funcOp.getNumArguments(); ++argNum) {
    if (auto oldSharding = funcOp.getArgAttrOfType<TensorShardingAttr>(
            argNum, kShardingAttr)) {
      setFrontendAttribute(
          funcOp, xla::ToStringRef(HloSharding::kShardingFrontendAttrName),
          oldSharding, argNum);
    }
  }

  Operation* terminatorOp = mlir::sdy::getBodyTerminator(funcOp);
  builder.setInsertionPoint(terminatorOp);
  for (mlir::OpOperand& returnOperand : terminatorOp->getOpOperands()) {
    if (auto sharding = funcOp.getResultAttrOfType<TensorShardingAttr>(
            returnOperand.getOperandNumber(), kShardingAttr)) {
      // We cannot save the result shardings as frontend attributes.
      // StableHLO->HLO conversion converts `mhlo.sharding`s on the results to a
      // tuple sharding on the ROOT instruction, but it discards the frontend
      // attributes on the MLIR results. Thus, instead of making the conversion
      // handle that, push the shardings to a temporary custom op for each
      // FuncOp result with a sharding. Then during import, we will copy the
      // Op's sharding to the FuncOp's result and delete te temporary custom
      // call.
      Value returnValue = returnOperand.get();
      auto customCallOp = CustomCallOp::create(
          builder, returnValue.getLoc(), returnValue.getType(), returnValue);
      customCallOp.setCallTargetName(kFuncResultShardingTargetName);
      // Want to prevent the canonicalizer from de-duplicating func sharding
      // custom calls which actually may have different sharding attributes.
      customCallOp.setHasSideEffect(true);
      saveOpShardingPerValueAttr(
          customCallOp,
          TensorShardingPerValueAttr::get(customCallOp.getContext(), sharding));
      returnOperand.set(customCallOp.getResult(0));
    }
  }

  funcOp.front().walk([&](Operation* op) {
    if (TensorShardingPerValueAttr oldShardingPerValue =
            mlir::sdy::getShardingPerValue(op)) {
      saveOpShardingPerValueAttr(op, oldShardingPerValue);
    }
  });

  return mlir::success();
}

class SdyRoundTripExportShardyAttrsPass
    : public PassWrapper<SdyRoundTripExportShardyAttrsPass,
                         mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      SdyRoundTripExportShardyAttrsPass)

  SdyRoundTripExportShardyAttrsPass(bool enableHloShardingV3)
      : enableHloShardingV3(enableHloShardingV3) {}

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext* context = moduleOp.getContext();
    auto builder = OpBuilder(context);

    if (enableHloShardingV3) {
      // If HloShardingV3 is enabled, frontend attributes are used only for
      // sharding rules
      for (auto funcOp : moduleOp.getOps<FuncOp>()) {
        exportShardingRules(funcOp);
      }
    } else {
      for (auto funcOp : moduleOp.getOps<FuncOp>()) {
        if (mlir::failed(exportFunc(funcOp, builder))) {
          signalPassFailure();
        }
        exportShardingRules(funcOp);
      }
      SmallVector<NamedAttribute> stablehloMeshes;
      // Saves the MeshOps for StableHLO<->HLO round-trip and removes them from
      // the ModuleOp.
      for (MeshOp meshOp : moduleOp.getOps<MeshOp>()) {
        stablehloMeshes.emplace_back(meshOp.getSymNameAttr(),
                                     meshOp.getMeshAttr());
      }
      if (!stablehloMeshes.empty()) {
        setFrontendAttribute(moduleOp, kMeshesRoundTripAttr,
                             DictionaryAttr::get(context, stablehloMeshes));
      }
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-export-shardy-attrs";
  }

  StringRef getDescription() const override {
    return "Converts the shardy attributes from "
           "kShardingAttr/kShardingRuleAttr to "
           "HloSharding::kShardingFrontendAttrName/kShardingRuleRoundTripAttr "
           "in the HLO frontend attributes and saves the mesh symbols as "
           "kMeshesRoundTripAttr in the module frontend attributes.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect, mlir::stablehlo::StablehloDialect>();
  }

 private:
  bool enableHloShardingV3;
};

}  // namespace

void registerSdyRoundTripExportShardyAttrsPass() {
  mlir::registerPass([]() {
    return createSdyRoundTripExportShardyAttrsPass(
        /*enableHloShardingV3=*/false);
  });
}

std::unique_ptr<Pass> createSdyRoundTripExportShardyAttrsPass(
    bool enableHloShardingV3) {
  return std::make_unique<SdyRoundTripExportShardyAttrsPass>(
      enableHloShardingV3);
}

}  // namespace sdy
}  // namespace xla
