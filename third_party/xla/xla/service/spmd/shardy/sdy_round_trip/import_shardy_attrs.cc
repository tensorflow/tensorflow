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

#include "xla/service/spmd/shardy/sdy_round_trip/import_shardy_attrs.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/sdy_round_trip/dedup_meshes.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::Attribute;
using ::mlir::DictionaryAttr;
using ::mlir::IRRewriter;
using ::mlir::LogicalResult;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::Operation;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::FuncOp;

using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::kShardingRuleAttr;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::OpShardingRuleAttr;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;
using ::mlir::stablehlo::CustomCallOp;

namespace stablehlo = ::mlir::stablehlo;

HloSharding parseShardingFromString(StringAttr sharding) {
  absl::StatusOr<xla::HloSharding> hloSharding =
      xla::ParseSharding(sharding.str());
  CHECK(hloSharding.ok());
  return *hloSharding;
}

CustomCallOp dynCastX64CombineCustomCall(Operation* op) {
  auto customCallOp = mlir::dyn_cast<CustomCallOp>(op);
  if (!customCallOp || customCallOp.getCallTargetName() != "X64Combine") {
    return nullptr;
  }
  return customCallOp;
}

CustomCallOp getX64CombineOnFuncResultSharding(
    CustomCallOp funcResultSharding) {
  if (funcResultSharding.getNumResults() != 2 ||
      !funcResultSharding.getResult(0).hasOneUse() ||
      !funcResultSharding.getResult(1).hasOneUse()) {
    return nullptr;
  }
  Operation* lhsUser = *funcResultSharding.getResult(0).user_begin();
  Operation* rhsUser = *funcResultSharding.getResult(1).user_begin();
  if (lhsUser != rhsUser) {
    return nullptr;
  }
  return dynCastX64CombineCustomCall(lhsUser);
}

// TODO(kostiantynl): b/448858211 when API is fixed, use
// sharding.openShardingDims() instead.
TensorShardingAttr openShardingDims(TensorShardingAttr sharding) {
  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings(
      sharding.getDimShardings().begin(), sharding.getDimShardings().end());
  for (auto& dimSharding : dimShardings) {
    dimSharding = mlir::sdy::DimensionShardingAttr::get(
        sharding.getContext(), dimSharding.getAxes(), /*isClosed=*/false,
        /*priority=*/dimSharding.getPriority());
  }
  return TensorShardingAttr::get(sharding.getContext(), sharding.getMeshOrRef(),
                                 dimShardings, sharding.getReplicatedAxes(),
                                 sharding.getUnreducedAxes());
}

void handleFuncResultSharding(CustomCallOp funcResultSharding, FuncOp funcOp,
                              DictionaryAttr dictAttr, IRRewriter& rewriter) {
  // This is a temporary CustomCallOp that holds the sharding from a
  // func result. When importing we want to move that sharding to the
  // func result and delete the CustomCallOp.
  auto shardingPerValueAttr = parseStringAttr<TensorShardingPerValueAttr>(
      dictAttr, xla::ToStringRef(HloSharding::kShardingFrontendAttrName));

  auto resultUses = funcResultSharding->getUses();
  auto x64CombineOp = getX64CombineOnFuncResultSharding(funcResultSharding);
  if (x64CombineOp) {
    // X64Rewriter pass will pass through the two split 32-bit operands to
    // the `xla.sdy.FuncResultSharding`, which will return two 32-bit results,
    // that would then be passed to a `X64Combine` custom-call. Therefore, we
    // need to look at the uses of the `X64Combine` instead to find the
    // corresponding `func.return` op.
    mlir::sdy::setShardings(x64CombineOp, shardingPerValueAttr);
    resultUses = x64CombineOp->getUses();
  } else if (auto* defOp = funcResultSharding.getOperand(0).getDefiningOp();
             defOp && funcResultSharding->use_empty()) {
    // It `funcResultSharding` has no uses, it is likely because it has a
    // dimension of size 0 (i.e. 0 num-elements), in which case its uses will be
    // replaced with a constant of the same shape, which will replace the
    // operand of the `funcResultSharding`.
    resultUses = defOp->getUses();
  }
  TensorShardingAttr sharding = shardingPerValueAttr.getSharding(0);
  bool hasNonFuncReturnUses = false;
  for (mlir::OpOperand& use : llvm::make_early_inc_range(resultUses)) {
    if (mlir::isa<mlir::func::ReturnOp>(use.getOwner())) {
      funcOp.setResultAttr(use.getOperandNumber(), kShardingAttr, sharding);
    } else if (use.getOwner() != funcResultSharding &&
               !dynCastX64CombineCustomCall(use.getOwner())) {
      hasNonFuncReturnUses = true;
    }
  }
  if (hasNonFuncReturnUses && !x64CombineOp) {
    // If there are users that are not the func return op, which might happen
    // due to inlined func ops that originally had result shardings, we replace
    // the `xla.sdy.FuncResultSharding` with a `ShardingConstraintOp` to
    // preserve the original func result sharding, but open all sharding
    // dimensions.
    rewriter.setInsertionPoint(funcResultSharding);
    CHECK_EQ(funcResultSharding.getNumOperands(), 1);
    rewriter.replaceOpWithNewOp<mlir::sdy::ShardingConstraintOp>(
        funcResultSharding, funcResultSharding.getOperand(0),
        openShardingDims(sharding));
  } else {
    rewriter.replaceOp(funcResultSharding, funcResultSharding.getOperands());
  }
}

// Builds the shardy attributes coming from Shardy previously. This means
// the module was exported from Shardy and we are now round-tripping back.
// This should happen after the meshes were created from the `ModuleOp` attrs
// (see `SdyRoundTripImportShardyAttrsPass`).
void convertShardyAttrs(FuncOp funcOp, IRRewriter& rewriter,
                        bool enableHloShardingV3) {
  // Copy over the argument shardings, but not the result shardings yet.
  // We need to wait until after we've converted all the Operations before
  // copying the result shardings.
  for (auto [argNum, argType] : llvm::enumerate(funcOp.getArgumentTypes())) {
    if (enableHloShardingV3) {
      if (auto oldSharding =
              funcOp.getArgAttrOfType<StringAttr>(argNum, kXlaShardingAttr)) {
        funcOp.setArgAttr(
            argNum, kShardingAttr,
            convertToSdyShardingAttr(parseShardingFromString(oldSharding),
                                     argType, funcOp.getContext()));
      }
    } else {
      // Attempt to extract the TensorShardingAttr from the frontend attributes
      // of the function argument/result.
      if (DictionaryAttr dictAttr = getFuncArgFrontendAttrs(funcOp, argNum)) {
        if (auto sharding = parseStringAttr<TensorShardingAttr>(
                dictAttr,
                xla::ToStringRef(HloSharding::kShardingFrontendAttrName))) {
          funcOp.setArgAttr(argNum, kShardingAttr, sharding);
          removeFrontendAttribute(
              funcOp, xla::ToStringRef(HloSharding::kShardingFrontendAttrName),
              argNum);
        }
      }
    }
    funcOp.removeArgAttr(argNum, kXlaShardingAttr);
  }

  // Due to `SdyRoundTripExportShardingsPass` keeping `mhlo.sharding`s, remove
  // them purely for cleanliness of the module.
  for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
    if (enableHloShardingV3) {
      // Result shardings only need to be handled in HloShardingV3 case where we
      // don't use FuncResultSharding custom call.
      if (auto oldSharding = funcOp.getResultAttrOfType<StringAttr>(
              resNum, kXlaShardingAttr)) {
        funcOp.setResultAttr(
            resNum, kShardingAttr,
            convertToSdyShardingAttr(parseShardingFromString(oldSharding),
                                     funcOp.getResultTypes()[resNum],
                                     funcOp.getContext()));
      }
    }
    funcOp.removeResultAttr(
        resNum, StringAttr::get(funcOp.getContext(), kXlaShardingAttr));
  }

  // Extract the round-tripped shardy attributes from the operations.
  funcOp.front().walk([&](Operation* op) {
    if (!enableHloShardingV3) {
      op->removeAttr(kXlaShardingAttr);
    }
    DictionaryAttr dictAttr = getFrontendAttrs(op);
    // No frontend attributes to import.
    if (!enableHloShardingV3 && !dictAttr) {
      return;
    }

    // Import sharding rules.
    if (dictAttr) {
      if (auto shardingRuleAttr = parseStringAttr<OpShardingRuleAttr>(
              dictAttr, kShardingRuleRoundTripAttr)) {
        op->setAttr(kShardingRuleAttr, shardingRuleAttr);
        removeFrontendAttribute(op, kShardingRuleRoundTripAttr);
      }
    }

    // No shardings to import.
    if (enableHloShardingV3 &&
        !op->getAttrOfType<StringAttr>(kXlaShardingAttr)) {
      return;
    }

    // `SendOp`, `RecvOp`, and `AfterAllOp` can have a sharding when doing TPU
    // callbacks through JAX.
    if (mlir::isa<stablehlo::SendOp, stablehlo::RecvOp, stablehlo::AfterAllOp>(
            op)) {
      if (enableHloShardingV3) {
        auto sharding = op->getAttrOfType<StringAttr>(kXlaShardingAttr);
        op->setAttr(kShardingAttr, convertToSdySharding(
                                       parseShardingFromString(sharding),
                                       op->getResultTypes(), op->getContext()));
      } else {
        if (auto sharding = parseStringAttr<TensorShardingPerValueAttr>(
                dictAttr,
                xla::ToStringRef(HloSharding::kShardingFrontendAttrName))) {
          op->setAttr(kShardingAttr, sharding);
        }
      }
    }
    // NOTE: we are only setting the sharding on known custom-calls. For any
    // other op that has a `HloSharding::kShardingFrontendAttrName` we discard
    // it. XLA sometimes creates new instructions, copying over the operand's
    // frontend attrs, which may mean the shapes are wrong when the new
    // instruction is a reshape for example. This does mean we can't fully
    // round-trip b/w HLO and MLIR after SDY propagation.
    if (auto customCallOp = mlir::dyn_cast<CustomCallOp>(op)) {
      StringRef targetName = customCallOp.getCallTargetName();
      if (targetName == kFuncResultShardingTargetName && !enableHloShardingV3) {
        handleFuncResultSharding(customCallOp, funcOp, dictAttr, rewriter);
        return;
      }
      if (targetName == kShardingCustomCallTargetName ||
          isPythonCallbackCustomCall(customCallOp)) {
        if (enableHloShardingV3) {
          auto sharding =
              customCallOp->getAttrOfType<StringAttr>(kXlaShardingAttr);
          customCallOp->setAttr(
              kShardingAttr,
              convertToSdySharding(parseShardingFromString(sharding),
                                   customCallOp->getResultTypes(),
                                   customCallOp->getContext()));
        } else {
          customCallOp->setAttr(
              kShardingAttr,
              parseStringAttr<TensorShardingPerValueAttr>(
                  dictAttr,
                  xla::ToStringRef(HloSharding::kShardingFrontendAttrName)));
        }
      }
    }

    op->removeAttr(kXlaShardingAttr);

    if (!enableHloShardingV3) {
      removeFrontendAttribute(
          op, xla::ToStringRef(HloSharding::kShardingFrontendAttrName));
    }
  });
}

using ShardingSetter =
    absl::AnyInvocable<void(FuncOp, int64_t, TensorShardingAttr)>;
LogicalResult handleFuncTupleInOutShardings(ModuleOp moduleOp, FuncOp funcOp,
                                            StringRef attrName,
                                            ShardingSetter shardingSetter,
                                            int64_t expectedNumShardings) {
  std::optional<TensorShardingPerValueAttr> shardings =
      tryGetFrontendAttr<TensorShardingPerValueAttr>(moduleOp, attrName);
  if (!shardings.has_value()) {
    return mlir::success();
  }

  if (shardings->size() != expectedNumShardings) {
    moduleOp.emitError() << "Number of actual shardings (" << shardings->size()
                         << ") does not match number of expected shardings ("
                         << expectedNumShardings << "for " << attrName
                         << ") for function.";
    return mlir::failure();
  }

  for (auto [argNum, argSharding] :
       llvm::enumerate(shardings->getShardings())) {
    shardingSetter(funcOp, argNum, argSharding);
  }

  removeFrontendAttribute(moduleOp, attrName);

  return mlir::success();
}

class SdyRoundTripImportShardyAttrsPass
    : public mlir::PassWrapper<SdyRoundTripImportShardyAttrsPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      SdyRoundTripImportShardyAttrsPass)

  SdyRoundTripImportShardyAttrsPass(bool enableHloShardingV3)
      : enableHloShardingV3(enableHloShardingV3) {}

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    IRRewriter rewriter(moduleOp);
    SymbolTable symbolTable(moduleOp);

    // We can use the saved string attributes to restore the original mesh and
    // value shardings with the original mesh axis names and priorities on the
    // sharding. If there is no `kMeshesRoundTripAttr, there were no meshes in
    // the original Shardy model.

    if (!enableHloShardingV3) {
      // Insert the meshes before any functions.
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      std::optional<DictionaryAttr> meshesAttr =
          tryGetFrontendAttr<DictionaryAttr>(moduleOp, kMeshesRoundTripAttr);
      mlir::ArrayRef<NamedAttribute> sdyMeshes =
          meshesAttr.has_value() ? meshesAttr.value().getValue()
                                 : mlir::ArrayRef<NamedAttribute>();

      for (NamedAttribute mesh : sdyMeshes) {
        auto meshAttr = mlir::cast<MeshAttr>(mesh.getValue());
        symbolTable.insert(mlir::sdy::MeshOp::create(
            rewriter, moduleOp.getLoc(), mesh.getName(), meshAttr));
      }
      removeFrontendAttribute(moduleOp, kMeshesRoundTripAttr);
    }

    // TODO (b/485486745): Remove kInTupleShardings and kOutTupleShardings
    // frontend attributes added directly at tf2xla level
    if (FuncOp mainFunc = moduleOp.lookupSymbol<FuncOp>("main")) {
      auto argShardingSetter = [](FuncOp funcOp, int64_t argNum,
                                  TensorShardingAttr argSharding) {
        setSharding(funcOp.getArgument(argNum), argSharding);
      };
      if (mlir::failed(handleFuncTupleInOutShardings(
              moduleOp, mainFunc, kInTupleShardings, argShardingSetter,
              mainFunc.getNumArguments()))) {
        signalPassFailure();
      }

      auto resultShardingSetter = [](FuncOp funcOp, int64_t resultNum,
                                     TensorShardingAttr resultSharding) {
        setFuncResultSharding(funcOp, resultNum, resultSharding);
      };
      if (mlir::failed(handleFuncTupleInOutShardings(
              moduleOp, mainFunc, kOutTupleShardings, resultShardingSetter,
              mainFunc.getNumResults()))) {
        signalPassFailure();
      }
    }

    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      convertShardyAttrs(funcOp, rewriter, enableHloShardingV3);
    }

    if (enableHloShardingV3) {
      // Lift inlined meshes, as meshes are inlined in HloShardingV3 and
      // therefore in sdy shardings generated from conversion.
      mlir::PassManager pm(moduleOp.getContext());
      pm.addPass(mlir::sdy::createLiftInlinedMeshesPass());
      pm.addPass(createSdyRoundTripDedupMeshesPass());
      if (mlir::failed(pm.run(moduleOp))) {
        signalPassFailure();
      }
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-import-shardy-attrs";
  }

  StringRef getDescription() const override {
    return "Converts the shardy attributes from strings in MHLO frontend "
           "attributes to SDY meshes, shardings and sharding rules.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect>();
  }

 private:
  bool enableHloShardingV3;
};

}  // namespace

std::unique_ptr<mlir::Pass> createSdyRoundTripImportShardyAttrsPass(
    bool enableHloShardingV3) {
  return std::make_unique<SdyRoundTripImportShardyAttrsPass>(
      enableHloShardingV3);
}

void registerSdyRoundTripImportShardyAttrsPass() {
  mlir::registerPass([]() {
    return createSdyRoundTripImportShardyAttrsPass(
        /*enableHloShardingV3=*/false);
  });
}

}  // namespace sdy
}  // namespace xla
