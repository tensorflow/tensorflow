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
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "llvm/ADT/DenseSet.h"
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
#include "mlir/IR/OperationSupport.h"
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
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"
#include "xla/side_effect_util.h"

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
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;

using ::mlir::sdy::getSharding;
using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::kShardingRuleAttr;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::OpShardingRuleAttr;
using ::mlir::sdy::PropagationBarrierOp;
using ::mlir::sdy::PropagationDirection;
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

bool handleFuncResultSharding(
    CustomCallOp funcResultSharding, FuncOp funcOp,
    llvm::SmallVector<DictionaryAttr>& funcResultAttrs, DictionaryAttr dictAttr,
    IRRewriter& rewriter) {
  // This is a temporary CustomCallOp that holds the sharding from a
  // func result. When importing we want to move that sharding to the
  // func result and delete the CustomCallOp.
  auto shardingPerValueAttr = parseStringAttr<TensorShardingPerValueAttr>(
      dictAttr, xla::ToStringRef(HloSharding::kShardingFrontendAttrName));

  auto resultUses = funcResultSharding->getUses();
  bool anyChanged = false;
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
      int64_t resNum = use.getOperandNumber();
      mlir::NamedAttrList attrs(funcResultAttrs[resNum]);
      attrs.set(kShardingAttr, sharding);
      funcResultAttrs[resNum] = attrs.getDictionary(funcOp.getContext());
      anyChanged = true;
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
  return anyChanged;
}

// The sharding information is in the `kXlaShardingAttr` attribute.
void convertShardyAttrsWithHloShardingV3(FuncOp funcOp) {
  for (auto [argNum, argType] : llvm::enumerate(funcOp.getArgumentTypes())) {
    if (auto oldSharding =
            funcOp.getArgAttrOfType<StringAttr>(argNum, kXlaShardingAttr)) {
      if (auto sdySharding = convertToSdyShardingAttr(
              parseShardingFromString(oldSharding), funcOp.getContext())) {
        funcOp.setArgAttr(argNum, kShardingAttr, sdySharding);
      }
    }
    funcOp.removeArgAttr(argNum, kXlaShardingAttr);
  }

  for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
    if (auto oldSharding =
            funcOp.getResultAttrOfType<StringAttr>(resNum, kXlaShardingAttr)) {
      if (auto sdySharding = convertToSdyShardingAttr(
              parseShardingFromString(oldSharding), funcOp.getContext())) {
        funcOp.setResultAttr(resNum, kShardingAttr, sdySharding);
      }
    }
    funcOp.removeResultAttr(resNum, kXlaShardingAttr);
  }

  if (funcOp.isExternal()) {
    return;
  }

  // Extract the round-tripped shardy attributes from the operations.
  funcOp.front().walk([&](Operation* op) {
    // Import sharding rules.
    if (DictionaryAttr dictAttr = getFrontendAttrs(op)) {
      if (auto shardingRuleAttr = parseStringAttr<OpShardingRuleAttr>(
              dictAttr, kShardingRuleRoundTripAttr)) {
        op->setAttr(kShardingRuleAttr, shardingRuleAttr);
        removeFrontendAttribute(op, kShardingRuleRoundTripAttr);
      }
    }

    auto shardingAttr = op->getAttrOfType<StringAttr>(kXlaShardingAttr);
    if (!shardingAttr) {
      return;
    }

    // `SendOp`, `RecvOp`, and `AfterAllOp` can have a sharding when doing TPU
    // callbacks through JAX. We also handle known custom-calls.
    //
    // Folloiwng the routine without HloShardingV3, we discard the
    // `kXlaShardingAttr` for any other op. We may revisit this decision in the
    // future.
    if (mlir::isa<stablehlo::SendOp, stablehlo::RecvOp, stablehlo::AfterAllOp>(
            op)) {
      op->setAttr(kShardingAttr,
                  convertToSdySharding(parseShardingFromString(shardingAttr),
                                       op->getContext()));
    } else if (auto customCallOp = mlir::dyn_cast<CustomCallOp>(op)) {
      StringRef targetName = customCallOp.getCallTargetName();
      if (targetName == kShardingCustomCallTargetName ||
          targetName == "X64Combine" ||
          isPythonCallbackCustomCall(customCallOp)) {
        customCallOp->setAttr(
            kShardingAttr,
            convertToSdySharding(parseShardingFromString(shardingAttr),
                                 customCallOp->getContext()));
      }
    }

    op->removeAttr(kXlaShardingAttr);
  });
}

// The sharding information is in the `kShardingFrontendAttrName` frontend
// attribute.
void convertShardyAttrsWithoutHloShardingV3(FuncOp funcOp,
                                            IRRewriter& rewriter) {
  // Copy over the argument shardings, but not the result shardings yet.
  // We need to wait until after we've converted all the Operations before
  // copying the result shardings.
  StringRef attributeName =
      xla::ToStringRef(HloSharding::kShardingFrontendAttrName);
  llvm::SmallVector<mlir::DictionaryAttr> funcArgAttrs;
  funcArgAttrs.reserve(funcOp.getNumArguments());
  for (int64_t argNum = 0; argNum < funcOp.getNumArguments(); argNum++) {
    // Attempt to extract the TensorShardingAttr from the frontend attributes
    // of the function argument/result.
    // TODO(b/510714593): Batch remove/set attributes through a shardy utility.
    mlir::NamedAttrList attrs(funcOp.getArgAttrDict(argNum));
    if (DictionaryAttr dictAttr = getFuncArgFrontendAttrs(funcOp, argNum)) {
      if (auto sharding =
              parseStringAttr<TensorShardingAttr>(dictAttr, attributeName)) {
        attrs.set(kShardingAttr, sharding);
        llvm::SmallVector<NamedAttribute> existingAttributes =
            getExistingFrontendAttributes(dictAttr,
                                          /*excludedAttribute=*/attributeName);
        if (!existingAttributes.empty()) {
          attrs.set(
              kFrontendAttributesAttr,
              DictionaryAttr::get(funcOp.getContext(), existingAttributes));
        } else {
          attrs.erase(kFrontendAttributesAttr);
        }
      }
    }
    attrs.erase(kXlaShardingAttr);
    funcArgAttrs.push_back(attrs.getDictionary(funcOp.getContext()));
  }
  funcOp.setAllArgAttrs(funcArgAttrs);

  // Due to `SdyRoundTripExportShardingsPass` keeping `kXlaShardingAttr`, remove
  // them purely for cleanliness of the module.
  llvm::SmallVector<mlir::DictionaryAttr> newResultAttrs;
  newResultAttrs.reserve(funcOp.getNumResults());
  for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
    mlir::DictionaryAttr dict = funcOp.getResultAttrDict(resNum);
    mlir::NamedAttrList attrs(dict);
    if (attrs.erase(kXlaShardingAttr)) {
      newResultAttrs.push_back(attrs.getDictionary(funcOp.getContext()));
    } else {
      newResultAttrs.push_back(dict);
    }
  }
  funcOp.setAllResultAttrs(newResultAttrs);

  if (funcOp.isExternal()) {
    return;
  }

  llvm::SmallVector<std::pair<CustomCallOp, DictionaryAttr>>
      funcResultShardingOps;

  // Extract the round-tripped shardy attributes from the operations.
  funcOp.front().walk([&](Operation* op) {
    // Preserve `kXlaShardingAttr` on infeed & outfeed ops, as frontend
    // attributes are not always added for them, and we don't propagate
    // shardings for these ops.
    if (!mlir::isa<stablehlo::InfeedOp, stablehlo::OutfeedOp>(op)) {
      op->removeAttr(kXlaShardingAttr);
    }
    DictionaryAttr dictAttr = getFrontendAttrs(op);
    if (!dictAttr) {
      return;
    }

    // Import sharding rules.
    if (auto shardingRuleAttr = parseStringAttr<OpShardingRuleAttr>(
            dictAttr, kShardingRuleRoundTripAttr)) {
      op->setAttr(kShardingRuleAttr, shardingRuleAttr);
      removeFrontendAttribute(op, kShardingRuleRoundTripAttr);
    }

    // `SendOp`, `RecvOp`, and `AfterAllOp` can have a sharding when doing TPU
    // callbacks through JAX. We also handle known custom-calls.
    //
    // For any other op with a `HloSharding::kShardingFrontendAttrName`, we
    // discard it. XLA sometimes creates new instructions, copying over the
    // operand's frontend attrs, which may mean the shapes are wrong when the
    // new instruction is a reshape for example. This does mean we can't fully
    // round-trip b/w HLO and MLIR after SDY propagation.
    if (mlir::isa<stablehlo::SendOp, stablehlo::RecvOp, stablehlo::AfterAllOp>(
            op)) {
      if (auto sharding = parseStringAttr<TensorShardingPerValueAttr>(
              dictAttr,
              xla::ToStringRef(HloSharding::kShardingFrontendAttrName))) {
        op->setAttr(kShardingAttr, sharding);
      }
    } else if (auto customCallOp = mlir::dyn_cast<CustomCallOp>(op)) {
      StringRef targetName = customCallOp.getCallTargetName();
      if (targetName == kFuncResultShardingTargetName) {
        funcResultShardingOps.push_back({customCallOp, dictAttr});
        return;
      }
      if (targetName == kShardingCustomCallTargetName ||
          isPythonCallbackCustomCall(customCallOp)) {
        customCallOp->setAttr(
            kShardingAttr,
            parseStringAttr<TensorShardingPerValueAttr>(
                dictAttr,
                xla::ToStringRef(HloSharding::kShardingFrontendAttrName)));
      }
    }

    removeFrontendAttribute(
        op, xla::ToStringRef(HloSharding::kShardingFrontendAttrName));
  });

  // TODO(b/510714593): Create a shardy utility to modify func result attributes
  // as below but in a more general way and re-use it.
  llvm::SmallVector<DictionaryAttr> funcResultAttrs;
  funcOp.getAllResultAttrs(funcResultAttrs);
  bool anyChanged = false;
  for (auto& [customCallOp, dictAttr] : funcResultShardingOps) {
    anyChanged |= handleFuncResultSharding(customCallOp, funcOp,
                                           funcResultAttrs, dictAttr, rewriter);
  }
  if (anyChanged) {
    funcOp.setAllResultAttrs(funcResultAttrs);
  }
}

// Builds the shardy attributes coming from Shardy previously. This means
// the module was exported from Shardy and we are now round-tripping back.
// This should happen after the meshes were created from the `ModuleOp` attrs
// (see `SdyRoundTripImportShardyAttrsPass`).
void convertShardyAttrs(FuncOp funcOp, IRRewriter& rewriter,
                        bool enableHloShardingV3) {
  if (enableHloShardingV3) {
    convertShardyAttrsWithHloShardingV3(funcOp);
  } else {
    convertShardyAttrsWithoutHloShardingV3(funcOp, rewriter);
  }
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

// Returns true if the call operation is annotated with
// _xla_compute_type="sparseoffload", either directly as an attribute or
// nested inside its frontend attributes.
bool hasSparseOffloadAttribute(CallOp callOp) {
  mlir::StringAttr computeTypeAttr =
      callOp->getAttrOfType<mlir::StringAttr>(xla::kXlaComputeTypeAttr);
  if (!computeTypeAttr) {
    if (DictionaryAttr frontendAttrs =
            getFrontendAttrs(callOp.getOperation())) {
      computeTypeAttr =
          frontendAttrs.getAs<mlir::StringAttr>(xla::kXlaComputeTypeAttr);
    }
  }
  return computeTypeAttr &&
         computeTypeAttr.getValue() == xla::kXlaComputeTypeSparseOffload;
}

// Returns true if the transition from callerSharding to calleeSharding
// contains a replicated-to-partitioned (slicing) transition in any dimension.
// When this is true, we will not insert a propagation barrier for the parameter
// to allow backward propagation of the callee's sharding constraint. This
// avoids forcing a slicing transition inside the callee, which would crash the
// compiler due to unsupported partition-id() instructions on SparseCore.
bool isSlicing(TensorShardingAttr callerSharding,
               TensorShardingAttr calleeSharding) {
  CHECK(calleeSharding);
  if (!callerSharding) {
    // Caller has no sharding (treated as replicated). If the callee has any
    // sharded axes, it is a slicing transition. We allow this layout to
    // propagate backward to the caller so that they end up with matching
    // layouts, avoiding a slicing transition inside the callee.
    return llvm::any_of(calleeSharding.getDimShardings(),
                        [](auto dim) { return !dim.emptyAxes(); });
  }

  auto callerDims = callerSharding.getDimShardings();
  auto calleeDims = calleeSharding.getDimShardings();
  if (callerDims.size() != calleeDims.size()) {
    return false;
  }

  for (int i = 0; i < callerDims.size(); ++i) {
    if (callerDims[i].emptyAxes() && !calleeDims[i].emptyAxes()) {
      return true;
    }
  }
  return false;
}

// Checks if any call site of funcOp passes a value that would cause a slicing
// transition (replicated-to-partitioned) inside the callee.
bool isSlicingAtCallSites(FuncOp funcOp, int argIdx,
                          TensorShardingAttr calleeSharding,
                          ModuleOp moduleOp) {
  auto symbolUses = mlir::SymbolTable::getSymbolUses(funcOp, moduleOp);
  if (!symbolUses) {
    return false;
  }
  return llvm::any_of(
      *symbolUses, [&](const mlir::SymbolTable::SymbolUse& use) {
        if (auto callOp = mlir::dyn_cast<CallOp>(use.getUser())) {
          return isSlicing(getSharding(callOp->getOperand(argIdx)),
                           calleeSharding);
        }
        return false;
      });
}

// Inserts propagation barriers with direction FORWARD on parameters of the
// function that have an internal sharding constraint.
//
// Inserting the barrier is bypassed (not done) if it would force a slicing
// (replicated-to-partitioned) transition inside the callee, which is
// unsupported on SparseCore and crashes the compiler.
//
// When we bypass the barrier, the callee's layout constraint propagates
// backward to the caller argument. If the caller argument has no sharding,
// it inherits the layout constraint directly, ensuring they match and
// avoiding slicing inside the callee.
void insertPropagationBarriers(FuncOp funcOp, ModuleOp moduleOp,
                               IRRewriter& rewriter) {
  if (funcOp.isExternal()) {
    return;
  }

  mlir::Block& entryBlock = funcOp.getBody().front();
  rewriter.setInsertionPointToStart(&entryBlock);

  for (const mlir::BlockArgument& arg : entryBlock.getArguments()) {
    auto findArgShardingUser =
        llvm::find_if(arg.getUses(), [](const mlir::OpOperand& use) {
          if (auto customCallOp =
                  mlir::dyn_cast<CustomCallOp>(use.getOwner())) {
            return customCallOp.getCallTargetName() ==
                   kShardingCustomCallTargetName;
          }
          return false;
        });

    if (findArgShardingUser != arg.getUses().end()) {
      TensorShardingAttr calleeSharding =
          getSharding(mlir::cast<CustomCallOp>(findArgShardingUser->getOwner())
                          .getResult(0));
      bool shouldInsertBarrier = true;
      if (calleeSharding) {
        if (isSlicingAtCallSites(funcOp, arg.getArgNumber(), calleeSharding,
                                 moduleOp)) {
          shouldInsertBarrier = false;
        }
      }

      if (shouldInsertBarrier) {
        auto barrierOp = PropagationBarrierOp::create(
            rewriter, funcOp.getLoc(), arg, PropagationDirection::FORWARD);
        rewriter.replaceAllUsesExcept(arg, barrierOp.getResult(), barrierOp);
      }
    }
  }
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
          meshesAttr.has_value() ? meshesAttr->getValue()
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

    llvm::DenseSet<FuncOp> modifiedCallees;
    moduleOp.walk([&](CallOp callOp) {
      if (hasSparseOffloadAttribute(callOp)) {
        const FuncOp callee = symbolTable.lookup<FuncOp>(callOp.getCallee());
        if (callee && modifiedCallees.insert(callee).second) {
          insertPropagationBarriers(callee, moduleOp, rewriter);
        }
      }
    });
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
