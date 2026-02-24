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

#include "xla/service/spmd/shardy/stablehlo_round_trip/shard_map_export.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // for CopyOp
#include "xla/service/spmd/shardy/constants.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ArrayRef;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::Operation;
using ::mlir::OperationPass;
using ::mlir::SmallVector;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::SymbolTableCollection;
using ::mlir::SymbolUserMap;
using ::mlir::Value;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::mhlo::CopyOp;
using ::mlir::stablehlo::CustomCallOp;

namespace sdy = ::mlir::sdy;

using sdy::kShardingAttr;
using sdy::ManualAxesAttr;
using sdy::ManualComputationOp;
using sdy::MeshAttr;
using sdy::NamedComputationOp;
using sdy::SdyDialect;
using sdy::TensorShardingAttr;
using sdy::TensorShardingPerValueAttr;

// Mapping from ManualComputationOp to all manual axes it's nested in.
using ManualComputationToParentManualAxes =
    llvm::SmallDenseMap<ManualComputationOp, SmallVector<StringAttr>>;

// Given an ManualComputationOp `op`, `op.getManualAxes()` is the local manual
// axes. `parent` is the manual axes of its parent ManualComputationOp,
// recursively. `region` is the concatenation of `parent` and
// `op.getManualAxes()`.
struct ManualAxesHierarchy {
  ArrayRef<StringAttr> parent;
  SmallVector<StringAttr> region;
};

ManualAxesHierarchy getManualAxesHierarchy(
    ManualComputationOp op,
    const ManualComputationToParentManualAxes& parentManualCompAxes) {
  ManualAxesHierarchy hierarchy;

  if (auto parentManualAxes = parentManualCompAxes.find(op);
      parentManualAxes != parentManualCompAxes.end()) {
    hierarchy.parent = parentManualAxes->getSecond();
  }

  hierarchy.region =
      SmallVector<StringAttr>(hierarchy.parent.begin(), hierarchy.parent.end());
  hierarchy.region.append(op.getManualAxes().begin(), op.getManualAxes().end());
  return hierarchy;
}

// Returns the first sharding of `op`. If there are no in/out shardings, returns
// nullptr.
TensorShardingAttr getFirstSharding(ManualComputationOp op) {
  auto inOutShardings = llvm::concat<const TensorShardingAttr>(
      op.getInShardings().getShardings(), op.getOutShardings().getShardings());
  if (inOutShardings.begin() == inOutShardings.end()) {
    return nullptr;
  }
  return *inOutShardings.begin();
}

void setFullyClosedShardingsIfMissing(Operation* op, StringRef meshName) {
  MLIRContext* context = op->getContext();

  if (NamedComputationOp namedComputationOp =
          mlir::dyn_cast<NamedComputationOp>(op)) {
    if (!namedComputationOp.getInShardings().has_value()) {
      namedComputationOp.setInShardingsAttr(
          TensorShardingPerValueAttr::getFullyClosed(
              context, op->getOperandTypes(), meshName));
    }
    if (!namedComputationOp.getOutShardings().has_value()) {
      namedComputationOp.setOutShardingsAttr(
          TensorShardingPerValueAttr::getFullyClosed(
              context, op->getResultTypes(), meshName));
    }
    return;
  }

  if (!op->hasAttrOfType<TensorShardingPerValueAttr>(kShardingAttr)) {
    SmallVector<TensorShardingAttr> shardings =
        sdy::getFullyClosedShardings(context, op->getResultTypes(), meshName);
    if (shardings.empty() && !op->hasTrait<mlir::OpTrait::IsTerminator>()) {
      shardings = {TensorShardingAttr::getFullyReplicated(
          context, /*rank=*/0, meshName, /*isClosed=*/true)};
    }
    sdy::setShardings(op, shardings);
  }
}

// Sets the manual axes of all operations in `op`'s body.
void setManualAxesForOpsInBody(
    ManualComputationOp op,
    const ManualComputationToParentManualAxes& parentManualCompAxes,
    const mlir::SymbolTable& symbolTable) {
  TensorShardingAttr sharding = getFirstSharding(op);
  if (!sharding) {
    // If there are no in/out shardings, op.getManualAxes() must be empty. We do
    // not need to set the manual axes of the body.
    return;
  }

  ManualAxesHierarchy manualAxes =
      getManualAxesHierarchy(op, parentManualCompAxes);
  if (manualAxes.region.empty()) {
    return;
  }

  MLIRContext* context = op.getContext();
  StringRef meshName = sharding.getMeshName();
  ManualAxesAttr manualAxesAttr =
      ManualAxesAttr::get(context, manualAxes.region);

  // Set the manual axes of all operations in the body.
  op.getBody().front().walk<mlir::WalkOrder::PreOrder>(
      [&](Operation* opInBody) {
        if (mlir::isa<ManualComputationOp>(opInBody)) {
          // Skip `ManualComputationOp`s and their nested operations, they will
          // be handled separately.
          return mlir::WalkResult::skip();
        }

        // TODO(b/415378067). Polish how we handle shardings with different
        // meshes.
        bool hasOtherMesh = false;
        for (TensorShardingAttr opInBodySharding :
             sdy::getShardings(opInBody)) {
          if (opInBodySharding.getMeshName() != meshName) {
            hasOtherMesh = true;
            MeshAttr otherMesh = opInBodySharding.getMesh(opInBody);
            CHECK(otherMesh.getAxes().empty() || otherMesh.isMaximal());
          }
        }
        if (hasOtherMesh) {
          // Must be fully manual.
          CHECK(manualAxes.region.size() ==
                sharding.getMesh(symbolTable).getAxes().size());
          opInBody->removeAttr(kShardingAttr);
        }

        setFullyClosedShardingsIfMissing(opInBody, meshName);
        opInBody->setAttr(kManualAxes, manualAxesAttr);
        return mlir::WalkResult::advance();
      });
}

void setNonEmptyManualAxes(Operation* op, ManualAxesAttr manualAxesAttr) {
  if (!manualAxesAttr.empty()) {
    op->setAttr(kManualAxes, manualAxesAttr);
  }
}

void setBlockArgManualAxes(FuncOp funcOp, mlir::BlockArgument blockArg,
                           ManualAxesAttr manualAxesAttr) {
  if (!manualAxesAttr.empty()) {
    funcOp.setArgAttr(blockArg.getArgNumber(), kManualAxes, manualAxesAttr);
  }
}

void setFuncResultManualAxes(FuncOp funcOp, int64_t resultIndex,
                             ManualAxesAttr manualAxesAttr) {
  if (!manualAxesAttr.empty()) {
    funcOp.setResultAttr(resultIndex, kManualAxes, manualAxesAttr);
  }
}

// Creates a sharding constraint op. If `createHloShardingConstraints` is true,
// creates a `stablehlo.custom_call` op with `call_target_name` equal to
// "Sharding", otherwise creates a `mhlo.copy` op.
Operation* createShardingConstraint(mlir::IRRewriter& rewriter,
                                    mlir::Location loc, Value value,
                                    bool createHloShardingConstraints) {
  if (createHloShardingConstraints) {
    auto customCallOp =
        CustomCallOp::create(rewriter, loc, value.getType(), value);
    customCallOp.setCallTargetName(kShardingCustomCallTargetName);
    return customCallOp;
  }
  return CopyOp::create(rewriter, loc, value);
}

// Converts `op` to the pattern that XLA recognizes.
//
// The pattern is:
// 1. Sharding-constraint for each operand.
// 2. CustomCall @SPMDFullToShardShape for each operand.
// 3. Inline the body.
// 4. Sharding-constraint for each result.
// 5. CustomCall @SPMDShardToFullShape for each result.
//
// The shardings and manual axes of the sharding-constraints and CustomCall ops
// are set based on the in/out shardings of `op`.
//
// If `createHloShardingConstraints` is true, the sharding-constraints are
// @Sharding custom calls, otherwise they MHLO copy ops.
void convertManualComputationOp(
    ManualComputationOp op,
    const ManualComputationToParentManualAxes& parentManualCompAxes,
    mlir::SymbolTable& symbolTable, bool createHloShardingConstraints) {
  MLIRContext* context = op.getContext();
  mlir::IRRewriter rewriter(op);
  TensorShardingAttr sharding = getFirstSharding(op);
  if (!sharding) {
    // If there are no in/out shardings, op.getManualAxes() must be empty. We
    // directly inline the body and erase the op.
    rewriter.eraseOp(sdy::getBodyTerminator(op));
    rewriter.inlineBlockBefore(&op.getBody().front(), op, op.getOperands());
    rewriter.eraseOp(op);
    return;
  }

  // For a ManualComputationOp, all in/out shardings, shardings in the body,
  // and manual axes must refer to the same mesh.
  MeshAttr mesh = sharding.getMesh(symbolTable);
  CHECK(mesh);

  // The axes that are manual inside `op`'s region.
  ManualAxesHierarchy manualAxes =
      getManualAxesHierarchy(op, parentManualCompAxes);
  ManualAxesAttr parentManualAxesAttr =
      ManualAxesAttr::get(context, manualAxes.parent);
  ManualAxesAttr regionManualAxesAttr =
      ManualAxesAttr::get(context, manualAxes.region);

  mlir::Location loc = op.getLoc();
  // Add sharding-constraint and custom_call @SPMDFullToShardShape for each
  // operand.
  SmallVector<Value> fullToShardResults;
  for (auto [globalOperand, localArgumentType, inSharding] :
       llvm::zip_equal(op.getOperands(), op.getBody().getArgumentTypes(),
                       op.getInShardings().getShardings())) {
    if (!mlir::isa<mlir::ShapedType>(localArgumentType)) {
      fullToShardResults.push_back(globalOperand);
      continue;
    }
    Operation* shardingConstraint = createShardingConstraint(
        rewriter, loc, globalOperand, createHloShardingConstraints);
    sdy::setShardings(shardingConstraint, inSharding);
    setNonEmptyManualAxes(shardingConstraint, parentManualAxesAttr);

    auto fullToShard = CustomCallOp::create(rewriter, loc, localArgumentType,
                                            shardingConstraint->getResult(0));
    fullToShard.setCallTargetName(kSPMDFullToShardShapeCallTargetName);
    sdy::setShardings(fullToShard,
                      eraseManualAxes(inSharding, manualAxes.region));
    setNonEmptyManualAxes(fullToShard, regionManualAxesAttr);

    fullToShardResults.push_back(fullToShard.getResult(0));
  }

  rewriter.setInsertionPointToEnd(
      &op->getParentOfType<ModuleOp>().getRegion().front());
  Operation* terminator = sdy::getBodyTerminator(op);
  auto funcOp =
      FuncOp::create(rewriter, loc, kInlineableManualComputationFuncName,
                     rewriter.getFunctionType(op.getBody().getArgumentTypes(),
                                              terminator->getOperandTypes()));
  mlir::StringAttr funcName = symbolTable.insert(funcOp);

  rewriter.setInsertionPointAfter(op);
  auto callOp = CallOp::create(rewriter, loc, terminator->getOperandTypes(),
                               funcName, fullToShardResults);
  setNonEmptyManualAxes(callOp, regionManualAxesAttr);
  sdy::inlineRegionAndConvertTerminatorOp<mlir::func::ReturnOp>(
      op.getBody(), funcOp.getBody());
  for (auto [blockArg, sharding] : llvm::zip_equal(
           funcOp.getArguments(), op.getInShardings().getShardings())) {
    if (sharding) {
      setSharding(blockArg, eraseManualAxes(sharding, manualAxes.region));
      setBlockArgManualAxes(funcOp, blockArg, regionManualAxesAttr);
    }
  }
  for (auto [i, sharding] :
       llvm::enumerate(op.getOutShardings().getShardings())) {
    if (sharding) {
      sdy::setFuncResultSharding(funcOp, i,
                                 eraseManualAxes(sharding, manualAxes.region));
      setFuncResultManualAxes(funcOp, i, regionManualAxesAttr);
    }
  }

  SmallVector<TensorShardingAttr> erasedManualAxisOutShardings;
  erasedManualAxisOutShardings.reserve(op.getNumResults());
  // Add custom_call @SPMDShardToFullShape and sharding-constraint for each
  // operand of terminator.
  for (auto [localResult, oldResult, outSharding] :
       llvm::zip_equal(callOp->getResults(), op.getResults(),
                       op.getOutShardings().getShardings())) {
    erasedManualAxisOutShardings.push_back(
        eraseManualAxes(outSharding, manualAxes.region));
    if (!mlir::isa<mlir::ShapedType>(oldResult.getType())) {
      oldResult.replaceAllUsesWith(localResult);
      continue;
    }
    Operation* shardingConstraint = createShardingConstraint(
        rewriter, loc, localResult, createHloShardingConstraints);
    sdy::setShardings(shardingConstraint, erasedManualAxisOutShardings.back());
    setNonEmptyManualAxes(shardingConstraint, regionManualAxesAttr);

    auto shardToFull = CustomCallOp::create(rewriter, loc, oldResult.getType(),
                                            shardingConstraint->getResult(0));
    shardToFull.setCallTargetName(kSPMDShardToFullShapeCallTargetName);
    sdy::setShardings(shardToFull, outSharding);
    setNonEmptyManualAxes(shardToFull, parentManualAxesAttr);

    oldResult.replaceAllUsesWith(shardToFull.getResult(0));
  }

  setShardings(callOp, erasedManualAxisOutShardings);
  rewriter.eraseOp(op);
}

class ShardMapExportPass
    : public mlir::PassWrapper<ShardMapExportPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShardMapExportPass)

  explicit ShardMapExportPass(bool createHloShardingConstraints) {
    this->createHloShardingConstraints = createHloShardingConstraints;
  }

  ShardMapExportPass() = default;

  explicit ShardMapExportPass(const ShardMapExportPass& other) {
    this->createHloShardingConstraints = other.createHloShardingConstraints;
  }

 private:
  void runOnOperation() final {
    ManualComputationToParentManualAxes parentManualCompAxes;
    ModuleOp module = getOperation();
    SymbolTableCollection symbolTableCollection;
    SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(module);
    // Need to go from parent into child/nested `ManualComputationOp`s in order
    // to make sure we have the latest `parentManualCompAxes`, so do a pre-order
    // walk.
    module->walk<mlir::WalkOrder::PreOrder>([&](ManualComputationOp op) {
      if (auto parentOp = op->getParentOfType<ManualComputationOp>()) {
        SmallVector<StringAttr>& parentAxes = parentManualCompAxes[op];
        parentAxes = parentManualCompAxes[parentOp];
        parentAxes.insert(parentAxes.end(), parentOp.getManualAxes().begin(),
                          parentOp.getManualAxes().end());
      }
      setManualAxesForOpsInBody(op, parentManualCompAxes, symbolTable);
    });

    // Need to do a separate post order walk to inline the
    // `ManualComputationOp`s. Inlining causes the walk order to change, which
    // invalidates the existing walk if it were a pre-order walk like the one
    // above.
    module->walk([&](ManualComputationOp op) {
      convertManualComputationOp(op, parentManualCompAxes, symbolTable,
                                 createHloShardingConstraints);
    });

    // Drop uncalled inlineable manual computation funcs.
    // TODO(enver): Drop generically, not just inlined manual computation funcs.
    llvm::SmallVector<FuncOp> uncalledInlineableManualComputationFuncs;
    SymbolUserMap symbolUserMap(symbolTableCollection, module);
    for (FuncOp funcOp : module.getOps<FuncOp>()) {
      if (StringRef funcSymName = funcOp.getName();
          funcSymName.contains(kInlineableManualComputationFuncName) &&
          symbolUserMap.useEmpty(funcOp)) {
        uncalledInlineableManualComputationFuncs.push_back(funcOp);
      }
    }
    // TODO(enver): Erase directly without collecting on a vector.
    for (FuncOp funcOp : uncalledInlineableManualComputationFuncs) {
      symbolTable.erase(funcOp);
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-stablehlo-round-trip-shard-map-export";
  }

  StringRef getDescription() const override {
    return "Replaces sdy::ManualComputationOp with the pattern that XLA "
           "recognizes.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<SdyDialect, mlir::mhlo::MhloDialect,
                    mlir::stablehlo::StablehloDialect>();
  }

  Option<bool> createHloShardingConstraints{
      *this, "create-hlo-sharding-constraints",
      llvm::cl::desc(
          "Whether to create @Sharding custom calls or MHLO copy ops."),
      llvm::cl::init(false)};
};

}  // namespace

std::unique_ptr<mlir::Pass> createStablehloRoundTripShardMapExportPass(
    bool createHloShardingConstraints) {
  return std::make_unique<ShardMapExportPass>(createHloShardingConstraints);
}

void registerStablehloRoundTripShardMapExportPass() {
  mlir::registerPass(std::make_unique<ShardMapExportPass>);
}

}  // namespace sdy
}  // namespace xla
