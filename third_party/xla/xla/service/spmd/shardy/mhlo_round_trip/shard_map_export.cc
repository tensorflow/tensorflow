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

#include "xla/service/spmd/shardy/mhlo_round_trip/shard_map_export.h"

#include <cassert>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/mhlo_round_trip/export_shardings.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::Operation;
using ::mlir::OperationPass;
using ::mlir::SmallVector;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::Value;
using ::mlir::mhlo::CopyOp;
using ::mlir::stablehlo::CustomCallOp;

namespace sdy = ::mlir::sdy;
using sdy::kShardingAttr;
using sdy::ManualComputationOp;
using sdy::MeshAttr;
using sdy::SdyDialect;
using sdy::TensorShardingAttr;
using sdy::TensorShardingPerValueAttr;

// Mapping from ManualComputationOp to all manual axes it's nested in.
using ManualComputationToParentManualAxes =
    llvm::SmallDenseMap<ManualComputationOp, SmallVector<StringAttr>>;

// Populates `regionManualAxes` with the manual axes of `op`. If `op` is nested
// in another manual computation, also returns the manual axes of the parent
// computation.
mlir::ArrayRef<StringAttr> populateRegionManualAxes(
    SmallVector<StringAttr>& regionManualAxes, ManualComputationOp op,
    const ManualComputationToParentManualAxes& parentManualCompAxes) {
  regionManualAxes = SmallVector<StringAttr>(op.getManualAxes().begin(),
                                             op.getManualAxes().end());
  mlir::ArrayRef<StringAttr> parentManualAxesRef;
  if (auto parentManualAxes = parentManualCompAxes.find(op);
      parentManualAxes != parentManualCompAxes.end()) {
    parentManualAxesRef = parentManualAxes->getSecond();
    regionManualAxes.append(parentManualAxes->getSecond().begin(),
                            parentManualAxes->getSecond().end());
  }
  return parentManualAxesRef;
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

// Converts the shardings of all operations in `op`'s body to MHLO shardings.
void convertShardingsToMhloShardings(
    ManualComputationOp op,
    const ManualComputationToParentManualAxes& parentManualCompAxes,
    const mlir::SymbolTable& symbolTable) {
  TensorShardingAttr sharding = getFirstSharding(op);
  if (!sharding) {
    // If there are no in/out shardings, op.getManualAxes() must be empty. No
    // sharding conversion is needed.
    return;
  }

  // For a ManualComputationOp, all in/out shardings, shardings in the body,
  // and manual axes must refer to the same mesh.
  StringRef meshName = sharding.getMeshName();
  MeshAttr mesh = sharding.getMesh(symbolTable);
  CHECK(mesh);

  // The axes that are manual inside `op`'s region.
  SmallVector<StringAttr> regionManualAxes;
  (void)populateRegionManualAxes(regionManualAxes, op, parentManualCompAxes);
  MLIRContext* context = op.getContext();
  std::function<StringAttr(const HloSharding&)> getStringAttr =
      [&](const HloSharding& hloSharding) {
        return StringAttr::get(context, hloSharding.ToString());
      };

  if (mesh.getAxes().size() == regionManualAxes.size()) {
    // All operations in the body have fully manual sharding.
    StringAttr fullyManualSharding = getStringAttr(HloSharding::Manual());
    op.getBody().front().walk<mlir::WalkOrder::PreOrder>(
        [&](Operation* opInBody) {
          if (mlir::isa<ManualComputationOp>(opInBody)) {
            // Skip `ManualComputationOp`s, they will be converted separately.
            return mlir::WalkResult::skip();
          }
          opInBody->setAttr(kXlaShardingAttr, fullyManualSharding);
          // Remove the possible fully replicated sdy.sharding attribute.
          opInBody->removeAttr(kShardingAttr);
          return mlir::WalkResult::advance();
        });
  } else {
    auto getMeshAttr = [&](TensorShardingAttr) { return mesh; };
    // All operations in the body must be sharded or replicated along free
    // axes. If an operation does not have sharding annotation, it is fully
    // replicated along free axes.
    op.getBody().front().walk<mlir::WalkOrder::PreOrder>(
        [&](Operation* opInBody) {
          if (mlir::isa<ManualComputationOp>(opInBody)) {
            return mlir::WalkResult::skip();
          }
          TensorShardingPerValueAttr shardingPerValue =
              opInBody->getAttrOfType<TensorShardingPerValueAttr>(
                  kShardingAttr);
          if (!shardingPerValue) {
            shardingPerValue = TensorShardingPerValueAttr::getFullyOpen(
                context, opInBody->getResultTypes(), meshName);
          }
          opInBody->setAttr(kXlaShardingAttr,
                            convertToHloShardingAttr(
                                opInBody, shardingPerValue.getShardings(),
                                getMeshAttr, getStringAttr, regionManualAxes));
          opInBody->removeAttr(kShardingAttr);
          return mlir::WalkResult::advance();
        });
  }
}

// Converts `op` to the pattern that XLA recognizes.
//
// The pattern is:
// 1. Copy for each operand.
// 2. CustomCall @SPMDFullToShardShape for each operand.
// 3. Inline the body.
// 4. Copy for each result.
// 5. CustomCall @SPMDShardToFullShape for each result.
//
// The shardings of the Copy and CustomCall ops are set based on the in/out
// shardings of `op`.
void convertManualComputationOp(
    ManualComputationOp op,
    const ManualComputationToParentManualAxes& parentManualCompAxes,
    const mlir::SymbolTable& symbolTable) {
  mlir::IRRewriter rewriter(op);
  TensorShardingAttr sharding = getFirstSharding(op);
  if (!sharding) {
    // If there are no in/out shardings, op.getManualAxes() must be empty. We
    // directly inline the body and erase the op.
    rewriter.eraseOp(getBodyTerminator(op));
    rewriter.inlineBlockBefore(&op.getBody().front(), op, op.getOperands());
    rewriter.eraseOp(op);
    return;
  }

  // For a ManualComputationOp, all in/out shardings, shardings in the body,
  // and manual axes must refer to the same mesh.
  MeshAttr mesh = sharding.getMesh(symbolTable);
  CHECK(mesh);

  // The axes that are manual inside `op`'s region.
  SmallVector<StringAttr> regionManualAxes;
  mlir::ArrayRef<StringAttr> parentManualAxes =
      populateRegionManualAxes(regionManualAxes, op, parentManualCompAxes);
  std::function<StringAttr(const HloSharding&)> getStringAttr =
      [&](const HloSharding& hloSharding) {
        return rewriter.getStringAttr(hloSharding.ToString());
      };

  StringAttr fullyManualSharding = getStringAttr(HloSharding::Manual());
  auto createAttributes =
      [&](StringRef callTargetName) -> SmallVector<NamedAttribute, 2> {
    return {rewriter.getNamedAttr("call_target_name",
                                  rewriter.getStringAttr(callTargetName)),
            rewriter.getNamedAttr(kXlaShardingAttr, fullyManualSharding)};
  };
  SmallVector<NamedAttribute, 2> fullToShardAttributes =
      createAttributes(kSPMDFullToShardShapeCallTargetName);
  SmallVector<NamedAttribute, 2> shardToFullAttributes =
      createAttributes(kSPMDShardToFullShapeCallTargetName);

  bool fullyManual = mesh.getAxes().size() == regionManualAxes.size();
  mlir::Location loc = op.getLoc();
  auto getMeshAttr = [&](TensorShardingAttr) { return mesh; };
  // Add copy and custom_call @SPMDFullToShardShape for each operand. The
  // copy corresponds to custom_call @Sharding before sharding propagation.
  SmallVector<Value> fullToShardResults;
  for (auto [operand_index, args] : llvm::enumerate(
           llvm::zip_equal(op.getOperands(), op.getBody().getArgumentTypes(),
                           op.getInShardings().getShardings()))) {
    auto [globalOperand, localArgumentType, inSharding] = args;
    auto copy = rewriter.create<CopyOp>(loc, globalOperand);
    copy->setAttr(kXlaShardingAttr,
                  getStringAttr(convertToHloSharding(inSharding, getMeshAttr,
                                                     parentManualAxes)));
    fullToShardAttributes.back() = rewriter.getNamedAttr(
        kXlaShardingAttr,
        fullyManual ? fullyManualSharding
                    : getStringAttr(convertToHloSharding(
                          op.getInShardingWithoutManualAxes(operand_index),
                          getMeshAttr, regionManualAxes)));
    auto fullToShard = rewriter.create<CustomCallOp>(
        loc, localArgumentType, copy.getResult(), fullToShardAttributes);
    fullToShardResults.push_back(fullToShard.getResult(0));
  }
  Operation* terminator = getBodyTerminator(op);
  // Add custom_call @SPMDShardToFullShape and copy for each operand of
  // terminator.
  rewriter.setInsertionPointAfter(op);
  for (auto [terminatorOperand, opResult, outSharding] :
       llvm::zip_equal(terminator->getOpOperands(), op.getResults(),
                       op.getOutShardings().getShardings())) {
    auto copy = rewriter.create<CopyOp>(loc, terminatorOperand.get());
    copy->setAttr(kXlaShardingAttr,
                  fullyManual ? fullyManualSharding
                              : getStringAttr(convertToHloSharding(
                                    op.getOutShardingWithoutManualAxes(
                                        terminatorOperand.getOperandNumber()),
                                    getMeshAttr, regionManualAxes)));
    shardToFullAttributes.back() = rewriter.getNamedAttr(
        kXlaShardingAttr, getStringAttr(convertToHloSharding(
                              outSharding, getMeshAttr, parentManualAxes)));
    auto shardToFull = rewriter.create<CustomCallOp>(
        loc, opResult.getType(), copy.getResult(), shardToFullAttributes);
    opResult.replaceAllUsesWith(shardToFull.getResult(0));
  }
  rewriter.inlineBlockBefore(&op.getBody().front(), op, fullToShardResults);
  rewriter.eraseOp(terminator);
  rewriter.eraseOp(op);
}

class ShardMapExportPass
    : public mlir::PassWrapper<ShardMapExportPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShardMapExportPass)

 private:
  void runOnOperation() final {
    ManualComputationToParentManualAxes parentManualCompAxes;
    ModuleOp module = getOperation();
    mlir::SymbolTable symbolTable(module);
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
      convertShardingsToMhloShardings(op, parentManualCompAxes, symbolTable);
    });

    // Need to do a separate post order walk to inline the
    // `ManualComputationOp`s. Inlining causes the walk order to change, which
    // invalidates the existing walk if it were a pre-order walk like the one
    // above.
    module->walk([&](ManualComputationOp op) {
      convertManualComputationOp(op, parentManualCompAxes, symbolTable);
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-mhlo-round-trip-shard-map-export";
  }

  StringRef getDescription() const override {
    return "Replaces sdy::ManualComputationOp with the pattern that XLA "
           "recognizes.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<SdyDialect, mlir::mhlo::MhloDialect>();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createMhloRoundTripShardMapExportPass() {
  return std::make_unique<ShardMapExportPass>();
}

void registerMhloRoundTripShardMapExportPass() {
  mlir::registerPass(createMhloRoundTripShardMapExportPass);
}

}  // namespace sdy
}  // namespace xla
