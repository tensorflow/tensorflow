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
#include <numeric>
#include <tuple>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
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
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
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
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
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
using ::mlir::Value;
using ::mlir::mhlo::CopyOp;
using ::mlir::stablehlo::CustomCallOp;

namespace sdy = ::mlir::sdy;
using sdy::AxisRefAttr;
using sdy::DimensionShardingAttr;
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

// Given the `sharding` and `manualAxes`, remove any auto axes that would cause
// padding.
TensorShardingAttr removeAutoAxesToAvoidPadding(TensorShardingAttr sharding,
                                                ArrayRef<StringAttr> manualAxes,
                                                mlir::Type type,
                                                MeshAttr mesh) {
  MLIRContext* context = sharding.getContext();

  SmallVector<DimensionShardingAttr> newDimShardings;
  newDimShardings.reserve(sharding.getRank());
  for (auto [dimSize, dimSharding] :
       llvm::zip_equal(mlir::cast<mlir::RankedTensorType>(type).getShape(),
                       sharding.getDimShardings())) {
    ArrayRef<AxisRefAttr> dimAxes = dimSharding.getAxes();
    SmallVector<AxisRefAttr> newDimAxes;
    ArrayRef<AxisRefAttr>::const_iterator freeAxisIt =
        sdy::getFirstFreeAxisIter(dimAxes, manualAxes);

    // Keep all manual axes.
    int64_t dimAxesSize = 1;
    for (const auto* it = dimAxes.begin(); it != freeAxisIt; ++it) {
      dimAxesSize *= it->getSize(mesh);
      newDimAxes.push_back(*it);
    }

    // The manual axes cannot introduce padding. The dimension size must be
    // divisible by the corresponding manual axes size.
    assert(dimSize % dimAxesSize == 0);
    int64_t capacity = dimSize / dimAxesSize;

    // Keep all free axes that divide the dimension size.
    for (ArrayRef<AxisRefAttr>::const_iterator it = freeAxisIt;
         it != dimAxes.end() && capacity > 1; ++it) {
      int64_t gcd = std::gcd(capacity, it->getSize(mesh));
      if (gcd == it->getSize(mesh)) {
        newDimAxes.push_back(*it);
        capacity /= gcd;
      } else {
        if (gcd != 1) {
          newDimAxes.push_back(AxisRefAttr::get(context, it->getName(),
                                                it->getSubAxisPreSize(), gcd));
        }
        break;
      }
    }

    newDimShardings.push_back(DimensionShardingAttr::get(
        context, newDimAxes, dimSharding.getIsClosed(),
        dimSharding.getPriority()));
  }
  return TensorShardingAttr::get(context, sharding.getMeshOrRef(),
                                 newDimShardings, sharding.getReplicatedAxes());
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
    sdy::setShardings(op, sdy::getFullyClosedShardings(
                              context, op->getResultTypes(), meshName));
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
             mlir::sdy::getShardings(opInBody)) {
          if (opInBodySharding.getMeshName() != meshName) {
            hasOtherMesh = true;
            CHECK(opInBodySharding.getMesh(opInBody).empty());
          }
        }
        if (hasOtherMesh) {
          // must be fully manual
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
  if (!manualAxesAttr.getValue().empty()) {
    op->setAttr(kManualAxes, manualAxesAttr);
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
// The shardings and manual axes of the Copy and CustomCall ops are set based on
// the in/out shardings of `op`.
void convertManualComputationOp(
    ManualComputationOp op,
    const ManualComputationToParentManualAxes& parentManualCompAxes,
    const mlir::SymbolTable& symbolTable) {
  MLIRContext* context = op.getContext();
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
  ManualAxesHierarchy manualAxes =
      getManualAxesHierarchy(op, parentManualCompAxes);
  ManualAxesAttr parentManualAxesAttr =
      ManualAxesAttr::get(context, manualAxes.parent);
  ManualAxesAttr regionManualAxesAttr =
      ManualAxesAttr::get(context, manualAxes.region);

  mlir::Location loc = op.getLoc();
  // Add copy and custom_call @SPMDFullToShardShape for each operand. The
  // copy corresponds to custom_call @Sharding before sharding propagation.
  SmallVector<Value> fullToShardResults;
  for (auto [globalOperand, localArgumentType, inSharding] :
       llvm::zip_equal(op.getOperands(), op.getBody().getArgumentTypes(),
                       op.getInShardings().getShardings())) {
    if (!isa<mlir::ShapedType>(localArgumentType)) {
      fullToShardResults.push_back(globalOperand);
      continue;
    }
    TensorShardingAttr newSharding = removeAutoAxesToAvoidPadding(
        inSharding, manualAxes.region, globalOperand.getType(), mesh);
    auto copy = rewriter.create<CopyOp>(loc, globalOperand);
    sdy::setShardings(copy, newSharding);
    setNonEmptyManualAxes(copy, parentManualAxesAttr);

    auto fullToShard =
        rewriter.create<CustomCallOp>(loc, localArgumentType, copy.getResult());
    fullToShard.setCallTargetName(kSPMDFullToShardShapeCallTargetName);
    sdy::setShardings(fullToShard,
                      eraseManualAxes(newSharding, manualAxes.region));
    setNonEmptyManualAxes(fullToShard, regionManualAxesAttr);

    fullToShardResults.push_back(fullToShard.getResult(0));
  }

  Operation* terminator = getBodyTerminator(op);
  // Add custom_call @SPMDShardToFullShape and copy for each operand of
  // terminator.
  rewriter.setInsertionPointAfter(op);

  for (auto [terminatorOperand, opResult, outSharding] :
       llvm::zip_equal(terminator->getOpOperands(), op.getResults(),
                       op.getOutShardings().getShardings())) {
    if (!isa<mlir::ShapedType>(opResult.getType())) {
      opResult.replaceAllUsesWith(terminatorOperand.get());
      continue;
    }
    TensorShardingAttr newSharding = removeAutoAxesToAvoidPadding(
        outSharding, manualAxes.region, opResult.getType(), mesh);

    auto copy = rewriter.create<CopyOp>(loc, terminatorOperand.get());
    sdy::setShardings(copy, eraseManualAxes(newSharding, manualAxes.region));
    setNonEmptyManualAxes(copy, regionManualAxesAttr);

    auto shardToFull = rewriter.create<CustomCallOp>(loc, opResult.getType(),
                                                     copy.getResult());
    shardToFull.setCallTargetName(kSPMDShardToFullShapeCallTargetName);
    sdy::setShardings(shardToFull, newSharding);
    setNonEmptyManualAxes(shardToFull, parentManualAxesAttr);

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
      setManualAxesForOpsInBody(op, parentManualCompAxes, symbolTable);
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
};

}  // namespace

std::unique_ptr<mlir::Pass> createStablehloRoundTripShardMapExportPass() {
  return std::make_unique<ShardMapExportPass>();
}

void registerStablehloRoundTripShardMapExportPass() {
  mlir::registerPass(createStablehloRoundTripShardMapExportPass);
}

}  // namespace sdy
}  // namespace xla
