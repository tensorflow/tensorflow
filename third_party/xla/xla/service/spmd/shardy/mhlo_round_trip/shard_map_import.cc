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

#include "xla/service/spmd/shardy/mhlo_round_trip/shard_map_import.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ModuleOp;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::OperationPass;
using ::mlir::SmallVector;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::Value;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::stablehlo::CustomCallOp;

namespace sdy = ::mlir::sdy;
using sdy::AxisRefAttr;
using sdy::DimensionShardingAttr;
using sdy::kShardingAttr;
using sdy::ManualComputationOp;
using sdy::MeshAttr;
using sdy::ReturnOp;
using sdy::SdyDialect;
using sdy::TensorShardingAttr;
using sdy::TensorShardingPerValueAttr;

// A pair of custom calls. `shardingOp` has target name "Sharding".
// `shapeTransformOp` has target name "SPMDFullToShardShape" or
// "SPMDShardToFullShape". `shardingOp` has exactly one user, which is
// shapeTransformOp. `shapeTransformOp` has exactly one operand, which is
// `shardingOp`.
//
// Both `shardingOp` and `shapeTransformOp` will be nullptr for unused results
// of the shard map `func.call`.
struct ShardMapCustomCallPair {
  CustomCallOp shardingOp;
  CustomCallOp shapeTransformOp;
};

// The arguments and results of a shard map function call.
struct ShardMapArgumentsResults {
  SmallVector<ShardMapCustomCallPair> argumentOps;
  SmallVector<ShardMapCustomCallPair> resultOps;
};

struct ShardMapOps {
  ShardMapArgumentsResults shardMapArgumentsResults;
  CallOp callOp;
  ReturnOp returnOp;
};

// Checks whether the custom call op from the shard_map has:
// 1. One result
// 2. At least one use of the result
// 3. One operand
// 4. Right target name
bool isShardMapCustomCall(CustomCallOp op, StringRef targetName) {
  return op->getNumResults() == 1 && !op->getResult(0).use_empty() &&
         op->getNumOperands() == 1 && op.getCallTargetName() == targetName;
}

// Inserts all used axes in `tensorSharding` into `axesSet`. We assume that
// `tensorSharding` does not use sub axes.
void insertAxisNamesFromSharding(mlir::MLIRContext* context,
                                 TensorShardingAttr tensorSharding,
                                 llvm::SmallDenseSet<StringAttr>& axesSet) {
  auto checkFullAxisAndInsertToResult = [&](AxisRefAttr axis) {
    CHECK(!axis.getSubAxisInfo());
    axesSet.insert(StringAttr::get(context, axis.getName()));
  };
  for (DimensionShardingAttr dimSharding : tensorSharding.getDimShardings()) {
    for (AxisRefAttr axis : dimSharding.getAxes()) {
      checkFullAxisAndInsertToResult(axis);
    }
  }
  for (AxisRefAttr axis : tensorSharding.getReplicatedAxes()) {
    checkFullAxisAndInsertToResult(axis);
  }
}

// Assumptions to confirm this is a shard_map pattern:
// 1. All operands are the result of a `SPMDFullToShardShape` custom call, which
//    is the result of a `Sharding` custom call.
// 2. All results are the operands of a `Sharding` custom call, which is the
//    operand of a `SPMDShardToFullShape`.
absl::StatusOr<ShardMapArgumentsResults> getJaxShardMapPatternOps(CallOp op) {
  SmallVector<ShardMapCustomCallPair> argumentOps;
  argumentOps.reserve(op.getNumOperands());
  for (Value operand : op.getOperands()) {
    auto fullToShardCustomCall = operand.getDefiningOp<CustomCallOp>();
    if (!fullToShardCustomCall) {
      return absl::NotFoundError("expecting CustomCallOp as operand");
    }
    if (!isShardMapCustomCall(fullToShardCustomCall,
                              kSPMDFullToShardShapeCallTargetName)) {
      return absl::NotFoundError(
          "expecting SPMDFullToShardShape custom call as operand");
    }
    auto shardingCustomCall =
        fullToShardCustomCall->getOperand(0).getDefiningOp<CustomCallOp>();
    if (!shardingCustomCall) {
      return absl::NotFoundError(
          "expecting CustomCallOp as operand of SPMDFullToShardShape");
    }
    if (!isShardMapCustomCall(shardingCustomCall,
                              kShardingCustomCallTargetName)) {
      return absl::NotFoundError(
          "expecting Sharding CustomCallOp as operand of SPMDFullToShardShape");
    }
    argumentOps.push_back(
        ShardMapCustomCallPair{shardingCustomCall, fullToShardCustomCall});
  }

  SmallVector<ShardMapCustomCallPair> resultOps;
  resultOps.reserve(op.getNumResults());
  for (Value result : op.getResults()) {
    if (result.use_empty()) {
      // Result is unused, so we don't need to add it to the
      // ManualComputationOp.
      resultOps.push_back(ShardMapCustomCallPair{nullptr, nullptr});
      continue;
    }
    if (!result.hasOneUse()) {
      return absl::NotFoundError(
          "expecting each result of shmap_body to have one or no uses");
    }
    Operation* sharding = *result.user_begin();
    auto shardingCustomCall = mlir::dyn_cast<CustomCallOp>(sharding);
    if (!shardingCustomCall) {
      return absl::NotFoundError(
          "expecting CustomCallOp as the use of the result of the CallOp");
    }
    if (!isShardMapCustomCall(shardingCustomCall,
                              kShardingCustomCallTargetName)) {
      return absl::NotFoundError(
          "expecting Sharding CustomCallOp as the use of the result of the "
          "CallOp");
    }
    if (!shardingCustomCall->hasOneUse()) {
      return absl::NotFoundError(
          "expecting Sharding CustomCallOp user of the result to have one use");
    }
    auto shardToFullCustomCall =
        mlir::dyn_cast<CustomCallOp>(*sharding->user_begin());
    if (!shardToFullCustomCall) {
      return absl::NotFoundError(
          "expecting CustomCallOp as the use of Sharding CustomCallOp");
    }
    if (!isShardMapCustomCall(shardToFullCustomCall,
                              kSPMDShardToFullShapeCallTargetName)) {
      return absl::NotFoundError(
          "expecting SPMDShardToFullShape CustomCallOp as the use of Sharding "
          "CustomCallOp");
    }
    resultOps.push_back(
        ShardMapCustomCallPair{shardingCustomCall, shardToFullCustomCall});
  }

  return ShardMapArgumentsResults{argumentOps, resultOps};
}

// When calling `jax.shard_map`, we have the following pattern in the MHLO.
// ```
// %shard_arg0_0 = custom_call @Sharding(%0)
// %shard_arg0_1 = custom_call @SPMDFullToShardShape(%shard_arg0_0)
// ...
// %shard_argN_0 = custom_call @Sharding(%N)
// %shard_argN_1 = custom_call @SPMDFullToShardShape(%shard_argN_0)
//
// %shard_result0, ..., %shard_resultN = func.call @shmap_body(%shard_arg0_1,
//                                                             ...,
//                                                             %shard_argN_1)
//
// %shard_result0_0 = custom_call @Sharding(%shard_result0)
// %shard_result0_1 = custom_call @SPMDShardToFullShape(%shard_result0_0)
// ...
// %shard_resultN_0 = custom_call @Sharding(%shard_resultN)
// %shard_resultN_1 = custom_call @SPMDShardToFullShape(%shard_resultN_0)
// ```
//
// We intend to
// 1. Remove the 2 argument custom calls and pass the SSA values that went into
//    the first custom call `@Sharding` into the introduced
//    `ManualComputationOp`. The initial SSA value has the global tensor shape,
//    which is what `ManualComputationOp` expects.
// 2. Create a `ManualComputationOp` by inlining or cloning the body of the
//    FuncOp `shmap_body` into the region of the created `ManualComputationOp`.
// 3. Remove the 2 result custom calls and use the types of the SSA values that
//    were the result of the last custom call `@SPMDShardToFullShape` as the
//    resulting type of the introduced `ManualComputationOp`. The last SSA
//    result value has the global tensor shape, which is what
//    `ManualComputationOp` expects.
class ShardMapImportPass
    : public mlir::PassWrapper<ShardMapImportPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShardMapImportPass)

 private:
  void runOnOperation() final {
    // TODO(zixuanjiang), b/324320384. Handle unspecified_dims and free axes.
    mlir::MLIRContext* context = &getContext();
    OpBuilder builder(context);
    ModuleOp module = getOperation();

    mlir::SymbolTableCollection symbolTableCollection;
    mlir::SymbolTable& symbolTable =
        symbolTableCollection.getSymbolTable(module);

    // The walk will attempt to walk over all ops, even if the op was deleted
    // while walking an earlier op. Hence, we cannot erase ops while walking the
    // ModuleOp. Instead, we save and delete them after the walk finishes.
    SmallVector<ShardMapOps> jaxShardMapOpsToBeErased;
    // For every shard map symbol name, the first CallOp encountered with that
    // symbol will move the body of the callee into the created
    // ManualComputationOp, and map the symbol name to the moved region.
    // Subsequent CallOps with that symbol will clone the mapped region.
    llvm::SmallDenseMap<StringRef, mlir::Region*> shardMapNameToMovedRegion;
    bool success = true;
    module->walk([&](CallOp op) {
      if (!op.getCallee().contains("shmap_body")) {
        return mlir::WalkResult::advance();
      }

      builder.setInsertionPoint(op.getOperation());

      // Only rewrite CallOps that match how JAX outputs shard_map.
      absl::StatusOr<ShardMapArgumentsResults> patternOps =
          getJaxShardMapPatternOps(op);
      if (!patternOps.ok()) {
        op.emitError(patternOps.status().message());
        success = false;
        return mlir::WalkResult::interrupt();
      }

      // Get the operands and their shardings of the new ManualComputationOp.
      SmallVector<Value> newOperands;
      SmallVector<TensorShardingAttr> inShardings;
      newOperands.reserve(op.getNumOperands());
      inShardings.reserve(op.getNumOperands());
      for (const ShardMapCustomCallPair& argumentCustomCalls :
           patternOps->argumentOps) {
        newOperands.push_back(argumentCustomCalls.shardingOp->getOperand(0));
        inShardings.push_back(
            argumentCustomCalls.shardingOp
                ->getAttrOfType<TensorShardingPerValueAttr>(kShardingAttr)
                .getShardings()
                .front());
      }

      // Get the results and their shardings of the new ManualComputationOp.
      SmallVector<mlir::Type> resultTypes;
      SmallVector<TensorShardingAttr> outShardings;
      resultTypes.reserve(op.getNumResults());
      outShardings.reserve(op.getNumResults());
      for (const ShardMapCustomCallPair& resultCustomCalls :
           patternOps->resultOps) {
        if (!resultCustomCalls.shapeTransformOp) {
          // Skip unused result.
          continue;
        }
        resultTypes.push_back(
            resultCustomCalls.shapeTransformOp->getResultTypes().front());
        outShardings.push_back(
            resultCustomCalls.shapeTransformOp
                ->getAttrOfType<TensorShardingPerValueAttr>(kShardingAttr)
                .getShardings()
                .front());
      }

      // Collect the manual axes if inOutShardings is not empty.
      SmallVector<StringAttr> manualAxes;
      if (!inShardings.empty() || !outShardings.empty()) {
        auto inOutShardings =
            llvm::concat<TensorShardingAttr>(inShardings, outShardings);
        // All in/out shardings must refer to the same mesh.
        MeshAttr mesh = mlir::sdy::getCommonMesh(inShardings, outShardings, op);
        if (!mesh) {
          op.emitError("Multiple meshes in a single manual computation.");
          success = false;
          return mlir::WalkResult::interrupt();
        }

        // Manual axes are the union of the axes in the in/out shardings.
        llvm::SmallDenseSet<StringAttr> manualAxesSet;
        for (TensorShardingAttr tensorSharding : inOutShardings) {
          insertAxisNamesFromSharding(context, tensorSharding, manualAxesSet);
        }

        manualAxes.assign(manualAxesSet.begin(), manualAxesSet.end());
        llvm::sort(manualAxes, mesh.getAxisNameComparator());
      }

      auto manualComputationOp = builder.create<ManualComputationOp>(
          op->getLoc(), resultTypes, newOperands,
          TensorShardingPerValueAttr::get(context, inShardings),
          TensorShardingPerValueAttr::get(context, outShardings), manualAxes);

      // Inline or clone the called function.
      mlir::Region& manualComputationRegion = manualComputationOp.getRegion();
      if (auto movedRegionIt = shardMapNameToMovedRegion.find(op.getCallee());
          movedRegionIt != shardMapNameToMovedRegion.end()) {
        builder.cloneRegionBefore(*movedRegionIt->second,
                                  manualComputationRegion,
                                  manualComputationRegion.begin());
      } else {
        auto calleeOp = mlir::cast<FuncOp>(
            mlir::cast<mlir::CallOpInterface>(*op).resolveCallableInTable(
                &symbolTableCollection));
        sdy::inlineRegionAndConvertTerminatorOp<ReturnOp>(
            calleeOp.getBody(), manualComputationRegion);
        shardMapNameToMovedRegion[op.getCallee()] = &manualComputationRegion;
      }

      int64_t index = 0;
      for (ShardMapCustomCallPair& resultCustomCalls : patternOps->resultOps) {
        // If there is no `shapeTransformOp`, it means the shard map result was
        // unused, and therefore we didn't add it to the `ManualComputationOp`.
        // We will remove the corresponding operand from the `ReturnOp` outside
        // the walk as we might need to clone the `manualComputationRegion` with
        // the original return values.
        if (resultCustomCalls.shapeTransformOp) {
          resultCustomCalls.shapeTransformOp->replaceAllUsesWith(
              mlir::ValueRange(manualComputationOp.getResult(index++)));
        }
      }

      // Save the CallOp and corresponding CustomCallOps to be deleted after the
      // walk, since these ops become dead.
      jaxShardMapOpsToBeErased.push_back(
          ShardMapOps{std::move(patternOps).value(), op,
                      mlir::cast<ReturnOp>(
                          mlir::sdy::getBodyTerminator(manualComputationOp))});

      return mlir::WalkResult::advance();
    });

    if (!success) {
      signalPassFailure();
      return;
    }

    for (auto& [patternOps, callOp, returnOp] : jaxShardMapOpsToBeErased) {
      // We need to remove the operands of `return_op` that correspond to unused
      // shard map results.
      mlir::BitVector returnValuesToRemove(returnOp->getNumOperands(), false);
      for (const auto& [index, resultCustomCalls] :
           llvm::enumerate(patternOps.resultOps)) {
        if (!resultCustomCalls.shapeTransformOp) {
          returnValuesToRemove.set(index);
        }
      }
      returnOp->eraseOperands(returnValuesToRemove);

      // If this callee op has multiple CallOps, then after it is erased for the
      // first CallOp, resolveCallable will return nullptr.
      if (Operation* calleeOp =
              mlir::cast<mlir::CallOpInterface>(*callOp).resolveCallableInTable(
                  &symbolTableCollection)) {
        symbolTable.erase(calleeOp);
      }

      // Erase the result custom calls.
      for (auto [shardingOp, shapeTransformOp] : patternOps.resultOps) {
        if (!shardingOp) {
          continue;
        }
        // `shapeTransformOp` is always the user of `shardingOp`, so
        // `shapeTransformOp` needs to be erased first.
        CHECK(shapeTransformOp && shapeTransformOp->use_empty());
        shapeTransformOp.erase();
        CHECK(shardingOp->use_empty());
        shardingOp.erase();
      }

      // Erase the call op itself.
      CHECK(callOp.use_empty());
      callOp.erase();

      // Erase the argument custom calls.
      llvm::SmallDenseSet<Operation*> seenShapeTransformOps;
      for (auto [shardingOp, shapeTransformOp] : patternOps.argumentOps) {
        CHECK(shardingOp && shapeTransformOp);
        if (!seenShapeTransformOps.insert(shapeTransformOp).second) {
          // This means the same `SPMDFullToShardShape` custom call is used
          // multiple times by the same shard map call op, therefore we have
          // already deleted it (as well as its corresponding `Sharding`
          // custom call).
          continue;
        }
        // `shapeTransformOp` is always the user of `shardingOp`, so
        // `shapeTransformOp` needs to be erased first. We check if the ops
        // still have uses because they can have multiple uses.
        if (shapeTransformOp->use_empty()) {
          shapeTransformOp.erase();
        }
        if (shardingOp->use_empty()) {
          shardingOp.erase();
        }
      }
    }
  }

  StringRef getArgument() const override {
    return "xla-mhlo-round-trip-shard-map-import";
  }

  StringRef getDescription() const override {
    return "Replaces a CallOp pattern unique to JAX shard_map through GSPMD "
           "lowering with a ManualComputationOp.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<SdyDialect>();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createMhloRoundTripShardMapImportPass() {
  return std::make_unique<ShardMapImportPass>();
}

void registerMhloRoundTripShardMapImportPass() {
  mlir::registerPass(createMhloRoundTripShardMapImportPass);
}

}  // namespace sdy
}  // namespace xla
