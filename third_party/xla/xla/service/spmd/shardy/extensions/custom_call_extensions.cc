/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/spmd/shardy/extensions/custom_call_extensions.h"

#include <cassert>
#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/mlir_hlo/utils/unregistered_attributes.h"
#include "xla/service/custom_call_sharding_helper.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/extensions/partial_reduce_custom_call.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"
#include "xla/sharding_op_util.h"

namespace xla {
namespace sdy {

namespace {

constexpr char kInspectShardingTargetName[] = "InspectSharding";

// Converts `value`'s Shardy tensor sharding into an `xla::HloSharding`
// (defaulting to replicated when unannotated), or emits an error on `op` and
// returns `std::nullopt` if the referenced mesh symbol does not exist.
std::optional<xla::HloSharding> convertValueShardingToHloSharding(
    mlir::Operation* op, mlir::Value value,
    const mlir::SymbolTable& symbolTable) {
  auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value);
  mlir::sdy::TensorShardingAttr sdySharding =
      (blockArg &&
       (!blockArg.getOwner() || !blockArg.getOwner()->getParentOp()))
          ? nullptr
          : mlir::sdy::getSharding(value);
  if (!sdySharding) {
    return xla::HloSharding::Replicate();
  }
  mlir::sdy::MeshAttr mesh = sdySharding.getMesh(symbolTable);
  if (!mesh) {
    op->emitOpError("unknown mesh: ") << sdySharding.getMeshOrRef();
    return std::nullopt;
  }
  return xla::sdy::convertToHloSharding(
      sdySharding, [&](mlir::sdy::TensorShardingAttr) { return mesh; });
}

// Extracts the raw `backend_config` byte slice from `customCall` without
// null-terminated C-string conversion, or returns an empty view if absent.
absl::string_view getRawBackendConfig(
    mlir::stablehlo::CustomCallOp customCall) {
  if (std::optional<mlir::Attribute> backendConfigAttr =
          customCall.getBackendConfig()) {
    if (auto strAttr = llvm::dyn_cast<mlir::StringAttr>(*backendConfigAttr)) {
      llvm::StringRef ref = strAttr.getValue();
      return absl::string_view(ref.data(), ref.size());
    }
  }
  return {};
}

// Resolves the input sharding of an `@InspectSharding` custom call and converts
// it to an `xla::HloSharding`. If `operand(0)` is a `BlockArgument` whose
// owning block was already unlinked by `FuncOp` or `ManualComputationOp`
// signature conversion, `sdy::getSharding(operand(0))` cannot walk to the
// parent op; in that case, reads the propagated inner sharding from `result(0)`
// (which has manual axes stripped inside `sdy.manual_computation`) or falls
// back to `convertedOperand` (which is attached to the converted `FuncOp` entry
// block).
std::optional<xla::HloSharding> getInspectShardingInputHloSharding(
    mlir::stablehlo::CustomCallOp customCall, mlir::Value convertedOperand,
    const mlir::SymbolTable& symbolTable) {
  mlir::Value value = customCall.getOperand(0);
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value);
      blockArg &&
      (!blockArg.getOwner() || !blockArg.getOwner()->getParentOp())) {
    value = mlir::sdy::getSharding(customCall.getResult(0))
                ? customCall.getResult(0)
                : convertedOperand;
  }
  return convertValueShardingToHloSharding(customCall, value, symbolTable);
}

// Lowers `stablehlo.custom_call @InspectSharding` by invoking the registered
// sharding-inspection callback on the input operand's sharding and replacing
// the op with its localized operand.
mlir::LogicalResult convertInspectShardingToLocal(
    mlir::stablehlo::CustomCallOp customCall, mlir::ValueRange operands,
    mlir::ConversionPatternRewriter& rewriter,
    const mlir::SymbolTable& symbolTable) {
  if (customCall.getNumOperands() != 1 || customCall.getNumResults() != 1 ||
      operands.size() != 1) {
    return customCall.emitOpError(
        "expected InspectSharding custom call to have 1 operand and 1 result");
  }

  std::optional<xla::HloSharding> hloSharding =
      getInspectShardingInputHloSharding(customCall, operands[0], symbolTable);
  if (!hloSharding.has_value()) {
    return mlir::failure();
  }

  if (const xla::CustomCallPartitioner* partitioner =
          xla::GetCustomCallPartitioner(kInspectShardingTargetName)) {
    absl::Status status = partitioner->InspectSharding(
        *hloSharding, getRawBackendConfig(customCall));
    if (!status.ok()) {
      return customCall.emitOpError("InspectSharding callback failed: ")
             << status.message();
    }
  }

  rewriter.replaceOp(customCall, operands[0]);
  return mlir::success();
}

// Provides `OpShardingRuleAttr` definitions for post-HLO XLA-internal
// `stablehlo.custom_call` targets by `call_target_name`.
struct CustomCallShardingRuleOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          CustomCallShardingRuleOpInterface, mlir::stablehlo::CustomCallOp> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation* op) const {
    auto customCall = llvm::cast<mlir::stablehlo::CustomCallOp>(op);
    llvm::StringRef callTargetName = customCall.getCallTargetName();
    auto hasDynamicTensor = [](mlir::TypeRange types) {
      return llvm::any_of(types, [](mlir::Type t) {
        auto rankedTy = llvm::dyn_cast<mlir::RankedTensorType>(t);
        return rankedTy && !rankedTy.hasStaticShape();
      });
    };
    if (hasDynamicTensor(customCall.getOperandTypes()) ||
        hasDynamicTensor(customCall.getResultTypes())) {
      return mlir::sdy::OpShardingRuleAttr();
    }
    if (callTargetName == "mhlo.bounded_dynamic_cast" ||
        callTargetName == kShardingCustomCallTargetName) {
      return mlir::sdy::OpShardingRuleBuilder::buildPointwise(customCall);
    }
    if (callTargetName == kSPMDFullToShardShapeCallTargetName ||
        callTargetName == kSPMDShardToFullShapeCallTargetName) {
      std::vector<int64_t> unspecDims;
      if (!xla::sharding_op_util::ParseAttributes(
               getRawBackendConfig(customCall), &unspecDims)
               .ok() ||
          unspecDims.empty() || customCall.getNumOperands() != 1 ||
          customCall.getNumResults() != 1) {
        return mlir::sdy::OpShardingRuleAttr();
      }
      llvm::ArrayRef<int64_t> inShape =
          mlir::sdy::getTensorShape(customCall.getOperand(0));
      llvm::ArrayRef<int64_t> outShape =
          mlir::sdy::getTensorShape(customCall.getResult(0));
      if (inShape.size() != outShape.size()) {
        return mlir::sdy::OpShardingRuleAttr();
      }
      mlir::sdy::OpShardingRuleBuilder builder(customCall);
      for (int64_t d = 0; d < static_cast<int64_t>(inShape.size()); ++d) {
        if (llvm::is_contained(unspecDims, d) && inShape[d] == outShape[d]) {
          builder.addFactor(d, d, inShape[d]);
        } else {
          builder.addFactor(d, mlir::sdy::kNullDim, inShape[d],
                            mlir::sdy::FactorType::kNeedReplication,
                            /*isBlocked=*/true);
          builder.addFactor(mlir::sdy::kNullDim, d, outShape[d],
                            mlir::sdy::FactorType::kNeedReplication,
                            /*isBlocked=*/true);
        }
      }
      return builder.build();
    }
    if (op->hasAttr(kOrigResultTypesAttr) ||
        llvm::any_of(op->getOperands(), [](mlir::Value v) {
          mlir::Operation* defOp = v.getDefiningOp();
          return defOp && defOp->hasAttr(kOrigResultTypesAttr);
        })) {
      return mlir::sdy::OpShardingRuleAttr();
    }
    if (callTargetName == "CompactWyHelper" ||
        callTargetName == "compute-on-done" ||
        callTargetName == "InvertDiagBlocksLowerTriangular" ||
        callTargetName == "InvertDiagBlocksUpperTriangular" ||
        callTargetName == "X64SplitHigh" || callTargetName == "X64SplitLow" ||
        callTargetName == "_SPMDInternalOp_MultiPad" ||
        callTargetName == "_SPMDInternalOp_RotateRight" ||
        callTargetName == "_SPMDInternalOp_MultiRotate" ||
        callTargetName == "_SPMDInternalOp_MultiSlice" ||
        callTargetName == "_SPMDInternalOp_Wrap") {
      return mlir::sdy::OpShardingRuleBuilder::buildPointwise(customCall);
    }
    if (callTargetName == "QrDecompositionBlock") {
      assert(customCall.getNumOperands() == 1 &&
             customCall.getNumResults() == 2);
      // Propagate batch dimensions pointwise and require replication on the
      // trailing matrix and `tau` dimensions (`QrExpander`).
      llvm::ArrayRef<int64_t> inShape =
          mlir::sdy::getTensorShape(customCall.getOperand(0));
      llvm::ArrayRef<int64_t> outShapeRhs =
          mlir::sdy::getTensorShape(customCall.getResult(1));
      int64_t nonBatchDim1 = inShape.size() - 2;
      int64_t nonBatchDim2 = inShape.size() - 1;
      return mlir::sdy::OpShardingRuleBuilder(customCall)
          .addPointwise(inShape.drop_back(2))
          .addFactor(nonBatchDim1, {nonBatchDim1, mlir::sdy::kNullDim},
                     inShape[nonBatchDim1],
                     mlir::sdy::FactorType::kNeedReplication)
          .addFactor(nonBatchDim2, {nonBatchDim2, mlir::sdy::kNullDim},
                     inShape[nonBatchDim2],
                     mlir::sdy::FactorType::kNeedReplication)
          .addFactor(mlir::sdy::kNullDim, {mlir::sdy::kNullDim, nonBatchDim1},
                     outShapeRhs[nonBatchDim1],
                     mlir::sdy::FactorType::kNeedReplication)
          .build();
    }
    if (callTargetName == "LuRealDecompositionBlock" ||
        callTargetName == "LuComplexDecompositionBlock") {
      int64_t numOperands = customCall.getNumOperands();
      assert(numOperands >= 1 && customCall.getNumResults() == numOperands + 2);
      // Propagate batch dimensions pointwise across `[...batch, M, N]` inputs,
      // `lu` results, `pivots` (`[...batch, min(M, N)]`), and `permutation`
      // (`[...batch, M]`), requiring replication on non-batch dimensions.
      llvm::ArrayRef<int64_t> inShape =
          mlir::sdy::getTensorShape(customCall.getOperand(0));
      llvm::ArrayRef<int64_t> pivotsShape =
          mlir::sdy::getTensorShape(customCall.getResult(numOperands));
      int64_t nonBatchDim1 = inShape.size() - 2;
      int64_t nonBatchDim2 = inShape.size() - 1;
      llvm::SmallVector<int64_t> nonBatchDim1Operands(numOperands,
                                                      nonBatchDim1);
      llvm::SmallVector<int64_t> nonBatchDim1Results(numOperands + 2,
                                                     nonBatchDim1);
      nonBatchDim1Results[numOperands] = mlir::sdy::kNullDim;
      llvm::SmallVector<int64_t> nonBatchDim2Operands(numOperands,
                                                      nonBatchDim2);
      llvm::SmallVector<int64_t> nonBatchDim2Results(numOperands + 2,
                                                     nonBatchDim2);
      nonBatchDim2Results[numOperands] = mlir::sdy::kNullDim;
      nonBatchDim2Results[numOperands + 1] = mlir::sdy::kNullDim;
      llvm::SmallVector<int64_t> pivotsOperands(numOperands,
                                                mlir::sdy::kNullDim);
      llvm::SmallVector<int64_t> pivotsResults(numOperands + 2,
                                               mlir::sdy::kNullDim);
      pivotsResults[numOperands] = nonBatchDim1;
      return mlir::sdy::OpShardingRuleBuilder(customCall)
          .addPointwise(inShape.drop_back(2))
          .addFactor(nonBatchDim1Operands, nonBatchDim1Results,
                     inShape[nonBatchDim1],
                     mlir::sdy::FactorType::kNeedReplication)
          .addFactor(nonBatchDim2Operands, nonBatchDim2Results,
                     inShape[nonBatchDim2],
                     mlir::sdy::FactorType::kNeedReplication)
          .addFactor(pivotsOperands, pivotsResults, pivotsShape[nonBatchDim1],
                     mlir::sdy::FactorType::kNeedReplication)
          .build();
    }
    if (callTargetName == "TopK") {
      assert(customCall.getNumOperands() == 1 &&
             customCall.getNumResults() == 2);
      return mlir::sdy::OpShardingRuleBuilder(customCall)
          .addPointwiseWithDiffTypeForMismatch(
              mlir::sdy::getTensorShape(customCall.getOperand(0)),
              mlir::sdy::getTensorShape(customCall.getResult(0)),
              mlir::sdy::FactorType::kNeedReplication,
              /*mismatchFactorIsBlocked=*/true)
          .build();
    }
    if (callTargetName == kPartialReduceTargetName) {
      return createPartialReduceShardingRule(customCall);
    }
    if (callTargetName == "X64Combine") {
      // Replicate and block propagation when `X64Combine` feeds
      // `RngBitGeneratorOp`, whose 1D `ui64` initial state must stay replicated
      // and cannot be resharded.
      if (llvm::any_of(customCall->getUsers(), [](mlir::Operation* user) {
            return llvm::isa<mlir::stablehlo::RngBitGeneratorOp>(user);
          })) {
        mlir::sdy::OpShardingRuleBuilder builder(customCall);
        llvm::ArrayRef<int64_t> shape =
            mlir::sdy::getTensorShape(customCall.getResult(0));
        assert(shape.size() == 1);
        builder.addFactor(0, shape.front(),
                          mlir::sdy::FactorType::kNeedReplication,
                          /*isBlocked=*/true);
        return builder.build();
      }
      return mlir::sdy::OpShardingRuleBuilder::buildPointwise(customCall);
    }
    // TODO(b/327191011): output unregistered op stats instead.
    static llvm::once_flag onceFlag;
    mlir::sdy::emitOpWarningOnce(
        onceFlag, customCall,
        llvm::formatv(
            "custom call @{0} is unknown to SDY sharding rule registry",
            callTargetName)
            .str());
    return mlir::sdy::OpShardingRuleAttr();
  }
};

using ConvertGlobalToLocalHandler = mlir::LogicalResult (*)(
    mlir::stablehlo::CustomCallOp, mlir::ValueRange,
    mlir::ConversionPatternRewriter&, const mlir::SymbolTable&);

// Dispatches `stablehlo.custom_call` global-to-local lowering by
// `call_target_name`, returning `std::nullopt` for unhandled targets so Shardy
// falls back to generic op localization.
struct CustomCallPartitioningOpInterface
    : public mlir::sdy::PartitioningOpInterface::ExternalModel<
          CustomCallPartitioningOpInterface, mlir::stablehlo::CustomCallOp> {
  std::optional<mlir::LogicalResult> resolvePermutationFactors(
      mlir::Operation* op, mlir::sdy::OpShardingRuleAttr shardingRule,
      mlir::IRRewriter& rewriter, const mlir::SymbolTable& symbolTable) const {
    auto customCall = llvm::cast<mlir::stablehlo::CustomCallOp>(op);
    if (customCall.getCallTargetName() == kPartialReduceTargetName) {
      return resolvePartialReduce(customCall, shardingRule, rewriter,
                                  symbolTable);
    }
    return std::nullopt;
  }

  std::optional<mlir::LogicalResult> convertGlobalToLocal(
      mlir::Operation* op, mlir::ValueRange operands,
      mlir::ConversionPatternRewriter& rewriter,
      const mlir::TypeConverter& /*typeConverter*/,
      const mlir::SymbolTable& symbolTable) const {
    auto customCall = llvm::cast<mlir::stablehlo::CustomCallOp>(op);
    if (customCall.getCallTargetName() == kPartialReduceTargetName) {
      // Strip stale global `xla_shape` in place and return `std::nullopt` so
      // Shardy's generic op localizer narrows result types to shard-local
      // shapes instead of running a replacement `ConvertGlobalToLocalHandler`.
      rewriter.modifyOpInPlace(
          customCall, [&]() { customCall->removeAttr(xla::kXlaShape); });
      return std::nullopt;
    }
    ConvertGlobalToLocalHandler handler =
        llvm::StringSwitch<ConvertGlobalToLocalHandler>(
            customCall.getCallTargetName())
            .Case(kInspectShardingTargetName, &convertInspectShardingToLocal)
            .Default(nullptr);
    if (!handler) {
      return std::nullopt;
    }
    return handler(customCall, operands, rewriter, symbolTable);
  }
};

}  // namespace

void registerCustomCallExtensions(mlir::DialectRegistry& registry) {
  registry.addExtension(+[](mlir::MLIRContext* ctx,
                            mlir::stablehlo::StablehloDialect*) {
    mlir::stablehlo::CustomCallOp::attachInterface<
        CustomCallShardingRuleOpInterface, CustomCallPartitioningOpInterface>(
        *ctx);
  });
}

}  // namespace sdy
}  // namespace xla
