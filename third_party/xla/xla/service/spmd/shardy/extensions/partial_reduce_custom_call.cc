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

#include "xla/service/spmd/shardy/extensions/partial_reduce_custom_call.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/builder/lib/approx_topk_shape.h"
#include "xla/mlir_hlo/utils/unregistered_attributes.h"

namespace xla {
namespace sdy {
namespace {

// Discardable attribute and `@PartialReduce` `backend_config` keys/values.
constexpr llvm::StringLiteral kMhloBackendConfigAttr = "mhlo.backend_config";
constexpr llvm::StringLiteral kLog2ReductionKey = "log2_reduction";
constexpr llvm::StringLiteral kReductionDimKey = "reduction_dim";
constexpr llvm::StringLiteral kTopKKey = "top_k";
constexpr llvm::StringLiteral kRecallTargetKey = "recall_target";
constexpr llvm::StringLiteral kToApplyTypeKey = "to_apply_type";
constexpr llvm::StringLiteral kComparatorToApplyType = "comparator";
constexpr llvm::StringLiteral kCombinerToApplyType = "combiner";

// Specifies how the `called_computations` body of `@PartialReduce` combines
// candidate elements during Stage 2 folding.
enum class ToApplyKind {
  // Computation takes grouped `(lhs_0..lhs_{K-1}, rhs_0..rhs_{K-1})` and
  // returns `K` scalar tensors (or a `K`-element tuple).
  kCombiner,
  // Computation takes interleaved `(lhs_0, rhs_0, ..., lhs_{K-1}, rhs_{K-1})`
  // and returns a single `tensor<i1>` predicate selecting `lhs` vs `rhs`.
  kComparator,
};

// Holds raw fields decoded from `backend_config` and `mhlo.backend_config`,
// giving per-field precedence to `backend_config`.
struct RawPartialReduceBackendConfig {
  std::optional<int64_t> reductionDim;
  std::optional<int64_t> topK;
  std::optional<double> recallTarget;
  std::optional<std::string> rawToApplyType;
  // Additional JSON fields from `backend_config` preserved when serializing
  // Stage 1's `backend_config`.
  llvm::json::Object extraJsonFields;
};

// Holds validated arity and shape metadata shared between propagation rule
// construction and two-stage lowering.
struct PartialReduceShapeSignature {
  int64_t numInputs = 0;
  int64_t rank = 0;
  llvm::ArrayRef<int64_t> inShape;
  llvm::ArrayRef<int64_t> outShape;
};

// Returns true if `lhs` and `rhs` represent equivalent Shardy tensor shardings,
// treating a missing sharding attribute as replicated.
bool areShardingsEquivalent(mlir::sdy::TensorShardingAttr lhs,
                            mlir::sdy::TensorShardingAttr rhs) {
  return lhs == rhs || (lhs ? lhs.isEquivalent(rhs) : rhs.isEquivalent(lhs));
}

// Populates unset fields in `out` from a dictionary or JSON string backend
// config attribute `attr`. Returns `failure()` if `attr` is a non-empty string
// that is not a JSON object (emitting an op error when `emitErrorOnInvalidJson`
// is true).
mlir::LogicalResult mergePartialReduceConfigAttr(
    mlir::stablehlo::CustomCallOp customCall, mlir::Attribute attr,
    bool emitErrorOnInvalidJson, RawPartialReduceBackendConfig& out) {
  if (!attr) {
    return mlir::success();
  }
  if (auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr)) {
    // `DictionaryAttr` configs come from pre-HLO StableHLO attributes (e.g.,
    // `mhlo.backend_config`); TPU `PartialReduce` lowering only consumes
    // `log2_reduction`, `reduction_dim`, and `to_apply_type`, so any extra
    // pre-HLO keys (such as `aggregate_to_topk`) are intentionally omitted from
    // Stage 1's JSON `backend_config`.
    if (!out.reductionDim) {
      if (auto v = dict.getAs<mlir::IntegerAttr>(kReductionDimKey)) {
        out.reductionDim = v.getInt();
      }
    }
    if (!out.topK) {
      if (auto v = dict.getAs<mlir::IntegerAttr>(kTopKKey)) {
        out.topK = v.getInt();
      }
    }
    if (!out.recallTarget) {
      if (auto v = dict.getAs<mlir::FloatAttr>(kRecallTargetKey)) {
        out.recallTarget = v.getValueAsDouble();
      }
    }
    if (!out.rawToApplyType) {
      if (auto v = dict.getAs<mlir::StringAttr>(kToApplyTypeKey)) {
        out.rawToApplyType = v.getValue().str();
      }
    }
    return mlir::success();
  }

  auto str = llvm::dyn_cast<mlir::StringAttr>(attr);
  if (!str || str.getValue().empty()) {
    return mlir::success();
  }
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(str.getValue());
  if (!parsed || !parsed->getAsObject()) {
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
    }
    if (emitErrorOnInvalidJson) {
      customCall.emitOpError(
          "expected JSON object in PartialReduce backend_config");
    }
    return mlir::failure();
  }

  const llvm::json::Object* obj = parsed->getAsObject();
  for (const auto& kv : *obj) {
    out.extraJsonFields.try_emplace(kv.first, kv.second);
  }
  if (!out.reductionDim) {
    out.reductionDim = obj->getInteger(kReductionDimKey);
  }
  if (!out.topK) {
    out.topK = obj->getInteger(kTopKKey);
  }
  if (!out.recallTarget) {
    out.recallTarget = obj->getNumber(kRecallTargetKey);
  }
  if (!out.rawToApplyType) {
    if (std::optional<llvm::StringRef> v = obj->getString(kToApplyTypeKey)) {
      out.rawToApplyType = v->str();
    }
  }
  return mlir::success();
}

// Decodes `backend_config` and `mhlo.backend_config` on `customCall`, giving
// per-field precedence to `backend_config` and falling back to
// `mhlo.backend_config` only for fields still unset.
std::optional<RawPartialReduceBackendConfig> decodePartialReduceBackendConfig(
    mlir::stablehlo::CustomCallOp customCall, bool emitErrorOnInvalidJson) {
  RawPartialReduceBackendConfig raw;
  if (std::optional<mlir::Attribute> backendAttr =
          customCall.getBackendConfig()) {
    if (mlir::failed(mergePartialReduceConfigAttr(
            customCall, *backendAttr, emitErrorOnInvalidJson, raw)) &&
        emitErrorOnInvalidJson) {
      return std::nullopt;
    }
  }
  if (mlir::failed(mergePartialReduceConfigAttr(
          customCall, customCall->getAttr(kMhloBackendConfigAttr),
          emitErrorOnInvalidJson, raw)) &&
      emitErrorOnInvalidJson) {
    return std::nullopt;
  }
  return raw;
}

// Infers `reduction_dim` when omitted from backend config by finding the unique
// dimension whose input and output sizes differ.
std::optional<int64_t> inferReductionDimFromShapes(
    llvm::ArrayRef<int64_t> inShape, llvm::ArrayRef<int64_t> outShape) {
  std::optional<int64_t> reductionDim;
  for (int64_t dim = 0; dim < static_cast<int64_t>(inShape.size()); ++dim) {
    if (inShape[dim] == outShape[dim]) {
      continue;
    }
    if (reductionDim.has_value()) {
      return std::nullopt;
    }
    reductionDim = dim;
  }
  return reductionDim;
}

// Validates the `2 * K -> K` arity and static tensor shapes of `customCall`
// without requiring init scalar element types to match data operand element
// types (propagation tests allow heterogeneous dummy init scalars).
std::optional<PartialReduceShapeSignature> parsePartialReduceShapeSignature(
    mlir::stablehlo::CustomCallOp customCall, bool emitError) {
  const int64_t numInputs = customCall.getNumResults();
  const int64_t numOperands = customCall.getNumOperands();
  if (numInputs < 1 || numOperands != 2 * numInputs) {
    if (emitError) {
      customCall.emitOpError(
          "expected PartialReduce to have K >= 1 results and 2 * K operands, "
          "got ")
          << numOperands << " operands and " << numInputs << " results";
    }
    return std::nullopt;
  }

  auto firstInType = llvm::dyn_cast<mlir::RankedTensorType>(
      customCall.getOperand(0).getType());
  auto firstOutType =
      llvm::dyn_cast<mlir::RankedTensorType>(customCall.getResult(0).getType());
  if (!firstInType || !firstOutType || !firstInType.hasStaticShape() ||
      !firstOutType.hasStaticShape() || firstInType.getRank() < 1 ||
      firstInType.getRank() != firstOutType.getRank()) {
    if (emitError) {
      customCall.emitOpError(
          "expected static non-scalar RankedTensorType operands and results of "
          "matching rank on PartialReduce");
    }
    return std::nullopt;
  }

  llvm::ArrayRef<int64_t> inShape = firstInType.getShape();
  llvm::ArrayRef<int64_t> outShape = firstOutType.getShape();
  for (int64_t i = 0; i < numInputs; ++i) {
    auto inType = llvm::dyn_cast<mlir::RankedTensorType>(
        customCall.getOperand(i).getType());
    auto outType = llvm::dyn_cast<mlir::RankedTensorType>(
        customCall.getResult(i).getType());
    auto initType = llvm::dyn_cast<mlir::RankedTensorType>(
        customCall.getOperand(numInputs + i).getType());
    if (!inType || !outType || !initType || initType.getRank() != 0 ||
        inType.getShape() != inShape || outType.getShape() != outShape) {
      if (emitError) {
        customCall.emitOpError(
            "mismatched operand, result, or scalar init types on "
            "PartialReduce");
      }
      return std::nullopt;
    }
  }

  PartialReduceShapeSignature sig;
  sig.numInputs = numInputs;
  sig.rank = firstInType.getRank();
  sig.inShape = inShape;
  sig.outShape = outShape;
  return sig;
}

}  // namespace

mlir::sdy::OpShardingRuleAttr createPartialReduceShardingRule(
    mlir::stablehlo::CustomCallOp customCall) {
  std::optional<PartialReduceShapeSignature> sig =
      parsePartialReduceShapeSignature(customCall, /*emitError=*/false);
  if (!sig) {
    return mlir::sdy::OpShardingRuleAttr();
  }

  RawPartialReduceBackendConfig rawConfig =
      decodePartialReduceBackendConfig(customCall,
                                       /*emitErrorOnInvalidJson=*/false)
          .value_or(RawPartialReduceBackendConfig());
  std::optional<int64_t> reductionDim =
      rawConfig.reductionDim
          ? rawConfig.reductionDim
          : inferReductionDimFromShapes(sig->inShape, sig->outShape);
  if (!reductionDim || *reductionDim < 0 || *reductionDim >= sig->rank) {
    return mlir::sdy::OpShardingRuleAttr();
  }

  mlir::sdy::OpShardingRuleBuilder builder(customCall);
  for (int64_t dim = 0; dim < sig->rank; ++dim) {
    int64_t inDimSize = sig->inShape[dim];
    int64_t outDimSize = sig->outShape[dim];
    if (dim != *reductionDim) {
      if (inDimSize != outDimSize) {
        return mlir::sdy::OpShardingRuleAttr();
      }
      builder.addFactor(dim, inDimSize);
      continue;
    }

    // Mark input `reduction_dim` as `kPermutation` so operands align their
    // reduction shardings and trigger `resolvePartialReduce` during export;
    // mark output `reduction_dim` as `kNeedReplication` + `isBlocked` because
    // the reduced top-K output window must be replicated along the reduction
    // axis.
    builder.addFactorSameForAllOperands(dim, inDimSize,
                                        mlir::sdy::FactorType::kPermutation);
    builder.addFactorSameForAllResults(dim, outDimSize,
                                       mlir::sdy::FactorType::kNeedReplication,
                                       /*isBlocked=*/true);
  }
  return builder.build();
}

namespace {

// Holds validated `@PartialReduce` backend configuration for two-stage
// lowering.
struct PartialReduceConfig {
  int64_t reductionDim = 0;
  int64_t topK = 0;
  double recallTarget = 0.0;
  ToApplyKind toApplyKind = ToApplyKind::kCombiner;
  // Original `to_apply_type` string from backend config (if present), preserved
  // verbatim when re-serializing Stage 1's `backend_config`.
  std::optional<std::string> rawToApplyType;
  // Unrecognized JSON fields preserved when serializing Stage 1's
  // `backend_config`.
  llvm::json::Object extraJsonFields;
};

// Holds fully validated tensor shapes, element types, and scalar identity
// operands for two-stage `@PartialReduce` lowering.
struct PartialReduceSignature {
  int64_t numInputs = 0;
  int64_t rank = 0;
  int64_t reductionDim = 0;
  int64_t databaseSize = 0;
  int64_t outputSize = 0;
  llvm::SmallVector<mlir::Type> elemTypes;
  llvm::SmallVector<mlir::Value> initVals;
};

// Bundles validated op metadata, configuration, shardings, and Stage 1 shape
// parameters across the two-stage `@PartialReduce` lowering.
struct PartialReduceLoweringContext {
  mlir::stablehlo::CustomCallOp customCall;
  mlir::Location loc;
  PartialReduceSignature sig;
  PartialReduceConfig config;
  mlir::func::FuncOp toApplyFunc;
  mlir::sdy::MeshAttr mesh;
  mlir::sdy::TensorShardingAttr firstInSharding;
  mlir::sdy::TensorShardingAttr reductionDimReplicatedSharding;
  llvm::SmallVector<mlir::sdy::AxisRefAttr> manualAxes;
  int64_t databaseShards;
  int64_t databaseSizePerShard;
  int64_t stageOneOutSize;
  int64_t stageOneLog2Reduction;

  // Returns true if each shard runs a local Stage 1 `@PartialReduce` before
  // all-gathering candidates across `reduction_dim`.
  bool runsStageOnePartialReduce() const { return stageOneLog2Reduction > 0; }

  // Returns the per-shard candidate count along `reduction_dim` entering the
  // cross-shard all-gather.
  int64_t localCandidatesSize() const {
    return runsStageOnePartialReduce() ? stageOneOutSize : databaseSizePerShard;
  }
};

// Extracts the unique `operand(0)` dimension mapped to `shardingRule`'s single
// `kPermutation` factor, or returns `std::nullopt` if `shardingRule` does not
// match `@PartialReduce`'s rule layout.
std::optional<int64_t> findPermutationReductionDim(
    mlir::sdy::OpShardingRuleAttr shardingRule) {
  if (!shardingRule || shardingRule.getNumOperands() < 2 ||
      shardingRule.getPermutationFactors().size() != 1) {
    return std::nullopt;
  }
  const int64_t permFactor = shardingRule.getPermutationFactors().front();
  std::optional<int64_t> reductionDim;
  for (auto [dim, dimMapping] :
       llvm::enumerate(shardingRule.getOperandMapping(0).getDimMappings())) {
    if (!llvm::is_contained(dimMapping.getFactorIndices(), permFactor)) {
      continue;
    }
    if (dimMapping.getFactorIndices().size() != 1 || reductionDim.has_value()) {
      return std::nullopt;
    }
    reductionDim = static_cast<int64_t>(dim);
  }
  return reductionDim;
}

// Validates `customCall` shapes, element-type equality across `(in_i, out_i,
// init_i)`, non-reduction dimension agreement, and positive reduction sizes.
std::optional<PartialReduceSignature> validatePartialReduceForLowering(
    mlir::stablehlo::CustomCallOp customCall, int64_t reductionDim) {
  std::optional<PartialReduceShapeSignature> shapeSig =
      parsePartialReduceShapeSignature(customCall, /*emitError=*/true);
  if (!shapeSig) {
    return std::nullopt;
  }
  if (reductionDim < 0 || reductionDim >= shapeSig->rank) {
    customCall.emitOpError("PartialReduce reduction_dim out of bounds");
    return std::nullopt;
  }

  PartialReduceSignature sig;
  sig.numInputs = shapeSig->numInputs;
  sig.rank = shapeSig->rank;
  sig.reductionDim = reductionDim;
  sig.elemTypes.reserve(sig.numInputs);
  sig.initVals.reserve(sig.numInputs);

  for (int64_t i = 0; i < sig.numInputs; ++i) {
    auto inType =
        llvm::cast<mlir::RankedTensorType>(customCall.getOperand(i).getType());
    auto outType =
        llvm::cast<mlir::RankedTensorType>(customCall.getResult(i).getType());
    mlir::Value initVal = customCall.getOperand(sig.numInputs + i);
    auto initType = llvm::cast<mlir::RankedTensorType>(initVal.getType());
    if (inType.getElementType() != outType.getElementType() ||
        inType.getElementType() != initType.getElementType()) {
      customCall.emitOpError(
          "mismatched operand, result, or scalar init types on PartialReduce");
      return std::nullopt;
    }
    sig.elemTypes.push_back(inType.getElementType());
    sig.initVals.push_back(initVal);
  }

  for (int64_t dim = 0; dim < sig.rank; ++dim) {
    if (dim != reductionDim &&
        shapeSig->inShape[dim] != shapeSig->outShape[dim]) {
      customCall.emitOpError(
          "PartialReduce non-reduction dimension mismatch at dim ")
          << dim;
      return std::nullopt;
    }
  }

  sig.databaseSize = shapeSig->inShape[reductionDim];
  sig.outputSize = shapeSig->outShape[reductionDim];
  if (sig.databaseSize <= 0 || sig.outputSize <= 0) {
    customCall.emitOpError(
        "expected positive database_size and output_size on PartialReduce");
    return std::nullopt;
  }
  return sig;
}

// Parses and validates `PartialReduceConfig` for two-stage lowering, verifying
// `reduction_dim == expectedReductionDim`, `top_k >= 1`, `recall_target`, and
// `to_apply_type`.
std::optional<PartialReduceConfig> parsePartialReduceConfig(
    mlir::stablehlo::CustomCallOp customCall, int64_t expectedReductionDim) {
  std::optional<RawPartialReduceBackendConfig> raw =
      decodePartialReduceBackendConfig(customCall,
                                       /*emitErrorOnInvalidJson=*/true);
  if (!raw) {
    return std::nullopt;
  }

  const int64_t reductionDim = raw->reductionDim.value_or(expectedReductionDim);
  if (reductionDim != expectedReductionDim) {
    customCall.emitOpError("PartialReduce reduction_dim ")
        << reductionDim << " does not match sharding rule dimension "
        << expectedReductionDim;
    return std::nullopt;
  }
  if (!raw->topK || *raw->topK < 1 || !raw->recallTarget) {
    customCall.emitOpError(
        "expected PartialReduce backend_config with top_k >= 1 and "
        "recall_target");
    return std::nullopt;
  }

  ToApplyKind toApplyKind = ToApplyKind::kCombiner;
  if (raw->rawToApplyType.has_value()) {
    llvm::StringRef toApplyType(*raw->rawToApplyType);
    if (toApplyType.equals_insensitive(kComparatorToApplyType)) {
      toApplyKind = ToApplyKind::kComparator;
    } else if (!toApplyType.equals_insensitive(kCombinerToApplyType)) {
      customCall.emitOpError("unsupported PartialReduce to_apply_type: ")
          << toApplyType;
      return std::nullopt;
    }
  }

  PartialReduceConfig config;
  config.reductionDim = reductionDim;
  config.topK = *raw->topK;
  config.recallTarget = *raw->recallTarget;
  config.toApplyKind = toApplyKind;
  config.rawToApplyType = std::move(raw->rawToApplyType);
  config.extraJsonFields = std::move(raw->extraJsonFields);
  return config;
}

// Serializes the JSON `backend_config` string for the Stage 1 per-shard
// `@PartialReduce` call with `log2_reduction = stageOneLog2Reduction`.
std::string formatStageOneBackendConfig(const PartialReduceConfig& config,
                                        int64_t stageOneLog2Reduction) {
  llvm::json::Object obj = config.extraJsonFields;
  obj[kLog2ReductionKey] = stageOneLog2Reduction;
  obj[kReductionDimKey] = config.reductionDim;
  if (config.rawToApplyType.has_value()) {
    obj[kToApplyTypeKey] = *config.rawToApplyType;
  } else {
    obj.erase(kToApplyTypeKey);
  }
  obj[kTopKKey] = config.topK;
  obj[kRecallTargetKey] = config.recallTarget;
  std::string out;
  llvm::raw_string_ostream os(out);
  os << llvm::json::Value(std::move(obj));
  return out;
}

// Returns a copy of `type` with dimension `dim` resized to `newDimSize`.
mlir::RankedTensorType withReplacedDimSize(mlir::RankedTensorType type,
                                           int64_t dim, int64_t newDimSize) {
  llvm::SmallVector<int64_t> shape(type.getShape());
  shape[dim] = newDimSize;
  return mlir::RankedTensorType::get(shape, type.getElementType());
}

// Reshards `val` to `targetSharding` if its current sharding is not equivalent,
// treating a null `targetSharding` as fully replicated.
mlir::Value reshardIfNeeded(mlir::Location loc, mlir::Value val, int64_t rank,
                            mlir::sdy::TensorShardingAttr targetSharding,
                            mlir::IRRewriter& rewriter) {
  mlir::sdy::TensorShardingAttr curSharding = mlir::sdy::getSharding(val);
  if (areShardingsEquivalent(curSharding, targetSharding)) {
    return val;
  }
  mlir::sdy::TensorShardingAttr dstSharding =
      targetSharding
          ? targetSharding
          : mlir::sdy::TensorShardingAttr::getFullyReplicated(
                rewriter.getContext(), rank, curSharding.getMeshOrRef(),
                /*isClosed=*/true);
  return mlir::sdy::ReshardOp::create(rewriter, loc, val.getType(), val,
                                      dstSharding);
}

// Aligns all `numInputs` data operands of `customCall` in place to
// `targetSharding` when `customCall` remains a single-stage op.
void alignPartialReduceDataOperands(
    mlir::stablehlo::CustomCallOp customCall, int64_t numInputs, int64_t rank,
    mlir::sdy::TensorShardingAttr targetSharding, mlir::IRRewriter& rewriter) {
  mlir::Location loc = customCall.getLoc();
  llvm::SmallVector<std::pair<int64_t, mlir::Value>> updates;
  for (int64_t i = 0; i < numInputs; ++i) {
    mlir::Value origVal = customCall.getOperand(i);
    mlir::Value alignedVal =
        reshardIfNeeded(loc, origVal, rank, targetSharding, rewriter);
    if (alignedVal != origVal) {
      updates.emplace_back(i, alignedVal);
    }
  }
  if (!updates.empty()) {
    rewriter.modifyOpInPlace(customCall, [&]() {
      for (auto [idx, resharded] : updates) {
        customCall.setOperand(idx, resharded);
      }
    });
  }
}

// Computes the `manual_axes` list for Stage 1's `sdy.manual_computation`, or
// returns `std::nullopt` when `mesh` or `firstInSharding` uses sub-axes or
// nested manual axes that `sdy.manual_computation` cannot represent cleanly.
std::optional<llvm::SmallVector<mlir::sdy::AxisRefAttr>>
computeEligibleStageOneManualAxes(
    mlir::stablehlo::CustomCallOp customCall,
    mlir::sdy::TensorShardingAttr firstInSharding,
    llvm::ArrayRef<mlir::sdy::AxisRefAttr> reductionAxes,
    mlir::sdy::MeshAttr mesh) {
  if (!mesh || mesh.isSingleDevice()) {
    return std::nullopt;
  }

  // Collect manual axes already claimed by enclosing `ManualComputationOp`s.
  llvm::SmallVector<mlir::sdy::AxisRefAttr> parentManualAxes;
  for (auto manualOp =
           customCall->getParentOfType<mlir::sdy::ManualComputationOp>();
       manualOp;
       manualOp = manualOp->getParentOfType<mlir::sdy::ManualComputationOp>()) {
    llvm::append_range(parentManualAxes, manualOp.getManualAxes());
  }

  // Reject sub-axis reduction shardings or reduction axes already manual in an
  // outer `sdy.manual_computation`.
  for (mlir::sdy::AxisRefAttr redAxis : reductionAxes) {
    if (redAxis.getSubAxisInfo() != nullptr) {
      return std::nullopt;
    }
    if (llvm::any_of(parentManualAxes, [&](mlir::sdy::AxisRefAttr parentAxis) {
          return redAxis.overlaps(parentAxis);
        })) {
      return std::nullopt;
    }
  }

  mlir::MLIRContext* mlirCtx = customCall.getContext();
  llvm::SmallVector<mlir::sdy::AxisRefAttr> allMeshAxes;
  allMeshAxes.reserve(mesh.getAxes().size());
  for (mlir::sdy::MeshAxisAttr axis : mesh.getAxes()) {
    allMeshAxes.push_back(mlir::sdy::AxisRefAttr::get(mlirCtx, axis.getName()));
  }
  llvm::SmallVector<mlir::sdy::AxisRefAttr> manualAxes =
      mlir::sdy::getAxisSetDiff(allMeshAxes, parentManualAxes, mesh);

  // Guard against conflicting sub-axes when an enclosing
  // `sdy.manual_computation` (`parentManualAxes`) splits a mesh axis so
  // `manualAxes` contains a sub-axis (mirroring `ManualComputationOp`
  // verification; full manual axes always coexist with sub-axis batch
  // shardings).
  for (mlir::sdy::AxisRefAttr manualAxis : manualAxes) {
    if (firstInSharding.anyOfAxisRef([&](mlir::sdy::AxisRefAttr axisRef) {
          return !axisRef.canCoexist(manualAxis);
        })) {
      return std::nullopt;
    }
  }
  return manualAxes;
}

// Resolves `customCall` in place without two-stage decomposition when
// `reduction_dim` is unsharded, sharded only on size-1 axes, or sharded on
// axes incompatible with Stage 1 `sdy.manual_computation`. Returns true if
// resolved as a single stage, or false if two-stage lowering is needed
// (populating `manualAxesOut`).
bool tryResolveAsSingleStage(
    mlir::stablehlo::CustomCallOp customCall, const PartialReduceSignature& sig,
    const mlir::SymbolTable& symbolTable, mlir::IRRewriter& rewriter,
    llvm::SmallVector<mlir::sdy::AxisRefAttr>& manualAxesOut) {
  mlir::sdy::TensorShardingAttr firstInSharding =
      mlir::sdy::getSharding(customCall.getOperand(0));
  if (!firstInSharding ||
      firstInSharding.getDimSharding(sig.reductionDim).emptyAxes()) {
    alignPartialReduceDataOperands(customCall, sig.numInputs, sig.rank,
                                   firstInSharding, rewriter);
    return true;
  }

  mlir::MLIRContext* mlirCtx = customCall.getContext();
  llvm::ArrayRef<mlir::sdy::AxisRefAttr> reductionAxes =
      firstInSharding.getDimSharding(sig.reductionDim).getAxes();
  mlir::sdy::MeshAttr mesh = firstInSharding.getMesh(symbolTable);
  mlir::sdy::TensorShardingAttr reductionDimReplicatedSharding =
      firstInSharding.replaceDimSharding(sig.reductionDim,
                                         mlir::sdy::DimensionShardingAttr::get(
                                             mlirCtx, {}, /*is_closed=*/true));

  std::optional<llvm::SmallVector<mlir::sdy::AxisRefAttr>> eligibleManualAxes =
      computeEligibleStageOneManualAxes(customCall, firstInSharding,
                                        reductionAxes, mesh);
  if (!eligibleManualAxes) {
    // Fall back to replicating `reduction_dim` so single-stage `@PartialReduce`
    // sees the full database on every shard.
    alignPartialReduceDataOperands(customCall, sig.numInputs, sig.rank,
                                   reductionDimReplicatedSharding, rewriter);
    return true;
  }

  const int64_t databaseShards =
      firstInSharding.getDimSharding(sig.reductionDim).getShardedSize(mesh);
  if (databaseShards <= 1) {
    alignPartialReduceDataOperands(customCall, sig.numInputs, sig.rank,
                                   firstInSharding, rewriter);
    return true;
  }

  manualAxesOut = std::move(*eligibleManualAxes);
  return false;
}

// Validates `toApplyFunc`'s block structure, `2 * K` scalar argument types,
// and comparator (`tensor<i1>`) or combiner (`K` scalars or `K`-tuple) return
// types before any Stage 1 or Stage 2 IR is emitted.
mlir::LogicalResult validateToApplyFunction(
    mlir::stablehlo::CustomCallOp customCall, mlir::func::FuncOp toApplyFunc,
    const PartialReduceSignature& sig, ToApplyKind toApplyKind) {
  llvm::StringRef calleeName = toApplyFunc.getSymName();
  const int64_t numInputs = sig.numInputs;
  if (!toApplyFunc.getBody().hasOneBlock()) {
    return customCall.emitOpError(
               "expected single-block PartialReduce computation @")
           << calleeName;
  }
  mlir::Block& body = toApplyFunc.getBody().front();
  if (static_cast<int64_t>(body.getNumArguments()) != 2 * numInputs) {
    return customCall.emitOpError("expected PartialReduce computation @")
           << calleeName << " to have " << (2 * numInputs) << " arguments";
  }

  const bool isComparator = toApplyKind == ToApplyKind::kComparator;
  for (int64_t i = 0; i < numInputs; ++i) {
    auto expectedScalar = mlir::RankedTensorType::get({}, sig.elemTypes[i]);
    mlir::Type lhsType = body.getArgument(isComparator ? 2 * i : i).getType();
    mlir::Type rhsType =
        body.getArgument(isComparator ? 2 * i + 1 : numInputs + i).getType();
    if (lhsType != expectedScalar || rhsType != expectedScalar) {
      return customCall.emitOpError(
                 "computation argument types do not match PartialReduce "
                 "element types in @")
             << calleeName;
    }
  }

  mlir::Operation* terminator = body.getTerminator();
  if (isComparator) {
    if (terminator->getNumOperands() != 1) {
      return customCall.emitOpError("expected comparator @")
             << calleeName << " to return 1 predicate";
    }
    auto predType = llvm::dyn_cast<mlir::RankedTensorType>(
        terminator->getOperand(0).getType());
    if (!predType || predType.getRank() != 0 ||
        !predType.getElementType().isInteger(1)) {
      return customCall.emitOpError("expected comparator @")
             << calleeName << " to return tensor<i1>";
    }
    return mlir::success();
  }

  llvm::SmallVector<mlir::Type> retTypes;
  if (static_cast<int64_t>(terminator->getNumOperands()) == numInputs &&
      !llvm::isa<mlir::TupleType>(terminator->getOperand(0).getType())) {
    llvm::append_range(retTypes, terminator->getOperandTypes());
  } else if (terminator->getNumOperands() == 1 &&
             llvm::isa<mlir::TupleType>(terminator->getOperand(0).getType())) {
    auto tupleType =
        llvm::cast<mlir::TupleType>(terminator->getOperand(0).getType());
    if (static_cast<int64_t>(tupleType.size()) != numInputs) {
      return customCall.emitOpError("expected combiner @")
             << calleeName << " tuple return size " << numInputs;
    }
    llvm::append_range(retTypes, tupleType.getTypes());
  } else {
    return customCall.emitOpError("expected combiner @")
           << calleeName << " to return " << numInputs << " scalar values";
  }

  for (int64_t i = 0; i < numInputs; ++i) {
    if (retTypes[i] != mlir::RankedTensorType::get({}, sig.elemTypes[i])) {
      return customCall.emitOpError("combiner @")
             << calleeName << " return type mismatch at index " << i;
    }
  }
  return mlir::success();
}

// Parses backend config, validates the `called_computations` callee, and
// computes Stage 1 per-shard output sizing for two-stage lowering.
std::optional<PartialReduceLoweringContext> buildPartialReduceLoweringContext(
    mlir::stablehlo::CustomCallOp customCall, const PartialReduceSignature& sig,
    llvm::SmallVector<mlir::sdy::AxisRefAttr> manualAxes,
    const mlir::SymbolTable& symbolTable) {
  std::optional<PartialReduceConfig> config =
      parsePartialReduceConfig(customCall, sig.reductionDim);
  if (!config) {
    return std::nullopt;
  }

  if (customCall.getCalledComputations().size() != 1) {
    customCall.emitOpError(
        "expected PartialReduce to have exactly 1 called_computation");
    return std::nullopt;
  }
  auto calleeRef = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(
      customCall.getCalledComputations()[0]);
  mlir::func::FuncOp toApplyFunc =
      calleeRef ? symbolTable.lookup<mlir::func::FuncOp>(calleeRef.getValue())
                : nullptr;
  if (!toApplyFunc) {
    customCall.emitOpError("unknown PartialReduce computation");
    return std::nullopt;
  }
  if (mlir::failed(validateToApplyFunction(customCall, toApplyFunc, sig,
                                           config->toApplyKind))) {
    return std::nullopt;
  }

  mlir::sdy::TensorShardingAttr firstInSharding =
      mlir::sdy::getSharding(customCall.getOperand(0));
  mlir::sdy::MeshAttr mesh = firstInSharding.getMesh(symbolTable);
  const int64_t databaseShards =
      firstInSharding.getDimSharding(sig.reductionDim).getShardedSize(mesh);
  const int64_t databaseSizePerShard =
      llvm::divideCeil(sig.databaseSize, databaseShards);

  // Pass global `sig.databaseSize` as `input_size_override` so Stage 1's
  // per-shard reduction factor preserves the global `recall_target` across all
  // shards (matching GSPMD's `PartialReduce` partitioner).
  absl::StatusOr<std::pair<int64_t, int64_t>> stageOneShapeOr =
      xla::ApproxTopKReductionOutputSize(
          databaseSizePerShard, sig.rank, config->topK,
          static_cast<float>(config->recallTarget),
          /*aggregate_to_topk=*/false,
          /*input_size_override=*/sig.databaseSize);
  if (!stageOneShapeOr.ok()) {
    customCall.emitOpError("ApproxTopKReductionOutputSize failed: ")
        << stageOneShapeOr.status().message();
    return std::nullopt;
  }

  return PartialReduceLoweringContext{
      /*customCall=*/customCall,
      /*loc=*/customCall.getLoc(),
      /*sig=*/sig,
      /*config=*/std::move(*config),
      /*toApplyFunc=*/toApplyFunc,
      /*mesh=*/mesh,
      /*firstInSharding=*/firstInSharding,
      /*reductionDimReplicatedSharding=*/
      firstInSharding.replaceDimSharding(
          sig.reductionDim,
          mlir::sdy::DimensionShardingAttr::get(customCall.getContext(), {},
                                                /*is_closed=*/true)),
      /*manualAxes=*/std::move(manualAxes),
      /*databaseShards=*/databaseShards,
      /*databaseSizePerShard=*/databaseSizePerShard,
      /*stageOneOutSize=*/stageOneShapeOr->first,
      /*stageOneLog2Reduction=*/stageOneShapeOr->second,
  };
}

// Aligns all `K` data operands to `ctx.firstInSharding` and high-pads any
// shard-indivisible dimensions with each operand's scalar `init_i` identity.
llvm::SmallVector<mlir::Value> alignAndPadDataOperands(
    const PartialReduceLoweringContext& ctx,
    const mlir::SymbolTable& symbolTable, mlir::IRRewriter& rewriter) {
  mlir::stablehlo::CustomCallOp customCall = ctx.customCall;
  llvm::SmallVector<mlir::Value> paddedVals;
  paddedVals.reserve(ctx.sig.numInputs);
  for (int64_t i = 0; i < ctx.sig.numInputs; ++i) {
    mlir::Value val =
        reshardIfNeeded(ctx.loc, customCall.getOperand(i), ctx.sig.rank,
                        ctx.firstInSharding, rewriter);
    auto paddedInType =
        llvm::cast<mlir::RankedTensorType>(mlir::sdy::getDivisiblePaddedType(
            val.getType(), ctx.firstInSharding, symbolTable));
    paddedVals.push_back(mlir::sdy::padHighSideToType(
        ctx.loc, val, paddedInType, ctx.firstInSharding, rewriter,
        /*paddingValue=*/ctx.sig.initVals[i]));
  }
  return paddedVals;
}

// Rewrites a cloned Stage 1 `@PartialReduce` op for shard-local execution
// inside `sdy.manual_computation`, scrubbing global shape/sharding attributes
// and serializing the Stage 1 JSON `backend_config`.
void configureLocalStageOneCall(mlir::stablehlo::CustomCallOp stageOneCall,
                                llvm::ArrayRef<mlir::Type> localOutTypes,
                                const PartialReduceConfig& config,
                                int64_t stageOneLog2Reduction,
                                mlir::IRRewriter& rewriter) {
  for (int64_t i = 0; i < static_cast<int64_t>(localOutTypes.size()); ++i) {
    stageOneCall.getResult(i).setType(localOutTypes[i]);
  }
  stageOneCall->removeAttr(xla::kXlaShape);
  stageOneCall->removeAttr(kMhloBackendConfigAttr);
  stageOneCall->removeAttr(mlir::sdy::kShardingAttr);
  stageOneCall->removeAttr(mlir::sdy::kShardingRuleAttr);

  // TPU `PartialReduce` expects a JSON string `backend_config`, whereas typed
  // FFI custom calls require a `DictionaryAttr`; downgrade typed FFI to
  // `API_VERSION_ORIGINAL` before attaching the Stage 1 JSON string.
  if (stageOneCall.getApiVersion() ==
      mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI) {
    stageOneCall.setApiVersion(
        mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL);
  }
  stageOneCall.setBackendConfigAttr(rewriter.getStringAttr(
      formatStageOneBackendConfig(config, stageOneLog2Reduction)));
}

// Emits Stage 1 per-shard `@PartialReduce` inside `sdy.manual_computation`
// (when `ctx.runsStageOnePartialReduce()`) followed by `sdy.reshard` to
// replicate `reduction_dim` across shards.
llvm::SmallVector<mlir::Value> emitStageOneAndAllGather(
    const PartialReduceLoweringContext& ctx,
    llvm::ArrayRef<mlir::Value> paddedInputs, mlir::IRRewriter& rewriter) {
  const int64_t numInputs = ctx.sig.numInputs;
  llvm::SmallVector<mlir::Value> stageOneVals(paddedInputs);

  if (ctx.runsStageOnePartialReduce()) {
    mlir::sdy::TensorShardingAttr scalarRepSharding =
        mlir::sdy::TensorShardingAttr::getFullyReplicated(
            rewriter.getContext(), /*rank=*/0,
            ctx.firstInSharding.getMeshOrRef(),
            /*isClosed=*/true);
    llvm::SmallVector<mlir::Value> manualOperands(paddedInputs);
    llvm::append_range(manualOperands, ctx.sig.initVals);
    llvm::SmallVector<mlir::sdy::TensorShardingAttr> manualInShardings(
        numInputs, ctx.firstInSharding);
    manualInShardings.append(numInputs, scalarRepSharding);
    llvm::SmallVector<mlir::sdy::TensorShardingAttr> manualOutShardings(
        numInputs, ctx.firstInSharding);

    llvm::SmallVector<mlir::Type> manualGlobalOutTypes;
    llvm::SmallVector<mlir::Type> localOutTypes;
    manualGlobalOutTypes.reserve(numInputs);
    localOutTypes.reserve(numInputs);
    for (mlir::Value paddedVal : paddedInputs) {
      auto paddedType = llvm::cast<mlir::RankedTensorType>(paddedVal.getType());
      mlir::RankedTensorType globalOutType =
          withReplacedDimSize(paddedType, ctx.sig.reductionDim,
                              ctx.databaseShards * ctx.stageOneOutSize);
      manualGlobalOutTypes.push_back(globalOutType);
      localOutTypes.push_back(ctx.firstInSharding.getLocalType(
          globalOutType, ctx.mesh, /*allowNonDivisible=*/false));
    }

    auto manualComp = mlir::sdy::ManualComputationOp::create(
        rewriter, ctx.loc, manualGlobalOutTypes, manualOperands,
        manualInShardings, manualOutShardings,
        llvm::ArrayRef<mlir::sdy::AxisRefAttr>(ctx.manualAxes));
    mlir::Block& bodyBlock = manualComp.getBody().emplaceBlock();
    for (mlir::Value paddedVal : paddedInputs) {
      bodyBlock.addArgument(
          ctx.firstInSharding.getLocalType(
              llvm::cast<mlir::RankedTensorType>(paddedVal.getType()), ctx.mesh,
              /*allowNonDivisible=*/false),
          ctx.loc);
    }
    for (mlir::Value initVal : ctx.sig.initVals) {
      bodyBlock.addArgument(initVal.getType(), ctx.loc);
    }
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&bodyBlock);
      mlir::stablehlo::CustomCallOp origCall = ctx.customCall;
      auto stageOneCall = llvm::cast<mlir::stablehlo::CustomCallOp>(
          rewriter.clone(*origCall.getOperation()));
      stageOneCall->setOperands(bodyBlock.getArguments());
      configureLocalStageOneCall(stageOneCall, localOutTypes, ctx.config,
                                 ctx.stageOneLog2Reduction, rewriter);
      mlir::sdy::ReturnOp::create(rewriter, ctx.loc, stageOneCall.getResults());
    }
    stageOneVals.assign(manualComp.result_begin(), manualComp.result_end());
  }

  llvm::SmallVector<mlir::Value> gatheredVals;
  gatheredVals.reserve(numInputs);
  for (mlir::Value val : stageOneVals) {
    gatheredVals.push_back(
        mlir::sdy::ReshardOp::create(rewriter, ctx.loc, val.getType(), val,
                                     ctx.reductionDimReplicatedSharding));
  }
  return gatheredVals;
}

// Emits `stablehlo.select` across all `ctx.sig.numInputs` pairs using a
// `"comparator"` computation's scalar `tensor<i1>` predicate and terminates
// `reduceBlock` (`validateToApplyFunction` has already verified the signature).
void emitComparatorSelectReturn(const PartialReduceLoweringContext& ctx,
                                mlir::Block& reduceBlock,
                                mlir::Operation* terminator,
                                const mlir::IRMapping& mapping,
                                mlir::IRRewriter& rewriter) {
  const int64_t numInputs = ctx.sig.numInputs;
  mlir::Value pred = mapping.lookupOrDefault(terminator->getOperand(0));
  llvm::SmallVector<mlir::Value> selectedVals;
  selectedVals.reserve(numInputs);
  for (int64_t i = 0; i < numInputs; ++i) {
    mlir::Value lhs = reduceBlock.getArgument(i);
    mlir::Value rhs = reduceBlock.getArgument(numInputs + i);
    selectedVals.push_back(mlir::stablehlo::SelectOp::create(
        rewriter, ctx.loc, lhs.getType(), pred, lhs, rhs));
  }
  mlir::stablehlo::ReturnOp::create(rewriter, ctx.loc, selectedVals);
}

// Unpacks a `"combiner"` computation's `ctx.sig.numInputs` scalar return values
// (either variadic scalars or an HLO round-trip `tuple`) and terminates
// `reduceBlock` (`validateToApplyFunction` has already verified the signature).
void emitCombinerReturn(const PartialReduceLoweringContext& ctx,
                        mlir::Block& reduceBlock, mlir::Operation* terminator,
                        const mlir::IRMapping& mapping,
                        mlir::IRRewriter& rewriter) {
  const int64_t numInputs = ctx.sig.numInputs;
  llvm::SmallVector<mlir::Value> reducedVals;
  reducedVals.reserve(numInputs);
  if (static_cast<int64_t>(terminator->getNumOperands()) == numInputs &&
      !llvm::isa<mlir::TupleType>(terminator->getOperand(0).getType())) {
    for (mlir::Value retOperand : terminator->getOperands()) {
      reducedVals.push_back(mapping.lookupOrDefault(retOperand));
    }
  } else {
    // Tuple-returning combiners wrap scalar results in a single `tuple`;
    // unwrap `stablehlo.tuple` directly or extract tuple elements.
    mlir::Value clonedTuple =
        mapping.lookupOrDefault(terminator->getOperand(0));
    if (auto tupleOp = clonedTuple.getDefiningOp<mlir::stablehlo::TupleOp>()) {
      llvm::append_range(reducedVals, tupleOp.getVal());
      if (tupleOp->use_empty()) {
        rewriter.eraseOp(tupleOp);
      }
    } else {
      for (int64_t i = 0; i < numInputs; ++i) {
        reducedVals.push_back(mlir::stablehlo::GetTupleElementOp::create(
            rewriter, ctx.loc, clonedTuple, i));
      }
    }
  }
  mlir::stablehlo::ReturnOp::create(rewriter, ctx.loc, reducedVals);
}

// Populates `reduceRegion` for the Stage 2 fold `stablehlo.reduce` by cloning
// `ctx.toApplyFunc` and adapting interleaved `"comparator"` vs grouped
// `"combiner"` parameter conventions.
void buildStageTwoFoldReduceBody(const PartialReduceLoweringContext& ctx,
                                 mlir::Region& reduceRegion,
                                 mlir::IRRewriter& rewriter) {
  mlir::func::FuncOp toApplyFunc = ctx.toApplyFunc;
  mlir::Block& toApplyBlock = toApplyFunc.getBody().front();
  const int64_t numInputs = ctx.sig.numInputs;

  // `stablehlo.reduce` orders block arguments as
  // `(lhs_0..lhs_{K-1}, rhs_0..rhs_{K-1})`.
  mlir::Block& reduceBlock = reduceRegion.emplaceBlock();
  for (int side = 0; side < 2; ++side) {
    for (mlir::Type elemType : ctx.sig.elemTypes) {
      reduceBlock.addArgument(mlir::RankedTensorType::get({}, elemType),
                              ctx.loc);
    }
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&reduceBlock);
  mlir::IRMapping mapping;
  const bool isComparator = ctx.config.toApplyKind == ToApplyKind::kComparator;
  for (int64_t i = 0; i < numInputs; ++i) {
    // Map interleaved comparator args `(lhs_i, rhs_i)` at `(2*i, 2*i+1)` vs
    // grouped combiner args at `(i, K+i)`.
    mapping.map(toApplyBlock.getArgument(isComparator ? 2 * i : i),
                reduceBlock.getArgument(i));
    mapping.map(
        toApplyBlock.getArgument(isComparator ? 2 * i + 1 : numInputs + i),
        reduceBlock.getArgument(numInputs + i));
  }

  mlir::Operation* terminator = toApplyBlock.getTerminator();
  for (mlir::Operation& op : toApplyBlock.without_terminator()) {
    rewriter.clone(op, mapping);
  }

  if (isComparator) {
    emitComparatorSelectReturn(ctx, reduceBlock, terminator, mapping, rewriter);
  } else {
    emitCombinerReturn(ctx, reduceBlock, terminator, mapping, rewriter);
  }
}

// High-pads gathered candidates to `numFolds * outputSize` along
// `reduction_dim`, reshapes into `(numFolds, outputSize)` when `numFolds > 1`,
// and folds across `numFolds` via `stablehlo.reduce`.
llvm::SmallVector<mlir::Value> emitStageTwoFoldReduce(
    const PartialReduceLoweringContext& ctx,
    llvm::SmallVector<mlir::Value> stageTwoVals, mlir::IRRewriter& rewriter) {
  const int64_t numInputs = ctx.sig.numInputs;
  const int64_t reductionDim = ctx.sig.reductionDim;
  const int64_t outputSize = ctx.sig.outputSize;
  const int64_t gatheredSize = ctx.databaseShards * ctx.localCandidatesSize();
  const int64_t numFolds = llvm::divideCeil(gatheredSize, outputSize);

  if (outputSize * numFolds > gatheredSize) {
    for (int64_t i = 0; i < numInputs; ++i) {
      auto curType =
          llvm::cast<mlir::RankedTensorType>(stageTwoVals[i].getType());
      stageTwoVals[i] = mlir::sdy::padHighSideToType(
          ctx.loc, stageTwoVals[i],
          withReplacedDimSize(curType, reductionDim, outputSize * numFolds),
          ctx.reductionDimReplicatedSharding, rewriter,
          /*paddingValue=*/ctx.sig.initVals[i]);
    }
  }

  if (numFolds <= 1) {
    return stageTwoVals;
  }

  mlir::MLIRContext* mlirCtx = rewriter.getContext();
  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> reshapedDimShardings(
      ctx.reductionDimReplicatedSharding.getDimShardings());
  reshapedDimShardings.insert(
      reshapedDimShardings.begin() + reductionDim,
      mlir::sdy::DimensionShardingAttr::get(mlirCtx, {}, /*is_closed=*/true));
  auto reshapedSharding = mlir::sdy::TensorShardingAttr::get(
      mlirCtx, ctx.reductionDimReplicatedSharding.getMeshOrRef(),
      reshapedDimShardings,
      ctx.reductionDimReplicatedSharding.getReplicatedAxes(),
      ctx.reductionDimReplicatedSharding.getUnreducedAxes(),
      ctx.reductionDimReplicatedSharding.getReductionOp());

  llvm::SmallVector<mlir::Value> reshapedVals;
  llvm::SmallVector<mlir::Type> stageTwoOutTypes;
  reshapedVals.reserve(numInputs);
  stageTwoOutTypes.reserve(numInputs);
  for (int64_t i = 0; i < numInputs; ++i) {
    auto curType =
        llvm::cast<mlir::RankedTensorType>(stageTwoVals[i].getType());
    llvm::SmallVector<int64_t> reshapedShape(curType.getShape());
    reshapedShape[reductionDim] = outputSize;
    reshapedShape.insert(reshapedShape.begin() + reductionDim, numFolds);
    auto reshapedOp = mlir::stablehlo::ReshapeOp::create(
        rewriter, ctx.loc,
        mlir::RankedTensorType::get(reshapedShape, ctx.sig.elemTypes[i]),
        stageTwoVals[i]);
    mlir::sdy::setSharding(reshapedOp.getResult(), reshapedSharding);
    reshapedVals.push_back(reshapedOp.getResult());
    stageTwoOutTypes.push_back(
        withReplacedDimSize(curType, reductionDim, outputSize));
  }

  auto reduceOp = mlir::stablehlo::ReduceOp::create(
      rewriter, ctx.loc, stageTwoOutTypes, reshapedVals, ctx.sig.initVals,
      rewriter.getDenseI64ArrayAttr({reductionDim}));
  mlir::sdy::setShardings(reduceOp,
                          llvm::SmallVector<mlir::sdy::TensorShardingAttr>(
                              numInputs, ctx.reductionDimReplicatedSharding));
  buildStageTwoFoldReduceBody(ctx, reduceOp.getBody(), rewriter);

  stageTwoVals.assign(reduceOp.result_begin(), reduceOp.result_end());
  return stageTwoVals;
}

// Slices off any high-side batch divisibility padding added before Stage 1 and
// reshards each result to match the original output sharding.
llvm::SmallVector<mlir::Value> unpadAndReshardResults(
    const PartialReduceLoweringContext& ctx,
    llvm::ArrayRef<mlir::Value> reducedVals, mlir::IRRewriter& rewriter) {
  mlir::stablehlo::CustomCallOp customCall = ctx.customCall;
  llvm::SmallVector<mlir::Value> finalResults;
  finalResults.reserve(ctx.sig.numInputs);
  for (int64_t i = 0; i < ctx.sig.numInputs; ++i) {
    mlir::Value val = reducedVals[i];
    mlir::Value origResult = customCall.getResult(i);
    if (val.getType() != origResult.getType()) {
      val = mlir::sdy::sliceHighSideToType(ctx.loc, val, origResult.getType(),
                                           ctx.reductionDimReplicatedSharding,
                                           rewriter);
    }
    val = reshardIfNeeded(ctx.loc, val, ctx.sig.rank,
                          mlir::sdy::getSharding(origResult), rewriter);
    finalResults.push_back(val);
  }
  return finalResults;
}

}  // namespace

std::optional<mlir::LogicalResult> resolvePartialReduce(
    mlir::stablehlo::CustomCallOp customCall,
    mlir::sdy::OpShardingRuleAttr shardingRule, mlir::IRRewriter& rewriter,
    const mlir::SymbolTable& symbolTable) {
  std::optional<int64_t> reductionDim =
      findPermutationReductionDim(shardingRule);
  if (!reductionDim) {
    return std::nullopt;
  }

  std::optional<PartialReduceSignature> sig =
      validatePartialReduceForLowering(customCall, *reductionDim);
  if (!sig) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::sdy::AxisRefAttr> manualAxes;
  if (tryResolveAsSingleStage(customCall, *sig, symbolTable, rewriter,
                              manualAxes)) {
    return mlir::success();
  }

  std::optional<PartialReduceLoweringContext> ctx =
      buildPartialReduceLoweringContext(customCall, *sig, std::move(manualAxes),
                                        symbolTable);
  if (!ctx) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> paddedInputs =
      alignAndPadDataOperands(*ctx, symbolTable, rewriter);
  llvm::SmallVector<mlir::Value> gatheredVals =
      emitStageOneAndAllGather(*ctx, paddedInputs, rewriter);
  llvm::SmallVector<mlir::Value> reducedVals =
      emitStageTwoFoldReduce(*ctx, std::move(gatheredVals), rewriter);
  llvm::SmallVector<mlir::Value> finalResults =
      unpadAndReshardResults(*ctx, reducedVals, rewriter);
  rewriter.replaceOp(customCall, finalResults);
  return mlir::success();
}

}  // namespace sdy
}  // namespace xla
