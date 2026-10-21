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

#include "xla/service/spmd/shardy/extensions/mhlo_extensions.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mhlo = ::mlir::mhlo;

namespace xla {
namespace sdy {

namespace {

using ::mlir::ArrayRef;
using ::mlir::sdy::FactorType;
using ::mlir::sdy::getTensorShape;
using ::mlir::sdy::kNullDim;
using ::mlir::sdy::OpShardingRuleBuilder;

template <typename OpTy>
struct PointwiseShardingRuleOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          PointwiseShardingRuleOpInterface<OpTy>, OpTy> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation* op) const {
    return OpShardingRuleBuilder::buildPointwise(op);
  }
};

template <typename... OpTys>
void attachPointwiseShardingRule(mlir::MLIRContext* ctx) {
  (OpTys::template attachInterface<PointwiseShardingRuleOpInterface<OpTys>>(
       *ctx),
   ...);
}

enum RaggedDotMode {
  // Ragged non-contracting (m): [b,m,k], [g,b,k,n], [b,g] -> [b,m,n].
  kNonContracting,
  // Ragged contracting (k):     [b,m,k], [b,k,n],   [b,g] -> [g,b,m,n].
  kContracting,
  // Ragged batch (b):           [b,m,k], [b,k,n],   [g]   -> [b,m,n].
  kBatch,
};

struct RaggedDotShardingRuleOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          RaggedDotShardingRuleOpInterface, mhlo::RaggedDotOp> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation* op) const {
    mhlo::RaggedDotOp raggedDot = llvm::cast<mhlo::RaggedDotOp>(op);
    mhlo::RaggedDotDimensionNumbersAttr raggedDotDimNumbers =
        raggedDot.getRaggedDotDimensionNumbers();
    mhlo::DotDimensionNumbersAttr dotDimNumbers =
        raggedDotDimNumbers.getDotDimensionNumbers();

    ArrayRef<int64_t> lhsBatchingDims =
        dotDimNumbers.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchingDims =
        dotDimNumbers.getRhsBatchingDimensions();
    ArrayRef<int64_t> lhsContractingDims =
        dotDimNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dotDimNumbers.getRhsContractingDimensions();

    ArrayRef<int64_t> lhsRaggedDims =
        raggedDotDimNumbers.getLhsRaggedDimensions();
    CHECK_EQ(lhsRaggedDims.size(), 1);
    int64_t lhsRaggedDim = lhsRaggedDims.front();
    ArrayRef<int64_t> rhsGroupDims =
        raggedDotDimNumbers.getRhsGroupDimensions();

    RaggedDotMode mode;
    if (llvm::is_contained(lhsContractingDims, lhsRaggedDim)) {
      CHECK(rhsGroupDims.empty());
      mode = RaggedDotMode::kContracting;
    } else if (llvm::is_contained(lhsBatchingDims, lhsRaggedDim)) {
      CHECK(rhsGroupDims.empty());
      mode = RaggedDotMode::kBatch;
    } else {
      CHECK_EQ(rhsGroupDims.size(), 1);
      mode = RaggedDotMode::kNonContracting;
    }

    OpShardingRuleBuilder builder(raggedDot);

    mlir::RankedTensorType lhsType = raggedDot.getLhs().getType();
    mlir::RankedTensorType rhsType = raggedDot.getRhs().getType();
    mlir::RankedTensorType groupSizesType = raggedDot.getGroupSizes().getType();

    int64_t groupSizesDim = 0;
    int64_t outputDim = (mode == RaggedDotMode::kContracting) ? 1 : 0;

    // batching dimensions
    for (auto [lhsDim, rhsDim] :
         llvm::zip_equal(lhsBatchingDims, rhsBatchingDims)) {
      builder.addFactor(
          {lhsDim, rhsDim,
           mode != RaggedDotMode::kBatch ? groupSizesDim++ : kNullDim},
          outputDim++, lhsType.getDimSize(lhsDim));
    }

    // lhs non-contracting dimensions
    bool addLhsNonContractingDimsInGroupSizes =
        mode == RaggedDotMode::kNonContracting;
    for (int64_t i = 0; i < lhsType.getRank(); i++) {
      if (!llvm::is_contained(lhsContractingDims, i) &&
          !llvm::is_contained(lhsBatchingDims, i)) {
        FactorType factorType = FactorType::kPassThrough;
        if (i == lhsRaggedDim) {
          // We only add the non-contracting dimensions before the lhs ragged
          // dimension to the group sizes in the kNonContracting mode.
          addLhsNonContractingDimsInGroupSizes = false;
          factorType = FactorType::kNeedReplication;
        }
        builder.addFactor(
            {i, kNullDim,
             addLhsNonContractingDimsInGroupSizes ? groupSizesDim++ : kNullDim},
            outputDim++, lhsType.getDimSize(i), factorType);
      }
    }

    // rhs non-contracting dimensions
    for (int64_t i = 0; i < rhsType.getRank(); i++) {
      if (!llvm::is_contained(rhsContractingDims, i) &&
          !llvm::is_contained(rhsBatchingDims, i) &&
          !llvm::is_contained(rhsGroupDims, i)) {
        builder.addFactor({kNullDim, i, kNullDim}, outputDim++,
                          rhsType.getDimSize(i));
      }
    }

    // contracting dimensions
    bool addContractingDimsInGroupSizes = mode == RaggedDotMode::kContracting;
    for (auto [lhsDim, rhsDim] :
         llvm::zip_equal(lhsContractingDims, rhsContractingDims)) {
      FactorType factorType = FactorType::kReduction;
      if (lhsDim == lhsRaggedDim) {
        // We only add the contracting dimensions before the lhs ragged
        // dimension to the group sizes in the kContracting mode.
        addContractingDimsInGroupSizes = false;
        factorType = FactorType::kNeedReplication;
      }
      builder.addFactor(
          {lhsDim, rhsDim,
           addContractingDimsInGroupSizes ? groupSizesDim++ : kNullDim},
          kNullDim, lhsType.getDimSize(lhsDim), factorType);
    }

    switch (mode) {
      case RaggedDotMode::kNonContracting: {
        CHECK_EQ(groupSizesDim, groupSizesType.getRank() - 1);
        int64_t rhsGroupDim = rhsGroupDims.front();
        builder.addFactor({kNullDim, rhsGroupDim, groupSizesDim}, kNullDim,
                          rhsType.getDimSize(rhsGroupDim),
                          FactorType::kNeedReplication);
        break;
      }
      case RaggedDotMode::kContracting: {
        CHECK_EQ(groupSizesDim, groupSizesType.getRank() - 1);
        builder.addFactor({kNullDim, kNullDim, groupSizesDim}, 0,
                          groupSizesType.getDimSize(groupSizesDim),
                          FactorType::kNeedReplication);
        break;
      }
      case RaggedDotMode::kBatch: {
        break;
      }
    }

    return builder.build();
  }
};

struct TopKShardingRuleOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          TopKShardingRuleOpInterface, mhlo::TopKOp> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation* op) const {
    mhlo::TopKOp topK = llvm::cast<mhlo::TopKOp>(op);
    return OpShardingRuleBuilder(topK)
        .addPointwiseWithDiffTypeForMismatch(getTensorShape(topK.getOperand()),
                                             getTensorShape(topK.getResult(0)),
                                             FactorType::kNeedReplication,
                                             /*mismatchFactorIsBlocked=*/true)
        .build();
  }
};

// Returns true if the scan body applies its combiner pointwise: it only
// contains elementwise ops, scalar constants, and broadcasts of scalars
// (plus the terminator). Pointwise bodies are the ones whose non-scan
// dimensions are independent across iterations, so those dimensions can be
// sharded (the body can be evaluated on any slice of them).
bool isElementwiseScanBody(mhlo::ScanOp scan) {
  if (scan.getBody().empty()) {
    return false;
  }
  mlir::Block& bodyBlock = scan.getBody().front();
  for (mlir::Operation& bodyOp : bodyBlock.without_terminator()) {
    if (bodyOp.hasTrait<mlir::OpTrait::Elementwise>()) {
      continue;
    }
    if (bodyOp.hasTrait<mlir::OpTrait::ConstantLike>()) {
      auto type =
          llvm::dyn_cast<mlir::RankedTensorType>(bodyOp.getResult(0).getType());
      if (!type || type.getRank() != 0) {
        return false;
      }
      continue;
    }
    // Select applies pointwise when every operand (including the predicate)
    // has the result shape; MLIR marks it with a broadcasting trait instead
    // of Elementwise, but the partitioner's body analysis accepts the HLO
    // form, so accept it here too.
    if (llvm::isa<mhlo::SelectOp, mlir::stablehlo::SelectOp>(bodyOp)) {
      auto resultType =
          llvm::dyn_cast<mlir::RankedTensorType>(bodyOp.getResult(0).getType());
      if (!resultType) {
        return false;
      }
      bool pointwise =
          llvm::all_of(bodyOp.getOperands(), [&](mlir::Value operand) {
            auto operandType =
                llvm::dyn_cast<mlir::RankedTensorType>(operand.getType());
            return operandType &&
                   operandType.getShape() == resultType.getShape();
          });
      if (!pointwise) {
        return false;
      }
      continue;
    }
    // Recomposed scan bodies hold stablehlo ops; handle both dialects. A
    // broadcast of a scalar is rebuildable for per-shard shapes.
    if (llvm::isa<mhlo::BroadcastInDimOp, mlir::stablehlo::BroadcastInDimOp>(
            bodyOp)) {
      auto operandType = llvm::dyn_cast<mlir::RankedTensorType>(
          bodyOp.getOperand(0).getType());
      if (operandType && operandType.getRank() == 0) {
        continue;
      }
    }
    return false;
  }
  return true;
}

// Returns `type` with dimension `dim` removed.
mlir::RankedTensorType eraseDim(mlir::RankedTensorType type, int64_t dim) {
  mlir::SmallVector<int64_t> shape(type.getShape());
  shape.erase(shape.begin() + dim);
  return mlir::RankedTensorType::get(shape, type.getElementType());
}

// Detect the "shape-preserving" pattern produced by chlo.ScanOp/JAX. In that
// pattern the per-iteration body output has the same shape as the
// per-iteration body input, so each input[i] has a corresponding output of
// identical shape, and each init[j] has shape equal to input[j] with the scan
// dim removed.
bool isShapePreservingScan(mhlo::ScanOp scan) {
  int64_t scanDim = scan.getDimension();
  mlir::ValueRange inputs = scan.getInputs();
  mlir::ValueRange inits = scan.getInits();
  mlir::ValueRange outputs = scan.getOutputs();
  int64_t numInputs = inputs.size();
  int64_t numInits = inits.size();
  if (numInputs == 0) {
    return false;
  }
  auto representative =
      llvm::dyn_cast<mlir::RankedTensorType>(inputs.front().getType());
  if (!representative || scanDim < 0 || scanDim >= representative.getRank()) {
    return false;
  }
  if (numInputs != outputs.size() || numInputs != numInits) {
    return false;
  }
  for (int64_t i = 0; i < numInputs; ++i) {
    auto inT = llvm::dyn_cast<mlir::RankedTensorType>(inputs[i].getType());
    auto outT = llvm::dyn_cast<mlir::RankedTensorType>(outputs[i].getType());
    // The mapping loop below indexes every input and init with the
    // representative's dimensions, so all inputs must share the
    // representative's shape, not just match their own output.
    if (!inT || !outT || inT.getShape() != outT.getShape() ||
        inT.getShape() != representative.getShape()) {
      return false;
    }
  }
  for (int64_t j = 0; j < numInits; ++j) {
    auto initT = llvm::dyn_cast<mlir::RankedTensorType>(inits[j].getType());
    auto inT = llvm::dyn_cast<mlir::RankedTensorType>(inputs[j].getType());
    if (!initT || !inT || inT.getRank() != initT.getRank() + 1) {
      return false;
    }
    for (int64_t d = 0, m = 0; d < inT.getRank(); ++d) {
      if (d == scanDim) {
        continue;
      }
      if (inT.getDimSize(d) != initT.getDimSize(m++)) {
        return false;
      }
    }
  }
  return true;
}

// Returns true if `sharding` shards dimension `dim`.
bool isDimSharded(mlir::sdy::TensorShardingAttr sharding, int64_t dim) {
  return sharding && dim < sharding.getRank() &&
         !sharding.getDimSharding(dim).emptyAxes();
}

// Inserts a dimension of size `dimSize` at `scanDim` into `baseCarryType`.
mlir::RankedTensorType insertDimOfSize(mlir::RankedTensorType baseCarryType,
                                       int64_t scanDim, int64_t dimSize) {
  mlir::SmallVector<int64_t> shape(baseCarryType.getShape());
  shape.insert(shape.begin() + scanDim, dimSize);
  return mlir::RankedTensorType::get(shape, baseCarryType.getElementType());
}

// Replaces dimension `dim` of `type` with `dimSize`.
mlir::RankedTensorType withDimSize(mlir::Type type, int64_t dim,
                                   int64_t dimSize) {
  auto rankedType = llvm::cast<mlir::RankedTensorType>(type);
  mlir::SmallVector<int64_t> shape(rankedType.getShape());
  shape[dim] = dimSize;
  return mlir::RankedTensorType::get(shape, rankedType.getElementType());
}

// Reshapes a rank-(R-1) carry value into a rank-R slice of size 1 along
// `scanDim` and attaches `sliceSharding`.
mlir::Value expandCarryToSingleSlice(
    mlir::IRRewriter& rewriter, mlir::Location loc, mlir::Value carryValue,
    int64_t scanDim, mlir::sdy::TensorShardingAttr sliceSharding) {
  auto carryType = llvm::cast<mlir::RankedTensorType>(carryValue.getType());
  mlir::RankedTensorType singleSliceType =
      insertDimOfSize(carryType, scanDim, 1);
  auto reshapeOp = mlir::stablehlo::ReshapeOp::create(
      rewriter, loc, singleSliceType, carryValue);
  mlir::sdy::setShardings(reshapeOp, sliceSharding);
  return reshapeOp.getResult();
}

// Reshapes a rank-R slice of size 1 along `scanDim` into a rank-(R-1) carry
// value and attaches `carrySharding`.
mlir::Value collapseSingleSliceToCarry(
    mlir::IRRewriter& rewriter, mlir::Location loc, mlir::Value sliceValue,
    int64_t scanDim, mlir::sdy::TensorShardingAttr carrySharding) {
  auto sliceType = llvm::cast<mlir::RankedTensorType>(sliceValue.getType());
  mlir::RankedTensorType carryType = eraseDim(sliceType, scanDim);
  auto reshapeOp =
      mlir::stablehlo::ReshapeOp::create(rewriter, loc, carryType, sliceValue);
  mlir::sdy::setShardings(reshapeOp, carrySharding);
  return reshapeOp.getResult();
}

// Clones `scan` with updated operands and result shardings, inferring output
// types from `inputs`, carry types from `inits`, and `scan_dim_size` from
// `inputs[0]`.
mhlo::ScanOp createScanWithClonedBody(
    mhlo::ScanOp scan, mlir::ValueRange inputs, mlir::ValueRange inits,
    ArrayRef<mlir::sdy::TensorShardingAttr> outputShardings,
    ArrayRef<mlir::sdy::TensorShardingAttr> carryShardings,
    mlir::IRRewriter& rewriter) {
  mlir::SmallVector<mlir::Value> operands;
  operands.reserve(inputs.size() + inits.size());
  operands.append(inputs.begin(), inputs.end());
  operands.append(inits.begin(), inits.end());

  mlir::NamedAttrList discardableAttrs(scan->getDiscardableAttrDictionary());
  discardableAttrs.erase(mlir::sdy::kShardingRuleAttr);
  discardableAttrs.erase(mlir::sdy::kShardingAttr);

  auto newScan = mhlo::ScanOp::create(
      rewriter, scan.getLoc(), mlir::ValueRange(operands).getTypes(), operands,
      scan.getProperties(), discardableAttrs.getAttrs());
  if (scan.getScanDimSize().has_value()) {
    auto in0Type = llvm::cast<mlir::RankedTensorType>(inputs[0].getType());
    newScan.setScanDimSize(in0Type.getDimSize(scan.getDimension()));
  }
  rewriter.cloneRegionBefore(scan.getBody(), newScan.getBody(),
                             newScan.getBody().end());

  mlir::SmallVector<mlir::sdy::TensorShardingAttr> resultShardings;
  resultShardings.reserve(outputShardings.size() + carryShardings.size());
  resultShardings.append(outputShardings.begin(), outputShardings.end());
  resultShardings.append(carryShardings.begin(), carryShardings.end());
  mlir::sdy::setShardings(newScan, resultShardings);
  return newScan;
}

// Slices `input` along `scanDim` in `[startIndex, limitIndex)` with `sharding`.
mlir::Value createSliceAlongDim(mlir::IRRewriter& rewriter, mlir::Location loc,
                                mlir::Value input, int64_t scanDim,
                                int64_t startIndex, int64_t limitIndex,
                                mlir::sdy::TensorShardingAttr sharding) {
  auto inputType = llvm::cast<mlir::RankedTensorType>(input.getType());
  int64_t rank = inputType.getRank();
  mlir::SmallVector<int64_t> starts(rank, 0);
  mlir::SmallVector<int64_t> limits(inputType.getShape());
  mlir::SmallVector<int64_t> strides(rank, 1);
  starts[scanDim] = startIndex;
  limits[scanDim] = limitIndex;

  mlir::SmallVector<int64_t> outShape(inputType.getShape());
  outShape[scanDim] = limitIndex - startIndex;
  auto outType =
      mlir::RankedTensorType::get(outShape, inputType.getElementType());
  auto sliceOp = mlir::stablehlo::SliceOp::create(
      rewriter, loc, outType, input, rewriter.getDenseI64ArrayAttr(starts),
      rewriter.getDenseI64ArrayAttr(limits),
      rewriter.getDenseI64ArrayAttr(strides));
  mlir::sdy::setShardings(sliceOp, sharding);
  return sliceOp.getResult();
}

// Sharding rule for `mhlo.scan`.
//
// `mhlo.scan` has variadic operands `(inputs..., inits...)` and variadic
// results `(outputs..., carries...)`. In the general case the body may
// produce per-iteration outputs of a different shape than its per-iteration
// inputs (the result types are inferred from the body terminator), so
// `outputs[i].shape` is not guaranteed to equal `inputs[i].shape` and the
// init/carry shapes need not relate to the input shapes at all.
//
// In practice (chlo.ScanOp from JAX, etc.) the body is shape-preserving:
//   * #inputs == #outputs == #inits == #carries,
//   * `inputs[i].shape == outputs[i].shape`,
//   * `inits[j].shape == inputs[j].shape` with the scan dim removed
//     (so `carries[j].shape` matches as well).
// In that "shape-preserving" pattern, all non-scan dimensions are
// independent across iterations and propagate point-wise between
// input/output/init/carry. The scan dimension's treatment depends on
// `is_associative`:
//   * associative: the SPMD partitioner implements a distributed parallel
//     prefix (local scans per shard plus an inter-shard combine of the
//     carry-sized shard totals). The dimension is permutable
//     (`kPermutation`) so propagation may shard it at the cost of data
//     movement, mirroring how `stablehlo.reduce_window` handles window dims
//     with size > 1.
//   * non-associative or unspecified: the body has a true sequential
//     dependency along the scan dim, so it must be replicated
//     (`kNeedReplication`).
// Sharding the scan dimension only changes how many slices each shard
// processes, never the slice shapes, so it places no constraint on the
// body. Sharding a NON-scan dimension slices every value in the body, so
// those dimensions additionally require a pointwise body (elementwise over
// slices and scalars); for other bodies the non-scan dimensions are marked
// `kNeedReplication` and Shardy inserts the reshards up front.
//
// For shape-changing scans, we fall back to a conservative rule that only
// constrains the scan dim and blocks propagation through every other dim.
struct ScanShardingRuleOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          ScanShardingRuleOpInterface, mhlo::ScanOp> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation* op) const {
    mhlo::ScanOp scan = llvm::cast<mhlo::ScanOp>(op);
    OpShardingRuleBuilder builder(scan);

    // The `mhlo.scan` verifier rejects the both-empty case
    // (`inputs.empty() && outputs.empty()`); defensively bail out with an
    // empty (no-factor) rule for malformed IR rather than crashing on
    // `.front()`.
    mlir::ValueRange inputs = scan.getInputs();
    mlir::ValueRange outputs = scan.getOutputs();
    if (inputs.empty() && outputs.empty()) {
      return builder.build();
    }
    mlir::ValueRange inits = scan.getInits();
    mlir::ValueRange carries = scan.getCarries();
    int64_t scanDim = scan.getDimension();
    int64_t numInputs = inputs.size();
    int64_t numInits = inits.size();
    bool isAssociative = scan.getIsAssociative().value_or(false);

    // Pick a representative input/output tensor to derive the rank and the
    // scan dim size. Bail out with an empty (no-factor) rule if the
    // representative is not a ranked tensor; in well-formed `mhlo.scan` IR
    // it always is.
    auto representative = llvm::dyn_cast<mlir::RankedTensorType>(
        (!inputs.empty() ? inputs.front() : outputs.front()).getType());
    if (!representative) {
      return builder.build();
    }
    if (scanDim < 0 || scanDim >= representative.getRank()) {
      return builder.build();
    }
    int64_t rank = representative.getRank();
    int64_t scanDimSize = representative.getDimSize(scanDim);
    bool isShapePreserving = isShapePreservingScan(scan);

    // The scan dimension can be sharded when the scan is associative (the
    // distributed parallel prefix) and shape preserving; sharding it never
    // changes the slice shapes, so the body needs no rebuild. Sharding a
    // non-scan dimension slices every value in the body, which additionally
    // requires a pointwise body. Reflect that here so propagation does not
    // shard dimensions the partitioner would replicate anyway.
    FactorType scanDimFactorType = isAssociative && isShapePreserving
                                       ? FactorType::kPermutation
                                       : FactorType::kNeedReplication;

    // Layout of the operand/result mapping vectors mirrors the variadic op:
    //   operandDims = [input_0, ..., input_{N-1}, init_0, ..., init_{M-1}]
    //   resultDims  = [output_0, ..., output_{N-1}, carry_0, ..., carry_{M-1}]
    int64_t numOperands = numInputs + numInits;
    int64_t numResults = outputs.size() + carries.size();
    mlir::SmallVector<int64_t> operandDims(numOperands, kNullDim);
    mlir::SmallVector<int64_t> resultDims(numResults, kNullDim);

    if (!isShapePreserving) {
      // Conservative fallback: only model the scan dim factor (which we know
      // is shared across all inputs and outputs at position `scanDim`), and
      // leave every other dim unmapped. Unmapped dims are treated as needing
      // replication by Shardy, which is safe for arbitrary body shapes.
      std::fill_n(operandDims.begin(), numInputs, scanDim);
      std::fill_n(resultDims.begin(), outputs.size(), scanDim);
      builder.addFactor(operandDims, resultDims, scanDimSize,
                        scanDimFactorType);
      return builder.build();
    }

    bool elementwiseBody = isElementwiseScanBody(scan);
    FactorType nonScanDimFactorType = elementwiseBody
                                          ? FactorType::kPassThrough
                                          : FactorType::kNeedReplication;

    // Shape-preserving common case: one factor per input dim, shared across
    // input[i]/output[i] (and init[j]/carry[j] for the non-scan dims).
    int64_t initDim = 0;
    for (int64_t inDim = 0; inDim < rank; ++inDim) {
      int64_t dimSize = representative.getDimSize(inDim);
      int64_t carryDim = (inDim == scanDim) ? kNullDim : initDim++;
      FactorType factorType =
          (inDim == scanDim) ? scanDimFactorType : nonScanDimFactorType;
      std::fill_n(operandDims.begin(), numInputs, inDim);
      std::fill_n(resultDims.begin(), numInputs, inDim);
      std::fill_n(operandDims.begin() + numInputs, numInits, carryDim);
      std::fill_n(resultDims.begin() + numInputs, numInits, carryDim);
      builder.addFactor(operandDims, resultDims, dimSize, factorType);
    }

    return builder.build();
  }
};

// Returns true if the scan dimension is sharded on any output of `scan`.
// The sharding rule only allows shape-preserving scans to partition the scan
// dimension, in which case `numInputs == numOutputs` and the scan dimension is
// sharded the same way in all inputs and outputs.
bool isScanDimSharded(mhlo::ScanOp scan) {
  int64_t scanDim = scan.getDimension();
  return llvm::any_of(scan.getOutputs(), [scanDim](mlir::Value val) {
    return isDimSharded(mlir::sdy::getSharding(val), scanDim);
  });
}

// Updates non-scalar ranked results in `bodyBlock` to `perStepSliceShape`.
void localizeElementwiseBodyToPerStepSliceShape(
    mlir::Block& bodyBlock, ArrayRef<int64_t> perStepSliceShape) {
  for (mlir::Operation& bodyOp : bodyBlock.without_terminator()) {
    if (bodyOp.hasTrait<mlir::OpTrait::ConstantLike>()) {
      continue;
    }
    for (mlir::OpResult res : bodyOp.getResults()) {
      if (auto resType = llvm::dyn_cast<mlir::RankedTensorType>(res.getType());
          resType && resType.getRank() > 0) {
        res.setType(mlir::RankedTensorType::get(perStepSliceShape,
                                                resType.getElementType()));
      }
    }
  }
}

// Rebuilds `scan` with updated `operands`, converted result types, and
// per-step slice shapes in its body.
mlir::LogicalResult rebuildScanWithUpdatedOperandsAndResultTypes(
    mhlo::ScanOp scan, mlir::ValueRange operands,
    const mlir::TypeConverter& typeConverter,
    mlir::ConversionPatternRewriter& rewriter) {
  mlir::SmallVector<mlir::Type> newResultTypes;
  if (mlir::failed(
          typeConverter.convertTypes(scan->getResults(), newResultTypes))) {
    return mlir::failure();
  }

  int64_t numInputs = scan.getInputs().size();
  int64_t scanDim = scan.getDimension();

  // Update scan_dim_size to the local size and drop sdy.sharding_rule.
  mlir::NamedAttrList discardableAttrs(scan->getDiscardableAttrDictionary());
  discardableAttrs.erase(mlir::sdy::kShardingRuleAttr);

  auto newScan =
      mhlo::ScanOp::create(rewriter, scan.getLoc(), newResultTypes, operands,
                           scan.getProperties(), discardableAttrs.getAttrs());
  if (scan.getScanDimSize().has_value() && numInputs > 0) {
    auto localIn0Type =
        llvm::dyn_cast<mlir::RankedTensorType>(operands[0].getType());
    if (localIn0Type && !localIn0Type.isDynamicDim(scanDim)) {
      newScan.setScanDimSize(localIn0Type.getDimSize(scanDim));
    }
  }
  rewriter.cloneRegionBefore(scan.getBody(), newScan.getBody(),
                             newScan.getBody().end());

  mlir::Block& bodyBlock = newScan.getBody().front();
  bool perStepSliceShapeChanged = false;
  for (auto [i, arg] : llvm::enumerate(bodyBlock.getArguments())) {
    auto oldArgType = llvm::cast<mlir::RankedTensorType>(arg.getType());
    auto operandType =
        llvm::cast<mlir::RankedTensorType>(operands[i].getType());
    mlir::RankedTensorType newArgType;
    if (static_cast<int64_t>(i) < numInputs) {
      // Strip scanDim from scan input operands to form the per-step slice type.
      newArgType = eraseDim(operandType, scanDim);
    } else {
      // Use the full init operand type for carry block arguments.
      newArgType = operandType;
    }
    if (oldArgType.getShape() != newArgType.getShape()) {
      perStepSliceShapeChanged = true;
    }
    arg.setType(newArgType);
  }

  if (perStepSliceShapeChanged) {
    // If the per-step slice shape is changed, it means that non-scan dimensions
    // are sharded or padded, and that we need to retype the body.
    auto arg0Type =
        llvm::cast<mlir::RankedTensorType>(bodyBlock.getArgument(0).getType());
    localizeElementwiseBodyToPerStepSliceShape(bodyBlock, arg0Type.getShape());
  }

  rewriter.replaceOp(scan, newScan.getResults());
  return mlir::success();
}

// Shared lowering context for the 3-phase distributed prefix scan.
struct ScanLoweringContext {
  mhlo::ScanOp scan;
  mlir::Location loc;
  int64_t scanDim;
  bool isReverse;
  int64_t numInputs;
  int64_t numPartitions;
  int64_t localScanSize;
  mlir::ValueRange scanInputs;
  mlir::ValueRange inits;
  mlir::SmallVector<mlir::StringAttr> manualAxes;
  mlir::SmallVector<mlir::sdy::TensorShardingAttr> inShardings;
  mlir::SmallVector<mlir::sdy::TensorShardingAttr> outShardings;
  mlir::SmallVector<mlir::sdy::TensorShardingAttr> carryShardings;
  mlir::SmallVector<mlir::sdy::TensorShardingAttr> scanReplicatedInShardings;
  mlir::SmallVector<mlir::sdy::TensorShardingAttr> scanReplicatedOutShardings;
  mlir::SmallVector<mlir::sdy::TensorShardingAttr> innerCarryShardings;
};

// Output of Phase 2 (`buildCrossShardPrefixScan`).
struct CrossShardPrefixScanResult {
  mhlo::ScanOp prefixScan;
  mlir::SmallVector<mlir::Value> shardedIncs;
};

// Validates preconditions and populates `ScanLoweringContext` for `scan`.
std::optional<ScanLoweringContext> createScanLoweringContext(
    mhlo::ScanOp scan, const mlir::SymbolTable& symbolTable,
    mlir::IRRewriter& rewriter) {
  mlir::ValueRange inputs = scan.getInputs();
  mlir::ValueRange inits = scan.getInits();
  mlir::ValueRange outputs = scan.getOutputs();
  mlir::ValueRange carries = scan.getCarries();
  if (!scan.getIsAssociative().value_or(false) ||
      !isShapePreservingScan(scan) ||
      !llvm::equal(inputs.getTypes(), outputs.getTypes()) ||
      !llvm::equal(inits.getTypes(), carries.getTypes())) {
    return std::nullopt;
  }

  // Require the body terminator to yield matching output and carry values.
  int64_t numInputs = inputs.size();
  mlir::ValueRange termOperands = mlir::sdy::getBodyTerminatorOperands(scan);
  if (termOperands.take_front(numInputs) !=
      termOperands.drop_front(numInputs)) {
    return std::nullopt;
  }

  int64_t scanDim = scan.getDimension();
  mlir::sdy::TensorShardingAttr in0Sharding = mlir::sdy::getSharding(inputs[0]);
  if (!in0Sharding) {
    return std::nullopt;
  }
  ArrayRef<mlir::sdy::AxisRefAttr> scanAxes =
      in0Sharding.getDimSharding(scanDim).getAxes();
  if (scanAxes.empty()) {
    return std::nullopt;
  }

  mlir::SmallVector<mlir::StringAttr> manualAxes;
  manualAxes.reserve(scanAxes.size());
  for (mlir::sdy::AxisRefAttr axisRef : scanAxes) {
    if (axisRef.getSubAxisInfo()) {
      return std::nullopt;
    }
    manualAxes.push_back(rewriter.getStringAttr(axisRef.getName()));
  }

  mlir::sdy::MeshAttr mesh = in0Sharding.getMesh(symbolTable);
  if (!mesh) {
    return std::nullopt;
  }
  int64_t numPartitions =
      in0Sharding.getDimSharding(scanDim).getShardedSize(mesh);
  auto in0Type = llvm::cast<mlir::RankedTensorType>(inputs[0].getType());
  if (in0Type.isDynamicDim(scanDim)) {
    return std::nullopt;
  }
  int64_t globalScanSize = in0Type.getDimSize(scanDim);
  if (numPartitions <= 1 || globalScanSize <= 0) {
    return std::nullopt;
  }
  // TODO(b/537364058): Support indivisible scan dimension padding.
  if (globalScanSize % numPartitions != 0) {
    return std::nullopt;
  }
  int64_t localScanSize = globalScanSize / numPartitions;
  if (localScanSize <= 1) {
    return std::nullopt;
  }

  ScanLoweringContext ctx{
      /*scan=*/scan,
      /*loc=*/scan.getLoc(),
      /*scanDim=*/scanDim,
      /*isReverse=*/scan.getIsReverse(),
      /*numInputs=*/numInputs,
      /*numPartitions=*/numPartitions,
      /*localScanSize=*/localScanSize,
      /*scanInputs=*/inputs,
      /*inits=*/inits,
      /*manualAxes=*/std::move(manualAxes),
  };
  ctx.inShardings.reserve(numInputs);
  ctx.outShardings.reserve(numInputs);
  ctx.carryShardings.reserve(numInputs);
  ctx.scanReplicatedInShardings.reserve(numInputs);
  ctx.scanReplicatedOutShardings.reserve(numInputs);
  ctx.innerCarryShardings.reserve(numInputs);

  mlir::Attribute meshOrRef = in0Sharding.getMeshOrRef();
  for (int64_t i = 0; i < numInputs; ++i) {
    mlir::sdy::TensorShardingAttr inSharding = mlir::sdy::getOrCreateSharding(
        inputs[i], meshOrRef, /*closedIfMissing=*/true);
    mlir::sdy::TensorShardingAttr outSharding = mlir::sdy::getOrCreateSharding(
        outputs[i], meshOrRef, /*closedIfMissing=*/true);
    mlir::sdy::TensorShardingAttr carrySharding =
        mlir::sdy::getOrCreateSharding(carries[i], meshOrRef,
                                       /*closedIfMissing=*/true);
    ctx.inShardings.push_back(inSharding);
    ctx.outShardings.push_back(outSharding);
    ctx.carryShardings.push_back(carrySharding);
    ctx.scanReplicatedInShardings.push_back(
        mlir::sdy::eraseManualAxes(inSharding, ctx.manualAxes));
    ctx.scanReplicatedOutShardings.push_back(
        mlir::sdy::eraseManualAxes(outSharding, ctx.manualAxes));
    ctx.innerCarryShardings.push_back(
        mlir::sdy::eraseManualAxes(carrySharding, ctx.manualAxes));
  }
  return ctx;
}

// Phase 1: Emits a per-shard local reduction via `sdy.manual_computation`,
// returning the per-shard totals stacked along `scanDim` (shape `P` along
// `scanDim`).
mlir::sdy::ManualComputationOp buildLocalScanReduction(
    const ScanLoweringContext& ctx, mlir::IRRewriter& rewriter) {
  mlir::SmallVector<mlir::Type> stackedTotalTypes;
  stackedTotalTypes.reserve(ctx.numInputs);
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    stackedTotalTypes.push_back(withDimSize(ctx.scanInputs[i].getType(),
                                            ctx.scanDim, ctx.numPartitions));
  }

  auto phase1ManualComp = mlir::sdy::ManualComputationOp::create(
      rewriter, ctx.loc, stackedTotalTypes, ctx.scanInputs, ctx.inShardings,
      ctx.inShardings, ctx.manualAxes);
  mlir::Block& phase1Block = phase1ManualComp.getBody().emplaceBlock();
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    phase1Block.addArgument(withDimSize(ctx.scanInputs[i].getType(),
                                        ctx.scanDim, ctx.localScanSize),
                            ctx.loc);
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&phase1Block);

  mlir::SmallVector<mlir::Value> restInputs;
  mlir::SmallVector<mlir::Value> localInits;
  restInputs.reserve(ctx.numInputs);
  localInits.reserve(ctx.numInputs);

  int64_t initStart = ctx.isReverse ? ctx.localScanSize - 1 : 0;
  int64_t initLimit = ctx.isReverse ? ctx.localScanSize : 1;
  int64_t restStart = ctx.isReverse ? 0 : 1;
  int64_t restLimit = ctx.isReverse ? ctx.localScanSize - 1 : ctx.localScanSize;

  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    mlir::Value localIn = phase1Block.getArgument(i);
    mlir::Value initSlice =
        createSliceAlongDim(rewriter, ctx.loc, localIn, ctx.scanDim, initStart,
                            initLimit, ctx.scanReplicatedInShardings[i]);
    localInits.push_back(collapseSingleSliceToCarry(
        rewriter, ctx.loc, initSlice, ctx.scanDim, ctx.innerCarryShardings[i]));
    restInputs.push_back(createSliceAlongDim(rewriter, ctx.loc, localIn,
                                             ctx.scanDim, restStart, restLimit,
                                             ctx.scanReplicatedInShardings[i]));
  }

  mhlo::ScanOp localScan = createScanWithClonedBody(
      ctx.scan, restInputs, localInits, ctx.scanReplicatedInShardings,
      ctx.innerCarryShardings, rewriter);

  mlir::SmallVector<mlir::Value> localTotals;
  localTotals.reserve(ctx.numInputs);
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    localTotals.push_back(expandCarryToSingleSlice(
        rewriter, ctx.loc, localScan.getCarries()[i], ctx.scanDim,
        ctx.scanReplicatedInShardings[i]));
  }
  mlir::sdy::ReturnOp::create(rewriter, ctx.loc, localTotals);
  return phase1ManualComp;
}

// Phase 2: All-gathers per-shard totals along `scanDim`, runs a length-`P`
// prefix scan in global IR, and scatters the exclusive incoming carry to each
// shard.
CrossShardPrefixScanResult buildCrossShardPrefixScan(
    const ScanLoweringContext& ctx, mlir::ValueRange stackedTotals,
    mlir::IRRewriter& rewriter) {
  mlir::SmallVector<mlir::Value> gatheredTotals;
  gatheredTotals.reserve(ctx.numInputs);
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    auto reshardOp = mlir::sdy::ReshardOp::create(
        rewriter, ctx.loc, stackedTotals[i], ctx.scanReplicatedInShardings[i]);
    gatheredTotals.push_back(reshardOp.getResult());
  }

  CrossShardPrefixScanResult result;
  result.prefixScan = createScanWithClonedBody(
      ctx.scan, gatheredTotals, ctx.inits, ctx.scanReplicatedInShardings,
      ctx.carryShardings, rewriter);

  result.shardedIncs.reserve(ctx.numInputs);
  int64_t prefStart = ctx.isReverse ? 1 : 0;
  int64_t prefLimit = ctx.isReverse ? ctx.numPartitions : ctx.numPartitions - 1;
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    mlir::Value initSlice =
        expandCarryToSingleSlice(rewriter, ctx.loc, ctx.inits[i], ctx.scanDim,
                                 ctx.scanReplicatedInShardings[i]);
    mlir::Value prefixSlice = createSliceAlongDim(
        rewriter, ctx.loc, result.prefixScan.getOutputs()[i], ctx.scanDim,
        prefStart, prefLimit, ctx.scanReplicatedInShardings[i]);

    mlir::SmallVector<mlir::Value, 2> concatOperands =
        ctx.isReverse
            ? mlir::SmallVector<mlir::Value, 2>{prefixSlice, initSlice}
            : mlir::SmallVector<mlir::Value, 2>{initSlice, prefixSlice};
    auto allInc = mlir::stablehlo::ConcatenateOp::create(
        rewriter, ctx.loc, concatOperands, ctx.scanDim);
    mlir::sdy::setShardings(allInc, ctx.scanReplicatedInShardings[i]);

    auto shardedInc = mlir::sdy::ReshardOp::create(
        rewriter, ctx.loc, allInc.getResult(), ctx.inShardings[i]);
    result.shardedIncs.push_back(shardedInc.getResult());
  }
  return result;
}

// Phase 3: Emits the final per-shard scan via `sdy.manual_computation`, using
// each shard's exclusive incoming carry from `shardedIncs` as its initial
// state.
mlir::sdy::ManualComputationOp buildFinalLocalScan(
    const ScanLoweringContext& ctx, mlir::ValueRange shardedIncs,
    mlir::IRRewriter& rewriter) {
  mlir::SmallVector<mlir::Value> phase3Inputs;
  mlir::SmallVector<mlir::sdy::TensorShardingAttr> phase3InShardings;
  phase3Inputs.reserve(2 * ctx.numInputs);
  phase3InShardings.reserve(2 * ctx.numInputs);
  phase3Inputs.append(ctx.scanInputs.begin(), ctx.scanInputs.end());
  phase3Inputs.append(shardedIncs.begin(), shardedIncs.end());
  phase3InShardings.append(ctx.inShardings.begin(), ctx.inShardings.end());
  phase3InShardings.append(ctx.inShardings.begin(), ctx.inShardings.end());

  auto phase3ManualComp = mlir::sdy::ManualComputationOp::create(
      rewriter, ctx.loc, ctx.scanInputs.getTypes(), phase3Inputs,
      phase3InShardings, ctx.outShardings, ctx.manualAxes);
  mlir::Block& phase3Block = phase3ManualComp.getBody().emplaceBlock();
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    phase3Block.addArgument(withDimSize(ctx.scanInputs[i].getType(),
                                        ctx.scanDim, ctx.localScanSize),
                            ctx.loc);
  }
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    phase3Block.addArgument(
        withDimSize(ctx.scanInputs[i].getType(), ctx.scanDim, 1), ctx.loc);
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&phase3Block);

  mlir::SmallVector<mlir::Value> finalLocalInputs;
  mlir::SmallVector<mlir::Value> finalLocalInits;
  finalLocalInputs.reserve(ctx.numInputs);
  finalLocalInits.reserve(ctx.numInputs);
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    finalLocalInputs.push_back(phase3Block.getArgument(i));
    finalLocalInits.push_back(collapseSingleSliceToCarry(
        rewriter, ctx.loc, phase3Block.getArgument(ctx.numInputs + i),
        ctx.scanDim, ctx.innerCarryShardings[i]));
  }

  mhlo::ScanOp finalScan = createScanWithClonedBody(
      ctx.scan, finalLocalInputs, finalLocalInits,
      ctx.scanReplicatedOutShardings, ctx.innerCarryShardings, rewriter);
  mlir::sdy::ReturnOp::create(rewriter, ctx.loc, finalScan.getOutputs());
  return phase3ManualComp;
}

struct ScanPartitioningOpInterface
    : public mlir::sdy::PartitioningOpInterface::ExternalModel<
          ScanPartitioningOpInterface, mhlo::ScanOp> {
  std::optional<mlir::LogicalResult> resolvePermutationFactors(
      mlir::Operation* op, mlir::sdy::OpShardingRuleAttr shardingRule,
      mlir::IRRewriter& rewriter, const mlir::SymbolTable& symbolTable) const {
    rewriter.setInsertionPoint(op);
    auto scan = llvm::cast<mhlo::ScanOp>(op);
    if (scan.getInputs().empty() || !isScanDimSharded(scan)) {
      return mlir::success();
    }
    std::optional<ScanLoweringContext> scanContext =
        createScanLoweringContext(scan, symbolTable, rewriter);
    if (!scanContext) {
      // Fall back to default replication behavior if not eligible for lowering.
      return std::nullopt;
    }

    mlir::sdy::ManualComputationOp phaseOneManualComp =
        buildLocalScanReduction(*scanContext, rewriter);
    CrossShardPrefixScanResult prefixResult = buildCrossShardPrefixScan(
        *scanContext, phaseOneManualComp.getResults(), rewriter);
    mlir::sdy::ManualComputationOp phaseThreeManualComp =
        buildFinalLocalScan(*scanContext, prefixResult.shardedIncs, rewriter);

    mlir::SmallVector<mlir::Value> replacements;
    replacements.reserve(2 * scanContext->numInputs);
    replacements.append(phaseThreeManualComp.result_begin(),
                        phaseThreeManualComp.result_end());
    mlir::ValueRange globalCarries = prefixResult.prefixScan.getCarries();
    replacements.append(globalCarries.begin(), globalCarries.end());
    rewriter.replaceOp(scan, replacements);
    return mlir::success();
  }

  // TODO(b/537364058): Support padForDivisibility for indivisible dimensions.
  std::optional<mlir::LogicalResult> convertGlobalToLocal(
      mlir::Operation* op, mlir::ValueRange operands,
      mlir::ConversionPatternRewriter& rewriter,
      const mlir::TypeConverter& typeConverter,
      const mlir::SymbolTable& symbolTable) const {
    auto scan = llvm::cast<mhlo::ScanOp>(op);
    if (isScanDimSharded(scan)) {
      return scan.emitOpError(
          "expected scan dimension to be unsharded during global-to-local "
          "conversion");
    }
    return rebuildScanWithUpdatedOperandsAndResultTypes(
        scan, operands, typeConverter, rewriter);
  }
};

}  // namespace

void registerMhloExtensions(mlir::DialectRegistry& registry) {
  registry.addExtension(+[](mlir::MLIRContext* ctx, mhlo::MhloDialect*) {
    attachPointwiseShardingRule<mhlo::AcosOp, mhlo::AcoshOp, mhlo::AsinOp,
                                mhlo::AsinhOp, mhlo::AtanhOp, mhlo::CopyOp,
                                mhlo::CoshOp, mhlo::ErfOp, mhlo::MulhiOp,
                                mhlo::SinhOp, mhlo::StochasticConvertOp>(ctx);
    mhlo::RaggedDotOp::attachInterface<RaggedDotShardingRuleOpInterface>(*ctx);
    mhlo::ScanOp::attachInterface<ScanShardingRuleOpInterface,
                                  ScanPartitioningOpInterface>(*ctx);
    mhlo::TopKOp::attachInterface<TopKShardingRuleOpInterface>(*ctx);
  });
}

}  // namespace sdy
}  // namespace xla
