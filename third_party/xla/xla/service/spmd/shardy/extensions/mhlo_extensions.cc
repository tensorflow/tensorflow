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
#include "llvm/Support/MathExtras.h"
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
#include "shardy/dialect/sdy/transforms/export/utils.h"
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

// Retypes `bodyBlock` arguments to match the per-step slice shapes of `inputs`
// and `inits`, and updates elementwise body operations if the slice shape
// changed.
void retypeScanBodyToOperandShapes(mlir::Block& bodyBlock,
                                   mlir::ValueRange inputs,
                                   mlir::ValueRange inits, int64_t scanDim) {
  int64_t numInputs = inputs.size();
  bool perStepSliceShapeChanged = false;
  for (auto [i, arg] : llvm::enumerate(bodyBlock.getArguments())) {
    auto oldArgType = llvm::cast<mlir::RankedTensorType>(arg.getType());
    int64_t idx = static_cast<int64_t>(i);
    mlir::RankedTensorType newArgType =
        idx < numInputs
            ? eraseDim(
                  llvm::cast<mlir::RankedTensorType>(inputs[idx].getType()),
                  scanDim)
            : llvm::cast<mlir::RankedTensorType>(
                  inits[idx - numInputs].getType());
    if (oldArgType.getShape() != newArgType.getShape()) {
      perStepSliceShapeChanged = true;
    }
    arg.setType(newArgType);
  }

  if (perStepSliceShapeChanged && !bodyBlock.getArguments().empty()) {
    auto arg0Type =
        llvm::cast<mlir::RankedTensorType>(bodyBlock.getArgument(0).getType());
    localizeElementwiseBodyToPerStepSliceShape(bodyBlock, arg0Type.getShape());
  }
}

// Clones `scan` with updated operands and result shardings, inferring output
// types from `inputs`, carry types from `inits`, and `scan_dim_size` from
// `inputs[0]`.
mhlo::ScanOp createScanWithClonedBody(
    mhlo::ScanOp scan, mlir::ValueRange inputs, mlir::ValueRange inits,
    ArrayRef<mlir::sdy::TensorShardingAttr> outputShardings,
    ArrayRef<mlir::sdy::TensorShardingAttr> carryShardings,
    mlir::IRRewriter& rewriter) {
  mlir::SmallVector<mlir::Value> operands(inputs.begin(), inputs.end());
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
  retypeScanBodyToOperandShapes(newScan.getBody().front(), inputs, inits,
                                scan.getDimension());

  mlir::SmallVector<mlir::sdy::TensorShardingAttr> allShardings(
      outputShardings.begin(), outputShardings.end());
  allShardings.append(carryShardings.begin(), carryShardings.end());
  mlir::sdy::setShardings(newScan, allShardings);
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

  // Update scan_dim_size to the converted input's scan-dim size and drop
  // sdy.sharding_rule. All scan inputs share the same scan dimension size.
  mlir::NamedAttrList discardableAttrs(scan->getDiscardableAttrDictionary());
  discardableAttrs.erase(mlir::sdy::kShardingRuleAttr);

  auto newScan =
      mhlo::ScanOp::create(rewriter, scan.getLoc(), newResultTypes, operands,
                           scan.getProperties(), discardableAttrs.getAttrs());
  if (scan.getScanDimSize().has_value() && numInputs > 0) {
    auto newIn0Type =
        llvm::dyn_cast<mlir::RankedTensorType>(operands[0].getType());
    if (newIn0Type && !newIn0Type.isDynamicDim(scanDim)) {
      newScan.setScanDimSize(newIn0Type.getDimSize(scanDim));
    }
  }
  rewriter.cloneRegionBefore(scan.getBody(), newScan.getBody(),
                             newScan.getBody().end());
  retypeScanBodyToOperandShapes(newScan.getBody().front(),
                                operands.take_front(numInputs),
                                operands.drop_front(numInputs), scanDim);

  rewriter.replaceOp(scan, newScan.getResults());
  return mlir::success();
}

// Shared lowering context for the 3-phase distributed prefix scan.
struct ScanLoweringContext {
  mhlo::ScanOp scan;
  mlir::Location loc;
  mlir::sdy::MeshAttr mesh;
  int64_t scanDim;
  bool isReverse;
  int64_t numInputs;
  int64_t numPartitions;
  int64_t localScanSize;
  int64_t globalScanSize;
  int64_t lastShardValidSize = 0;
  bool needsScanPadding = false;
  mlir::SmallVector<mlir::Value> scanInputs;
  mlir::SmallVector<mlir::Value> scanInits;
  mlir::Value globalIdx;
  mlir::sdy::TensorShardingAttr globalIdxSharding;
  mlir::sdy::TensorShardingAttr localIdxSharding;
  mlir::SmallVector<mlir::sdy::AxisRefAttr> manualAxes;
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

// Pads `ctx.scanInputs` and `ctx.scanInits` along any indivisible sharded
// dimensions (both non-scan dimensions and `ctx.scanDim`) before building the
// manual computations, and creates the 1D global index `iota` tensor when
// `ctx.needsScanPadding` is true.
void maybePadScanInputsAndCreateGlobalIdx(ScanLoweringContext& ctx,
                                          mlir::IRRewriter& rewriter) {
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    auto inType =
        llvm::cast<mlir::RankedTensorType>(ctx.scanInputs[i].getType());
    mlir::RankedTensorType paddedInType = mlir::sdy::getDivisiblePaddedType(
        inType, ctx.inShardings[i], ctx.outShardings[i], ctx.mesh);
    if (paddedInType == inType) {
      continue;
    }
    ctx.scanInputs[i] = mlir::sdy::padHighSideToType(
        ctx.loc, ctx.scanInputs[i], paddedInType, ctx.inShardings[i], rewriter);
    mlir::RankedTensorType paddedInitType = eraseDim(paddedInType, ctx.scanDim);
    if (paddedInitType != ctx.scanInits[i].getType()) {
      mlir::sdy::TensorShardingAttr initSharding =
          mlir::sdy::getOrCreateSharding(ctx.scanInits[i],
                                         ctx.inShardings[i].getMeshOrRef(),
                                         /*closedIfMissing=*/true);
      ctx.scanInits[i] = mlir::sdy::padHighSideToType(
          ctx.loc, ctx.scanInits[i], paddedInitType, initSharding, rewriter);
    }
  }
  if (!ctx.needsScanPadding) {
    return;
  }
  int64_t paddedGlobalScanSize = ctx.numPartitions * ctx.localScanSize;
  auto globalIdxType = mlir::RankedTensorType::get({paddedGlobalScanSize},
                                                   rewriter.getI32Type());
  auto globalIdxOp = mlir::stablehlo::IotaOp::create(
      rewriter, ctx.loc, globalIdxType, rewriter.getI64IntegerAttr(0));
  mlir::Attribute meshOrRef = ctx.inShardings[0].getMeshOrRef();
  ctx.globalIdxSharding = mlir::sdy::TensorShardingAttr::get(
      rewriter.getContext(), meshOrRef,
      {ctx.inShardings[0].getDimShardings()[ctx.scanDim]},
      /*replicatedAxes=*/{}, /*unreducedAxes=*/{});
  ctx.localIdxSharding = mlir::sdy::TensorShardingAttr::get(
      rewriter.getContext(), meshOrRef,
      {mlir::sdy::DimensionShardingAttr::get(rewriter.getContext(), {},
                                             /*isClosed=*/true)},
      /*replicatedAxes=*/{}, /*unreducedAxes=*/{});
  mlir::sdy::setShardings(globalIdxOp, ctx.globalIdxSharding);
  ctx.globalIdx = globalIdxOp.getResult();
}

// Validates preconditions and populates `ScanLoweringContext` for `scan`.
std::optional<ScanLoweringContext> createScanLoweringContext(
    mhlo::ScanOp scan, const mlir::SymbolTable& symbolTable) {
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

  mlir::SmallVector<mlir::sdy::AxisRefAttr> manualAxes;
  manualAxes.reserve(scanAxes.size());
  for (mlir::sdy::AxisRefAttr axisRef : scanAxes) {
    if (axisRef.getSubAxisInfo()) {
      return std::nullopt;
    }
    manualAxes.push_back(axisRef);
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
  int64_t paddedScanSize = llvm::alignTo(globalScanSize, numPartitions);
  int64_t localScanSize = paddedScanSize / numPartitions;
  bool needsScanPadding = paddedScanSize != globalScanSize;
  int64_t lastShardValidSize =
      globalScanSize - (numPartitions - 1) * localScanSize;
  // Require localScanSize > 1 (so Phase 1 can split each shard into 1 init
  // slice and localScanSize - 1 rest slices) and at least one valid element per
  // shard (lastShardValidSize >= 1). If a trailing shard had only padded
  // elements, seeding its Phase 1 local carry from a padded slice without an
  // algebraic identity element would corrupt the cross-shard prefix scan.
  if (localScanSize <= 1 || lastShardValidSize <= 0) {
    return std::nullopt;
  }

  ScanLoweringContext ctx{
      /*scan=*/scan,
      /*loc=*/scan.getLoc(),
      /*mesh=*/mesh,
      /*scanDim=*/scanDim,
      /*isReverse=*/scan.getIsReverse(),
      /*numInputs=*/numInputs,
      /*numPartitions=*/numPartitions,
      /*localScanSize=*/localScanSize,
      /*globalScanSize=*/globalScanSize,
      /*lastShardValidSize=*/lastShardValidSize,
      /*needsScanPadding=*/needsScanPadding,
      /*scanInputs=*/
      mlir::SmallVector<mlir::Value>(inputs.begin(), inputs.end()),
      /*scanInits=*/
      mlir::SmallVector<mlir::Value>(inits.begin(), inits.end()),
      /*globalIdx=*/mlir::Value(),
      /*globalIdxSharding=*/mlir::sdy::TensorShardingAttr(),
      /*localIdxSharding=*/mlir::sdy::TensorShardingAttr(),
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

// Splits a Phase 1 local shard tensor of length `ctx.localScanSize` along
// `ctx.scanDim` into its initial carry value (first slice for forward scan,
// last slice for reverse scan) and the remaining `ctx.localScanSize - 1`
// slices.
std::pair<mlir::Value, mlir::Value> splitIntoLocalInitAndRest(
    const ScanLoweringContext& ctx, mlir::Value localOperand,
    mlir::sdy::TensorShardingAttr sliceSharding,
    mlir::sdy::TensorShardingAttr carrySharding, mlir::IRRewriter& rewriter) {
  int64_t initStart = ctx.isReverse ? ctx.localScanSize - 1 : 0;
  int64_t initLimit = ctx.isReverse ? ctx.localScanSize : 1;
  int64_t restStart = ctx.isReverse ? 0 : 1;
  int64_t restLimit = ctx.isReverse ? ctx.localScanSize - 1 : ctx.localScanSize;
  mlir::Value initSlice =
      createSliceAlongDim(rewriter, ctx.loc, localOperand, ctx.scanDim,
                          initStart, initLimit, sliceSharding);
  mlir::Value localInit = collapseSingleSliceToCarry(
      rewriter, ctx.loc, initSlice, ctx.scanDim, carrySharding);
  mlir::Value restInput =
      createSliceAlongDim(rewriter, ctx.loc, localOperand, ctx.scanDim,
                          restStart, restLimit, sliceSharding);
  return {localInit, restInput};
}

// Compares the last element of 1D `localIdx` (`tensor<Lxi32>`) against
// `ctx.globalScanSize` (`GE`) to produce a scalar `tensor<i1>` predicate that
// is true only on the last shard (`P - 1`).
mlir::Value createIsLastShardPred(const ScanLoweringContext& ctx,
                                  mlir::Value localIdx,
                                  mlir::IRRewriter& rewriter) {
  mlir::Value lastIdxSlice = createSliceAlongDim(
      rewriter, ctx.loc, localIdx, /*scanDim=*/0, ctx.localScanSize - 1,
      ctx.localScanSize, ctx.localIdxSharding);
  auto scalarI32Type = mlir::RankedTensorType::get({}, rewriter.getI32Type());
  auto lastIdx = mlir::stablehlo::ReshapeOp::create(
      rewriter, ctx.loc, scalarI32Type, lastIdxSlice);
  auto limitConst = mlir::stablehlo::ConstantOp::create(
      rewriter, ctx.loc,
      mlir::DenseIntElementsAttr::get(
          scalarI32Type, {static_cast<int32_t>(ctx.globalScanSize)}));
  auto isLastShard = mlir::stablehlo::CompareOp::create(
      rewriter, ctx.loc, lastIdx, limitConst,
      mlir::stablehlo::ComparisonDirection::GE);
  return isLastShard.getResult();
}

// On the last shard (`isLastShard`), shifts the `ctx.lastShardValidSize` valid
// slices of `localTensor` along `ctx.scanDim` to the high side (`[K : L]` when
// `shiftToHighSide` is true, before a reverse local scan) or back to the low
// side (`[0 : V]` when `shiftToHighSide` is false, after Phase 3's reverse
// local scan).
mlir::Value shiftLastShardValidSlice(const ScanLoweringContext& ctx,
                                     mlir::Value localTensor,
                                     mlir::Value isLastShard,
                                     bool shiftToHighSide,
                                     mlir::sdy::TensorShardingAttr sharding,
                                     mlir::IRRewriter& rewriter) {
  int64_t validSize = ctx.lastShardValidSize;
  int64_t padSize = ctx.localScanSize - validSize;
  int64_t sliceStart = shiftToHighSide ? 0 : padSize;
  int64_t sliceLimit = shiftToHighSide ? validSize : ctx.localScanSize;
  mlir::Value validSlice =
      createSliceAlongDim(rewriter, ctx.loc, localTensor, ctx.scanDim,
                          sliceStart, sliceLimit, sharding);

  auto localType = llvm::cast<mlir::RankedTensorType>(localTensor.getType());
  int64_t rank = localType.getRank();
  mlir::SmallVector<int64_t> low(rank, 0);
  mlir::SmallVector<int64_t> high(rank, 0);
  mlir::SmallVector<int64_t> interior(rank, 0);
  if (shiftToHighSide) {
    low[ctx.scanDim] = padSize;
  } else {
    high[ctx.scanDim] = padSize;
  }
  auto scalarType = mlir::RankedTensorType::get({}, localType.getElementType());
  mlir::Value zeroConst =
      mlir::sdy::createZeroConstant(ctx.loc, scalarType, rewriter);
  auto padOp = mlir::stablehlo::PadOp::create(
      rewriter, ctx.loc, localType, validSlice, zeroConst,
      rewriter.getDenseI64ArrayAttr(low), rewriter.getDenseI64ArrayAttr(high),
      rewriter.getDenseI64ArrayAttr(interior));
  mlir::sdy::setShardings(padOp, sharding);

  auto selectOp = mlir::stablehlo::SelectOp::create(
      rewriter, ctx.loc, localType, isLastShard, padOp.getResult(),
      localTensor);
  mlir::sdy::setShardings(selectOp, sharding);
  return selectOp.getResult();
}

// Selects the Phase 1 shard reduction carry: `fullCarry` on full shards
// (`0 .. P - 2`), and the prefix carry after `ctx.lastShardValidSize` valid
// elements (`localInit` when `V == 1`, or `localScanOutput` at `validOutIdx`
// when `V >= 2`) on the last shard (`P - 1`).
mlir::Value selectPhaseOneShardCarry(
    const ScanLoweringContext& ctx, mlir::Value localInit,
    mlir::Value localScanOutput, mlir::Value fullCarry, mlir::Value isLastShard,
    mlir::sdy::TensorShardingAttr carrySharding, mlir::IRRewriter& rewriter) {
  mlir::Value partialCarry = localInit;
  if (ctx.lastShardValidSize >= 2) {
    int64_t validOutIdx = ctx.isReverse
                              ? (ctx.localScanSize - ctx.lastShardValidSize)
                              : (ctx.lastShardValidSize - 2);
    mlir::Value validOutSlice = createSliceAlongDim(
        rewriter, ctx.loc, localScanOutput, ctx.scanDim, validOutIdx,
        validOutIdx + 1, mlir::sdy::getSharding(localScanOutput));
    partialCarry = collapseSingleSliceToCarry(rewriter, ctx.loc, validOutSlice,
                                              ctx.scanDim, carrySharding);
  }
  auto selectOp =
      mlir::stablehlo::SelectOp::create(rewriter, ctx.loc, fullCarry.getType(),
                                        isLastShard, partialCarry, fullCarry);
  mlir::sdy::setShardings(selectOp, carrySharding);
  return selectOp.getResult();
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

  mlir::SmallVector<mlir::Value> phaseOneInputs(ctx.scanInputs.begin(),
                                                ctx.scanInputs.end());
  mlir::SmallVector<mlir::sdy::TensorShardingAttr> phaseOneInShardings(
      ctx.inShardings.begin(), ctx.inShardings.end());
  if (ctx.needsScanPadding) {
    phaseOneInputs.push_back(ctx.globalIdx);
    phaseOneInShardings.push_back(ctx.globalIdxSharding);
  }

  auto phaseOneManualComp = mlir::sdy::ManualComputationOp::create(
      rewriter, ctx.loc, stackedTotalTypes, phaseOneInputs, phaseOneInShardings,
      ctx.inShardings, ctx.manualAxes);
  mlir::Block& phaseOneBlock = phaseOneManualComp.getBody().emplaceBlock();
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    phaseOneBlock.addArgument(withDimSize(ctx.scanInputs[i].getType(),
                                          ctx.scanDim, ctx.localScanSize),
                              ctx.loc);
  }
  if (ctx.needsScanPadding) {
    phaseOneBlock.addArgument(
        mlir::RankedTensorType::get({ctx.localScanSize}, rewriter.getI32Type()),
        ctx.loc);
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&phaseOneBlock);

  mlir::Value isLastShard;
  if (ctx.needsScanPadding) {
    isLastShard = createIsLastShardPred(
        ctx, phaseOneBlock.getArgument(ctx.numInputs), rewriter);
  }

  mlir::SmallVector<mlir::Value> restInputs;
  mlir::SmallVector<mlir::Value> localInits;
  restInputs.reserve(ctx.numInputs);
  localInits.reserve(ctx.numInputs);

  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    mlir::Value localIn = phaseOneBlock.getArgument(i);
    if (ctx.needsScanPadding && ctx.isReverse) {
      localIn = shiftLastShardValidSlice(
          ctx, localIn, isLastShard, /*shiftToHighSide=*/true,
          ctx.scanReplicatedInShardings[i], rewriter);
    }
    auto [localInit, restInput] = splitIntoLocalInitAndRest(
        ctx, localIn, ctx.scanReplicatedInShardings[i],
        ctx.innerCarryShardings[i], rewriter);
    localInits.push_back(localInit);
    restInputs.push_back(restInput);
  }

  mhlo::ScanOp localScan = createScanWithClonedBody(
      ctx.scan, restInputs, localInits, ctx.scanReplicatedInShardings,
      ctx.innerCarryShardings, rewriter);

  mlir::SmallVector<mlir::Value> localTotals;
  localTotals.reserve(ctx.numInputs);
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    mlir::Value shardCarry = localScan.getCarries()[i];
    if (ctx.needsScanPadding) {
      shardCarry = selectPhaseOneShardCarry(
          ctx, localInits[i], localScan.getOutputs()[i], shardCarry,
          isLastShard, ctx.innerCarryShardings[i], rewriter);
    }
    localTotals.push_back(
        expandCarryToSingleSlice(rewriter, ctx.loc, shardCarry, ctx.scanDim,
                                 ctx.scanReplicatedInShardings[i]));
  }
  mlir::sdy::ReturnOp::create(rewriter, ctx.loc, localTotals);
  return phaseOneManualComp;
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
      ctx.scan, gatheredTotals, ctx.scanInits, ctx.scanReplicatedInShardings,
      ctx.carryShardings, rewriter);

  result.shardedIncs.reserve(ctx.numInputs);
  int64_t prefStart = ctx.isReverse ? 1 : 0;
  int64_t prefLimit = ctx.isReverse ? ctx.numPartitions : ctx.numPartitions - 1;
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    mlir::Value initSlice =
        expandCarryToSingleSlice(rewriter, ctx.loc, ctx.scanInits[i],
                                 ctx.scanDim, ctx.scanReplicatedInShardings[i]);
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
  bool needsReverseShift = ctx.needsScanPadding && ctx.isReverse;
  mlir::SmallVector<mlir::Value> phaseThreeInputs;
  mlir::SmallVector<mlir::sdy::TensorShardingAttr> phaseThreeInShardings;
  phaseThreeInputs.reserve(2 * ctx.numInputs + (needsReverseShift ? 1 : 0));
  phaseThreeInShardings.reserve(2 * ctx.numInputs +
                                (needsReverseShift ? 1 : 0));
  phaseThreeInputs.append(ctx.scanInputs.begin(), ctx.scanInputs.end());
  phaseThreeInputs.append(shardedIncs.begin(), shardedIncs.end());
  phaseThreeInShardings.append(ctx.inShardings.begin(), ctx.inShardings.end());
  phaseThreeInShardings.append(ctx.inShardings.begin(), ctx.inShardings.end());
  if (needsReverseShift) {
    phaseThreeInputs.push_back(ctx.globalIdx);
    phaseThreeInShardings.push_back(ctx.globalIdxSharding);
  }

  auto phaseThreeManualComp = mlir::sdy::ManualComputationOp::create(
      rewriter, ctx.loc, mlir::ValueRange(ctx.scanInputs).getTypes(),
      phaseThreeInputs, phaseThreeInShardings, ctx.outShardings,
      ctx.manualAxes);
  mlir::Block& phaseThreeBlock = phaseThreeManualComp.getBody().emplaceBlock();
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    phaseThreeBlock.addArgument(withDimSize(ctx.scanInputs[i].getType(),
                                            ctx.scanDim, ctx.localScanSize),
                                ctx.loc);
  }
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    phaseThreeBlock.addArgument(
        withDimSize(ctx.scanInputs[i].getType(), ctx.scanDim, 1), ctx.loc);
  }
  if (needsReverseShift) {
    phaseThreeBlock.addArgument(
        mlir::RankedTensorType::get({ctx.localScanSize}, rewriter.getI32Type()),
        ctx.loc);
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&phaseThreeBlock);

  mlir::Value isLastShard;
  if (needsReverseShift) {
    isLastShard = createIsLastShardPred(
        ctx, phaseThreeBlock.getArgument(2 * ctx.numInputs), rewriter);
  }

  mlir::SmallVector<mlir::Value> finalLocalInputs;
  mlir::SmallVector<mlir::Value> finalLocalInits;
  finalLocalInputs.reserve(ctx.numInputs);
  finalLocalInits.reserve(ctx.numInputs);
  for (int64_t i = 0; i < ctx.numInputs; ++i) {
    mlir::Value localIn = phaseThreeBlock.getArgument(i);
    if (needsReverseShift) {
      localIn = shiftLastShardValidSlice(
          ctx, localIn, isLastShard, /*shiftToHighSide=*/true,
          ctx.scanReplicatedInShardings[i], rewriter);
    }
    finalLocalInputs.push_back(localIn);
    finalLocalInits.push_back(collapseSingleSliceToCarry(
        rewriter, ctx.loc, phaseThreeBlock.getArgument(ctx.numInputs + i),
        ctx.scanDim, ctx.innerCarryShardings[i]));
  }

  mhlo::ScanOp finalScan = createScanWithClonedBody(
      ctx.scan, finalLocalInputs, finalLocalInits,
      ctx.scanReplicatedOutShardings, ctx.innerCarryShardings, rewriter);
  mlir::SmallVector<mlir::Value> finalOutputs(finalScan.getOutputs().begin(),
                                              finalScan.getOutputs().end());
  if (needsReverseShift) {
    for (int64_t i = 0; i < ctx.numInputs; ++i) {
      finalOutputs[i] = shiftLastShardValidSlice(
          ctx, finalOutputs[i], isLastShard, /*shiftToHighSide=*/false,
          ctx.scanReplicatedOutShardings[i], rewriter);
    }
  }
  mlir::sdy::ReturnOp::create(rewriter, ctx.loc, finalOutputs);
  return phaseThreeManualComp;
}

// Slices Phase 3 outputs and Phase 2 carries back to the original unpadded
// result shapes when any dimension was padded for divisibility.
mlir::SmallVector<mlir::Value> maybeUnpadScanOutputs(
    const ScanLoweringContext& ctx, mlir::ValueRange phaseThreeOutputs,
    mlir::ValueRange globalCarries, mlir::IRRewriter& rewriter) {
  mlir::SmallVector<mlir::Value> results;
  results.reserve(2 * ctx.numInputs);
  auto appendUnpadded = [&](mlir::ValueRange values,
                            mlir::ValueRange origResults,
                            ArrayRef<mlir::sdy::TensorShardingAttr> shardings) {
    for (auto [val, origRes, sharding] :
         llvm::zip_equal(values, origResults, shardings)) {
      if (val.getType() != origRes.getType()) {
        val = mlir::sdy::sliceHighSideToType(ctx.loc, val, origRes.getType(),
                                             sharding, rewriter);
      }
      results.push_back(val);
    }
  };
  mhlo::ScanOp scan = ctx.scan;
  appendUnpadded(phaseThreeOutputs, scan.getOutputs(), ctx.outShardings);
  appendUnpadded(globalCarries, scan.getCarries(), ctx.carryShardings);
  return results;
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
        createScanLoweringContext(scan, symbolTable);
    if (!scanContext) {
      // Fall back to default replication behavior if not eligible for lowering.
      return std::nullopt;
    }

    maybePadScanInputsAndCreateGlobalIdx(*scanContext, rewriter);
    mlir::sdy::ManualComputationOp phaseOneManualComp =
        buildLocalScanReduction(*scanContext, rewriter);
    CrossShardPrefixScanResult prefixResult = buildCrossShardPrefixScan(
        *scanContext, phaseOneManualComp.getResults(), rewriter);
    mlir::sdy::ManualComputationOp phaseThreeManualComp =
        buildFinalLocalScan(*scanContext, prefixResult.shardedIncs, rewriter);

    mlir::SmallVector<mlir::Value> replacements =
        maybeUnpadScanOutputs(*scanContext, phaseThreeManualComp.getResults(),
                              prefixResult.prefixScan.getCarries(), rewriter);
    rewriter.replaceOp(scan, replacements);
    return mlir::success();
  }

  // Rebuilds `mhlo.scan` when non-scan dimensions are padded for divisibility.
  // Because `sdy-resolve-permutation-factors` runs before
  // `sdy-pad-for-divisibility`, any sharded scan dimension has already been
  // lowered into the 3-phase scan or replicated.
  std::optional<mlir::LogicalResult> padForDivisibility(
      mlir::Operation* op, mlir::ValueRange operands,
      mlir::ConversionPatternRewriter& rewriter,
      const mlir::TypeConverter& typeConverter,
      const mlir::SymbolTable& /*symbolTable*/) const {
    return rebuildScanWithUpdatedOperandsAndResultTypes(
        llvm::cast<mhlo::ScanOp>(op), operands, typeConverter, rewriter);
  }

  std::optional<mlir::LogicalResult> convertGlobalToLocal(
      mlir::Operation* op, mlir::ValueRange operands,
      mlir::ConversionPatternRewriter& rewriter,
      const mlir::TypeConverter& typeConverter,
      const mlir::SymbolTable& /*symbolTable*/) const {
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
