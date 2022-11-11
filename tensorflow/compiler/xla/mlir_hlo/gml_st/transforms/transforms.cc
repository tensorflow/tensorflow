/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "gml_st/transforms/transforms.h"

#include <tuple>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Matchers.h"

namespace mlir {
namespace gml_st {
bool isZero(Value v) { return matchPattern(v, m_Zero()); }
namespace {
using ::mlir::linalg::LinalgOp;

void generateLoopNest(OpBuilder &b, Location loc, ArrayRef<Range> loopRanges,
                      LinalgOp linalgOp, ArrayRef<Attribute> iteratorTypes,
                      function_ref<scf::ValueVector(OpBuilder &, Location,
                                                    ValueRange, ValueRange)>
                          bodyBuilderFn,
                      ArrayRef<StringRef> distributionTypes) {
  SmallVector<OpFoldResult, 4> lbs, ubs, steps;
  for (Range range : loopRanges) {
    lbs.emplace_back(range.offset);
    ubs.emplace_back(range.size);
    steps.emplace_back(range.stride);
  }

  auto wrappedBuilderFn = [&](OpBuilder &nestedBuilder, Location nestedLoc,
                              ValueRange ivs, ValueRange inputs,
                              ValueRange outputs) {
    SmallVector<Value> operandValuesToUse = inputs;
    operandValuesToUse.append(outputs.begin(), outputs.end());
    scf::ValueVector results =
        bodyBuilderFn(nestedBuilder, nestedLoc, ivs, operandValuesToUse);
    nestedBuilder.create<gml_st::YieldOp>(nestedLoc, results);
  };

  SmallVector<Value> inputs{linalgOp.getDpsInputOperands()};
  SmallVector<Value> outputs{linalgOp.getDpsInitOperands()};

  SmallVector<Value> lbsValue =
      mlir::getValueOrCreateConstantIndexOp(b, loc, lbs);
  SmallVector<Value> ubsValue =
      mlir::getValueOrCreateConstantIndexOp(b, loc, ubs);
  SmallVector<Value> stepsValue =
      mlir::getValueOrCreateConstantIndexOp(b, loc, steps);
  auto tiledLoop =
      b.create<LoopOp>(loc, lbsValue, ubsValue, stepsValue, inputs, outputs,
                       b.getArrayAttr(iteratorTypes), wrappedBuilderFn);
  if (!distributionTypes.empty())
    tiledLoop.setDistributionTypes(b, distributionTypes);
}

// Insert a tile `source` into the destination tensor `dest`. The position at
// which the tile is inserted (as well as size of tile) is taken from a given
// ExtractSliceOp `sliceOp`.
Value insertSliceIntoTensor(RewriterBase &b, Location loc,
                            tensor::ExtractSliceOp sliceOp, Value source,
                            Value dest) {
  return b.create<tensor::InsertSliceOp>(
      loc, sliceOp.getSource().getType(), source, dest, sliceOp.getOffsets(),
      sliceOp.getSizes(), sliceOp.getStrides(), sliceOp.getStaticOffsets(),
      sliceOp.getStaticSizes(), sliceOp.getStaticStrides());
}

FailureOr<linalg::TiledLinalgOp> tileLinalgOpImpl(
    RewriterBase &b, LinalgOp op, ValueRange tileSizes,
    const linalg::LinalgTilingOptions &options) {
  auto nLoops = op.getNumLoops();
  // Initial tile sizes may be too big, only take the first nLoops.
  tileSizes = tileSizes.take_front(nLoops);

  if (llvm::all_of(tileSizes, isZero)) {
    linalg::TiledLinalgOp tiledOp;
    tiledOp.op = cast<LinalgOp>(b.clone(*op.getOperation()));
    tiledOp.tensorResults.assign(tiledOp.op->result_begin(),
                                 tiledOp.op->result_end());
    return tiledOp;
  }

  SmallVector<OpFoldResult> tileSizesFold;
  for (Value tileSize : tileSizes) tileSizesFold.push_back(tileSize);

  // 1. Build the tiled loop ranges.
  auto allShapeSizes = op.createFlatListOfOperandDims(b, op.getLoc());
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap) return failure();

  SmallVector<Range, 4> loopRanges;
  mlir::linalg::LoopIndexToRangeIndexMap loopIndexToRangeIndex;
  std::tie(loopRanges, loopIndexToRangeIndex) =
      mlir::linalg::makeTiledLoopRanges(b, op.getLoc(), shapeSizesToLoopsMap,
                                        allShapeSizes, tileSizesFold);

  SmallVector<Attribute, 4> iteratorTypes;
  for (const auto &attr : enumerate(op.getIteratorTypesArray())) {
    if (loopIndexToRangeIndex.count(attr.index()))
      iteratorTypes.push_back(
          IteratorTypeAttr::get(b.getContext(), attr.value()));
  }

  // 2. Create the tiled loops.
  LinalgOp res = op;
  SmallVector<Value, 4> ivs, tensorResults;
  auto tiledLoopBodyBuilder =
      [&](OpBuilder & /*builder*/, Location loc, ValueRange localIvs,
          ValueRange operandValuesToUse) -> scf::ValueVector {
    ivs.assign(localIvs.begin(), localIvs.end());

    // Tile the `operandValuesToUse` that either match the `op` operands
    // themselves or the tile loop arguments forwarding them.
    assert(operandValuesToUse.size() == op->getNumOperands() &&
           "expect the number of operands and inputs and outputs to match");
    SmallVector<Value> valuesToTile = operandValuesToUse;
    auto sizeBounds = makeComposedFoldedMultiResultAffineApply(
        b, loc, shapeSizesToLoopsMap, allShapeSizes);
    SmallVector<OpFoldResult> ivsFold(ivs.begin(), ivs.end());
    SmallVector<Value, 4> tiledOperands = makeTiledShapes(
        b, loc, op, valuesToTile, ivsFold, tileSizesFold, sizeBounds,
        /*omitPartialTileCheck=*/false);

    SmallVector<Type, 4> resultTensorTypes;
    for (OpOperand *opOperand : op.getDpsInitOperands())
      resultTensorTypes.push_back(
          tiledOperands[opOperand->getOperandNumber()].getType());

    res = op.clone(b, loc, resultTensorTypes, tiledOperands);

    // Insert a insert_slice for each output tensor.
    unsigned resultIdx = 0;
    for (OpOperand *opOperand : op.getDpsInitOperands()) {
      Value outputTensor = tiledOperands[opOperand->getOperandNumber()];
      IRRewriter rewriter(b);
      if (auto sliceOp = outputTensor.getDefiningOp<tensor::ExtractSliceOp>()) {
        tensorResults.push_back(insertSliceIntoTensor(rewriter, loc, sliceOp,
                                                      res->getResult(resultIdx),
                                                      sliceOp.getSource()));
      } else {
        tensorResults.push_back(res->getResult(resultIdx));
      }
      ++resultIdx;
    }
    return scf::ValueVector(tensorResults.begin(), tensorResults.end());
  };
  generateLoopNest(b, op.getLoc(), loopRanges, op, iteratorTypes,
                   tiledLoopBodyBuilder, options.distributionTypes);

  // 3. Transform IndexOp results w.r.t. the tiling.
  mlir::linalg::transformIndexOps(b, res, ivs, loopIndexToRangeIndex);

  // 4. Gather the newly created loops and return them with the new op.
  SmallVector<Operation *, 8> loops;
  loops.reserve(ivs.size());
  for (auto iv : ivs) {
    if (iv.isa<BlockArgument>()) {
      loops.push_back(iv.cast<BlockArgument>().getOwner()->getParentOp());
      assert(loops.back() && "no owner found for induction variable!");
    } else {
      loops.push_back(nullptr);
    }
  }

  // 5. Get the tensor results from the outermost loop if available. Otherwise
  // use the previously captured `tensorResults`.
  Operation *outermostLoop = nullptr;
  for (Operation *loop : loops)
    if ((outermostLoop = loop)) break;

  return linalg::TiledLinalgOp{
      res, loops, outermostLoop ? outermostLoop->getResults() : tensorResults};
}
}  // namespace

FailureOr<linalg::TiledLinalgOp> tileLinalgOp(
    RewriterBase &b, linalg::LinalgOp op,
    const linalg::LinalgTilingOptions &options) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);

  if (!options.tileSizeComputationFunction) return failure();

  // Enforce the convention that "tiling by zero" skips tiling a particular
  // dimension. This convention is significantly simpler to handle instead of
  // adjusting affine maps to account for missing dimensions.
  auto nLoops = op.getNumLoops();
  SmallVector<Value, 4> tileSizeVector =
      options.tileSizeComputationFunction(b, op);
  if (tileSizeVector.size() < nLoops) {
    auto zero = b.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    tileSizeVector.append(nLoops - tileSizeVector.size(), zero);
  }

  return tileLinalgOpImpl(b, op, tileSizeVector, options);
}

void setTransformationAttr(mlir::OpBuilder &b, Operation *op, StringRef name) {
  op->setAttr(name, b.getBoolAttr(true));
}

void removeTransformationAttr(Operation *op, StringRef name) {
  op->removeAttr(name);
}

bool hasTransformationAttr(Operation *op, StringRef name) {
  auto marker = op->getAttr(name);
  if (!marker) return false;
  return marker && marker.cast<BoolAttr>().getValue();
}

constexpr llvm::StringLiteral kOpLabel = "op_label";

bool hasMatchingLabel(Operation *op, StringRef label) {
  auto opLabelAttr = op->getAttr(kOpLabel);
  if (!opLabelAttr) return false;

  return opLabelAttr.cast<StringAttr>().getValue() == label;
}

}  // namespace gml_st
}  // namespace mlir
