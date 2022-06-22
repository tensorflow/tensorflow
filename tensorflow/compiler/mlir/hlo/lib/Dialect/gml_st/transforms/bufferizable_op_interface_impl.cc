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

#include "mlir-hlo/Dialect/gml_st/transforms/bufferizable_op_interface_impl.h"

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using mlir::bufferization::AnalysisState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::BufferRelation;
using mlir::bufferization::ToMemrefOp;
using mlir::bufferization::ToTensorOp;

namespace mlir {
namespace gml_st {
namespace {

/// Bufferization of gml_st.loop. Replace with a new gml_st.loop
/// that operates entirely on memrefs.
struct LoopOpInterface
    : public BufferizableOpInterface::ExternalModel<LoopOpInterface, LoopOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto loopOp = cast<LoopOp>(op);

    // gml_st.loop operands alone do not bufferize to a memory read, but
    // one of the uses of their matching bbArgs may.
    return state.isValueRead(loopOp.getTiedBlockArgument(opOperand));
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Only operands with an aliasing OpResult (i.e., output operands) bufferize
    // to a memory write.
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    return !bufferizableOp.getAliasingOpResult(opOperand, state).empty();
  }

  SmallVector<OpResult> getAliasingOpResult(
      Operation *op, OpOperand &opOperand,
      const AnalysisState & /*state*/) const {
    auto loopOp = cast<LoopOp>(op);

    // Output operands are tied to their corresponding OpResults.
    OpResult opResult = loopOp.getTiedOpResult(opOperand);
    if (!opResult) return {};
    return {opResult};
  }

  BufferRelation bufferRelation(Operation * /*op*/, OpResult /*opResult*/,
                                const AnalysisState & /*state*/) const {
    return BufferRelation::Equivalent;
  }

  bool isWritable(Operation * /*op*/, Value /*value*/,
                  const AnalysisState & /*state*/) const {
    // Interestingly, LoopOp's bbArgs can **always** be viewed
    // inplace from the perspective of nested ops:
    //   1. Either the matching iter operand is not bufferized inplace and an
    //      alloc + optional copy makes the bbArg itself inplaceable.
    //   2. Or the matching iter operand is bufferized inplace and bbArg just
    //      bufferizes to that too.
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto loopOp = cast<LoopOp>(op);

    // Compute new inputs, outputs and results.
    SmallVector<Value> newInputs, newOutputs, newResults;
    for (unsigned i = loopOp.getNumControlOperands();
         i < loopOp->getNumOperands(); ++i) {
      OpOperand &operand = loopOp->getOpOperand(i);
      Value rewrittenValue = operand.get();
      if (rewrittenValue.getType().isa<TensorType>()) {
        rewrittenValue = getBuffer(rewriter, operand.get(), options);
      }
      if (i < loopOp.getNumControlOperands() + loopOp.getNumInputs()) {
        newInputs.push_back(rewrittenValue);
      } else {
        newOutputs.push_back(rewrittenValue);
        if (operand.get().getType().isa<TensorType>())
          newResults.push_back(rewrittenValue);
      }
    }

    // Create new TiledLoopOp.
    auto newLoopOp = rewriter.create<LoopOp>(
        loopOp.getLoc(), loopOp.lowerBound(), loopOp.upperBound(),
        loopOp.step(), newInputs, newOutputs, loopOp.iterator_types(),
        loopOp.distribution_types());

    // Remove terminator.
    if (!newLoopOp.getBody()->empty())
      rewriter.eraseOp(loopOp.getBody()->getTerminator());

    // Compute new loop body arguments.
    SmallVector<Value> newBlockArgs, newRegionInOutArgs, oldRegionInOutArgs;
    ValueRange newInductionVars = newLoopOp.getInductionVars();
    newBlockArgs.append(newInductionVars.begin(), newInductionVars.end());

    ValueRange newRegionInArgs = newLoopOp.getRegionInputArgs();
    ValueRange newRegionOutArgs = newLoopOp.getRegionOutputArgs();
    newRegionInOutArgs.append(newRegionInArgs.begin(), newRegionInArgs.end());
    newRegionInOutArgs.append(newRegionOutArgs.begin(), newRegionOutArgs.end());

    ValueRange oldRegionInArgs = loopOp.getRegionInputArgs();
    ValueRange oldRegionOutArgs = loopOp.getRegionOutputArgs();
    oldRegionInOutArgs.append(oldRegionInArgs.begin(), oldRegionInArgs.end());
    oldRegionInOutArgs.append(oldRegionOutArgs.begin(), oldRegionOutArgs.end());
    assert(newRegionInArgs.size() == oldRegionInArgs.size() &&
           "expected same number of input args");
    assert(newRegionOutArgs.size() == oldRegionOutArgs.size() &&
           "expected same number of output args");

    for (auto it : llvm::zip(oldRegionInOutArgs, newRegionInOutArgs)) {
      Value oldArg = std::get<0>(it);
      Value newArg = std::get<1>(it);
      rewriter.setInsertionPointToStart(newLoopOp.getBody());
      if (oldArg.getType().isa<TensorType>()) {
        newBlockArgs.push_back(rewriter.create<bufferization::ToTensorOp>(
            oldArg.getLoc(), newArg));
      } else {
        newBlockArgs.push_back(newArg);
      }
    }

    // Move old body into new loop.
    rewriter.mergeBlocks(loopOp.getBody(), newLoopOp.getBody(), newBlockArgs);

    // Replace previous terminator with a new one that does not yield anything.
    auto oldTerminator =
        cast<gml_st::YieldOp>(newLoopOp.getBody()->getTerminator());
    rewriter.setInsertionPointToEnd(newLoopOp.getBody());
    auto newTerminator =
        rewriter.create<gml_st::YieldOp>(oldTerminator->getLoc());

    // Copy buffer of yielded tensor to output buffer. If everything bufferized
    // inplace, this copy will fold away.
    rewriter.setInsertionPoint(newTerminator);
    for (auto it : llvm::zip(oldTerminator.values(), newOutputs)) {
      Value output = std::get<1>(it);
      Value toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(
          newTerminator.getLoc(), output.getType(), std::get<0>(it));
      if (failed(options.createMemCpy(rewriter, newTerminator.getLoc(),
                                      toMemrefOp, output)))
        return failure();
    }

    // Erase old terminator.
    rewriter.eraseOp(oldTerminator);

    // Replace results and delete old op.
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newResults);

    return success();
  }
};

// Returns the subset chain in reverse order, i.e. from subset to space.
// The space operation itself is not included.
FailureOr<SmallVector<Operation *>> findSubsetChain(Value subset) {
  SmallVector<Operation *> subsets;
  Operation *current = subset.getDefiningOp();
  while (current) {
    if (auto space = dyn_cast<SpaceOp>(*current)) break;

    subsets.push_back(current);
    // TODO(pifon): It might be useful to have a subset interface.
    if (auto tile = dyn_cast<TileOp>(*current)) {
      current = tile.subset().getDefiningOp();
      continue;
    }
    if (auto point = dyn_cast<PointOp>(*current)) {
      current = point.subset().getDefiningOp();
      continue;
    }
    return failure();
  }
  return subsets;
}

// TODO(pifon): Clean this up, for example, by using ViewLikeInterface.
SmallVector<Value> getPointIndicesValues(OpBuilder &b, PointOp pointOp) {
  SmallVector<Value> indices;
  unsigned rank = pointOp.getRank();
  indices.reserve(rank);
  unsigned numDynamic = 0;
  for (auto staticIndex : pointOp.static_indices().getAsRange<IntegerAttr>()) {
    if (ShapedType::isDynamicStrideOrOffset(staticIndex.getInt())) {
      indices.push_back(pointOp.dynamic_indices()[numDynamic++]);
    } else {
      Value indexValue = b.create<arith::ConstantIndexOp>(pointOp.getLoc(),
                                                          staticIndex.getInt());
      indices.push_back(indexValue);
    }
  }
  return indices;
}

// Returns a scalar or a memref type result of `gml_st.materialize` op after
// bufferization.
FailureOr<Value> materializeExtraction(OpBuilder &b, Value memref,
                                       Value subset) {
  auto subsetsOr = findSubsetChain(subset);
  if (failed(subsetsOr)) return failure();

  // Find subset use-def chain from space to the subset.
  // Create subview or load ops for the subset computation.
  OpBuilder::InsertionGuard g(b);
  Value result = memref;
  for (auto *subset : llvm::reverse(*subsetsOr)) {
    Location loc = subset->getLoc();
    b.setInsertionPointAfter(subset);
    if (auto tile = dyn_cast<TileOp>(*subset)) {
      result = b.create<memref::SubViewOp>(loc, result, tile.getMixedOffsets(),
                                           tile.getMixedSizes(),
                                           tile.getMixedOffsets());
      continue;
    }
    if (auto point = dyn_cast<PointOp>(*subset)) {
      result = b.create<memref::LoadOp>(loc, result,
                                        getPointIndicesValues(b, point));
      continue;
    }
    return failure();
  }
  return result;
}

struct MaterializeOpInterface
    : public BufferizableOpInterface::ExternalModel<MaterializeOpInterface,
                                                    MaterializeOp> {
  bool bufferizesToMemoryRead(Operation * /*op*/, OpOperand & /*opOperand*/,
                              const AnalysisState & /*state*/) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation * /*op*/, OpOperand & /*opOperand*/,
                               const AnalysisState & /*state*/) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(
      Operation *op, OpOperand &opOperand,
      const AnalysisState & /*state*/) const {
    auto result = op->getOpResult(0);
    if (result.getType().isa<RankedTensorType>() &&
        &opOperand == &op->getOpOperand(0))
      return {op->getOpResult(0)};
    return {};
  }

  BufferRelation bufferRelation(Operation * /*op*/, OpResult /*opResult*/,
                                const AnalysisState & /*state*/) const {
    return BufferRelation::Equivalent;
  }

  bool isWritable(Operation * /*op*/, Value /*value*/,
                  const AnalysisState & /*state*/) const {
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto materializeOp = cast<MaterializeOp>(op);

    FailureOr<Value> bufferOr =
        getBuffer(rewriter, materializeOp->getOpOperand(0).get(), options);
    if (failed(bufferOr)) return failure();

    FailureOr<Value> resultOr =
        materializeExtraction(rewriter, *bufferOr, materializeOp.subset());

    if (failed(resultOr)) return failure();

    bufferization::replaceOpWithBufferizedValues(rewriter, op, *resultOr);
    return success();
  }
};

}  // namespace
}  // namespace gml_st
}  // namespace mlir

void mlir::gml_st::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, gml_st::GmlStDialect * /*dialect*/) {
        LoopOp::attachInterface<LoopOpInterface>(*ctx);
        MaterializeOp::attachInterface<MaterializeOpInterface>(*ctx);
      });
}
