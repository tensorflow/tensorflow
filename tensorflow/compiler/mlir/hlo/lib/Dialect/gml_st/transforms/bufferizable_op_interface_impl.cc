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

#include <iterator>
#include <tuple>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LogicalResult.h"

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

  FailureOr<BaseMemRefType> getBufferType(
      Operation *op, BlockArgument bbArg,
      const BufferizationOptions &options) const {
    auto loopOp = cast<LoopOp>(op);
    return bufferization::getBufferType(loopOp.getTiedOperand(bbArg).get(),
                                        options);
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
        FailureOr<Value> maybeBuffer =
            getBuffer(rewriter, operand.get(), options);
        if (failed(maybeBuffer)) return failure();
        rewrittenValue = *maybeBuffer;
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

// Returns the set chain in reverse order, i.e. from set to space.
// The space operation itself is not included.
FailureOr<SmallVector<Operation *>> findSetChain(Value set) {
  SmallVector<Operation *> sets;
  Operation *current = set.getDefiningOp();
  while (current) {
    if (auto space = dyn_cast<SpaceOp>(*current)) break;

    sets.push_back(current);
    // TODO(pifon): It might be useful to have a set interface.
    if (auto tile = dyn_cast<TileOp>(*current)) {
      current = tile.superset().getDefiningOp();
      continue;
    }
    if (auto point = dyn_cast<PointOp>(*current)) {
      current = point.superset().getDefiningOp();
      continue;
    }
    return failure();
  }
  return sets;
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
FailureOr<Value> materializeExtraction(OpBuilder &b, Value memref, Value set) {
  auto setsOr = findSetChain(set);
  if (failed(setsOr)) return failure();

  // Find set use-def chain from space to the set.
  // Create subview or load ops for the set computation.
  OpBuilder::InsertionGuard g(b);
  Value result = memref;
  for (auto *set : llvm::reverse(*setsOr)) {
    Location loc = set->getLoc();
    b.setInsertionPointAfter(set);
    if (auto tile = dyn_cast<TileOp>(*set)) {
      result = b.create<memref::SubViewOp>(loc, result, tile.getMixedOffsets(),
                                           tile.getMixedSizes(),
                                           tile.getMixedStrides());
      continue;
    }
    if (auto point = dyn_cast<PointOp>(*set)) {
      result = b.create<memref::LoadOp>(loc, result,
                                        getPointIndicesValues(b, point));
      continue;
    }
    return failure();
  }
  return result;
}

LogicalResult materializeInsertion(OpBuilder &b, Value update, Value set,
                                   Value memref,
                                   const BufferizationOptions &options) {
  auto sets = findSetChain(set);
  if (failed(sets)) return failure();

  if (sets->empty())
    return options.createMemCpy(b, update.getLoc(), update, memref);

  // Create subviews or store ops for the set computation.
  OpBuilder::InsertionGuard g(b);
  auto *it = std::prev(sets->end());
  // The first element for the use-def chain is the `gml_st.space` op and it
  // should be ignored for now.
  for (; it != sets->begin(); --it) {
    Location loc = (*it)->getLoc();
    b.setInsertionPointAfter(*it);

    auto tile = dyn_cast<TileOp>(*it);
    if (!tile) return failure();

    memref = b.create<memref::SubViewOp>(loc, memref, tile.getMixedOffsets(),
                                         tile.getMixedSizes(),
                                         tile.getMixedStrides());
  }
  Location loc = (*it)->getLoc();
  if (auto point = dyn_cast<PointOp>(*it)) {
    b.create<memref::StoreOp>(loc, update, memref,
                              getPointIndicesValues(b, point));
    return success();
  }
  if (auto tile = dyn_cast<TileOp>(*it)) {
    memref = b.create<memref::SubViewOp>(loc, memref, tile.getMixedOffsets(),
                                         tile.getMixedSizes(),
                                         tile.getMixedOffsets());
    return success();
  }
  llvm_unreachable("Unknown set type");
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
        opOperand.getOperandNumber() == 0)
      return {result};
    return {};
  }

  BufferRelation bufferRelation(Operation * /*op*/, OpResult /*opResult*/,
                                const AnalysisState & /*state*/) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto materializeOp = cast<MaterializeOp>(op);

    FailureOr<Value> bufferOr =
        getBuffer(rewriter, materializeOp->getOpOperand(0).get(), options);
    if (failed(bufferOr)) return failure();

    FailureOr<Value> resultOr =
        materializeExtraction(rewriter, *bufferOr, materializeOp.set());

    if (failed(resultOr)) return failure();

    bufferization::replaceOpWithBufferizedValues(rewriter, op, *resultOr);
    return success();
  }
};

struct ParallelOpInterface
    : public BufferizableOpInterface::ExternalModel<ParallelOpInterface,
                                                    ParallelOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(
      Operation *op, OpResult opResult, const AnalysisState & /*state*/) const {
    auto parallelOp = cast<ParallelOp>(op);
    return {
        parallelOp.getTerminator().getDstOperand(opResult.getResultNumber())};
  }

  bool isMemoryWrite(Operation *, OpResult, const AnalysisState &) const {
    // This op is a memory write. Stop lookup here to avoid finding false
    // conflicts involving this op and one of the ops in the region. This is
    // similar to how scf.if ops are analyzed.
    return true;
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
                          const BufferizationOptions & /*options*/) const {
    auto loopOp = cast<ParallelOp>(op);

    // Create new TiledLoopOp.
    auto newLoopOp = rewriter.create<ParallelOp>(
        loopOp.getLoc(), TypeRange{llvm::None}, loopOp.lowerBound(),
        loopOp.upperBound(), loopOp.step(), nullptr);

    // Move the old body into the new loop.
    rewriter.mergeBlocks(loopOp.getBody(), newLoopOp.getBody(),
                         newLoopOp.getInductionVars());

    // Remove the old op.
    rewriter.eraseOp(op);
    return success();
  }
};

struct ForOpInterface
    : public BufferizableOpInterface::ExternalModel<ForOpInterface, ForOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(
      Operation *op, OpResult opResult, const AnalysisState & /*state*/) const {
    auto forOp = cast<gml_st::ForOp>(op);
    return {&forOp.getOpOperandForResult(opResult)};
  }

  bool isWritable(Operation * /*op*/, Value /*value*/,
                  const AnalysisState & /*state*/) const {
    // Interestingly, ForOp's bbArg can **always** be viewed
    // inplace from the perspective of ops nested under:
    //   1. Either the matching iter operand is not bufferized inplace and an
    //      alloc + optional copy makes the bbArg itself inplaceable.
    //   2. Or the matching iter operand is bufferized inplace and bbArg just
    //      bufferizes to that too.
    return true;
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto forOp = cast<gml_st::ForOp>(op);
    return state.isValueRead(forOp.getRegionOutputArgForOpOperand(opOperand));
  }

  bool bufferizesToMemoryWrite(Operation * /*op*/, OpOperand & /*opOperand*/,
                               const AnalysisState & /*state*/) const {
    return true;
  }

  SmallVector<OpResult> getAliasingOpResult(
      Operation *op, OpOperand &opOperand,
      const AnalysisState & /*state*/) const {
    auto forOp = cast<gml_st::ForOp>(op);
    return {forOp.getResultForOpOperand(opOperand)};
  }

  BufferRelation bufferRelation(Operation * /*op*/, OpResult /*opResult*/,
                                const AnalysisState & /*state*/) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto forOp = cast<ForOp>(op);
    Location loc = forOp.getLoc();

    // Get the bufferized output arguments.
    SmallVector<Value> bufferizedOutputs;
    bufferizedOutputs.reserve(forOp.getNumOutputs());
    for (Value output : forOp.outputs()) {
      FailureOr<Value> maybeBuffer = getBuffer(rewriter, output, options);
      if (failed(maybeBuffer)) return failure();
      bufferizedOutputs.push_back(*maybeBuffer);
    }

    // Create new ForOp.
    auto newForOp = rewriter.create<ForOp>(loc, TypeRange{}, forOp.lowerBound(),
                                           forOp.upperBound(), forOp.step(),
                                           ValueRange{}, nullptr);
    Block *loopBody = newForOp.getBody();

    // Add conversions to tensor so that we can reuse the old loop body.
    rewriter.setInsertionPointToStart(loopBody);
    SmallVector<Value> outputsToTensors;
    for (auto buf : bufferizedOutputs) {
      Value tensor = rewriter.create<bufferization::ToTensorOp>(loc, buf);
      outputsToTensors.push_back(tensor);
    }
    SmallVector<Value> blockArgs = newForOp.getInductionVars();
    blockArgs.append(outputsToTensors);

    // Move old body into new for loop.
    rewriter.mergeBlocks(forOp.getBody(), loopBody, blockArgs);

    // Replace results and delete old op.
    bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                 bufferizedOutputs);
    return success();
  }
};

struct SetYieldOpInterface
    : public BufferizableOpInterface::ExternalModel<SetYieldOpInterface,
                                                    SetYieldOp> {
  SmallVector<OpResult> getAliasingOpResult(
      Operation *op, OpOperand &opOperand,
      const AnalysisState & /*state*/) const {
    auto yieldOp = cast<SetYieldOp>(op);
    if (!yieldOp.isDstOperand(opOperand)) return {};

    auto loopResult = yieldOp.getTiedOpResult(opOperand);
    assert(succeeded(loopResult) && "didn't find a corresponding loop result");
    return {*loopResult};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto yieldOp = cast<SetYieldOp>(op);
    Operation *loop = yieldOp->getParentOp();
    if (!isa<ForOp, ParallelOp>(loop))
      return yieldOp->emitError("unsupported gml_st::SetYieldOp parent");

    for (const auto &it :
         llvm::enumerate(llvm::zip(yieldOp.srcs(), yieldOp.dsts(),
                                   yieldOp.sets(), loop->getResults()))) {
      Value src, dst, set, loopResult;
      std::tie(src, dst, set, loopResult) = it.value();

      // `src` can be a scalar, that's `getBuffer()` should be called only for
      // tensor types.
      if (src.getType().isa<RankedTensorType>()) {
        FailureOr<Value> srcBufferOr = getBuffer(rewriter, src, options);
        if (failed(srcBufferOr)) return failure();

        src = *srcBufferOr;
      }

      FailureOr<Value> dstBufferOr = getBuffer(rewriter, dst, options);
      if (failed(dstBufferOr)) return failure();
      Value dstBuffer = *dstBufferOr;

      if (failed(materializeInsertion(rewriter, src, set, dstBuffer, options)))
        return failure();
      if (auto parallelOp =
              dyn_cast<gml_st::ParallelOp>(yieldOp->getParentOp())) {
        // Replace results of the enclosing loop with `to_tensor(dst)`.
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointAfter(loop);

        Value resultToTensor =
            rewriter.create<ToTensorOp>(loop->getLoc(), dstBuffer);
        for (OpOperand &use : loopResult.getUses()) {
          rewriter.updateRootInPlace(use.getOwner(),
                                     [&]() { use.set(resultToTensor); });
        }
      }
    }
    rewriter.replaceOpWithNewOp<SetYieldOp>(op);
    return success();
  }

  bool bufferizesToMemoryRead(Operation * /*op*/, OpOperand & /*opOperand*/,
                              const AnalysisState & /*state*/) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState & /*state*/) const {
    return cast<SetYieldOp>(op).isDstOperand(opOperand);
  }

  bool isNotConflicting(Operation * /*op*/, OpOperand * /*uRead*/,
                        OpOperand * /*uConflictingWrite*/,
                        const AnalysisState & /*state*/) const {
    // TODO(pifon): Implement proper analysis here similar to InsertSliceOp.
    return true;
  }
};

}  // namespace
}  // namespace gml_st
}  // namespace mlir

void mlir::gml_st::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, gml_st::GmlStDialect * /*dialect*/) {
        ForOp::attachInterface<ForOpInterface>(*ctx);
        LoopOp::attachInterface<LoopOpInterface>(*ctx);
        MaterializeOp::attachInterface<MaterializeOpInterface>(*ctx);
        ParallelOp::attachInterface<ParallelOpInterface>(*ctx);
        SetYieldOp::attachInterface<SetYieldOpInterface>(*ctx);
      });
}
