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

#include "gml_st/interfaces/bufferizable_op_interface_impl.h"

#include <iterator>
#include <optional>
#include <tuple>

#include "gml_st/IR/gml_st_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LogicalResult.h"

using mlir::bufferization::AliasingOpOperandList;
using mlir::bufferization::AliasingOpResultList;
using mlir::bufferization::AnalysisState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::BufferRelation;
using mlir::bufferization::ToTensorOp;
using mlir::tensor::ExtractSliceOp;

namespace mlir {
namespace gml_st {
namespace {


LogicalResult materializeInsertion(OpBuilder &b, Value update, Value set,
                                   Value memref,
                                   const BufferizationOptions &options) {
  Location loc = update.getLoc();

  Operation *setDefiningOp = set.getDefiningOp();

  // Create subviews or store ops for the set computation.
  auto tile = dyn_cast<TileOp>(setDefiningOp);
  if (!tile) {
    // TODO(bchetioui): this check for an unrealized conversion cast does not
    // belong here. This workaround will have to be deleted once SetYieldOp can
    // be canonicalized correctly.

    // If constants were folded into the tile type during canonicalization,
    // tile creation is followed by an UnrealizedConversionCastOp on the tile.
    auto castOp = dyn_cast<UnrealizedConversionCastOp>(setDefiningOp);
    if (!castOp) return failure();

    tile = dyn_cast<TileOp>(castOp->getOperand(0).getDefiningOp());
    if (!tile) return failure();
  }

  if (!update.getType().isa<ShapedType>()) {
    auto indices =
        getValueOrCreateConstantIndexOp(b, loc, tile.getMixedOffsets());
    b.create<memref::StoreOp>(loc, update, memref, indices);
    return success();
  }

  memref =
      b.create<memref::SubViewOp>(loc, memref, tile.getMixedOffsets(),
                                  tile.getMixedSizes(), tile.getMixedStrides());
  return options.createMemCpy(b, loc, update, memref);
}

struct ParallelOpInterface
    : public BufferizableOpInterface::ExternalModel<ParallelOpInterface,
                                                    ParallelOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto parallelOp = cast<ParallelOp>(op);

    // gml_st.parallel alone doesn't bufferize to a memory read, one of the uses
    // of its matching bbArg may.
    return state.isValueRead(
        parallelOp.getRegionOutputArgForOpOperand(opOperand));
  }

  bool bufferizesToMemoryWrite(Operation * /*op*/, OpOperand & /*opOperand*/,
                               const AnalysisState & /*state*/) const {
    // Outputs of gml_st::ParallelOp are always considered as a write.
    return true;
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &) const {
    auto parallelOp = cast<ParallelOp>(op);
    return {{parallelOp.getResultForOpOperand(opOperand),
             BufferRelation::Equivalent}};
  }

  bool isWritable(Operation * /*op*/, Value /*value*/,
                  const AnalysisState & /*state*/) const {
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto parallelOp = cast<ParallelOp>(op);

    // Get the bufferized output arguments.
    Location loc = op->getLoc();
    SmallVector<Value> bufferizedOutputs;
    bufferizedOutputs.reserve(parallelOp.getNumOutputs());
    for (Value output : parallelOp.getOutputs()) {
      FailureOr<Value> maybeBuffer = getBuffer(rewriter, output, options);
      if (failed(maybeBuffer)) return failure();
      bufferizedOutputs.push_back(*maybeBuffer);
    }

    // Create new ParallelOp.
    std::optional<StringAttr> distTypeAttr;
    if (auto distType = cast<ParallelOp>(op).getDistributionType())
      distTypeAttr = rewriter.getStringAttr(*distType);

    auto newParallelOp = rewriter.create<ParallelOp>(
        loc, TypeRange{}, parallelOp.getLowerBound(),
        parallelOp.getUpperBound(), parallelOp.getStep(), ValueRange{},
        distTypeAttr, nullptr);
    Block *loopBody = newParallelOp.getBody();

    // Add conversions to tensor so that we can reuse the old loop body.
    rewriter.setInsertionPointToStart(loopBody);
    SmallVector<Value> outputsToTensors;
    for (auto buf : bufferizedOutputs) {
      Value tensor = rewriter.create<bufferization::ToTensorOp>(loc, buf);
      outputsToTensors.push_back(tensor);
    }
    SmallVector<Value> blockArgs = newParallelOp.getInductionVars();
    blockArgs.append(outputsToTensors);

    // Move old body into new for loop.
    rewriter.mergeBlocks(parallelOp.getBody(), loopBody, blockArgs);

    // Replace results and delete old op.
    bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                 bufferizedOutputs);
    return success();
  }

  FailureOr<BaseMemRefType> getBufferType(
      Operation *op, Value value, const BufferizationOptions &options,
      const DenseMap<Value, BaseMemRefType> &fixedTypes) const {
    auto parallelOp = cast<ParallelOp>(op);

    if (auto bbArg = value.dyn_cast<BlockArgument>()) {
      // A tensor block argument has the same bufferized type as the
      // corresponding output operand.
      return bufferization::getBufferType(
          parallelOp.getOpOperandForRegionOutputArg(bbArg).get(), options,
          fixedTypes);
    }

    // The bufferized result type is the same as the bufferized type of the
    // corresponding output operand.
    return bufferization::getBufferType(
        parallelOp.getOutputs()[value.cast<OpResult>().getResultNumber()],
        options, fixedTypes);
  }

  bool isRepetitiveRegion(Operation * /*op*/, unsigned /*index*/) const {
    return true;
  }
};

struct SetYieldOpInterface
    : public BufferizableOpInterface::ExternalModel<SetYieldOpInterface,
                                                    SetYieldOp> {
  AliasingOpResultList getAliasingOpResults(
      Operation * /*op*/, OpOperand & /*opOperand*/,
      const AnalysisState & /*state*/) const {
    return {};
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState & /*state*/) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState & /*state*/) const {
    return cast<SetYieldOp>(op).isDstOperand(opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto yieldOp = cast<SetYieldOp>(op);
    Operation *loop = yieldOp->getParentOp();

    rewriter.setInsertionPoint(op);
    for (const auto &it :
         llvm::enumerate(llvm::zip(yieldOp.getSrcs(), yieldOp.getDsts(),
                                   yieldOp.getSets(), loop->getResults()))) {
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
        for (OpOperand &use :
             llvm::make_early_inc_range(loopResult.getUses())) {
          rewriter.updateRootInPlace(use.getOwner(),
                                     [&]() { use.set(resultToTensor); });
        }
      }
    }
    rewriter.replaceOpWithNewOp<SetYieldOp>(op);
    return success();
  }

  LogicalResult resolveConflicts(Operation *op, RewriterBase &rewriter,
                                 const AnalysisState &state) const {
    OpBuilder::InsertionGuard g(rewriter);
    SmallVector<OpOperand *> outOfPlaceOpOperands;
    DenseSet<OpOperand *> copiedOpOperands;
    DenseSet<OpOperand *> escapingOpOperandCopies;

    // Find all out-of-place OpOperands.
    for (OpOperand &opOperand : op->getOpOperands()) {
      Type operandType = opOperand.get().getType();
      if (!operandType.isa<TensorType>()) continue;
      if (state.isInPlace(opOperand)) continue;
      if (operandType.isa<UnrankedTensorType>())
        return op->emitError("copies of unranked tensors are not supported");

      AliasingOpResultList aliasingOpResults =
          state.getAliasingOpResults(opOperand);
      // Is the result yielded from a block? Or are deallocations turned off
      // entirely? In either case, mark the allocation as "escaping", so that it
      // will not be deallocated.
      bool escape = !state.getOptions().createDeallocs ||
                    llvm::any_of(aliasingOpResults,
                                 [&](bufferization::AliasingOpResult a) {
                                   return state.isTensorYielded(a.opResult);
                                 });

      // In all other cases, make a copy of the OpOperand.
      outOfPlaceOpOperands.push_back(&opOperand);
      if (!state.canOmitTensorCopy(opOperand))
        copiedOpOperands.insert(&opOperand);
      if (escape) escapingOpOperandCopies.insert(&opOperand);
    }

    // Insert copies of OpOperands before the loop.
    rewriter.setInsertionPoint(op->getParentOp());
    for (OpOperand *opOperand : outOfPlaceOpOperands) {
      FailureOr<Value> copy = allocateTensorForShapedValue(
          rewriter, op->getLoc(), opOperand->get(),
          escapingOpOperandCopies.contains(opOperand), state.getOptions(),
          copiedOpOperands.contains(opOperand));
      if (failed(copy)) return failure();
      rewriter.updateRootInPlace(op, [&]() { opOperand->set(*copy); });
    }

    return success();
  }

  bool areEquivalentSlices(const AnalysisState &state,
                           ExtractSliceOp extractSliceOp, SetYieldOp setYieldOp,
                           int64_t updateIdx) const {
    if (!extractSliceOp || !setYieldOp) return false;
    if (extractSliceOp != setYieldOp &&
        !state.areEquivalentBufferizedValues(extractSliceOp.getSource(),
                                             setYieldOp.getDsts()[updateIdx])) {
      return false;
    }
    if (!sameOffsetsSizesAndStrides(
            extractSliceOp,
            setYieldOp.getSets()[updateIdx].getDefiningOp<TileOp>(),
            isEqualConstantIntOrValue))
      return false;
    return true;
  }

  /// Return true if `value` is originating from an ExtractSliceOp that matches
  /// the given SetYieldOp.
  bool matchesInsertDestination(const AnalysisState &state, Value value,
                                SetYieldOp setYieldOp,
                                int64_t updateIdx) const {
    // Look for matching slices.
    auto matchesSlice = [&](Value val) {
      if (auto materializeOp = val.getDefiningOp<ExtractSliceOp>()) {
        if (areEquivalentSlices(state, materializeOp, setYieldOp, updateIdx)) {
          return true;
        }
      }
      return false;
    };
    return llvm::all_of(
        state.findValueInReverseUseDefChain(value, matchesSlice), matchesSlice);
  }

  // Copied and modified for gml_st.materialize/gml_st.set_yield pairs from
  // mlir/lib/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.cpp
  // Takes into account that gml_st.set_yield can have multiple src/dst pairs.
  bool isNotConflicting(Operation *op, OpOperand *uRead,
                        OpOperand *uConflictingWrite,
                        const AnalysisState &state) const {
    Operation *readingOp = uRead->getOwner();
    Operation *conflictingWritingOp = uConflictingWrite->getOwner();

    // Special rules for matching SetYieldOp/ExtractSliceOp pairs. If
    // uRead is an SetYieldOp...
    if (auto setYieldOp = dyn_cast<SetYieldOp>(readingOp)) {
      for (int64_t updateIdx :
           llvm::seq<int64_t>(0, setYieldOp.getNumUpdates())) {
        OpOperand &srcOpOperand = setYieldOp->getOpOperand(updateIdx);
        OpOperand *dstOpOperand = setYieldOp.getDstOperand(updateIdx);

        if (uRead == dstOpOperand /*dest*/ &&
            matchesInsertDestination(state, uConflictingWrite->get(),
                                     setYieldOp, updateIdx))
          return true;

        if (uRead == &srcOpOperand /*source*/ &&
            uConflictingWrite == dstOpOperand /*dest*/ &&
            matchesInsertDestination(state, uRead->get(), setYieldOp,
                                     updateIdx))
          return true;
      }
    }

    // If uConflictingWrite is an SetYieldOp...
    if (auto setYieldOp = dyn_cast<SetYieldOp>(conflictingWritingOp)) {
      for (int64_t updateIdx :
           llvm::seq<int64_t>(0, setYieldOp.getNumUpdates())) {
        if (uConflictingWrite == setYieldOp.getDstOperand(updateIdx) &&
            state.areEquivalentBufferizedValues(
                uRead->get(), setYieldOp.getSrcs()[updateIdx]) &&
            matchesInsertDestination(state, setYieldOp.getSrcs()[updateIdx],
                                     setYieldOp, updateIdx))
          return true;
      }
    }

    return false;
  }
};

}  // namespace
}  // namespace gml_st
}  // namespace mlir

void mlir::gml_st::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, gml_st::GmlStDialect * /*dialect*/) {
        ParallelOp::attachInterface<ParallelOpInterface>(*ctx);
        SetYieldOp::attachInterface<SetYieldOpInterface>(*ctx);
      });
}
