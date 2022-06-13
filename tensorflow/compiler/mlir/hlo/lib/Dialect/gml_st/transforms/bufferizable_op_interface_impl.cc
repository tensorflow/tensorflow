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
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationState;
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
                              const BufferizationState &state) const {
    auto loopOp = cast<LoopOp>(op);

    // gml_st.loop operands alone do not bufferize to a memory read, but
    // one of the uses of their matching bbArgs may.
    return state.getAnalysisState().isValueRead(
        loopOp.getTiedBlockArgument(opOperand));
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    // Only operands with an aliasing OpResult (i.e., output operands) bufferize
    // to a memory write.
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    return !bufferizableOp
                .getAliasingOpResult(opOperand, state.getAnalysisState())
                .empty();
  }

  SmallVector<OpResult> getAliasingOpResult(
      Operation *op, OpOperand &opOperand,
      const BufferizationState &state) const {
    auto loopOp = cast<LoopOp>(op);

    // Output operands are tied to their corresponding OpResults.
    OpResult opResult = loopOp.getTiedOpResult(opOperand);
    if (!opResult) return {};
    return {opResult};
  }

  BufferRelation bufferRelation(Operation *op, OpResult op_result,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  bool isWritable(Operation *op, Value value,
                  const BufferizationState &state) const {
    // Interestingly, LoopOp's bbArgs can **always** be viewed
    // inplace from the perspective of nested ops:
    //   1. Either the matching iter operand is not bufferized inplace and an
    //      alloc + optional copy makes the bbArg itself inplaceable.
    //   2. Or the matching iter operand is bufferized inplace and bbArg just
    //      bufferizes to that too.
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    auto loopOp = cast<LoopOp>(op);

    // Compute new inputs, outputs and results.
    SmallVector<Value> newInputs, newOutputs, newResults;
    for (unsigned i = loopOp.getNumControlOperands();
         i < loopOp->getNumOperands(); ++i) {
      OpOperand &operand = loopOp->getOpOperand(i);
      Value rewrittenValue = operand.get();
      if (rewrittenValue.getType().isa<TensorType>()) {
        FailureOr<Value> bufferOrFailure = state.getBuffer(rewriter, operand);
        if (failed(bufferOrFailure)) return failure();
        rewrittenValue = *bufferOrFailure;
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
      if (failed(state.getOptions().createMemCpy(
              rewriter, newTerminator.getLoc(), toMemrefOp, output)))
        return failure();
    }

    // Erase old terminator.
    rewriter.eraseOp(oldTerminator);

    // Replace results and delete old op.
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newResults);

    return success();
  }
};

}  // namespace
}  // namespace gml_st
}  // namespace mlir

void mlir::gml_st::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, gml_st::GmlStDialect *dialect) {
    LoopOp::attachInterface<LoopOpInterface>(*ctx);
  });
}
