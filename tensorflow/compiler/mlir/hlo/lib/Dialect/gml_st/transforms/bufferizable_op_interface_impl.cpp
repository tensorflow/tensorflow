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
  bool bufferizesToMemoryRead(Operation *op, OpOperand &op_operand,
                              const BufferizationState &state) const {
    auto loop_op = cast<LoopOp>(op);

    // gml_st.loop operands alone do not bufferize to a memory read, but
    // one of the uses of their matching bbArgs may.
    return state.getAnalysisState().isValueRead(
        loop_op.getTiedBlockArgument(op_operand));
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &op_operand,
                               const BufferizationState &state) const {
    // Only operands with an aliasing OpResult (i.e., output operands) bufferize
    // to a memory write.
    auto bufferizable_op = cast<BufferizableOpInterface>(op);
    return !bufferizable_op
                .getAliasingOpResult(op_operand, state.getAnalysisState())
                .empty();
  }

  SmallVector<OpResult> getAliasingOpResult(
      Operation *op, OpOperand &op_operand,
      const BufferizationState &state) const {
    auto loop_op = cast<LoopOp>(op);

    // Output operands are tied to their corresponding OpResults.
    OpResult op_result = loop_op.getTiedOpResult(op_operand);
    if (!op_result) return {};
    return {op_result};
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

  bool isAllocationHoistingBarrier(Operation *op) const { return true; }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    auto loop_op = cast<LoopOp>(op);

    // Compute new inputs, outputs and results.
    SmallVector<Value> new_inputs, new_outputs, new_results;
    for (unsigned i = loop_op.getNumControlOperands();
         i < loop_op->getNumOperands(); ++i) {
      OpOperand &operand = loop_op->getOpOperand(i);
      Value rewritten_value = operand.get();
      if (rewritten_value.getType().isa<TensorType>()) {
        FailureOr<Value> buffer_or_failure = state.getBuffer(rewriter, operand);
        if (failed(buffer_or_failure)) return failure();
        rewritten_value = *buffer_or_failure;
      }
      if (i < loop_op.getNumControlOperands() + loop_op.getNumInputs()) {
        new_inputs.push_back(rewritten_value);
      } else {
        new_outputs.push_back(rewritten_value);
        if (operand.get().getType().isa<TensorType>())
          new_results.push_back(rewritten_value);
      }
    }

    // Create new TiledLoopOp.
    auto new_loop_op = rewriter.create<LoopOp>(
        loop_op.getLoc(), loop_op.lowerBound(), loop_op.upperBound(),
        loop_op.step(), new_inputs, new_outputs, loop_op.iterator_types(),
        loop_op.distribution_types());

    // Remove terminator.
    if (!new_loop_op.getBody()->empty())
      rewriter.eraseOp(loop_op.getBody()->getTerminator());

    // Compute new loop body arguments.
    SmallVector<Value> new_block_args, new_region_in_out_args,
        old_region_in_out_args;
    ValueRange newInductionVars = new_loop_op.getInductionVars();
    new_block_args.append(newInductionVars.begin(), newInductionVars.end());

    ValueRange new_region_in_args = new_loop_op.getRegionInputArgs();
    ValueRange new_region_out_args = new_loop_op.getRegionOutputArgs();
    new_region_in_out_args.append(new_region_in_args.begin(),
                                  new_region_in_args.end());
    new_region_in_out_args.append(new_region_out_args.begin(),
                                  new_region_out_args.end());

    ValueRange old_region_in_args = loop_op.getRegionInputArgs();
    ValueRange old_region_out_args = loop_op.getRegionOutputArgs();
    old_region_in_out_args.append(old_region_in_args.begin(),
                                  old_region_in_args.end());
    old_region_in_out_args.append(old_region_out_args.begin(),
                                  old_region_out_args.end());
    assert(new_region_in_args.size() == old_region_in_args.size() &&
           "expected same number of input args");
    assert(new_region_out_args.size() == old_region_out_args.size() &&
           "expected same number of output args");

    for (auto it : llvm::zip(old_region_in_out_args, new_region_in_out_args)) {
      Value old_arg = std::get<0>(it);
      Value new_arg = std::get<1>(it);
      rewriter.setInsertionPointToStart(new_loop_op.getBody());
      if (old_arg.getType().isa<TensorType>()) {
        new_block_args.push_back(rewriter.create<bufferization::ToTensorOp>(
            old_arg.getLoc(), new_arg));
      } else {
        new_block_args.push_back(new_arg);
      }
    }

    // Move old body into new loop.
    rewriter.mergeBlocks(loop_op.getBody(), new_loop_op.getBody(),
                         new_block_args);

    // Replace previous terminator with a new one that does not yield anything.
    auto old_terminator =
        cast<gml_st::YieldOp>(new_loop_op.getBody()->getTerminator());
    rewriter.setInsertionPointToEnd(new_loop_op.getBody());
    auto new_terminator =
        rewriter.create<gml_st::YieldOp>(old_terminator->getLoc());

    // Copy buffer of yielded tensor to output buffer. If everything bufferized
    // inplace, this copy will fold away.
    rewriter.setInsertionPoint(new_terminator);
    for (auto it : llvm::zip(old_terminator.values(), new_outputs)) {
      Value output = std::get<1>(it);
      Value to_memref_op = rewriter.create<bufferization::ToMemrefOp>(
          new_terminator.getLoc(), output.getType(), std::get<0>(it));
      if (failed(createMemCpy(rewriter, new_terminator.getLoc(), to_memref_op,
                              output, state.getOptions())))
        return failure();
    }

    // Erase old terminator.
    rewriter.eraseOp(old_terminator);

    // Replace results and delete old op.
    bufferization::replaceOpWithBufferizedValues(rewriter, op, new_results);

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
