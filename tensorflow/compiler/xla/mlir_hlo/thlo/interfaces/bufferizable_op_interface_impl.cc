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

#include "thlo/interfaces/bufferizable_op_interface_impl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir {
namespace thlo {
namespace {

using mlir::bufferization::AnalysisState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::BufferRelation;

// We can reuse the upstream implementation when DestinationStyleOpInterface
// is moved out of linalg.
static LogicalResult bufferizeDestinationStyleOpInterface(
    RewriterBase &rewriter, DestinationStyleOpInterface op,
    const BufferizationOptions &options) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (op.hasBufferSemantics()) return success();

  if (!op.hasTensorSemantics())
    return op->emitError() << "expected either buffer or tensor semantics";

  size_t numOutputs = op.getNumDpsInits();

  // New operands for the cloned op.
  SmallVector<Value> newOperands;
  newOperands.reserve(op.getNumDpsInputs() + numOutputs);

  for (OpOperand *opOperand : op.getDpsInputOperands()) {
    if (op.isScalar(opOperand)) {
      newOperands.push_back(opOperand->get());
      continue;
    }
    FailureOr<Value> buffer = getBuffer(rewriter, opOperand->get(), options);
    if (failed(buffer)) return failure();
    newOperands.push_back(*buffer);
  }

  // New output operands for the cloned op.
  SmallVector<Value> newOutputs;
  newOutputs.reserve(numOutputs);

  for (OpResult opResult : op->getOpResults()) {
    OpOperand *opOperand = op.getDpsInitOperand(opResult.getResultNumber());
    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, opOperand->get(), options);
    if (failed(resultBuffer)) return failure();
    newOutputs.push_back(*resultBuffer);
  }

  newOperands.append(newOutputs.begin(), newOutputs.end());

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);

  // Clone the op, but use the new operands. Move the existing block into the
  // new op. Since the new op does not have any tensor results, it does not
  // return anything.
  auto newOp = cast<DestinationStyleOpInterface>(cloneWithoutRegions(
      rewriter, op, /*resultTypes=*/TypeRange{}, newOperands));

  assert(op->getNumRegions() <= 1);
  if (op->getNumRegions() == 1) {
    rewriter.inlineRegionBefore(op->getRegion(0), newOp->getRegion(0),
                                newOp->getRegion(0).begin());
  }

  // Replace the results of the old op with the new output buffers.
  bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);

  return success();
}

struct ThloSortOpBufferizationModel
    : public BufferizableOpInterface::ExternalModel<
          ThloSortOpBufferizationModel, SortOp> {
  bool bufferizesToMemoryRead(Operation * /*op*/, OpOperand & /*opOperand*/,
                              const AnalysisState & /*state*/) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState & /*state*/) const {
    return cast<DestinationStyleOpInterface>(op).isDpsInit(&opOperand);
  }

  SmallVector<OpOperand *> getAliasingOpOperand(
      Operation *op, OpResult opResult, const AnalysisState & /*state*/) const {
    auto dstStyleOp = cast<DestinationStyleOpInterface>(op);

    // The i-th OpResult may alias with the i-th "out" tensor.
    return {dstStyleOp.getDpsInitOperand(opResult.getResultNumber())};
  }

  SmallVector<OpResult> getAliasingOpResult(
      Operation *op, OpOperand &opOperand,
      const AnalysisState & /*state*/) const {
    auto dstStyleOp = cast<DestinationStyleOpInterface>(op);

    // The i-th "out" tensor may alias with the i-th OpResult.
    if (dstStyleOp.isDpsInit(&opOperand))
      return {dstStyleOp.getTiedOpResult(&opOperand)};
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }

  BufferRelation bufferRelation(Operation * /*op*/, OpResult /*opResult*/,
                                const AnalysisState & /*state*/) const {
    return BufferRelation::Equivalent;
  }
};

}  // namespace

}  // namespace thlo
}  // namespace mlir

void mlir::thlo::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, thlo::THLODialect * /*dialect*/) {
    SortOp::attachInterface<ThloSortOpBufferizationModel>(*ctx);
  });
}
