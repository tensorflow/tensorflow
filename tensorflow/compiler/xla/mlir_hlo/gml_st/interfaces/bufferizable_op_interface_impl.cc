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

#include <optional>
#include <tuple>

#include "gml_st/IR/gml_st_ops.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace gml_st {
namespace {

using mlir::bufferization::AliasingOpOperandList;
using mlir::bufferization::AliasingOpResultList;
using mlir::bufferization::AnalysisState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::BufferRelation;

struct FusionOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          FusionOpBufferizationInterface, FusionOp> {
  bool bufferizesToMemoryRead(Operation * /*op*/, OpOperand & /*opOperand*/,
                              const AnalysisState & /*state*/) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState & /*state*/) const {
    return cast<FusionOp>(op).isDpsInit(&opOperand);
  }

  AliasingOpOperandList getAliasingOpOperands(
      Operation *op, OpResult opResult, const AnalysisState & /*state*/) const {
    auto fusionOp = cast<FusionOp>(op);

    // The i-th OpResult aliases with the i-th "out" tensor.
    return {{fusionOp.getDpsInitOperand(opResult.getResultNumber()),
             BufferRelation::Equivalent}};
  }

  AliasingOpResultList getAliasingOpResults(
      Operation *op, OpOperand &opOperand,
      const AnalysisState & /*state*/) const {
    auto fusionOp = cast<FusionOp>(op);

    // The i-th "out" tensor aliases with the i-th OpResult.
    if (fusionOp.isDpsInit(&opOperand)) {
      return {
          {fusionOp.getTiedOpResult(&opOperand), BufferRelation::Equivalent}};
    }
    return {};
  }

  bool isWritable(Operation * /*op*/, Value /*value*/,
                  const AnalysisState & /*state*/) const {
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);

    auto loc = op->getLoc();
    FusionOp fusionOp = cast<FusionOp>(op);

    // Nothing to do. This op is already bufferized.
    if (fusionOp.hasBufferSemantics()) return success();

    if (!fusionOp.hasTensorSemantics()) {
      return op->emitError() << "expected either buffer or tensor semantics";
    }

    size_t numOutputs = fusionOp.getNumDpsInits();

    // New operands for the cloned op.
    SmallVector<Value> newOperands;
    newOperands.reserve(fusionOp.getNumDpsInputs() + numOutputs);

    for (OpOperand *opOperand : fusionOp.getDpsInputOperands()) {
      if (fusionOp.isScalar(opOperand)) {
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

    for (OpResult opResult : fusionOp->getOpResults()) {
      OpOperand *opOperand =
          fusionOp.getDpsInitOperand(opResult.getResultNumber());
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
    auto newFusionOp = cast<FusionOp>(cloneWithoutRegions(
        rewriter, op, /*resultTypes=*/TypeRange{}, newOperands));

    // Create empty region in the new bufferized op.
    Region &region = newFusionOp.getRegion();
    SmallVector<Type, 4> blockArgTypes =
        llvm::to_vector(TypeRange(ValueRange(newOperands)));
    SmallVector<Location, 4> blockArgLocs(blockArgTypes.size(), loc);
    rewriter.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);

    ArrayRef<BlockArgument> bbArgs =
        newFusionOp.getRegion().front().getArguments();
    SmallVector<Value> bbArgsToTensors;
    for (auto buf : bbArgs) {
      if (isa<MemRefType>(buf.getType())) {
        Value tensor = rewriter.create<bufferization::ToTensorOp>(loc, buf);
        bbArgsToTensors.push_back(tensor);
      } else {
        bbArgsToTensors.push_back(buf);
      }
    }

    // Move old body into new fusion op.
    rewriter.mergeBlocks(fusionOp.getBody(), newFusionOp.getBody(),
                         bbArgsToTensors);

    // Copy results to output memrefs. In most of the cases it's not necessary,
    // because clusters are constructed in a way that the result is produced by
    // an dst-style op that already put everything in the output memrefs, but
    // there are corner cases when it doesn't happen. For example, tiled 1d
    // linalg.reduce.
    rewriter.setInsertionPoint(newFusionOp.getTerminator());
    for (auto [bbArg, resultValue] :
         llvm::zip(bbArgs.take_back(numOutputs),
                   newFusionOp.getTerminator().getValues())) {
      if (auto toTensorOp =
              resultValue.getDefiningOp<bufferization::ToTensorOp>()) {
        rewriter.create<memref::CopyOp>(loc, toTensorOp.getMemref(), bbArg);
      }
    }

    // Replace gml_st.yield values with output buffers.
    rewriter.replaceOpWithNewOp<gml_st::YieldOp>(newFusionOp.getTerminator(),
                                                 bbArgs.take_back(numOutputs));

    // Replace the results of the old op with the new output buffers.
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);

    return success();
  }

  FailureOr<BaseMemRefType> getBufferType(
      Operation *op, Value value, const BufferizationOptions &options,
      const DenseMap<Value, BaseMemRefType> &fixedTypes) const {
    auto fusionOp = cast<FusionOp>(op);

    if (auto bbArg = value.dyn_cast<BlockArgument>()) {
      // A tensor block argument has the same bufferized type as the
      // corresponding output operand.
      return bufferization::getBufferType(
          fusionOp->getOpOperand(bbArg.getArgNumber()).get(), options,
          fixedTypes);
    }

    // The bufferized result type is the same as the bufferized type of the
    // corresponding output operand.
    return bufferization::getBufferType(
        fusionOp.getDpsInitOperand(value.cast<OpResult>().getResultNumber())
            ->get(),
        options, fixedTypes);
  }
};

}  // namespace
}  // namespace gml_st
}  // namespace mlir

void mlir::gml_st::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, gml_st::GmlStDialect * /*dialect*/) {
        FusionOp::attachInterface<FusionOpBufferizationInterface>(*ctx);
      });
}
