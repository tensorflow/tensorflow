/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFTPU {
namespace {

constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";
constexpr char kTPUEmbeddingAttr[] = "_tpu_embedding_layer";

#define GEN_PASS_DEF_TPUUPDATEEMBEDDINGENQUEUEOPINPUTSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct TPUUpdateEmbeddingEnqueueOpInputsPass
    : public impl::TPUUpdateEmbeddingEnqueueOpInputsPassBase<
          TPUUpdateEmbeddingEnqueueOpInputsPass> {
  void runOnOperation() override;
};

// Extracts `_tpu_embedding_layer` attribute from TPU embedding ops and
// clear the attribute from the operation. This ensures that future optimization
// passes does not trigger additional logic due to presence of this attribute.
LogicalResult ExtractEmbeddingAttribute(
    Operation* op, llvm::StringMap<Operation*>* embedding_op_map) {
  auto embedding_attr = op->getAttrOfType<StringAttr>(kTPUEmbeddingAttr);
  if (!embedding_attr) return mlir::success();

  if (!embedding_op_map->insert({embedding_attr.getValue(), op}).second)
    return op->emitOpError(
        "found duplicate TPU embedding ops potentially from multiple "
        "TPUEmbedding layers");

  op->removeAttr(kTPUEmbeddingAttr);
  return success();
}

LogicalResult FindTPUEmbeddingOps(
    func::FuncOp func_op, llvm::StringMap<Operation*>* enqueue_op_map,
    llvm::StringMap<Operation*>* recv_activation_op_map,
    llvm::StringMap<Operation*>* send_gradient_op_map) {
  auto walk_result = func_op.walk([&](Operation* op) {
    if (llvm::isa<TF::RecvTPUEmbeddingActivationsOp>(op))
      if (failed(ExtractEmbeddingAttribute(op, recv_activation_op_map)))
        return WalkResult::interrupt();

    if (llvm::isa<TF::SendTPUEmbeddingGradientsOp>(op))
      if (failed(ExtractEmbeddingAttribute(op, send_gradient_op_map)))
        return WalkResult::interrupt();

    if (llvm::isa<TF::EnqueueTPUEmbeddingSparseTensorBatchOp,
                  TF::EnqueueTPUEmbeddingRaggedTensorBatchOp,
                  TF::EnqueueTPUEmbeddingArbitraryTensorBatchOp>(op))
      if (failed(ExtractEmbeddingAttribute(op, enqueue_op_map)))
        return WalkResult::interrupt();

    return WalkResult::advance();
  });
  return failure(walk_result.wasInterrupted());
}

// Updates the operand of TPU embedding enqueue ops depending on whether
// the graph is in training mode or in non-training mode.
// If SendTPUEmbeddingGradients op is present, this means that graph is in
// training mode. As so, correctly feed in `then` branch value of SelectV2
// operand as inputs to the TPU embedding enqueue ops.
LogicalResult UpdateEmbeddingEnqueueOpInput(
    const llvm::StringMap<Operation*>& enqueue_op_map,
    const llvm::StringMap<Operation*>& recv_activation_op_map,
    const llvm::StringMap<Operation*>& send_gradient_op_map,
    OpBuilder* builder) {
  for (const auto& it : enqueue_op_map) {
    const auto& embedding_attr = it.getKey();
    Operation* embedding_op = it.second;
    if (!recv_activation_op_map.count(embedding_attr))
      return embedding_op->emitOpError()
             << "must have a corresponding '"
             << TF::RecvTPUEmbeddingActivationsOp::getOperationName() << "' op";

    // TPU Embedding enqueue ops take different inputs depending on whether
    // graph is in training mode or in eval/prediction mode. During training,
    // the mode parameter for TPUEmbeddingEnqueue op must be `train` and for
    // evaluation or prediction, mode must be set to `inference`.
    // If SendTPUEmbeddingGradients op exists in the graph, then graph is
    // in training mode, so create a const op with value `train` use the
    // output value of the constant as an operand to the TPU embedding
    // enqueue op.
    bool is_training = send_gradient_op_map.count(embedding_attr);

    // The last operand of TPUEmbeddingEnqueue ops is the mode which
    // represents whether graph is in training mode or in evaluation mode.
    auto& mode_enqueue_operand =
        embedding_op->getOpOperand(embedding_op->getNumOperands() - 1);

    llvm::SmallVector<StringRef, 1> mode_string_value;
    mode_string_value.emplace_back(is_training ? "train" : "inference");
    builder->setInsertionPoint(embedding_op);
    auto enqueue_mode = builder->create<TF::ConstOp>(
        embedding_op->getLoc(),
        DenseStringElementsAttr::get(
            RankedTensorType::get({}, builder->getType<TF::StringType>()),
            mode_string_value));

    auto outside_compilation_attr =
        embedding_op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr);
    if (outside_compilation_attr)
      enqueue_mode->setAttr(kXlaOutsideCompilationAttr,
                            outside_compilation_attr);

    mode_enqueue_operand.set(enqueue_mode);
  }

  return success();
}

void TPUUpdateEmbeddingEnqueueOpInputsPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto func_op = getOperation();

  // All TPU embedding layer related ops are annotated with
  // `_tpu_embedding_layer` attribute along with corresponding string attribute.
  // Store all tpu embedding layer related ops with value of
  // `_tpu_embedding_layer` attribute as map key.
  llvm::StringMap<Operation*> enqueue_op_map;
  llvm::StringMap<Operation*> recv_activation_op_map;
  llvm::StringMap<Operation*> send_gradient_op_map;
  if (failed(FindTPUEmbeddingOps(func_op, &enqueue_op_map,
                                 &recv_activation_op_map,
                                 &send_gradient_op_map)))
    return signalPassFailure();

  if (enqueue_op_map.size() != recv_activation_op_map.size()) {
    func_op.emitError() << "expects the number of embedding enqueue ops to "
                           "match the number of '"
                        << TF::RecvTPUEmbeddingActivationsOp::getOperationName()
                        << "' ops";
    return signalPassFailure();
  }

  if (failed(UpdateEmbeddingEnqueueOpInput(enqueue_op_map,
                                           recv_activation_op_map,
                                           send_gradient_op_map, &builder)))
    return signalPassFailure();
}

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateTPUUpdateEmbeddingEnqueueOpInputsPass() {
  return std::make_unique<TPUUpdateEmbeddingEnqueueOpInputsPass>();
}

}  // namespace TFTPU
}  // namespace mlir
