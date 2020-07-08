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

#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFTPU {
namespace {

constexpr char kTPUEmbeddingAttr[] = "_tpu_embedding_layer";

struct TPUUpdateEmbeddingEnqueueOpInputs
    : public PassWrapper<TPUUpdateEmbeddingEnqueueOpInputs, FunctionPass> {
  void runOnFunction() override;
};

// Extracts `_tpu_embedding_layer` attribute from TPU embedding ops and
// clear the attribute from the operation. This ensures that future optimization
// passes does not trigger additional logic due to presence of this attribute.
LogicalResult ExtractEmbeddingAttribute(
    Operation* op, std::map<std::string, mlir::Operation*>* embedding_op_map) {
  auto embedding_attr = op->getAttrOfType<StringAttr>(kTPUEmbeddingAttr);
  if (!embedding_attr)
    return op->emitOpError(
        "missing required attribute: `_tpu_embedding_layer`");

  auto it = embedding_op_map->emplace(embedding_attr.getValue().str(), op);
  if (!it.second)
    return op->emitOpError(
        "found duplicate tpu embedding ops. This usually happens when "
        "there are multiple TPUEmbedding layers.");

  op->removeAttr(kTPUEmbeddingAttr);
  return success();
}

LogicalResult FindTPUEmbeddingOps(
    FuncOp func_op, std::map<std::string, Operation*>* enqueue_op_map,
    std::map<std::string, Operation*>* recv_activation_op_map,
    std::map<std::string, Operation*>* send_gradient_op_map) {
  auto walk_result = func_op.walk([&](Operation* op) {
    if (llvm::isa<TF::RecvTPUEmbeddingActivationsOp>(op)) {
      if (failed(ExtractEmbeddingAttribute(op, recv_activation_op_map)))
        return WalkResult::interrupt();
    }

    if (llvm::isa<TF::SendTPUEmbeddingGradientsOp>(op)) {
      if (failed(ExtractEmbeddingAttribute(op, send_gradient_op_map)))
        return WalkResult::interrupt();
    }

    if (llvm::isa<TF::EnqueueTPUEmbeddingSparseTensorBatchOp>(op) ||
        llvm::isa<TF::EnqueueTPUEmbeddingRaggedTensorBatchOp>(op)) {
      if (failed(ExtractEmbeddingAttribute(op, enqueue_op_map)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) return failure();

  return success();
}

// Updates the operand of TPU embedding enqueue ops depending on whether
// the graph is in training mode or in non-training mode.
// If SendTPUEmbeddingGradients op is present, this means that graph is in
// training mode. As so, correctly feed in `then` branch value of SelectV2
// operand as inputs to the TPU embedding enqueue ops.
LogicalResult UpdateEmbeddingEnqueueOpInput(
    const std::map<std::string, Operation*>& enqueue_op_map,
    const std::map<std::string, Operation*>& recv_activation_op_map,
    const std::map<std::string, Operation*>& send_gradient_op_map) {
  for (const auto& it : enqueue_op_map) {
    const auto embedding_attr = it.first;
    Operation* embedding_op = it.second;
    if (!recv_activation_op_map.count(embedding_attr))
      return embedding_op->emitOpError(
          "TPU embedding enqueue op must have corresponding "
          "RecvTPUEmbeddingActivations op");

    // TPU Embedding enqueue ops take different inputs depending on whether
    // graph is in training mode or in eval/prediction mode. The inputs to the
    // enqueue ops are present/listed as operands to SelectV2 op. Then branch
    // operand of the SelectV2 op represents input to take during training
    // and else branch operand represents input to take during
    // prediction/evaluation. If SendTPUEmbeddingGradients op exists in the
    // graph, then graph is in training mode, so correctly forward the input
    // of SelectV2 op as operand to the TPU embedding enqueue op.
    bool is_training = send_gradient_op_map.count(embedding_attr);
    for (auto enqueue_operand : embedding_op->getOperands()) {
      if (auto select = llvm::dyn_cast_or_null<TF::SelectV2Op>(
              enqueue_operand.getDefiningOp())) {
        enqueue_operand.replaceAllUsesWith(is_training ? select.t()
                                                       : select.e());
      }
    }
  }

  return success();
}

void TPUUpdateEmbeddingEnqueueOpInputs::runOnFunction() {
  OpBuilder builder(&getContext());
  auto func_op = getFunction();

  // All TPU embedding layer related ops are annotated with
  // `_tpu_embedding_layer` attribute along with corresponding string attribute.
  // Store all tpu embedding layer related ops with value of
  // `_tpu_embedding_layer` attribute as map key.
  std::map<std::string, mlir::Operation*> enqueue_op_map;
  std::map<std::string, mlir::Operation*> recv_activation_op_map;
  std::map<std::string, mlir::Operation*> send_gradient_op_map;
  if (failed(FindTPUEmbeddingOps(func_op, &enqueue_op_map,
                                 &recv_activation_op_map,
                                 &send_gradient_op_map)))
    return signalPassFailure();

  if (enqueue_op_map.size() != recv_activation_op_map.size()) {
    func_op.emitError(
        "Number of embedding enqueue ops must match the number "
        "of RecvTPUEmbeddingActivation op");
    return signalPassFailure();
  }

  if (failed(UpdateEmbeddingEnqueueOpInput(
          enqueue_op_map, recv_activation_op_map, send_gradient_op_map)))
    return signalPassFailure();
}

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateTPUUpdateEmbeddingEnqueueOpInputsPass() {
  return std::make_unique<TPUUpdateEmbeddingEnqueueOpInputs>();
}

static PassRegistration<TPUUpdateEmbeddingEnqueueOpInputs> pass(
    "tf-tpu-update-embedding-enqueue-op-inputs",
    "Updates inputs to TPU embedding enqueue ops depending on whether graph "
    "is in training mode or in evaluation mode.");

}  // namespace TFTPU
}  // namespace mlir
