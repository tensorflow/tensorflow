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

#include <memory>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/mlir/dtensor_send_recv.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORINFERSHAPESFORRESTOREV2OP
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

// From the Operation that produces `value`, set the result type to `type`.
//
// Recursively set the result type to `type` going backward toward
// the tf.RestoreV2Op that produced the unknown shape associated with `value`.
mlir::LogicalResult BackwardShapeInferenceToRestoreOp(mlir::ModuleOp module,
                                                      mlir::OpBuilder* builder,
                                                      mlir::Value value,
                                                      mlir::Type type) {
  mlir::Operation* op = value.getDefiningOp();
  if (op == nullptr) return mlir::success();
  if (!llvm::isa<mlir::TF::IdentityOp, mlir::TF::CastOp, mlir::TF::DTensorRecv,
                 mlir::TF::RestoreV2Op>(op)) {
    return op->emitOpError(
        llvm::formatv("Expected an Identity, Cast, DTensorRecv, or RestoreV2 "
                      "op, but got: {0}. Please file a bug to the DTensor team."
                      "(component id: 833864)",
                      op->getName().getStringRef()));
  }

  builder->setInsertionPointAfter(op);

  // Base case: If we got to the RestoreV2Op, then we got to the root
  // of the unknown shape result. Set the type to `type` of the result index
  // from `value`.
  if (auto restore_op = llvm::dyn_cast_or_null<mlir::TF::RestoreV2Op>(op)) {
    // This is usually a dangerous operation, but since we are backward
    // propagating shapes and correctly setting the shapes backwards,
    // we can modify the value itself here instead of creating a new
    // RestoreV2 op.
    //
    // Creating a new RestoreV2 op and replacing all uses will make this
    // algorithm run in O(N^2) where N = number of outputs of RestoreV2.
    //
    // Using setType(type) modifies in place and makes this algorithm run in
    // O(N).
    value.setType(type);
  } else if (auto cast_op = llvm::dyn_cast_or_null<mlir::TF::CastOp>(op)) {
    auto new_cast_op = builder->create<mlir::TF::CastOp>(cast_op.getLoc(), type,
                                                         cast_op.getOperand());
    cast_op.replaceAllUsesWith(new_cast_op.getResult());
    cast_op.erase();

    // Cast ops have differing operand and output element type, so update
    // the type to the operand element type.
    mlir::RankedTensorType new_type = mlir::RankedTensorType::get(
        GetShapeOfValue(new_cast_op.getResult()).value(),
        new_cast_op.getOperand()
            .getType()
            .cast<mlir::TensorType>()
            .getElementType());

    // Recursively shape inference to the input of the cast op with the
    // new type.
    return BackwardShapeInferenceToRestoreOp(
        module, builder, new_cast_op.getOperand(), new_type);
  } else if (auto identity_op =
                 llvm::dyn_cast_or_null<mlir::TF::IdentityOp>(op)) {
    auto new_identity_op = builder->create<mlir::TF::IdentityOp>(
        identity_op.getLoc(), type, identity_op.getInput());
    identity_op.getOutput().replaceAllUsesWith(new_identity_op.getOutput());
    identity_op.erase();

    // Recursively shape inference to the input of the identity op.
    return BackwardShapeInferenceToRestoreOp(module, builder,
                                             new_identity_op.getInput(), type);
  } else if (auto recv_op = llvm::dyn_cast_or_null<mlir::TF::DTensorRecv>(op)) {
    // If we have a DTensorRecv, then there is cross mesh action and the
    // RestoreV2Op we want to fix is on the mesh of the corresponding
    // DTensorSend. Set shape of this DTensorRecv first and go to the
    // corresponding DTensorSend.
    auto new_recv_op = builder->create<mlir::TF::DTensorRecv>(
        recv_op.getLoc(), type, builder->getStringAttr(recv_op.getKey()),
        mlir::TF::ShapeAttr::get(builder->getContext(),
                                 type.dyn_cast<mlir::TensorType>()),
        mlir::dtensor::LayoutAttr::get(builder->getContext(),
                                       recv_op.getLayout()));

    recv_op.replaceAllUsesWith(new_recv_op.getOutput());
    recv_op.erase();

    auto send_op = GetCorrespondingDTensorSendRecvOp<mlir::TF::DTensorRecv>(
        module, new_recv_op);

    if (!send_op.ok())
      return recv_op.emitOpError(tsl::NullTerminatedMessage(send_op.status()));

    // Recursively shape inference to the input of the send op.
    return BackwardShapeInferenceToRestoreOp(
        module, builder, send_op.value()->getOperand(0), type);
  }
  return mlir::success();
}

// From every AssignVariableOp, if the value X that we are assigning to the
// resource tensor has unknown shape information, then value X might be
// from the result of a tf.RestoreV2 op.
//
// We can infer the unknown shape of the result of a tf.RestoreV2 op through
// the resource tensors of AssignVariableOps that consume the results.
//
// Thus, we propagate the underlying resource tensor shape and dtype backwards
// leading up to the tf.RestoreV2 op.
mlir::LogicalResult PropagateShapeInformationFromAssignVariableOp(
    mlir::ModuleOp module) {
  module.walk([&](mlir::TF::AssignVariableOp assign_op) {
    // Check that the `value` has an unknown shape.
    if (ValueRank(assign_op.getValue()) == -1) {
      StatusOr<llvm::ArrayRef<int64_t>> shape =
          GetShapeOfValue(assign_op.getResource());
      if (!shape.ok()) {
        assign_op->emitOpError(
            "Resource tensor was expected to have shape information but was "
            "missing it during CheckpointShapeInference.");
        return mlir::WalkResult::interrupt();
      }
      // Propagete shape backwards to all the ops that use or produce
      // the value with missing shape.
      mlir::OpBuilder builder(assign_op);
      mlir::Type known_type = GetSubtypeOrSelf(assign_op.getResource());
      if (mlir::failed(BackwardShapeInferenceToRestoreOp(
              module, &builder, assign_op.getValue(), known_type))) {
        assign_op->emitOpError(
            "Error doing Backward shape inference from AssignVariableOp during "
            "CheckpointShapeInference.");
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });

  return mlir::success();
}

struct DTensorInferShapesForRestoreV2Op
    : public impl::DTensorInferShapesForRestoreV2OpBase<
          DTensorInferShapesForRestoreV2Op> {
  void runOnOperation() override {
    auto module = getOperation();
    if (failed(PropagateShapeInformationFromAssignVariableOp(module)))
      return signalPassFailure();
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorInferShapesForRestoreV2Op() {
  return std::make_unique<DTensorInferShapesForRestoreV2Op>();
}

}  // namespace dtensor
}  // namespace tensorflow
