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
#include <string>
#include <utility>

#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

// This pass marks unsupported ops in a device cluster with
// `_xla_outside_compilation` attribute so the operations will run on the host
// instead of the device.  Unsupported ops are ops that can not be code
// generated to run on the device for the cluster.
struct MarkOpsForOutsideCompilation
    : public PassWrapper<MarkOpsForOutsideCompilation,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

// TODO(b/159128666): Check the control flow legalization passes instead once
// added.
void AddSupportedControlFlowOps(MLIRContext* context,
                                llvm::DenseSet<OperationName>* supported_ops) {
  supported_ops->insert(OperationName("tf.IfRegion", context));
  supported_ops->insert(OperationName("tf.Yield", context));
}

bool HasStringOperand(Operation& op) {
  for (auto operand : op.getOperands()) {
    if (getElementTypeOrSelf(operand).isa<TF::StringType>()) return true;
  }
  return false;
}

bool HasStringResult(Operation& op) {
  for (auto result : op.getResults()) {
    if (getElementTypeOrSelf(result).isa<TF::StringType>()) return true;
  }
  return false;
}

bool MatchesPattern(Operation& op,
                    const llvm::DenseSet<OperationName>& supported_ops) {
  return (supported_ops.contains(op.getName()));
}

// Checks if the op is supported inside of a device cluster.
bool IsSupportedOp(Operation& op,
                   const llvm::DenseSet<OperationName>& supported_ops) {
  // TODO(b/161726307): Check the allowed ops list in LegalizeTfWithTf2XlaPass
  // as well.
  return !HasStringOperand(op) && !HasStringResult(op) &&
         (MatchesPattern(op, supported_ops) ||
          mhlo::IsOpAllowedTf2XlaFallback(&op));
}

bool HasCapturedStringOperand(TF::IfRegionOp* if_op) {
  bool string_operand = false;
  mlir::visitUsedValuesDefinedAbove(
      if_op->then_branch(), if_op->then_branch(),
      [&](mlir::OpOperand* operand) {
        if (getElementTypeOrSelf(operand->get()).isa<TF::StringType>())
          string_operand = true;
      });
  if (string_operand) return string_operand;
  mlir::visitUsedValuesDefinedAbove(
      if_op->else_branch(), if_op->else_branch(),
      [&](mlir::OpOperand* operand) {
        if (getElementTypeOrSelf(operand->get()).isa<TF::StringType>())
          string_operand = true;
      });
  return string_operand;
}

LogicalResult MarkUncompilableOps(
    Block* block, llvm::DenseSet<OperationName>& supported_ops) {
  block->walk([&](Operation* op) {
    if (!IsSupportedOp(*op, supported_ops)) {
      op->setAttr(kXlaOutsideCompilationAttr,
                  StringAttr::get("auto", op->getContext()));
    }
    if (auto if_op = llvm::dyn_cast<TF::IfRegionOp>(op)) {
      if (HasCapturedStringOperand(&if_op)) {
        op->setAttr(kXlaOutsideCompilationAttr,
                    StringAttr::get("auto", op->getContext()));
      }
    }
  });
  return success();
}

void MarkOpsForOutsideCompilation::runOnOperation() {
  auto module = getOperation();
  OwningRewritePatternList patterns;
  mhlo::PopulateLegalizeTfPatterns(module.getContext(), &patterns);

  // `supported_ops` contains the name of all of the ops that can potentially be
  // lowered into HLO on the device. This doesn't always mean that the op can
  // be lowered in the future passes but if the op is not in this set, it can't
  // be lowered in a subsequent pass.
  llvm::DenseSet<OperationName> supported_ops;
  for (auto& pattern : patterns) {
    supported_ops.insert(*pattern->getRootKind());
  }
  AddSupportedControlFlowOps(module.getContext(), &supported_ops);

  auto result = module.walk([&](tf_device::ClusterOp cluster) {
    if (failed(MarkUncompilableOps(&cluster.GetBody(), supported_ops)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateMarkOpsForOutsideCompilationPass() {
  return std::make_unique<MarkOpsForOutsideCompilation>();
}

static PassRegistration<MarkOpsForOutsideCompilation> pass(
    "tf-mark-ops-for-outside-compilation",
    "Marks unsupported ops a device cluster for outside compilation.");

}  // namespace TFDevice
}  // namespace mlir
