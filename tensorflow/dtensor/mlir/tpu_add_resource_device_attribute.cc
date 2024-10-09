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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORTPUADDRESOURCEDEVICEATTRIBUTE
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

constexpr char kFuncDeviceAttr[] = "tf.device";

// Adds device attribute to `arg` with the device placement of `execute_op`
void AddPlaceholderDeviceAttributeToResource(
    mlir::BlockArgument arg, mlir::TF::TPUExecuteOp execute_op) {
  // TPUExecute op is wrapped inside tf_device.Launch op for device assignment.
  auto tpu_execute_device_launch =
      execute_op->getParentOfType<mlir::tf_device::LaunchOp>();
  mlir::StringRef tpu_device_attr = tpu_execute_device_launch.getDevice();

  auto function = execute_op->getParentOfType<mlir::func::FuncOp>();
  mlir::OpBuilder builder(execute_op);
  function.setArgAttr(arg.getArgNumber(), kFuncDeviceAttr,
                      builder.getStringAttr(tpu_device_attr));
}

// Returns AssignVariableOp that consumes output of `val`. `val` is a output
// from TPUExecute op which is wrapped inside a single tf_device.Launch
// operation. As so, output of parent launch op is queried to identify connected
// AssignVariable op.
mlir::Operation* IdentifyConnectedAssignVariableOp(mlir::Value val) {
  for (mlir::OpOperand& use : val.getUses()) {
    auto return_op = llvm::dyn_cast<mlir::tf_device::ReturnOp>(use.getOwner());
    if (!return_op) continue;

    auto parent_launch =
        val.getDefiningOp()->getParentOfType<mlir::tf_device::LaunchOp>();
    mlir::Value launch_output = parent_launch.getResult(use.getOperandNumber());
    for (mlir::Operation* user : launch_output.getUsers()) {
      auto assign_variable = llvm::dyn_cast<mlir::TF::AssignVariableOp>(user);
      if (!assign_variable) continue;

      return assign_variable;
    }
  }
  return nullptr;
}

struct DTensorTpuAddResourceDeviceAttribute
    : public impl::DTensorTpuAddResourceDeviceAttributeBase<
          DTensorTpuAddResourceDeviceAttribute> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder op_builder(&context);
    mlir::ModuleOp module = getOperation();
    // For each resource value that is input or that is consumed by TPUExecute
    // op, add placeholder device attribute to the resource argument.
    mlir::WalkResult walk_result =
        module.walk([](mlir::TF::TPUExecuteOp tpu_execute) {
          for (mlir::Value tpu_input : tpu_execute.getOperands()) {
            if (mlir::isa<mlir::BlockArgument>(tpu_input) &&
                IsResourceType(tpu_input))
              AddPlaceholderDeviceAttributeToResource(
                  mlir::cast<mlir::BlockArgument>(tpu_input), tpu_execute);

            mlir::Operation* input_op = tpu_input.getDefiningOp();
            auto read_variable_op =
                llvm::dyn_cast_or_null<mlir::TF::ReadVariableOp>(input_op);
            if (!read_variable_op) continue;

            AddPlaceholderDeviceAttributeToResource(
                mlir::cast<mlir::BlockArgument>(read_variable_op.getResource()),
                tpu_execute);
          }

          for (mlir::Value result : tpu_execute.getResults()) {
            mlir::Operation* assign_variable =
                IdentifyConnectedAssignVariableOp(result);
            if (assign_variable == nullptr) continue;

            AddPlaceholderDeviceAttributeToResource(
                mlir::cast<mlir::BlockArgument>(
                    llvm::cast<mlir::TF::AssignVariableOp>(assign_variable)
                        .getResource()),
                tpu_execute);
          }

          return mlir::WalkResult::advance();
        });

    if (walk_result.wasInterrupted()) return signalPassFailure();
  };
};

}  // namespace

// Adds placeholder device attributes to resource arguments of TPU functions.
// Device attribute added is consistent with device placement of TPUExecute op.
// This is required for enabling CreateTPUMergeVariablesWithExecutePass as the
// pass checks that all resources must have consistent device placement with
// TPUExecute op in order to enable buffer aliasing.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorTpuAddResourceDeviceAttribute() {
  return std::make_unique<DTensorTpuAddResourceDeviceAttribute>();
}

}  // namespace dtensor
}  // namespace tensorflow
