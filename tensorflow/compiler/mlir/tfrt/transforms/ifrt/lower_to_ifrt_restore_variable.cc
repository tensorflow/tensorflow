/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <vector>

#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"  // IWYU pragma: keep
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DEF_LOWERTOIFRTRESTOREVARIABLEPASS
#define GEN_PASS_DECL_LOWERTOIFRTRESTOREVARIABLEPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

class LowerToIfrtRestoreVariablePass
    : public impl::LowerToIfrtRestoreVariablePassBase<
          LowerToIfrtRestoreVariablePass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    std::vector<mlir::TF::RestoreV2Op> restore_ops;
    module.walk([&](mlir::TF::RestoreV2Op restore_op) {
      restore_ops.push_back(restore_op);
    });

    for (const auto& restore_op : restore_ops) {
      if (mlir::failed(RewriteRestore(restore_op))) {
        return signalPassFailure();
      }
    }
  }

 private:
  mlir::LogicalResult RewriteRestore(mlir::TF::RestoreV2Op restore_op) {
    std::vector<mlir::Value> var_handle_values;
    std::vector<mlir::TF::AssignVariableOp> assign_variable_ops;

    var_handle_values.reserve(restore_op.getTensors().size());
    assign_variable_ops.reserve(restore_op.getTensors().size());
    for (const auto& out_tensor : restore_op.getTensors()) {
      for (mlir::Operation* user : out_tensor.getUsers()) {
        if (auto assign_variable_op =
                llvm::dyn_cast<mlir::TF::AssignVariableOp>(user)) {
          assign_variable_ops.push_back(assign_variable_op);
          if (auto var_handle_op = llvm::dyn_cast<mlir::TF::VarHandleOp>(
                  assign_variable_op.getResource().getDefiningOp())) {
            var_handle_values.push_back(var_handle_op.getResult());
          } else {
            return assign_variable_op->emitOpError()
                   << "does not have any associated VarHandle";
          }
        }
      }
    }

    if (var_handle_values.size() != restore_op.getTensors().size()) {
      return restore_op->emitOpError()
             << "expects " << restore_op.getTensors().size()
             << " VarHandleOps, but got " << var_handle_values.size();
    }

    std::vector<mlir::Attribute> dtypes;
    for (const auto& dtype : restore_op.getDtypes()) {
      dtypes.push_back(mlir::TypeAttr::get(dtype));
    }

    // Insert at the end of the block so that all dependencies are satisfied.
    mlir::OpBuilder builder =
        mlir::OpBuilder::atBlockTerminator(restore_op->getBlock());
    builder.create<mlir::TF::IfrtRestoreVariableOp>(
        restore_op->getLoc(), restore_op.getPrefix(),
        restore_op.getTensorNames(), restore_op.getShapeAndSlices(),
        var_handle_values, builder.getArrayAttr(dtypes));

    for (auto& assign_variable_op : assign_variable_ops) {
      assign_variable_op.erase();
    }
    if (!restore_op->use_empty()) {
      return restore_op->emitOpError()
             << "failed to identify all AssignVariableOps "
                "associated with this RestoreV2Op.";
    } else {
      restore_op.erase();
    }

    return mlir::success();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateLowerToIfrtRestoreVariablePass() {
  return std::make_unique<LowerToIfrtRestoreVariablePass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
