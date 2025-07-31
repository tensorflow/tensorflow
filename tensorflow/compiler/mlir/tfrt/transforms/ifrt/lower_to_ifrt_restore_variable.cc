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

#include "llvm/ADT/SmallVector.h"
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
  // Returns true if the given user is a VarHandleOp.
  struct RestoredTensorUser {
    // Path to the AssignVariableOp from the RestoreV2Op. Those ops are deleted
    // after rewriting RestoreV2Op to IfrtRestoreVariableOp because we want to
    // handle the host tensor restore asynchronously in IfrtRestoreVariableOp.
    std::vector<mlir::Operation*> path_to_assign_variable_op;

    // The VarHandleOp associated with the AssignVariableOp.
    mlir::TF::VarHandleOp var_handle_op;
    bool truncate_in_cast =
        false;  // value of the truncate attribute in the CastOp.
                // Default to false if CastOp is not present.
  };

  mlir::LogicalResult ValidateThenUpdateUser(
      mlir::Operation* user,
      std::vector<RestoredTensorUser>& restored_tensor_users) {
    // Traverses the user chain of a restored tensor, validates it, and collects
    // user information.
    RestoredTensorUser restored_tensor_user;
    for (;;) {
      restored_tensor_user.path_to_assign_variable_op.push_back(user);
      // The user chain must consist of zero or more `TF::CastOp`s followed by
      // a `TF::AssignVariableOp`.
      if (auto cast_op = llvm::dyn_cast<mlir::TF::CastOp>(user)) {
        // The chain can contain intermediate `CastOp`s.
        if (!cast_op.getResult().hasOneUse()) {
          return cast_op->emitOpError()
                 << " has more than one use in the restore user chain";
        }
        restored_tensor_user.truncate_in_cast = cast_op.getTruncate();
        // Move to the next operation in the user chain.
        user = *cast_op.getResult().getUsers().begin();
      } else if (auto assign_variable_op =
                     llvm::dyn_cast<mlir::TF::AssignVariableOp>(user)) {
        // The chain must end with an `AssignVariableOp`.
        if (auto var_handle_op = llvm::dyn_cast<mlir::TF::VarHandleOp>(
                assign_variable_op.getResource().getDefiningOp())) {
          // The AssignVariableOp must be associated with a VarHandleOp.
          restored_tensor_user.var_handle_op = var_handle_op;
          break;
        } else {
          return assign_variable_op->emitOpError()
                 << "does not have any associated VarHandle";
        }
      } else {
        // Any other op in the user chain is not supported.
        return user->emitOpError() << "is not a supported user of RestoreV2Op";
      }
    }

    // If the user chain is valid, add the collected information.
    restored_tensor_users.push_back(restored_tensor_user);
    return mlir::success();
  }

  mlir::LogicalResult RewriteRestore(mlir::TF::RestoreV2Op restore_op) {
    // Find and validate all users of the RestoreV2Op's output tensors.
    std::vector<RestoredTensorUser> restored_tensor_users;

    for (const auto& out_tensor : restore_op.getTensors()) {
      for (mlir::Operation* user : out_tensor.getUsers()) {
        if (mlir::failed(ValidateThenUpdateUser(user, restored_tensor_users))) {
          return mlir::failure();
        }
      }
    }

    // Check that each tensor has exactly one valid user chain.
    if (restored_tensor_users.size() != restore_op.getTensors().size()) {
      return restore_op->emitOpError()
             << "expects " << restore_op.getTensors().size()
             << " valid users, but got " << restored_tensor_users.size();
    }

    // Collect tensor dtypes for the new op.
    std::vector<mlir::Attribute> dtypes;
    for (const auto& dtype : restore_op.getDtypes()) {
      dtypes.push_back(mlir::TypeAttr::get(dtype));
    }

    std::vector<mlir::Value> var_handle_values;
    // Collect attributes from users and delete the old user op chain.
    llvm::SmallVector<bool, 4> truncate_in_cast;
    var_handle_values.reserve(restored_tensor_users.size());
    truncate_in_cast.reserve(restored_tensor_users.size());
    for (auto& restored_tensor_user : restored_tensor_users) {
      var_handle_values.push_back(
          restored_tensor_user.var_handle_op.getResult());

      truncate_in_cast.push_back(restored_tensor_user.truncate_in_cast);

      // Delete the path from the RestoreV2Op to the AssignVariableOp in reverse
      // order.
      for (auto r = restored_tensor_user.path_to_assign_variable_op.rbegin();
           r != restored_tensor_user.path_to_assign_variable_op.rend(); ++r) {
        (*r)->erase();
      }
    }

    // Create the new IfrtRestoreVariableOp.
    // Insert at the end of the block so that all dependencies are satisfied.
    mlir::OpBuilder builder =
        mlir::OpBuilder::atBlockTerminator(restore_op->getBlock());
    mlir::TF::IfrtRestoreVariableOp::create(
        builder, restore_op->getLoc(), restore_op.getPrefix(),
        restore_op.getTensorNames(), restore_op.getShapeAndSlices(),
        var_handle_values, builder.getArrayAttr(dtypes),
        builder.getDenseBoolArrayAttr(truncate_in_cast));

    // Finally, erase the original RestoreV2Op.
    if (!restore_op->use_empty()) {
      return restore_op->emitOpError() << "failed to identify all users"
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
