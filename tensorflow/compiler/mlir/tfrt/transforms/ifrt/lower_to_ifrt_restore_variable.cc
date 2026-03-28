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
#include <optional>
#include <vector>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/ir/types/dialect.h"
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
  // Holds information about a user of a restored tensor from RestoreV2Op.
  struct RestoreOpUser {
    // The chain of ops from RestoreV2Op's user. If var_handle_op is
    // not null, this chain ends with AssignVariableOp and will be deleted.
    // Otherwise, restored_tensor_to_return will be set, and this chain
    // ends with an op that uses the restored tensor, which should be returned
    // by IfrtRestoreVariableOp.
    std::vector<mlir::Operation*> op_chain_from_restore_output;

    // The name of the restored tensor.
    mlir::StringRef tensor_name;

    // The VarHandleOp associated with the AssignVariableOp.
    mlir::TF::VarHandleOp var_handle_op;
    bool truncate_in_cast =
        false;  // value of the truncate attribute in the CastOp.
                // Default to false if CastOp is not present.

    // The restored tensor that is not assigned to a variable and should be
    // returned by IfrtRestoreVariableOp.
    std::optional<mlir::Value> restored_tensor_to_return;
  };

  mlir::LogicalResult ValidateThenUpdateUser(
      mlir::Value source_tensor, mlir::Operation* user,
      mlir::StringRef tensor_name,
      std::vector<RestoreOpUser>& restored_tensor_users) {
    // Traverses the user chain of a restored tensor, validates it, and collects
    // user information.
    RestoreOpUser restored_tensor_user;
    restored_tensor_user.tensor_name = tensor_name;
    for (;;) {
      restored_tensor_user.op_chain_from_restore_output.push_back(user);
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
        // If the restored tensor is not assigned to a variable, it is returned
        // by the IfrtRestoreVariableOp. A VarHandleOp is created to track the
        // restored tensor.
        mlir::OpBuilder builder(user);

        auto resource_type = mlir::RankedTensorType::get(
            {}, mlir::tf_type::ResourceType::get(
                    {mlir::cast<mlir::TensorType>(source_tensor.getType())},
                    builder.getContext()));
        mlir::OpBuilder var_handle_builder =
            mlir::OpBuilder::atBlockBegin(user->getBlock());
        auto var_handle_op = var_handle_builder.create<mlir::TF::VarHandleOp>(
            user->getLoc(), resource_type,
            /*container=*/builder.getStringAttr(""),
            /*shared_name=*/builder.getStringAttr(tensor_name));

        restored_tensor_user.var_handle_op = var_handle_op;
        restored_tensor_user.restored_tensor_to_return = source_tensor;
        break;
      }
    }

    // If the user chain is valid, add the collected information.
    restored_tensor_users.push_back(restored_tensor_user);
    return mlir::success();
  }

  mlir::LogicalResult RewriteRestore(mlir::TF::RestoreV2Op restore_op) {
    // Find and validate all users of the RestoreV2Op's output tensors.
    std::vector<RestoreOpUser> restored_tensor_users;

    mlir::DenseStringElementsAttr tensor_names_attr;
    if (!matchPattern(restore_op.getTensorNames(),
                      m_Constant(&tensor_names_attr))) {
      return restore_op.emitOpError("expects tensor_names to be a constant");
    }
    auto name_values = tensor_names_attr.getValues<mlir::StringRef>();
    llvm::SmallVector<mlir::StringRef> tensor_names_vec(name_values.begin(),
                                                        name_values.end());

    if (restore_op.getTensors().size() != tensor_names_vec.size()) {
      return restore_op.emitOpError(absl::StrCat(
          "expects the number of tensors and tensor names to be the same, got ",
          restore_op.getTensors().size(), " tensors and ",
          tensor_names_vec.size(), " tensor names."));
    }

    for (int i = 0; i < restore_op.getTensors().size(); ++i) {
      mlir::Value out_tensor = restore_op.getTensors()[i];
      auto tensor_name = tensor_names_vec[i];
      for (mlir::Operation* user : out_tensor.getUsers()) {
        if (mlir::failed(ValidateThenUpdateUser(out_tensor, user, tensor_name,
                                                restored_tensor_users))) {
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

    llvm::SmallVector<mlir::Type> result_types;
    llvm::DenseMap<mlir::Value, int> returned_tensors_indices;
    llvm::SmallVector<mlir::StringRef, 4> returned_tensor_names;
    for (auto& restored_tensor_user : restored_tensor_users) {
      truncate_in_cast.push_back(restored_tensor_user.truncate_in_cast);

      // This tensor is assigned to a variable.
      var_handle_values.push_back(
          restored_tensor_user.var_handle_op.getResult());
      if (!restored_tensor_user.restored_tensor_to_return.has_value()) {
        // Delete the path from the RestoreV2Op to the AssignVariableOp in
        // reverse order.
        for (auto r =
                 restored_tensor_user.op_chain_from_restore_output.rbegin();
             r != restored_tensor_user.op_chain_from_restore_output.rend();
             ++r) {
          (*r)->erase();
        }
      } else {
        // This tensor is not assigned to a variable and should be returned by
        // IfrtRestoreVariableOp.
        mlir::Value tensor = *restored_tensor_user.restored_tensor_to_return;
        if (returned_tensors_indices.find(tensor) ==
            returned_tensors_indices.end()) {
          returned_tensors_indices.insert({tensor, result_types.size()});
          result_types.push_back(tensor.getType());
          returned_tensor_names.push_back(restored_tensor_user.tensor_name);
        }
      }
    }

    // Create the new IfrtRestoreVariableOp.
    // Insert at the end of the block so that all dependencies are satisfied.
    mlir::OpBuilder builder =
        mlir::OpBuilder::atBlockTerminator(restore_op->getBlock());
    auto ifrt_restore_variable_op = mlir::TF::IfrtRestoreVariableOp::create(
        builder, restore_op->getLoc(), result_types, restore_op.getPrefix(),
        restore_op.getTensorNames(), restore_op.getShapeAndSlices(),
        var_handle_values, builder.getArrayAttr(dtypes),
        builder.getDenseBoolArrayAttr(truncate_in_cast),
        builder.getStrArrayAttr(returned_tensor_names));

    // Replace the original restored tensors with the results of the new
    // IfrtRestoreVariableOp.
    llvm::SmallPtrSet<mlir::Value, 4> replaced_tensors;
    for (auto& restored_tensor_user : restored_tensor_users) {
      if (restored_tensor_user.restored_tensor_to_return.has_value()) {
        mlir::Value tensor = *restored_tensor_user.restored_tensor_to_return;
        // Replace all uses of the original tensor with the new op's result.
        // The `replaced_tensors` set ensures that we only do this once per
        // tensor.
        if (replaced_tensors.insert(tensor).second) {
          tensor.replaceAllUsesWith(ifrt_restore_variable_op.getResult(
              returned_tensors_indices[tensor]));
        }
        // Move the user chain of the restored tensor after the new
        // IfrtRestoreVariableOp to maintain a valid IR.
        for (auto* op : restored_tensor_user.op_chain_from_restore_output) {
          op->moveAfter(ifrt_restore_variable_op);
        }
        // Ensure that the users of the user chain are also moved after.
        for (auto* op : restored_tensor_user.op_chain_from_restore_output) {
          for (mlir::Value result : op->getResults()) {
            llvm::SmallVector<mlir::Operation*, 4> users(result.user_begin(),
                                                         result.user_end());
            for (mlir::Operation* user : users) {
              if (user->getBlock() == ifrt_restore_variable_op->getBlock()) {
                user->moveAfter(op);
              }
            }
          }
        }
      }
    }

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
