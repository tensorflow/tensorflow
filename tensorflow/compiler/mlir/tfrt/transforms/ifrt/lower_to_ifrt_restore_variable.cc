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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
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

    // The attributes of VarHandleOp associated with the AssignVariableOp.
    mlir::Type var_handle_type;
    mlir::StringAttr container;
    mlir::StringAttr shared_name;
    bool truncate_in_cast =
        false;  // value of the truncate attribute in the CastOp.
                // Default to false if CastOp is not present.

    // The restored tensor that is not assigned to a variable and should be
    // returned by IfrtRestoreVariableOp.
    std::optional<mlir::Value> restored_tensor_to_return;

    // The VarHandleOp associated with the AssignVariableOp, if it exists in
    // the same block or function.
    mlir::TF::VarHandleOp original_var_handle_op;
  };

  // Groups parameters being collected for IfrtRestoreVariableOp.
  struct RestorationParams {
    llvm::SmallVectorImpl<std::string>& tensor_names;
    llvm::SmallVectorImpl<std::string>& shape_and_slices;
    llvm::SmallVectorImpl<mlir::Attribute>& dtypes;
    llvm::SmallVectorImpl<bool>& truncate_in_cast;
    std::vector<mlir::Value>& var_handle_values;
  };

  // Holds information about an intercepted Read-Assign chain.
  struct InterceptedChain {
    mlir::TF::AssignVariableOp assign_op;
    mlir::TF::VarHandleOp target_handle;
    bool truncate = false;
    std::vector<mlir::Operation*> op_chain;
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
        restored_tensor_user.truncate_in_cast =
            restored_tensor_user.truncate_in_cast || cast_op.getTruncate();
        // Move to the next operation in the user chain.
        user = *cast_op.getResult().getUsers().begin();
      } else if (auto assign_variable_op =
                     llvm::dyn_cast<mlir::TF::AssignVariableOp>(user)) {
        // The chain must end with an `AssignVariableOp`.
        if (auto var_handle_op = llvm::dyn_cast<mlir::TF::VarHandleOp>(
                assign_variable_op.getResource().getDefiningOp())) {
          // The AssignVariableOp must be associated with a VarHandleOp.
          restored_tensor_user.var_handle_type = var_handle_op.getType();
          restored_tensor_user.container = var_handle_op.getContainerAttr();
          restored_tensor_user.shared_name = var_handle_op.getSharedNameAttr();
          restored_tensor_user.original_var_handle_op = var_handle_op;
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

        mlir::Type resource_type = mlir::RankedTensorType::get(
            /*shape=*/{}, /*element_type=*/mlir::tf_type::ResourceType::get(
                {mlir::cast<mlir::TensorType>(source_tensor.getType())},
                builder.getContext()));
        restored_tensor_user.var_handle_type = resource_type;
        // In TensorFlow, variables are identified by a (container,
        // shared_name) pair. An empty string for `container` is valid and
        // indicates that the default container should be used. For restored
        // tensors that are returned as outputs instead of being assigned to
        // variables, we create a logical VarHandleOp with an empty container
        // and use `tensor_name` as `shared_name`.
        restored_tensor_user.container = builder.getStringAttr("");
        restored_tensor_user.shared_name = builder.getStringAttr(tensor_name);

        restored_tensor_user.restored_tensor_to_return = source_tensor;
        break;
      }
    }

    // If the user chain is valid, add the collected information.
    restored_tensor_users.push_back(restored_tensor_user);
    return mlir::success();
  }

  // Traces and validates an operation chain starting from a `ReadVariableOp`
  // user. Returns the chain if it's a valid pattern (optional `Cast`s,
  // `Slice`s, or `Identity`s followed by an `AssignVariableOp`).
  std::optional<InterceptedChain> TraceReadChain(mlir::Operation* read_user) {
    InterceptedChain chain;
    mlir::Operation* current = read_user;
    while (true) {
      if (auto cast_op = llvm::dyn_cast<mlir::TF::CastOp>(current)) {
        chain.truncate |= cast_op.getTruncate();
        chain.op_chain.push_back(cast_op);
        if (!cast_op.getResult().hasOneUse()) return std::nullopt;
        current = *cast_op.getResult().getUsers().begin();
      } else if (auto slice_op = llvm::dyn_cast<mlir::TF::SliceOp>(current)) {
        chain.op_chain.push_back(slice_op);
        if (!slice_op.getOutput().hasOneUse()) return std::nullopt;
        current = *slice_op.getOutput().getUsers().begin();
      } else if (auto identity_op =
                     llvm::dyn_cast<mlir::TF::IdentityOp>(current)) {
        chain.op_chain.push_back(identity_op);
        if (!identity_op.getOutput().hasOneUse()) return std::nullopt;
        current = *identity_op.getOutput().getUsers().begin();
      } else {
        break;
      }
    }

    if (auto assign_op = llvm::dyn_cast<mlir::TF::AssignVariableOp>(current)) {
      if (auto target_handle = llvm::dyn_cast<mlir::TF::VarHandleOp>(
              assign_op.getResource().getDefiningOp())) {
        chain.assign_op = assign_op;
        chain.target_handle = target_handle;
        chain.op_chain.push_back(assign_op);
        return chain;
      }
    }
    return std::nullopt;
  }

  // Processes a validly intercepted chain: updates restoration parameters and
  // manages `VarHandleOp`s.
  void ProcessInterceptedChain(InterceptedChain& chain,
                               mlir::StringRef tensor_name,
                               mlir::StringRef slice, mlir::Attribute dtype,
                               RestorationParams& params,
                               mlir::OpBuilder& handle_builder,
                               mlir::Operation* restore_op) {
    params.tensor_names.push_back(tensor_name.str());
    params.shape_and_slices.push_back(slice.str());
    params.dtypes.push_back(dtype);
    params.truncate_in_cast.push_back(chain.truncate);

    mlir::Value handle_result;
    if (chain.target_handle->getBlock() == restore_op->getBlock()) {
      handle_result = chain.target_handle.getResult();
      // Since we are reusing an op that originally appeared later, we need to
      // move it before the new `IfrtRestoreVariableOp` to avoid dominance
      // violations.
      chain.target_handle->moveBefore(restore_op);
    } else {
      auto derived_local_handle = handle_builder.create<mlir::TF::VarHandleOp>(
          restore_op->getLoc(), chain.target_handle.getType(),
          chain.target_handle.getContainerAttr(),
          chain.target_handle.getSharedNameAttr());
      handle_result = derived_local_handle.getResult();
    }
    params.var_handle_values.push_back(handle_result);
  }

  // Identifies and handles cases where a restored variable is read and
  // subsequently assigned to another variable. For example, if VarA is restored
  // from a checkpoint, and VarB is initialized with VarA's value (e.g., via
  // ReadVariable(VarA) -> AssignVariable(VarB)), then VarB should also be
  // treated as being restored from the checkpoint. This function detects such
  // patterns, adds VarB to list of tensors to be restored by
  // IfrtRestoreVariableOp, and marks the intermediate ReadVariableOp and
  // AssignVariableOp chain for deletion.
  // Returns intercepted chains if all users of a `ReadVariableOp` lead to
  // assignments.
  std::optional<std::vector<InterceptedChain>> GetInterceptedChains(
      mlir::TF::ReadVariableOp read_op) {
    std::vector<InterceptedChain> intercepted_chains;
    // Follow the user chain for each user of the `ReadVariableOp`.
    for (mlir::Operation* read_user : read_op.getResult().getUsers()) {
      if (std::optional<InterceptedChain> intercepted_chain =
              TraceReadChain(read_user)) {
        intercepted_chains.push_back(std::move(*intercepted_chain));
      } else {
        return std::nullopt;
      }
    }
    return intercepted_chains;
  }

  // Identifies and handles cases where a restored variable is read and
  // subsequently assigned to another variable. For example, if VarA is restored
  // from a checkpoint, and VarB is initialized with VarA's value (e.g., via
  // ReadVariable(VarA) -> AssignVariable(VarB)), then VarB should also be
  // treated as being restored from the checkpoint. This function detects such
  // patterns, adds VarB to list of tensors to be restored by
  // IfrtRestoreVariableOp, and marks the intermediate ReadVariableOp and
  // AssignVariableOp chain for deletion.
  void InterceptReadAssignVariable(
      const llvm::StringMap<std::vector<mlir::TF::ReadVariableOp>>&
          shared_name_to_reads,
      mlir::StringRef tensor_name, mlir::StringRef slice, mlir::Attribute dtype,
      RestoreOpUser& restored_tensor_user, mlir::OpBuilder& handle_builder,
      mlir::Operation* restore_op, RestorationParams& params,
      llvm::SmallSetVector<mlir::Operation*, 16>& ops_to_erase) {
    auto it =
        shared_name_to_reads.find(restored_tensor_user.shared_name.getValue());
    // If no `ReadVariableOp` reads the variable associated with
    // `restored_tensor_user`, then there is no read-assign chain to intercept.
    if (it == shared_name_to_reads.end()) {
      return;
    }

    auto add_to_erase = [&](mlir::Operation* op) { ops_to_erase.insert(op); };

    // Iterate through all `ReadVariableOp`s that read the variable.
    for (mlir::TF::ReadVariableOp read_op : it->second) {
      // If all users of this specific `ReadVariableOp` instance were
      // successfully mapped to `AssignVariableOp` chains, we can erase the
      // `ReadVariableOp` and all intermediate ops in these chains.
      if (auto intercepted_chains = GetInterceptedChains(read_op)) {
        if (intercepted_chains->empty()) continue;

        add_to_erase(read_op);
        for (auto& chain : *intercepted_chains) {
          ProcessInterceptedChain(chain, tensor_name, slice, dtype, params,
                                  handle_builder, restore_op);
          for (mlir::Operation* op : chain.op_chain) {
            add_to_erase(op);
          }
        }
      }
    }
  }

  // Creates a TF::ConstOp with a 1D tensor of strings from the given values.
  mlir::Value CreateStringTensorConst(
      mlir::OpBuilder& builder, mlir::Location loc,
      const llvm::SmallVector<std::string, 4>& values) {
    llvm::SmallVector<llvm::StringRef> ref_values;
    for (const auto& value : values) {
      ref_values.push_back(value);
    }
    auto type = mlir::RankedTensorType::get(
        {static_cast<int64_t>(values.size())},
        mlir::tf_type::StringType::get(builder.getContext()));

    return builder.create<mlir::TF::ConstOp>(
        loc, mlir::DenseStringElementsAttr::get(type, ref_values));
  }

  mlir::LogicalResult RewriteRestore(mlir::TF::RestoreV2Op restore_op) {
    // Find and validate all users of the RestoreV2Op's output tensors.

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

    // The RestoreV2 op contains shape and slice specifications for each tensor
    // to be restored. These are expected to be provided as a constant string
    // tensor.
    mlir::DenseStringElementsAttr shape_and_slices_attr;
    if (!matchPattern(restore_op.getShapeAndSlices(),
                      m_Constant(&shape_and_slices_attr))) {
      return restore_op.emitOpError(
          "expects shape_and_slices to be a constant");
    }
    // Extract shape and slice strings into a vector.
    // This vector is used later to associate each restored tensor with its
    // corresponding shape and slice specification when creating the
    // IfrtRestoreVariableOp.
    auto slice_values = shape_and_slices_attr.getValues<mlir::StringRef>();
    llvm::SmallVector<mlir::StringRef> shape_and_slices_vec(
        slice_values.begin(), slice_values.end());

    std::vector<RestoreOpUser> restored_tensor_users;
    for (int i = 0; i < restore_op.getTensors().size(); ++i) {
      mlir::Value out_tensor = restore_op.getTensors()[i];
      mlir::StringRef tensor_name = tensor_names_vec[i];
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

    // The transformation needs to identify cases where a restored variable is
    // read and then assigned to another variable (e.g., RestoreV2 ->
    // AssignVariable(var1), ReadVariable(var1) -> AssignVariable(var2)).
    // In such cases, `var2` should also be treated as a restored variable
    // initialized from the checkpoint. To achieve this, we first map all
    // variable shared names to their VarHandleOps and all ReadVariableOps
    // that read them. This allows us to trace reads of restored variables and
    // check if they are part of an assignment to another variable.
    llvm::StringMap<mlir::TF::VarHandleOp> shared_name_to_handle;
    llvm::StringMap<std::vector<mlir::TF::ReadVariableOp>> shared_name_to_reads;
    restore_op->getParentOfType<mlir::ModuleOp>().walk([&](mlir::Operation*
                                                               op) {
      if (auto var_handle = llvm::dyn_cast<mlir::TF::VarHandleOp>(op)) {
        shared_name_to_handle[var_handle.getSharedName()] = var_handle;
      } else if (auto read_op = llvm::dyn_cast<mlir::TF::ReadVariableOp>(op)) {
        if (auto var_handle =
                read_op.getResource().getDefiningOp<mlir::TF::VarHandleOp>()) {
          shared_name_to_reads[var_handle.getSharedName()].push_back(read_op);
        }
      }
    });

    // Collect tensor dtypes for the new op.
    std::vector<mlir::Attribute> dtypes;
    for (const auto& dtype : restore_op.getDtypes()) {
      dtypes.push_back(mlir::TypeAttr::get(dtype));
    }

    // The following vectors collect information for each tensor that will be
    // restored by the new IfrtRestoreVariableOp. This includes tensors
    // directly restored from RestoreV2Op, as well as variables initialized
    // by reading other restored variables (e.g., ReadVariable(restored_var) ->
    // AssignVariable(new_var)). This information is used to construct the
    // arguments for the IfrtRestoreVariableOp that replaces RestoreV2Op and
    // its user chains.
    llvm::SmallVector<std::string, 4> new_tensor_names;
    llvm::SmallVector<std::string, 4> new_shape_and_slices;
    llvm::SmallVector<mlir::Attribute, 4> new_dtypes;
    llvm::SmallVector<bool, 4> new_truncate_in_cast;

    std::vector<mlir::Value> var_handle_values;

    RestorationParams params{new_tensor_names, new_shape_and_slices, new_dtypes,
                             new_truncate_in_cast, var_handle_values};

    llvm::SmallVector<mlir::Type> result_types;
    llvm::DenseMap<mlir::Value, int> returned_tensors_indices;
    llvm::SmallVector<mlir::StringRef, 4> returned_tensor_names;

    // The following data structures are used to manage operations that need to
    // be erased after the IfrtRestoreVariableOp is created. This includes
    // chains of operations from RestoreV2Op to AssignVariableOp, as well as
    // ReadVariableOp -> AssignVariableOp chains that are optimized away by
    // InterceptReadAssignVariable.
    llvm::SmallSetVector<mlir::Operation*, 16> ops_to_erase;
    // Helper function to add an operation to ops_to_erase, ensuring no
    // duplicates are added.
    auto add_to_erase = [&](mlir::Operation* op) { ops_to_erase.insert(op); };
    // OpBuilder for creating new VarHandleOps before the RestoreV2Op.
    mlir::OpBuilder handle_builder(restore_op);

    for (int i = 0; i < restored_tensor_users.size(); ++i) {
      RestoreOpUser& restored_tensor_user = restored_tensor_users[i];
      mlir::StringRef tensor_name = tensor_names_vec[i];
      mlir::StringRef slice = shape_and_slices_vec[i];
      mlir::Attribute dtype = dtypes[i];

      params.tensor_names.push_back(tensor_name.str());
      params.shape_and_slices.push_back(slice.str());
      params.dtypes.push_back(dtype);
      params.truncate_in_cast.push_back(restored_tensor_user.truncate_in_cast);

      // Reuse the original VarHandleOp if it exists and is in the same block as
      // the RestoreV2Op. This avoids redundant VarHandleOps in the IR and
      // makes the output cleaner.
      mlir::Value handle_result;
      if (restored_tensor_user.original_var_handle_op &&
          restored_tensor_user.original_var_handle_op->getBlock() ==
              restore_op->getBlock()) {
        handle_result = restored_tensor_user.original_var_handle_op.getResult();
        // Since we are reusing an op that originally appeared later, we need to
        // move it before the new IfrtRestoreVariableOp to avoid dominance
        // violations.
        restored_tensor_user.original_var_handle_op->moveBefore(restore_op);
      } else {
        mlir::TF::VarHandleOp local_handle =
            handle_builder.create<mlir::TF::VarHandleOp>(
                restore_op->getLoc(), restored_tensor_user.var_handle_type,
                restored_tensor_user.container,
                restored_tensor_user.shared_name);
        handle_result = local_handle.getResult();
      }
      params.var_handle_values.push_back(handle_result);

      if (!restored_tensor_user.restored_tensor_to_return.has_value()) {
        // This tensor is assigned to a variable.
        // Delete the path from the RestoreV2Op to the AssignVariableOp.
        for (mlir::Operation* op :
             restored_tensor_user.op_chain_from_restore_output) {
          add_to_erase(op);
        }

        // Intercept usages where the restored variable is assigned to a var
        // handle 1, and then read from varhandle1 and assigned to var handle 2.
        InterceptReadAssignVariable(shared_name_to_reads, tensor_name, slice,
                                    dtype, restored_tensor_user, handle_builder,
                                    restore_op, params, ops_to_erase);
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

    for (mlir::Operation* op : llvm::reverse(ops_to_erase)) {
      if (!op->use_empty()) {
        return op->emitOpError() << "is expected to be erased but has uses";
      }
      op->erase();
    }

    mlir::OpBuilder builder(restore_op);

    // If tensor_names or shape_and_slices have been modified (e.g., due to
    // intercepted read-assign variables), create new constant tensors.
    // Otherwise, reuse the original tensors from RestoreV2Op to avoid
    // redundancy.
    mlir::Value new_tensor_names_op;
    if (new_tensor_names.size() == tensor_names_vec.size() &&
        std::equal(new_tensor_names.begin(), new_tensor_names.end(),
                   tensor_names_vec.begin())) {
      new_tensor_names_op = restore_op.getTensorNames();
    } else {
      new_tensor_names_op = CreateStringTensorConst(
          builder, restore_op->getLoc(), new_tensor_names);
    }
    mlir::Value new_shape_and_slices_op;
    if (new_shape_and_slices.size() == shape_and_slices_vec.size() &&
        std::equal(new_shape_and_slices.begin(), new_shape_and_slices.end(),
                   shape_and_slices_vec.begin())) {
      new_shape_and_slices_op = restore_op.getShapeAndSlices();
    } else {
      new_shape_and_slices_op = CreateStringTensorConst(
          builder, restore_op->getLoc(), new_shape_and_slices);
    }

    auto ifrt_restore_variable_op = mlir::TF::IfrtRestoreVariableOp::create(
        builder, restore_op->getLoc(), result_types, restore_op.getPrefix(),
        new_tensor_names_op, new_shape_and_slices_op, var_handle_values,
        builder.getArrayAttr(new_dtypes),
        builder.getDenseBoolArrayAttr(new_truncate_in_cast),
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
