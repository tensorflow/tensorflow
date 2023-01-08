/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

// This pass removes tf.If ops' operands that are produced by tf.Const ops.
// These constants can be moved into branches' function body for further
// optimziation.
class RemoveTfIfConstArgs
    : public mlir::PassWrapper<RemoveTfIfConstArgs,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveTfIfConstArgs)

 private:
  llvm::StringRef getArgument() const final {
    return "tfrt-remove-tf-if-const-args";
  }
  llvm::StringRef getDescription() const final {
    return "Remove const args from tf.If ops";
  }

  void runOnOperation() override {
    auto module = getOperation();
    for (auto func_op :
         llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
      ProcessFunction(func_op);
    }
  }

  void ProcessFunction(mlir::func::FuncOp op) {
    // Set the insertion point to the current function, as we will insert new
    // functions here.
    mlir::OpBuilder builder(op);
    for (mlir::Operation &op : op.front()) {
      auto if_op = llvm::dyn_cast<mlir::TF::IfOp>(&op);
      if (!if_op) continue;

      // Record the operands that are produced by tf.Const ops.
      llvm::SmallVector<mlir::TF::ConstOp, 2> const_args;
      // Record these operands's corresponding operand indices.
      llvm::SmallVector<unsigned, 2> const_arg_indices;
      // Record the remaining operands that won't be removed.
      llvm::SmallVector<mlir::Value, 2> remaining_args;
      for (auto iter : llvm::enumerate(if_op.getInput())) {
        mlir::Value operand = iter.value();
        if (auto const_op = operand.getDefiningOp<mlir::TF::ConstOp>()) {
          const_args.push_back(const_op);
          const_arg_indices.push_back(iter.index());
        } else {
          remaining_args.push_back(operand);
        }
      }

      if (const_args.empty()) continue;

      RemoveConstArgsFromTfIfOp(builder, if_op, const_args, const_arg_indices,
                                remaining_args);
    }
  }

  void RemoveConstArgsFromTfIfOp(mlir::OpBuilder &builder, mlir::TF::IfOp if_op,
                                 llvm::ArrayRef<mlir::TF::ConstOp> const_args,
                                 llvm::ArrayRef<unsigned> const_arg_indices,
                                 llvm::ArrayRef<mlir::Value> remaining_args) {
    auto branch_suffix = absl::StrCat("_removed_const_args_", id_++);

    // Create wrapper functions with the new arguments (as const args are
    // removed) for both then function and else function.
    auto new_then_function_name =
        CreateBranchFunction(builder, if_op.then_function(), branch_suffix,
                             const_args, const_arg_indices);
    auto new_else_function_name =
        CreateBranchFunction(builder, if_op.else_function(), branch_suffix,
                             const_args, const_arg_indices);

    // Change the if_op's argumetns to the new arguments, branches to new
    // branches. Note that the outputs are not changed.
    if_op.getInputMutable().assign(remaining_args);
    if_op.setThenBranchAttr(
        mlir::SymbolRefAttr::get(builder.getContext(), new_then_function_name));
    if_op.setElseBranchAttr(
        mlir::SymbolRefAttr::get(builder.getContext(), new_else_function_name));
  }

  llvm::StringRef CreateBranchFunction(
      mlir::OpBuilder &builder, mlir::func::FuncOp branch,
      absl::string_view branch_suffix,
      llvm::ArrayRef<mlir::TF::ConstOp> const_args,
      llvm::ArrayRef<unsigned> const_arg_indices) {
    // Get the new function type as const args are removed.
    llvm::BitVector const_arg_indices_bv(branch.getNumArguments());
    for (auto i : const_arg_indices) const_arg_indices_bv.set(i);
    auto new_branch_type = branch.getFunctionType().getWithoutArgsAndResults(
        const_arg_indices_bv, {});
    std::string new_branch_name =
        absl::StrCat(branch.getSymName().str(), branch_suffix);
    // Create the wrapper function with the new arguments that calls the
    // original branch.
    auto new_branch = builder.create<mlir::func::FuncOp>(
        branch.getLoc(), new_branch_name, new_branch_type);
    new_branch.setVisibility(mlir::func::FuncOp::Visibility::Private);

    // In its function body, we will add the corresponding const ops and call
    // the original branch.

    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *block = new_branch.addEntryBlock();
    builder.setInsertionPointToStart(block);

    // Prepare the function arguments of the original branch.
    llvm::SmallVector<mlir::Value, 4> call_args(branch.getNumArguments());

    // For those removed const args, we copy the tf.Const op, and use that as
    // the corresponding argument when calling the original branch.
    for (const auto &iter : llvm::zip(const_args, const_arg_indices)) {
      auto const_op =
          llvm::cast<mlir::TF::ConstOp>(builder.clone(*std::get<0>(iter)));
      unsigned index = std::get<1>(iter);
      call_args[index] = const_op;
    }

    // For the rest, they are now coming from the wrapper function's arguments
    // in the original order.
    for (int i = 0, j = 0; i < call_args.size(); ++i) {
      if (!call_args[i]) {
        assert(j < block->getNumArguments());
        call_args[i] = block->getArgument(j++);
      }
    }

    // Now create the call op to the original branch.
    auto call_op = builder.create<mlir::TF::StatefulPartitionedCallOp>(
        new_branch.getLoc(), new_branch_type.getResults(), call_args,
        branch.getSymName(), "", "", "");
    // Note that the outputs are not changed.
    builder.create<mlir::func::ReturnOp>(new_branch.getLoc(),
                                         call_op.getOutput());

    return new_branch.getSymName();
  }

  int id_ = 0;
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateRemoveTfIfConstArgsPass() {
  return std::make_unique<RemoveTfIfConstArgs>();
}

static mlir::PassRegistration<RemoveTfIfConstArgs> register_pass(
    CreateRemoveTfIfConstArgsPass);

}  // namespace tfrt_compiler
}  // namespace tensorflow
