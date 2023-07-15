/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

struct ResultMapping {
  bool has_duplicate = false;
  llvm::SmallVector<int> old_to_new;
  llvm::SmallVector<int> new_to_old;
};

ResultMapping FindDuplicateResultIndices(mlir::func::FuncOp func) {
  llvm::SmallDenseMap<mlir::Value, int> results;
  auto &block = func.getBody().front();

  auto return_op = llvm::cast<mlir::func::ReturnOp>(block.getTerminator());

  ResultMapping mapping;
  mapping.old_to_new.reserve(return_op.getNumOperands());

  int j = 0;
  for (int i = 0; i < return_op.getNumOperands(); ++i) {
    auto value = return_op.getOperand(i);
    auto iter = results.find(value);
    if (iter != results.end()) {
      mapping.has_duplicate = true;
      mapping.old_to_new.push_back(iter->second);
    } else {
      results[value] = j;
      mapping.old_to_new.push_back(j++);
    }
  }

  mapping.new_to_old.resize(j);
  for (int i = 0; i < mapping.old_to_new.size(); ++i) {
    mapping.new_to_old[mapping.old_to_new[i]] = i;
  }

  return mapping;
}

mlir::func::FuncOp CreateBranchFunctionWithDeduplicatedResults(
    mlir::OpBuilder &builder, mlir::Location loc, absl::string_view name,
    mlir::func::FuncOp func, const ResultMapping &mapping) {
  auto arg_types = func.getFunctionType().getInputs();
  auto result_types = func.getFunctionType().getResults();

  llvm::SmallVector<mlir::Type> new_result_types;
  new_result_types.reserve(mapping.new_to_old.size());

  for (int i : mapping.new_to_old) {
    new_result_types.push_back(result_types[i]);
  }

  auto new_func_type = mlir::FunctionType::get(builder.getContext(), arg_types,
                                               new_result_types);

  auto new_func = builder.create<mlir::func::FuncOp>(loc, name, new_func_type);
  new_func.setVisibility(mlir::func::FuncOp::Visibility::Private);

  mlir::OpBuilder::InsertionGuard guard(builder);

  // In the body of newly created function, we insert
  // tf.PartitionedCall ops to call the original func.
  auto *block = new_func.addEntryBlock();
  builder.setInsertionPointToStart(block);
  auto empty_string_attr = builder.getStringAttr("");

  llvm::SmallVector<mlir::Value, 4> results;
  results.reserve(new_func_type.getNumResults());

  // Create the call op to the original func. The arguments are simply
  // the arguments from the wrapper function.
  auto call_op = builder.create<mlir::TF::PartitionedCallOp>(
      loc, result_types, block->getArguments(),
      mlir::FlatSymbolRefAttr::get(func.getSymNameAttr()), empty_string_attr,
      empty_string_attr, empty_string_attr);

  for (int i : mapping.new_to_old) {
    results.push_back(call_op.getResult(i));
  }

  builder.create<mlir::func::ReturnOp>(loc, results);

  return new_func;
}

void DeduplicateIfOps(mlir::ModuleOp module) {
  mlir::SymbolTable symbol_table(module);

  llvm::DenseMap<mlir::func::FuncOp, ResultMapping> visited;
  llvm::DenseMap<mlir::func::FuncOp, mlir::func::FuncOp> new_funcs;
  llvm::SmallVector<mlir::TF::IfOp> candidates;

  module.walk([&](mlir::TF::IfOp op) { candidates.push_back(op); });

  llvm::SmallVector<mlir::TF::IfOp> to_erase;

  mlir::OpBuilder builder(module.getRegion());
  for (auto op : candidates) {
    auto find_mapping = [&](mlir::func::FuncOp func) {
      if (auto iter = visited.find(func); iter != visited.end()) {
        return iter->second;
      }

      auto mapping = FindDuplicateResultIndices(func);
      visited[func] = mapping;
      return mapping;
    };

    auto then_branch =
        symbol_table.lookup<mlir::func::FuncOp>(op.getThenBranch());
    auto else_branch =
        symbol_table.lookup<mlir::func::FuncOp>(op.getElseBranch());

    auto then_mapping = find_mapping(then_branch);
    auto else_mapping = find_mapping(else_branch);

    if (then_mapping.has_duplicate &&
        then_mapping.old_to_new == else_mapping.old_to_new) {
      auto get_or_create = [&](mlir::func::FuncOp func,
                               const ResultMapping &mapping) {
        auto iter = new_funcs.find(func);
        if (iter != new_funcs.end()) {
          return iter->second;
        }

        auto new_func = CreateBranchFunctionWithDeduplicatedResults(
            builder, op.getLoc(),
            absl::StrCat(func.getSymName().str(), "/tfrt_dedup_results"), func,
            mapping);
        new_funcs[func] = new_func;
        return new_func;
      };
      auto new_then_func = get_or_create(then_branch, then_mapping);
      auto new_else_func = get_or_create(else_branch, else_mapping);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(op);

      llvm::SmallVector<mlir::Type> new_result_types;
      for (int i : then_mapping.new_to_old) {
        new_result_types.push_back(op->getResult(i).getType());
      }

      auto new_if_op = builder.create<mlir::TF::IfOp>(
          op.getLoc(), new_result_types, op.getCond(), op.getInput(),
          new_then_func.getSymName(), new_else_func.getSymName(),
          op.getIsStateless());

      DCHECK_EQ(then_mapping.old_to_new.size(), op.getNumResults());
      for (int i = 0; i < then_mapping.old_to_new.size(); ++i) {
        op.getResult(i).replaceAllUsesWith(
            new_if_op.getResult(then_mapping.old_to_new[i]));
      }
      to_erase.push_back(op);
    }
  }

  for (auto op : to_erase) {
    op->erase();
  }
}

class DeduplicateIfResultPass
    : public mlir::PassWrapper<DeduplicateIfResultPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  llvm::StringRef getArgument() const final {
    return "tfrt-deduplicate-if-result";
  }
  llvm::StringRef getDescription() const final {
    return "Deduplicate the results of tf.If ops";
  }

  void runOnOperation() override { DeduplicateIfOps(getOperation()); }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeduplicateIfResultPass)
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDeduplicateIfResultPass() {
  return std::make_unique<DeduplicateIfResultPass>();
}

static mlir::PassRegistration<DeduplicateIfResultPass> register_pass(
    CreateDeduplicateIfResultPass);

}  // namespace tfrt_compiler
}  // namespace tensorflow
