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
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/async_while.h"

#include <linux/limits.h>

#include <algorithm>
#include <deque>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h"

namespace tensorflow {
namespace mlrt_compiler {
namespace {

// Move await right before its tensor is used.
void MoveTFAwaitOpToFirstUse(mlir::Block &block) {
  llvm::SmallVector<tf_mlrt::TFAwaitOp> await_ops;
  for (auto &op : block) {
    if (auto await_op = llvm::dyn_cast<tf_mlrt::TFAwaitOp>(&op)) {
      await_ops.push_back(await_op);
    }
  }

  for (auto op : await_ops) {
    auto result = op.getResult();
    if (result.use_empty()) continue;

    mlir::Operation *first_user = *result.user_begin();
    for (auto *user : result.getUsers()) {
      if (user->isBeforeInBlock(first_user)) {
        first_user = user;
      }
    }

    op->moveBefore(first_user);
  }
}

// Move promise right after the tensor becomes available.
void MoveTFPromiseOpRightAfterAvailable(mlir::Block &block) {
  llvm::SmallVector<tf_mlrt::TFPromiseOp> promise_ops;
  for (auto &op : block) {
    if (auto promise_op = llvm::dyn_cast<tf_mlrt::TFPromiseOp>(&op)) {
      promise_ops.push_back(promise_op);
    }
  }

  for (auto promise_op : promise_ops) {
    auto tensor = promise_op.getOperand(1);
    // Promise is on variant tensor, which must have a defining op inside the
    // block.
    DCHECK(tensor.getDefiningOp() != nullptr);
    promise_op->moveAfter(tensor.getDefiningOp());
  }
}

// Converts applicable tf.While to tf_mlrt.AsyncWhile.
//
// In pseudo code, the following
//
//  %res0, %res1 = tf.While(%arg0, %arg1)(body = @while_body, cond =
//  @while_predicate)
//
// is converted to
//
//  %0 = tf.PartitionedCall(%res0) (f =
//  @while_predicate/TfMlrtAsyncWhilePredicate) %1, %2, %3 =
//  tf_mlrt.async_while(%0, %arg0, %arg1) (body =
//  @while_body/TfMlrtAsyncWhileBody, immutable_size = 1) %res0 =
//  tf_mlrt.await(%2) %res1 = tf_mlrt.await(%3)
//
class AsyncWhilePass
    : public mlir::PassWrapper<AsyncWhilePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  AsyncWhilePass() = default;
  AsyncWhilePass &operator=(const AsyncWhilePass &) = delete;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AsyncWhilePass)

 private:
  // The input to while op will be re-order and supplied as input to async_while
  // op.
  struct ArgumentRemap {
    std::vector<int> new_to_old_remap;
    std::vector<int> old_to_new_remap;
  };

  // The input to while op will grouped into immutables(invariants) between
  // iterations and mutables(variants).
  struct ArgumentGroups {
    std::vector<int> immutables;
    std::vector<int> mutables;
  };

  // First argument of the new async_while is a predicate.
  static constexpr int kNonPredicateStartIndex = 1;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<tensorflow::tf_mlrt::TensorflowMlrtDialect>();
    registry.insert<mlrt::compiler::MlrtDialect>();
  }

  llvm::StringRef getArgument() const final { return "tf-mlrt-async-while"; }

  llvm::StringRef getDescription() const final {
    return "Convert tf.while to tf_mlrt.async_while when possible for parallel "
           "execution.";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::SymbolTable symbol_table(module);
    mlir::TF::SideEffectAnalysis side_effect_analysis(module);

    // Keep a record of predicate and body functions that we already used to
    // create new predicate and body functions.
    // key is the original function name and value is the newly created
    // function.
    // We keep our own record instead of relying on SymbolTable to know whether
    // a function is already converted is because SymbolTable might create new
    // symbol name for inserted function to avoid conflict.
    llvm::SmallDenseMap<llvm::StringRef, mlir::func::FuncOp>
        processed_functions;
    // Use make_early_inc_range because the processing might insert new node
    // into the list
    for (auto func_op :
         llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
      MayConvertWhileToAsyncWhile(
          func_op, symbol_table, processed_functions,
          side_effect_analysis.GetAnalysisForFunc(func_op));
    }
  }

  void MayConvertWhileToAsyncWhile(
      mlir::func::FuncOp op, mlir::SymbolTable &symbol_table,
      llvm::SmallDenseMap<llvm::StringRef, mlir::func::FuncOp>
          &processed_functions,
      const mlir::TF::SideEffectAnalysis::Info &side_effect_analysis) {
    mlir::OpBuilder builder(op);
    for (mlir::Operation &op : llvm::make_early_inc_range(op.front())) {
      auto while_op = llvm::dyn_cast<mlir::TF::WhileOp>(&op);
      if (!while_op) continue;

      // Fill in bodies
      mlir::func::FuncOp body_fn =
          symbol_table.lookup<mlir::func::FuncOp>(while_op.getBody());

      mlir::func::FuncOp predicate_fn =
          symbol_table.lookup<mlir::func::FuncOp>(while_op.getCond());

      if (ShouldConvertWhileToAsyncWhile(body_fn, predicate_fn)) {
        std::vector<int> predicate_determinat =
            GetPredicateDeterminat(predicate_fn);

        ArgumentGroups argument_groups = GetArgumentGroups(body_fn);
        ArgumentRemap argument_remap =
            CreateArgumentRemap(body_fn, argument_groups.immutables);

        mlir::func::FuncOp new_predicate_fn =
            CreatePredicate(builder, symbol_table, processed_functions,
                            predicate_fn, predicate_determinat);

        mlir::func::FuncOp new_body =
            CreateBody(builder, symbol_table, processed_functions, body_fn,
                       new_predicate_fn, predicate_determinat, argument_groups,
                       argument_remap, side_effect_analysis);

        mlir::OpBuilder::InsertionGuard insertion_guard(builder);
        builder.setInsertionPoint(while_op);

        // Call the predicate function first. async_while requires the first
        // predicate to be given.
        llvm::SmallVector<mlir::Value> predicate_input;
        for (int i : predicate_determinat) {
          predicate_input.push_back(while_op->getOperand(i));
        }

        auto empty_string_attr = builder.getStringAttr("");
        auto init_predicate = builder.create<mlir::TF::PartitionedCallOp>(
            while_op->getLoc(), predicate_fn.getResultTypes()[0],
            predicate_input, /*args_attrs=*/nullptr, /*res_attrs=*/nullptr,
            mlir::FlatSymbolRefAttr::get(new_predicate_fn.getSymNameAttr()),
            empty_string_attr, empty_string_attr, empty_string_attr);

        // These arguments exclude predicate.
        llvm::SmallVector<mlir::Value> async_while_input;
        async_while_input.reserve(while_op->getNumOperands());
        for (auto old_arg_idx_iter =
                 argument_remap.new_to_old_remap.begin() + 1;
             old_arg_idx_iter != argument_remap.new_to_old_remap.end();
             ++old_arg_idx_iter) {
          async_while_input.push_back(while_op.getOperand(*old_arg_idx_iter));
        }

        llvm::SmallVector<mlir::Type> async_while_output_types;
        async_while_output_types.resize(1 + while_op->getNumResults());
        std::fill(async_while_output_types.begin(),
                  async_while_output_types.end(),
                  builder.getType<mlrt::compiler::FutureType>());

        auto async_while_op = builder.create<tf_mlrt::TFAsyncWhileOp>(
            while_op.getLoc(), async_while_output_types,
            init_predicate->getResult(0), async_while_input,
            new_body.getSymName(), argument_groups.immutables.size());

        for (int i = 0; i < while_op.getResults().size(); ++i) {
          if (!while_op.getResult(i).use_empty()) {
            auto async_while_result = builder.create<tf_mlrt::TFAwaitOp>(
                while_op.getLoc(), while_op->getResultTypes()[i],
                async_while_op.getResult(argument_remap.old_to_new_remap[i]));
            while_op.getResult(i).replaceAllUsesWith(
                async_while_result.getResult());
          }
        }

        while_op.erase();
      }
    }

    // TFAwaitOp may have been inserted, move them to first use.
    MoveTFAwaitOpToFirstUse(op.getBody().front());
  }

  // Creates a new predicate function that returns a boolean tensor. This new
  // predicate function is converted from the original predicate function after
  // removing the unused input arguments. We want the inputs to the new
  // predicate function to be the minimal set because the new body function
  // wants to call the predicate as early as possible to maximize the chances of
  // parallerization. Also since the new body function reorders the input
  // arguments, the new predicate function follows that new order to be
  // consistent (just like the original predicate function and body function
  // have the same order of arguments).
  mlir::func::FuncOp CreatePredicate(
      mlir::OpBuilder &builder, mlir::SymbolTable &symbol_table,
      llvm::SmallDenseMap<llvm::StringRef, mlir::func::FuncOp>
          &processed_functions,
      mlir::func::FuncOp original_predicate_fn,
      const std::vector<int> &predicate_determinat) {
    auto &processed_function =
        processed_functions[original_predicate_fn.getName()];
    if (processed_function) {
      return processed_function;
    }
    mlir::OpBuilder::InsertionGuard insertion_guard(builder);
    builder.setInsertionPointAfter(original_predicate_fn);

    // Input to the new predicate only contains arguments listed in
    // predicate_determinat.
    llvm::SmallVector<mlir::Type> remapped_input_types;
    for (int i : predicate_determinat) {
      remapped_input_types.push_back(
          original_predicate_fn.getArgumentTypes()[i]);
    }

    std::string new_predicate_fn_name = absl::StrCat(
        original_predicate_fn.getName().str(), "/TfMlrtAsyncWhilePredicate");
    mlir::func::FuncOp new_predicate_fn = builder.create<mlir::func::FuncOp>(
        original_predicate_fn->getLoc(), new_predicate_fn_name,
        mlir::FunctionType::get(original_predicate_fn.getContext(),
                                remapped_input_types,
                                original_predicate_fn.getResultTypes()));

    processed_function = new_predicate_fn;

    new_predicate_fn.setPrivate();

    builder.setInsertionPointToEnd(new_predicate_fn.addEntryBlock());

    // Inputs will be reordered according to the body functions' reorder.
    mlir::IRMapping remap;
    int j = 0;
    for (int i : predicate_determinat) {
      remap.map(original_predicate_fn.getArgument(i),
                new_predicate_fn.getArgument(j));
      j++;
    }

    for (auto &op_it : original_predicate_fn.getBody().front()) {
      builder.clone(op_it, remap);
    }

    symbol_table.insert(new_predicate_fn);
    return new_predicate_fn;
  }

  // Creates the new body function for the async while.
  // The new body function has the input signature of
  // (predicate_promise, mutable0_future, mutable0_promise, mutable1_future,
  // mutable1_promise, ..., immutable0, immutable1,...) The new body is mostly a
  // clone from the old body function, but the input arguments are reordered
  // such that the immutables(invariants) between iterations are always at the
  // bottom of the argument list. Additionaly, the new body function calls the
  // new predicate function and puts the boolean predicate for the next
  // iteration into the predicate_promise.
  mlir::func::FuncOp CreateBody(
      mlir::OpBuilder &builder, mlir::SymbolTable &symbol_table,
      llvm::SmallDenseMap<llvm::StringRef, mlir::func::FuncOp>
          &processed_functions,
      mlir::func::FuncOp original_body_fn, mlir::func::FuncOp new_predicate_fn,
      const std::vector<int> &predicate_determinat,
      const ArgumentGroups &argument_groups,
      const ArgumentRemap &argument_remap,
      const mlir::TF::SideEffectAnalysis::Info &side_effect_analysis) {
    auto &processed_function = processed_functions[original_body_fn.getName()];
    if (processed_function) {
      return processed_function;
    }

    // Safe to do topology sort here b/c there are no promise/future involved
    // here.
    SortBodyFnByFirstReadyFirstRun(original_body_fn, predicate_determinat,
                                   side_effect_analysis);

    mlir::OpBuilder::InsertionGuard insertion_guard(builder);
    builder.setInsertionPointAfter(original_body_fn);

    // Create the new body function signature.
    std::vector<mlir::Type> remapped_input_types;
    std::vector<mlir::Type> remapped_output_types;

    // predicate promise is the first argument.
    remapped_input_types.push_back(
        builder.getType<mlrt::compiler::PromiseType>());

    // future and promise pairs for the mutables follows.
    for (int i = 0; i < argument_groups.mutables.size(); ++i) {
      remapped_input_types.push_back(
          builder.getType<mlrt::compiler::FutureType>());
      remapped_input_types.push_back(
          builder.getType<mlrt::compiler::PromiseType>());
    }

    // immutables are always at the bottom of the argument list.
    for (int i : argument_groups.immutables) {
      remapped_input_types.push_back(original_body_fn.getArgumentTypes()[i]);
    }

    std::string new_body_fn_name =
        absl::StrCat(original_body_fn.getName().str(), "/TfMlrtAsyncWhileBody");
    auto body_fn = builder.create<mlir::func::FuncOp>(
        original_body_fn->getLoc(), new_body_fn_name,
        mlir::FunctionType::get(original_body_fn.getContext(),
                                remapped_input_types, remapped_output_types));
    processed_function = body_fn;

    body_fn.setPrivate();

    // Inserts await for futures so that all mutable tensors are ready.
    // We will move these awaits to first use location later on.
    builder.setInsertionPointToEnd(body_fn.addEntryBlock());
    // First future starts at index 1 after predicate promise.
    int future_index = 1;
    mlir::IRMapping mapping;
    int body_arg_idx = kNonPredicateStartIndex;
    for (int i : argument_groups.mutables) {
      auto future_value = builder.create<tf_mlrt::TFAwaitOp>(
          body_fn->getLoc(), original_body_fn.getArgumentTypes()[i],
          body_fn.getArgument(future_index));
      mapping.map(original_body_fn.getArgument(i), future_value);
      // Move by 2 b/c each variant correspond to one future and one promise
      body_arg_idx += 2;
      future_index += 2;
    }

    for (int i : argument_groups.immutables) {
      mapping.map(original_body_fn.getArgument(i),
                  body_fn.getArgument(body_arg_idx));
      body_arg_idx++;
    }

    // All future values are ready; so we can clone from the original body
    // function.
    for (auto &op : original_body_fn.getBody().front()) {
      builder.clone(op, mapping);
    }

    auto return_op = body_fn.getBody().front().getTerminator();

    // Find the earliest location that the new predicate function can execute.
    auto *body_block = &body_fn.getBody().front();
    auto *earliest_op = &body_block->front();
    for (int i = 0; i < return_op->getOperands().size(); ++i) {
      if (std::find(predicate_determinat.begin(), predicate_determinat.end(),
                    i) != predicate_determinat.end()) {
        auto latest_op = return_op->getOperands()[i].getDefiningOp();
        if (latest_op != nullptr && earliest_op->isBeforeInBlock(latest_op)) {
          earliest_op = latest_op;
        }
      }
    }

    llvm::SmallVector<mlir::Value> predicate_inputs;
    for (int i : predicate_determinat) {
      predicate_inputs.push_back(return_op->getOperands()[i]);
    }
    // Call new predicate when all dependencies are ready.
    builder.setInsertionPointAfter(earliest_op);
    llvm::SmallVector<mlir::Type> predicate_returned_types;
    auto empty_string_attr = builder.getStringAttr("");
    auto predicate_op = builder.create<mlir::TF::PartitionedCallOp>(
        earliest_op->getLoc(), new_predicate_fn.getResultTypes(),
        predicate_inputs, /*args_attrs=*/nullptr, /*res_attrs=*/nullptr,
        mlir::FlatSymbolRefAttr::get(new_predicate_fn.getSymNameAttr()),
        empty_string_attr, empty_string_attr, empty_string_attr);

    // Insert promise right before return
    builder.setInsertionPoint(return_op);
    // builder.create<mlrt::compiler::PromiseOp>(return_op->getLoc());
    auto predicate_tensor = predicate_op->getResult(0);

    const int kPredicatePromiseIndex = 0;
    builder.create<tf_mlrt::TFPromiseOp>(
        return_op->getLoc(), body_fn.getArgument(kPredicatePromiseIndex),
        predicate_tensor);
    // First variant promise start at 2 as [predicate_promise, arg0_future,
    // arg0_promise, ...]
    int promise_index = 2;
    for (int i : argument_groups.mutables) {
      builder.create<tf_mlrt::TFPromiseOp>(return_op->getLoc(),
                                           body_fn.getArgument(promise_index),
                                           return_op->getOperand(i));
      promise_index += 2;
    }
    builder.create<mlir::func::ReturnOp>(return_op->getLoc());
    return_op->erase();

    // Reorder await and promise to reduce blocking waits.
    MoveTFAwaitOpToFirstUse(body_fn.getBody().front());
    MoveTFPromiseOpRightAfterAvailable(body_fn.getBody().front());

    symbol_table.insert(body_fn);
    return body_fn;
  }

  // Decides whether we should convert a while to async_while.
  // This determination is largely by heuristic.
  bool ShouldConvertWhileToAsyncWhile(mlir::func::FuncOp body_fn,
                                      mlir::func::FuncOp predicate_fn) {
    // If the predicate are NOT determined by simple ops such as (less)
    // comparison and logic and, do not apply conversion.
    for (auto &op : predicate_fn.getBody().front()) {
      if (!llvm::isa<mlir::TF::ConstOp, mlir::TF::LessOp, mlir::TF::IdentityOp,
                     mlir::TF::LogicalAndOp, mlir::TF::ToBoolOp,
                     mlir::func::ReturnOp>(&op))
        return false;
    }

    // We then apply some heuristics:
    // 1. If all variants are needed to determine the predicate, do not convert
    // to async while because the next iteration has a low chance of overlapping
    // with the current iteration.
    // 2. The predicate shall be updated before any other variants is updated.
    std::vector<int> predicate_determinat =
        GetPredicateDeterminat(predicate_fn);
    ArgumentGroups argument_groups = GetArgumentGroups(body_fn);

    // Find out defining ops for predicate and non-predicate inside the body.
    llvm::SmallSetVector<mlir::Operation *, 4> predicate_determinant_ops;
    llvm::SmallSetVector<mlir::Operation *, 4> worker_ops;
    for (int i : argument_groups.mutables) {
      auto *defining_op = body_fn.getBody()
                              .front()
                              .getTerminator()
                              ->getOperand(i)
                              .getDefiningOp();
      if (std::find(predicate_determinat.begin(), predicate_determinat.end(),
                    i) != predicate_determinat.end()) {
        predicate_determinant_ops.insert(defining_op);
      } else {
        worker_ops.insert(defining_op);
      }
    }

    // simple heuristic to reject pipelining: predicate are determined by all
    // variants.
    if (worker_ops.empty()) return false;

    // Validate that the determinant of predicate must be updated before any
    // other variants and variants are determined in a cluster.
    int predicate_determinant_cnt = 0;
    for (auto &body_op : body_fn.getBlocks().front()) {
      if (predicate_determinant_cnt < predicate_determinant_ops.size()) {
        if (worker_ops.contains(&body_op)) {
          return false;
        } else if (predicate_determinant_ops.contains(&body_op)) {
          predicate_determinant_cnt++;
          if (predicate_determinant_cnt >= predicate_determinant_ops.size()) {
            break;
          }
        }
      }
    }

    return true;
  }

  // Returns the list of minimal argument indices that determines the predicate.
  std::vector<int> GetPredicateDeterminat(mlir::func::FuncOp predicate_fn) {
    std::vector<int> determinat;
    for (auto &op : predicate_fn.getArguments()) {
      if (!op.use_empty()) {
        determinat.push_back(op.getArgNumber());
      }
    }
    return determinat;
  }

  // Groups the arguments of the original body function into two groups:
  // mutables between iterations and immutables across iterations.
  ArgumentGroups GetArgumentGroups(mlir::func::FuncOp body_fn) {
    ArgumentGroups groups;
    auto *return_op = body_fn.getBlocks().front().getTerminator();

    int i = 0;
    for (auto return_op_operand : return_op->getOperands()) {
      if (std::find(body_fn.getArguments().begin(),
                    body_fn.getArguments().end(),
                    return_op_operand) != body_fn.getArguments().end()) {
        groups.immutables.push_back(i);
      } else {
        groups.mutables.push_back(i);
      }
      i++;
    }
    return groups;
  }

  // Re-orders the argument such that all the immutables are at the bottom of
  // the argument list. This function returns a mapping for such re-order.
  ArgumentRemap CreateArgumentRemap(mlir::func::FuncOp original_body_fn,
                                    std::vector<int> invariant) {
    absl::flat_hash_set<int> invariant_set(invariant.begin(), invariant.end());
    ArgumentRemap argument_remap;  // new to old
    argument_remap.old_to_new_remap.resize(
        original_body_fn.getArguments().size());
    argument_remap.new_to_old_remap.resize(
        original_body_fn.getArguments().size() + 1);

    const int invariant_start_index =
        original_body_fn.getArguments().size() - invariant_set.size();
    int variant_index = 0 + kNonPredicateStartIndex;
    int invariant_index = invariant_start_index + kNonPredicateStartIndex;
    for (int i = 0; i < original_body_fn.getArguments().size(); ++i) {
      if (invariant_set.contains(i)) {
        argument_remap.old_to_new_remap[i] = invariant_index;
        argument_remap.new_to_old_remap[invariant_index] = i;
        invariant_index++;
      } else {
        argument_remap.old_to_new_remap[i] = variant_index;
        argument_remap.new_to_old_remap[variant_index] = i;
        variant_index++;
      }
    }
    DCHECK_EQ(variant_index, invariant_start_index + kNonPredicateStartIndex);
    DCHECK_EQ(invariant_index,
              original_body_fn.getArguments().size() + kNonPredicateStartIndex);
    return argument_remap;
  }

  // Sorts the operations in the block to an order of first ready first run to
  // increase the parallelism between iterations. The order of ops before the
  // predicate can be computed remains unchanged because we prefer to have the
  // next iteration starts as soon as possible. For ops that are
  // ready at the same time, ops producing the final results will be executed
  // first.
  void SortBodyFnByFirstReadyFirstRun(
      mlir::func::FuncOp body_fn, const std::vector<int> &predicate_determinat,
      const mlir::TF::SideEffectAnalysis::Info &side_effect_analysis) {
    mlir::Block &body_block = body_fn.getBody().front();
    mlir::Operation *return_op = body_block.getTerminator();
    if (!return_op) return;

    // Ops in final execution order.
    std::vector<mlir::Operation *> sorted_ops;
    // Number of outstanding (not executed yet) dependency for an op. When that
    // number hits zero, this op is ready to be executed.
    llvm::DenseMap<mlir::Operation *, int> dependency_cnt;

    // Build the graph and mark ops that depends on input arguments in the
    // `new_ready_ops`. We will prioritize final result producing ops later on
    // before finalizing their order.
    std::list<mlir::Operation *> new_ready_ops;
    for (mlir::Operation &op : body_block.without_terminator()) {
      // Existing side effecting ops are dependencies on all subsequent ops.
      dependency_cnt[&op] =
          op.getNumOperands() +
          side_effect_analysis.DirectControlPredecessors(&op).size();

      // Block argument is viewed as ready right away
      for (const mlir::Value &operand : op.getOperands()) {
        if (!operand.getDefiningOp()) {
          dependency_cnt[&op]--;
        }
      }
      if (!dependency_cnt[&op]) new_ready_ops.push_back(&op);
    }

    // Ops that produces final returned results have priority among ops that are
    // ready at the same time.
    llvm::DenseSet<mlir::Operation *> final_result_producing_ops;
    for (const auto &operand : return_op->getOperands()) {
      if (auto *op = operand.getDefiningOp()) {
        final_result_producing_ops.insert(op);
      }
    }

    // The order of all ops before the predicate for the next iteration
    // can be computed remains unchanged because we want to kick off next
    // iteration preparation asap by making the predicate ready asap.
    llvm::SmallDenseSet<mlir::Operation *> predicate_determined_ops;
    for (int i : predicate_determinat) {
      if (auto *defining_op = return_op->getOperand(i).getDefiningOp()) {
        predicate_determined_ops.insert(defining_op);
      }
    }
    auto iter = body_block.without_terminator().begin();
    // Make those ops before predicate ready and remember them so that they
    // don't get resorted.
    llvm::SmallDenseSet<mlir::Operation *> pre_scheduled_ops;
    while (!predicate_determined_ops.empty()) {
      if (predicate_determined_ops.contains(&*iter)) {
        predicate_determined_ops.erase(&*iter);
      }
      sorted_ops.push_back(&*iter);
      pre_scheduled_ops.insert(&*iter);
      iter++;
    }

    // Topologically sorts the ops in a breadth first order.
    std::deque<mlir::Operation *> ready_ops;
    MoveNewReadyOps(ready_ops, std::move(new_ready_ops),
                    final_result_producing_ops);
    while (!ready_ops.empty()) {
      mlir::Operation *active_op = ready_ops.front();
      if (!pre_scheduled_ops.contains(active_op))
        sorted_ops.push_back(active_op);
      ready_ops.pop_front();
      // Do not directly push new ready ops into ready_ops yet.
      // We will re-arrange the new ready ops by moving those producing the
      // final results to the top so that they get executed first.
      std::list<mlir::Operation *> new_ready_ops;
      for (const auto &result : active_op->getOpResults()) {
        // Uses instead of users in case users can dedup.
        for (const auto &use : result.getUses()) {
          int cnt = --dependency_cnt[use.getOwner()];
          if (!cnt) new_ready_ops.push_back(use.getOwner());
        }
      }
      for (auto *control_dependent :
           side_effect_analysis.DirectControlSuccessors(active_op)) {
        int cnt = --dependency_cnt[control_dependent];
        if (!cnt) new_ready_ops.push_back(control_dependent);
      }

      MoveNewReadyOps(ready_ops, std::move(new_ready_ops),
                      final_result_producing_ops);
    }

    // Update the block with the new order.
    sorted_ops.push_back(return_op);
    for (mlir::Operation *op : sorted_ops) {
      op->remove();
      body_block.push_back(op);
    }
  }

  // Add ops in the `new_ready_ops` to `ready_ops` in the following order:
  // ops that produces final returned result are added first and then the rest
  // ops.
  void MoveNewReadyOps(
      std::deque<mlir::Operation *> &ready_ops,
      std::list<mlir::Operation *> new_ready_ops,
      const llvm::DenseSet<mlir::Operation *> &final_result_producing_ops) {
    auto iter = new_ready_ops.begin();
    while (iter != new_ready_ops.end()) {
      if (final_result_producing_ops.contains(*iter)) {
        ready_ops.push_back(*iter);
        iter = new_ready_ops.erase(iter);
      } else {
        iter++;
      }
    }
    for (mlir::Operation *op : new_ready_ops) {
      ready_ops.push_back(op);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateAsyncWhilePass() {
  return std::make_unique<AsyncWhilePass>();
}

}  // namespace mlrt_compiler
}  // namespace tensorflow
