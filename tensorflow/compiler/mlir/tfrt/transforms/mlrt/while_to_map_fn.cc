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
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/while_to_map_fn.h"

#include <linux/limits.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h"

namespace tensorflow {
namespace mlrt_compiler {
namespace {

void RemoveIdentityOp(mlir::func::FuncOp func) {
  auto &block = func.getBody().front();
  llvm::SmallVector<mlir::TF::IdentityOp> identity_ops;
  for (auto &op : block) {
    if (auto identity_op = llvm::dyn_cast<mlir::TF::IdentityOp>(&op)) {
      identity_ops.push_back(identity_op);
    }
  }

  for (auto op : llvm::reverse(identity_ops)) {
    op.getOutput().replaceAllUsesWith(op.getInput());
  }

  for (auto op : identity_ops) {
    op->erase();
  }

  auto return_op = llvm::cast<mlir::func::ReturnOp>(block.getTerminator());

  auto func_type = mlir::FunctionType::get(
      func.getContext(), func.getArgumentTypes(), return_op.getOperandTypes());

  func.setType(func_type);
}

// tf.map_fn (https://www.tensorflow.org/api_docs/python/tf/map_fn) is converted
// to tf.while during lowering. tf.map_fn expects parallel execution of its body
// function but not all tf.while can guarantee parallel executions. The tf.while
// op that is converted from tf.map_fn has distinct programming patterns. This
// pass matches those patterns to convert applicable tf.while to tf_mlrt.map_fn
// for parallel execution of the body function.
//
// For example, tf.map_fn(fn, elems, ...) can be converted to the following:
//
// %tensor_list =  "tf.TensorListReserve"(%per_iteration_shape, %max_iterations)
//
// %while_outputs:7 =  "tf.While"(%loop_counter,
// %tensor_list_index, %other_args, %tensor_list) {body = @while_body, cond =
// @while_cond}
//
// %outputs =  "tf.TensorListStack"(%while_outputs#2, %output_shape)
//
// in which
//
// while_cond: check loop_counter and tensor_list_index both smaller than
// max_iterations.
//
// while_body: loop_counter and tensor_list_index is incremented and returned;
// also gather input from elems based on un-incremented tensor_list_index,
// call fn and set output into a TensorList at tensor_list_index.
//
// This pass additionally assumes the following patterns to identify a tf.While
// that are converted from tf.map_fn:
// 1. Arguments have one loop_counter and one element_index that are initialized
// to be 0.
// 2. TensorList or TensorArray is reserved with max_iterations size. The
// max_iterations shall be a constant.
// 3. The predicate function check both loop_counter and element_index is less
// than max_iterations.
// 4. The body function increase loop_counter and element_index by 1 and use
// element_index to stores its result into  tensor list or tensor array such
// that there is no overlap in write between iterations
// 5. The body function does not have side effects such that one iteration will
// impact the next iteration outside #4.
//
// After conversion, the pseudocode is
//
// %tensor_list =  "tf.TensorListReserve"(%per_iteration_shape, %max_iterations)
//
// %updated_tensor_list =  "tf_mlrt.map_fn" (%max_iterations, %tensor_list,
// %other_args) {body = @map_fn_body}
//
// %outputs = "tf.TensorListStack"(%updated_tensor_list, %output_shape)
//
// where
//
// tf_mlrt.map_fn leads to a blocking call and
// the argument list of tf_mlrt.map_fn is (%max_iterations, %tensor_list,
// tf.while's argument list minus loop_counter, tensor_list_index and
// tensor_list). tf_mlrt.map_fn is a block call and returns the updated tensor
// list.
//
// map_fn_body has an input signature of (%in_tensor_list_future,
// %out_tensor_list_promise, %loop_counter, %tensor_list_index, %other_args) and
// has not return values (the updated_tensor_list is delivered through
// %out_tensor_list_promise).
//
class WhileToMapFnPass
    : public mlir::PassWrapper<WhileToMapFnPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  WhileToMapFnPass() = default;
  WhileToMapFnPass &operator=(const WhileToMapFnPass &) = delete;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WhileToMapFnPass)

 private:
  struct LoopInfo {
    // Argument indices in while op of key loop variables.
    int loop_counter = -1;
    int element_index = -1;
    std::vector<int> tensor_list_or_flow_in;
    // Max iteration may be passed in as an argument to while op.
    std::optional<int> max_iterations_arg_idx;
    // Max itertions may be hard coded as constant inside while predicate
    // function.
    std::optional<int> max_iterations_value;
    // Defining Op of max_iterations.
    mlir::Value max_iterations;
  };

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<tensorflow::tf_mlrt::TensorflowMlrtDialect>();
    registry.insert<mlrt::compiler::MlrtDialect>();
  }

  llvm::StringRef getArgument() const final {
    return "tf-mlrt-while-to-map-fn";
  }

  llvm::StringRef getDescription() const final {
    return "Convert tf.while to tf_mlrt.map_fn when possible for parallel "
           "execution.";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::SymbolTable symbol_table(module);

    // Use make_early_inc_range because the processing might insert new node
    // into the list
    for (auto func_op :
         llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
      MayConvertWhileToMapFn(func_op, symbol_table);
    }
  }

  // We match while op's predicate function and body function with known
  // patterns from tf.map_fn. If matched, tf.while is converted to
  // tf_mlrt.map_fn.
  void MayConvertWhileToMapFn(mlir::func::FuncOp op,
                              mlir::SymbolTable &symbol_table) {
    mlir::OpBuilder builder(op);
    for (mlir::Operation &op : llvm::make_early_inc_range(op.front())) {
      auto while_op = llvm::dyn_cast<mlir::TF::WhileOp>(&op);
      if (!while_op) continue;
      LoopInfo loop_info;
      if (mlir::succeeded(MatchPredicate(while_op.getCondAttr(), symbol_table,
                                         loop_info)) &&
          mlir::succeeded(
              MatchBody(while_op.getBodyAttr(), symbol_table, loop_info)) &&
          mlir::succeeded(MatchInputSource(while_op, loop_info)) &&
          mlir::succeeded(MatchOutputUse(while_op, loop_info))) {
        // Input, predicate function, body function and output are all following
        // patterns, we can convert it to tf_mlrt.map_fn.
        mlir::func::FuncOp while_body_func =
            symbol_table.lookup<mlir::func::FuncOp>(while_op.getBody());
        auto map_fn_body_func = CreateMapFnBodyFunction(
            builder, while_body_func, symbol_table, loop_info);

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(while_op);
        std::vector<mlir::Value> invariant_arguments;
        invariant_arguments.reserve(while_op->getNumOperands());

        absl::flat_hash_set<int> variant_arguments = {loop_info.loop_counter,
                                                      loop_info.element_index};
        variant_arguments.insert(loop_info.tensor_list_or_flow_in.begin(),
                                 loop_info.tensor_list_or_flow_in.end());
        for (int i = 0; i < while_op->getNumOperands(); ++i) {
          if (variant_arguments.contains(i)) {
            continue;
          }
          invariant_arguments.push_back(while_op.getOperand(i));
        }

        llvm::SmallVector<mlir::Type> result_types;
        llvm::SmallVector<mlir::Value> tensor_list_operands;
        for (int i = 0; i < loop_info.tensor_list_or_flow_in.size(); ++i) {
          tensor_list_operands.push_back(
              while_op.getOperand(loop_info.tensor_list_or_flow_in[i]));
          result_types.push_back(
              while_op.getResult(loop_info.tensor_list_or_flow_in[i])
                  .getType());
        }

        auto map_fn_op = builder.create<tf_mlrt::TFMapFnOp>(
            while_op.getLoc(), result_types, loop_info.max_iterations,
            tensor_list_operands, invariant_arguments,
            map_fn_body_func.getSymName(),
            loop_info.tensor_list_or_flow_in.size());

        // MatchOutputUse already makes sure only the tensor_list or
        // tensor_array output is used.
        absl::flat_hash_map<int, int> old_arg_indx_to_new_index;
        for (int i = 0; i < loop_info.tensor_list_or_flow_in.size(); ++i) {
          old_arg_indx_to_new_index.insert(
              {loop_info.tensor_list_or_flow_in[i], i});
        }
        for (int i = 0; i < while_op.getResults().size(); ++i) {
          if (!old_arg_indx_to_new_index.contains(i)) {
            while_op.getResult(i).dropAllUses();
          } else {
            while_op.getResult(i).replaceAllUsesWith(
                map_fn_op.getResult()[old_arg_indx_to_new_index[i]]);
          }
        }

        while_op.erase();
      }
    }
  }

  // Match that (a) the tensor list or tensor array are reserved with
  // max_iterations size such that parallel operations on tensor list or tensor
  // array is thread safe; (b) loop_counter and element_index starts with 0.
  // Also may identify source of max_iterations.
  mlir::LogicalResult MatchInputSource(mlir::TF::WhileOp while_op,
                                       LoopInfo &loop_info) {
    // Element index and loop counter should start from 0.
    if (!mlir::matchPattern(while_op.getOperand(loop_info.loop_counter),
                            mlir::m_Zero()) ||
        !mlir::matchPattern(while_op.getOperand(loop_info.element_index),
                            mlir::m_Zero())) {
      return mlir::failure();
    }

    DCHECK_GE(loop_info.tensor_list_or_flow_in.size(), 1);
    // Tensor list or a tensor array are reserved

    for (auto tensor_list_index : loop_info.tensor_list_or_flow_in) {
      mlir::Operation *tensor_list_or_flow_in_defining_op =
          while_op.getOperand(tensor_list_index).getDefiningOp();
      if (tensor_list_or_flow_in_defining_op == nullptr) {
        return mlir::failure();
      }

      mlir::Operation *max_iterations = nullptr;
      if (loop_info.max_iterations_arg_idx.has_value()) {
        max_iterations =
            while_op.getOperand(loop_info.max_iterations_arg_idx.value())
                .getDefiningOp();
      }
      if (auto tensor_list_reserve =
              llvm::dyn_cast<mlir::TF::TensorListReserveOp>(
                  tensor_list_or_flow_in_defining_op)) {
        // Tensor list should resever for max_iterations.
        mlir::Operation *tensor_list_reserve_size =
            tensor_list_reserve.getNumElements().getDefiningOp();

        if (tensor_list_reserve_size != max_iterations) {
          // if tensor list is not reserved by max_iteration variable, then
          // another acceptable case is that both contain same constant values.
          llvm::APInt reserved_cst;
          if (!mlir::matchPattern(tensor_list_reserve_size,
                                  mlir::m_ConstantInt(&reserved_cst)) ||
              !loop_info.max_iterations_value.has_value() ||
              reserved_cst.getZExtValue() !=
                  loop_info.max_iterations_value.value()) {
            return mlir::failure();
          }
        }
        // TensorListReserveOp has only one result and is already in used by
        // while.
        loop_info.max_iterations = tensor_list_reserve.getNumElements();
      } else if (auto tensor_array = llvm::dyn_cast<mlir::TF::TensorArrayV3Op>(
                     tensor_list_or_flow_in_defining_op)) {
        mlir::Operation *tensor_array_size =
            tensor_array.getOperand().getDefiningOp();
        if (tensor_array_size != max_iterations) {
          // if tensor array is not reserved by max_iteration variable, then
          // another acceptable case is that both contain same constant values.
          llvm::APInt reserved_cst;
          if (!mlir::matchPattern(tensor_array_size,
                                  mlir::m_ConstantInt(&reserved_cst)) ||
              !loop_info.max_iterations_value.has_value() ||
              reserved_cst.getZExtValue() !=
                  loop_info.max_iterations_value.value()) {
            return mlir::failure();
          }
        }

        // Other than flow_in, the tensor array should be used by while as well.
        if (!llvm::is_contained(while_op.getOperands(),
                                tensor_array.getHandle())) {
          return mlir::failure();
        }
        loop_info.max_iterations = tensor_array.getSize();
      } else {
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  // Match the map_attern that output of while op is subsequentially stacked.
  mlir::LogicalResult MatchOutputUse(mlir::TF::WhileOp &while_op,
                                     const LoopInfo &loop_info) {
    absl::flat_hash_set<int> used_results;
    used_results.insert(loop_info.tensor_list_or_flow_in.begin(),
                        loop_info.tensor_list_or_flow_in.end());
    for (int i = 0; i < while_op->getResults().size(); ++i) {
      if (used_results.contains(i)) {
        // Tensor list or flow in should be used next.
        if (!while_op->getResult(i).hasOneUse()) {
          return mlir::failure();
        }
      } else {
        // No other result should be used.
        if (!while_op->getResult(i).use_empty()) {
          return mlir::failure();
        }
      }
    }

    for (auto result_index : loop_info.tensor_list_or_flow_in) {
      // Finds the use of the tensor list or flow in is a tensor list stack or
      // tensor array gather. This maybe over-conservative, but we rather be
      // correct than sorry.
      mlir::Operation *use_op =
          *while_op->getResult(result_index).getUsers().begin();
      if (llvm::isa<mlir::TF::StopGradientOp>(use_op)) {
        use_op = *use_op->getUsers().begin();
      }

      if (!llvm::isa<mlir::TF::TensorListStackOp,
                     mlir::TF::TensorArrayGatherV3Op>(use_op)) {
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  // Match that the while predicate function is doing just
  // loop_counter < max iterations && element_index < max_iterations.
  // Through this pattern, we also update the argument index of
  // loop_counter, element_index and possibly max_iterations.
  mlir::LogicalResult MatchPredicate(mlir::FlatSymbolRefAttr predicate_fn,
                                     const mlir::SymbolTable &symbol_table,
                                     LoopInfo &loop_info) {
    mlir::func::FuncOp predicate_fn_op =
        symbol_table.lookup<mlir::func::FuncOp>(predicate_fn.getValue());

    // The body of the predicate function should have two LessOp and one
    // LogicalAndOp. It can optionally has IdentityOp and ToBoolOp.
    enum class PredicateBodyExpectingOp {
      kExpectFirstLess,
      kExpectSecondLess,
      kExpectLogicalAnd,
      kExpectTerminator
    };
    std::vector<mlir::Operation *> less_ops;
    less_ops.reserve(2);
    PredicateBodyExpectingOp expecting_op =
        PredicateBodyExpectingOp::kExpectFirstLess;
    for (auto &body_op : predicate_fn_op.getBody().front()) {
      switch (expecting_op) {
        case PredicateBodyExpectingOp::kExpectFirstLess:
          if (llvm::isa<mlir::TF::LessOp>(body_op)) {
            expecting_op = PredicateBodyExpectingOp::kExpectSecondLess;
            less_ops.push_back(&body_op);
          } else if (!llvm::isa<mlir::TF::ConstOp, mlir::TF::IdentityOp>(
                         body_op)) {
            return mlir::failure();
          }
          break;
        case PredicateBodyExpectingOp::kExpectSecondLess:
          if (llvm::isa<mlir::TF::LessOp>(body_op)) {
            expecting_op = PredicateBodyExpectingOp::kExpectLogicalAnd;
            less_ops.push_back(&body_op);
          } else if (!llvm::isa<mlir::TF::ConstOp, mlir::TF::IdentityOp>(
                         body_op)) {
            return mlir::failure();
          }
          break;
        case PredicateBodyExpectingOp::kExpectLogicalAnd:
          if (llvm::isa<mlir::TF::LogicalAndOp>(body_op)) {
            expecting_op = PredicateBodyExpectingOp::kExpectTerminator;
          } else if (!llvm::isa<mlir::TF::IdentityOp>(body_op)) {
            return mlir::failure();
          }
          break;
        case PredicateBodyExpectingOp::kExpectTerminator:
          if (!llvm::isa<mlir::TF::ToBoolOp, mlir::func::ReturnOp,
                         mlir::TF::IdentityOp>(body_op)) {
            return mlir::failure();
          }
          break;
        default:
          return mlir::failure();
      }
    }

    // Identify loop_counter
    int counter_index = -1;
    auto counter_iter =
        llvm::find(predicate_fn_op.getArguments(), less_ops[0]->getOperand(0));
    if (counter_iter != predicate_fn_op.getArguments().end()) {
      counter_index = counter_iter->getArgNumber();
      if (!IsScalarOrUnrankedI32Tensor(
              predicate_fn_op.getArgument(counter_index))) {
        return mlir::failure();
      }
    }

    // Find upper bound on loop_counter.
    int max_iter_index_from_counter = -1;
    int max_iter_value_from_counter = -1;
    if (auto max_iter_iter = llvm::find(predicate_fn_op.getArguments(),
                                        less_ops[0]->getOperand(1));
        max_iter_iter != predicate_fn_op.getArguments().end()) {
      // Upper bound on loop_counter is from one argument.
      max_iter_index_from_counter = max_iter_iter->getArgNumber();
      // Argument has to be int32
      if (!IsScalarOrUnrankedI32Tensor(
              predicate_fn_op.getArgument(max_iter_index_from_counter))) {
        return mlir::failure();
      }
    } else {
      // If upper bound is not passed in, it has to be a constant
      llvm::APInt value;
      if (!mlir::matchPattern(less_ops[0]->getOperand(1).getDefiningOp(),
                              mlir::m_ConstantInt(&value))) {
        return mlir::failure();
      }
      max_iter_value_from_counter = value.getZExtValue();
    }

    // Identify element_index
    int element_index = -1;
    auto element_index_iter =
        llvm::find(predicate_fn_op.getArguments(), less_ops[1]->getOperand(0));
    if (element_index_iter != predicate_fn_op.getArguments().end()) {
      element_index = element_index_iter->getArgNumber();
      if (!IsScalarOrUnrankedI32Tensor(
              predicate_fn_op.getArgument(element_index))) {
        return mlir::failure();
      }
    }

    // Find upper bound on element_index.
    int max_iter_index_from_element = -1;
    int max_iter_value_from_element = -1;
    if (auto max_iter_iter = llvm::find(predicate_fn_op.getArguments(),
                                        less_ops[1]->getOperand(1));
        max_iter_iter != predicate_fn_op.getArguments().end()) {
      // Upper bound on element_index is from one argument.
      max_iter_index_from_element = max_iter_iter->getArgNumber();
      // Upper bound argument needs to be int32
      if (!IsScalarOrUnrankedI32Tensor(
              predicate_fn_op.getArgument(max_iter_index_from_element))) {
        return mlir::failure();
      }
    } else {
      // If upper bound is not passed in, it has to be a constant
      llvm::APInt value;
      if (!mlir::matchPattern(less_ops[1]->getOperand(1).getDefiningOp(),
                              mlir::m_ConstantInt(&value))) {
        return mlir::failure();
      }
      max_iter_value_from_element = value.getZExtValue();
    }

    // Loop_counter is always available.
    if (counter_index < 0) return mlir::failure();
    // element_index can change its location, but will always be provided.
    if (element_index < 0) return mlir::failure();

    std::optional<int> max_iter_const;
    std::optional<int> max_iter_index;
    if (max_iter_index_from_counter < 0 && max_iter_index_from_element < 0) {
      // If both loop counter and element index are not upper bounded by passing
      // in arguments, they shall be upper bounded by constants of same value.
      if (max_iter_value_from_element != max_iter_value_from_counter ||
          max_iter_value_from_element < 0 || max_iter_value_from_counter < 0) {
        return mlir::failure();
      } else {
        max_iter_const = max_iter_value_from_element;
      }
    } else if (max_iter_index_from_counter >= 0 &&
               max_iter_index_from_element >= 0) {
      // Loop counter or element are upper bounded by pass-in arguments.
      // They need to be upper bounded by the same argument
      if (max_iter_index_from_element != max_iter_index_from_counter) {
        return mlir::failure();
      } else {
        max_iter_index = max_iter_index_from_counter;
      }
    } else {
      // TODO(deqiangc): remove this clause after verifying grappler pass remove
      // the case that one of them is bounded by pass-in argument and the other
      // is bounded by constants.
      max_iter_index =
          std::max(max_iter_index_from_counter, max_iter_index_from_element);
      max_iter_const =
          std::max(max_iter_value_from_element, max_iter_value_from_counter);
    }

    // Update hypothesis
    loop_info.loop_counter = counter_index;
    loop_info.element_index = element_index;
    loop_info.max_iterations_arg_idx = max_iter_index;
    loop_info.max_iterations_value = max_iter_const;
    return mlir::success();
  }

  // Match that the current hypothesis of current loop_counter and element_index
  // in the while body function based on the following simple pattern:
  // %updated_loop_counter = %loop_counter + 1
  // %updated_element_index = %element_index + 1
  // %loaded_elem = tf.Gather(.., %element_index,... )
  // DoSomething
  // tf.TensorListSetItem(.., %element_index)
  // return %update_loop_counter, %updated_element_index,
  // %tensor_array_list, %max_iterations, %other_args
  mlir::LogicalResult MatchLoopCounterElementIndexInBody(
      mlir::func::FuncOp while_body_func, LoopInfo &loop_info) {
    mlir::Block &block = while_body_func.getBlocks().front();

    // Verify argument loop_counter is +1 and returned at the same location.
    mlir::BlockArgument loop_counter =
        block.getArgument(loop_info.loop_counter);
    llvm::SmallVector<mlir::Operation *> loop_counter_users =
        GetUsersIgnoringIdentityOp(loop_counter);
    if (loop_counter_users.size() != 1 ||
        !llvm::isa<mlir::TF::AddOp, mlir::TF::AddV2Op>(
            loop_counter_users.front()) ||
        !mlir::matchPattern(
            loop_counter_users.front()->getOperand(1).getDefiningOp(),
            mlir::m_One())) {
      return mlir::failure();
    }

    // loop_counter + 1 is in ReturnOp's operand.
    if (loop_counter_users.front() !=
        GetDefiningOpIgnoringIdentityOp(
            GetReturnedOperand(while_body_func, loop_info.loop_counter))) {
      return mlir::failure();
    }

    // Verify element_index's usage and also identify the argument index of
    // tensor list or tensor array flow_in.
    std::vector<int> tensor_list_or_flow_in_index;
    mlir::BlockArgument element_index =
        block.getArgument(loop_info.element_index);
    for (auto *element_index_use : GetUsersIgnoringIdentityOp(element_index)) {
      if (llvm::isa<mlir::TF::AddOp, mlir::TF::AddV2Op>(element_index_use)) {
        // One use of element_index is +1 and then returned at the same
        // location.
        if (!mlir::matchPattern(
                element_index_use->getOperand(1).getDefiningOp(),
                mlir::m_One()) ||
            element_index_use !=
                GetDefiningOpIgnoringIdentityOp(GetReturnedOperand(
                    while_body_func, loop_info.element_index))) {
          return mlir::failure();
        }
      } else if (llvm::isa<mlir::TF::TensorListSetItemOp>(element_index_use)) {
        if (auto tensor_list_index = MayGetArgumentIndexIgnoringIdentityOp(
                while_body_func,
                llvm::dyn_cast<mlir::TF::TensorListSetItemOp>(element_index_use)
                    .getInputHandle());
            !tensor_list_index.has_value()) {
          return mlir::failure();
        } else {
          tensor_list_or_flow_in_index.push_back(tensor_list_index.value());
        }
      } else if (llvm::isa<mlir::TF::TensorArrayWriteV3Op>(element_index_use)) {
        if (auto flow_in_index = MayGetArgumentIndexIgnoringIdentityOp(
                while_body_func, llvm::dyn_cast<mlir::TF::TensorArrayWriteV3Op>(
                                     element_index_use)
                                     .getFlowIn());
            !flow_in_index.has_value()) {
          return mlir::failure();
        } else {
          tensor_list_or_flow_in_index.push_back(flow_in_index.value());
        }
      } else if (!llvm::isa<mlir::TF::GatherOp, mlir::TF::GatherV2Op,
                            mlir::TF::TensorListGetItemOp,
                            mlir::TF::TensorArrayReadV3Op>(element_index_use)) {
        // The only other use is to either gather the input or set output.
        return mlir::failure();
      }
    }

    if (tensor_list_or_flow_in_index.empty()) {
      return mlir::failure();
    }

    // Update hypothesis
    loop_info.tensor_list_or_flow_in = std::move(tensor_list_or_flow_in_index);

    return mlir::success();
  }

  // Match that the while body function is the following simple pattern:
  // %updated_loop_counter = %loop_counter + 1
  // %updated_element_index = %element_index + 1
  // %loaded_elem = tf.Gather(.., %element_index,... )
  // DoSomething
  // tf.TensorListSetItem(.., %element_index)
  // return %update_loop_counter, %updated_element_index,
  // %tensor_array_list, %max_iterations, %other_args
  //
  // in which
  // DoSomething has no side-effect on the next iteration.
  //
  // Also identify argument index for TensorList or TensorArray flow_in.
  mlir::LogicalResult MatchBody(mlir::FlatSymbolRefAttr while_body_func_name,
                                const mlir::SymbolTable &symbol_table,
                                LoopInfo &loop_info) {
    mlir::func::FuncOp while_body_func =
        symbol_table.lookup<mlir::func::FuncOp>(
            while_body_func_name.getValue());

    if (mlir::failed(
            MatchLoopCounterElementIndexInBody(while_body_func, loop_info))) {
      // Swap the order of loop_counter and element_index in the current
      // hypothesis and try again
      int swap = loop_info.loop_counter;
      loop_info.loop_counter = loop_info.element_index;
      loop_info.element_index = swap;
      if (mlir::failed(
              MatchLoopCounterElementIndexInBody(while_body_func, loop_info))) {
        return mlir::failure();
      }
    }

    // The next iteration of while_body does not depend on the previous
    // iteration except loop_counter, element_index, tensor_list_or_flow_in, and
    // max_iterations.
    absl::flat_hash_set<int> allowed_variable_between_iterations;
    allowed_variable_between_iterations.insert(loop_info.loop_counter);
    allowed_variable_between_iterations.insert(loop_info.element_index);
    if (loop_info.max_iterations_arg_idx.has_value()) {
      allowed_variable_between_iterations.insert(
          loop_info.max_iterations_arg_idx.value());
    }
    allowed_variable_between_iterations.insert(
        loop_info.tensor_list_or_flow_in.begin(),
        loop_info.tensor_list_or_flow_in.end());
    for (int j = 0; j < while_body_func.getNumArguments(); j++) {
      if (!allowed_variable_between_iterations.contains(j)) {
        if (GetReturnedOperand(while_body_func, j) !=
            while_body_func.getArgument(j)) {
          return mlir::failure();
        }
      }
    }

    return mlir::success();
  }

  // The map_fn body function is a clone of the while_body_func that
  // canonicalize loop_counter and tensor_list_index to be the first two
  // arguments.
  mlir::func::FuncOp CreateMapFnBodyFunction(mlir::OpBuilder &builder,
                                             mlir::func::FuncOp while_body_func,
                                             mlir::SymbolTable &symbol_table,
                                             const LoopInfo &loop_info) {
    std::string map_fn_body_name =
        absl::StrCat(while_body_func.getSymName().str(), "/MapFnBody");

    if (auto func = symbol_table.lookup<mlir::func::FuncOp>(map_fn_body_name)) {
      return func;
    }

    RemoveIdentityOp(while_body_func);

    absl::flat_hash_set<int> variant_arguments = {loop_info.loop_counter,
                                                  loop_info.element_index};
    variant_arguments.insert(loop_info.tensor_list_or_flow_in.begin(),
                             loop_info.tensor_list_or_flow_in.end());
    llvm::SmallVector<mlir::Type> remapped_input_type;

    for (int i = 0; i < loop_info.tensor_list_or_flow_in.size(); i++) {
      remapped_input_type.push_back(
          builder.getType<mlrt::compiler::FutureType>());
      remapped_input_type.push_back(
          builder.getType<mlrt::compiler::PromiseType>());
    }

    remapped_input_type.push_back(
        while_body_func.getFunctionType().getInput(loop_info.loop_counter));
    remapped_input_type.push_back(
        while_body_func.getFunctionType().getInput(loop_info.element_index));
    for (int i = 0; i < while_body_func.getFunctionType().getNumInputs(); i++) {
      if (!variant_arguments.contains(i)) {
        remapped_input_type.push_back(
            while_body_func.getFunctionType().getInput(i));
      }
    }
    mlir::OpBuilder::InsertionGuard insertion_guard(builder);
    builder.setInsertionPointAfter(while_body_func);
    auto map_fn_body_func = builder.create<mlir::func::FuncOp>(
        while_body_func.getLoc(), map_fn_body_name,
        mlir::FunctionType::get(while_body_func.getContext(),
                                remapped_input_type, {}));

    map_fn_body_func->setAttr(
        "tfrt.cost_threshold",
        builder.getI64IntegerAttr(std::numeric_limits<uint32_t>::max()));

    if (while_body_func.getArgAttrs().has_value()) {
      llvm::SmallVector<mlir::Attribute> remapped_input_attributes;
      // No attributes carry over for tensor list future/promise.
      for (int i = 0; i < loop_info.tensor_list_or_flow_in.size(); i++) {
        remapped_input_attributes.push_back(mlir::Attribute());
        remapped_input_attributes.push_back(mlir::Attribute());
      }
      auto args_attrs = while_body_func.getArgAttrs().value();
      remapped_input_attributes.push_back(args_attrs[loop_info.loop_counter]);
      remapped_input_attributes.push_back(args_attrs[loop_info.element_index]);
      for (int i = 0; i < args_attrs.size(); i++) {
        if (!variant_arguments.contains(i)) {
          remapped_input_attributes.push_back(args_attrs[i]);
        }
      }
      map_fn_body_func.setAllArgAttrs(remapped_input_attributes);
    }
    auto future_index = [](int i) { return 2 * i; };
    auto promise_index = [](int i) { return 2 * i + 1; };

    if (while_body_func.getResAttrs().has_value()) {
      // The order and types of results remain the same; so does attributes.
      map_fn_body_func.setAllResultAttrs(while_body_func.getResAttrs().value());
    }
    map_fn_body_func.setVisibility(mlir::func::FuncOp::Visibility::Private);

    builder.setInsertionPointToEnd(map_fn_body_func.addEntryBlock());

    mlir::IRMapping mapping;
    std::vector<tf_mlrt::TFAwaitOp> await_ops;
    for (int i = 0; i < loop_info.tensor_list_or_flow_in.size(); i++) {
      await_ops.push_back(builder.create<tf_mlrt::TFAwaitOp>(
          while_body_func.getLoc(),
          while_body_func.getArgument(loop_info.tensor_list_or_flow_in.at(i))
              .getType(),
          map_fn_body_func.getArgument(future_index(i))));

      mapping.map(
          while_body_func.getArgument(loop_info.tensor_list_or_flow_in.at(i)),
          await_ops.at(i));
    }
    // Rest of argument start after promise
    int map_fn_argument_index =
        promise_index(loop_info.tensor_list_or_flow_in.size() - 1);
    mapping.map(while_body_func.getArgument(loop_info.loop_counter),
                map_fn_body_func.getArgument(++map_fn_argument_index));
    mapping.map(while_body_func.getArgument(loop_info.element_index),
                map_fn_body_func.getArgument(++map_fn_argument_index));
    for (int i = 0; i < while_body_func.getNumArguments(); i++) {
      if (!variant_arguments.contains(i)) {
        mapping.map(while_body_func.getArgument(i),
                    map_fn_body_func.getArgument(++map_fn_argument_index));
      }
    }

    for (auto &op : while_body_func.getBody().front()) {
      builder.clone(op, mapping);
    }

    auto return_op = map_fn_body_func.getBody().front().getTerminator();

    mlir::Operation *first_write = nullptr;
    // Move tensor list write to the end of the block.
    for (int index : loop_info.tensor_list_or_flow_in) {
      auto *def = return_op->getOperand(index).getDefiningOp();
      CHECK(def);  // Crash OK
      def->moveBefore(return_op);
      if (!first_write) first_write = def;
    }

    // Move the await op before the first write.
    for (auto tensor_list_or_flow_in : await_ops) {
      tensor_list_or_flow_in->moveBefore(first_write);
    }

    // Insert promise right before return
    builder.setInsertionPoint(return_op);
    for (int i = 0; i < await_ops.size(); i++) {
      builder.create<tf_mlrt::TFPromiseOp>(
          return_op->getLoc(), map_fn_body_func.getArgument(promise_index(i)),
          return_op->getOperand(loop_info.tensor_list_or_flow_in.at(i)));
    }
    builder.create<mlir::func::ReturnOp>(return_op->getLoc());
    return_op->erase();

    symbol_table.insert(map_fn_body_func);

    return map_fn_body_func;
  }

  std::optional<int> MayGetArgumentIndexIgnoringIdentityOp(
      mlir::func::FuncOp func, mlir::Value value) const {
    // Value may go through some identify chains.
    while (value.getDefiningOp()) {
      if (!llvm::isa<mlir::TF::IdentityOp>(value.getDefiningOp())) {
        return std::nullopt;
      }
      value = value.getDefiningOp()->getOperand(0);
    }

    // Value is directly from argument since it has no defining op.
    auto argument_iter = llvm::find(func.getArguments(), value);
    if (argument_iter == func.getArguments().end()) {
      return std::nullopt;
    }
    return argument_iter->getArgNumber();
  }

  // Given a value, find its use ignoring identify op.
  // For example, given the below chains:
  //
  // %original_value = OriginalDefinedOp()
  // %value1 = tf.IdentifyOp(original_value)
  // %value2 = tf.IdentifyOp(value1)
  // UseOp(%value2)
  //
  // GetUseIgnroningIdentifyOp(%original_value) will return UseOp
  llvm::SmallVector<mlir::Operation *> GetUsersIgnoringIdentityOp(
      mlir::Value value) {
    llvm::SmallVector<mlir::Operation *> users;
    std::vector<mlir::Operation *> users_stack;

    for (auto *direct_user : value.getUsers()) {
      users_stack.push_back(direct_user);
    }

    while (!users_stack.empty()) {
      mlir::Operation *descendent_user = users_stack.back();
      users_stack.pop_back();

      if (!llvm::isa<mlir::TF::IdentityOp>(descendent_user)) {
        users.push_back(descendent_user);
      } else {
        // User of identify op is considered as user.
        for (auto *user : descendent_user->getResult(0).getUsers()) {
          users_stack.push_back(user);
        }
      }
    }
    return users;
  }

  // Given a value, find its source defined op ignoring identify op.
  // For example, given the below chains:
  //
  // %original_value = OriginalDefinedOp()
  // %value1 = tf.IdentifyOp(original_value)
  // %value2 = tf.IdentifyOp(value1)
  // UseOp(%value2)
  //
  // GetDefiningOpIgnroningIdentifyOp(%value2) will return OriginalDefinedOp
  mlir::Operation *GetDefiningOpIgnoringIdentityOp(mlir::Value value) {
    mlir::Operation *source_op = value.getDefiningOp();
    while (llvm::isa<mlir::TF::IdentityOp>(source_op)) {
      source_op = source_op->getOperand(0).getDefiningOp();
    }
    return source_op;
  }

  mlir::Value GetReturnedOperand(const mlir::func::FuncOp func,
                                 uint32_t result_index) {
    auto return_op = llvm::dyn_cast<mlir::func::ReturnOp>(
        func->getRegion(0).front().getTerminator());
    DCHECK_NE(return_op, nullptr);
    return return_op->getOperand(result_index);
  }

  bool IsScalarI32Tensor(mlir::Value value) const {
    if (auto value_type = llvm::dyn_cast<mlir::TensorType>(value.getType())) {
      if (value_type.getElementType().isInteger(32) && value_type.hasRank() &&
          value_type.getRank() == 0) {
        return true;
      }
    }
    return false;
  }

  bool IsScalarOrUnrankedI32Tensor(mlir::Value value) const {
    if (auto value_type = llvm::dyn_cast<mlir::TensorType>(value.getType())) {
      if (value_type.getElementType().isInteger(32) &&
          ((value_type.hasRank() && value_type.getRank() == 0) ||
           !value_type.hasRank())) {
        return true;
      }
    }
    return false;
  }
};
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateWhileToMapFnPass() {
  return std::make_unique<WhileToMapFnPass>();
}

}  // namespace mlrt_compiler
}  // namespace tensorflow
