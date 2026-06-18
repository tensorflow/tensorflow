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

#include "tensorflow/dtensor/mlir/topological_iterator.h"

#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/mlir/op_utils.h"

namespace tensorflow {
namespace dtensor {

TopologicalIterator::TopologicalIterator(mlir::func::FuncOp main_func)
    : ops_to_visit_{&main_func.front().front()} {
  funcs_visited_.insert(main_func.getName());
  funcs_visited_in_call_stack_.insert(main_func.getName());
}

mlir::Operation* TopologicalIterator::next() {
  if (!hasNext()) return nullptr;

  auto* op = ops_to_visit_.pop_back_val();
  auto* next_op = op->getNextNode();
  if (next_op) ops_to_visit_.push_back(next_op);

  // If this is a function call op, push the first op of the function body so
  // that the function body is converted before the call site.
  std::optional<mlir::func::FuncOp> func = MaybeFindFunction(op);
  if (func.has_value()) {
    mlir::StringRef func_name = func->getName();

    if (funcs_visited_.contains(func_name)) return next();

    ops_to_visit_.push_back(&(func->front().front()));
    funcs_visited_.insert(func_name);
  }

  // If we have reached the end of a function body, remove the function from
  // our active set.
  if (!next_op && !funcs_visited_in_call_stack_.empty())
    if (auto func = op->getParentOfType<mlir::func::FuncOp>())
      funcs_visited_in_call_stack_.erase(func.getName());

  if (auto cluster_op = mlir::dyn_cast<mlir::tf_device::ClusterOp>(op))
    ops_to_visit_.push_back(&cluster_op.GetBody().front());

  if (auto while_op = mlir::dyn_cast<mlir::TF::WhileRegionOp>(op)) {
    ops_to_visit_.push_back(&while_op.getCond().front().front());
    ops_to_visit_.push_back(&while_op.getBody().front().front());
  }

  if (auto if_op = mlir::dyn_cast<mlir::TF::IfRegionOp>(op)) {
    ops_to_visit_.push_back(&if_op.getThenBranch().front().front());
    ops_to_visit_.push_back(&if_op.getElseBranch().front().front());
  }
  return op;
}

bool TopologicalIterator::hasNext() { return !ops_to_visit_.empty(); }

}  // namespace dtensor
}  // namespace tensorflow
