/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/ifrt_interfaces.h"

#include <optional>

#include "absl/log/check.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"

// Generated definitions.

#define GET_ATTR_INTERFACE_CLASSES
#include "xla/python/ifrt/ir/ifrt_attr_interfaces.cc.inc"

#define GET_OP_INTERFACE_CLASSES
#include "xla/python/ifrt/ir/ifrt_op_interfaces.cc.inc"

namespace xla {
namespace ifrt {
namespace detail {
namespace {

// Returns the mutable `control_inputs` operand range of `op` if `op` accepts
// variadic IFRT control dependencies, or `std::nullopt` otherwise.
std::optional<mlir::MutableOperandRange> GetControlInputsMutable(
    mlir::Operation* op) {
  return llvm::TypeSwitch<mlir::Operation*,
                          std::optional<mlir::MutableOperandRange>>(op)
      .Case<IfrtArrayOpInterface, AssembleOp, DisassembleOp>(
          [](auto control_op) { return control_op.getControlInputsMutable(); })
      .Default([](mlir::Operation*) { return std::nullopt; });
}

}  // namespace

void ForwardControlInputsToUsers(mlir::Value control_output,
                                 mlir::ValueRange control_inputs,
                                 mlir::RewriterBase& rewriter) {
  if (control_output.use_empty()) {
    return;
  }

  llvm::SmallVector<mlir::Value> unique_control_inputs;
  unique_control_inputs.reserve(control_inputs.size());
  llvm::SmallDenseSet<mlir::Value, 8> seen_inputs;
  for (mlir::Value control_input : control_inputs) {
    if (seen_inputs.insert(control_input).second) {
      unique_control_inputs.push_back(control_input);
    }
  }

  llvm::SmallVector<mlir::Operation*> users;
  llvm::SmallPtrSet<mlir::Operation*, 4> seen_users;
  for (mlir::Operation* user : control_output.getUsers()) {
    if (seen_users.insert(user).second) {
      users.push_back(user);
    }
  }

  for (mlir::Operation* user : users) {
    std::optional<mlir::MutableOperandRange> user_control_inputs =
        GetControlInputsMutable(user);
    if (user_control_inputs.has_value()) {
      mlir::OperandRange current_control_inputs =
          user_control_inputs->getAsOperandRange();
      llvm::SmallVector<mlir::Value> new_control_inputs;
      new_control_inputs.reserve(current_control_inputs.size() +
                                 unique_control_inputs.size());
      llvm::SmallDenseSet<mlir::Value, 8> seen_values;
      for (mlir::Value control_value : current_control_inputs) {
        if (control_value == control_output) {
          for (mlir::Value forwarded_input : unique_control_inputs) {
            if (seen_values.insert(forwarded_input).second) {
              new_control_inputs.push_back(forwarded_input);
            }
          }
        } else if (seen_values.insert(control_value).second) {
          new_control_inputs.push_back(control_value);
        }
      }
      rewriter.modifyOpInPlace(
          user, [&]() { user_control_inputs->assign(new_control_inputs); });
      continue;
    }

    CHECK_EQ(unique_control_inputs.size(), 1)
        << "Cannot forward " << unique_control_inputs.size()
        << " unique control inputs to non-variadic user op: "
        << user->getName().getStringRef().str();
    const mlir::Value single_control_input = unique_control_inputs.front();
    rewriter.modifyOpInPlace(user, [&]() {
      for (mlir::OpOperand& operand : user->getOpOperands()) {
        if (operand.get() == control_output) {
          operand.set(single_control_input);
        }
      }
    });
  }
}

}  // namespace detail
}  // namespace ifrt
}  // namespace xla

namespace mlir {
namespace OpTrait {
namespace xla {
namespace ifrt {
namespace impl {

LogicalResult verifyNestedInIfrtFunc(Operation* op) {
  auto func_op = op->getParentOfType<func::FuncOp>();
  if (func_op != nullptr &&
      !func_op->hasAttr(::xla::ifrt::kIfrtFunctionAttrName) &&
      !func_op->hasAttr(::xla::ifrt::kIfrtReshardFunctionAttrName)) {
    return op->emitOpError()
           << "must be in a FuncOp with attr `"
           << ::xla::ifrt::kIfrtFunctionAttrName << "` or atttr `"
           << ::xla::ifrt::kIfrtReshardFunctionAttrName << "`";
  }
  return success();
}

}  // namespace impl
}  // namespace ifrt
}  // namespace xla
}  // namespace OpTrait
}  // namespace mlir
