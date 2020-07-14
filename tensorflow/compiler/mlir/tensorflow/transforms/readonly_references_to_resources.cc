/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {
namespace {

// Location attribute.
constexpr StringRef kClassAttr = "_class";
constexpr StringRef kLocationPrefix = "loc:@";

// A pass that converts readonly reference variables to the corresponding
// resource variables.
//
// It converts (VariableV2 -> Identity) to (VarHandle -> ReadVariable).
//
// For the background, this pass is a part of hoisting VariableV2 ops by
// re-using the pipeline for hoisting (VarHandle -> ReadVariable) cases, which
//  can be done by the following passes:
//  - Capturing resource values into global tensors (importing saved model).
//  - Promoting VarHandle ops to function input/outputs.
//  - Freezing global tensor pass.
//
// This path assumes that all the VariableV2 ops is read-only via verifying the
// heuristic method that assumes that all the users of them is Identity op,
// fed directly.
class ConvertReadonlyReferenceVariablesToResourceVariablesPass
    : public PassWrapper<
          ConvertReadonlyReferenceVariablesToResourceVariablesPass,
          FunctionPass> {
 public:
  void runOnFunction() override;
};

// Parse node name from "_class" attribute.
StringRef GetNodeNameFromClassAttr(Operation *op) {
  ArrayAttr classes_attr = op->getAttrOfType<ArrayAttr>(kClassAttr);
  if (!classes_attr) {
    op->emitOpError() << "has no '_class' attribute";
    return StringRef();
  }

  StringRef result;
  for (Attribute class_attr : classes_attr) {
    StringRef node_name = class_attr.cast<StringAttr>().getValue();
    if (!node_name.startswith(kLocationPrefix)) {
      continue;
    }
    if (!result.empty()) {
      // Invalid case since there are multiple loc:@ attributes.
      op->emitOpError()
          << "expects only one named location in '_class' attribute, but got "
          << classes_attr;
      return StringRef();
    }
    result = node_name.drop_front(kLocationPrefix.size());
  }
  if (result.empty()) {
    op->emitOpError() << "expects variable name in '_class' attribute, but got "
                      << classes_attr;
  }
  return result;
}

void ConvertReadonlyReferenceVariablesToResourceVariablesPass::runOnFunction() {
  FuncOp func = getFunction();

  OpBuilder builder(func.getContext());
  SmallVector<VariableV2Op, 4> variable_v2s_to_replace;

  // Checks all the VariableV2 ops is read-only via verifying the heuristic
  // method that assumes that all the users of them is Identity op, feeded
  // directly.
  auto read_only_vars_fn = [&variable_v2s_to_replace](
                               VariableV2Op variable_v2_op) {
    if (variable_v2_op.getResult().use_empty()) {
      // Erase the op when there is no user.
      variable_v2_op.erase();
      return mlir::WalkResult::advance();
    }
    if (!all_of(variable_v2_op.getResult().getUsers(), [&variable_v2_op](
                                                           Operation *user) {
          if (!isa<IdentityOp>(user)) {
            variable_v2_op.emitOpError()
                << "expects all users to be 'tf.Identity', but got user "
                << user->getName();
            return false;
          }
          return true;
        })) {
      return mlir::WalkResult::interrupt();
    }
    variable_v2s_to_replace.push_back(variable_v2_op);
    return mlir::WalkResult::advance();
  };

  WalkResult walk_res = func.walk(read_only_vars_fn);
  if (walk_res.wasInterrupted()) return signalPassFailure();

  for (VariableV2Op variable_v2_op : variable_v2s_to_replace) {
    builder.setInsertionPoint(variable_v2_op);
    ShapedType shaped_type =
        variable_v2_op.getResult().getType().cast<ShapedType>();
    TensorType tensor_type = DropRefType(shaped_type).cast<TensorType>();
    StringAttr device_attr = variable_v2_op.getAttrOfType<StringAttr>("device");
    if (!device_attr) device_attr = builder.getStringAttr("");
    StringRef variable_name = GetNodeNameFromClassAttr(variable_v2_op);
    if (variable_name.empty()) {
      return signalPassFailure();
    }
    VarHandleOp var_handle_op = builder.create<VarHandleOp>(
        variable_v2_op.getLoc(),
        ArrayRef<Type>{RankedTensorType::get(
            {}, TF::ResourceType::get(ArrayRef<TensorType>{tensor_type},
                                      builder.getContext()))},
        ArrayRef<Value>{},
        ArrayRef<NamedAttribute>{
            builder.getNamedAttr("device", device_attr),
            builder.getNamedAttr("container", variable_v2_op.containerAttr()),
            builder.getNamedAttr("shared_name",
                                 builder.getStringAttr(variable_name))});
    for (Operation *user :
         make_early_inc_range(variable_v2_op.getResult().getUsers())) {
      builder.setInsertionPoint(user);
      ReadVariableOp read_variable_op = builder.create<ReadVariableOp>(
          user->getLoc(), ArrayRef<Type>{tensor_type},
          ArrayRef<Value>{var_handle_op}, ArrayRef<NamedAttribute>{});
      user->getResult(0).replaceAllUsesWith(read_variable_op.getResult());
      user->erase();
    }
    variable_v2_op.erase();
  }
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateConvertReadonlyReferenceVariablesToResourceVariablesPass() {
  return std::make_unique<
      ConvertReadonlyReferenceVariablesToResourceVariablesPass>();
}

static PassRegistration<
    ConvertReadonlyReferenceVariablesToResourceVariablesPass>
    pass("tf-readonly-references-to-resources",
         "Convert readonly reference variables to resource variables.");

}  // namespace TF

}  // namespace mlir
