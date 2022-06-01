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
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TF {
namespace {

// Location attribute.
constexpr StringRef kClassAttr = "_class";
constexpr StringRef kSharedNameAttr = "shared_name";
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
    : public ConvertReadonlyReferenceVariablesToResourceVariablesPassBase<
          ConvertReadonlyReferenceVariablesToResourceVariablesPass> {
  void runOnOperation() override;
};

// Parse node name from "_class" or "shared_name" attributes.
StringRef GetNodeNameFromClassAttrOrSharedNameAttr(Operation *op) {
  // Parse node name from the `shared_name` attribute first. The variable v2 op
  // relies on the share name to look up from the TensorFlow's resource manager.
  StringAttr shared_name_attr = op->getAttrOfType<StringAttr>(kSharedNameAttr);
  if (shared_name_attr) {
    auto shared_name = StringRef(shared_name_attr.getValue());
    if (!shared_name.empty()) {
      return shared_name;
    }
  }
  // Attempt to parse "_class" attribute if there is no "shared_name"
  // attribute.
  ArrayAttr classes_attr = op->getAttrOfType<ArrayAttr>(kClassAttr);
  if (!classes_attr) {
    // Attempt to parse "_class" from the IdentityOp that follows VariableV2.
    // For read-only reference variables, IdentityOp should be the only user of
    // VariableV2.
    auto identity_op = op->getUsers().begin();
    classes_attr = identity_op->getAttrOfType<ArrayAttr>(kClassAttr);
    if (!classes_attr) {
      op->emitOpError() << "has no '_class' and 'shared_name' attributes";
      return StringRef();
    }
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

void ConvertReadonlyReferenceVariablesToResourceVariablesPass::
    runOnOperation() {
  func::FuncOp func = getOperation();

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
    StringAttr device_attr =
        variable_v2_op->getAttrOfType<StringAttr>("device");
    if (!device_attr) device_attr = builder.getStringAttr("");
    StringRef variable_name =
        GetNodeNameFromClassAttrOrSharedNameAttr(variable_v2_op);
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
          ArrayRef<Value>{var_handle_op});
      user->getResult(0).replaceAllUsesWith(read_variable_op.getResult());
      user->erase();
    }
    variable_v2_op.erase();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertReadonlyReferenceVariablesToResourceVariablesPass() {
  return std::make_unique<
      ConvertReadonlyReferenceVariablesToResourceVariablesPass>();
}

}  // namespace TF

}  // namespace mlir
