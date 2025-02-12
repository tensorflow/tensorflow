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

// This pass removes the device attribute from every corert.executeop.

#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tfrt/core_runtime/opdefs/core_runtime.h"  // from @tf_runtime

namespace tensorflow {

namespace {

constexpr const char* kDevice = "device";

struct RemoveDeviceAttributePass
    : public PassWrapper<RemoveDeviceAttributePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveDeviceAttributePass)

  llvm::StringRef getArgument() const final {
    return "tfrt-remove-device-attribute";
  }
  llvm::StringRef getDescription() const final {
    return "This pass removes the device attribute from every corert.executeop";
  }

  void runOnOperation() override;
};

void RemoveDeviceAttributePass::runOnOperation() {
  ModuleOp module = getOperation();

  module.walk([&](tfrt::corert::ExecuteOp execute_op) {
    SmallVector<std::pair<StringRef, Attribute>, 4> op_func_attrs;
    SmallVector<std::pair<StringRef, Attribute>, 4> op_attrs;
    SmallVector<std::pair<StringRef, Attribute>, 4> new_op_attrs;
    execute_op.getOpFuncAttrs(&op_func_attrs);
    execute_op.getOpAttrs(&op_attrs);
    for (std::pair<StringRef, Attribute> attr : op_attrs) {
      if (attr.first != kDevice) {
        new_op_attrs.push_back(attr);
      }
    }
    if (op_attrs.size() == new_op_attrs.size()) return WalkResult::advance();

    OpBuilder builder(execute_op);
    auto new_execute_op = builder.create<tfrt::corert::ExecuteOp>(
        execute_op.getLoc(), execute_op.getResultTypes(),
        execute_op.getOpHandler(), execute_op.getArguments(), new_op_attrs,
        op_func_attrs, execute_op.getOpName());
    execute_op.replaceAllUsesWith(new_execute_op.getResults());
    execute_op.erase();
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateRemoveDeviceAttributePass() {
  return std::make_unique<RemoveDeviceAttributePass>();
}

static PassRegistration<RemoveDeviceAttributePass> pass;

}  // namespace tensorflow
