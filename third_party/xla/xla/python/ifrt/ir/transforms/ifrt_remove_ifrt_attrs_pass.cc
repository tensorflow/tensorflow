/* Copyright 2024 The OpenXLA Authors.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla {
namespace ifrt {

namespace {

#define GEN_PASS_DEF_IFRTREMOVEIFRTATTRSPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

class IfrtRemoveIfrtAttrsPass
    : public impl::IfrtRemoveIfrtAttrsPassBase<IfrtRemoveIfrtAttrsPass> {
 public:
  void runOnOperation() override;
};

void IfrtRemoveIfrtAttrsPass::runOnOperation() {
  mlir::ModuleOp module_op = getOperation();
  module_op->removeAttr(kIfrtNumDevicesAttrName);
  module_op->removeAttr(kIfrtLocalViewAttrName);
  module_op.walk([&](mlir::func::FuncOp func_op) {
    // Remove from function attributes.
    for (auto attribute_name : {kIfrtDevicesAttrName, kIfrtMemoryKindAttrName,
                                kIfrtShardingAttrName}) {
      func_op->removeAttr(attribute_name);
    }

    // Remove from argument attributes.
    for (int i = 0; i < func_op.getNumArguments(); ++i) {
      mlir::NamedAttrList arg_attrs = func_op.getArgAttrDict(i);
      for (auto attribute_name : {kIfrtDevicesAttrName, kIfrtMemoryKindAttrName,
                                  kIfrtShardingAttrName}) {
        arg_attrs.erase(attribute_name);
      }
      func_op.setArgAttrs(i, arg_attrs);
    }
    // Remove from result attributes.
    for (int i = 0; i < func_op.getNumResults(); ++i) {
      mlir::NamedAttrList res_attrs = func_op.getResultAttrDict(i);
      for (auto attribute_name : {kIfrtDevicesAttrName, kIfrtMemoryKindAttrName,
                                  kIfrtShardingAttrName}) {
        res_attrs.erase(attribute_name);
      }
      func_op.setResultAttrs(i, res_attrs);
    }
  });
}
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtRemoveIfrtAttrsPass() {
  return std::make_unique<IfrtRemoveIfrtAttrsPass>();
}

}  // namespace ifrt
}  // namespace xla
