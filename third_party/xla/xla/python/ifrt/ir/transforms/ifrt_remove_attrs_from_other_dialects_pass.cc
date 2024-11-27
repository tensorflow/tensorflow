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

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla {
namespace ifrt {

namespace {

#define GEN_PASS_DEF_IFRTREMOVEATTRSFROMOTHERDIALECTSPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

class IfrtRemoveAttrsFromOtherDialectsPass
    : public impl::IfrtRemoveAttrsFromOtherDialectsPassBase<
          IfrtRemoveAttrsFromOtherDialectsPass> {
 public:
  void runOnOperation() override;
};

// Returns true if the given `NamedAttribute` is from the IFRT or builtin
// dialect.
bool isBuiltinOrIfrtAttr(mlir::NamedAttribute attr) {
  if (!attr.getNameDialect()) {
    return true;
  }
  auto dialect_namespace = attr.getNameDialect()->getNamespace();
  return dialect_namespace == mlir::BuiltinDialect::getDialectNamespace() ||
         dialect_namespace == IfrtDialect::getDialectNamespace();
}

bool isBuiltinOrIfrtAttr(mlir::Attribute attr) {
  auto dialect_namespace = attr.getDialect().getNamespace();
  return dialect_namespace == mlir::BuiltinDialect::getDialectNamespace() ||
         dialect_namespace == IfrtDialect::getDialectNamespace();
}

// Returns true if the given `Operation` is an IFRT op, or if it is a FuncOp or
// ReturnOp of an IFRT function.
bool isIfrtOpOrFunc(mlir::Operation* op) {
  if (op->getDialect()->getNamespace() == IfrtDialect::getDialectNamespace()) {
    return true;
  }
  if (auto func_op = llvm::dyn_cast_or_null<mlir::func::FuncOp>(op)) {
    return op->hasAttr(kIfrtFunctionAttrName);
  } else if (auto return_op =
                 llvm::dyn_cast_or_null<mlir::func::ReturnOp>(op)) {
    return return_op->getParentOp()->hasAttr(kIfrtFunctionAttrName);
  }
  return false;
}

// Recursively removes attributes from the given `Attribute` that are not from
// the IFRT or builtin dialect.
mlir::FailureOr<mlir::Attribute> removeAttrsFromOtherDialects(
    mlir::Attribute attr) {
  // Remove invalid attributes from container attributes.
  if (auto array_attr = llvm::dyn_cast<mlir::ArrayAttr>(attr)) {
    llvm::SmallVector<mlir::Attribute> elements;
    for (auto element : array_attr.getValue()) {
      if (auto converted_element_or = removeAttrsFromOtherDialects(element);
          mlir::succeeded(converted_element_or)) {
        elements.push_back(*converted_element_or);
      }
    }
    auto res = mlir::ArrayAttr::get(attr.getContext(), elements);
    return res;
  } else if (auto dict_attr = llvm::dyn_cast<mlir::DictionaryAttr>(attr)) {
    llvm::SmallVector<mlir::NamedAttribute> kept_attrs;
    for (auto named_attr : dict_attr.getValue()) {
      if (isBuiltinOrIfrtAttr(named_attr)) {
        if (auto new_attr_or =
                removeAttrsFromOtherDialects(named_attr.getValue());
            mlir::succeeded(new_attr_or)) {
          kept_attrs.push_back(
              mlir::NamedAttribute(named_attr.getName(), *new_attr_or));
        }
      }
    }
    auto res = mlir::DictionaryAttr::get(attr.getContext(), kept_attrs);
    return res;
  }
  if (isBuiltinOrIfrtAttr(attr)) {
    return attr;
  }
  return mlir::failure();
}

mlir::LogicalResult removeAttrsFromOtherDialects(mlir::Operation* op) {
  auto attr_dict_or = removeAttrsFromOtherDialects(op->getAttrDictionary());
  if (mlir::succeeded(attr_dict_or)) {
    if (auto attr_dict =
            llvm::dyn_cast_or_null<mlir::DictionaryAttr>(*attr_dict_or)) {
      op->setAttrs(attr_dict);
    } else {
      op->emitOpError() << "Failed to remove attrs from other dialects. Remove "
                           "returned a non-dictionary attribute";
      return mlir::failure();
    }
  } else {
    op->setAttrs(mlir::DictionaryAttr::get(op->getContext(), {}));
  }
  return mlir::success();
}

void IfrtRemoveAttrsFromOtherDialectsPass::runOnOperation() {
  mlir::ModuleOp module_op = getOperation();
  if (mlir::failed(removeAttrsFromOtherDialects(module_op))) {
    signalPassFailure();
    return;
  }
  auto result = module_op.walk([&](mlir::Operation* op) {
    if (isIfrtOpOrFunc(op) && mlir::failed(removeAttrsFromOtherDialects(op))) {
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtRemoveAttrsFromOtherDialectsPass() {
  return std::make_unique<IfrtRemoveAttrsFromOtherDialectsPass>();
}

}  // namespace ifrt
}  // namespace xla
