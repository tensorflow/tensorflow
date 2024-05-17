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

#ifndef XLA_PYTHON_IFRT_IR_IFRT_INTERFACES_H_
#define XLA_PYTHON_IFRT_IR_IFRT_INTERFACES_H_

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"

namespace mlir {
namespace OpTrait {
namespace xla {
namespace ifrt {

namespace impl {

// Verifies `op` used in a FuncOp with `ifrt.function` attr.
LogicalResult verifyNestedInIfrtFunc(Operation* op);

}  // namespace impl

template <typename ConcreteType>
class NestedInIfrtFuncTrait
    : public TraitBase<ConcreteType, NestedInIfrtFuncTrait> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return impl::verifyNestedInIfrtFunc(op);
  }
};

template <typename CalleeOpType>
class IfrtCallLikeTrait {
 public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, Impl> {
   public:
    // Verifies getCallee() is a valid SymbolRefAttr to CalleeOpType.
    static LogicalResult verifyTrait(Operation* op) {
      mlir::SymbolTableCollection symbol_table;
      ConcreteType concrete = llvm::cast<ConcreteType>(op);
      CalleeOpType callee = concrete.getCalleeOp(symbol_table);
      if (callee == nullptr) {
        return op->emitOpError() << "requires '" << concrete.getCallee()
                                 << "' to reference a valid `"
                                 << CalleeOpType::getOperationName() << "`";
      }
      if (callee->hasAttr(::xla::ifrt::kIfrtFunctionAttrName)) {
        return op->emitOpError() << "requires callee not with attr `"
                                 << ::xla::ifrt::kIfrtFunctionAttrName << "`";
      }
      return success();
    }

    CalleeOpType getCalleeOp(mlir::SymbolTableCollection& symbol_table) {
      SymbolRefAttr callee_attr = static_cast<ConcreteType*>(this)->getCallee();
      return symbol_table.lookupNearestSymbolFrom<CalleeOpType>(
          this->getOperation(), callee_attr);
    }
  };
};

}  // namespace ifrt
}  // namespace xla
}  // namespace OpTrait
}  // namespace mlir

// IWYU pragma: begin_exports

// Generated definitions.
#define GET_ATTR_INTERFACE_CLASSES
#include "xla/python/ifrt/ir/ifrt_attr_interfaces.h.inc"

#define GET_OP_INTERFACE_CLASSES
#include "xla/python/ifrt/ir/ifrt_op_interfaces.h.inc"

// IWYU pragma: end_exports

#endif  // XLA_PYTHON_IFRT_IR_IFRT_INTERFACES_H_
