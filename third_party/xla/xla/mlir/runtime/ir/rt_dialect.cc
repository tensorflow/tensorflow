/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/mlir/runtime/ir/rt_dialect.h"

#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project  // IWYU pragma: keep
#include "xla/mlir/runtime/ir/rt_interfaces.h"
#include "xla/mlir/runtime/ir/rt_ops.h"
#include "xla/runtime/constraints.h"

//===----------------------------------------------------------------------===//
// RT Dialect
//===----------------------------------------------------------------------===//

#include "xla/mlir/runtime/ir/rt_dialect.cc.inc"

namespace xla {
namespace runtime {

static bool IsRtConstraintAttr(mlir::Attribute attr) {
  // If attribute is not defined it means that there is no constraint
  if (!attr) return true;
  auto str = attr.dyn_cast_or_null<mlir::StringAttr>();
  absl::StatusOr<ArgumentConstraint> constraint =
      ParseArgumentConstraint(str.getValue());
  return constraint.ok();
}

void RuntimeDialect::initialize() {
  allowUnknownTypes();

  addOperations<
#define GET_OP_LIST
#include "xla/mlir/runtime/ir/rt_ops.cc.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "xla/mlir/runtime/ir/rt_types.cc.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "xla/mlir/runtime/ir/rt_attrs.cc.inc"
      >();
}

mlir::LogicalResult RuntimeDialect::verifyOperationAttribute(
    mlir::Operation *op, mlir::NamedAttribute attribute) {
  // Only functions can be marked as exported.
  if (attribute.getName() == "rt.exported") {
    if (!llvm::isa<mlir::IntegerAttr>(attribute.getValue())) {
      return op->emitOpError() << "requires " << attribute.getName()
                               << " to be an integer attribute";
    }

    auto func = llvm::dyn_cast<mlir::FunctionOpInterface>(op);
    if (!func) {
      return op->emitError()
             << attribute.getName() << " can only be applied to a function";
    }
    if (func.empty()) {
      return op->emitOpError()
             << "requires non-empty body for function with attribute "
             << attribute.getName();
    }
  }

  // Custom call attribute can be defined only on a function declaration.
  if (attribute.getName() == "rt.custom_call") {
    if (!(attribute.getValue().isa<mlir::StringAttr>())) {
      return op->emitOpError() << "requires " << attribute.getName()
                               << " to only accept string value";
    }

    auto func = llvm::dyn_cast<mlir::func::FuncOp>(op);
    if (!func) {
      return op->emitError()
             << attribute.getName() << " can only be applied to a function";
    }
    if (!func.empty()) {
      return op->emitOpError() << "requires " << attribute.getName()
                               << " to only apply to a function declaration";
    }
  }

  // Dynamic custom call attribute can be applied only to a custom call
  // declaration.
  if (attribute.getName() == "rt.dynamic") {
    if (!op->hasAttr("rt.custom_call")) {
      return op->emitOpError()
             << attribute.getName()
             << " can only be applied to a custom call declaration";
    }
  }

  // Trace annotation should implement an attribute interface.
  if (attribute.getName() == "rt.trace") {
    if (!attribute.getValue().isa<TraceAnnotationAttrInterface>()) {
      return op->emitOpError() << " requires " << attribute.getName()
                               << " to be a trace annotation attribute";
    }
  }

  // Check constraints for all function arguments.
  if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
    for (int i = 0; i < func.getNumArguments(); ++i) {
      if (!IsRtConstraintAttr(
              func.getArgAttr(i, kArgumentConstraintAttrName))) {
        return op->emitOpError()
               << "has illegal attribute value of "
               << kArgumentConstraintAttrName << " for argument " << i;
      }
    }
  }

  return mlir::success();
}
}  // namespace runtime
}  // namespace xla

#define GET_TYPEDEF_CLASSES
#include "xla/mlir/runtime/ir/rt_types.cc.inc"

#define GET_ATTRDEF_CLASSES
#include "xla/mlir/runtime/ir/rt_attrs.cc.inc"
