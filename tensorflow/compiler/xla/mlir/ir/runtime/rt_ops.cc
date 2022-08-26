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

#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_ops.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/xla/runtime/constraints.h"

//===----------------------------------------------------------------------===//
// RT Dialect
//===----------------------------------------------------------------------===//

#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_dialect.cpp.inc"

namespace xla {
namespace runtime {

using llvm::Expected;

static bool IsRtConstraintAttr(mlir::Attribute attr) {
  // If attribute is not defined it means that there is no constraint
  if (!attr) return true;
  auto str = attr.dyn_cast_or_null<mlir::StringAttr>();
  Expected<ArgumentConstraint> constraint =
      ParseArgumentConstraint(str.getValue());
  return !constraint.takeError();
}

void RuntimeDialect::initialize() {
  allowUnknownTypes();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/xla/mlir/ir/runtime//rt_ops.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_types.cpp.inc"
      >();
}

mlir::LogicalResult RuntimeDialect::verifyOperationAttribute(
    mlir::Operation *op, mlir::NamedAttribute attribute) {
  if (attribute.getName() == "rt.entrypoint") {
    if (!(attribute.getValue().isa<mlir::UnitAttr>())) {
      return op->emitOpError()
             << "requires " << attribute.getName() << " to be a unit attribute";
    }

    auto func = llvm::dyn_cast<mlir::func::FuncOp>(op);
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

  if (attribute.getName() == "rt.custom_call" ||
      attribute.getName() == "rt.direct_custom_call") {
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

#define GET_OP_CLASSES
#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_ops.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_types.cpp.inc"
