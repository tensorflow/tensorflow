/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements the named operations for the "Control Flow" dialect of
// TensorFlow graphs

#include "tensorflow/compiler/mlir/tensorflow/ir/control_flow_ops.h"

#include "mlir/IR/DialectImplementation.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/OpImplementation.h"  // TF:local_config_mlir

namespace mlir {
namespace TFControlFlow {

// TODO(ycao): Implement following verify methods when we know more about their
// invariant.
LogicalResult EnterOp::verify() { return success(); }

LogicalResult MergeOp::verify() { return success(); }

LogicalResult NextIterationSourceOp::verify() { return success(); }

LogicalResult NextIterationSinkOp::verify() { return success(); }

LogicalResult LoopCondOp::verify() { return success(); }

LogicalResult SwitchOp::verify() { return success(); }

LogicalResult ExitOp::verify() { return success(); }

TFControlFlowDialect::TFControlFlowDialect(MLIRContext *context)
    : Dialect(/*name=*/"_tf", context) {
  addOperations<SwitchOp, MergeOp, EnterOp, NextIterationSourceOp,
                NextIterationSinkOp, ExitOp, LoopCondOp>();
  addTypes<TFControlType>();

  // We allow unregistered TensorFlow operations in the control dialect.
  allowUnknownOperations();
}

// Parses a type registered to this dialect.
Type TFControlFlowDialect::parseType(DialectAsmParser &parser) const {
  if (parser.parseKeyword("control", ": unknown TFControl type")) return Type();

  return TFControlType::get(getContext());
}

// Prints a type registered to this dialect.
void TFControlFlowDialect::printType(Type type, DialectAsmPrinter &os) const {
  assert(type.isa<TFControlType>());
  os << "control";
}

}  // namespace TFControlFlow
}  // namespace mlir
