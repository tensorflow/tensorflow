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

// This file defines the operations used in the disc_ral dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_DISC_RAL_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_DISC_RAL_OPS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class OpBuilder;

namespace disc_ral {

class RalDialect : public Dialect {
 public:
  explicit RalDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "disc_ral"; }

  // Parses a type registered to this dialect.
  Type parseType(DialectAsmParser& parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter& os) const override;
};

class RalExecutionContextType
    : public Type::TypeBase<RalExecutionContextType, Type, TypeStorage> {
 public:
  using Base::Base;
};

}  // end namespace disc_ral
}  // end namespace mlir

#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.h.inc"

#endif  //  TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_DISC_RAL_OPS_H_
