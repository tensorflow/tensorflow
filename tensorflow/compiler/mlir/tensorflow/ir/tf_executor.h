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

// This file defines the tf_executor dialect: it models the TensorFlow executor
// semantics and can represent arbitrary TensorFlow graphs. As such it follows
// the existing execution model that includes deadness propagation, concurrent
// semantics, and control dependencies.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_EXECUTOR_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_EXECUTOR_H_

#include "mlir/Dialect/Traits.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Dialect.h"  // TF:llvm-project
#include "mlir/IR/Matchers.h"  // TF:llvm-project
#include "mlir/IR/OpImplementation.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace tf_executor {

class TensorFlowExecutorDialect : public Dialect {
 public:
  explicit TensorFlowExecutorDialect(MLIRContext *context);

  // Parses a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;
};

namespace TFTypes {
enum Kind {
  Control = Type::FIRST_TENSORFLOW_EXECUTOR_TYPE,
  Token,
};
}  // namespace TFTypes

// The Control type is a token-like value that models control dependencies from
// TensorFlow graphs.
class ControlType : public Type::TypeBase<ControlType, Type> {
 public:
  using Base::Base;

  static ControlType get(MLIRContext *context) {
    return Base::get(context, TFTypes::Control);
  }

  // Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == TFTypes::Control; }
};

class TokenType : public Type::TypeBase<TokenType, Type> {
 public:
  using Base::Base;

  static TokenType get(MLIRContext *context) {
    return Base::get(context, TFTypes::Token);
  }

  // Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == TFTypes::Token; }
};

// Declares the operations for this dialect using the generated header.
#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h.inc"

}  // namespace tf_executor
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_EXECUTOR_H_
