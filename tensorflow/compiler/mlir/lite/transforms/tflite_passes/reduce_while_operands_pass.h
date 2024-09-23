/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TFLITE_PASSES_REDUCE_WHILE_OPERANDS_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TFLITE_PASSES_REDUCE_WHILE_OPERANDS_PASS_H_

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_options.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

namespace mlir {
namespace TFL {

// This is a pass to reduce operands without changing the outcome.

class ReduceWhileOperandsPass
    : public Pass<ReduceWhileOperandsPass, EmptyPassOptions, func::FuncOp> {
 public:
  ReduceWhileOperandsPass() = default;
  ReduceWhileOperandsPass(const ReduceWhileOperandsPass &other) {}

  void runOnOperation() final;

  /// Returns the command-line argument attached to this pass.
  static llvm::StringRef GetArgument() { return "tfl-reduce-while"; }

  static llvm::StringRef GetDescription() {
    return "Reduce the number of operands and results of a whlieOp.";
  }

  /// Returns the derived pass name.
  static llvm::StringRef GetName() { return "ReduceWhileOperandsPass"; }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<TFL::TensorFlowLiteDialect, TF::TensorFlowDialect>();
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit
  /// private instantiation because Pass classes should only be visible by the
  /// current library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReduceWhileOperandsPass)
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TFLITE_PASSES_REDUCE_WHILE_OPERANDS_PASS_H_
