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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TFLITE_PASSES_SPLIT_MERGED_OPERANDS_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TFLITE_PASSES_SPLIT_MERGED_OPERANDS_PASS_H_

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

namespace mlir {
namespace TFL {

// Background info:
// Currently the model taken to MLIRConverter is frozen (all the variables have
// been converted to constants, all the assign ops are gone, etc.). However,
// TFLite has these variable tensors semantics. So the variable mapping from TF
// to TFLite is actually broken here, we sort of hard-code the variable tensors
// based on the actual ops using them, such as unidirectional_sequence_lstm.
//
// MLIRConverter also benefits from lots of typical compiler optimization like
// merging same input values if they're identical. These optimizations are
// desirable but not for those TFLite ops which have variable tensors as inputs.
// Yes, they have identical input values, but those identical values are
// "stateful", their values can change during invocations.
//
// A typical example is unidirectional_sequence_lstm have two variable tensor
// inputs: activation state & cell state. They may have same initial values
// (typical zero-initialized), but their values will be changed. So we cannot
// just merge those values.
//
// This pass is more like short-term workaround since we don't have a good
// variable representation right now.
//
// This pass will duplicate input values for those variable tensor inputs.

class SplitMergedOperandsPass
    : public TFL::Pass<SplitMergedOperandsPass, EmptyPassOptions,
                       func::FuncOp> {
 public:
  SplitMergedOperandsPass() = default;
  SplitMergedOperandsPass(const SplitMergedOperandsPass &other) {}

  void runOnOperation() final;

  /// Returns the command-line argument attached to this pass.
  static llvm::StringRef GetArgument() { return "tfl-split-merged-operands"; }

  static llvm::StringRef GetDescription() {
    return "Split merged stateful operands for tfl operations.";
  }

  /// Returns the derived pass name.
  static llvm::StringRef GetName() { return "SplitMergedOperandsPass"; }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit
  /// private instantiation because Pass classes should only be visible by the
  /// current library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitMergedOperandsPass)
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TFLITE_PASSES_SPLIT_MERGED_OPERANDS_PASS_H_
