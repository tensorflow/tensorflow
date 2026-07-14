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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PUSH_TRANSPOSE_THROUGH_EWISE_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PUSH_TRANSPOSE_THROUGH_EWISE_PASS_H_

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass.h"

namespace mlir {
namespace TFL {

class PushTransposeThroughEwisePass
    : public Pass<PushTransposeThroughEwisePass> {
 public:
  PushTransposeThroughEwisePass() = default;
  PushTransposeThroughEwisePass(const PushTransposeThroughEwisePass &other) {}

  void runOnOperation() final;

  /// Returns the command-line argument attached to this pass.
  static llvm::StringRef GetArgument() {
    return "tfl-push-transpose-through-ewise";
  }

  static llvm::StringRef GetDescription() {
    return "Push transpose ops through element-wise ops.";
  }

  /// Returns the derived pass name.
  static llvm::StringRef GetName() { return "PushTransposeThroughEwisePass"; }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit
  /// private instantiation because Pass classes should only be visible by the
  /// current library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PushTransposeThroughEwisePass)
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PUSH_TRANSPOSE_THROUGH_EWISE_PASS_H_
