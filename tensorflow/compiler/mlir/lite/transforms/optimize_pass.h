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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_PASS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/optimize_pass_options.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass.h"

namespace mlir {
namespace TFL {

// Optimize TFLite operations in functions.
class OptimizePass
    : public Pass<OptimizePass, OptimizePassOptions, func::FuncOp> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizePass)

  OptimizePass() = default;
  OptimizePass(const OptimizePass &) {}
  explicit OptimizePass(const mlir::detail::PassOptions &options)
      : Pass<OptimizePass, OptimizePassOptions, func::FuncOp>(options) {}

  /// Returns the command-line argument attached to this pass.
  static llvm::StringRef GetArgument() { return "tfl-optimize"; }

  static llvm::StringRef GetDescription() {
    return "Optimize within the TensorFlow Lite dialect";
  }

  /// Returns the derived pass name.
  static llvm::StringRef GetName() { return "OptimizePass"; }

  void runOnOperation() override;
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_PASS_H_
