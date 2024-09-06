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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_H_

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

struct OptimizePassOptions {
  bool enable_canonicalization = true;
  bool disable_fuse_mul_and_fc = false;
};

class OptimizePass
    : public mlir::PassWrapper<OptimizePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizePass)

  OptimizePass() = default;
  OptimizePass(const OptimizePass &) {}
  explicit OptimizePass(bool enable_canonicalization,
                        bool disable_fuse_mul_and_fc = false) {
    this->enable_canonicalization_ = enable_canonicalization;
    this->disable_fuse_mul_and_fc_ = disable_fuse_mul_and_fc;
  }

  explicit OptimizePass(const OptimizePassOptions &options) {
    this->enable_canonicalization_ = options.enable_canonicalization;
    this->disable_fuse_mul_and_fc_ = options.disable_fuse_mul_and_fc;
  }

  void runOnOperation() final;

  /// Returns the command-line argument attached to this pass.
  static constexpr llvm::StringLiteral getArgumentName() {
    return llvm::StringLiteral("tfl-optimize");
  }
  llvm::StringRef getArgument() const final { return "tfl-optimize"; }

  llvm::StringRef getDescription() const final {
    return "Optimize within the TensorFlow Lite dialect";
  }

  /// Returns the derived pass name.
  static constexpr llvm::StringLiteral getPassName() {
    return llvm::StringLiteral("OptimizePass");
  }
  llvm::StringRef getName() const final { return "OptimizePass"; }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }

 private:
  mlir::Pass::Option<bool> enable_canonicalization_{
      *this, "enable-canonicalization",
      llvm::cl::desc("Enable canonicalization during optimization pass."),
      llvm::cl::init(true)};
  mlir::Pass::Option<bool> disable_fuse_mul_and_fc_{
      *this, "disable-fuse-mul-and-fc",
      llvm::cl::desc("Disable folding mul and fully connected ops during "
                     "optimization pass."),
      llvm::cl::init(false)};
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_H_
