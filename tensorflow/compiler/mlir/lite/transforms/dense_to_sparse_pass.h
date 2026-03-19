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

// This transformation pass convert dense tensor to sparse format.
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_DENSE_TO_SPARSE_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_DENSE_TO_SPARSE_PASS_H_

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

// This pass encodes sparse weights in the model in the proper format, and adds
// Densify() op if necessary. The general algorithm is:
//   1. Get list of operands (weights) of an op that can be sparse.
//   2. Get list of supported block configurations of the op.
//   3. Calculate random sparsity of the weight.
//     3.1. If sparsity level is below the encoding threshold, keep in dense.
//     3.2. If sparsity level is above the encoding threshold, go to 4.
//   4. Try to encode the weight with supported block configurations. If the
//      weight was pruned with the same block config, the blocked sparsity level
//      should match the random sparsity.
//     4.1. Return the matching block config if found.
//     4.2. If no matching block config is found, encode the weight with random
//          sparsity, and add Densify() op to fall back to dense execution.

class DenseToSparsePass
    : public Pass<DenseToSparsePass, EmptyPassOptions, func::FuncOp> {
 public:
  DenseToSparsePass() = default;
  DenseToSparsePass(const DenseToSparsePass &other) {}

  void runOnOperation() final;

  /// Returns the command-line argument attached to this pass.
  static llvm::StringRef GetArgument() { return "tfl-dense-to-sparse"; }

  static llvm::StringRef GetDescription() {
    return "Convert dense tensor to sparse format.";
  }

  /// Returns the derived pass name.
  static llvm::StringRef GetName() { return "DenseToSparsePass"; }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit
  /// private instantiation because Pass classes should only be visible by the
  /// current library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DenseToSparsePass)
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_DENSE_TO_SPARSE_PASS_H_
