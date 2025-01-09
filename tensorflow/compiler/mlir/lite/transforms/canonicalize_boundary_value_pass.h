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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_CANONICALIZE_BOUNDARY_VALUE_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_CANONICALIZE_BOUNDARY_VALUE_PASS_H_

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/transforms/pass.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

namespace mlir {
namespace TFL {

// Pass to canonicalize the IR representations of boundary values.

class CanonicalizeBoundaryValuePass
    : public TFL::Pass<CanonicalizeBoundaryValuePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeBoundaryValuePass)

  CanonicalizeBoundaryValuePass() = default;
  CanonicalizeBoundaryValuePass(const CanonicalizeBoundaryValuePass&) {};

  void runOnOperation() override;
  static llvm::StringRef GetName() { return "CanonicalizeBoundaryValuePass"; }
  static llvm::StringRef GetArgument() {
    return "tfl-canonicalize-boundary-value";
  }
  static llvm::StringRef GetDescription() {
    return "Pass to canonicalize the IR representations of boundary values";
  }

 private:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect, mlir::stablehlo::StablehloDialect,
                    mlir::arith::ArithDialect>();
  }
};
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_CANONICALIZE_BOUNDARY_VALUE_PASS_H_
