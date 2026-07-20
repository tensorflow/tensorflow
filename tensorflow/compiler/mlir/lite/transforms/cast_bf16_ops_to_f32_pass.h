/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_CAST_BF16_OPS_TO_F32_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_CAST_BF16_OPS_TO_F32_PASS_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_options.h"

namespace mlir {
namespace TFL {

// Pass to cast bfloat16 operations to float32 in TFLite dialect.
// Currently, this pass eliminates BF16 operations even if the ops natively
// support BF16, with the intention of having all operations run in FP32.
// If needed, an option can be added later to allow ops that support BF16
// to directly consume BF16 without casting.
class CastBf16OpsToF32Pass
    : public TFL::Pass<CastBf16OpsToF32Pass, EmptyPassOptions, func::FuncOp> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CastBf16OpsToF32Pass)

  CastBf16OpsToF32Pass() = default;
  CastBf16OpsToF32Pass(const CastBf16OpsToF32Pass&) {};

  void runOnOperation() override;
  static llvm::StringRef GetName() { return "CastBf16OpsToF32Pass"; }
  static llvm::StringRef GetArgument() { return "tfl-cast-bf16-ops-to-f32"; }
  static llvm::StringRef GetDescription() {
    return "Cast BF16 operations to F32 in TFLite dialect.";
  }

 private:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<func::FuncDialect, mlir::TFL::TensorFlowLiteDialect>();
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> CreateCastBf16OpsToF32Pass();

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_CAST_BF16_OPS_TO_F32_PASS_H_
