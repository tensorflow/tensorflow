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

// This transformation pass applies quantization on TFLite dialect.

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Matchers.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Support/Functional.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Quantize Pass.
//
namespace {

// Full integer quantization rewrite pattern for TFLite.
struct TFLFullQuantization
    : public QuantizationPattern<TFLFullQuantization, QuantizeOp,
                                 DequantizeOp> {
  explicit TFLFullQuantization(MLIRContext* ctx) : BaseType(ctx) {}
  static bool AllowHybridOperand() { return false; }
  static bool AllowHybridResult() { return false; }
};

// Applies quantization on the model in TFL dialect.
struct QuantizePass : public FunctionPass<QuantizePass> {
  void runOnFunction() override;
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_quantize.inc"

void QuantizePass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();
  auto* ctx = func.getContext();
  TFL::populateWithGenerated(ctx, &patterns);
  patterns.insert<TFLFullQuantization>(ctx);
  applyPatternsGreedily(func, patterns);
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect QuantizeTFL pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateQuantizePass() {
  return std::make_unique<QuantizePass>();
}

static PassRegistration<QuantizePass> pass(
    "tfl-quantize", "Apply quantization on models in TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
