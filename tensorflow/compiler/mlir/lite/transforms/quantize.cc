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
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_numeric_verify(
    "tfl-numeric-verify", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether verify numericals at runtime."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<float> error_tolerance(
    "tfl-error-tolerance", llvm::cl::value_desc("float"),
    llvm::cl::desc("Error tolerance for numeric verify. Valid when "
                   "`-tfl-numeric-verify` is set."),
    llvm::cl::init(5.0));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_single_layer_verify(
    "tfl-single-layer-verify", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether verify numericals layer by layer. Valid when "
                   "`-tfl-numeric-verify` is set."),
    llvm::cl::init(true));

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Quantize Pass.
//
namespace {

// Full integer quantization rewrite pattern for TFLite.
struct TFLFullQuantization
    : public quant::QuantizationPattern<TFLFullQuantization, QuantizeOp,
                                        DequantizeOp, NumericVerifyOp> {
  explicit TFLFullQuantization(MLIRContext* ctx, bool verify_numeric,
                               float tolerance, bool verify_single_layer)
      : BaseType(ctx, verify_numeric, tolerance, verify_single_layer) {}
  static bool AllowHybridOperand() { return false; }
  static bool AllowHybridResult() { return false; }
};

// Applies quantization on the model in TFL dialect.
struct QuantizePass : public PassWrapper<QuantizePass, FunctionPass> {
  void runOnFunction() override;
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_quantize.inc"

void QuantizePass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();
  auto* ctx = func.getContext();
  TFL::populateWithGenerated(ctx, &patterns);
  patterns.insert<TFLFullQuantization>(
      ctx, enable_numeric_verify, error_tolerance, enable_single_layer_verify);
  applyPatternsAndFoldGreedily(func, patterns);
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect QuantizeTFL pass.
std::unique_ptr<OperationPass<FuncOp>> CreateQuantizePass() {
  return std::make_unique<QuantizePass>();
}

static PassRegistration<QuantizePass> pass(
    "tfl-quantize", "Apply quantization on models in TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
