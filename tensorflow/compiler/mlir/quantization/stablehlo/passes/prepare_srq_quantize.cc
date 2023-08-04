/* Copyright 2023 The StableHLO Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/fill_quantization_options.h"

//===----------------------------------------------------------------------===//
// The prepare-srq-quantize Pass.
//===----------------------------------------------------------------------===//
namespace mlir {
namespace stablehlo {

namespace {
#define GEN_PASS_DEF_PREPARESRQQUANTIZEPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

using QuantizationUnits = llvm::SetVector<std::pair<Operation*, int>>;
using ::stablehlo::quantization::QuantizationOptions;

class PrepareSrqQuantizePass
    : public impl::PrepareSrqQuantizePassBase<PrepareSrqQuantizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareSrqQuantizePass)

  // Constructor used by the PassRegistration and enforce post training int8
  // static range quantization. This is only used by test.
  explicit PrepareSrqQuantizePass() {
    QuantizationOptions preset_srq;
    preset_srq.mutable_quantization_method()
        ->mutable_preset_quantization_method()
        ->set_preset_method(
            ::stablehlo::quantization::PresetQuantizationMethod::
                POST_TRAINING_QUANTIZATION_STATIC_RANGE);

    quantization_options_ = FillPresetQuantizationOptions(preset_srq);
  }

  explicit PrepareSrqQuantizePass(QuantizationOptions quantization_options)
      : quantization_options_(quantization_options) {}

 private:
  void runOnOperation() override;
  QuantizationOptions quantization_options_;
};

using ReplaceStatsWithQDQs =
    quant::ConvertStatsToQDQs<quantfork::QuantizeCastOp,
                              quantfork::DequantizeCastOp>;

#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/prepare_srq_quantize.inc"

void PrepareSrqQuantizePass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();
  RewritePatternSet patterns(ctx);

  populateWithGenerated(patterns);

  // TODO: b/288046643 - Implement different activation bit width per op/op
  // instance.
  int bit_width;
  if (stablehlo::GetActivationBitWidth(quantization_options_, &bit_width)
          .succeeded()) {
    // Convert quant stats to quantization parameters.
    // Only activation stats are imported during calibration, so narrow_range =
    // false.
    patterns.add<ReplaceStatsWithQDQs>(bit_width, /*narrow_range*/ false,
                                       /*is_signed*/ true,
                                       /*legacy_float_scale=*/false, ctx);
  }

  FrozenRewritePatternSet frozen_patterns(std::move(patterns));
  if (failed(applyPatternsAndFoldGreedily(func, frozen_patterns))) {
    signalPassFailure();
  }
}

static PassRegistration<PrepareSrqQuantizePass> pass;

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareSrqQuantizePass(
    QuantizationOptions quantization_options) {
  return std::make_unique<PrepareSrqQuantizePass>(quantization_options);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareSrqQuantizePass() {
  return std::make_unique<PrepareSrqQuantizePass>();
}
}  // namespace stablehlo
}  // namespace mlir
