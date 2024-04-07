/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/config.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/pre_calibration.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir::quant::stablehlo::testing {

#define GEN_PASS_DEF_TESTPRECALIBRATIONCOMPONENTPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/testing/passes.h.inc"

namespace {

using ::stablehlo::quantization::ExpandPresets;
using ::stablehlo::quantization::PopulateDefaults;
using ::stablehlo::quantization::QuantizationConfig;

class TestPreCalibrationComponentPass
    : public impl::TestPreCalibrationComponentPassBase<
          TestPreCalibrationComponentPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPreCalibrationComponentPass)

 private:
  void runOnOperation() override;
};

void TestPreCalibrationComponentPass::runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext& ctx = getContext();

  // Simply runs the PreCalibrationComponent with a default configuration.
  PreCalibrationComponent component(&ctx);
  QuantizationConfig quantization_config{};
  quantization_config.mutable_static_range_ptq_preset();
  quantization_config = ExpandPresets(PopulateDefaults(quantization_config));
  if (!component.Run(module_op, quantization_config).ok()) {
    signalPassFailure();
  }
}

}  // namespace
}  // namespace mlir::quant::stablehlo::testing
