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
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantize_passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/fill_quantization_options.h"

namespace stablehlo {
namespace quantization {

void AddQuantizationPasses(mlir::PassManager& pass_manager,
                           const QuantizationOptions& quantization_options) {
  QuantizationOptions quantization_options_ = quantization_options;
  if (quantization_options.quantization_method()
          .has_preset_quantization_method()) {
    quantization_options_ =
        mlir::quant::stablehlo::FillPresetQuantizationOptions(
            quantization_options);
  }

  // TODO: b/276999414 - Add activation and bias quantization component as
  // respective quantization passes are created.
  QuantizationComponentSpec weight_component;
  for (const auto& component : quantization_options_.quantization_method()
                                   .custom_quantization_method()
                                   .quantization_component_spec()) {
    switch (component.quantization_component()) {
      case QuantizationComponentSpec::COMPONENT_WEIGHT:
        weight_component = component;
        break;
      default:
        break;
    }
  }
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::stablehlo::CreateQuantizeWeightPass(weight_component));
}

}  // namespace quantization
}  // namespace stablehlo
