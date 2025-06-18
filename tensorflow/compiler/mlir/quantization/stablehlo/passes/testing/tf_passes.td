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

// Passes only used for testing purposes.

include "mlir/Pass/PassBase.td"

def TestPreCalibrationComponentPass : Pass<"tf-stablehlo-test-pre-calibration-component", "mlir::ModuleOp"> {
  let summary = "Test-only pass to test the PreCalibrationComponent.";
  let description = [{
    Runs the pre calibration passes for post-training quantization with default
    configuration.
  }];
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect", "mlir::TF::TensorFlowDialect",
    "mlir::func::FuncDialect", "mlir::tf_executor::TensorFlowExecutorDialect",
    "mlir::mhlo::MhloDialect", "mlir::vhlo::VhloDialect",
  ];
}

def TestPostCalibrationComponentPass : Pass<"tf-stablehlo-test-post-calibration-component", "mlir::ModuleOp"> {
  let summary = "Test-only pass to test the PostCalibrationComponent.";
  let description = [{
    Runs the post-calibration passes for post-training quantization.
  }];
  let options = [
    Option<"unpack_quantized_types_", "unpack-quantized-types", "bool",
      /*default=*/"true", "Unpacks ops with uniform quantized types into "
      "operations without uniform quantized types (mostly i8 or i32).">
  ];
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect", "mlir::TF::TensorFlowDialect",
    "mlir::func::FuncDialect", "mlir::mhlo::MhloDialect",
    "mlir::quant::QuantDialect", "mlir::chlo::ChloDialect",
    "mlir::vhlo::VhloDialect", "mlir::shape::ShapeDialect",
    "mlir::quant::ir::TFQuantDialect",
  ];
}

def TestTFToStablehloPass : Pass<"tf-stablehlo-test-tf-to-stablehlo", "mlir::ModuleOp"> {
  let summary = "Test-only pass to test TFToStablehloPasses.";
  let description = [{
    Runs the TFToStablehloPasses.
  }];
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect", "mlir::TF::TensorFlowDialect",
    "mlir::chlo::ChloDialect", "mlir::quant::QuantDialect",
    "mlir::mhlo::MhloDialect", "mlir::shape::ShapeDialect",
    "mlir::sparse_tensor::SparseTensorDialect", "mlir::ub::UBDialect",
    "mlir::vhlo::VhloDialect",
  ];
}

def TestLiftQuantizableSpotsAsFunctionsWithQuantizationSpecsPass :
    Pass<"tf-stablehlo-test-lift-quantizable-spots-as-functions-with-quantization-specs", "mlir::ModuleOp"> {
  let summary = "Test-only pass for testing the LiftQuantizableSpotsAsFunctionsPass with a predefined QuantizationSpecs.";
  let description = [{
    This test-only pass is the same as `LiftQuantizableSpotsAsFunctionsPass` but
    has predefined `QuantizationSpecs` to make FileCheck testing easier.
  }];
  let options = [
    Option<"quantization_specs_", "quantization-specs",
      "mlir::tf_quant::stablehlo::testing::TestQuantizationSpecs",
      /*default=*/"mlir::tf_quant::stablehlo::testing::TestQuantizationSpecs::kEmpty",
      "Sets one of the predefined `QuantizationSpecs` for testing.",
      [{llvm::cl::values(
        clEnumValN(mlir::tf_quant::stablehlo::testing::TestQuantizationSpecs::kEmpty,
          "empty", "Uses empty (default) QuantizationSpecs."),
        clEnumValN(mlir::tf_quant::stablehlo::testing::TestQuantizationSpecs::kDisableAllDotGeneral,
          "disable-all-dot-general", "Disables all dot_general ops by matching lifted function names"),
        clEnumValN(mlir::tf_quant::stablehlo::testing::TestQuantizationSpecs::kStaticRangePtqToAll,
          "static-range-ptq-to-all", "Applies `StaticRangePtq` to all quantizable units."),
        clEnumValN(mlir::tf_quant::stablehlo::testing::TestQuantizationSpecs::kStaticRangePtqToComputeHeavy,
          "static-range-ptq-to-compute-heavy", "Applies `StaticRangePtq` to only compute heavy units.")
      )}]>
  ];
  let dependentDialects = [
      "mlir::func::FuncDialect",
      "mlir::stablehlo::StablehloDialect",
      "TF::TensorFlowDialect",
  ];
}
