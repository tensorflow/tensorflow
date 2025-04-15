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

include "mlir/Pass/PassBase.td"

def ComposeUniformQuantizedTypePass : Pass<"compose-uniform-quantized-type", "ModuleOp"> {
  let summary = "Compose uniform quantized types in StableHLO.";
  let constructor = "mlir::odml::CreateComposeUniformQuantizedTypePass()";
  let description = [{
    Identifies uniform quantization patterns and composes them to uniform
    quantized types. This pass targets a specific set of models that are
    quantized from the framework level, which produces "decomposed"
    quantization patterns due to the limitations of the framework.

    In the long term it is desired that the framework produces uniform
    quantized types directly and deprecate this pass.

    The target patterns should have the following characteristics, with some
    variations depending on the op being quantized.

    * A tensor is quantized using a `func::FuncOp` whose name contains
      "uniform_quantize". The first argument is the tensor to be quantized,
      the second argument is the zero point constant (element type: int) and
      the third argument is the inverse scale constant (element type: float).
    * A tensor is dequantized using a `func::FuncOp` whose name contains
      "uniform_dequantize". The first argument is the tensor to be quantized,
      the second argument is the zero point constant (element type: int) and
      the third argument is the inverse scale constant (element type: float).
    * Inputs to the target quantized op is quantized and the outputs are
      dequantized.
    * All quantization parameters (zero points and scales) are available,
      usually as constants.

    The resulting pattern for a quantized op looks roughly as follows:

    ```
    %0 = stablehlo.constant() : () -> tensor<...x!quant.uniform<i8:f32:3 ...>>
    %1 = stablehlo.uniform_quantize %2 : (tensor<...xf32>) -> tensor<...x!quant.uniform<i8:f32 ...>>
    %3 = stablehlo.my_quantized_op(%1, %0) : (tensor<...x!quant.uniform<i8:f32 ...>>, tensor<...x!quant.uniform<i8:f32:3 ...>>) -> tensor<...x!quant.uniform<i8:f32 ...>>
    %4 = stablehlo.dequantize %3 : (tensor<...x!quant.uniform<i8:f32 ...>>) -> tensor<...xf32>
    ```
  }];
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "quant::QuantDialect",
  ];
}

def UniformQuantizedStableHloToTflPass
    : Pass<"uniform-quantized-stablehlo-to-tfl", "mlir::func::FuncOp"> {
  let summary = "Converts StableHLO ops using uniform quantized types to equivalent TFL ops.";
  let constructor = "mlir::odml::CreateUniformQuantizedStableHloToTflPass()";
  let description = [{
    Converts StableHLO ops that accept or return uniform quantized types to
    equivalent ops in the TFLite dialect.
  }];
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "quant::QuantDialect",
    "mlir::TFL::TFLDialect",
  ];
}

def LegalizeHloToTfPass : Pass<"tf-legalize-hlo", "ModuleOp"> {
  let summary = "Legalize from MHLO to the TF dialect";
  let dependentDialects = ["TF::TensorFlowDialect"];
  let constructor = "mlir::odml::CreateLegalizeHloToTfPass()";
}

def LegalizeHloToTfLitePass : Pass<"tfl-legalize-hlo", "mlir::ModuleOp"> {
  let summary = "Legalize from MHLO to the TFLite dialect";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
  let constructor = "mlir::odml::CreateLegalizeHloToTfLitePass()";
}

def LegalizeStablehloToVhloPass : Pass<"stablehlo-legalize-vhlo", "ModuleOp"> {
  let summary = "Legalize StableHLO to VHLO for ODML.";
  let dependentDialects = ["mlir::vhlo::VhloDialect"];
}

def LegalizeVhloToStablehloPass : Pass<"vhlo-legalize-stablehlo", "ModuleOp"> {
  let summary = "Legalize VHLO to StableHLO for ODML.";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def LegalizeCompositeToCustomOpPass : Pass<"stablehlo-composite-legalize-tfl-custom", "func::FuncOp"> {
  let summary = "Legalize supported StableHLO CompositeOps to TFL CustomOp";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}
def LegalizeStablehloCustomCallToCompositePass : Pass<"stablehlo-custom-call-legalize-composite", "ModuleOp"> {
  let summary = "Legalize StableHLO custom call ops where the call target is 'stablehlo.composite' to composite ops.";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def UnfoldSplatConstantPass : Pass<"unfold-splat-constant-pass", "ModuleOp"> {
  let summary = "Replaces a splat constant tensor with a BroadcastInDim op.";
  let constructor = "mlir::odml::CreateUnfoldSplatConstantPass()";
}

def CompositeLoweringPass : Pass<"composite-lowering", "ModuleOp"> {
  let summary = "Lowers mhlo composites directly to tflite ops (when possible).";
  let dependentDialects = ["mlir::mhlo::MhloDialect", "TFL::TensorFlowLiteDialect"];
  let constructor = "mlir::odml::CreateCompositeLoweringPass()";
}

def TransposeCommuteOpsPass : Pass<"transpose-commute-ops", "ModuleOp"> {
  let summary = "Move transpose through specific ops";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
  let constructor = "mlir::odml::CreateTransposeCommuteOpsPass()";
}

def LegalizeChloToTflPass
    : Pass<"tfl-legalize-chlo", "mlir::func::FuncOp"> {
  let summary = "Lower chlo directly to tfl.";
  let constructor = "mlir::odml::CreateLegalizeChloToTflPass()";
  let description = [{
    Decompose chlo ops directly to tflite. Additionally, apply any chlo/shlo->tflite
    patterns that must be applied before chlo->shlo decomposition.
  }];
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "mlir::TFL::TensorFlowLiteDialect",
    "chlo::ChloDialect",
    "func::FuncDialect"
  ];
}

def PrepareHloPass
    : Pass<"prepare-hlo", "mlir::func::FuncOp"> {
  let summary = "Prepare mhlo for mhlo->tfl legalization.";
  let constructor = "mlir::odml::CreatePrepareHloPass()";
  let description = [{
    Apply generic mhlo->mhlo rewrites to make the graph more amenable for
    legalizing to tfl.
  }];
  let dependentDialects = [
    "mlir::mhlo::MhloDialect",
    "func::FuncDialect",
    "TFL::TensorFlowLiteDialect"
  ];
}

def LiftCallSiteLocCallerPass : Pass<"lift-callsite-loc-caller", "ModuleOp"> {
  let summary = "Lifts CallSites in pytorch generated stablehlo.";
  let description = [{
    Lifts CallSites in pytorch generated stablehlo to make the Loc's consitent
    after inlining.
  }];
  let dependentDialects = ["func::FuncDialect"];
}

def LegalizeQDQCustomCallPass : Pass<"legalize-qdq-custom-call", "ModuleOp"> {
  let summary = "Legalize stablehlo.custom_call for QDQ ops from odml-torch.";
  let description = [{
    Legalize stablehlo.custom_call for Q/DQ ops from odml-torch. Quant dialect
    is not registered in Python, so odml-torch uses custom_call to as carriers.
  }];
  let dependentDialects = ["func::FuncDialect", "quant::QuantDialect", "stablehlo::StablehloDialect"];
}

def BuildStableHLOCompositePass : Pass<"build-stablehlo-composite", "ModuleOp"> {
  let summary = "Build stablehlo.composite from inlined stablehlo.custom_call mark_tensor ops.";
  let description = [{
    Build stablehlo.composite from inlined stablehlo.custom_call mark_tensor ops.
  }];
  let dependentDialects = ["func::FuncDialect", "stablehlo::StablehloDialect"];
}

def StablehloUnfuseBatchNormPass : Pass<"stablehlo-unfuse-batch-norm", "mlir::func::FuncOp"> {
  let summary = "Unfuses stablehlo.batch_norm_(inference,training) ops into stablehlo arithmetic ops.";
  let dependentDialects = ["stablehlo::StablehloDialect", "TF::TensorFlowDialect", "mlir::shape::ShapeDialect"];
}

def StablehloFuseConvolutionPass : Pass<"stablehlo-fuse-convolution", "mlir::func::FuncOp"> {
  let summary = "Fuses stablehlo.convolution and binary element-wise ops.";
}
