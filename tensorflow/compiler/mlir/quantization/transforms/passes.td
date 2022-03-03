/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// Quant pass definition file

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_PASSES_QUANT_PASSES
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_PASSES_QUANT_PASSES

include "mlir/Pass/PassBase.td"

def QuantConvertConst : Pass<"quant-convert-const", "FuncOp"> {
  let summary = "Converts constants followed by qbarrier to actual quantized "
                "values";
  let constructor = "mlir::quant::createConvertConstPass()";
}

def QuantConvertSimulatedQuant
    : Pass<"quant-convert-simulated-quantization", "FuncOp"> {
  let summary = "Converts training-time simulated quantization ops to "
                "corresponding quantize/dequantize casts";
  let constructor = "mlir::quant::createConvertSimulatedQuantPass()";
}

#endif // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_PASSES_QUANT_PASSES
