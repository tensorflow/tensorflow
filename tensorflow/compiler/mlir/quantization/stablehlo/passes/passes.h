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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_PASSES_H_

#include <memory>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_options.pb.h"

namespace mlir::quant::stablehlo {

// Creates a `QuantizePass` that quantizes ops according to surrounding qcast /
// dcast ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateQuantizePass(
    const quant::QuantizationSpecs& quantization_specs);

// Creates a pass that quantizes weight component of StableHLO graph.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizeWeightPass(
    const ::stablehlo::quantization::QuantizationComponentSpec&
        quantization_component_spec = {});

// Creates an instance of the StableHLO dialect PrepareQuantize pass without any
// arguments. Preset method of SRQ is set to the quantization option by default.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass(
    bool enable_per_channel_quantization = false, int bit_width = 8);

// Converts a serialized StableHLO module to bfloat16 and output serialized
// module.
FailureOr<std::string> ConvertSerializedStableHloModuleToBfloat16(
    MLIRContext* context, StringRef serialized_stablehlo_module);

// Adds generated pass default constructors or options definitions.
#define GEN_PASS_DECL
// Adds generated pass registration functions.
#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_PASSES_H_
