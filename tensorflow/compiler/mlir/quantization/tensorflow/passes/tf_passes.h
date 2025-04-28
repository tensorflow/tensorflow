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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_TF_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_TF_PASSES_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/tf_quantization_lib/tf_quantization_config.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace mlir {
namespace quant {

// Creates a pass that add QuantizationUnitLoc to quantizable layers.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTFAddQuantizationUnitLocPass();

// Creates an instance of the PrepareQuantize pass, which will perform similar
// transformations as TFL::PrepareQuantizePass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFPrepareQuantizePass(
    const tf_quant::QuantizationSpecs& quant_specs,
    tensorflow::quantization::QuantizationMethod::PresetMethod
        quantization_method);

// Creates an instance of the PrepareQuantizeDRQ pass, which will
// perform similar transformations as TFL::PrepareQuantizeDynamicRangePass.
std::unique_ptr<OperationPass<ModuleOp>> CreateTFPrepareQuantizeDRQPass(
    const tf_quant::QuantizationSpecs& quant_specs,
    tensorflow::quantization::OpSet op_set);

// Converts FakeQuant ops to quant.qcast and quant.dcast (QDQ) pairs.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTFConvertFakeQuantToQdqPass();

// Apply graph optimizations such as fusing and constant folding to prepare
// lifting.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFPrepareLiftingPass(
    tensorflow::quantization::OpSet target_opset);

// Creates an instance of the PostQuantize pass, which will remove unnecessary
// ops from the final quantized graph.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizePass();

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_TF_PASSES_H_
