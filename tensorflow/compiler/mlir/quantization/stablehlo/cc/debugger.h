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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_DEBUGGER_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_DEBUGGER_H_

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"

namespace stablehlo::quantization {

// Disables debugging on `DumpTensor` ops.
void DisableDebugging(mlir::ModuleOp module_op);

// Enables debugging on `DumpTensor` ops.
void EnableDebugging(tensorflow::quantization::ExportedModel& exported_model);

// Changes the filename from `unquantized_tensor_data.pb` to
// `quantized_tensor_data.pb`.
void ChangeToQuantizedFilename(mlir::ModuleOp module_op);

}  // namespace stablehlo::quantization

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_DEBUGGER_H_
