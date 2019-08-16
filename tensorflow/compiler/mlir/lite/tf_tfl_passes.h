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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_PASSES_H_

#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"  // TF:local_config_mlir

namespace tensorflow {

// Quantization passess will run only when the user specifies a quantized type
// in the `-tf-inference-type` flag, which is converted to the function
// attribute "tf.quantize" by the importer module.
// TODO(fengliuai): switch to the cmd flag once the flags are moved to this
// file with main method.
bool ShouldRunQuantizePasses(mlir::ModuleOp m);

// TODO(b/139535802) - Simplify this signature, and fix the comments.
// Add the MLIR passes that convert TF dialect to TF Lite dialect
// to a MLIR `pass_manager`. These passes first raise the control flow in the TF
// control flow dialect, decode the constant tensors, and then legalize the
// module to TF Lite dialect with some optimizations afterwards.
// If `emit_builtin_tflite_ops` is true, TF Lite legalization passes will be
// added, which produces TF Lite ops. If `run_quantize` is true, quantization
// passes will be added. If `emit_quant_adaptor_ops` is true, Quantize and
// Dequantize ops are added to the inputs and outputs of the quantized model.
// If `lower_tensor_list_ops` is true, tensorlist ops will be lowered to basic
// TF ops before legalization to TF Lite dialect.
void AddTFToTFLConversionPasses(bool emit_builtin_tflite_ops, bool run_quantize,
                                bool emit_quant_adaptor_ops,
                                bool lower_tensor_list_ops,
                                mlir::PassManager* pass_manager);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_PASSES_H_
