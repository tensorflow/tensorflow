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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_COMMON_TFL_PASS_CONFIG_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_COMMON_TFL_PASS_CONFIG_H_

#include <string>

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace TFL {

// A config that controls which passes get run as part TFLite translation.
struct PassConfig {
  // If `emit_builtin_tflite_ops` is true, TF Lite legalization passes will be
  // added, which produces TF Lite ops.
  bool emit_builtin_tflite_ops;
  // If `emit_quant_adaptor_ops` is true, Quantize and
  // Dequantize ops are added to the inputs and outputs of the quantized model.
  bool emit_quant_adaptor_ops;
  // If `lower_tensor_list_ops` is true, tensorlist ops will be lowered to basic
  // TF ops before legalization to TF Lite dialect.
  bool lower_tensor_list_ops;
  // If run_quantize is true, quantization passes will be added.
  bool run_quantize;
  // The whitelist of functions that would be preserved after trimming. If not
  // specified/empty, this pass is a no-op.
  llvm::ArrayRef<std::string> trim_functions_whitelist;
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_COMMON_TFL_PASS_CONFIG_H_
