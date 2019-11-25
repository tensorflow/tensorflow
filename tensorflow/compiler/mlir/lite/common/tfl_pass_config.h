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
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"

namespace mlir {
namespace TFL {

// A config that controls which passes get run as part TFLite converter.
struct PassConfig {
  explicit PassConfig(QuantizationSpecs specs)
      : emit_builtin_tflite_ops(true),
        lower_tensor_list_ops(false),
        trim_functions_whitelist({}),
        quant_specs(specs),
        skip_control_dialect(false),
        form_clusters(false),
        inline_functions(false) {}

  // If `emit_builtin_tflite_ops` is true, TF Lite legalization passes will be
  // added, which produces TF Lite ops.
  bool emit_builtin_tflite_ops;
  // If `lower_tensor_list_ops` is true, tensorlist ops will be lowered to basic
  // TF ops before legalization to TF Lite dialect.
  bool lower_tensor_list_ops;
  // The whitelist of functions that would be preserved after trimming.
  llvm::ArrayRef<std::string> trim_functions_whitelist;
  // All information about quantization.
  QuantizationSpecs quant_specs;
  // If `skip_control_dialect` is true, TF executor dialect is not converted to
  // TF control dialect prior to legalization to TF Lite.
  // TODO(b/142911013): Remove flag once control dialect is removed.
  bool skip_control_dialect;
  // If `form_clusters` is true (and `skip_control_dialect` is true), clusters
  // are formed by grouping consecutive ops of the same device, under a
  // `tf_device.launch` op.
  bool form_clusters;
  // Inline function calls within the main function in the MLIR module, prior
  // to legalization to TFLite.
  bool inline_functions;
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_COMMON_TFL_PASS_CONFIG_H_
