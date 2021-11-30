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
        trim_functions_allowlist({}),
        quant_specs(std::move(specs)),
        form_clusters(false),
        unfold_batch_matmul(true),
        legalize_tf_while(true),
        shape_inference(true),
        runtime_verification(true),
        enable_tflite_variables(false),
        disable_variable_freezing(false),
        unfold_large_splat_constant(false),
        guarantee_all_funcs_one_use(false),
        enable_hlo_to_tf_conversion(false) {}

  // If `emit_builtin_tflite_ops` is true, TF Lite legalization passes will be
  // added, which produces TF Lite ops.
  bool emit_builtin_tflite_ops;
  // If `lower_tensor_list_ops` is true, tensorlist ops will be lowered to basic
  // TF ops before legalization to TF Lite dialect.
  bool lower_tensor_list_ops;
  // The allowlist of functions that would be preserved after trimming.
  llvm::ArrayRef<std::string> trim_functions_allowlist;
  // All information about quantization.
  QuantizationSpecs quant_specs;
  // If `form_clusters` is true , clusters are formed by grouping consecutive
  // ops of the same device, under a `tf_device.launch` op.
  bool form_clusters;
  // if `unfold_batch_matmul` is true, the tf.BatchMatMul is unfolded to a set
  // of tfl.fully_connected ops.
  bool unfold_batch_matmul;
  // Whether to legalize TF While to TFL While.
  // Note: This is staging step and will be removed.
  // TODO(b/137395003): Remove post switching legalization.
  bool legalize_tf_while;
  // Whether to outline WhileOp at the end of the pipeline.
  bool outline_tf_while = false;
  // Whether to do shape inference.
  bool shape_inference;
  // Whether to do TFLite runtime verification.
  bool runtime_verification;
  // Whether to enable TFLite variables or not, this will allow
  // mutable variables and produce ReadVariable/AssignVariable ops in TFLite.
  bool enable_tflite_variables;
  // Whether to disable the variable freezing pass or not.
  // By default we freeze all variables and disallow mutable variables. When
  // 'enable_tflite_variables' is true then we allow mutable variable only.
  bool disable_variable_freezing;
  // Whether to unfold large splat constant tensors and replace them with
  // fill operation.
  bool unfold_large_splat_constant;
  // Whether to run the `GuaranteeAllFuncsOneUsePass` to ensure each function
  // has a single use.
  bool guarantee_all_funcs_one_use;
  // Whether to enable the hlo to tf conversion.
  bool enable_hlo_to_tf_conversion;
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_COMMON_TFL_PASS_CONFIG_H_
