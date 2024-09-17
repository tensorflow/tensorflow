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

#include "tensorflow/compiler/mlir/lite/python/graphdef_to_tfl_flatbuffer.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/model_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.h"
#include "tensorflow/compiler/mlir/lite/types.pb.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

absl::Status ConvertGraphDefToTFLiteFlatBuffer(
    const tflite::ModelFlags& model_flags,
    tflite::ConverterFlags& converter_flags, const GraphDebugInfo& debug_info,
    const GraphDef& input, std::string* result) {
  auto context = std::make_unique<mlir::MLIRContext>();
  GraphImportConfig specs;
  mlir::quant::QuantizationSpecs quant_specs;

  // Parse input arrays.
  std::vector<std::string> node_names;
  std::vector<std::string> node_dtypes;
  std::vector<std::optional<std::vector<int>>> node_shapes;
  std::vector<std::optional<double>> node_mins;
  std::vector<std::optional<double>> node_maxs;

  // Populate quantization specs.
  TF_RETURN_IF_ERROR(internal::PopulateQuantizationSpecs(
      model_flags, converter_flags, &quant_specs, &node_names, &node_dtypes,
      &node_shapes, &node_mins, &node_maxs));

  TF_RETURN_IF_ERROR(
      ParseInputArrayInfo(node_names, node_dtypes, node_shapes, &specs.inputs));

  // Parse output arrays.
  std::vector<std::string> output_arrays(model_flags.output_arrays().begin(),
                                         model_flags.output_arrays().end());
  TF_RETURN_IF_ERROR(ParseOutputArrayInfo(output_arrays, &specs.outputs));

  // Parse control output arrays.
  std::vector<std::string> control_output_arrays(
      model_flags.control_output_arrays().begin(),
      model_flags.control_output_arrays().end());
  TF_RETURN_IF_ERROR(
      ParseOutputArrayInfo(control_output_arrays, &specs.control_outputs));

  specs.prune_unused_nodes = true;
  specs.convert_legacy_fed_inputs = true;
  specs.graph_as_function = false;
  specs.upgrade_legacy = true;
  specs.unconditionally_use_set_output_shapes = true;
  internal::WarningUnusedFlags(model_flags, converter_flags);

  // Register all custom ops, including user-specified custom ops.
  TF_RETURN_IF_ERROR(internal::RegisterAllCustomOps(converter_flags));

  TF_ASSIGN_OR_RETURN(auto module, ConvertGraphdefToMlir(input, debug_info,
                                                         specs, context.get()));

  mlir::TFL::PassConfig pass_config(quant_specs);
  bool emit_builtin_tflite_ops = !converter_flags.force_select_tf_ops();
  pass_config.emit_builtin_tflite_ops = emit_builtin_tflite_ops;
  pass_config.unfold_batch_matmul = converter_flags.unfold_batchmatmul();
  pass_config.lower_tensor_list_ops = converter_flags.lower_tensor_list_ops();
  pass_config.legalize_custom_tensor_list_ops =
      converter_flags.legalize_custom_tensor_list_ops();
  // Disable the unfolding of the 16x16 TF::BatchMatMulOp to avoid the
  // conversion to an unsupported 16x16 TFL::FullyConnectedOp.
  if (converter_flags.inference_type() == tflite::IODataType::QUANTIZED_INT16) {
    pass_config.unfold_batch_matmul = false;
  }
  pass_config.unfold_large_splat_constant =
      converter_flags.unfold_large_splat_constant();
  pass_config.enable_dynamic_update_slice =
      converter_flags.enable_dynamic_update_slice();
  pass_config.preserve_assert_op = converter_flags.preserve_assert_op();
  pass_config.guarantee_all_funcs_one_use =
      converter_flags.guarantee_all_funcs_one_use();
  pass_config.enable_stablehlo_conversion =
      converter_flags.convert_to_stablehlo();
  pass_config.canonicalizing_inf_as_min_max_float =
      converter_flags.canonicalizing_inf_as_min_max_float();

  // StableHLO Quantizer is not supported for GraphDef inputs, so
  // quantization_py_function_lib is set to nullptr.
  return internal::ConvertMLIRToTFLiteFlatBuffer(
      model_flags, converter_flags, std::move(context), std::move(module),
      pass_config,
      /*saved_model_tags=*/{}, result,
      /*quantization_py_function_lib=*/nullptr);
}

}  // namespace tensorflow
