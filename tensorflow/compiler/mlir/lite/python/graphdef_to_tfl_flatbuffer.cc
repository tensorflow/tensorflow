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

#include "tensorflow/compiler/mlir/lite/python/graphdef_to_tfl_flatbuffer.h"

#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
#include "tensorflow/lite/toco/types.pb.h"

namespace tensorflow {

// Converts the toco::IODataType to tensorflow::DataType. Only contains the
// conversion mapping for constants defined in TFLite Python API.
DataType ConvertIODataTypeToDataType(toco::IODataType dtype) {
  switch (dtype) {
    case toco::IODataType::FLOAT:
      return DT_FLOAT;
    case toco::IODataType::QUANTIZED_UINT8:
      return DT_QUINT8;
    case toco::IODataType::INT32:
      return DT_INT32;
    case toco::IODataType::INT64:
      return DT_INT64;
    case toco::IODataType::STRING:
      return DT_STRING;
    case toco::IODataType::BOOL:
      return DT_BOOL;
    default:
      return DT_INVALID;
  }
}

// Give a warning for any unused flags that have been specified.
void WarningUnusedFlags(const toco::ModelFlags& model_flags,
                        const toco::TocoFlags& toco_flags) {
  if (toco_flags.inference_input_type()) {
    LOG(WARNING) << "Ignored inference_input_type.";
  }
  if (toco_flags.output_format()) {
    LOG(WARNING) << "Ignored output_format.";
  }
  if (toco_flags.default_ranges_min() || toco_flags.default_ranges_max()) {
    LOG(WARNING) << "Ignored default_ranges_stats.";
  }
  if (toco_flags.drop_control_dependency()) {
    LOG(WARNING) << "Ignored drop_control_dependency.";
  }
  if (toco_flags.reorder_across_fake_quant()) {
    LOG(WARNING) << "Ignored reorder_across_fake_quant.";
  }
  if (model_flags.change_concat_input_ranges()) {
    LOG(WARNING) << "Ignored change_concat_input_ranges.";
  }
  if (toco_flags.post_training_quantize()) {
    LOG(WARNING) << "Ignored post_training_quantize.";
  }
  if (toco_flags.dump_graphviz_dir().empty()) {
    LOG(WARNING) << "Ignored dump_graphviz_dir.";
  }
  if (toco_flags.dump_graphviz_include_video()) {
    LOG(WARNING) << "Ignored dump_graphviz_video.";
  }
  if (model_flags.allow_nonexistent_arrays()) {
    LOG(WARNING) << "Allow allow_nonexistent_arrays.";
  }
}

Status ConvertGraphDefToTFLiteFlatBuffer(const toco::ModelFlags& model_flags,
                                         const toco::TocoFlags& toco_flags,
                                         const GraphDebugInfo& debug_info,
                                         const GraphDef& input,
                                         string* result) {
  mlir::MLIRContext context;
  NodeSpecs specs;

  // Parse input arrays.
  std::vector<string> node_names;
  std::vector<string> node_dtypes;
  std::vector<std::vector<int>> node_shapes;
  std::vector<float> node_mins;
  std::vector<float> node_maxs;
  tensorflow::DataType inference_type =
      ConvertIODataTypeToDataType(toco_flags.inference_type());
  for (auto& flag : model_flags.input_arrays()) {
    node_names.push_back(flag.name());
    node_dtypes.push_back(
        DataType_Name(ConvertIODataTypeToDataType(flag.data_type())));
    node_shapes.push_back(std::vector<int>(flag.shape().dims().begin(),
                                           flag.shape().dims().end()));

    const float mean_value = flag.mean_value();
    const float std_value = flag.std_value();
    const float qmin = 0, qmax = 255;
    node_mins.push_back((qmin - mean_value) / std_value);
    node_maxs.push_back((qmax - mean_value) / std_value);
  }
  TF_RETURN_IF_ERROR(tensorflow::ParseInputArrayInfo(
      node_names, node_dtypes, node_shapes, inference_type, node_mins,
      node_maxs, &specs.inputs));

  // Parse output arrays.
  std::vector<string> output_arrays(model_flags.output_arrays().begin(),
                                    model_flags.output_arrays().end());
  TF_RETURN_IF_ERROR(tensorflow::ParseOutputArrayInfo(
      output_arrays, &specs.output_arrays, &specs.output_arrays_order));

  // Other flags.
  bool emit_builtin_tflite_ops = !toco_flags.force_select_tf_ops();
  bool emit_select_tf_ops = toco_flags.enable_select_tf_ops();
  bool emit_custom_ops = toco_flags.allow_custom_ops();
  specs.prune_unused_nodes = true;
  specs.convert_legacy_fed_inputs = true;
  WarningUnusedFlags(model_flags, toco_flags);

  bool emit_quant_adaptor_ops = false;
  bool lower_tensor_list_ops = true;
  TF_ASSIGN_OR_RETURN(
      auto module, ConvertGraphdefToMlir(input, debug_info, specs, &context));
  return ConvertTFExecutorToTFLOrFlatbuffer(
      module.get(), /*export_to_mlir=*/false, emit_builtin_tflite_ops,
      emit_select_tf_ops, emit_custom_ops, emit_quant_adaptor_ops,
      lower_tensor_list_ops, result);
}

}  // namespace tensorflow
