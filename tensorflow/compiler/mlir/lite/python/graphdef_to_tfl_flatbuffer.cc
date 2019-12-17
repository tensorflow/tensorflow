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

#include <ostream>
#include <utility>

#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Support/FileUtilities.h"  // TF:local_config_mlir
#include "mlir/Transforms/ViewOpGraph.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
#include "tensorflow/lite/toco/types.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

using stream_executor::port::StatusOr;

namespace tensorflow {

namespace {
// Op def string for TFLite_Detection_PostProcess Op.
const char kDetectionPostProcessOp[] =
    "name: 'TFLite_Detection_PostProcess' input_arg: { name: "
    "'raw_outputs/box_encodings' type: DT_FLOAT } input_arg: { name: "
    "'raw_outputs/class_predictions' type: DT_FLOAT } input_arg: { name: "
    "'anchors' type: DT_FLOAT } output_arg: { name: "
    "'TFLite_Detection_PostProcess' type: DT_FLOAT } output_arg: { name: "
    "'TFLite_Detection_PostProcess:1' type: DT_FLOAT } output_arg: { name: "
    "'TFLite_Detection_PostProcess:2' type: DT_FLOAT } output_arg: { name: "
    "'TFLite_Detection_PostProcess:3' type: DT_FLOAT } attr : { name: "
    "'h_scale' type: 'float'} attr : { name: 'max_classes_per_detection' "
    "type: 'int'} attr : { name: 'max_detections' type: 'int'} attr : { "
    "name: 'nms_iou_threshold' type: 'float'} attr : { name: "
    "'nms_score_threshold' type: 'float'} attr : { name: 'num_classes' type: "
    "'int'} attr : { name: 'w_scale' type: 'float'} attr : { name: 'x_scale' "
    "type: 'float'} attr : { name: 'y_scale' type: 'float'} attr { name: "
    "'detections_per_class' type: 'int' default_value { i : 100 }} attr { "
    "name: 'use_regular_nms' type: 'bool' default_value { b : false }}";

// Converts the toco::IODataType to tensorflow::DataType. Only contains the
// conversion mapping for constants defined in TFLite Python API.
DataType ConvertIODataTypeToDataType(toco::IODataType dtype) {
  switch (dtype) {
    case toco::IODataType::FLOAT:
      return DT_FLOAT;
    case toco::IODataType::QUANTIZED_UINT8:
      return DT_QUINT8;
    case toco::IODataType::INT8:
      return DT_QINT8;
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

StatusOr<std::pair<double, double>> InputStatsToMinMax(double mean, double std,
                                                       DataType type) {
  // Only qint8 and quint8 are considered here.
  double qmin, qmax;
  if (type == DT_QUINT8) {
    qmin = 0.0;
    qmax = 255.0;
  } else if (type == DT_QINT8) {
    qmin = -128.0;
    qmax = 127.0;
  } else {
    return errors::InvalidArgument("Only int8 and uint8 are considered.");
  }
  return std::make_pair((qmin - mean) / std, (qmax - mean) / std);
}

// Give a warning for any unused flags that have been specified.
void WarningUnusedFlags(const toco::ModelFlags& model_flags,
                        const toco::TocoFlags& toco_flags) {
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
  if (toco_flags.dump_graphviz_include_video()) {
    LOG(WARNING) << "Ignored dump_graphviz_video.";
  }
  if (model_flags.allow_nonexistent_arrays()) {
    LOG(WARNING) << "Allow allow_nonexistent_arrays.";
  }
}

// Dumps the op graph of the `module` to `filename` in DOT format.
Status DumpOpGraphToFile(mlir::ModuleOp module, const std::string& filename) {
  std::string error_message;
  auto output = mlir::openOutputFile(filename, &error_message);
  if (!error_message.empty()) {
    return errors::InvalidArgument("Failed to open file in %s.", filename);
  }
  mlir::PassManager pm(module.getContext());
  pm.addPass(mlir::createPrintOpGraphPass(output->os()));
  if (failed(pm.run(module))) {
    return errors::Unknown("Failed to dump Op Graph from MLIR module.");
  }
  output->keep();
  return Status::OK();
}

Status RegisterCustomBuiltinOps(const std::vector<string> extra_tf_opdefs) {
  for (const auto& tf_opdefs_string : extra_tf_opdefs) {
    tensorflow::OpDef opdef;
    if (!tensorflow::protobuf::TextFormat::ParseFromString(tf_opdefs_string,
                                                           &opdef)) {
      return errors::InvalidArgument("fail to parse extra OpDef");
    }
    // Make sure the op is not already registered. If registered continue.
    const OpRegistrationData* op_reg = nullptr;
    auto status =
        tensorflow::OpRegistry::Global()->LookUp(opdef.name(), &op_reg);
    if (status.ok()) continue;

    tensorflow::OpRegistry::Global()->Register(
        [opdef](tensorflow::OpRegistrationData* op_reg_data) -> Status {
          *op_reg_data = tensorflow::OpRegistrationData(opdef);
          return Status::OK();
        });
  }
  return Status::OK();
}

}  // namespace

Status ConvertGraphDefToTFLiteFlatBuffer(const toco::ModelFlags& model_flags,
                                         const toco::TocoFlags& toco_flags,
                                         const GraphDebugInfo& debug_info,
                                         const GraphDef& input,
                                         string* result) {
  mlir::MLIRContext context;
  GraphImportConfig specs;
  mlir::TFL::QuantizationSpecs quant_specs;

  // Parse input arrays.
  std::vector<string> node_names;
  std::vector<string> node_dtypes;
  std::vector<std::vector<int>> node_shapes;
  std::vector<double> node_mins;
  std::vector<double> node_maxs;
  quant_specs.inference_input_type =
      ConvertIODataTypeToDataType(toco_flags.inference_input_type());
  tensorflow::DataType inference_type =
      ConvertIODataTypeToDataType(toco_flags.inference_type());
  // Use non-float flag `inference_input_type` to override the `inference_type`
  // because we have to apply quantization to satisfy that.
  if (quant_specs.inference_input_type != tensorflow::DT_FLOAT) {
    inference_type = quant_specs.inference_input_type;
  }

  for (auto& flag : model_flags.input_arrays()) {
    node_names.push_back(flag.name());
    // TOCO doesn't required `data_type` to be filled for every input.
    // If it's not filled, make it an empty string so the importer will use
    // the data type in the NodeDef.
    auto toco_data_type = flag.data_type();
    if (toco_data_type == ::toco::IODataType::IO_DATA_TYPE_UNKNOWN) {
      node_dtypes.push_back("");
    } else {
      node_dtypes.push_back(
          DataType_Name(ConvertIODataTypeToDataType(toco_data_type)));
    }
    node_shapes.push_back(std::vector<int>(flag.shape().dims().begin(),
                                           flag.shape().dims().end()));
    // Currently, only UINT8 and INT8 require inputs stats
    if (inference_type == DT_QINT8 || inference_type == DT_QUINT8) {
      TF_ASSIGN_OR_RETURN(
          auto min_max, InputStatsToMinMax(flag.mean_value(), flag.std_value(),
                                           inference_type));
      node_mins.push_back(min_max.first);
      node_maxs.push_back(min_max.second);
    }
  }
  TF_RETURN_IF_ERROR(tensorflow::ParseInputArrayInfo(
      node_names, node_dtypes, node_shapes, &specs.inputs));
  if (mlir::TFL::GetInputNodeQuantSpecs(node_names, node_mins, node_maxs,
                                        inference_type, &quant_specs)) {
    return errors::InvalidArgument("Failed to get input quant spec.");
  }

  // Some extra flag related to post training quantization. If post-training
  // quantization is enabled, `inference_type` and `inference_input_type` are
  // not used by MLIR passes.
  if (toco_flags.post_training_quantize()) {
    quant_specs.weight_quantization = true;
    if (toco_flags.quantize_to_float16()) {
      quant_specs.inference_type = tensorflow::DT_HALF;
      quant_specs.inference_input_type = tensorflow::DT_HALF;
    } else {
      quant_specs.inference_type = tensorflow::DT_QINT8;
      quant_specs.inference_input_type = tensorflow::DT_QINT8;
    }
  }

  // Parse output arrays.
  std::vector<string> output_arrays(model_flags.output_arrays().begin(),
                                    model_flags.output_arrays().end());
  TF_RETURN_IF_ERROR(
      tensorflow::ParseOutputArrayInfo(output_arrays, &specs.outputs));

  // Other flags.
  bool emit_builtin_tflite_ops = !toco_flags.force_select_tf_ops();
  bool emit_select_tf_ops = toco_flags.enable_select_tf_ops();
  bool emit_custom_ops = toco_flags.allow_custom_ops();
  specs.prune_unused_nodes = true;
  specs.convert_legacy_fed_inputs = true;
  specs.graph_as_function = false;
  specs.upgrade_legacy = true;
  WarningUnusedFlags(model_flags, toco_flags);

  // Register any custom OpDefs.
  std::vector<string> extra_tf_opdefs(toco_flags.custom_opdefs().begin(),
                                      toco_flags.custom_opdefs().end());
  extra_tf_opdefs.push_back(kDetectionPostProcessOp);
  TF_RETURN_IF_ERROR(RegisterCustomBuiltinOps(extra_tf_opdefs));

  TF_ASSIGN_OR_RETURN(
      auto module, ConvertGraphdefToMlir(input, debug_info, specs, &context));

  if (toco_flags.has_dump_graphviz_dir()) {
    TF_RETURN_IF_ERROR(DumpOpGraphToFile(
        module.get(),
        // rename once we enable the new converter feature flag.
        absl::StrCat(toco_flags.dump_graphviz_dir(), "/toco_AT_IMPORT.dot")));
  }

  mlir::PassManager pm(module->getContext());
  mlir::TFL::PassConfig pass_config(quant_specs);
  pass_config.emit_builtin_tflite_ops = emit_builtin_tflite_ops;
  pass_config.lower_tensor_list_ops = true;

  tensorflow::AddTFToTFLConversionPasses(pass_config, &pm);

  auto status = ConvertTFExecutorToTFLOrFlatbuffer(
      module.get(), /*export_to_mlir=*/false, emit_builtin_tflite_ops,
      emit_select_tf_ops, emit_custom_ops, quant_specs, result, &pm);

  if (toco_flags.has_dump_graphviz_dir()) {
    TF_RETURN_IF_ERROR(DumpOpGraphToFile(
        // rename once we enable the new converter feature flag.
        module.get(), absl::StrCat(toco_flags.dump_graphviz_dir(),
                                   "/toco_AFTER_TRANSFORMATIONS.dot")));
  }

  return status;
}

}  // namespace tensorflow
