/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/model_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/lite/tools/optimize/reduced_precision_metadata.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/types.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace tensorflow {
namespace internal {
namespace {

using ::tensorflow::quantization::PyFunctionLibrary;
using ::tflite::optimize::ReducedPrecisionSupport;

// Op def string for TFLite_Detection_PostProcess Op.
constexpr mlir::StringRef kDetectionPostProcessOp =
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

constexpr mlir::StringRef kUnidirectionalSequenceLstmOp =
    "name: 'UnidirectionalSequenceLstm' input_arg: {name: 'Input' type: "
    "DT_FLOAT} input_arg: { name: 'InputToInputWeights' type: DT_FLOAT } "
    "input_arg: { name: 'InputToForgetWeights' type: DT_FLOAT } input_arg: { "
    "name: 'InputToCellWeights' type: DT_FLOAT} input_arg: { name: "
    "'InputToOutputWeights' type: DT_FLOAT } input_arg: { name: "
    "'RecurrentToInputWeights' type: DT_FLOAT} input_arg: { name: "
    "'RecurrentToForgetWeights' type: DT_FLOAT} input_arg: { name: "
    "'RecurrentToCellWeights' type: DT_FLOAT } input_arg: { name: "
    "'RecurrentToOutputWeights' type: DT_FLOAT } input_arg: { name: "
    "'CellToInputWeights' type: DT_FLOAT} input_arg: { name: "
    "'CellToForgetWeights' type: DT_FLOAT } input_arg: { name: "
    "'CellToOutputWeights' type: DT_FLOAT } input_arg: { name: 'InputGateBias' "
    "type: DT_FLOAT } input_arg: { name: 'ForgetGateBias' type: DT_FLOAT } "
    "input_arg: { name: 'kCellGateBias' type: DT_FLOAT } input_arg: { name: "
    "'OutputGateBias' type: DT_FLOAT } input_arg: { name: 'ProjectionWeights' "
    "type: DT_FLOAT } input_arg: { name: 'ProjectionBias' type: DT_FLOAT } "
    "input_arg: { name: 'InputActivationState' type: DT_FLOAT} input_arg: { "
    "name: 'InputCellStateTensor' type: DT_FLOAT } "
    "output_arg: { name: 'Concat' type: DT_FLOAT} "
    "output_arg: { name: "
    "'LastState' type: DT_FLOAT } output_arg: { name: 'Output' type: DT_FLOAT} "
    "attr : { name: '_tflite_input_indices' type: 'list(int)'}";

constexpr mlir::StringRef kUnidirectionalSequenceRnnOp =
    "name: 'UnidirectionalSequenceRnn' input_arg: {name: 'Input' type: "
    "DT_FLOAT} input_arg: { name: 'Weights' type: DT_FLOAT } "
    "input_arg: { name: 'RecurrentWeights' type: DT_FLOAT } input_arg: { "
    "name: 'Bias' type: DT_FLOAT} "
    "input_arg: { name: 'HiddenState' type: DT_FLOAT} "
    "output_arg: { name: "
    "'LastState' type: DT_FLOAT } output_arg: { name: 'Output' type: "
    "DT_FLOAT} "
    "attr : { name: '_tflite_input_indices' type: 'list(int)'}";

// Converts the tflite::IODataType to tensorflow::DataType. Only contains the
// conversion mapping for constants defined in TFLite Python API.
DataType ConvertIODataTypeToDataType(tflite::IODataType dtype) {
  switch (dtype) {
    case tflite::IODataType::FLOAT:
      return DT_FLOAT;
    case tflite::IODataType::FLOAT16:
      return DT_HALF;
    case tflite::IODataType::FLOAT64:
      return DT_DOUBLE;
    case tflite::IODataType::QUANTIZED_UINT8:
      return DT_QUINT8;
    case tflite::IODataType::QUANTIZED_INT8:
      return DT_QINT8;
    case tflite::IODataType::QUANTIZED_INT16:
      return DT_QINT16;
    case tflite::IODataType::INT8:
      return DT_INT8;
    case tflite::IODataType::INT16:
      return DT_INT16;
    case tflite::IODataType::UINT16:
      return DT_UINT16;
    case tflite::IODataType::INT32:
      return DT_INT32;
    case tflite::IODataType::UINT32:
      return DT_UINT32;
    case tflite::IODataType::INT64:
      return DT_INT64;
    case tflite::IODataType::UINT8:
      return DT_UINT8;
    case tflite::IODataType::UINT64:
      return DT_UINT64;
    case tflite::IODataType::STRING:
      return DT_STRING;
    case tflite::IODataType::BOOL:
      return DT_BOOL;
    case tflite::IODataType::COMPLEX64:
      return DT_COMPLEX64;
    case tflite::IODataType::COMPLEX128:
      return DT_COMPLEX128;
    case tflite::IODataType::RESOURCE:
      return DT_RESOURCE;
    case tflite::IODataType::VARIANT:
      return DT_VARIANT;
    default:
      return DT_INVALID;
  }
}

absl::StatusOr<std::pair<double, double>> InputStatsToMinMax(double mean,
                                                             double std,
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
    return absl::InvalidArgumentError("Only int8 and uint8 are considered.");
  }
  return std::make_pair((qmin - mean) / std, (qmax - mean) / std);
}

absl::Status RegisterCustomBuiltinOps(
    const std::vector<std::string> extra_tf_opdefs) {
  for (const auto& tf_opdefs_string : extra_tf_opdefs) {
    OpDef opdef;
    // NOLINTNEXTLINE: Use tsl::protobuf to be compatible with OSS.
    if (!tsl::protobuf::TextFormat::ParseFromString(tf_opdefs_string, &opdef)) {
      return errors::InvalidArgument("fail to parse extra OpDef");
    }
    // Make sure the op is not already registered. If registered continue.
    const OpRegistrationData* op_reg =
        OpRegistry::Global()->LookUp(opdef.name());
    if (op_reg) continue;

    OpRegistry::Global()->Register(
        [opdef](OpRegistrationData* op_reg_data) -> absl::Status {
          *op_reg_data = OpRegistrationData(opdef);
          return absl::OkStatus();
        });
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status RegisterAllCustomOps(
    const tflite::ConverterFlags& converter_flags) {
  // Register any custom OpDefs.
  std::vector<std::string> extra_tf_opdefs(
      converter_flags.custom_opdefs().begin(),
      converter_flags.custom_opdefs().end());
  extra_tf_opdefs.push_back(kDetectionPostProcessOp.str());
  extra_tf_opdefs.push_back(kUnidirectionalSequenceLstmOp.str());
  extra_tf_opdefs.push_back(kUnidirectionalSequenceRnnOp.str());
  return RegisterCustomBuiltinOps(extra_tf_opdefs);
}

absl::Status PopulateQuantizationSpecs(
    const tflite::ModelFlags& model_flags,
    tflite::ConverterFlags& converter_flags,
    mlir::TFL::QuantizationSpecs* quant_specs,
    std::vector<std::string>* node_names, std::vector<std::string>* node_dtypes,
    std::vector<std::optional<std::vector<int>>>* node_shapes,
    std::vector<std::optional<double>>* node_mins,
    std::vector<std::optional<double>>* node_maxs) {
  quant_specs->inference_input_type =
      ConvertIODataTypeToDataType(converter_flags.inference_input_type());
  DataType inference_type =
      ConvertIODataTypeToDataType(converter_flags.inference_type());
  // Use non-float flag `inference_input_type` to override the `inference_type`
  // because we have to apply quantization to satisfy that.
  if (quant_specs->inference_input_type != DT_FLOAT) {
    inference_type = quant_specs->inference_input_type;
  }

  for (auto& flag : model_flags.input_arrays()) {
    node_names->push_back(flag.name());
    // TOCO doesn't required `data_type` to be filled for every input.
    // If it's not filled, make it an empty string so the importer will use
    // the data type in the NodeDef.
    auto tflite_data_type = flag.data_type();
    if (tflite_data_type == tflite::IODataType::IO_DATA_TYPE_UNKNOWN) {
      node_dtypes->push_back("");
    } else {
      node_dtypes->push_back(
          DataType_Name(ConvertIODataTypeToDataType(tflite_data_type)));
    }
    if (flag.shape().unknown_rank()) {
      node_shapes->push_back(std::nullopt);
    } else {
      node_shapes->push_back(std::vector<int>(flag.shape().dims().begin(),
                                              flag.shape().dims().end()));
    }
    // Currently, only UINT8 and INT8 require inputs stats
    if (inference_type == DT_QINT8 || inference_type == DT_QUINT8) {
      if (flag.has_mean_value() && flag.has_std_value()) {
        TF_ASSIGN_OR_RETURN(
            auto min_max, InputStatsToMinMax(flag.mean_value(),
                                             flag.std_value(), inference_type));
        node_mins->push_back(min_max.first);
        node_maxs->push_back(min_max.second);
      } else {
        node_mins->push_back(std::nullopt);
        node_maxs->push_back(std::nullopt);
      }
    }
  }

  if (mlir::TFL::GetInputNodeQuantSpecs(*node_names, *node_mins, *node_maxs,
                                        inference_type, quant_specs)) {
    return errors::InvalidArgument("Failed to get input quant spec.");
  }

  // Some extra flag related to post training quantization. If post-training
  // quantization is enabled, `inference_type` and `inference_input_type` are
  // not used by MLIR passes.
  if (converter_flags.post_training_quantize()) {
    quant_specs->weight_quantization = true;
    quant_specs->disable_per_channel =
        converter_flags.disable_per_channel_quantization();
    if (converter_flags.quantize_to_float16()) {
      quant_specs->inference_type = DT_HALF;
      quant_specs->inference_input_type = DT_HALF;
    } else {
      quant_specs->inference_type = DT_QINT8;
      quant_specs->inference_input_type = DT_QINT8;
    }
  } else {
    // These flags are incompatible with post_training_quantize() as only
    // QAT models can provide required ranges.
    quant_specs->disable_infer_tensor_range =
        converter_flags.disable_infer_tensor_range();
    quant_specs->use_fake_quant_num_bits =
        converter_flags.use_fake_quant_num_bits();
  }

  // Add information about half-precision support if fp16 quantization applies.
  // TODO(b/195945955): Add e2e test for this.
  if (converter_flags.quantize_to_float16() ||
      converter_flags.allow_bfloat16()) {
    ReducedPrecisionSupport mask = ReducedPrecisionSupport::None;
    if (converter_flags.quantize_to_float16()) {
      mask |= ReducedPrecisionSupport::Float16Inference;
    }
    if (converter_flags.allow_bfloat16()) {
      mask |= ReducedPrecisionSupport::Bfloat16Inference;
    }
    if (converter_flags.accumulation_type() == tflite::IODataType::FLOAT16) {
      mask |= ReducedPrecisionSupport::Float16Accumulation;
    } else {
      mask |= ReducedPrecisionSupport::Float32Accumulation;
    }
    quant_specs->support_mask = mask;
  }

  // Other flags.
  if (converter_flags.has_default_ranges_min()) {
    quant_specs->default_ranges.first = converter_flags.default_ranges_min();
  }
  if (converter_flags.has_default_ranges_max()) {
    quant_specs->default_ranges.second = converter_flags.default_ranges_max();
  }
  quant_specs->enable_mlir_dynamic_range_quantizer =
      converter_flags.enable_mlir_dynamic_range_quantizer();
  quant_specs->enable_mlir_variable_quantization =
      converter_flags.enable_mlir_variable_quantization();
  quant_specs->disable_per_channel_for_dense_layers =
      converter_flags.disable_per_channel_quantization_for_dense_layers();
  return absl::OkStatus();
}

absl::Status ConvertMLIRToTFLiteFlatBuffer(
    const tflite::ModelFlags& model_flags,
    tflite::ConverterFlags& converter_flags,
    std::unique_ptr<mlir::MLIRContext>&& context,
    mlir::OwningOpRef<mlir::ModuleOp> module,
    const mlir::TFL::PassConfig& pass_config,
    const std::unordered_set<std::string>& saved_model_tags,
    std::string* result,
    const PyFunctionLibrary* quantization_py_function_lib) {
  mlir::TFL::PassConfig pass_config_copy = pass_config;
  pass_config_copy.outline_tf_while = true;

  // Checks whether the model contains an `XlaCallModuleOp` operation which
  // is a wrapper around StableHLO.
  // This option is mutually exclusive to `enable_stablehlo_conversion`, the
  // latter of which takes precedence.
  // TODO(b/290109282): explore removing the enable_hlo_to_tf_conversion flag
  // entirely, such that the added passes are no-ops in the non-shlo case.
  module->walk([&](mlir::TF::XlaCallModuleOp xla_call_module_op) {
    pass_config_copy.enable_hlo_to_tf_conversion = true;
    mlir::WalkResult::interrupt();
  });

  auto status = ConvertTFExecutorToTFLOrFlatbuffer(
      std::move(context), std::move(module), converter_flags, pass_config_copy,
      saved_model_tags, model_flags.saved_model_dir(), result,
      /*export_to_mlir=*/false, quantization_py_function_lib);

  return status;
}

void WarningUnusedFlags(const tflite::ModelFlags& model_flags,
                        const tflite::ConverterFlags& converter_flags) {
  if (converter_flags.output_format()) {
    LOG(WARNING) << "Ignored output_format.";
  }
  if (converter_flags.drop_control_dependency()) {
    LOG(WARNING) << "Ignored drop_control_dependency.";
  }
  if (converter_flags.reorder_across_fake_quant()) {
    LOG(WARNING) << "Ignored reorder_across_fake_quant.";
  }
  if (model_flags.change_concat_input_ranges()) {
    LOG(WARNING) << "Ignored change_concat_input_ranges.";
  }
  if (converter_flags.dump_graphviz_include_video()) {
    LOG(WARNING) << "Ignored dump_graphviz_video.";
  }
  if (model_flags.allow_nonexistent_arrays()) {
    LOG(WARNING) << "Allow allow_nonexistent_arrays.";
  }
}

}  // namespace internal
}  // namespace tensorflow
