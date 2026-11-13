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

// This header file defines node specs for quantization and the methods to parse
// command line flags to these specs.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_COMMON_QUANTIZATION_LIB_QUANTIZATION_CONFIG_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_COMMON_QUANTIZATION_LIB_QUANTIZATION_CONFIG_H_

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/lite/tools/optimize/reduced_precision_metadata.h"

namespace mlir {
namespace TFL {

// Stores information about how to quantize a user-specified custom operation.
struct CustomOpInfo {
  std::vector<std::int32_t> quantizable_input_indices;
  bool is_weight_only = false;
  bool no_side_effect = true;
};

using CustomOpMap = std::unordered_map<std::string, CustomOpInfo>;
enum CustomOpUpdateOptions { kInputIndices, kWeightOnly, kNoSideEffect };
enum class QDQConversionMode { kQDQNone, kQDQStatic, kQDQDynamic, kQDQStrict };

// The data type that a model is converted to for inference. This is the subset
// of `tensorflow::DataType` that the TFLite quantization passes use, so that
// the quantization config does not depend on TensorFlow. The conversion from
// `tensorflow::DataType` is in `tf_quantization_config.h`.
enum class InferenceDataType {
  kInvalid,  // DT_INVALID, and any data type that is not listed below.
  kFloat,    // DT_FLOAT
  kHalf,     // DT_HALF
  kInt8,     // DT_INT8
  kUInt8,    // DT_UINT8
  kInt16,    // DT_INT16
  kUInt16,   // DT_UINT16
  kInt32,    // DT_INT32
  kQInt8,    // DT_QINT8
  kQUInt8,   // DT_QUINT8
  kQInt16,   // DT_QINT16
  kQUInt16,  // DT_QUINT16
  kQInt32,   // DT_QINT32
};

// Returns the name of `type`, e.g. "DT_QINT8".
absl::string_view InferenceDataTypeName(InferenceDataType type);

// Returns whether `type` is one of the quantized data types (`kQInt8`,
// `kQUInt8`, `kQInt16`, `kQUInt16` and `kQInt32`).
bool IsQuantizedInferenceDataType(InferenceDataType type);

struct QuantizationSpecs {
  // Which function this node quant specifications belong to.
  std::string target_func = "main";

  // Whether to trigger quantization passses for post-training quantization.
  // If true, the model input doesn't require user specified input ranges.
  bool post_training_quantization = false;

  // Whether to allow dynamic range quantization. This is the easiest
  // quantization mode which doesn't require QAT or sample inputs.
  // This option only targets `kHalf` and `kQInt8` inference type.
  bool weight_quantization = false;

  // Whether to use the MLIR dynamic range quantizer instead of TOCO.
  bool enable_mlir_dynamic_range_quantizer = false;

  // Whether to allow weight-only quantization. This scheme quantizes
  // weights but will dequantize them back at runtime which is useful for
  // memory bound case without kernel support available in lower precisions.
  // Used in MLIR dynamic range quantizer.
  bool weight_only_quantization = false;

  // The minimum number of elements in a weights array required to apply
  // quantization. This is especially useful not to quantize small tensors as
  // it is hard to get performance benefits from them with quantization. Used
  // in MLIR dynamic range quantizer with int8 weight data type.
  int64_t minimum_elements_for_weights = 1024;

  // Whether to calculate scales in float to keep quantized values the same with
  // old TOCO quantizer.
  bool legacy_float_scale = false;

  // Whether to perform per-tensor quantization. Currently, this option is only
  // valid when the quantization parameters need to be created by scanning the
  // constant content (post-training quantization or QAT without weight
  // FakeQuant).
  bool disable_per_channel = false;

  // Whether to disable per-channel weight quantization and enable legacy per
  // tensor quantization. The legacy quantization for Dense layers is
  // inconsistent with Conv 1x1 which always performs per channel quantization.
  bool disable_per_channel_for_dense_layers = false;

  // Whether to use fixed output ranges of the activation ops (tanh, sigmoid,
  // etc.) and not infer weight constants.
  // If this option is set, quantization emulation ops should be placed after
  // the ops in the input graph. This flag should be set to false for
  // post-training quantization.
  bool disable_infer_tensor_range = false;

  // Whether to use the unfrozen variable quantization in MLIR. Typically,
  // variables are frozen for passing passes, but some variables aren't frozen.
  // If it is true, QuantizeVariables pass will be added after the
  // PrepareQuantizePass.
  bool enable_mlir_variable_quantization = false;

  // The node type when the model is exported. Currently this is limited to
  // kFloat, kHalf, kQInt8, and kQUInt8. When kHalf is used, the
  // `weight_quantization` flag needs to set to true. When kQUInt8 is used,
  // the `weight_quantization` flag needs to set to false.
  InferenceDataType inference_type = InferenceDataType::kFloat;

  // The input and output data type during inference. This flag is only used
  // when `inference_type` is different from kFloat. This flag can only be set
  // to kFloat or as same as `inference_type`. If this flag is different
  // from `inference_type`, adaptor ops are inserted as heading and tailing ops
  // in the result model.
  InferenceDataType inference_input_type = InferenceDataType::kFloat;

  // Input node ranges. These ranges are stored as the same order of function
  // arguments. They are only used when `weight_quantization` is set to false,
  // and the model is required to have quantization parameters, either from
  // quantization aware training or calibration, for the remaining tensors.
  std::vector<std::pair<std::optional<double>, std::optional<double>>>
      input_ranges;

  // Whether to disable setting the quantization parameters of the input nodes
  // using input ranges.
  bool disable_set_input_nodes_quantization_params = false;

  // The default ranges can be used when a tensor doesn't have quantization
  // parameters and couldn't be quantized. Used only for latency tests.
  std::pair<std::optional<double>, std::optional<double>> default_ranges;

  // A serialized "QuantizationInfo" object to specify value ranges for some of
  // the tensors with known names.
  std::string serialized_quant_stats = "";

  // A bitmask to encode support for reduced precision inference in the model.
  tflite::optimize::ReducedPrecisionSupport support_mask =
      tflite::optimize::ReducedPrecisionSupport::None;

  // Whether to run the passes to propagate the quantization parameters and
  // graph rewrites. Returns false if the inference_type is kFloat or
  // `weight_quantization` flag is set.
  bool RunPropagationAndRewriteQuantizationPasses() const {
    return inference_type != InferenceDataType::kFloat && !weight_quantization;
  }

  // TODO: b/202075505 - make implicit weight type clearer
  // Whether run the passes and graph rewrites for dynamic range quantization.
  bool RunAndRewriteDynamicRangeQuantizationPasses() const {
    bool dynamic_range_quantize =
        (inference_type != InferenceDataType::kFloat) && weight_quantization &&
        !post_training_quantization && !disable_infer_tensor_range &&
        enable_mlir_dynamic_range_quantizer;
    return dynamic_range_quantize;
  }

  // Returns whether this inference type represents a signed storage type.
  bool IsSignedInferenceType() const {
    switch (inference_type) {
      case InferenceDataType::kQUInt8:
      case InferenceDataType::kQUInt16:
        return false;
      default:
        return true;
    }
  }

  // Gets the width of this quantization type. Returns 0 if it isn't a
  // quantization type.
  int64_t GetQuantizationTypeWidth() const {
    switch (inference_type) {
      case InferenceDataType::kInt8:
      case InferenceDataType::kUInt8:
      case InferenceDataType::kQInt8:
      case InferenceDataType::kQUInt8:
        return 8;
      case InferenceDataType::kInt16:
      case InferenceDataType::kUInt16:
      case InferenceDataType::kQInt16:
      case InferenceDataType::kQUInt16:
        return 16;
      case InferenceDataType::kInt32:
      case InferenceDataType::kQInt32:
        return 32;
      default:
        return 0;
    }
  }

  // Whether to add the NumericVerify ops to verify numbers before and after
  // quantization.
  bool verify_numeric = false;
  // Whether to add verification for layer by layer, or on whole model. When
  // disabled (per-layer) float and quantized ops will be run from same input
  // (output of previous quantized layer). When enabled, float and quantized ops
  // will run with respective float and quantized output of previous ops.
  bool whole_model_verify = false;

  // Whether to use fake quant attributes to calculate quantization parameters.
  bool use_fake_quant_num_bits = false;

  // Names of ops to block from quantization. Used in QuantizePass.
  // For dynamic range quantization, ops in blocklist are quantized in weight-
  // only manner.
  absl::flat_hash_set<std::string> ops_blocklist;

  // Names of locations to block from quantization. Used in QuantizePass.
  absl::flat_hash_set<std::string> nodes_blocklist;

  // Map from custom op code to custom op quantization information.
  // For dynamic range quantization, among the custom ops in the graph those
  // specified in this map are subject to quantization.
  CustomOpMap custom_map;

  // If other than kQDQNone, the model is a floating point graph with QDQ ops
  // to be eliminated and fused into quantized kernels.
  QDQConversionMode qdq_conversion_mode = QDQConversionMode::kQDQNone;

  // When set, adheres to the QDQ annotations added by the framework when
  // possible rather than quantizing any op that is possible to quantize.
  bool strict_qdq_mode = false;
};

// Parses the command line flag strings to the CustomOpMap specification.
void ParseCustomOpSpecs(absl::string_view node_names,
                        const CustomOpUpdateOptions& update_option,
                        CustomOpMap& custom_op_map);

// Gets the quantization specification for input arrays. The array names are not
// stored in the spec, and will be matched by position. The min/max will be
// ignored if the inference_type isn't a quantized type. Returns true if failed.
bool GetInputNodeQuantSpecs(const std::vector<std::string>& node_names,
                            const std::vector<std::optional<double>>& node_mins,
                            const std::vector<std::optional<double>>& node_maxs,
                            InferenceDataType inference_type,
                            QuantizationSpecs* quant_specs);

// Returns a human-readable string of the QDQQuantMode enum class
std::string GetQDQQuantModeString(QDQConversionMode mode);

// Returns the QDQQuantMode enum class from a human-readable string
QDQConversionMode GetQDQQuantModeFromString(const std::string& mode_str);
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_COMMON_QUANTIZATION_LIB_QUANTIZATION_CONFIG_H_
