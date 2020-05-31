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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_CONFIG_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_CONFIG_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mlir {
namespace TFL {

struct QuantizationSpecs {
  // Which function this node quant specifications belong to.
  std::string target_func = "main";

  // Whether allow weight-only quantization. This is the easiest quantization
  // mode which doesn't require QAT or sample inputs. But it can only target
  // DT_HALF and DT_QINT8 inference type.
  bool weight_quantization = false;

  // Whether the quantization passes are triggered for post-training
  // quantization. If it is true, the model input doesn't require user specified
  // input ranges.
  // TODO(fengliuai): The `weight_quantization` is just a special case of
  // post-training quantization. We need to deprecate the `weight_quantization`.
  bool post_training_quantization = false;

  // When set to true, quantization will be done per-tensor. Currently, this
  // option is only valid when the quantization parameters need to be created by
  // scanning the constant content (post-training quantization or QAT without
  // weight FakeQuant).
  bool disable_per_channel = false;

  // The node type when the model is exported. Currently this is limited to
  // DT_FLOAT, DT_HALF, DT_QINT8, and DT_QUINT8. When DT_HALF is used, the
  // `weight_quantization` flag needs to set to true. When DT_QUINT8 is used,
  // the `weight_quantization` flag needs to set to false.
  tensorflow::DataType inference_type = tensorflow::DT_FLOAT;

  // The input and output data type during inference. This flag is only used
  // when `inference_type` is different from DT_FLOAT. This flag can only be set
  // to DT_FLOAT or as same as `inference_type`. If this flag is different
  // from `inference_type`, adaptor ops are inserted as heading and tailing ops
  // in the result model.
  tensorflow::DataType inference_input_type = tensorflow::DT_FLOAT;

  // Input node ranges. These ranges are stored as the same order of function
  // arguments. They are only used when `weight_quantization` is set to false,
  // and the model is required to have quantization parameters, either from
  // quantization aware training or calibration, for the remaining tensors.
  std::vector<std::pair<llvm::Optional<double>, llvm::Optional<double>>>
      input_ranges;

  // The default ranges can be used when a tensor doesn't have quantization
  // parameters and couldn't be quantized. Used only for latency tests.
  std::pair<llvm::Optional<double>, llvm::Optional<double>> default_ranges;

  // A serialized "QuantizationInfo" object to specify value ranges for some of
  // the tensors with known names.
  std::string serialized_quant_stats = "";

  // Whether run the passes to propagate the quantization parameters and graph
  // rewrites. Returns false if the inference_type is DT_FLOAT or
  // `weight_quantization` flag is set.
  bool RunPropagationAndRewriteQuantizationPasses() const {
    return inference_type != tensorflow::DT_FLOAT && !weight_quantization;
  }

  // Whether run the passes to only quantize the weights.
  bool RunWeightQuantization() const { return weight_quantization; }

  // Whether this inference type represents a signed storage type.
  bool IsSignedInferenceType() const {
    switch (inference_type) {
      case tensorflow::DT_QUINT8:
      case tensorflow::DT_QUINT16:
        return false;
      default:
        return true;
    }
  }

  // Gets the width of this quantization type. Returns 0 if it isn't a
  // quantization type.
  int64_t GetQuantizationTypeWidth() const {
    switch (inference_type) {
      case tensorflow::DT_QINT8:
      case tensorflow::DT_QUINT8:
        return 8;
      case tensorflow::DT_QINT16:
      case tensorflow::DT_QUINT16:
        return 16;
      case tensorflow::DT_QINT32:
        return 32;
      default:
        return 0;
    }
  }
};

// Parses the command line flag strings to the quantization specification for
// input arrays of a graph. The array names are not stored in the spec, and will
// be matched by position. Returns true if failed.
bool ParseInputNodeQuantSpecs(absl::string_view node_names,
                              absl::string_view min_values,
                              absl::string_view max_values,
                              absl::string_view inference_type,
                              QuantizationSpecs* quant_specs);

// Gets the quantization specification for input arrays. The array names are not
// stored in the spec, and will be matched by position. The min/max will be
// ignored if the inference_type isn't a quantized type. Returns true if failed.
bool GetInputNodeQuantSpecs(
    const std::vector<std::string>& node_names,
    const std::vector<llvm::Optional<double>>& node_mins,
    const std::vector<llvm::Optional<double>>& node_maxs,
    tensorflow::DataType inference_type, QuantizationSpecs* quant_specs);

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_CONFIG_H_
