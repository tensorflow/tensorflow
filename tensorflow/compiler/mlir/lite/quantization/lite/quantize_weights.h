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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_QUANTIZE_WEIGHTS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_QUANTIZE_WEIGHTS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace mlir {
namespace lite {

// Supported resulting types from quantization process.
enum class BufferType { QUANTIZED_INT8, QUANTIZED_FLOAT16 };

// Stores information about how to quantize a user-specified custom operation.
// CustomOpInfo contains info of its corresponding CustomOp registered in the
// CustomOpMap. 'quantizable_input_indices' is used to determine which indices
// of the CustomOp are quantizable. 'is_weight_only' is used specify whether the
// custom op is quantized only for storage and dequantized at runtime.
// 'no_side_effect' is used to determine whether the op can be pruned if
// considered as trivially dead.
struct CustomOpInfo {
  std::vector<std::int32_t> quantizable_input_indices;
  bool is_weight_only = false;
  bool no_side_effect = true;
};

using BuiltinOperatorSet = absl::flat_hash_set<tflite::BuiltinOperator>;
// Map from custom op code to custom op quantization information.
using CustomOpMap = std::unordered_map<std::string, CustomOpInfo>;

// Applies dynamic range quantization for the given model wehre the input_model
// type is flatbuffer but is converted to MLIR during quantization process and
// then converted back to flatbuffer for return. Note that this is part of
// reaching feature parity with the old quantizer for dynamic range
// quantization, specifically for
// third_party/tensorflow/lite/tools/optimize/quantize_weights.h.
// TODO(b/202468183): Selective quantization + quant debugger support for
// dynamic range quantization for verify_numeric and whole_model_verify flags.
absl::Status QuantizeWeights(
    flatbuffers::FlatBufferBuilder* builder, const tflite::Model* input_model,
    const tflite::TensorType& inference_type,
    const absl::flat_hash_set<std::string>& denylisted_ops,
    const CustomOpMap& custom_op_map,
    int64_t minimum_elements_for_weights = 1024,
    bool disable_per_channel = false, bool weight_only_quantization = false,
    bool legacy_float_scale = false);

// Overloading methods to support old quantizer versions API
absl::Status QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const tflite::Model* input_model,
                             int64_t weights_min_num_elements,
                             bool use_hybrid_evaluation = true);

absl::Status QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const tflite::Model* input_model,
                             BufferType quant_type = BufferType::QUANTIZED_INT8,
                             bool use_updated_hybrid_scheme = true);

absl::Status QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const tflite::Model* input_model,
                             int64_t weights_min_num_elements,
                             const CustomOpMap& custom_op_map,
                             bool use_updated_hybrid_scheme = true,
                             const BuiltinOperatorSet& op_denylist = {});

}  // namespace lite
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_QUANTIZE_WEIGHTS_H_
