/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TOCO_LEGACY_QUANTIZE_WEIGHTS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TOCO_LEGACY_QUANTIZE_WEIGHTS_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace mlir {
namespace lite {
namespace toco_legacy {

using ::tflite::BuiltinOperator;
using ::tflite::Model;

// Supported resulting types from quantization process.
enum class BufferType { QUANTIZED_INT8, QUANTIZED_FLOAT16 };
enum class QuantizerType { OLD_QUANTIZER, MLIR_QUANTIZER };

// Stores information about how to quantize a user-specified custom operation.
struct CustomOpInfo {
  std::vector<std::int32_t> quantizable_input_indices;
  bool is_hybrid;
};

// Map from custom op code to custom op quantization information.
using CustomOpMap = std::unordered_map<std::string, CustomOpInfo>;

// This macro is for internal use for conversions requiring previous behavior.
#ifdef TFLITE_USE_PREVIOUS_HYBRID_SCHEME
// Use asymmetric quantized activations and per-channel quantized weights.
constexpr bool kUseUpdatedHybridSchemeDefault = false;
#else
// Use symmetric quantized activations and per-channel quantized weights.
constexpr bool kUseUpdatedHybridSchemeDefault = true;
#endif

// Quantizes input_model and populates the provided builder with the new model.
// By default only weights tensors weight more than 1024 elements will be
// quantized.
//
// A tflite::Model can be obtained from the builder with:
//   const uint8_t* buffer = builder->GetBufferPointer();
//   tflite::Model* model = GetModel(buffer);
absl::Status QuantizeWeights(
    flatbuffers::FlatBufferBuilder* builder, const Model* input_model,
    BufferType quant_type = BufferType::QUANTIZED_INT8,
    bool use_updated_hybrid_scheme = kUseUpdatedHybridSchemeDefault,
    QuantizerType quantizer_type = QuantizerType::OLD_QUANTIZER);

// Same as above, but only weights with greater than or equal
// weights_min_num_elements elements will be quantized.
absl::Status QuantizeWeights(
    flatbuffers::FlatBufferBuilder* builder, const Model* input_model,
    uint64_t weights_min_num_elements,
    QuantizerType quantizer_type = QuantizerType::OLD_QUANTIZER);

// Same as above, but with entry point of quantizing custom ops.
absl::Status QuantizeWeights(
    flatbuffers::FlatBufferBuilder* builder, const Model* input_model,
    uint64_t weights_min_num_elements, const CustomOpMap& custom_op_map,
    QuantizerType quantizer_type = QuantizerType::OLD_QUANTIZER);

// Same as above, but if use updated_hybrid_scheme is false,
// use previous quantization scheme. Optional op_denylist argument
// disables hybrid evaluation for provided BuiltinOperators.
absl::Status QuantizeWeights(
    flatbuffers::FlatBufferBuilder* builder, const Model* input_model,
    uint64_t weights_min_num_elements, const CustomOpMap& custom_op_map,
    bool use_updated_hybrid_scheme,
    const absl::flat_hash_set<BuiltinOperator>& op_denylist = {},
    QuantizerType quantizer_type = QuantizerType::OLD_QUANTIZER);

namespace internal {
// If use_hybrid_evaluation is false, will disable using hybrid eval for
// operations that support it.
//
// We use this internal QuantizeWeights call to test models with hybrid
// evaluation disabled.
absl::Status QuantizeWeights(
    flatbuffers::FlatBufferBuilder* builder, const Model* input_model,
    uint64_t weights_min_num_elements, bool use_hybrid_evaluation,
    QuantizerType quantizer_type = QuantizerType::OLD_QUANTIZER);
}  // namespace internal

}  // namespace toco_legacy
}  // namespace lite
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TOCO_LEGACY_QUANTIZE_WEIGHTS_H_
