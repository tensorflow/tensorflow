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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZE_WEIGHTS_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZE_WEIGHTS_H_

#include <cstdint>
#include <memory>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {

// Supported resulting types from quantization process.
enum class BufferType { QUANTIZED_INT8, QUANTIZED_FLOAT16 };

// This macro is for internal use for conversions requiring previous behavior.
#ifdef TFLITE_USE_PREVIOUS_HYBRID_SCHEME
constexpr bool kUseUpdatedHybridSchemeDefault = false;
#else
constexpr bool kUseUpdatedHybridSchemeDefault = true;
#endif

// Quantizes input_model and populates the provided builder with the new model.
// By default only weights tensors weight more than 1024 elements will be
// quantized.
//
// A tflite::Model can be obtained from the builder with:
//   const uint8_t* buffer = builder->GetBufferPointer();
//   tflite::Model* model = GetModel(buffer);
TfLiteStatus QuantizeWeights(
    flatbuffers::FlatBufferBuilder* builder, const Model* input_model,
    BufferType quant_type = BufferType::QUANTIZED_INT8);

// Same as above, but only weights with greater than or equal
// weights_min_num_elements elements will be quantized.
TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements);

// Stores information about how to quantize a user-specified custom operation.
typedef struct {
  std::vector<std::int32_t> quantizable_input_indices;
  bool is_hybrid;
} CustomOpInfo;

// Map from custom op code to custom op quantization information.
typedef std::unordered_map<string, CustomOpInfo> CustomOpMap;

// Same as above, but with entry point of quantizing custom ops.
TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements,
                             const CustomOpMap& custom_op_map);

// Same as above, but if use updated_hybrid_scheme is false,
// use previous quantization scheme.
TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements,
                             const CustomOpMap& custom_op_map,
                             bool use_updated_hybrid_scheme);

namespace internal {
// If use_hybrid_evaluation is false, will disable using hybrid eval for
// operations that support it.
//
// We use this internal QuantizeWeights call to test models with hybrid
// evaluation disabled.
TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements,
                             bool use_hybrid_evaluation);
}  // namespace internal

}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZE_WEIGHTS_H_
