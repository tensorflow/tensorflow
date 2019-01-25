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

#include <memory>
#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {

// Quantizes input_model and populates the provided builder with the new model.
// By default only weights tensors weight more than 1024 elements will be
// quantized.
//
// A tflite::Model can be obtained from the builder with:
//   const uint8_t* buffer = builder->GetBufferPointer();
//   tflite::Model* model = GetModel(buffer);
TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model);

// Same as above, but only weights with greater than or equal
// weights_min_num_elements elements will be quantized.
TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements);

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
