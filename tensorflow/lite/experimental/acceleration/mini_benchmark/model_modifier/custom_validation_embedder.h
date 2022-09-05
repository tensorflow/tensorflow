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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_MODIFIER_CUSTOM_VALIDATION_EMBEDDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_MODIFIER_CUSTOM_VALIDATION_EMBEDDER_H_

#include <utility>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/stderr_reporter.h"

namespace tflite {
namespace acceleration {

// Create a model with custom validation graph.
//
// 'validation model' (new subgraph)
// input (batch_size)
//           |
// +-----------------------+
// |'main_model' (0)       |
// | +---------------+     |
// | |input          +---+ |
// | +---------------+   | |
// |                     ~ |
// | +---------------+   | |
// | |outputs        +<--+ |
// | +---------------+     |
// |                       |
// +-----------------------+
//           |
// output (batch_size)
//
// The new model contains all the information from main_model, with an extra
// subgraph for validation purposes. The validation graph calls the primary
// subgraph with batch_size. The input data is embedded to the validation graph.
// custom_input should have the same order as the input in the main_model. E.g.
// custom_input[i] will be mapped to main_model.input[i].
class CustomValidationEmbedder {
 public:
  CustomValidationEmbedder(const Model& main_model, int batch_size,
                           std::vector<std::vector<uint8_t>> custom_input)
      : main_model_(main_model),
        batch_size_(batch_size),
        custom_input_(std::move(custom_input)) {
    main_model.UnPackTo(&main_model_obj_);
  }

  // Move only.
  CustomValidationEmbedder(CustomValidationEmbedder&&) = default;
  CustomValidationEmbedder& operator=(CustomValidationEmbedder&&) = default;

  // Build the final model with main_model and validation subgraph.
  MinibenchmarkStatus BuildModel(flatbuffers::FlatBufferBuilder& fbb);

 private:
  // Helper function to create tensors in validation graph based on primary
  // subgraph. This function creates new tensors and buffers based on the
  // from_subgraphs.tensors[from_indexes]. The new tensors will have shape[0]
  // set to batch_size_, and indexes stored in new_indexes.
  // New buffers will be created for each of the new tensors, and buffer data is
  // copied from the corresponding buffer_content.
  void CreateTensorsFrom(const SubGraph& from_subgraph,
                         const std::vector<int>& from_indexes,
                         std::vector<std::vector<uint8_t>>* buffer_content,
                         flatbuffers::FlatBufferBuilder& fbb,
                         std::vector<int>& new_indexes,
                         std::vector<flatbuffers::Offset<Buffer>>& buffers,
                         std::vector<flatbuffers::Offset<Tensor>>& tensors);

  const Model& main_model_;
  // Same as main_model_ with objective API.
  ModelT main_model_obj_;
  int batch_size_;
  std::vector<std::vector<uint8_t>> custom_input_;
  ErrorReporter* error_reporter_ = tflite::DefaultErrorReporter();
};

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_MODIFIER_CUSTOM_VALIDATION_EMBEDDER_H_
