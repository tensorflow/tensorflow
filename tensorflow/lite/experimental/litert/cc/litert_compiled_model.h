// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_COMPILED_MODEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_COMPILED_MODEL_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_handle.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"

namespace litert {

// The CompiledModel is a higher level inference API. It is created by
// provided model with compilation options. Internally, it instantiates runtime
// and applies Delegates mapped to the compilation options.
// It also supports getting BufferRequirements to create input/output
// TensorBuffers, and it allows to invoke the model with the input/output
// TensorBuffers.
//
// Example user flow:
//
// 1. Create CompiledModel
// 2. Query the model input/output requirements
// 3. Create input/output TensorBuffers
// 4. Fill the input TensorBuffers with input data
// 5. Invoke the model with the input/output TensorBuffers
// 6. Evaluate the output TensorBuffers

class CompiledModel
    : public internal::Handle<LiteRtCompiledModel, LiteRtDestroyCompiledModel> {
 public:
  CompiledModel() = default;

  // Parameter `owned` indicates if the created CompiledModel object should take
  // ownership of the provided `compiled_model` handle.
  explicit CompiledModel(Model* model, LiteRtCompiledModel compiled_model,
                         bool owned = true)
      : internal::Handle<LiteRtCompiledModel, LiteRtDestroyCompiledModel>(
            compiled_model, owned),
        model_(model) {}

  // Creates a CompiledModel from a TFLite file.
  // The model is loaded into memory and the caller takes ownership of the
  // returned object.
  static Expected<CompiledModel> Create(
      litert::Model& model,
      LiteRtCompilationOptions compilation_options = kLiteRtHwAccelatorNone) {
    LiteRtCompiledModel compiled_model;
    if (auto status = LiteRtCreateCompiledModel(
            model.Get(), compilation_options, &compiled_model);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to create compiled model");
    }
    return CompiledModel(&model, compiled_model);
  }

  // Returns the buffer requirements for the given n-th input tensor. The
  // returned TensorBufferRequirements is used to create the input tensor
  // buffer.
  litert::Expected<TensorBufferRequirements> GetInputBufferRequirements(
      size_t signature_index, size_t input_index) {
    LiteRtTensorBufferRequirements buffer_requirements;
    if (auto status = LiteRtGetCompiledModelInputBufferRequirements(
            Get(), signature_index, input_index, &buffer_requirements);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to get input buffer requirements");
    }
    return TensorBufferRequirements(buffer_requirements, /*owned=*/false);
  }

  // Returns the buffer requirements for the given output tensor. The returned
  // TensorBufferRequirements is used to create the output tensor
  // buffer.
  litert::Expected<TensorBufferRequirements> GetOutputBufferRequirements(
      size_t signature_index, size_t output_index) {
    LiteRtTensorBufferRequirements buffer_requirements;
    if (auto status = LiteRtGetCompiledModelOutputBufferRequirements(
            Get(), signature_index, output_index, &buffer_requirements);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to get output buffer requirements");
    }
    return TensorBufferRequirements(buffer_requirements, /*owned=*/false);
  }

  // A helper function to creates the input tensor buffers for the given
  // signature. It uses BufferRequirements and RankedTensorType to create the
  // input tensor buffers.
  Expected<std::vector<TensorBuffer>> CreateInputBuffers(
      size_t signature_index);

  // A helper function to creates the output tensor buffers for the given
  // signature. It uses BufferRequirements and RankedTensorType to create the
  // output tensor buffers.
  Expected<std::vector<TensorBuffer>> CreateOutputBuffers(
      size_t signature_index);

  // Runs the model of the given signature with the provided input/output
  // TensorBuffers.
  Expected<void> Run(size_t signature_index,
                     const std::vector<TensorBuffer>& input_buffers,
                     const std::vector<TensorBuffer>& output_buffers);

  // Runs the model of the given signature with the provided input/output
  // TensorBuffer map.
  Expected<void> Run(
      size_t signature_index,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map);

 private:
  Model* model_;
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_COMPILED_MODEL_H_
