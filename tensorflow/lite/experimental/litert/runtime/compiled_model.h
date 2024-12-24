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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_COMPILED_MODEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_COMPILED_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/runtime/external_litert_buffer_context.h"
#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"

// The LiteRtCompiledModelT is internal implementation of CompiledModel C++ API.
class LiteRtCompiledModelT {
 public:
  using Ptr = std::unique_ptr<LiteRtCompiledModelT>;

  LiteRtCompiledModelT() = default;
  ~LiteRtCompiledModelT() = default;

  // Creates a LiteRtCompiledModelT from a LiteRtModel object.
  // The model is loaded into memory and the caller takes ownership of the
  // returned object.
  static litert::Expected<Ptr> Create(
      LiteRtModel model, LiteRtCompilationOptions compilation_options);

  // Returns the buffer requirements for the n-th input tensor. The returned
  // LiteRtTensorBufferRequirements is used to create the input tensor
  // buffer.
  litert::Expected<LiteRtTensorBufferRequirements> GetInputBufferRequirements(
      absl::string_view signature_key, size_t input_index);

  // The same as GetInputBufferRequirements() for C API.
  litert::Expected<LiteRtTensorBufferRequirements>
  GetInputBufferRequirementsCApi(size_t signature_index, size_t input_index) {
    if (signature_index >= signature_keys_.size()) {
      return litert::Unexpected(
          kLiteRtStatusErrorIndexOOB,
          "Signature index is out of range of signature keys");
    }
    return GetInputBufferRequirements(*signature_keys_[signature_index],
                                      input_index);
  }

  // Returns the buffer requirements for the n-th output tensor. The returned
  // LiteRtTensorBufferRequirements is used to create the output tensor
  // buffer.
  litert::Expected<LiteRtTensorBufferRequirements> GetOutputBufferRequirements(
      absl::string_view signature_key, size_t output_index);

  // The same as GetOutputBufferRequirements() for C API.
  litert::Expected<LiteRtTensorBufferRequirements>
  GetOutputBufferRequirementsCApi(size_t signature_index, size_t output_index) {
    if (signature_index >= signature_keys_.size()) {
      return litert::Unexpected(
          kLiteRtStatusErrorIndexOOB,
          "Signature index is out of range of signature keys");
    }
    return GetOutputBufferRequirements(*signature_keys_[signature_index],
                                       output_index);
  }

  // Runs the model of the given signature with the provided input/output
  // litert::TensorBuffers.
  litert::Expected<void> Run(
      absl::string_view signature_key,
      const std::vector<LiteRtTensorBuffer>& input_buffers,
      const std::vector<LiteRtTensorBuffer>& output_buffers);

  // The same as Run() for C API.
  litert::Expected<void> RunCApi(size_t signature_index,
                                 size_t num_input_buffers,
                                 LiteRtTensorBuffer* input_buffers,
                                 size_t num_output_buffers,
                                 LiteRtTensorBuffer* output_buffers);

 private:
  // Processes the model and initializes the internal states.
  // This is called in the public Create*() methods.
  litert::Expected<void> Initialize();

  // Returns the buffer requirements for the given tensor.
  litert::Expected<LiteRtTensorBufferRequirements> GetTensorBufferRequirements(
      const TfLiteTensor* tensor);

  // Returns the SignatureRunner for the given signature key.
  // If the signature key is not found, returns nullptr.
  tflite::SignatureRunner* GetSignatureRunner(absl::string_view signature_key);

  // Registers the TensorBuffer for the given tensor with the SignatureRunner.
  // If the TensorBuffer can be directly consumed as CPU Tensors, they'll be
  // locked and use it with CustomAllocation. The buffer is locked by
  // LiteRtTensorBufferScopedLock and kept in the `scoped_locks`. It will be
  // unlocked automatically when the `scoped_locks` are destroyed.
  litert::Expected<void> RegisterBuffer(
      tflite::SignatureRunner* runner, const TfLiteTensor* tensor,
      const char* tensor_name, LiteRtTensorBuffer buffer, bool is_input,
      std::vector<litert::TensorBufferScopedLock>& scoped_locks);

  void RegisterDelegate(tflite::TfLiteOpaqueDelegateUniquePtr&& delegate) {
    delegates_.push_back(std::move(delegate));
  }

  // Map from signature key to SignatureRunner. This is used to lazy calling
  // GetSignatureRunner() which is expensive.
  absl::flat_hash_map<absl::string_view, tflite::SignatureRunner*>
      signature_runners_;

  // The buffer requirement maps for CPU buffers. For delegates with CPU
  // buffers, they don't register TensorBufferRequirements. Instead, the
  // CompiledModel creates the TensorBufferRequirements and stores them
  // in this map.
  absl::flat_hash_map<const TfLiteTensor*, litert::TensorBufferRequirements>
      cpu_buffer_requirements_;

  // The Interpreter and related objects used to run the model.
  std::unique_ptr<::tflite::Interpreter> interp_;
  std::unique_ptr<::tflite::FlatBufferModel> fb_model_;
  std::unique_ptr<::tflite::Allocation> alloc_;
  litert::OwningBufferRef<uint8_t> model_buf_;
  std::vector<const std::string*> signature_keys_;

  // The ExternalLiteRtBufferContext used to register tensor buffers with
  // Delegates.
  // Note: The ExternalLiteRtBufferContext must be destroyed after the
  // Interpreter.
  std::unique_ptr<litert::internal::ExternalLiteRtBufferContext>
      buffer_context_;

  std::vector<tflite::TfLiteOpaqueDelegateUniquePtr> delegates_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_COMPILED_MODEL_H_
