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

#include "tensorflow/lite/experimental/litert/runtime/compiled_model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"
#include "tensorflow/lite/experimental/litert/runtime/external_litert_buffer_context.h"
#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/stderr_reporter.h"

using litert::Expected;
using litert::SmallVec;
using litert::TensorBuffer;
using litert::Unexpected;
using litert::internal::ExternalLiteRtBufferContext;

Expected<void> LiteRtCompiledModelT::Initialize() {
  // Use BuiltinOpResolverWithoutDefaultDelegates to avoid auto applying of
  // Xnnpack delegate with GetSignatureRunner() API.
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterBuilder(*fb_model_, resolver)(&interp_);
  if (interp_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }

  signature_keys_ = interp_->signature_keys();
  if (signature_keys_.empty()) {
    static std::string* default_signature_key =
        new std::string(LITERT_DEFAULT_SIGNATURE_KEY);
    signature_keys_.push_back(default_signature_key);
  }
  // Register the ExternalLiteRtBufferContext for TensorBuffer handshaking.
  buffer_context_ =
      std::make_unique<litert::internal::ExternalLiteRtBufferContext>();
  interp_->SetExternalContext(kTfLiteLiteRtBufferContext,
                              buffer_context_.get());

  return {};
}

Expected<LiteRtCompiledModelT::Ptr> LiteRtCompiledModelT::Create(
    LiteRtModel model, LiteRtComplicationOptions complication_options) {
  auto runtime = std::make_unique<LiteRtCompiledModelT>();

  const char* model_buffer = nullptr;
  size_t model_buffer_size = 0;
  // The following code gets the original FB pointer from LiteRtModel.
  // TODO b/383120429 - Use a better way of getting the FB pointer.
  if (model->model_buffer) {
    // Use the saved the original FB pointer when the LiteRtModel was created
    // from a buffer.
    model_buffer = reinterpret_cast<const char*>(model->model_buffer);
    model_buffer_size = model->model_buffer_size;
  } else {
    // TODO b/383120429 - Once LiteRtModel provide tflite::Model object, switch
    // to use it to initialize Interpreter instead of serializing LiteRtModel.
    auto [data, size, offset] = runtime->model_buf_.GetWeak();
    if (LiteRtSerializeModel(model, &data, &size, &offset,
                             /*destroy_model=*/false) != kLiteRtStatusOk) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure);
    }
    runtime->alloc_ = std::make_unique<tflite::MemoryAllocation>(
        runtime->model_buf_.Data(), runtime->model_buf_.Size(),
        tflite::DefaultErrorReporter());
    model_buffer = reinterpret_cast<const char*>(runtime->alloc_->base());
    model_buffer_size = runtime->alloc_->bytes();
  }
  runtime->fb_model_ =
      tflite::FlatBufferModel::BuildFromBuffer(model_buffer, model_buffer_size);
  if (runtime->fb_model_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorFileIO);
  }

  if (auto res = runtime->Initialize(); !res.HasValue()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }

  // TODO: b/379317134 - Support other delegates with compilation options.
  if (complication_options & kHwAccelNpu) {
    auto dispatch_delegate_options = litert::CreateDispatchDelegateOptionsPtr();
    LiteRtDispatchDelegateAddAllocBaseOption(dispatch_delegate_options.get(),
                                             model_buffer);
    auto dispatch_delegate =
        litert::CreateDispatchDelegatePtr(std::move(dispatch_delegate_options));
    if (auto status =
            runtime->interp_->ModifyGraphWithDelegate(dispatch_delegate.get());
        status != kTfLiteOk) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to modify graph with delegate");
    }
  }

  return runtime;
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtCompiledModelT::GetTensorBufferRequirements(const TfLiteTensor* tensor) {
  auto requirements = buffer_context_->GetBufferRequirement(tensor);
  if (requirements) {
    return (*requirements)->Get();
  }
  LiteRtTensorBufferRequirements litert_cpu_buffer_requirements;
  LiteRtTensorBufferType cpu_buffer_type[] = {
      kLiteRtTensorBufferTypeHostMemory};
  uint32_t cpu_buffer_strides[] = {0};
  auto res = LiteRtCreateTensorBufferRequirements(
      /*num_supported_tensor_buffer_types=*/1, cpu_buffer_type, tensor->bytes,
      /*num_strides=*/1, cpu_buffer_strides, &litert_cpu_buffer_requirements);
  if (res != kLiteRtStatusOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create CPU buffer requirements");
  }
  cpu_buffer_requirements_[tensor] =
      litert::TensorBufferRequirements(litert_cpu_buffer_requirements);
  return litert_cpu_buffer_requirements;
}

Expected<LiteRtTensorBufferRequirements>
LiteRtCompiledModelT::GetInputBufferRequirements(
    absl::string_view signature_key, size_t input_index) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "Failed to get signature runner");
  }
  auto input_names = runner->input_names();
  if (input_index >= input_names.size()) {
    return Unexpected(kLiteRtStatusErrorIndexOOB, "Input index out of range");
  }
  auto input_name = input_names[input_index];
  auto* input_tensor = runner->input_tensor(input_name);
  if (input_tensor == nullptr) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to get input tensor");
  }

  return GetTensorBufferRequirements(input_tensor);
}

Expected<LiteRtTensorBufferRequirements>
LiteRtCompiledModelT::GetOutputBufferRequirements(
    absl::string_view signature_key, size_t output_index) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "Failed to get signature runner");
  }
  auto output_names = runner->output_names();
  if (output_index >= output_names.size()) {
    return Unexpected(kLiteRtStatusErrorIndexOOB, "Output index out of range");
  }
  auto output_name = output_names[output_index];
  auto* output_tensor = runner->output_tensor(output_name);
  if (output_tensor == nullptr) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "Failed to get output tensor");
  }

  return GetTensorBufferRequirements(output_tensor);
}

tflite::SignatureRunner* LiteRtCompiledModelT::GetSignatureRunner(
    absl::string_view signature_key) {
  if (signature_runners_.contains(signature_key)) {
    return signature_runners_[signature_key];
  }
  auto runner =
      interp_->GetSignatureRunner(signature_key == LITERT_DEFAULT_SIGNATURE_KEY
                                      ? nullptr
                                      : std::string(signature_key).c_str());
  signature_runners_[signature_key] = runner;
  return runner;
}

Expected<void> LiteRtCompiledModelT::Run(
    absl::string_view signature_key,
    std::vector<LiteRtTensorBuffer>& input_buffers,
    std::vector<LiteRtTensorBuffer>& output_buffers) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "Failed to get signature runner");
  }
  if (input_buffers.size() != runner->input_names().size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Input buffer size mismatch");
  }
  if (output_buffers.size() != runner->output_names().size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Output buffer size mismatch");
  }

  for (int i = 0; i < runner->input_names().size(); ++i) {
    const auto& input_name = runner->input_names()[i];
    auto* input_tensor = runner->input_tensor(input_name);
    if (input_buffers[i]->buffer_type() == kLiteRtTensorBufferTypeHostMemory) {
      // Assign CPU buffer via CustomAllocation.
      TensorBuffer cpu_buffer(input_buffers[i], /*owned=*/false);
      auto lock_and_addr = litert::TensorBufferScopedLock::Create(cpu_buffer);
      TfLiteCustomAllocation custom_allocation{lock_and_addr->second,
                                               input_tensor->bytes};
      runner->SetCustomAllocationForInputTensor(input_name, custom_allocation,
                                                /*flags=*/0);
    } else {
      // Register tensor buffer for non CPU buffers.
      input_buffers[i]->Duplicate();
      TensorBuffer duplicated_buffer(input_buffers[i]);
      if (auto status = buffer_context_->RegisterTensorBuffer(
              input_tensor, std::move(duplicated_buffer));
          status != kLiteRtStatusOk) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to register input tensor buffer");
      }
    }
  }

  for (int i = 0; i < runner->output_names().size(); ++i) {
    const auto& output_name = runner->output_names()[i];
    auto* output_tensor = runner->output_tensor(output_name);
    if (output_buffers[i]->buffer_type() == kLiteRtTensorBufferTypeHostMemory) {
      // Assign CPU buffer via CustomAllocation.
      TensorBuffer cpu_buffer(output_buffers[i], /*owned=*/false);
      auto lock_and_addr = litert::TensorBufferScopedLock::Create(cpu_buffer);
      TfLiteCustomAllocation custom_allocation{lock_and_addr->second,
                                               output_tensor->bytes};
      runner->SetCustomAllocationForOutputTensor(output_name, custom_allocation,
                                                 /*flags=*/0);
    } else {
      output_buffers[i]->Duplicate();
      TensorBuffer duplicated_buffer(output_buffers[i]);
      if (auto status = buffer_context_->RegisterTensorBuffer(
              output_tensor, std::move(duplicated_buffer));
          status != kLiteRtStatusOk) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to register output tensor buffer");
      }
    }
  }

  if (auto res = runner->AllocateTensors(); res != kTfLiteOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to allocate tensors");
  }

  if (auto res = runner->Invoke(); res != kTfLiteOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to invoke");
  }

  return {};
}

litert::Expected<void> LiteRtCompiledModelT::RunCApi(
    size_t signature_index, size_t num_input_buffers,
    LiteRtTensorBuffer* input_buffers, size_t num_output_buffers,
    LiteRtTensorBuffer* output_buffers) {
  if (signature_index >= signature_keys_.size()) {
    return litert::Unexpected(
        kLiteRtStatusErrorIndexOOB,
        "Signature index is out of range of signature keys");
  }
  std::vector<LiteRtTensorBuffer> input_buffers_vec;
  input_buffers_vec.reserve(num_input_buffers);
  for (int i = 0; i < num_input_buffers; ++i) {
    input_buffers_vec.push_back(std::move(input_buffers[i]));
  }
  std::vector<LiteRtTensorBuffer> output_buffers_vec;
  output_buffers_vec.reserve(num_output_buffers);
  for (int i = 0; i < num_output_buffers; ++i) {
    output_buffers_vec.push_back(std::move(output_buffers[i]));
  }
  return Run(*signature_keys_[signature_index], input_buffers_vec,
             output_buffers_vec);
}
