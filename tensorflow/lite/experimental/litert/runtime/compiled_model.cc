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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/experimental/litert/cc/litert_event.h"

#if defined(__ANDROID__)
#include <android/hardware_buffer.h>
#endif

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/compiler/plugin/compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"
#include "tensorflow/lite/experimental/litert/runtime/external_litert_buffer_context.h"
#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/stderr_reporter.h"

using litert::Error;
using litert::Expected;
using litert::OwningBufferRef;
using litert::TensorBuffer;
using litert::TensorBufferScopedLock;
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
    static auto* default_signature_key =
        new std::string(LiteRtSignatureT::kDefaultSignatureKey);
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
    LiteRtModel model, LiteRtCompilationOptions compilation_options) {
  auto compiled_model = std::make_unique<LiteRtCompiledModelT>();

  std::optional<OwningBufferRef<uint8_t>> new_flatbuffer;
  // TODO: b/379317134 - Support other delegates with compilation options.
  if (compilation_options != kLiteRtHwAccelatorNone) {
    LITERT_LOG(LITERT_INFO, "Applying compiler plugins");
    if (auto flatbuffer =
            litert::internal::ApplyPlugins(model, compilation_options);
        !flatbuffer) {
      LITERT_LOG(LITERT_ERROR, "Failed to applying compiler plugins");
      return flatbuffer.Error();
    } else {
      new_flatbuffer = *flatbuffer;
    }
  }

  const char* model_buffer = nullptr;
  size_t model_buffer_size = 0;
  // The following code gets the original FB pointer from LiteRtModel.
  // TODO b/383120429 - Use a better way of getting the FB pointer.
  auto init_model_buffer = detail::GetTflInitFlatbuffer(*model);
  if (init_model_buffer.Size() != 0) {
    // Use the saved the original FB pointer when the LiteRtModel was created
    // from a buffer.
    model_buffer = init_model_buffer.StrData();
    model_buffer_size = init_model_buffer.Size();
  } else {
    // TODO b/383120429 - Once LiteRtModel provide tflite::Model object, switch
    // to use it to initialize Interpreter instead of serializing LiteRtModel.
    auto [data, size, offset] = compiled_model->model_buf_.GetWeak();
    if (LiteRtSerializeModel(model, &data, &size, &offset,
                             /*destroy_model=*/false) != kLiteRtStatusOk) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure);
    }
    compiled_model->alloc_ = std::make_unique<tflite::MemoryAllocation>(
        compiled_model->model_buf_.Data(), compiled_model->model_buf_.Size(),
        tflite::DefaultErrorReporter());
    model_buffer =
        reinterpret_cast<const char*>(compiled_model->alloc_->base());
    model_buffer_size = compiled_model->alloc_->bytes();
  }
  compiled_model->fb_model_ =
      tflite::FlatBufferModel::BuildFromBuffer(model_buffer, model_buffer_size);
  if (compiled_model->fb_model_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorFileIO);
  }

  if (auto res = compiled_model->Initialize(); !res.HasValue()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }

  // Apply the dispatch delegate, unconditionally, since the loaded model may
  // have been compiled for NPU at AOT.
  auto dispatch_delegate_options = litert::CreateDispatchDelegateOptionsPtr();
  LiteRtDispatchDelegateAddAllocBaseOption(dispatch_delegate_options.get(),
                                           model_buffer);
  auto dispatch_delegate =
      litert::CreateDispatchDelegatePtr(std::move(dispatch_delegate_options));
  if (auto status = compiled_model->interp_->ModifyGraphWithDelegate(
          dispatch_delegate.get());
      status != kTfLiteOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to modify graph with delegate");
  }

  compiled_model->RegisterDelegate(std::move(dispatch_delegate));

  return compiled_model;
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
  auto runner = interp_->GetSignatureRunner(
      signature_key == LiteRtSignatureT::kDefaultSignatureKey
          ? nullptr
          : std::string(signature_key).c_str());
  signature_runners_[signature_key] = runner;
  return runner;
}

Expected<void> LiteRtCompiledModelT::RegisterBuffer(
    tflite::SignatureRunner* runner, const TfLiteTensor* tensor,
    const char* tensor_name, LiteRtTensorBuffer buffer, bool is_input,
    std::vector<TensorBufferScopedLock>& scoped_locks) {
  bool backend_requires_cpu_buffer = false;

  auto requirements = buffer_context_->GetBufferRequirement(tensor);
  if (requirements) {
    auto supported_types = (*requirements)->SupportedTypes();
    if (!supported_types) {
      return supported_types.Error();
    }

    for (auto& type : *supported_types) {
      if (type == buffer->buffer_type()) {
        // Register tensor buffer if it can be used by the backend.
        buffer->Duplicate();
        TensorBuffer duplicated_buffer(buffer);
        if (auto status = buffer_context_->RegisterTensorBuffer(
                tensor, std::move(duplicated_buffer));
            status != kLiteRtStatusOk) {
          return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                            "Failed to register tensor buffer");
        }
        return {};
      }
      if (type == kLiteRtTensorBufferTypeHostMemory) {
        backend_requires_cpu_buffer = true;
      }
    }
  } else {
    // If the BufferRequirement is not registered, assumes the backend requires
    // CPU buffer.
    backend_requires_cpu_buffer = true;
  }

  if (backend_requires_cpu_buffer) {
    // When backend requires CPU buffer.
    bool bufer_is_cpu_compatible =
        buffer->buffer_type() == kLiteRtTensorBufferTypeHostMemory;
#if defined(__ANDROID__)
    if (buffer->buffer_type() == kLiteRtTensorBufferTypeAhwb) {
      if (__builtin_available(android 26, *)) {
        auto ahwb = buffer->GetAhwbBuffer();
        if (ahwb) {
          // TODO: b/382330322 - Update logic to check if the AHWB (stride) is
          // CPU compatible.
          AHardwareBuffer_Desc desc;
          AHardwareBuffer_describe(*ahwb, &desc);
          bufer_is_cpu_compatible = true;
        }
      }
    }
#endif
    if (bufer_is_cpu_compatible) {
      auto lock_and_addr = TensorBufferScopedLock::Create(buffer);
      if (!lock_and_addr) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          absl::StrCat("Failed to lock input tensor buffer: ",
                                       lock_and_addr.Error().Message()));
      }
      scoped_locks.push_back(std::move(lock_and_addr->first));
      TfLiteCustomAllocation custom_allocation{lock_and_addr->second,
                                               tensor->bytes};
      if (is_input) {
        runner->SetCustomAllocationForInputTensor(tensor_name,
                                                  custom_allocation,
                                                  /*flags=*/0);
      } else {
        runner->SetCustomAllocationForOutputTensor(tensor_name,
                                                   custom_allocation,
                                                   /*flags=*/0);
      }
      return {};
    }
  }
  // TODO: b/382330322 - Add buffer conversion logic instead of returning error.
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "The given buffer type is not supported.");
}

Expected<void> LiteRtCompiledModelT::Run(
    absl::string_view signature_key,
    const std::vector<LiteRtTensorBuffer>& input_buffers,
    const std::vector<LiteRtTensorBuffer>& output_buffers) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "Failed to get signature runner");
  }
  size_t num_inputs = input_buffers.size();
  if (num_inputs != runner->input_names().size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Input buffer size mismatch");
  }
  size_t num_outputs = output_buffers.size();
  if (num_outputs != runner->output_names().size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Output buffer size mismatch");
  }

  // In general output buffer events are assigned by the runtime and not the
  // caller; here we check for any violation of that condition.
  for (auto litert_output_buffer : output_buffers) {
    if (litert_output_buffer->HasEvent()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Output buffers cannot have events attached");
    }
  }

  // TODO: If input buffers have events, we wait on them before we launch the
  // inference. This is inefficient when using HW acceleration, since in that
  // case it would be best to make the HW accelerator wait for those events as
  // opposed to blocking the CPU here.
  for (auto input_buffer : input_buffers) {
    if (input_buffer->HasEvent()) {
      auto litert_event = input_buffer->GetEvent();
      if (!litert_event) {
        return litert_event.Error();
      }
      litert::Event event(*litert_event, /*owned=*/false);
      if (auto status = event.Wait(/*timeout_in_ms=*/-1); !status) {
        return status.Error();
      }
    }
  }

  std::vector<TensorBufferScopedLock> scoped_locks;
  scoped_locks.reserve(num_inputs + num_outputs);
  for (int i = 0; i < num_inputs; ++i) {
    const auto& input_name = runner->input_names()[i];
    auto* input_tensor = runner->input_tensor(input_name);
    auto res =
        RegisterBuffer(runner, input_tensor, input_name, input_buffers[i],
                       /*is_input=*/true, scoped_locks);
    if (!res) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        absl::StrCat("Failed to register input tensor buffer: ",
                                     res.Error().Message()));
    }
  }

  for (int i = 0; i < runner->output_names().size(); ++i) {
    const auto& output_name = runner->output_names()[i];
    auto* output_tensor = runner->output_tensor(output_name);
    auto res =
        RegisterBuffer(runner, output_tensor, output_name, output_buffers[i],
                       /*is_input=*/false, scoped_locks);
    if (!res) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrCat("Failed to register output tensor buffer: ",
                       res.Error().Message()));
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
