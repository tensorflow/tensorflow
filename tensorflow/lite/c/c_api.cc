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
#include "tensorflow/lite/c/c_api.h"

#include <memory>
#include <mutex>  // NOLINT
#include <utility>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/c/common_internal.h"
#include "tensorflow/lite/create_op_resolver.h"
#include "tensorflow/lite/delegates/interpreter_utils.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/version.h"

namespace {
class CallbackErrorReporter : public tflite::ErrorReporter {
 public:
  explicit CallbackErrorReporter(TfLiteErrorReporterCallback callback)
      : callback_(callback) {}

  int Report(const char* format, va_list args) override {
    callback_.error_reporter(callback_.user_data, format, args);
    return 0;
  }

 private:
  TfLiteErrorReporterCallback callback_;
};

}  // namespace

extern "C" {

// LINT.IfChange

const char* TfLiteVersion() { return TFLITE_VERSION_STRING; }

TfLiteModel* TfLiteModelCreate(const void* model_data, size_t model_size) {
  auto model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      static_cast<const char*>(model_data), model_size);
  std::shared_ptr<const tflite::FlatBufferModel> shared_model(model.release());
  return shared_model ? new TfLiteModel{std::move(shared_model)} : nullptr;
}

TfLiteModel* TfLiteModelCreateFromFile(const char* model_path) {
  auto model = tflite::FlatBufferModel::VerifyAndBuildFromFile(model_path);
  std::shared_ptr<const tflite::FlatBufferModel> shared_model(model.release());
  return shared_model ? new TfLiteModel{std::move(shared_model)} : nullptr;
}

void TfLiteModelDelete(TfLiteModel* model) { delete model; }

TfLiteRegistrationExternal* TfLiteRegistrationExternalCreate(
    const char* custom_name, const int version) {
  return new TfLiteRegistrationExternal{custom_name, version};
}

void TfLiteRegistrationExternalDelete(TfLiteRegistrationExternal* reg) {
  delete reg;
}

void TfLiteRegistrationExternalSetPrepare(
    TfLiteRegistrationExternal* registration,
    TfLiteStatus (*prepare)(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node)) {
  registration->prepare = prepare;
}

void TfLiteRegistrationExternalSetInvoke(
    TfLiteRegistrationExternal* registration,
    TfLiteStatus (*invoke)(TfLiteOpaqueContext* context,
                           TfLiteOpaqueNode* node)) {
  registration->invoke = invoke;
}

TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate() {
  return new TfLiteInterpreterOptions{};
}

void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions* options) {
  delete options;
}

void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions* options,
                                           int32_t num_threads) {
  options->num_threads = num_threads;
}

void TfLiteInterpreterOptionsAddDelegate(TfLiteInterpreterOptions* options,
                                         TfLiteDelegate* delegate) {
  options->delegates.push_back(delegate);
}

void TfLiteInterpreterOptionsSetErrorReporter(
    TfLiteInterpreterOptions* options,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data) {
  options->error_reporter_callback.error_reporter = reporter;
  options->error_reporter_callback.user_data = user_data;
}

void TfLiteInterpreterOptionsAddRegistrationExternal(
    TfLiteInterpreterOptions* options,
    TfLiteRegistrationExternal* registration) {
  options->op_registrations.push_back(registration);
}

static void InitTfLiteRegistration(
    TfLiteRegistration* registration,
    TfLiteRegistrationExternal* registration_external) {
  registration->custom_name = registration_external->custom_name;
  registration->version = registration_external->version;
  registration->registration_external = registration_external;
}

TfLiteInterpreter* TfLiteInterpreterCreate(
    const TfLiteModel* model,
    const TfLiteInterpreterOptions* optional_options) {
  std::unique_ptr<tflite::MutableOpResolver> resolver =
      tflite::CreateOpResolver();
  return tflite::internal::InterpreterCreateWithOpResolver(
      model, optional_options, resolver.get());
}

void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter) {
  delete interpreter;
}

int32_t TfLiteInterpreterGetInputTensorCount(
    const TfLiteInterpreter* interpreter) {
  return static_cast<int32_t>(interpreter->impl->inputs().size());
}

TfLiteTensor* TfLiteInterpreterGetInputTensor(
    const TfLiteInterpreter* interpreter, int32_t input_index) {
  return interpreter->impl->tensor(interpreter->impl->inputs()[input_index]);
}

TfLiteStatus TfLiteInterpreterResizeInputTensor(TfLiteInterpreter* interpreter,
                                                int32_t input_index,
                                                const int* input_dims,
                                                int32_t input_dims_size) {
  std::vector<int> dims{input_dims, input_dims + input_dims_size};
  return interpreter->impl->ResizeInputTensor(
      interpreter->impl->inputs()[input_index], dims);
}

TfLiteStatus TfLiteInterpreterAllocateTensors(TfLiteInterpreter* interpreter) {
  return interpreter->impl->AllocateTensors();
}

TfLiteStatus TfLiteInterpreterInvoke(TfLiteInterpreter* interpreter) {
  if (interpreter->enable_delegate_fallback) {
    return tflite::delegates::InterpreterUtils::InvokeWithCPUFallback(
        interpreter->impl.get());
  } else {
    return interpreter->impl->Invoke();
  }
}

int32_t TfLiteInterpreterGetOutputTensorCount(
    const TfLiteInterpreter* interpreter) {
  return static_cast<int32_t>(interpreter->impl->outputs().size());
}

const TfLiteTensor* TfLiteInterpreterGetOutputTensor(
    const TfLiteInterpreter* interpreter, int32_t output_index) {
  return interpreter->impl->tensor(interpreter->impl->outputs()[output_index]);
}

TfLiteType TfLiteTensorType(const TfLiteTensor* tensor) { return tensor->type; }

int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor) {
  return tensor->dims->size;
}

int32_t TfLiteTensorDim(const TfLiteTensor* tensor, int32_t dim_index) {
  return tensor->dims->data[dim_index];
}

size_t TfLiteTensorByteSize(const TfLiteTensor* tensor) {
  return tensor->bytes;
}

void* TfLiteTensorData(const TfLiteTensor* tensor) { return tensor->data.raw; }

const char* TfLiteTensorName(const TfLiteTensor* tensor) {
  return tensor->name;
}

TfLiteQuantizationParams TfLiteTensorQuantizationParams(
    const TfLiteTensor* tensor) {
  return tensor->params;
}

TfLiteStatus TfLiteTensorCopyFromBuffer(TfLiteTensor* tensor,
                                        const void* input_data,
                                        size_t input_data_size) {
  if (tensor->bytes != input_data_size) {
    return kTfLiteError;
  }
  memcpy(tensor->data.raw, input_data, input_data_size);
  return kTfLiteOk;
}

TfLiteStatus TfLiteTensorCopyToBuffer(const TfLiteTensor* tensor,
                                      void* output_data,
                                      size_t output_data_size) {
  if (tensor->bytes != output_data_size) {
    return kTfLiteError;
  }
  memcpy(output_data, tensor->data.raw, output_data_size);
  return kTfLiteOk;
}

// LINT.ThenChange(//tensorflow/lite/experimental/examples/unity/TensorFlowLitePlugin/Assets/TensorFlowLite/SDK/Scripts/Interpreter.cs)

}  // extern "C"

namespace tflite {
namespace internal {

// Implementation of CallbackOpResolver class which is defined in
// c_api_internal.h. CallbackOpResolver is a (C++) `tflite::OpResolver` that
// forwards the methods to (C ABI) callback functions from a
// `TfLiteOpResolverCallbacks` struct.

// FindOp for builtin op query.
const TfLiteRegistration* CallbackOpResolver::FindOp(tflite::BuiltinOperator op,
                                                     int version) const {
  // Use Registration V2 API to find op.
  if (op_resolver_callbacks_.find_builtin_op) {
    return op_resolver_callbacks_.find_builtin_op(
        op_resolver_callbacks_.user_data,
        static_cast<TfLiteBuiltinOperator>(op), version);
  }
  if (op_resolver_callbacks_.find_builtin_op_v1) {
    // Check if cached Registration is available.
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& created_registration : temporary_builtin_registrations_) {
      if (created_registration->builtin_code == op &&
          created_registration->version == version) {
        return created_registration.get();
      }
    }
    // Get a Registration V1 object and create a Registration V2 object.
    const TfLiteRegistration_V1* reg_v1 =
        op_resolver_callbacks_.find_builtin_op_v1(
            op_resolver_callbacks_.user_data,
            static_cast<TfLiteBuiltinOperator>(op), version);
    if (reg_v1) {
      TfLiteRegistration* new_registration = new TfLiteRegistration();
      memcpy(new_registration, reg_v1, sizeof(TfLiteRegistration_V1));
      new_registration->registration_external = nullptr;
      temporary_builtin_registrations_.push_back(
          std::unique_ptr<TfLiteRegistration>(new_registration));
      return new_registration;
    }
  }
  return nullptr;
}

// FindOp for custom op query.
const TfLiteRegistration* CallbackOpResolver::FindOp(const char* op,
                                                     int version) const {
  // Use Registration V2 API to find op.
  if (op_resolver_callbacks_.find_custom_op) {
    return op_resolver_callbacks_.find_custom_op(
        op_resolver_callbacks_.user_data, op, version);
  }
  if (op_resolver_callbacks_.find_custom_op_v1) {
    // Check if cached Registration is available.
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& created_registration : temporary_custom_registrations_) {
      if (strcmp(created_registration->custom_name, op) == 0 &&
          created_registration->version == version) {
        return created_registration.get();
      }
    }
    // Get a Registration V1 object and create a Registration V2 object.
    const TfLiteRegistration_V1* reg_v1 =
        op_resolver_callbacks_.find_custom_op_v1(
            op_resolver_callbacks_.user_data, op, version);
    if (reg_v1) {
      TfLiteRegistration* new_registration = new TfLiteRegistration();
      memcpy(new_registration, reg_v1, sizeof(TfLiteRegistration_V1));
      new_registration->registration_external = nullptr;
      temporary_custom_registrations_.push_back(
          std::unique_ptr<TfLiteRegistration>(new_registration));
      return new_registration;
    }
  }
  return nullptr;
}

TfLiteInterpreter* InterpreterCreateWithOpResolver(
    const TfLiteModel* model, const TfLiteInterpreterOptions* optional_options,
    tflite::MutableOpResolver* mutable_resolver) {
  TFLITE_DCHECK_NE(mutable_resolver, nullptr);
  if (!model || !model->impl) {
    return nullptr;
  }

  std::unique_ptr<tflite::ErrorReporter> optional_error_reporter;
  if (optional_options &&
      optional_options->error_reporter_callback.error_reporter != nullptr) {
    optional_error_reporter = std::make_unique<CallbackErrorReporter>(
        optional_options->error_reporter_callback);
  }

  // By default, we use the provided mutable_op_resolver, adding any builtin or
  // custom ops registered with `TfLiteInterpreterOptionsAddBuiltinOp` and/or
  // `TfLiteInterpreterOptionsAddCustomOp`.
  tflite::OpResolver* op_resolver = mutable_resolver;
  if (optional_options) {
    mutable_resolver->AddAll(optional_options->mutable_op_resolver);
    for (auto* registration_external : optional_options->op_registrations) {
      TfLiteRegistration registration{};
      InitTfLiteRegistration(&registration, registration_external);
      mutable_resolver->AddCustom(registration_external->custom_name,
                                  &registration,
                                  registration_external->version);
    }
  }
  // However, if `TfLiteInterpreterOptionsSetOpResolver` has been called with
  // a non-null callback parameter, then we instead use a
  // `CallbackOpResolver` that will forward to the callbacks provided there.
  CallbackOpResolver callback_op_resolver;
  if (optional_options &&
      (optional_options->op_resolver_callbacks.find_builtin_op != nullptr ||
       optional_options->op_resolver_callbacks.find_custom_op != nullptr ||
       optional_options->op_resolver_callbacks.find_builtin_op_v1 != nullptr ||
       optional_options->op_resolver_callbacks.find_custom_op_v1 != nullptr)) {
    callback_op_resolver.SetCallbacks(optional_options->op_resolver_callbacks);
    op_resolver = &callback_op_resolver;
  }

  tflite::ErrorReporter* error_reporter = optional_error_reporter
                                              ? optional_error_reporter.get()
                                              : tflite::DefaultErrorReporter();
  tflite::InterpreterBuilder builder(model->impl->GetModel(), *op_resolver,
                                     error_reporter);

  std::unique_ptr<tflite::Interpreter> interpreter;
  if (builder(&interpreter) != kTfLiteOk) {
    return nullptr;
  }

  if (optional_options) {
    if (optional_options->num_threads !=
        TfLiteInterpreterOptions::kDefaultNumThreads) {
      interpreter->SetNumThreads(optional_options->num_threads);
    }

    if (optional_options->use_nnapi) {
      if (interpreter->ModifyGraphWithDelegate(tflite::NnApiDelegate()) !=
          kTfLiteOk) {
        return nullptr;
      }
    }

    for (auto* delegate : optional_options->delegates) {
      if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        return nullptr;
      }
    }
  }

  bool enable_delegate_fallback =
      optional_options != nullptr && optional_options->enable_delegate_fallback;

  return new TfLiteInterpreter{model->impl, std::move(optional_error_reporter),
                               std::move(interpreter),
                               enable_delegate_fallback};
}

}  // namespace internal
}  // namespace tflite
