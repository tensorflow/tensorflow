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
#include "tensorflow/lite/core/c/c_api.h"

#include <memory>
#include <mutex>  // NOLINT
#include <utility>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/c/common_internal.h"
#include "tensorflow/lite/core/create_op_resolver.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/delegates/interpreter_utils.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
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

int TfLiteSchemaVersion() { return TFLITE_SCHEMA_VERSION; }

TfLiteModel* TfLiteModelCreate(const void* model_data, size_t model_size) {
  auto model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      static_cast<const char*>(model_data), model_size);
  std::shared_ptr<const tflite::FlatBufferModel> shared_model(model.release());
  return shared_model ? new TfLiteModel{std::move(shared_model)} : nullptr;
}

TfLiteModel* TfLiteModelCreateWithErrorReporter(
    const void* model_data, size_t model_size,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data) {
  struct TfLiteErrorReporterCallback er_cb = {user_data, reporter};
  auto error_reporter = std::make_unique<CallbackErrorReporter>(er_cb);
  auto model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      static_cast<const char*>(model_data), model_size, nullptr,
      error_reporter.get());
  std::shared_ptr<const tflite::FlatBufferModel> shared_model(model.release());
  return shared_model ? new TfLiteModel{std::move(shared_model)} : nullptr;
}

TfLiteModel* TfLiteModelCreateFromFile(const char* model_path) {
  auto model = tflite::FlatBufferModel::VerifyAndBuildFromFile(model_path);
  std::shared_ptr<const tflite::FlatBufferModel> shared_model(model.release());
  return shared_model ? new TfLiteModel{std::move(shared_model)} : nullptr;
}

TfLiteModel* TfLiteModelCreateFromFileWithErrorReporter(
    const char* model_path,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data) {
  struct TfLiteErrorReporterCallback er_cb = {user_data, reporter};
  auto error_reporter = std::make_unique<CallbackErrorReporter>(er_cb);
  auto model = tflite::FlatBufferModel::VerifyAndBuildFromFile(
      model_path, nullptr, error_reporter.get());
  std::shared_ptr<const tflite::FlatBufferModel> shared_model(model.release());
  return shared_model ? new TfLiteModel{std::move(shared_model)} : nullptr;
}

void TfLiteModelDelete(TfLiteModel* model) { delete model; }

TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate() {
  return new TfLiteInterpreterOptions{};
}

struct TfLiteInterpreterOptions* TfLiteInterpreterOptionsCopy(
    const struct TfLiteInterpreterOptions* from) {
  struct TfLiteInterpreterOptions* copy = new TfLiteInterpreterOptions{};
  *copy = *from;
  return copy;
}

void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions* options) {
  delete options;
}

void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions* options,
                                           int32_t num_threads) {
  options->num_threads = num_threads;
}

void TfLiteInterpreterOptionsAddDelegate(TfLiteInterpreterOptions* options,
                                         TfLiteOpaqueDelegate* delegate) {
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

TfLiteStatus TfLiteInterpreterOptionsEnableCancellation(
    TfLiteInterpreterOptions* options, bool enable) {
  options->enable_cancellation = enable;
  return kTfLiteOk;
}

static void InitTfLiteRegistration(
    TfLiteRegistration* registration,
    TfLiteRegistrationExternal* registration_external) {
  registration->builtin_code = registration_external->builtin_code;
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

const int* TfLiteInterpreterInputTensorIndices(
    const TfLiteInterpreter* interpreter) {
  return interpreter->impl->inputs().data();
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

TfLiteTensor* TfLiteInterpreterGetTensor(const TfLiteInterpreter* interpreter,
                                         int index) {
  return interpreter->impl->tensor(index);
}

const int* TfLiteInterpreterOutputTensorIndices(
    const TfLiteInterpreter* interpreter) {
  return interpreter->impl->outputs().data();
}

const TfLiteTensor* TfLiteInterpreterGetOutputTensor(
    const TfLiteInterpreter* interpreter, int32_t output_index) {
  return interpreter->impl->tensor(interpreter->impl->outputs()[output_index]);
}

TfLiteStatus TfLiteInterpreterCancel(const TfLiteInterpreter* interpreter) {
  return interpreter->impl->Cancel();
}

TfLiteType TfLiteTensorType(const TfLiteTensor* tensor) { return tensor->type; }

int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor) {
  if (!tensor->dims) {
    return -1;
  }
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

TfLiteRegistrationExternal* TfLiteRegistrationExternalCreate(
    TfLiteBuiltinOperator builtin_code, const char* custom_name, int version) {
  return new TfLiteRegistrationExternal{/*.custom_name =*/custom_name,
                                        /*.version =*/version,
                                        /*.init =*/nullptr,
                                        /*.free =*/nullptr,
                                        /*.prepare =*/nullptr,
                                        /*.invoke =*/nullptr,
                                        /*.async_kernel =*/nullptr,
                                        /*.builtin_code =*/builtin_code,
                                        /*.node_index =*/-1};
}

void TfLiteRegistrationExternalDelete(TfLiteRegistrationExternal* reg) {
  delete reg;
}

void TfLiteRegistrationExternalSetInit(
    TfLiteRegistrationExternal* registration,
    void* (*init)(TfLiteOpaqueContext* context, const char* buffer,
                  size_t length)) {
  registration->init = init;
}

void TfLiteRegistrationExternalSetFree(
    TfLiteRegistrationExternal* registration,
    void (*free)(TfLiteOpaqueContext* context, void* data)) {
  registration->free = free;
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

void TfLiteRegistrationExternalSetAsyncKernel(
    TfLiteRegistrationExternal* registration,
    TfLiteAsyncKernel* (*async_kernel)(TfLiteOpaqueContext* context,
                                       TfLiteOpaqueNode* node)) {
  registration->async_kernel = async_kernel;
}

TfLiteBuiltinOperator TfLiteRegistrationExternalGetBuiltInCode(
    const TfLiteRegistrationExternal* registration) {
  return static_cast<TfLiteBuiltinOperator>(registration->builtin_code);
}

int TfLiteRegistrationExternalGetVersion(
    const TfLiteRegistrationExternal* registration) {
  if (!registration) {
    return -1;
  }
  return registration->version;
}

const char* TfLiteRegistrationExternalGetCustomName(
    const TfLiteRegistrationExternal* registration) {
  return registration->custom_name;
}
// LINT.ThenChange(//tensorflow/lite/experimental/examples/unity/TensorFlowLitePlugin/Assets/TensorFlowLite/SDK/Scripts/Interpreter.cs)

}  // extern "C"

namespace tflite {
namespace internal {

static TfLiteRegistration* RegistrationExternalToRegistration(
    const TfLiteRegistrationExternal* registration_external) {
  // All TfLiteRegistrationExternal objects are dynamically allocated via
  // TfLiteRegistrationExternalCreate(), so they are guaranteed
  // to be mutable, hence the const_cast below should be safe.
  auto registration_external_non_const =
      const_cast<TfLiteRegistrationExternal*>(registration_external);
  TfLiteRegistration* new_registration = new TfLiteRegistration{};
  InitTfLiteRegistration(new_registration, registration_external_non_const);
  return new_registration;
}

// Implementation of CallbackOpResolver class which is defined in
// c_api_internal.h. CallbackOpResolver is a (C++) `tflite::OpResolver` that
// forwards the methods to (C ABI) callback functions from a
// `TfLiteOpResolverCallbacks` struct.

// FindOp for builtin op query.
const TfLiteRegistration* CallbackOpResolver::FindOp(tflite::BuiltinOperator op,
                                                     int version) const {
  // Use Registration V3 API to find op.
  if (op_resolver_callbacks_.find_builtin_op) {
    return op_resolver_callbacks_.find_builtin_op(
        op_resolver_callbacks_.user_data,
        static_cast<TfLiteBuiltinOperator>(op), version);
  }

  // Check if cached Registration is available.
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& created_registration : temporary_builtin_registrations_) {
    if (created_registration->builtin_code == op &&
        created_registration->version == version) {
      return created_registration.get();
    }
  }

  if (auto* registration =
          BuildBuiltinOpFromLegacyRegistration<TfLiteRegistration_V2>(
              op, version, op_resolver_callbacks_.find_builtin_op_v2);
      registration) {
    return registration;
  }
  if (auto* registration =
          BuildBuiltinOpFromLegacyRegistration<TfLiteRegistration_V1>(
              op, version, op_resolver_callbacks_.find_builtin_op_v1);
      registration) {
    return registration;
  }
  // Try using newer RegistrationExternal API.
  if (op_resolver_callbacks_.find_builtin_op_external) {
    // Get a RegistrationExternal object and create a Registration (V3) object.
    const TfLiteRegistrationExternal* registration_external =
        op_resolver_callbacks_.find_builtin_op_external(
            op_resolver_callbacks_.user_data,
            static_cast<TfLiteBuiltinOperator>(op), version);
    if (registration_external) {
      TfLiteRegistration* new_registration =
          RegistrationExternalToRegistration(registration_external);
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
  // Use TfLiteRegistration API to find op.
  if (op_resolver_callbacks_.find_custom_op) {
    return op_resolver_callbacks_.find_custom_op(
        op_resolver_callbacks_.user_data, op, version);
  }
  // Check if cached Registration is available.
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& created_registration : temporary_custom_registrations_) {
    if (strcmp(created_registration->custom_name, op) == 0 &&
        created_registration->version == version) {
      return created_registration.get();
    }
  }

  if (auto* registration =
          BuildCustomOpFromLegacyRegistration<TfLiteRegistration_V2>(
              op, version, op_resolver_callbacks_.find_custom_op_v2);
      registration) {
    return registration;
  }
  if (auto* registration =
          BuildCustomOpFromLegacyRegistration<TfLiteRegistration_V1>(
              op, version, op_resolver_callbacks_.find_custom_op_v1);
      registration) {
    return registration;
  }
  if (op_resolver_callbacks_.find_custom_op_external) {
    // Get a RegistrationExternal object and create a Registration (V2) object.
    const TfLiteRegistrationExternal* registration_external =
        op_resolver_callbacks_.find_custom_op_external(
            op_resolver_callbacks_.user_data, op, version);
    if (registration_external) {
      TfLiteRegistration* new_registration =
          RegistrationExternalToRegistration(registration_external);
      temporary_builtin_registrations_.push_back(
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
       optional_options->op_resolver_callbacks.find_custom_op_v1 != nullptr ||
       optional_options->op_resolver_callbacks.find_builtin_op_v2 != nullptr ||
       optional_options->op_resolver_callbacks.find_custom_op_v2 != nullptr ||
       optional_options->op_resolver_callbacks.find_builtin_op_external !=
           nullptr ||
       optional_options->op_resolver_callbacks.find_custom_op_external !=
           nullptr)) {
    callback_op_resolver.SetCallbacks(optional_options->op_resolver_callbacks);
    op_resolver = &callback_op_resolver;
  }

  tflite::ErrorReporter* error_reporter = optional_error_reporter
                                              ? optional_error_reporter.get()
                                              : tflite::DefaultErrorReporter();
  tflite::InterpreterBuilder builder(model->impl->GetModel(), *op_resolver,
                                     error_reporter);

  if (optional_options && optional_options->telemetry_profiler) {
    std::unique_ptr<tflite::telemetry::TelemetryProfiler> profiler;
    profiler.reset(tflite::telemetry::MakeTfLiteTelemetryProfiler(
        optional_options->telemetry_profiler));
    builder.SetTelemetryProfiler(std::move(profiler));
  }

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

    if (optional_options->enable_cancellation) {
      interpreter->EnableCancellation();
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
