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

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/version.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

namespace {
class CallbackErrorReporter : public tflite::ErrorReporter {
 public:
  using ErrorCallback = void (*)(void* user_data, const char* format,
                                 va_list args);

  CallbackErrorReporter(ErrorCallback callback, void* user_data)
      : callback_(callback), user_data_(user_data) {}

  int Report(const char* format, va_list args) override {
    callback_(user_data_, format, args);
    return 0;
  }

 private:
  ErrorCallback callback_;
  void* user_data_;
};
}  // namespace

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
  options->error_reporter = reporter;
  options->error_reporter_user_data = user_data;
}

TfLiteInterpreter* TfLiteInterpreterCreate(
    const TfLiteModel* model,
    const TfLiteInterpreterOptions* optional_options) {
  if (!model || !model->impl) {
    return nullptr;
  }

  std::unique_ptr<tflite::ErrorReporter> optional_error_reporter;
  if (optional_options && optional_options->error_reporter != nullptr) {
    optional_error_reporter.reset(
        new CallbackErrorReporter(optional_options->error_reporter,
                                  optional_options->error_reporter_user_data));
  }

  // TODO(b/111881878): Allow use of C API without pulling in all builtin ops.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  if (optional_options) {
    resolver.AddAll(optional_options->op_resolver);
  }
  tflite::ErrorReporter* error_reporter = optional_error_reporter
                                              ? optional_error_reporter.get()
                                              : tflite::DefaultErrorReporter();
  tflite::InterpreterBuilder builder(model->impl->GetModel(), resolver,
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

    for (auto* delegate : optional_options->delegates) {
      if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        return nullptr;
      }
    }
  }

  return new TfLiteInterpreter{model->impl, std::move(optional_error_reporter),
                               std::move(interpreter)};
}

void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter) {
  delete interpreter;
}

int32_t TfLiteInterpreterGetInputTensorCount(
    const TfLiteInterpreter* interpreter) {
  return static_cast<int>(interpreter->impl->inputs().size());
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
  return interpreter->impl->Invoke();
}

int32_t TfLiteInterpreterGetOutputTensorCount(
    const TfLiteInterpreter* interpreter) {
  return static_cast<int>(interpreter->impl->outputs().size());
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

void* TfLiteTensorData(const TfLiteTensor* tensor) {
  return static_cast<void*>(tensor->data.raw);
}

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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
