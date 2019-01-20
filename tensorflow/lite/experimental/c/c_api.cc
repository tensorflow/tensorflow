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
#include "tensorflow/lite/experimental/c/c_api.h"

#include <memory>

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/experimental/c/c_api_internal.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

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

TFL_Model* TFL_NewModel(const void* model_data, size_t model_size) {
  auto model = tflite::FlatBufferModel::BuildFromBuffer(
      static_cast<const char*>(model_data), model_size);
  std::shared_ptr<const tflite::FlatBufferModel> shared_model(model.release());
  return shared_model ? new TFL_Model{std::move(shared_model)} : nullptr;
}

TFL_Model* TFL_NewModelFromFile(const char* model_path) {
  auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
  std::shared_ptr<const tflite::FlatBufferModel> shared_model(model.release());
  return shared_model ? new TFL_Model{std::move(shared_model)} : nullptr;
}

void TFL_DeleteModel(TFL_Model* model) { delete model; }

TFL_InterpreterOptions* TFL_NewInterpreterOptions() {
  return new TFL_InterpreterOptions{};
}

void TFL_DeleteInterpreterOptions(TFL_InterpreterOptions* options) {
  delete options;
}

void TFL_InterpreterOptionsSetNumThreads(TFL_InterpreterOptions* options,
                                         int32_t num_threads) {
  options->num_threads = num_threads;
}

TFL_CAPI_EXPORT extern void TFL_InterpreterOptionsSetErrorReporter(
    TFL_InterpreterOptions* options,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data) {
  options->error_reporter = reporter;
  options->error_reporter_user_data = user_data;
}

TFL_Interpreter* TFL_NewInterpreter(
    const TFL_Model* model, const TFL_InterpreterOptions* optional_options) {
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
        TFL_InterpreterOptions::kDefaultNumThreads) {
      interpreter->SetNumThreads(optional_options->num_threads);
    }
  }

  return new TFL_Interpreter{model->impl, std::move(optional_error_reporter),
                             std::move(interpreter)};
}

void TFL_DeleteInterpreter(TFL_Interpreter* interpreter) { delete interpreter; }

int32_t TFL_InterpreterGetInputTensorCount(const TFL_Interpreter* interpreter) {
  return static_cast<int>(interpreter->impl->inputs().size());
}

TFL_Tensor* TFL_InterpreterGetInputTensor(const TFL_Interpreter* interpreter,
                                          int32_t input_index) {
  return interpreter->impl->tensor(interpreter->impl->inputs()[input_index]);
}

TFL_Status TFL_InterpreterResizeInputTensor(TFL_Interpreter* interpreter,
                                            int32_t input_index,
                                            const int* input_dims,
                                            int32_t input_dims_size) {
  std::vector<int> dims{input_dims, input_dims + input_dims_size};
  return interpreter->impl->ResizeInputTensor(
      interpreter->impl->inputs()[input_index], dims);
}

TFL_Status TFL_InterpreterAllocateTensors(TFL_Interpreter* interpreter) {
  return interpreter->impl->AllocateTensors();
}

TFL_Status TFL_InterpreterInvoke(TFL_Interpreter* interpreter) {
  return interpreter->impl->Invoke();
}

int32_t TFL_InterpreterGetOutputTensorCount(
    const TFL_Interpreter* interpreter) {
  return static_cast<int>(interpreter->impl->outputs().size());
}

const TFL_Tensor* TFL_InterpreterGetOutputTensor(
    const TFL_Interpreter* interpreter, int32_t output_index) {
  return interpreter->impl->tensor(interpreter->impl->outputs()[output_index]);
}

TFL_Type TFL_TensorType(const TFL_Tensor* tensor) { return tensor->type; }

int32_t TFL_TensorNumDims(const TFL_Tensor* tensor) {
  return tensor->dims->size;
}

int32_t TFL_TensorDim(const TFL_Tensor* tensor, int32_t dim_index) {
  return tensor->dims->data[dim_index];
}

size_t TFL_TensorByteSize(const TFL_Tensor* tensor) { return tensor->bytes; }

void* TFL_TensorData(const TFL_Tensor* tensor) {
  return static_cast<void*>(tensor->data.raw);
}

const char* TFL_TensorName(const TFL_Tensor* tensor) { return tensor->name; }

TFL_QuantizationParams TFL_TensorQuantizationParams(const TFL_Tensor* tensor) {
  return tensor->params;
}

TFL_Status TFL_TensorCopyFromBuffer(TFL_Tensor* tensor, const void* input_data,
                                    size_t input_data_size) {
  if (tensor->bytes != input_data_size) {
    return kTfLiteError;
  }
  memcpy(tensor->data.raw, input_data, input_data_size);
  return kTfLiteOk;
}

TFL_Status TFL_TensorCopyToBuffer(const TFL_Tensor* tensor, void* output_data,
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
