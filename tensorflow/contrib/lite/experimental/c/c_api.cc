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
#include "tensorflow/contrib/lite/experimental/c/c_api.h"

#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct _TFL_Interpreter {
  std::unique_ptr<tflite::Interpreter> impl;
};

// LINT.IfChange

TFL_Interpreter* TFL_NewInterpreter(const void* model_data,
                                    int32_t model_size) {
  auto model = tflite::FlatBufferModel::BuildFromBuffer(
      static_cast<const char*>(model_data), static_cast<size_t>(model_size));
  if (!model) {
    return nullptr;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter_impl;
  if (builder(&interpreter_impl) != kTfLiteOk) {
    return nullptr;
  }

  return new TFL_Interpreter{std::move(interpreter_impl)};
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

TFL_Status TFL_TensorCopyFromBuffer(TFL_Tensor* tensor, const void* input_data,
                                    int32_t input_data_size) {
  if (tensor->bytes != static_cast<size_t>(input_data_size)) {
    return kTfLiteError;
  }
  memcpy(tensor->data.raw, input_data, input_data_size);
  return kTfLiteOk;
}

TFL_Status TFL_TensorCopyToBuffer(const TFL_Tensor* tensor, void* output_data,
                                  int32_t output_data_size) {
  if (tensor->bytes != static_cast<size_t>(output_data_size)) {
    return kTfLiteError;
  }
  memcpy(output_data, tensor->data.raw, output_data_size);
  return kTfLiteOk;
}

// LINT.ThenChange(//tensorflow/contrib/lite/experimental/examples/unity/TensorFlowLitePlugin/Assets/TensorFlowLite/SDK/Scripts/Interpreter.cs)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
