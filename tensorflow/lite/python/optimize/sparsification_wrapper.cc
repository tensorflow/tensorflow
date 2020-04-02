/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/python/optimize/sparsification_wrapper.h"

#include <memory>
#include <sstream>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/mlir/lite/sparsity/sparsify_model.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/python/interpreter_wrapper/numpy.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_error_reporter.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"

#define TFLITE_PY_CHECK(x)               \
  if ((x) != kTfLiteOk) {                \
    return error_reporter_->exception(); \
  }

#define TFLITE_PY_ENSURE_VALID_INTERPRETER()                               \
  if (!interpreter_) {                                                     \
    PyErr_SetString(PyExc_ValueError, "Interpreter was not initialized."); \
    return nullptr;                                                        \
  }

namespace tflite {
namespace sparsification_wrapper {

namespace {

std::unique_ptr<tflite::ModelT> CreateMutableModel(const tflite::Model& model) {
  auto copied_model = absl::make_unique<tflite::ModelT>();
  model.UnPackTo(copied_model.get(), nullptr);
  return copied_model;
}

}  // namespace

SparsificationWrapper::SparsificationWrapper(
    std::unique_ptr<tflite::FlatBufferModel> model,
    std::unique_ptr<tflite::interpreter_wrapper::PythonErrorReporter>
        error_reporter)
    : model_(std::move(model)), error_reporter_(std::move(error_reporter)) {}
SparsificationWrapper::~SparsificationWrapper() {}

PyObject* SparsificationWrapper::SparsifyModel() {
  auto tflite_model = CreateMutableModel(*model_->GetModel());
  flatbuffers::FlatBufferBuilder builder;
  auto status = kTfLiteOk;
  status =
      mlir::lite::SparsifyModel(*tflite_model, &builder, error_reporter_.get());

  if (status != kTfLiteOk) {
    error_reporter_->exception();
    return nullptr;
  }

  return python_utils::ConvertToPyString(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
}

/*static*/ SparsificationWrapper*
SparsificationWrapper::CreateWrapperCPPFromBuffer(PyObject* data) {
  using tflite::interpreter_wrapper::PythonErrorReporter;
  char* buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);
  ::tflite::python::ImportNumpy();

  if (python_utils::ConvertFromPyString(data, &buf, &length) == -1) {
    return nullptr;
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(buf, length,
                                               error_reporter.get());
  if (!model) {
    PyErr_Format(PyExc_ValueError, "Invalid model");
    return nullptr;
  }

  auto wrapper =
      new SparsificationWrapper(std::move(model), std::move(error_reporter));
  return wrapper;
}

}  // namespace sparsification_wrapper
}  // namespace tflite
