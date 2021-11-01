/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/python/optimize/calibration_wrapper.h"

#include <memory>
#include <sstream>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/python/interpreter_wrapper/numpy.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_error_reporter.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"
#include "tensorflow/lite/shared_library.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_reader.h"
#include "tensorflow/lite/tools/optimize/calibration/calibrator.h"
#include "tensorflow/lite/tools/optimize/quantization_wrapper_utils.h"
#include "tensorflow/lite/tools/optimize/quantize_model.h"

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
namespace calibration_wrapper {

namespace {

using python_utils::PyDecrefDeleter;

std::unique_ptr<tflite::ModelT> CreateMutableModel(const tflite::Model& model) {
  auto copied_model = absl::make_unique<tflite::ModelT>();
  model.UnPackTo(copied_model.get(), nullptr);
  return copied_model;
}

bool NoOpModel(const tflite::FlatBufferModel& model) {
  return model->subgraphs()->size() == 1 &&
         (!model->subgraphs()->begin()->operators() ||
          model->subgraphs()->begin()->operators()->size() == 0);
}

inline TensorType TfLiteTypeToSchemaType(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return TensorType_FLOAT32;  // TODO(b/129336260): No schema type for none.
    case kTfLiteFloat32:
      return TensorType_FLOAT32;
    case kTfLiteFloat16:
      return TensorType_FLOAT16;
    case kTfLiteFloat64:
      return TensorType_FLOAT64;
    case kTfLiteInt32:
      return TensorType_INT32;
    case kTfLiteUInt32:
      return TensorType_UINT32;
    case kTfLiteUInt8:
      return TensorType_UINT8;
    case kTfLiteInt8:
      return TensorType_INT8;
    case kTfLiteInt64:
      return TensorType_INT64;
    case kTfLiteUInt64:
      return TensorType_UINT64;
    case kTfLiteString:
      return TensorType_STRING;
    case kTfLiteBool:
      return TensorType_BOOL;
    case kTfLiteInt16:
      return TensorType_INT16;
    case kTfLiteComplex64:
      return TensorType_COMPLEX64;
    case kTfLiteComplex128:
      return TensorType_COMPLEX128;
    case kTfLiteResource:
      return TensorType_RESOURCE;
    case kTfLiteVariant:
      return TensorType_VARIANT;
  }
  // No default to get compiler error when new type is introduced.
}

bool RegisterCustomOpByName(const char* registerer_name,
                            tflite::MutableOpResolver* resolver) {
  // Registerer functions take a pointer to a BuiltinOpResolver as an input
  // parameter and return void.
  // TODO(b/137576229): We should implement this functionality in a more
  // principled way.
  typedef void (*RegistererFunctionType)(tflite::MutableOpResolver*);

  // Look for the Registerer function by name.
  RegistererFunctionType registerer = reinterpret_cast<RegistererFunctionType>(
      SharedLibrary::GetSymbol(registerer_name));

  // Fail in an informative way if the function was not found.
  if (registerer == nullptr) {
    PyErr_Format(PyExc_ValueError,
                 "Looking up symbol '%s' failed with error '%s'.",
                 registerer_name, SharedLibrary::GetError());
    return false;
  }

  // Call the registerer with the resolver.
  registerer(resolver);
  return true;
}

// Returns the dimension from the stored list in the PyObject. If the given
// PyObject is not a list, it will return absl::optional and set the Python
// error message to notify users.
absl::optional<std::vector<int>> ConvertInputShapeToVector(
    PyObject* input_shapes, size_t index) {
  PyObject* shape = PyList_GetItem(input_shapes, index);
  if (!shape || !PyList_Check(shape)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid %ld input shape: expected to be a list.", index);
    return absl::nullopt;
  }
  size_t size = PyList_Size(shape);
  std::vector<int> dims(size);
  for (size_t dim_index = 0; dim_index < size; ++dim_index) {
    PyObject* dim = PyList_GetItem(shape, dim_index);
    dims[dim_index] = PyLong_AsLong(dim);
  }
  return dims;
}

}  // namespace

PyObject* AddIntermediateTensors(PyObject* data) {
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
  flatbuffers::FlatBufferBuilder builder;
  auto tflite_model = CreateMutableModel(*model->GetModel());
  if (optimize::AddIntermediateTensorsToFusedOp(&builder, tflite_model.get()) !=
      kTfLiteOk) {
    error_reporter->exception();
    return nullptr;
  }

  if (builder.GetSize()) {
    return python_utils::ConvertToPyString(
        reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
        builder.GetSize());
  } else {
    // When AddIntermediateTensorsToFusedOp early returns, return the model as
    // it is.
    return python_utils::ConvertToPyString(buf, length);
  }
}

CalibrationWrapper::CalibrationWrapper(
    std::unique_ptr<tflite::Interpreter> interpreter,
    std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver,
    std::unique_ptr<tflite::interpreter_wrapper::PythonErrorReporter>
        error_reporter,
    std::unique_ptr<tflite::FlatBufferModel> model,
    std::unique_ptr<tflite::optimize::calibration::CalibrationReader> reader,
    std::unique_ptr<std::string> model_str)
    : interpreter_(std::move(interpreter)),
      error_reporter_(std::move(error_reporter)),
      resolver_(std::move(resolver)),
      model_(std::move(model)),
      reader_(std::move(reader)),
      model_str_(std::move(model_str)) {}

CalibrationWrapper::~CalibrationWrapper() {}

PyObject* CalibrationWrapper::Prepare() {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->AllocateTensors());
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::Prepare(std::string signature_key) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  SignatureRunner* runner =
      interpreter_->GetSignatureRunner(signature_key.c_str());
  if (runner == nullptr) {
    PyErr_Format(PyExc_ValueError, "Invalid signature key: %s",
                 signature_key.c_str());
    return nullptr;
  }
  TFLITE_PY_CHECK(runner->AllocateTensors());
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::Prepare(PyObject* input_shapes,
                                      std::string signature_key) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (!PyList_Check(input_shapes)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input shapes: expected shapes to be a list.");
    return nullptr;
  }
  const int subgraph_index =
      interpreter_->GetSubgraphIndexFromSignature(signature_key.c_str());
  if (subgraph_index == -1) {
    PyErr_Format(PyExc_ValueError, "Invalid signature key: %s",
                 signature_key.c_str());
    return nullptr;
  }
  auto* subgraph = interpreter_->subgraph(subgraph_index);

  const size_t inputs_size = PyList_Size(input_shapes);
  if (inputs_size != subgraph->inputs().size()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input shapes: expected %ld items got %ld items.",
                 subgraph->inputs().size(), inputs_size);
    return nullptr;
  }

  for (size_t i = 0; i < inputs_size; ++i) {
    absl::optional<std::vector<int>> dims =
        ConvertInputShapeToVector(input_shapes, i);
    if (!dims.has_value()) {
      return nullptr;
    }
    int input_tensor_idx = subgraph->inputs()[i];
    if (subgraph->ResizeInputTensor(input_tensor_idx, *dims) != kTfLiteOk) {
      PyErr_Format(PyExc_ValueError, "Failed to resize %ld input tensor.", i);
      return nullptr;
    }
  }

  return Prepare(signature_key);
}

PyObject* CalibrationWrapper::Prepare(PyObject* input_shapes) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (!PyList_Check(input_shapes)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input shapes: expected shapes to be a list.");
    return nullptr;
  }

  const size_t inputs_size = PyList_Size(input_shapes);
  if (inputs_size != interpreter_->inputs().size()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input shapes: expected %ld items got %ld items.",
                 interpreter_->inputs().size(), inputs_size);
    return nullptr;
  }

  for (size_t i = 0; i < inputs_size; ++i) {
    absl::optional<std::vector<int>> dims =
        ConvertInputShapeToVector(input_shapes, i);
    if (!dims.has_value()) {
      return nullptr;
    }
    int input_tensor_idx = interpreter_->inputs()[i];
    if (interpreter_->ResizeInputTensor(input_tensor_idx, *dims) != kTfLiteOk) {
      PyErr_Format(PyExc_ValueError, "Failed to resize %ld input tensor.", i);
      return nullptr;
    }
  }

  return Prepare();
}

PyObject* CalibrationWrapper::FeedTensor(PyObject* input_value,
                                         std::string signature_key) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (!PyList_Check(input_value)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input type: expected input to be a list.");
    return nullptr;
  }
  const int subgraph_index =
      interpreter_->GetSubgraphIndexFromSignature(signature_key.c_str());
  if (subgraph_index == -1) {
    PyErr_Format(PyExc_ValueError, "Invalid signature key: %s",
                 signature_key.c_str());
    return nullptr;
  }
  const size_t inputs_size = PyList_Size(input_value);

  auto* subgraph = interpreter_->subgraph(subgraph_index);
  if (inputs_size != subgraph->inputs().size()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input size: expected %ld items got %ld items.",
                 subgraph->inputs().size(), inputs_size);
    return nullptr;
  }

  for (size_t i = 0; i < inputs_size; ++i) {
    PyObject* input = PyList_GetItem(input_value, i);
    if (!input) {
      return nullptr;
    }
    int input_tensor_idx = subgraph->inputs()[i];
    if (!SetTensor(input_tensor_idx, input, signature_key)) {
      return nullptr;
    }
  }

  TFLITE_PY_CHECK(subgraph->Invoke());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::FeedTensor(PyObject* input_value) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (!PyList_Check(input_value)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input type: expected input to be a list.");
    return nullptr;
  }

  const size_t inputs_size = PyList_Size(input_value);

  if (inputs_size != interpreter_->inputs().size()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input size: expected %ld items got %ld items.",
                 interpreter_->inputs().size(), inputs_size);
    return nullptr;
  }

  for (size_t i = 0; i < inputs_size; ++i) {
    PyObject* input = PyList_GetItem(input_value, i);
    if (!input) {
      return nullptr;
    }
    int input_tensor_idx = interpreter_->inputs()[i];
    if (!SetTensor(input_tensor_idx, input)) {
      return nullptr;
    }
  }

  TFLITE_PY_CHECK(interpreter_->Invoke());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::SetTensor(int index, PyObject* value,
                                        std::string signature_key) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert value into readable tensor.");
    return nullptr;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());

  const int subgraph_index =
      interpreter_->GetSubgraphIndexFromSignature(signature_key.c_str());
  if (subgraph_index == -1) {
    PyErr_Format(PyExc_ValueError, "Invalid signature key: %s",
                 signature_key.c_str());
    return nullptr;
  }
  auto* subgraph = interpreter_->subgraph(subgraph_index);
  const TfLiteTensor* tensor = subgraph->tensor(index);

  if (python_utils::TfLiteTypeFromPyArray(array) != tensor->type) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: "
                 "Got value of type %s "
                 "but expected type %s for input %d, name: %s ",
                 TfLiteTypeGetName(python_utils::TfLiteTypeFromPyArray(array)),
                 TfLiteTypeGetName(tensor->type), index, tensor->name);
    return nullptr;
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: Dimension count mismatch, expected %d "
                 "but found %d",
                 tensor->dims->size, PyArray_NDIM(array));
    return nullptr;
  }

  std::vector<int> dims(PyArray_NDIM(array));
  bool has_unknown_dims = false;
  for (int j = 0; j < PyArray_NDIM(array); ++j) {
    // Ensure the calibration data input shape is the same as the model input
    // shape unless the dimension is unknown.
    if (tensor->dims_signature != nullptr &&
        tensor->dims_signature->size == tensor->dims->size &&
        tensor->dims_signature->data[j] == -1) {
      has_unknown_dims = true;
    } else if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor: Size mismatch, expected %d for dim "
                   "%d but found %ld",
                   tensor->dims->data[j], j, PyArray_SHAPE(array)[j]);
      return nullptr;
    }
    dims[j] = PyArray_SHAPE(array)[j];
  }

  // Resize the input tensor if there are unknown dimensions.
  if (has_unknown_dims) {
    // Does strict checking on the `ResizeInputTensor` call.
    TFLITE_PY_CHECK(subgraph->ResizeInputTensorStrict(index, dims));
    TFLITE_PY_CHECK(subgraph->AllocateTensors());
  }

  // Re-read the updated tensor after the allocation is done.
  tensor = subgraph->tensor(index);

  size_t size = PyArray_NBYTES(array);

  if (tensor->type == kTfLiteString) {
    tflite::DynamicBuffer buffer;
    buffer.AddString(reinterpret_cast<const char*>(PyArray_BYTES(array)), size);
    buffer.WriteToTensor(subgraph->tensor(index), /*new_shape=*/nullptr);
    Py_RETURN_NONE;
  }

  if (size != tensor->bytes) {
    PyErr_Format(PyExc_ValueError,
                 "numpy array had %zu bytes but expected %zu bytes.", size,
                 tensor->bytes);
    return nullptr;
  }
  memcpy(tensor->data.raw, PyArray_DATA(array), size);
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::SetTensor(int index, PyObject* value) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();

  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert value into readable tensor.");
    return nullptr;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());
  const TfLiteTensor* tensor = interpreter_->tensor(index);

  if (python_utils::TfLiteTypeFromPyArray(array) != tensor->type) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: "
                 "Got value of type %s "
                 "but expected type %s for input %d, name: %s ",
                 TfLiteTypeGetName(python_utils::TfLiteTypeFromPyArray(array)),
                 TfLiteTypeGetName(tensor->type), index, tensor->name);
    return nullptr;
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    PyErr_Format(
        PyExc_ValueError,
        "Cannot set tensor: Dimension count mismatch, expected %d but found %d",
        tensor->dims->size, PyArray_NDIM(array));
    return nullptr;
  }

  std::vector<int> dims(PyArray_NDIM(array));
  bool has_unknown_dims = false;
  for (int j = 0; j < PyArray_NDIM(array); ++j) {
    // Ensure the calibration data input shape is the same as the model input
    // shape unless the dimension is unknown.
    if (tensor->dims_signature != nullptr &&
        tensor->dims_signature->size == tensor->dims->size &&
        tensor->dims_signature->data[j] == -1) {
      has_unknown_dims = true;
    } else if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor: Size mismatch, expected %d for dim "
                   "%d but found %ld",
                   tensor->dims->data[j], j, PyArray_SHAPE(array)[j]);
      return nullptr;
    }
    dims[j] = PyArray_SHAPE(array)[j];
  }

  // Resize the input tensor if there are unknown dimensions.
  if (has_unknown_dims) {
    // Does strict checking on the `ResizeInputTensor` call.
    TFLITE_PY_CHECK(interpreter_->ResizeInputTensorStrict(index, dims));
    TFLITE_PY_CHECK(interpreter_->AllocateTensors());
  }

  // Re-read the updated tensor after the allocation is done.
  tensor = interpreter_->tensor(index);

  size_t size = PyArray_NBYTES(array);

  if (tensor->type == kTfLiteString) {
    tflite::DynamicBuffer buffer;
    buffer.AddString(reinterpret_cast<const char*>(PyArray_BYTES(array)), size);
    buffer.WriteToTensor(interpreter_->tensor(index), /*new_shape=*/nullptr);
    Py_RETURN_NONE;
  }

  if (size != tensor->bytes) {
    PyErr_Format(PyExc_ValueError,
                 "numpy array had %zu bytes but expected %zu bytes.", size,
                 tensor->bytes);
    return nullptr;
  }
  memcpy(tensor->data.raw, PyArray_DATA(array), size);
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::Calibrate() {
  auto tflite_model = CreateMutableModel(*model_->GetModel());
  reader_->AddCalibrationToModel(tflite_model.get(), /*update=*/false);
  flatbuffers::FlatBufferBuilder builder;
  auto loc = tflite::Model::Pack(builder, tflite_model.get());
  tflite::FinishModelBuffer(builder, loc);
  return python_utils::ConvertToPyString(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
}

PyObject* CalibrationWrapper::QuantizeModel(int input_py_type,
                                            int output_py_type,
                                            bool allow_float,
                                            int activations_py_type) {
  return QuantizeModel(input_py_type, output_py_type, allow_float,
                       activations_py_type, /*disable_per_channel=*/false);
}

PyObject* CalibrationWrapper::QuantizeModel(int input_py_type,
                                            int output_py_type,
                                            bool allow_float,
                                            int activations_py_type,
                                            bool disable_per_channel) {
  if (NoOpModel(*model_)) {
    return python_utils::ConvertToPyString(model_str_->data(),
                                           model_str_->size());
  }

  TfLiteType input_type = python_utils::TfLiteTypeFromPyType(input_py_type);
  TfLiteType output_type = python_utils::TfLiteTypeFromPyType(output_py_type);
  TfLiteType activations_type =
      python_utils::TfLiteTypeFromPyType(activations_py_type);

  if (input_type == kTfLiteNoType || output_type == kTfLiteNoType) {
    PyErr_SetString(PyExc_ValueError,
                    "Input/output type cannot be kTfLiteNoType");
    return nullptr;
  }
  auto tflite_model = CreateMutableModel(*model_->GetModel());
  reader_->AddCalibrationToModel(tflite_model.get(), /*update=*/false);
  flatbuffers::FlatBufferBuilder builder;
  auto status = kTfLiteOk;

  status = tflite::optimize::QuantizeModelAllOperators(
      &builder, tflite_model.get(), TfLiteTypeToSchemaType(input_type),
      TfLiteTypeToSchemaType(output_type), allow_float,
      TfLiteTypeToSchemaType(activations_type), disable_per_channel,
      error_reporter_.get());

  if (status != kTfLiteOk) {
    error_reporter_->exception();
    return nullptr;
  }

  return python_utils::ConvertToPyString(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
}

PyObject* CalibrationWrapper::QuantizeModel(int input_py_type,
                                            int output_py_type,
                                            bool allow_float,
                                            const char* operator_output_name) {
  string op_name = std::string(operator_output_name);

  TfLiteType input_type = python_utils::TfLiteTypeFromPyType(input_py_type);
  TfLiteType output_type = python_utils::TfLiteTypeFromPyType(output_py_type);
  if (input_type == kTfLiteNoType || output_type == kTfLiteNoType) {
    PyErr_SetString(PyExc_ValueError,
                    "Input/output type cannot be kTfLiteNoType");
    return nullptr;
  }
  auto tflite_model = CreateMutableModel(*model_->GetModel());
  reader_->AddCalibrationToModel(tflite_model.get(), /*update=*/false);
  flatbuffers::FlatBufferBuilder builder;
  auto status = tflite::optimize::QuantizeModel(
      &builder, tflite_model.get(), TfLiteTypeToSchemaType(input_type),
      TfLiteTypeToSchemaType(output_type), allow_float, {op_name},
      TensorType_INT8, error_reporter_.get());
  if (status != kTfLiteOk) {
    error_reporter_->exception();
    return nullptr;
  }

  return python_utils::ConvertToPyString(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
}

/*static*/ CalibrationWrapper* CalibrationWrapper::CreateWrapperCPPFromBuffer(
    PyObject* data, const std::vector<std::string>& registerers_by_name,
    const std::vector<std::function<void(uintptr_t)>>& registerers_by_func,
    std::string* error_msg) {
  using tflite::interpreter_wrapper::PythonErrorReporter;
  char* buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);
  ::tflite::python::ImportNumpy();

  if (python_utils::ConvertFromPyString(data, &buf, &length) == -1) {
    *error_msg = "Failed to convert from python string";
    return nullptr;
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(buf, length,
                                               error_reporter.get());
  if (!model) {
    *error_msg = "Invalid model";
    return nullptr;
  }
  auto resolver = absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
  for (const auto& registerer : registerers_by_name) {
    if (!RegisterCustomOpByName(registerer.c_str(), resolver.get())) {
      *error_msg =
          absl::StrFormat("Looking up symbol '%s' failed with error '%s'.",
                          registerer.c_str(), SharedLibrary::GetError());
      return nullptr;
    }
  }
  for (const auto& registerer : registerers_by_func) {
    registerer(reinterpret_cast<uintptr_t>(resolver.get()));
  }
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<tflite::optimize::calibration::CalibrationReader> reader;
  auto status = tflite::optimize::calibration::BuildLoggingInterpreter(
      *model, *resolver, &interpreter, &reader);
  if (status != kTfLiteOk) {
    *error_msg = error_reporter->message();
    return nullptr;
  }

  auto model_str = std::make_unique<std::string>(buf, length);
  // If we are not going to use this string during quantization, reset the
  // pointer and release the memory.
  if (!NoOpModel(*model)) {
    model_str.reset();
  }

  auto wrapper = new CalibrationWrapper(
      std::move(interpreter), std::move(resolver), std::move(error_reporter),
      std::move(model), std::move(reader), std::move(model_str));
  return wrapper;
}

}  // namespace calibration_wrapper
}  // namespace tflite
