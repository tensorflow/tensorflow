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
#include "tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.h"

#include <sstream>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_error_reporter.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"

// Disallow Numpy 1.7 deprecated symbols.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#if PY_MAJOR_VERSION >= 3
#define PY_TO_CPPSTRING PyBytes_AsStringAndSize
#define CPP_TO_PYSTRING PyBytes_FromStringAndSize
#else
#define PY_TO_CPPSTRING PyString_AsStringAndSize
#define CPP_TO_PYSTRING PyString_FromStringAndSize
#endif

#define TFLITE_PY_CHECK(x)               \
  if ((x) != kTfLiteOk) {                \
    return error_reporter_->exception(); \
  }

#define TFLITE_PY_TENSOR_BOUNDS_CHECK(i)                                    \
  if (i >= interpreter_->tensors_size() || i < 0) {                         \
    PyErr_Format(PyExc_ValueError,                                          \
                 "Invalid tensor index %d exceeds max tensor index %lu", i, \
                 interpreter_->tensors_size());                             \
    return nullptr;                                                         \
  }

#define TFLITE_PY_ENSURE_VALID_INTERPRETER()                               \
  if (!interpreter_) {                                                     \
    PyErr_SetString(PyExc_ValueError, "Interpreter was not initialized."); \
    return nullptr;                                                        \
  }

namespace tflite {
namespace interpreter_wrapper {

namespace {

// Calls PyArray's initialization to initialize all the API pointers. Note that
// this usage implies only this translation unit can use the pointers. See
// tensorflow/python/core/numpy.cc for a strategy if we ever need to extend
// this further.
void ImportNumpy() { import_array1(); }

std::unique_ptr<tflite::Interpreter> CreateInterpreter(
    const tflite::FlatBufferModel* model,
    const tflite::ops::builtin::BuiltinOpResolver& resolver) {
  if (!model) {
    return nullptr;
  }

  ImportNumpy();

  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
    return nullptr;
  }
  return interpreter;
}

struct PyDecrefDeleter {
  void operator()(PyObject* p) const { Py_DECREF(p); }
};

PyObject* PyArrayFromIntVector(const int* data, npy_intp size) {
  void* pydata = malloc(size * sizeof(int));
  memcpy(pydata, data, size * sizeof(int));
  return PyArray_SimpleNewFromData(1, &size, NPY_INT32, pydata);
}

PyObject* PyTupleFromQuantizationParam(const TfLiteQuantizationParams& param) {
  PyObject* result = PyTuple_New(2);
  PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(param.scale));
  PyTuple_SET_ITEM(result, 1, PyLong_FromLong(param.zero_point));
  return result;
}

}  // namespace

InterpreterWrapper* InterpreterWrapper::CreateInterpreterWrapper(
    std::unique_ptr<tflite::FlatBufferModel> model,
    std::unique_ptr<PythonErrorReporter> error_reporter,
    std::string* error_msg) {
  if (!model) {
    *error_msg = error_reporter->message();
    return nullptr;
  }

  auto resolver = absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
  auto interpreter = CreateInterpreter(model.get(), *resolver);
  if (!interpreter) {
    *error_msg = error_reporter->message();
    return nullptr;
  }

  InterpreterWrapper* wrapper =
      new InterpreterWrapper(std::move(model), std::move(error_reporter),
                             std::move(resolver), std::move(interpreter));
  return wrapper;
}

InterpreterWrapper::InterpreterWrapper(
    std::unique_ptr<tflite::FlatBufferModel> model,
    std::unique_ptr<PythonErrorReporter> error_reporter,
    std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver,
    std::unique_ptr<tflite::Interpreter> interpreter)
    : model_(std::move(model)),
      error_reporter_(std::move(error_reporter)),
      resolver_(std::move(resolver)),
      interpreter_(std::move(interpreter)) {}

InterpreterWrapper::~InterpreterWrapper() {}

PyObject* InterpreterWrapper::AllocateTensors() {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->AllocateTensors());
  Py_RETURN_NONE;
}

PyObject* InterpreterWrapper::Invoke() {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->Invoke());
  Py_RETURN_NONE;
}

PyObject* InterpreterWrapper::InputIndices() const {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  PyObject* np_array = PyArrayFromIntVector(interpreter_->inputs().data(),
                                            interpreter_->inputs().size());

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* InterpreterWrapper::OutputIndices() const {
  PyObject* np_array = PyArrayFromIntVector(interpreter_->outputs().data(),
                                            interpreter_->outputs().size());

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* InterpreterWrapper::ResizeInputTensor(int i, PyObject* value) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();

  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert numpy value into readable tensor.");
    return nullptr;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());

  if (PyArray_NDIM(array) != 1) {
    PyErr_Format(PyExc_ValueError, "Shape should be 1D instead of %d.",
                 PyArray_NDIM(array));
    return nullptr;
  }

  if (PyArray_TYPE(array) != NPY_INT32) {
    PyErr_Format(PyExc_ValueError, "Shape must be type int32 (was %d).",
                 PyArray_TYPE(array));
    return nullptr;
  }

  std::vector<int> dims(PyArray_SHAPE(array)[0]);
  memcpy(dims.data(), PyArray_BYTES(array), dims.size() * sizeof(int));

  TFLITE_PY_CHECK(interpreter_->ResizeInputTensor(i, dims));
  Py_RETURN_NONE;
}

int InterpreterWrapper::NumTensors() const {
  if (!interpreter_) {
    return 0;
  }
  return interpreter_->tensors_size();
}

std::string InterpreterWrapper::TensorName(int i) const {
  if (!interpreter_ || i >= interpreter_->tensors_size() || i < 0) {
    return "";
  }

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  return tensor->name ? tensor->name : "";
}

PyObject* InterpreterWrapper::TensorType(int i) const {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  if (tensor->type == kTfLiteNoType) {
    PyErr_Format(PyExc_ValueError, "Tensor with no type found.");
    return nullptr;
  }

  int code = python_utils::TfLiteTypeToPyArrayType(tensor->type);
  if (code == -1) {
    PyErr_Format(PyExc_ValueError, "Invalid tflite type code %d", code);
    return nullptr;
  }
  return PyArray_TypeObjectFromType(code);
}

PyObject* InterpreterWrapper::TensorSize(int i) const {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  if (tensor->dims == nullptr) {
    PyErr_Format(PyExc_ValueError, "Tensor with no shape found.");
    return nullptr;
  }
  PyObject* np_array =
      PyArrayFromIntVector(tensor->dims->data, tensor->dims->size);

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* InterpreterWrapper::TensorQuantization(int i) const {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);
  const TfLiteTensor* tensor = interpreter_->tensor(i);
  return PyTupleFromQuantizationParam(tensor->params);
}

PyObject* InterpreterWrapper::SetTensor(int i, PyObject* value) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);

  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert value into readable tensor.");
    return nullptr;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());
  const TfLiteTensor* tensor = interpreter_->tensor(i);

  if (python_utils::TfLiteTypeFromPyArray(array) != tensor->type) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor:"
                 " Got tensor of type %d"
                 " but expected type %d for input %d ",
                 python_utils::TfLiteTypeFromPyArray(array), tensor->type, i);
    return nullptr;
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    PyErr_SetString(PyExc_ValueError, "Cannot set tensor: Dimension mismatch");
    return nullptr;
  }

  for (int j = 0; j < PyArray_NDIM(array); j++) {
    if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      PyErr_SetString(PyExc_ValueError,
                      "Cannot set tensor: Dimension mismatch");
      return nullptr;
    }
  }

  size_t size = PyArray_NBYTES(array);
  if (size != tensor->bytes) {
    PyErr_Format(PyExc_ValueError,
                 "numpy array had %zu bytes but expected %zu bytes.", size,
                 tensor->bytes);
    return nullptr;
  }
  memcpy(tensor->data.raw, PyArray_DATA(array), size);
  Py_RETURN_NONE;
}

namespace {

// Checks to see if a tensor access can succeed (returns nullptr on error).
// Otherwise returns Py_None.
PyObject* CheckGetTensorArgs(Interpreter* interpreter_, int tensor_index,
                             TfLiteTensor** tensor, int* type_num) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(tensor_index);

  *tensor = interpreter_->tensor(tensor_index);
  if ((*tensor)->bytes == 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid tensor size.");
    return nullptr;
  }

  *type_num = python_utils::TfLiteTypeToPyArrayType((*tensor)->type);
  if (*type_num == -1) {
    PyErr_SetString(PyExc_ValueError, "Unknown tensor type.");
    return nullptr;
  }

  if (!(*tensor)->data.raw) {
    PyErr_SetString(PyExc_ValueError, "Tensor data is null.");
    return nullptr;
  }

  Py_RETURN_NONE;
}

}  // namespace

PyObject* InterpreterWrapper::GetTensor(int i) const {
  // Sanity check accessor
  TfLiteTensor* tensor = nullptr;
  int type_num = 0;

  PyObject* check_result =
      CheckGetTensorArgs(interpreter_.get(), i, &tensor, &type_num);
  if (check_result == nullptr) return check_result;
  Py_XDECREF(check_result);

  std::vector<npy_intp> dims(tensor->dims->data,
                             tensor->dims->data + tensor->dims->size);
  // Make a buffer copy but we must tell Numpy It owns that data or else
  // it will leak.
  void* data = malloc(tensor->bytes);
  if (!data) {
    PyErr_SetString(PyExc_ValueError, "Malloc to copy tensor failed.");
    return nullptr;
  }
  memcpy(data, tensor->data.raw, tensor->bytes);
  PyObject* np_array =
      PyArray_SimpleNewFromData(dims.size(), dims.data(), type_num, data);
  PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(np_array),
                      NPY_ARRAY_OWNDATA);
  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* InterpreterWrapper::tensor(PyObject* base_object, int i) {
  // Sanity check accessor
  TfLiteTensor* tensor = nullptr;
  int type_num = 0;

  PyObject* check_result =
      CheckGetTensorArgs(interpreter_.get(), i, &tensor, &type_num);
  if (check_result == nullptr) return check_result;
  Py_XDECREF(check_result);

  std::vector<npy_intp> dims(tensor->dims->data,
                             tensor->dims->data + tensor->dims->size);
  PyArrayObject* np_array =
      reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(
          dims.size(), dims.data(), type_num, tensor->data.raw));
  Py_INCREF(base_object);  // SetBaseObject steals, so we need to add.
  PyArray_SetBaseObject(np_array, base_object);
  return PyArray_Return(np_array);
}

InterpreterWrapper* InterpreterWrapper::CreateWrapperCPPFromFile(
    const char* model_path, std::string* error_msg) {
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_path, error_reporter.get());
  return CreateInterpreterWrapper(std::move(model), std::move(error_reporter),
                                  error_msg);
}

InterpreterWrapper* InterpreterWrapper::CreateWrapperCPPFromBuffer(
    PyObject* data, std::string* error_msg) {
  char * buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);
  if (PY_TO_CPPSTRING(data, &buf, &length) == -1) {
    return nullptr;
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(buf, length,
                                               error_reporter.get());
  return CreateInterpreterWrapper(std::move(model), std::move(error_reporter),
                                  error_msg);
}

PyObject* InterpreterWrapper::ResetVariableTensors() {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  Py_RETURN_NONE;
}

}  // namespace interpreter_wrapper
}  // namespace tflite
