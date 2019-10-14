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

// Windows does not have dlfcn.h/dlsym, use GetProcAddress() instead.
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif  // defined(_WIN32)

#include <stdarg.h>

#include <sstream>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/python/interpreter_wrapper/numpy.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_error_reporter.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"
#include "tensorflow/lite/string_util.h"

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

using python_utils::PyDecrefDeleter;

std::unique_ptr<tflite::Interpreter> CreateInterpreter(
    const tflite::FlatBufferModel* model,
    const tflite::ops::builtin::BuiltinOpResolver& resolver) {
  if (!model) {
    return nullptr;
  }

  ::tflite::python::ImportNumpy();

  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
    return nullptr;
  }
  return interpreter;
}

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

bool RegisterCustomOpByName(const char* registerer_name,
                            tflite::MutableOpResolver* resolver,
                            std::string* error_msg) {
  // Registerer functions take a pointer to a BuiltinOpResolver as an input
  // parameter and return void.
  // TODO(b/137576229): We should implement this functionality in a more
  // principled way.
  typedef void (*RegistererFunctionType)(tflite::MutableOpResolver*);

  // Look for the Registerer function by name.
  RegistererFunctionType registerer = reinterpret_cast<RegistererFunctionType>(
  // We don't have dlsym on Windows, use GetProcAddress instead.
#if defined(_WIN32)
      GetProcAddress(nullptr, registerer_name)
#else
      dlsym(RTLD_DEFAULT, registerer_name)
#endif  // defined(_WIN32)
  );

  // Fail in an informative way if the function was not found.
  if (registerer == nullptr) {
    // We don't have dlerror on Windows, use GetLastError instead.
    *error_msg =
#if defined(_WIN32)
        absl::StrFormat("Looking up symbol '%s' failed with error (0x%x).",
                        registerer_name, GetLastError());
#else
        absl::StrFormat("Looking up symbol '%s' failed with error '%s'.",
                        registerer_name, dlerror());
#endif  // defined(_WIN32)
    return false;
  }

  // Call the registerer with the resolver.
  registerer(resolver);
  return true;
}

}  // namespace

InterpreterWrapper* InterpreterWrapper::CreateInterpreterWrapper(
    std::unique_ptr<tflite::FlatBufferModel> model,
    std::unique_ptr<PythonErrorReporter> error_reporter,
    const std::vector<std::string>& registerers, std::string* error_msg) {
  if (!model) {
    *error_msg = error_reporter->message();
    return nullptr;
  }

  auto resolver = absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
  for (const auto registerer : registerers) {
    if (!RegisterCustomOpByName(registerer.c_str(), resolver.get(), error_msg))
      return nullptr;
  }
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

  // Release the GIL so that we can run multiple interpreters in parallel
  TfLiteStatus status_code = kTfLiteOk;
  Py_BEGIN_ALLOW_THREADS;  // To return can happen between this and end!
  status_code = interpreter_->Invoke();
  Py_END_ALLOW_THREADS;

  TFLITE_PY_CHECK(
      status_code);  // don't move this into the Py_BEGIN/Py_End block

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
  TfLiteTensor* tensor = interpreter_->tensor(i);

  if (python_utils::TfLiteTypeFromPyArray(array) != tensor->type) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor:"
                 " Got value of type %s"
                 " but expected type %s for input %d, name: %s ",
                 TfLiteTypeGetName(python_utils::TfLiteTypeFromPyArray(array)),
                 TfLiteTypeGetName(tensor->type), i, tensor->name);
    return nullptr;
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: Dimension mismatch."
                 " Got %d"
                 " but expected %d for input %d.",
                 PyArray_NDIM(array), tensor->dims->size, i);
    return nullptr;
  }

  for (int j = 0; j < PyArray_NDIM(array); j++) {
    if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor: Dimension mismatch."
                   " Got %ld"
                   " but expected %d for dimension %d of input %d.",
                   PyArray_SHAPE(array)[j], tensor->dims->data[j], j, i);
      return nullptr;
    }
  }

  if (tensor->type != kTfLiteString) {
    size_t size = PyArray_NBYTES(array);
    if (size != tensor->bytes) {
      PyErr_Format(PyExc_ValueError,
                   "numpy array had %zu bytes but expected %zu bytes.", size,
                   tensor->bytes);
      return nullptr;
    }
    memcpy(tensor->data.raw, PyArray_DATA(array), size);
  } else {
    DynamicBuffer dynamic_buffer;
    if (!python_utils::FillStringBufferWithPyArray(value, &dynamic_buffer)) {
      return nullptr;
    }
    dynamic_buffer.WriteToTensor(tensor, nullptr);
  }
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
  if (tensor->type != kTfLiteString) {
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
  } else {
    // Create a C-order array so the data is contiguous in memory.
    const int32_t kCOrder = 0;
    PyObject* py_object =
        PyArray_EMPTY(dims.size(), dims.data(), NPY_OBJECT, kCOrder);

    if (py_object == nullptr) {
      PyErr_SetString(PyExc_MemoryError, "Failed to allocate PyArray.");
      return nullptr;
    }

    PyArrayObject* py_array = reinterpret_cast<PyArrayObject*>(py_object);
    PyObject** data = reinterpret_cast<PyObject**>(PyArray_DATA(py_array));
    auto num_strings = GetStringCount(tensor);
    for (int j = 0; j < num_strings; ++j) {
      auto ref = GetString(tensor, j);

      PyObject* bytes = PyBytes_FromStringAndSize(ref.str, ref.len);
      if (bytes == nullptr) {
        Py_DECREF(py_object);
        PyErr_Format(PyExc_ValueError,
                     "Could not create PyBytes from string %d of input %d.", j,
                     i);
        return nullptr;
      }
      // PyArray_EMPTY produces an array full of Py_None, which we must decref.
      Py_DECREF(data[j]);
      data[j] = bytes;
    }
    return py_object;
  }
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
    const char* model_path, const std::vector<std::string>& registerers,
    std::string* error_msg) {
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_path, error_reporter.get());
  return CreateInterpreterWrapper(std::move(model), std::move(error_reporter),
                                  registerers, error_msg);
}

InterpreterWrapper* InterpreterWrapper::CreateWrapperCPPFromBuffer(
    PyObject* data, const std::vector<std::string>& registerers,
    std::string* error_msg) {
  char* buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);

  if (python_utils::ConvertFromPyString(data, &buf, &length) == -1) {
    return nullptr;
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(buf, length,
                                               error_reporter.get());
  return CreateInterpreterWrapper(std::move(model), std::move(error_reporter),
                                  registerers, error_msg);
}

PyObject* InterpreterWrapper::ResetVariableTensors() {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  Py_RETURN_NONE;
}

PyObject* InterpreterWrapper::ModifyGraphWithDelegate(
    TfLiteDelegate* delegate) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->ModifyGraphWithDelegate(delegate));
  Py_RETURN_NONE;
}

}  // namespace interpreter_wrapper
}  // namespace tflite
