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
#include "tensorflow/contrib/lite/python/interpreter_wrapper/interpreter_wrapper.h"

#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/core/platform/logging.h"

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
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (interpreter) {
    for (const int input_index : interpreter->inputs()) {
      const TfLiteTensor* tensor = interpreter->tensor(input_index);
      CHECK(tensor);
      const TfLiteIntArray* dims = tensor->dims;
      if (!dims) {
        continue;
      }

      std::vector<int> input_dims(dims->data, dims->data + dims->size);
      interpreter->ResizeInputTensor(input_index, input_dims);
    }
  }
  return interpreter;
}

int TfLiteTypeToPyArrayType(TfLiteType tf_lite_type) {
  switch (tf_lite_type) {
    case kTfLiteFloat32:
      return NPY_FLOAT32;
    case kTfLiteInt32:
      return NPY_INT32;
    case kTfLiteInt16:
      return NPY_INT16;
    case kTfLiteUInt8:
      return NPY_UINT8;
    case kTfLiteInt64:
      return NPY_INT64;
    case kTfLiteString:
      return NPY_OBJECT;
    case kTfLiteBool:
      return NPY_BOOL;
    case kTfLiteComplex64:
      return NPY_COMPLEX64;
    case kTfLiteNoType:
      return -1;
  }
  LOG(ERROR) << "Unknown TfLiteType " << tf_lite_type;
  return -1;
}

TfLiteType TfLiteTypeFromPyArray(PyArrayObject* array) {
  int pyarray_type = PyArray_TYPE(array);
  switch (pyarray_type) {
    case NPY_FLOAT32:
      return kTfLiteFloat32;
    case NPY_INT32:
      return kTfLiteInt32;
    case NPY_INT16:
      return kTfLiteInt16;
    case NPY_UINT8:
      return kTfLiteUInt8;
    case NPY_INT64:
      return kTfLiteInt64;
    case NPY_BOOL:
      return kTfLiteBool;
    case NPY_OBJECT:
    case NPY_STRING:
    case NPY_UNICODE:
      return kTfLiteString;
    case NPY_COMPLEX64:
      return kTfLiteComplex64;
  }
  LOG(ERROR) << "Unknown PyArray dtype " << pyarray_type;
  return kTfLiteNoType;
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

InterpreterWrapper::InterpreterWrapper(
    std::unique_ptr<tflite::FlatBufferModel> model)
    : model_(std::move(model)),
      resolver_(absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>()),
      interpreter_(CreateInterpreter(model_.get(), *resolver_)) {}

InterpreterWrapper::~InterpreterWrapper() {}

bool InterpreterWrapper::AllocateTensors() {
  if (!interpreter_) {
    LOG(ERROR) << "Cannot allocate tensors: invalid interpreter.";
    return false;
  }

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Unable to allocate tensors.";
    return false;
  }

  return true;
}

bool InterpreterWrapper::Invoke() {
  return interpreter_ ? (interpreter_->Invoke() == kTfLiteOk) : false;
}

PyObject* InterpreterWrapper::InputIndices() const {
  PyObject* np_array = PyArrayFromIntVector(interpreter_->inputs().data(),
                                            interpreter_->inputs().size());

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* InterpreterWrapper::OutputIndices() const {
  PyObject* np_array = PyArrayFromIntVector(interpreter_->outputs().data(),
                                            interpreter_->outputs().size());

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

bool InterpreterWrapper::ResizeInputTensor(int i, PyObject* value) {
  if (!interpreter_) {
    LOG(ERROR) << "Invalid interpreter.";
    return false;
  }

  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    LOG(ERROR) << "Failed to convert value into readable tensor.";
    return false;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());

  if (PyArray_NDIM(array) != 1) {
    LOG(ERROR) << "Expected 1-D defining input shape.";
    return false;
  }

  if (PyArray_TYPE(array) != NPY_INT32) {
    LOG(ERROR) << "Shape must be an int32 array";
    return false;
  }

  std::vector<int> dims(PyArray_SHAPE(array)[0]);
  memcpy(dims.data(), PyArray_BYTES(array), dims.size() * sizeof(int));

  return (interpreter_->ResizeInputTensor(i, dims) == kTfLiteOk);
}

std::string InterpreterWrapper::TensorName(int i) const {
  if (!interpreter_ || i >= interpreter_->tensors_size() || i < 0) {
    return "";
  }

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  return tensor->name;
}

PyObject* InterpreterWrapper::TensorType(int i) const {
  if (!interpreter_ || i >= interpreter_->tensors_size() || i < 0) {
    return nullptr;
  }

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  int typenum = TfLiteTypeToPyArrayType(tensor->type);
  return PyArray_TypeObjectFromType(typenum);
}

PyObject* InterpreterWrapper::TensorSize(int i) const {
  if (!interpreter_ || i >= interpreter_->tensors_size() || i < 0) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  PyObject* np_array =
      PyArrayFromIntVector(tensor->dims->data, tensor->dims->size);

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* InterpreterWrapper::TensorQuantization(int i) const {
  if (!interpreter_ || i >= interpreter_->tensors_size() || i < 0) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  return PyTupleFromQuantizationParam(tensor->params);
}

bool InterpreterWrapper::SetTensor(int i, PyObject* value) {
  if (!interpreter_) {
    LOG(ERROR) << "Invalid interpreter.";
    return false;
  }

  if (i >= interpreter_->tensors_size()) {
    LOG(ERROR) << "Invalid tensor index: " << i << " exceeds max tensor index "
               << interpreter_->tensors_size();
    return false;
  }

  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    LOG(ERROR) << "Failed to convert value into readable tensor.";
    return false;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());
  const TfLiteTensor* tensor = interpreter_->tensor(i);

  if (TfLiteTypeFromPyArray(array) != tensor->type) {
    LOG(ERROR) << "Cannot set tensor:"
               << " Got tensor of type " << TfLiteTypeFromPyArray(array)
               << " but expected type " << tensor->type << " for input " << i;
    return false;
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    LOG(ERROR) << "Cannot set tensor: Dimension mismatch";
    return false;
  }

  for (int j = 0; j < PyArray_NDIM(array); j++) {
    if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      LOG(ERROR) << "Cannot set tensor: Dimension mismatch";
      return false;
    }
  }

  size_t size = PyArray_NBYTES(array);
  DCHECK_EQ(size, tensor->bytes);
  memcpy(tensor->data.raw, PyArray_DATA(array), size);
  return true;
}

namespace {

PyObject* CheckGetTensorArgs(Interpreter* interpreter, int tensor_index,
                             TfLiteTensor** tensor, int* type_num) {
  if (!interpreter) {
    LOG(ERROR) << "Invalid interpreter.";
    Py_INCREF(Py_None);
    return Py_None;
  }

  if (tensor_index >= interpreter->tensors_size() || tensor_index < 0) {
    LOG(ERROR) << "Invalid tensor index: " << tensor_index
               << " exceeds max tensor index " << interpreter->inputs().size();
    Py_INCREF(Py_None);
    return Py_None;
  }

  *tensor = interpreter->tensor(tensor_index);
  if ((*tensor)->bytes == 0) {
    LOG(ERROR) << "Invalid tensor size";
    Py_INCREF(Py_None);
    return Py_None;
  }

  *type_num = TfLiteTypeToPyArrayType((*tensor)->type);
  if (*type_num == -1) {
    LOG(ERROR) << "Unknown tensor type " << (*tensor)->type;
    Py_INCREF(Py_None);
    return Py_None;
  }

  if (!(*tensor)->data.raw) {
    LOG(ERROR) << "Tensor data is null.";
    Py_INCREF(Py_None);
    return Py_None;
  }

  return nullptr;
}

}  // namespace

PyObject* InterpreterWrapper::GetTensor(int i) const {
  // Sanity check accessor
  TfLiteTensor* tensor = nullptr;
  int type_num = 0;
  if (PyObject* pynone_or_nullptr =
          CheckGetTensorArgs(interpreter_.get(), i, &tensor, &type_num)) {
    return pynone_or_nullptr;
  }
  std::vector<npy_intp> dims(tensor->dims->data,
                             tensor->dims->data + tensor->dims->size);
  // Make a buffer copy but we must tell Numpy It owns that data or else
  // it will leak.
  void* data = malloc(tensor->bytes);
  if (!data) {
    LOG(ERROR) << "Malloc to copy tensor failed.";
    Py_INCREF(Py_None);
    return Py_None;
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
  if (PyObject* pynone_or_nullptr =
          CheckGetTensorArgs(interpreter_.get(), i, &tensor, &type_num)) {
    return pynone_or_nullptr;
  }

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
    const char* model_path) {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_path);
  return model ? new InterpreterWrapper(std::move(model)) : nullptr;
}

InterpreterWrapper* InterpreterWrapper::CreateWrapperCPPFromBuffer(
    PyObject* data) {
  char * buf = nullptr;
  Py_ssize_t length;
  if (PY_TO_CPPSTRING(data, &buf, &length) == -1) {
    return nullptr;
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(buf, length);
  return model ? new InterpreterWrapper(std::move(model)) : nullptr;
}

}  // namespace interpreter_wrapper
}  // namespace tflite
