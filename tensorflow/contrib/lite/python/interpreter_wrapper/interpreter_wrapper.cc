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
#include "tensorflow/python/lib/core/numpy.h"

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
std::unique_ptr<tflite::Interpreter> CreateInterpreter(
    const tflite::FlatBufferModel* model,
    const tflite::ops::builtin::BuiltinOpResolver& resolver) {
  if (!model) {
    return nullptr;
  }

  tensorflow::ImportNumpy();

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

PyObject* InterpreterWrapper::GetTensor(int i) const {
  if (!interpreter_) {
    LOG(ERROR) << "Invalid interpreter.";
    Py_INCREF(Py_None);
    return Py_None;
  }

  if (i >= interpreter_->tensors_size()) {
    LOG(ERROR) << "Invalid tensor index: " << i << " exceeds max tensor index "
               << interpreter_->inputs().size();
    Py_INCREF(Py_None);
    return Py_None;
  }

  const TfLiteTensor* output_tensor = interpreter_->tensor(i);
  const int tensor_size = output_tensor->bytes;
  if (tensor_size <= 0) {
    LOG(ERROR) << "Invalid tensor size";
    Py_INCREF(Py_None);
    return Py_None;
  }

  int type_num = TfLiteTypeToPyArrayType(output_tensor->type);
  if (type_num == -1) {
    LOG(ERROR) << "Unknown tensor type " << output_tensor->type;
    Py_INCREF(Py_None);
    return Py_None;
  }

  void* data = malloc(tensor_size);
  memcpy(data, output_tensor->data.raw, tensor_size);

  const TfLiteIntArray* output_dims = output_tensor->dims;
  std::vector<npy_intp> dims(output_dims->data,
                             output_dims->data + output_dims->size);
  PyObject* np_array =
      PyArray_SimpleNewFromData(dims.size(), dims.data(), type_num, data);

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
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
