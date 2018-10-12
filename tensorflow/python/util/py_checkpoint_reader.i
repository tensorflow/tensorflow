/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

%include "tensorflow/python/lib/core/strings.i"
%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/py_func.h"
#include "tensorflow/python/lib/core/safe_ptr.h"
%}

%typemap(out) const tensorflow::checkpoint::TensorSliceReader::VarToShapeMap& {
  tensorflow::Safe_PyObjectPtr output_map(tensorflow::make_safe(PyDict_New()));
  for (auto v : *$1) {
%#if PY_MAJOR_VERSION >= 3
    tensorflow::Safe_PyObjectPtr key(
        tensorflow::make_safe(PyUnicode_FromStringAndSize(v.first.c_str(),
            v.first.size())));
%#else
    tensorflow::Safe_PyObjectPtr key(
        tensorflow::make_safe(PyString_FromStringAndSize(v.first.c_str(),
            v.first.size())));
%#endif
    if (!key) {
      SWIG_fail;
    }
    size_t dims = v.second.dims();
    tensorflow::Safe_PyObjectPtr value(tensorflow::make_safe(PyList_New(dims)));
    if (!value) {
      SWIG_fail;
    }
    for (size_t i = 0; i < dims; ++i) {
%#if PY_MAJOR_VERSION >= 3
      tensorflow::Safe_PyObjectPtr dim_value(
          tensorflow::make_safe(PyLong_FromLong(v.second.dim_size(i))));
%#else
      tensorflow::Safe_PyObjectPtr dim_value(
          tensorflow::make_safe(PyInt_FromLong(v.second.dim_size(i))));
%#endif
      if (!dim_value) {
        SWIG_fail;
      }
      PyList_SET_ITEM(value.get(), i, dim_value.release());
    }
    if (PyDict_SetItem(output_map.get(), key.get(), value.get()) == -1) {
      SWIG_fail;
    } else {
      key.release();
      value.release();
    }
  }

  $result = output_map.release();
}

%typemap(out) const tensorflow::checkpoint::TensorSliceReader::VarToDataTypeMap& {
  tensorflow::Safe_PyObjectPtr output_map(tensorflow::make_safe(PyDict_New()));
  for (auto v : *$1) {
%#if PY_MAJOR_VERSION >= 3
    tensorflow::Safe_PyObjectPtr key(
        tensorflow::make_safe(PyUnicode_FromStringAndSize(v.first.c_str(), v.first.size())));
%#else
    tensorflow::Safe_PyObjectPtr key(
        tensorflow::make_safe(PyString_FromStringAndSize(v.first.c_str(), v.first.size())));
%#endif
    if (!key) {
      SWIG_fail;
    }
%#if PY_MAJOR_VERSION >= 3
    tensorflow::Safe_PyObjectPtr value(tensorflow::make_safe(PyLong_FromLong(v.second)));
%#else
    tensorflow::Safe_PyObjectPtr value(tensorflow::make_safe(PyInt_FromLong(v.second)));
%#endif
    if (!value) {
      SWIG_fail;
    }
    if (PyDict_SetItem(output_map.get(), key.get(), value.get()) == -1) {
      SWIG_fail;
    } else {
      key.release();
      value.release();
    }
  }

  $result = output_map.release();
}

%{
static PyObject* CheckpointReader_GetTensor(
      tensorflow::checkpoint::CheckpointReader* reader,
      const string& name,
      TF_Status* out_status) {
  PyObject* py_obj = Py_None;
  std::unique_ptr<tensorflow::Tensor> tensor;
  reader->GetTensor(name, &tensor, out_status);
  if (TF_GetCode(out_status) == TF_OK) {
    tensorflow::Status status =
        tensorflow::ConvertTensorToNdarray(*tensor.get(), &py_obj);
    if (!status.ok()) {
      Set_TF_Status_from_Status(out_status, status);
    }
  }
  return py_obj;
}
%}

// Wrap this function.
PyObject* CheckpointReader_GetTensor(
    tensorflow::checkpoint::CheckpointReader* reader,
    const string& name,
    TF_Status* out_status);

%ignoreall

%unignore tensorflow;
%unignore tensorflow::checkpoint;
%unignore tensorflow::checkpoint::CheckpointReader;
%unignore tensorflow::checkpoint::CheckpointReader::CheckpointReader;
%unignore tensorflow::checkpoint::CheckpointReader::~CheckpointReader;
%rename("debug_string") tensorflow::checkpoint::CheckpointReader::DebugString;
%rename("get_variable_to_shape_map") tensorflow::checkpoint::CheckpointReader::GetVariableToShapeMap;
%rename("_GetVariableToDataTypeMap") tensorflow::checkpoint::CheckpointReader::GetVariableToDataTypeMap;
%rename("_HasTensor") tensorflow::checkpoint::CheckpointReader::HasTensor;
%unignore CheckpointReader_GetTensor;

%extend tensorflow::checkpoint::CheckpointReader {
%insert("python") %{
  def get_variable_to_dtype_map(self):
    from tensorflow.python.framework import dtypes
    return {name: dtypes.DType(type_enum)
            for name, type_enum in self._GetVariableToDataTypeMap().items()}

  def has_tensor(self, tensor_str):
    from tensorflow.python.util import compat
    return self._HasTensor(compat.as_bytes(tensor_str))

  def get_tensor(self, tensor_str):
    from tensorflow.python.framework import errors
    with errors.raise_exception_on_not_ok_status() as status:
      from tensorflow.python.util import compat
      return CheckpointReader_GetTensor(self, compat.as_bytes(tensor_str),
                                        status)
%}
}

%insert("python") %{
def NewCheckpointReader(filepattern):
  from tensorflow.python.framework import errors
  with errors.raise_exception_on_not_ok_status() as status:
    from tensorflow.python.util import compat
    return CheckpointReader(compat.as_bytes(filepattern), status)

NewCheckpointReader._tf_api_names = ['train.NewCheckpointReader']
NewCheckpointReader._tf_api_names_v1 = ['train.NewCheckpointReader']
%}

%include "tensorflow/c/checkpoint_reader.h"
%unignoreall
