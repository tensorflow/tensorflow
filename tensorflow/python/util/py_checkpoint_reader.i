/* Copyright 2015 Google Inc. All Rights Reserved.

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

%include "tensorflow/python/lib/core/status.i"
%include "tensorflow/python/lib/core/strings.i"
%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/checkpoint_reader.h"
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

%typemap(in, numinputs=0)
    std::unique_ptr<tensorflow::checkpoint::CheckpointReader>* out_reader (
        std::unique_ptr<tensorflow::checkpoint::CheckpointReader> temp) {
  $1 = &temp;
}

%typemap(out) tensorflow::Status tensorflow::checkpoint::NewCheckpointReader {
  if (!$1.ok()) {
    RaiseStatusNotOK($1, $descriptor(tensorflow::Status*));
    SWIG_fail;
  }
}

%typemap(argout) std::unique_ptr<tensorflow::checkpoint::CheckpointReader>*
  out_reader {
  $result = SWIG_NewPointerObj(
      $1->release(), $descriptor(tensorflow::checkpoint::CheckpointReader*),
      SWIG_POINTER_OWN);
}

%ignoreall
%unignore tensorflow;
%unignore tensorflow::checkpoint;
%unignore tensorflow::checkpoint::CheckpointReader;
%unignore tensorflow::checkpoint::CheckpointReader::~CheckpointReader;
%unignore tensorflow::checkpoint::CheckpointReader::DebugString;
%unignore tensorflow::checkpoint::CheckpointReader::GetVariableToShapeMap;
%rename("_HasTensor") tensorflow::checkpoint::CheckpointReader::HasTensor;

%newobject tensorflow::checkpoint::CheckpointReader::HasTensor;

%extend tensorflow::checkpoint::CheckpointReader {
%insert("python") %{
  def HasTensor(self, tensor_str):
    from tensorflow.python.util import compat
    return self._HasTensor(compat.as_bytes(tensor_str))
%}
}

%rename("_NewCheckpointReader") tensorflow::checkpoint::NewCheckpointReader;

%newobject tensorflow::checkpoint::NewCheckpointReader;

%insert("python") %{
  def NewCheckpointReader(filepattern):
    from tensorflow.python.util import compat
    return _NewCheckpointReader(compat.as_bytes(filepattern))
%}

%include "tensorflow/core/util/checkpoint_reader.h"
%unignoreall
