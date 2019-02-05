/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// SWIG typemaps for TF_SessionRun_wrapper()

%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/python/client/tf_session_helper.h"
%}

// Required to use PyArray_* functions.
%init %{
tensorflow::ImportNumpy();
%}

// $input is a Python dict mapping wrapped TF_Outputs to ndarrays.
%typemap(in) (const std::vector<TF_Output>& inputs,
              const std::vector<PyObject*>& input_ndarrays)
    (std::vector<TF_Output> inputs, std::vector<PyObject*> input_ndarrays) {
  if (!PyDict_Check($input)) {
    SWIG_exception_fail(SWIG_TypeError, "$symname: expected dict");
  }
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  while (PyDict_Next($input, &pos, &key, &value)) {
    TF_Output* input_ptr;
    SWIG_ConvertPtr(key, reinterpret_cast<void**>(&input_ptr),
                    SWIGTYPE_p_TF_Output, 0);
    inputs.push_back(*input_ptr);

    if (!PyArray_Check(value)) {
      SWIG_exception_fail(
          SWIG_TypeError,
          "$symname: expected all values in input dict to be ndarray");
    }
    input_ndarrays.push_back(value);
  }
  $1 = &inputs;
  $2 = &input_ndarrays;
}

// $input is a Python list of wrapped TF_Operations
%typemap(in) (const std::vector<TF_Operation*>& targets)
    (std::vector<TF_Operation*> targets) {
  if (!PyList_Check($input)) {
    SWIG_exception_fail(SWIG_TypeError, "$symname: expected list");
  }
  size_t size = PyList_Size($input);
  for (int i = 0; i < size; ++i) {
    PyObject* item = PyList_GetItem($input, i);
    TF_Operation* oper_ptr;
    SWIG_ConvertPtr(item, reinterpret_cast<void**>(&oper_ptr),
                    SWIGTYPE_p_TF_Operation, 0);
    targets.push_back(oper_ptr);
  }
  $1 = &targets;
}

// $input is a Python list of wrapped TF_Outputs
%typemap(in) (const std::vector<TF_Output>& outputs)
    (std::vector<TF_Output> outputs) {
  string error_msg;
  if (!PyTensorListToVector($input, &outputs, &error_msg)) {
    SWIG_exception_fail(SWIG_TypeError, ("$symname: " + error_msg).c_str());
  }
  $1 = &outputs;
}

// Apply the typemap above to inputs as well
%typemap(in) (const std::vector<TF_Output>& inputs) =
             (const std::vector<TF_Output>& outputs);

// Create temporary py_outputs_vec variable to store return value
%typemap(in, numinputs=0) (std::vector<PyObject*>* py_outputs)
    (std::vector<PyObject*> py_outputs_vec) {
  $1 = &py_outputs_vec;
}

// Convert py_outputs to returned Python list
%typemap(argout) (std::vector<PyObject*>* py_outputs) {
  $result = PyList_New($1->size());
  if (!$result) {
    SWIG_exception_fail(SWIG_MemoryError, "$symname: couldn't create list");
  }
  for (int i = 0; i < $1->size(); ++i) {
    PyList_SET_ITEM($result, i, (*$1)[i]);
  }
}
