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

%ignore "";

%rename("%s") TFE_Py_RegisterExceptionClass;
%rename("%s") TFE_Py_NumpyToTensorHandle;
%rename("%s") TFE_NewContext;
%rename("%s") TFE_DeleteContext;
%rename("%s") TFE_ContextListDevices;
%rename("%s") TFE_TensorHandleDataType;
%rename("%s") TFE_TensorHandleNumDims;
%rename("%s") TFE_DeleteTensorHandle;
%rename("%s") TFE_Py_Execute;
%rename("%s") TFE_ContextAddFunctionDef;
%rename("%s") TFE_TensorHandleDim;
%rename("%s") TFE_TensorHandleDeviceName;
%rename("%s") TFE_TensorHandleCopyToDevice;
%rename("%s") TFE_NewOp;
%rename("%s") TFE_Py_TensorHandleToNumpy;
%rename("%s") TFE_OpGetAttrType;


%{
#include "tensorflow/python/eager/pywrap_tfe.h"
%}

%typemap(out) TF_DataType {
  $result = PyInt_FromLong($1);
}

%typemap(out) int64_t {
  $result = PyInt_FromLong($1);
}

%typemap(out) TF_AttrType {
  $result = PyInt_FromLong($1);
}

%typemap(in, numinputs=0) unsigned char* is_list (unsigned char tmp) {
  $1 = &tmp;
}

%typemap(argout) unsigned char* is_list {
  if (*$1 == 1) {
    PyObject* list = PyList_New(1);
    PyList_SetItem(list, 0, $result);
    $result = list;
  }
}

%typemap(in) const char* serialized_function_def {
  $1 = TFE_GetPyThonString($input);
}

%typemap(in) const char* device_name {
  if ($input == Py_None) {
    $1 = nullptr;
  } else {
    $1 = TFE_GetPyThonString($input);
  }
}

%typemap(in) const char* op_name {
  $1 = TFE_GetPyThonString($input);
}

%include "tensorflow/c/eager/c_api.h"

%typemap(in) TFE_InputTensorHandles* inputs (TFE_InputTensorHandles temp) {
  $1 = &temp;
  if ($input != Py_None) {
    if (!PyList_Check($input)) {
      SWIG_exception_fail(SWIG_TypeError,
                          "must provide a list of Tensors as inputs");
    }
    Py_ssize_t len = PyList_Size($input);
    $1->resize(len);
    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject* elem = PyList_GetItem($input, i);
      if (!elem) {
        SWIG_fail;
      }
      void* thp = nullptr;
      int res = SWIG_ConvertPtr(elem, &thp,
                                $descriptor(TFE_TensorHandle*), 0 | 0);
      if (!SWIG_IsOK(res)) {
        SWIG_exception_fail(SWIG_ArgError(res),
                            "provided list of inputs contains objects other "
                            "than 'TFE_TensorHandle*'");
      }
      (*$1)[i] = reinterpret_cast<TFE_TensorHandle*>(thp);
    }
  }
}

// Temporary for the argout
%typemap(in) TFE_OutputTensorHandles* outputs (TFE_OutputTensorHandles temp) {
  if (!PyInt_Check($input)) {
    SWIG_exception_fail(SWIG_TypeError,
                        "expected an integer value (size of the number of "
                        "outputs of the operation)");
  }
  $1 = &temp;
  $1->resize(PyInt_AsLong($input), nullptr);
}

// Create new Status object.
%typemap(in, numinputs=0) TF_Status *out_status {
  $1 = TF_NewStatus();
}

%typemap(freearg) (TF_Status* out_status) {
 TF_DeleteStatus($1);
}

%typemap(argout) (TFE_OutputTensorHandles* outputs, TF_Status* out_status) {
  if (TFE_Py_MayBeRaiseException($2)) {
    SWIG_fail;
  } else {
    int num_outputs = $1->size();
    $result = PyList_New(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      PyList_SetItem($result, i, SWIG_NewPointerObj(SWIG_as_voidptr($1->at(i)),
                                                    $descriptor(TFE_TensorHandle*),
                                                    0 | 0));
    }
  }
}

// Note that we need to use a typemap for TFE_TensorHandle* so that we can call
// SWIG_fail in case the value is nullptr.  Otherwise SWIG will wrap the
// nullptr and return it to python as an opaque object, and python does not
// know that it needs to check if an Exception has been raised.
// TODO(agarwal): check if we can get rid of this typemap.
%typemap(out) (TFE_TensorHandle*) {
  if ($1 == nullptr) {
    SWIG_fail;
  } else {
    $result = SWIG_NewPointerObj(SWIG_as_voidptr($1),
                                 $descriptor(TFE_TensorHandle*), 0 | 0);
  }
}

%include "tensorflow/python/eager/pywrap_tfe.h"


// Clear all typemaps127
%typemap(out) TF_DataType;
%typemap(out) int64_t;
%typemap(out) TF_AttrType;
%typemap(in, numinputs=0) TF_Status *out_status;
%typemap(argout) unsigned char* is_list;
%typemap(in) TFE_InputTensorHandles* inputs (TFE_InputTensorHandles temp);
%typemap(in) TFE_OutputTensorHandles* outputs (TFE_OutputTensorHandles temp);
%typemap(in, numinputs=0) TF_Status *out_status;
%typemap(freearg) (TF_Status* out_status);
%typemap(argout) (TFE_OutputTensorHandles* outputs, TF_Status* out_status);
%typemap(out) (TFE_TensorHandle*);
