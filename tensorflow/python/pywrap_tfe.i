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

%rename("%s") TFE_NewContext;
%rename("%s") TFE_DeleteContext;
%rename("%s") TFE_ContextListDevices;
%rename("%s") TFE_ContextAddFunction;
%rename("%s") TFE_ContextAddFunctionDef;
%rename("%s") TFE_ContextEnableRunMetadata;
%rename("%s") TFE_ContextDisableRunMetadata;
%rename("%s") TFE_ContextExportRunMetadata;
%rename("%s") TFE_ContextClearCaches;
%rename("%s") TFE_ContextGetDevicePlacementPolicy;
%rename("%s") TFE_ContextSetThreadLocalDevicePlacementPolicy;
%rename("%s") TFE_OpNameGetAttrType;
%rename("%s") TFE_Py_InitEagerTensor;
%rename("%s") TFE_Py_RegisterExceptionClass;
%rename("%s") TFE_Py_RegisterBackwardFunctionGetter;
%rename("%s") TFE_Py_RegisterFallbackExceptionClass;
%rename("%s") TFE_Py_RegisterResourceVariableType;
%rename("%s") TFE_Py_Execute;
%rename("%s") TFE_Py_FastPathExecute;
%rename("%s") TFE_Py_RecordGradient;
%rename("%s") TFE_Py_UID;
%rename("%s") TFE_Py_TapeSetNew;
%rename("%s") TFE_Py_TapeSetRemove;
%rename("%s") TFE_Py_TapeSetStopOnThread;
%rename("%s") TFE_Py_TapeSetRestartOnThread;
%rename("%s") TFE_Py_TapeSetIsEmpty;
%rename("%s") TFE_Py_TapeSetShouldRecord;
%rename("%s") TFE_Py_TapeSetWatch;
%rename("%s") TFE_Py_TapeSetDeleteTrace;
%rename("%s") TFE_Py_TapeSetRecordOperation;
%rename("%s") TFE_Py_TapeSetWatchVariable;
%rename("%s") TFE_Py_TapeGradient;
%rename("%s") TFE_Py_TapeWatchedVariables;
%rename("%s") TFE_NewContextOptions;
%rename("%s") TFE_ContextOptionsSetConfig;
%rename("%s") TFE_ContextOptionsSetDevicePlacementPolicy;
%rename("%s") TFE_DeleteContextOptions;
%rename("%s") TFE_Py_TensorShapeSlice;

%{
#include "tensorflow/python/eager/pywrap_tfe.h"
%}

%typemap(in) (const void* proto) {
  char* c_string;
  Py_ssize_t py_size;
  // PyBytes_AsStringAndSize() does not copy but simply interprets the input
  if (PyBytes_AsStringAndSize($input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    SWIG_fail;
  }
  $1 = static_cast<void*>(c_string);
}

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
  $1 = TFE_GetPythonString($input);
}

%typemap(in) const char* device_name {
  if ($input == Py_None) {
    $1 = nullptr;
  } else {
    $1 = TFE_GetPythonString($input);
  }
}

%typemap(in) const char* op_name {
  $1 = TFE_GetPythonString($input);
}

%typemap(in) (TFE_Context*) {
  $1 = (TFE_Context*)PyCapsule_GetPointer($input, nullptr);

}
%typemap(out) (TFE_Context*) {
  if ($1 == nullptr) {
    SWIG_fail;
  } else {
    $result = PyCapsule_New($1, nullptr, TFE_DeleteContextCapsule);
  }
}

%rename("%s") TFE_ContextDevicePlacementPolicy;
%rename("%s") TFE_DEVICE_PLACEMENT_EXPLICIT;
%rename("%s") TFE_DEVICE_PLACEMENT_WARN;
%rename("%s") TFE_DEVICE_PLACEMENT_SILENT;
%rename("%s") TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32;

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
      if (EagerTensor_CheckExact(elem)) {
        (*$1)[i] = EagerTensor_Handle(elem);
      } else {
        SWIG_exception_fail(SWIG_TypeError,
                            "provided list of inputs contains objects other "
                            "than 'EagerTensor'");
      }
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
  if (MaybeRaiseExceptionFromTFStatus($2, nullptr)) {
    SWIG_fail;
  } else {
    int num_outputs = $1->size();
    $result = PyList_New(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      PyObject *output;
      output = EagerTensorFromHandle($1->at(i));
      PyList_SetItem($result, i, output);
    }
  }
}

// SWIG usually unwraps the tuple that the native Python/C interface generates.
// Since we wanted to have a function with a variable length of arguments, we
// used the native Python/C interface directly (which by default supports
// passing all arguments as a tuple).
%native(TFE_Py_FastPathExecute) TFE_Py_FastPathExecute_C;

%include "tensorflow/python/eager/pywrap_tfe.h"

// Clear all typemaps.
%typemap(out) TF_DataType;
%typemap(out) int64_t;
%typemap(out) TF_AttrType;
%typemap(in, numinputs=0) TF_Status *out_status;
%typemap(argout) unsigned char* is_list;
%typemap(in) (TFE_Context*);
%typemap(out) (TFE_Context*);
%typemap(in) TFE_OutputTensorHandles* outputs (TFE_OutputTensorHandles temp);
%typemap(in, numinputs=0) TF_Status *out_status;
%typemap(freearg) (TF_Status* out_status);
%typemap(argout) (TFE_OutputTensorHandles* outputs, TF_Status* out_status);
%typemap(in) (const void* proto);
