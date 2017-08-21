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

%include "tensorflow/python/platform/base.i"

%{

#include "tensorflow/c/python_api.h"
#include "tensorflow/python/client/tf_session_helper.h"
#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/public/version.h"

%}

%include "tensorflow/python/client/tf_sessionrun_wrapper.i"

// Required to use PyArray_* functions.
%init %{
tensorflow::ImportNumpy();
%}

// TensorFlow version and GraphDef versions
%constant const char* __version__ = TF_VERSION_STRING;
%constant int GRAPH_DEF_VERSION = TF_GRAPH_DEF_VERSION;
%constant int GRAPH_DEF_VERSION_MIN_CONSUMER = TF_GRAPH_DEF_VERSION_MIN_CONSUMER;
%constant int GRAPH_DEF_VERSION_MIN_PRODUCER = TF_GRAPH_DEF_VERSION_MIN_PRODUCER;

// Git version information
%constant const char* __git_version__ = tf_git_version();

// Compiler
%constant const char* __compiler_version__ = tf_compiler_version();

// Release the Python GIL for the duration of most methods.
%exception {
  Py_BEGIN_ALLOW_THREADS;
  $action
  Py_END_ALLOW_THREADS;
}

// The target input to TF_SetTarget() is passed as a null-terminated
// const char*.
%typemap(in) (const char* target) {
  $1 = PyBytes_AsString($input);
   if (!$1) {
    // Python has raised an error.
    SWIG_fail;
  }
}

// Constants used by TensorHandle (get_session_handle).
%constant const char* TENSOR_HANDLE_KEY = tensorflow::SessionState::kTensorHandleResourceTypeName;

// Convert TF_OperationName output to unicode python string
%typemap(out) const char* TF_OperationName {
  $result = PyUnicode_FromString($1);
}

// Convert TF_OperationOpType output to unicode python string
%typemap(out) const char* TF_OperationOpType {
  $result = PyUnicode_FromString($1);
}

// We use TF_OperationGetControlInputs_wrapper instead of
// TF_OperationGetControlInputs
%ignore TF_OperationGetControlInputs;
%unignore TF_OperationGetControlInputs_wrapper;
// See comment for "%noexception TF_SessionRun_wrapper;"
%noexception TF_OperationGetControlInputs_wrapper;

// Build a Python list of TF_Operation* and return it.
%typemap(out) std::vector<TF_Operation*> tensorflow::TF_OperationGetControlInputs_wrapper {
  $result = PyList_New($1.size());
  if (!$result) {
    SWIG_exception_fail(SWIG_MemoryError, "$symname: couldn't create list");
  }

  for (size_t i = 0; i < $1.size(); ++i) {
    PyList_SET_ITEM($result, i, SWIG_NewPointerObj(
                            $1[i], SWIGTYPE_p_TF_Operation, 0));
  }
}


////////////////////////////////////////////////////////////////////////////////
// BEGIN TYPEMAPS FOR tensorflow::TF_Run_wrapper()
////////////////////////////////////////////////////////////////////////////////

// Converts a python list of strings to NameVector.
// Has multiple users including feeds/fetches names and function output names
%typemap(in) const tensorflow::NameVector& (
    tensorflow::NameVector temp,
    tensorflow::Safe_PyObjectPtr temp_string_list(
        tensorflow::make_safe(static_cast<PyObject*>(nullptr)))) {
  if (!PyList_Check($input)) {
    SWIG_exception_fail(
        SWIG_TypeError,
        tensorflow::strings::Printf(
            "Expected a python list for conversion "
            "to tensorflow::NameVector but got %s",
            Py_TYPE($input)->tp_name).c_str());
  }

  Py_ssize_t len = PyList_Size($input);

  temp_string_list = tensorflow::make_safe(PyList_New(len));
  if (!temp_string_list) {
    SWIG_exception_fail(
        SWIG_MemoryError,
        tensorflow::strings::Printf("Failed to create a list of size %zd",
                                    len).c_str());
  }

  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject* elem = PyList_GetItem($input, i);
    if (!elem) {
      SWIG_fail;
    }

    // Keep a reference to the string in case the incoming list is modified.
    PyList_SET_ITEM(temp_string_list.get(), i, elem);
    Py_INCREF(elem);

    char* string_elem = PyBytes_AsString(elem);
    if (!string_elem) {
      SWIG_exception_fail(
          SWIG_TypeError,
          tensorflow::strings::Printf(
              "Element %zd was of type %s instead of a string",
              i, Py_TYPE(elem)->tp_name).c_str());
    }

    // TODO(mrry): Avoid copying the fetch name in, if this impacts performance.
    temp.push_back(string_elem);
  }
  $1 = &temp;
}

// Define temporaries for the argout outputs.
%typemap(in, numinputs=0) tensorflow::PyObjectVector* out_values (
    tensorflow::PyObjectVector temp) {
  $1 = &temp;
}
// TODO(iga): move this and the corresponding typemap(argout) to
// tf_sessionrun_wrapper.i once we get rid of this code for DeprecatedSession.
%typemap(in, numinputs=0) char** out_handle (
    char* temp) {
  $1 = &temp;
}

// Build a Python list of outputs and return it.
%typemap(argout) tensorflow::PyObjectVector* out_values {
  std::vector<tensorflow::Safe_PyObjectPtr> out_values_safe;
  for (size_t i = 0; i < $1->size(); ++i) {
    out_values_safe.emplace_back(tensorflow::make_safe($1->at(i)));
  }

  $result = PyList_New($1->size());
  if (!$result) {
    SWIG_exception_fail(
        SWIG_MemoryError,
        tensorflow::strings::Printf("Failed to create a list of size %zd",
                                    $1->size()).c_str());
  }

  for (size_t i = 0; i < $1->size(); ++i) {
    PyList_SET_ITEM($result, i, $1->at(i));
    out_values_safe[i].release();
  }
}

// Return the handle as a python string object.
%typemap(argout) char** out_handle {
%#if PY_MAJOR_VERSION < 3
  $result = PyString_FromStringAndSize(
%#else
  $result = PyUnicode_FromStringAndSize(
%#endif
    *$1, *$1 == nullptr ? 0 : strlen(*$1));
  delete[] *$1;
}

////////////////////////////////////////////////////////////////////////////////
// END TYPEMAPS FOR tensorflow::TF_Run_wrapper()
////////////////////////////////////////////////////////////////////////////////

// Typemap for TF_Status* inputs that automatically unwraps a ScopedTFStatus.
// This can also handle a wrapped TF_Status* input.
%typemap(in) (TF_Status*) {
  PyObject* wrapped_tf_status;
  if (strcmp(Py_TYPE($input)->tp_name, "ScopedTFStatus") == 0) {
    DCHECK(PyObject_HasAttrString($input, "status"))
        << "ScopedTFStatus.status not found! Do you need to modify "
           "tf_session.i?";
    wrapped_tf_status = PyObject_GetAttrString($input, "status");
  } else {
    // Assume wrapped TF_Status*
    wrapped_tf_status = $input;
  }
  DCHECK_EQ(strcmp(Py_TYPE(wrapped_tf_status)->tp_name, "SwigPyObject"), 0)
      << Py_TYPE(wrapped_tf_status)->tp_name;

  // The following is the default SWIG code generated for TF_Status*
  void* tf_status = nullptr;
  int r = SWIG_ConvertPtr(wrapped_tf_status, &tf_status,
                          $descriptor(TF_Status*), 0 | 0);
  if (!SWIG_IsOK(r)) {
    SWIG_exception_fail(
        SWIG_ArgError(r),
        "in method '_TF_DeleteStatus', argument 1 of type 'TF_Status *'");
  }
  $1 = reinterpret_cast<TF_Status*>(tf_status);
}

// Typemap for functions that return a TF_Buffer struct. This typemap creates a
// Python string from the TF_Buffer and returns it. The TF_Buffer.data string
// is not expected to be NULL-terminated, and TF_Buffer.length does not count
// the terminator.
%typemap(out) TF_Buffer (TF_GetOpList,TF_GetBuffer) {
  $result = PyBytes_FromStringAndSize(
      reinterpret_cast<const char*>($1.data), $1.length);
}

%inline %{
// Helper function to convert a Python list of Tensors to a C++ vector of
// TF_Outputs.
//
// Returns true if successful. Otherwise, returns false and sets error_msg.
bool PyTensorListToVector(PyObject* py_tensor_list,
                          std::vector<TF_Output>* vec,
                          string* error_msg) {
  if (!PyList_Check(py_tensor_list)) {
    *error_msg = "expected Python list.";
    return false;
  }
  size_t size = PyList_Size(py_tensor_list);
  for (int i = 0; i < size; ++i) {
    PyObject* item = PyList_GetItem(py_tensor_list, i);
    TF_Output* input_ptr;
    if (!SWIG_IsOK(SWIG_ConvertPtr(item, reinterpret_cast<void**>(&input_ptr),
                                   SWIGTYPE_p_TF_Output, 0))) {
      *error_msg = "expected Python list of wrapped TF_Output objects. "
          "Found python list of something else.";
      return false;
    }
    vec->push_back(*input_ptr);
  }
  return true;
}
%}

// Converts input Python list of wrapped TF_Outputs into a single array
%typemap(in) (const TF_Output* inputs, int num_inputs)
    (std::vector<TF_Output> inputs) {
  string error_msg;
  if (!PyTensorListToVector($input, &inputs, &error_msg)) {
    SWIG_exception_fail(SWIG_TypeError, ("$symname: " + error_msg).c_str());
  }
  $1 = inputs.data();
  $2 = inputs.size();
}

// TODO(skyewm): SWIG emits a warning for the const char* in TF_WhileParams,
// skip for now
%ignore TF_WhileParams;
%ignore TF_NewWhile;
%ignore TF_FinishWhile;
%ignore TF_AbortWhile;

// These are defined below, avoid duplicate definitions
%ignore TF_Run;
%ignore TF_PRun;
%ignore TF_PRunSetup;

// We use TF_SessionRun_wrapper instead of TF_SessionRun
%ignore TF_SessionRun;
%unignore TF_SessionRun_wrapper;
// The %exception block above releases the Python GIL for the length of each
// wrapped method. We disable this behavior for TF_SessionRun_wrapper because it
// uses Python method(s) that expect the GIL to be held (at least
// PyArray_Return, maybe others).
%noexception TF_SessionRun_wrapper;

// We use TF_SessionPRunSetup_wrapper instead of TF_SessionPRunSetup
%ignore TF_SessionPRunSetup;
%unignore TF_SessionPRunSetup_wrapper;
// See comment for "%noexception TF_SessionRun_wrapper;"
%noexception TF_SessionPRunSetup_wrapper;

// We use TF_SessionPRun_wrapper instead of TF_SessionPRun
%ignore TF_SessionPRun;
%unignore TF_SessionPRun_wrapper;
// See comment for "%noexception TF_SessionRun_wrapper;"
%noexception TF_SessionPRun_wrapper;

%rename("_TF_SetTarget") TF_SetTarget;
%rename("_TF_SetConfig") TF_SetConfig;
%rename("_TF_NewSessionOptions") TF_NewSessionOptions;

%include "tensorflow/c/c_api.h"
%include "tensorflow/c/python_api.h"


%ignoreall
%insert("python") %{
  def TF_NewSessionOptions(target=None, config=None):
    # NOTE: target and config are validated in the session constructor.
    opts = _TF_NewSessionOptions()
    if target is not None:
      _TF_SetTarget(opts, target)
    if config is not None:
      from tensorflow.python.framework import errors
      with errors.raise_exception_on_not_ok_status() as status:
        config_str = config.SerializeToString()
        _TF_SetConfig(opts, config_str, status)
    return opts
%}

// Include the wrapper for TF_Run from tf_session_helper.h.

// The %exception block above releases the Python GIL for the length
// of each wrapped method. We disable this behavior for TF_Run
// because it uses the Python allocator.
%noexception tensorflow::TF_Run_wrapper;
%rename(TF_Run) tensorflow::TF_Run_wrapper;
%unignore tensorflow;
%unignore TF_Run;
%unignore EqualGraphDefWrapper;

// Include the wrapper for TF_PRunSetup from tf_session_helper.h.

// The %exception block above releases the Python GIL for the length
// of each wrapped method. We disable this behavior for TF_PRunSetup
// because it uses the Python allocator.
%noexception tensorflow::TF_PRunSetup_wrapper;
%rename(TF_PRunSetup) tensorflow::TF_PRunSetup_wrapper;
%unignore tensorflow;
%unignore TF_PRunSetup;

// Include the wrapper for TF_PRun from tf_session_helper.h.

// The %exception block above releases the Python GIL for the length
// of each wrapped method. We disable this behavior for TF_PRun
// because it uses the Python allocator.
%noexception tensorflow::TF_PRun_wrapper;
%rename(TF_PRun) tensorflow::TF_PRun_wrapper;
%unignore tensorflow;
%unignore TF_PRun;

%unignore tensorflow::TF_Reset_wrapper;
%insert("python") %{
def TF_Reset(target, containers=None, config=None):
  from tensorflow.python.framework import errors
  opts = TF_NewSessionOptions(target=target, config=config)
  try:
    with errors.raise_exception_on_not_ok_status() as status:
      TF_Reset_wrapper(opts, containers, status)
  finally:
    TF_DeleteSessionOptions(opts)
%}

%include "tensorflow/python/client/tf_session_helper.h"

%unignoreall
