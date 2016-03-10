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

// SWIG wrapper for lib::tensorflow::Status

%include "tensorflow/python/platform/base.i"
%include "tensorflow/python/lib/core/strings.i"

%apply int { tensorflow::error::Code };  // Treat the enum as an integer.

%{
#include "tensorflow/core/lib/core/status.h"
%}

%typemap(out, fragment="StatusNotOK") tensorflow::Status {
  if ($1.ok()) {
    $result = SWIG_Py_Void();
  } else {
    RaiseStatusNotOK($1, $descriptor(tensorflow::Status*));
    SWIG_fail;
  }
}

%init %{
// Setup the StatusNotOK exception class.
PyObject *pywrap_status = PyImport_ImportModuleNoBlock(
    "tensorflow.python.pywrap_tensorflow");
if (pywrap_status) {
  PyObject *exception = PyErr_NewException(
      "tensorflow.python.pywrap_tensorflow.StatusNotOK",
      NULL, NULL);
  if (exception) {
    PyModule_AddObject(pywrap_status, "StatusNotOK", exception);  // Steals ref.
  }
  Py_DECREF(pywrap_status);
}
%}

%fragment("StatusNotOK", "header") %{
#include "tensorflow/core/lib/core/status.h"

namespace {
// Initialized on the first call to RaiseStatusNotOK().
static PyObject *StatusNotOKError = nullptr;

inline void Py_DECREF_wrapper(PyObject *o) { Py_DECREF(o); }
typedef std::unique_ptr<PyObject, decltype(&Py_DECREF_wrapper)> SafePyObjectPtr;
SafePyObjectPtr make_safe(PyObject* o) {
  return SafePyObjectPtr(o, Py_DECREF_wrapper);
}

void RaiseStatusNotOK(const tensorflow::Status& status, swig_type_info *type) {
  const int code = status.code();
  string fullmsg = status.ToString();

  PyObject *exception = nullptr;

  // We're holding the Python GIL, so we don't need to synchronize
  // access to StatusNotOKError with a Mutex of our own.
  if (!StatusNotOKError) {
    PyObject *cls = nullptr;
    auto pywrap = make_safe(PyImport_ImportModule(
        "tensorflow.python.pywrap_tensorflow"));
    if (pywrap) {
      cls = PyObject_GetAttrString(pywrap.get(), "StatusNotOK");
    }
    if (!cls) {
      cls = Py_None;
      Py_INCREF(cls);
    }
    StatusNotOKError = cls;
  }

  if (StatusNotOKError != Py_None) {
    auto fullmsg_ptr = make_safe(_SwigSimpleStr_FromString(fullmsg));
    auto exception_ptr = make_safe(PyObject_CallFunctionObjArgs(
        StatusNotOKError, fullmsg_ptr.get(), NULL));
    exception = exception_ptr.get();
    if (exception) {
      auto pycode = make_safe(PyInt_FromLong(static_cast<long>(code)));
      auto pymsg = make_safe(_SwigSimpleStr_FromString(status.error_message()));
      auto pystatus = make_safe(SWIG_NewPointerObj(
          SWIG_as_voidptr(new tensorflow::Status(status)), type, SWIG_POINTER_OWN));
      PyObject_SetAttrString(exception, "code", pycode.get());
      PyObject_SetAttrString(exception, "error_message", pymsg.get());
      PyErr_SetObject(StatusNotOKError, exception);
    }
  }
  if (!exception) {
    fullmsg =
        ("could not construct StatusNotOK (original error "
         " was: " +
         fullmsg + ")");
    PyErr_SetString(PyExc_SystemError, fullmsg.c_str());
  }
}

}  // namespace
%}

%ignoreall

%unignore tensorflow;
%unignore tensorflow::lib;
%unignore tensorflow::Status;
%unignore tensorflow::Status::~Status;
%ignore tensorflow::Status::operator=;

%include "tensorflow/core/lib/core/status.h"

%unignoreall
