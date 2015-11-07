// SWIG wrapper for lib::tensorflow::Status

%include "tensorflow/python/platform/base.i"
%include "tensorflow/python/lib/core/strings.i"

%apply int { tensorflow::error::Code };  // Treat the enum as an integer.

%{
#include "tensorflow/core/public/status.h"
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
#include "tensorflow/core/public/status.h"

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
    auto fullmsg_ptr = make_safe(_SwigString_FromString(fullmsg));
    auto exception_ptr = make_safe(PyObject_CallFunctionObjArgs(
        StatusNotOKError, fullmsg_ptr.get(), NULL));
    exception = exception_ptr.get();
    if (exception) {
      auto pycode = make_safe(PyInt_FromLong(static_cast<long>(code)));
      auto pymsg = make_safe(_SwigString_FromString(status.error_message()));
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
%unignore tensorflow::Status::Status;
%unignore tensorflow::Status::Status(tensorflow::error::Code, StringPiece);
%unignore tensorflow::Status::~Status;
%unignore tensorflow::Status::code;
%unignore tensorflow::Status::ok;
%unignore tensorflow::Status::error_message;
%unignore tensorflow::Status::ToString;
%ignore tensorflow::Status::operator=;

%rename(__str__) tensorflow::Status::ToString;

%include "tensorflow/core/public/status.h"

%unignoreall
