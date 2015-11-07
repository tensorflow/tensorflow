%include "tensorflow/python/platform/base.i"

%{

#include "numpy/arrayobject.h"

#include "tensorflow/python/client/tf_session_helper.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/status.h"

%}

// Implements the StatusNotOK exception.
%import(module="tensorflow.python.pywrap_tensorflow") "tensorflow/python/lib/core/status.i"

// Required to use PyArray_* functions.
%include "tensorflow/python/platform/numpy.i"
%init %{
import_array();
%}

// Release the Python GIL for the duration of most methods.
%exception {
  Py_BEGIN_ALLOW_THREADS;
  $action
  Py_END_ALLOW_THREADS;
}

// Proto input arguments to C API functions are passed as a (const
// void*, size_t) pair. In Python, typemap these to a single string
// argument.
%typemap(in) (const void* proto, size_t proto_len) {
  char* c_string;
  Py_ssize_t py_size;
  if (PyBytes_AsStringAndSize($input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    SWIG_fail;
  }
  $1 = static_cast<void*>(c_string);
  $2 = static_cast<size_t>(py_size);
}

////////////////////////////////////////////////////////////////////////////////
// BEGIN TYPEMAPS FOR tensorflow::TF_Run_wrapper()
////////////////////////////////////////////////////////////////////////////////

// The wrapper takes a vector of pairs of feed names and feed
// values. In Python this is represented as dictionary mapping strings
// to numpy arrays.
%typemap(in) const tensorflow::FeedVector& inputs (
    tensorflow::FeedVector temp,
    tensorflow::Safe_PyObjectPtr temp_string_list(tensorflow::make_safe(nullptr)),
    tensorflow::Safe_PyObjectPtr temp_array_list(tensorflow::make_safe(nullptr))) {
  if (!PyDict_Check($input)) {
    SWIG_fail;
  }

  temp_string_list = tensorflow::make_safe(PyList_New(0));
  if (!temp_string_list) {
    SWIG_fail;
  }
  temp_array_list = tensorflow::make_safe(PyList_New(0));
  if (!temp_array_list) {
    SWIG_fail;
  }

  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  while (PyDict_Next($input, &pos, &key, &value)) {
    const char* key_string = PyString_AsString(key);
    if (!key_string) {
      SWIG_fail;
    }

    // The ndarray must be stored as contiguous bytes in C (row-major) order.
    PyObject* array_object = PyArray_FromAny(
        value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr);
    if (!array_object) {
      SWIG_fail;
    }
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_object);

    // Keep a reference to the key and the array, in case the incoming dict is
    // modified, and/or to avoid leaking references on failure.
    if (PyList_Append(temp_string_list.get(), key) == -1) {
      SWIG_fail;
    }
    if (PyList_Append(temp_array_list.get(), array_object) == -1) {
      SWIG_fail;
    }

    temp.push_back(std::make_pair(key_string, array));
  }
  $1 = &temp;
}

// The wrapper also takes a list of fetch and target names.  In Python this is
// represented as a list of strings.
%typemap(in) const tensorflow::NameVector& (
    tensorflow::NameVector temp,
    tensorflow::Safe_PyObjectPtr temp_string_list(tensorflow::make_safe(nullptr))) {
  if (!PyList_Check($input)) {
    SWIG_fail;
  }

  Py_ssize_t len = PyList_Size($input);

  temp_string_list = tensorflow::make_safe(PyList_New(len));
  if (!temp_string_list) {
    SWIG_fail;
  }

  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject* elem = PyList_GetItem($input, i);
    if (!elem) {
      SWIG_fail;
    }

    // Keep a reference to the string in case the incoming list is modified.
    PyList_SET_ITEM(temp_string_list.get(), i, elem);
    Py_INCREF(elem);

    const char* fetch_name = PyString_AsString(elem);
    if (!fetch_name) {
      PyErr_SetString(PyExc_TypeError,
                      "a fetch or target name was not a string");
      SWIG_fail;
    }

    // TODO(mrry): Avoid copying the fetch name in, if this impacts performance.
    temp.push_back(fetch_name);
  }
  $1 = &temp;
}


// The wrapper has two outputs: a tensorflow::Status, and a vector of
// PyObjects containing the fetch results (iff the status is OK). Since
// the interpretation of the vector depends on the status, we define
// them as two consecutive out arguments, so that they can be accessed
// together in a typemap.

// Define temporaries for the argout outputs.
%typemap(in, numinputs=0) tensorflow::Status* out_status (
    tensorflow::Status temp) {
  $1 = &temp;
}
%typemap(in, numinputs=0) tensorflow::PyObjectVector* out_values (
    tensorflow::PyObjectVector temp) {
  $1 = &temp;
}

// Raise a StatusNotOK exception if the out_status is not OK;
// otherwise build a Python list of outputs and return it.
%typemap(argout, fragment="StatusNotOK") (
    tensorflow::Status* out_status, tensorflow::PyObjectVector* out_values) {
  if (!$1->ok()) {
    RaiseStatusNotOK(*$1, $descriptor(tensorflow::Status*));
    SWIG_fail;
  } else {
    tensorflow::Safe_PyObjectVector out_values_safe;
    for (int i = 0; i < $2->size(); ++i) {
      out_values_safe.emplace_back(tensorflow::make_safe($2->at(i)));
    }

    $result = PyList_New($2->size());
    if (!$result) {
      SWIG_fail;
    }

    for (int i = 0; i < $2->size(); ++i) {
      PyList_SET_ITEM($result, i, $2->at(i));
      out_values_safe[i].release();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// END TYPEMAPS FOR tensorflow::TF_Run_wrapper()
////////////////////////////////////////////////////////////////////////////////



// Include the functions from tensor_c_api.h, except TF_Run.
%ignoreall
%unignore TF_Code;
%unignore TF_Status;
%unignore TF_NewStatus;
%unignore TF_DeleteStatus;
%unignore TF_GetCode;
%unignore TF_Message;
%unignore TF_SessionOptions;
%rename("_TF_SetTarget") TF_SetTarget;
%rename("_TF_SetConfig") TF_SetConfig;
%rename("_TF_NewSessionOptions") TF_NewSessionOptions;
%unignore TF_DeleteSessionOptions;
%unignore TF_NewSession;
%unignore TF_CloseSession;
%unignore TF_DeleteSession;
%unignore TF_ExtendGraph;
%include "tensorflow/core/public/tensor_c_api.h"
%ignoreall

%insert("python") %{
  def TF_NewSessionOptions(target=None, config=None):
    opts = _TF_NewSessionOptions()
    if target is not None:
      _TF_SetTarget(opts, target)
    if config is not None:
      from tensorflow.core.framework import config_pb2
      if not isinstance(config, config_pb2.ConfigProto):
        raise TypeError("Expected config_pb2.ConfigProto, "
                        "but got %s" % type(config))
      status = TF_NewStatus()
      config_str = config.SerializeToString()
      _TF_SetConfig(opts, config_str, len(config_str), status)
      if TF_GetCode(status) != 0:
        raise ValueError(TF_Message(status))
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

%include "tensorflow/python/client/tf_session_helper.h"

%unignoreall
