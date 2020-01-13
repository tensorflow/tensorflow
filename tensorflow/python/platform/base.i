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

// Helper macros and typemaps for use in TensorFlow swig files.
//
%{
  #include <memory>
  #include <vector>
  #include "tensorflow/c/tf_status.h"
  #include "tensorflow/core/platform/types.h"
  #include "tensorflow/c/tf_datatype.h"
  #include "tensorflow/python/lib/core/py_exception_registry.h"

  using tensorflow::int64;
  using tensorflow::uint64;
  using tensorflow::string;

  template<class T>
      bool _PyObjAs(PyObject *pystr, T* cstr) {
    T::undefined;  // You need to define specialization _PyObjAs<T>
    return false;
  }

  template<class T>
      PyObject *_PyObjFrom(const T& c) {
    T::undefined;  // You need to define specialization _PyObjFrom<T>
    return NULL;
  }

  template<>
      bool _PyObjAs(PyObject *pystr, std::string* cstr) {
    char *buf;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(pystr, &buf, &len) == -1) return false;
    if (cstr) cstr->assign(buf, len);
    return true;
  }
  template<>
      PyObject* _PyObjFrom(const std::string& c) {
    return PyBytes_FromStringAndSize(c.data(), c.size());
  }

  PyObject* _SwigBytes_FromString(const string& s) {
    return PyBytes_FromStringAndSize(s.data(), s.size());
  }

  // The string must be both ASCII and Unicode compatible, so this routine
  // should be used only for error messages and the like.
  PyObject* _SwigSimpleStr_FromString(const string& s) {
#if PY_MAJOR_VERSION < 3
    return PyString_FromStringAndSize(s.data(), s.size());
#else
    return PyUnicode_FromStringAndSize(s.data(), s.size());
#endif
  }

  template <class T>
  bool tf_vector_input_helper(PyObject * seq, std::vector<T> * out,
                              bool (*convert)(PyObject*, T * const)) {
    PyObject *item, *it = PyObject_GetIter(seq);
    if (!it) return false;
    while ((item = PyIter_Next(it))) {
      T elem;
      bool success = convert(item, &elem);
      Py_DECREF(item);
      if (!success) {
        Py_DECREF(it);
        return false;
      }
      if (out) out->push_back(elem);
    }
    Py_DECREF(it);
    return static_cast<bool>(!PyErr_Occurred());
  }
%}

%typemap(in) string {
  if (!_PyObjAs<string>($input, &$1)) return NULL;
}

%typemap(in) const string& (string temp) {
  if (!_PyObjAs<string>($input, &temp)) return NULL;
  $1 = &temp;
}

%typemap(out) int64_t {
  $result = PyLong_FromLongLong($1);
}

%typemap(out) string {
  $result = PyBytes_FromStringAndSize($1.data(), $1.size());
}

%typemap(out) const string& {
  $result = PyBytes_FromStringAndSize($1->data(), $1->size());
}

%typemap(in, numinputs = 0) string* OUTPUT (string temp) {
  $1 = &temp;
}

%typemap(argout) string * OUTPUT {
  PyObject *str = PyBytes_FromStringAndSize($1->data(), $1->length());
  if (!str) SWIG_fail;
  %append_output(str);
}

%typemap(argout) string* INOUT = string* OUTPUT;

%typemap(varout) string {
  $result = PyBytes_FromStringAndSize($1.data(), $1.size());
}

%define _LIST_OUTPUT_TYPEMAP(type, py_converter)
    %typemap(in) std::vector<type>(std::vector<type> temp) {
  if (!tf_vector_input_helper($input, &temp, _PyObjAs<type>)) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "sequence(type) expected");
    return NULL;
  }
  $1 = temp;
}
%typemap(in) const std::vector<type>& (std::vector<type> temp),
   const std::vector<type>* (std::vector<type> temp) {
  if (!tf_vector_input_helper($input, &temp, _PyObjAs<type>)) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "sequence(type) expected");
    return NULL;
  }
  $1 = &temp;
}
%typemap(in,numinputs=0)
std::vector<type>* OUTPUT (std::vector<type> temp),
   hash_set<type>* OUTPUT (hash_set<type> temp),
   set<type>* OUTPUT (set<type> temp) {
  $1 = &temp;
}
%enddef

_LIST_OUTPUT_TYPEMAP(string, _SwigBytes_FromString);
_LIST_OUTPUT_TYPEMAP(long long, PyLong_FromLongLong);
_LIST_OUTPUT_TYPEMAP(unsigned long long, PyLong_FromUnsignedLongLong);
_LIST_OUTPUT_TYPEMAP(unsigned int, PyLong_FromUnsignedLong);

%typemap(in) uint64 {
  // TODO(gps): Check if another implementation
  // from hosting/images/util/image-hosting-utils.swig is better. May be not.
%#if PY_MAJOR_VERSION < 3
  if (PyInt_Check($input)) {
    $1 = static_cast<uint64>(PyInt_AsLong($input));
  } else
%#endif
  if (PyLong_Check($input)) {
    $1 = static_cast<uint64>(PyLong_AsUnsignedLongLong($input));
  } else {
    PyErr_SetString(PyExc_TypeError,
                    "int or long value expected for argument \"$1_name\"");
  }
  // TODO(mrovner): Make consistent use of SWIG_fail vs. return NULL.
  if (PyErr_Occurred()) return NULL;
}

%define _COPY_TYPEMAPS(oldtype, newtype)
    typedef oldtype newtype;
%apply oldtype * OUTPUT { newtype * OUTPUT };
%apply oldtype & OUTPUT { newtype & OUTPUT };
%apply oldtype * INPUT { newtype * INPUT };
%apply oldtype & INPUT { newtype & INPUT };
%apply oldtype * INOUT { newtype * INOUT };
%apply oldtype & INOUT { newtype & INOUT };
%apply std::vector<oldtype> * OUTPUT { std::vector<newtype> * OUTPUT };
%enddef

_COPY_TYPEMAPS(unsigned long long, uint64);
_COPY_TYPEMAPS(long long, int64);
_COPY_TYPEMAPS(unsigned int, mode_t);

// Proto input arguments to C API functions are passed as a (const
// void*, size_t) pair. In Python, typemap these to a single string
// argument.  This typemap does *not* make a copy of the input.
%typemap(in) (const void* proto, size_t proto_len) {
  char* c_string;
  Py_ssize_t py_size;
  // PyBytes_AsStringAndSize() does not copy but simply interprets the input
  if (PyBytes_AsStringAndSize($input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    SWIG_fail;
  }
  $1 = static_cast<void*>(c_string);
  $2 = static_cast<size_t>(py_size);
}

// SWIG macros for explicit API declaration.
// Usage:
//
// %ignoreall
// %unignore SomeName;   // namespace / class / method
// %include "somelib.h"
// %unignoreall  // mandatory closing "bracket"
%define %ignoreall %ignore ""; %enddef
%define %unignore %rename("%s") %enddef
%define %unignoreall %rename("%s") ""; %enddef

#if SWIG_VERSION < 0x030000
// Define some C++11 keywords safe to ignore so older SWIG does not choke.
%define final %enddef
%define override %enddef
#endif


// This was originally included in pywrap_tfe.i, but is used by tf_session.i
%include "tensorflow/c/tf_status.h"
%include "tensorflow/c/tf_datatype.h"

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

%typemap(in) int64_t {
  $1 = PyLong_AsLongLong($input);
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
  tmp = 0;
  $1 = &tmp;
}

%typemap(argout) unsigned char* is_list {
  if (*$1 == 1) {
    PyObject* list = PyList_New(1);
    PyList_SetItem(list, 0, $result);
    $result = list;
  }
}

// Typemaps to automatically raise a Python exception from bad output TF_Status.
// TODO(b/77295559): expand this to all TF_Status* output params and deprecate
// raise_exception_on_not_ok_status (currently it only affects the C API).
%typemap(in, numinputs=0) TF_Status* status {
  $1 = TF_NewStatus();
}

%typemap(freearg) (TF_Status* status) {
 TF_DeleteStatus($1);
}

%typemap(argout) TF_Status* status {
  TF_Code code = TF_GetCode($1);
  if (code != TF_OK) {
    PyObject* exc = tensorflow::PyExceptionRegistry::Lookup(code);
    // Arguments to OpError.
    PyObject* exc_args = Py_BuildValue("sss", nullptr, nullptr, TF_Message($1));
    SWIG_SetErrorObj(exc, exc_args);
    SWIG_fail;
  }
}
