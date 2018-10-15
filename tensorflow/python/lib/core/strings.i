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

// Wrapper functions to provide a scripting-language-friendly interface
// to our string libraries.
//
// NOTE: as of 2005-01-13, this SWIG file is not used to generate a pywrap
//       library for manipulation of various string-related types or access
//       to the special string functions (Python has plenty). This SWIG file
//       should be %import'd so that other SWIG wrappers have proper access
//       to the types in //strings (such as the StringPiece object). We may
//       generate a pywrap at some point in the future.
//
// NOTE: (Dan Ardelean) as of 2005-11-15 added typemaps to convert Java String
//       arguments to C++ StringPiece& objects. This is required because a
//       StringPiece class does not make sense - the code SWIG generates for a
//       StringPiece class is useless, because it releases the buffer set in
//       StringPiece after creating the object. C++ StringPiece objects rely on
//       the buffer holding the data being allocated externally.

// NOTE: for now, we'll just start with what is needed, and add stuff
//       as it comes up.

%{
#include "absl/strings/string_view.h"

// Handles str in Python 2, bytes in Python 3.
// Returns true on success, false on failure.
bool _BytesToStringPiece(PyObject* obj, absl::string_view* result) {
  if (obj == Py_None) {
    *result = absl::string_view();
  } else {
    char* ptr;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(obj, &ptr, &len) == -1) {
      // Python has raised an error (likely TypeError or UnicodeEncodeError).
      return false;
    }
    *result = absl::string_view(ptr, len);
  }
  return true;
}
%}

%typemap(typecheck) absl::string_view = char *;
%typemap(typecheck) const absl::string_view & = char *;

// "absl::string_view" arguments must be specified as a 'str' or 'bytes' object.
%typemap(in) absl::string_view {
  if (!_BytesToStringPiece($input, &$1)) SWIG_fail;
}

// "const absl::string_view&" arguments can be provided the same as
// "absl::string_view", whose typemap is defined above.
%typemap(in) const absl::string_view & (absl::string_view temp) {
  if (!_BytesToStringPiece($input, &temp)) SWIG_fail;
  $1 = &temp;
}

// C++ functions returning absl::string_view will simply return bytes in
// Python, or None if the StringPiece contained a NULL pointer.
%typemap(out) absl::string_view {
  if ($1.data()) {
    $result = PyBytes_FromStringAndSize($1.data(), $1.size());
  } else {
    Py_INCREF(Py_None);
    $result = Py_None;
  }
}

// Converts a C++ string vector to a list of Python bytes objects.
%typemap(out) std::vector<string> {
  const int size = $1.size();
  auto temp_string_list = tensorflow::make_safe(PyList_New(size));
  if (!temp_string_list) {
    SWIG_fail;
  }
  std::vector<tensorflow::Safe_PyObjectPtr> converted;
  converted.reserve(size);
  for (const string& op : $1) {
    // Always treat strings as bytes, consistent with the typemap
    // for string.
    PyObject* py_str = PyBytes_FromStringAndSize(op.data(), op.size());
    if (!py_str) {
      SWIG_fail;
    }
    converted.emplace_back(tensorflow::make_safe(py_str));
  }
  for (int i = 0; i < converted.size(); ++i) {
    PyList_SET_ITEM(temp_string_list.get(), i, converted[i].release());
  }
  $result = temp_string_list.release();
}
