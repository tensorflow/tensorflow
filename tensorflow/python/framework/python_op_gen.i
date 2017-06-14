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
#include "tensorflow/python/framework/python_op_gen.h"
%}

// Input typemap for GetPythonWrappers.
// Accepts a python object of 'bytes' type, and converts it to
// a const char* pointer and size_t length. The default typemap
// going from python bytes to const char* tries to decode the
// contents from utf-8 to unicode for Python version >= 3, but
// we want the bytes to be uninterpreted.
%typemap(in) (const char* op_list_buf, size_t op_list_len) {
  char* c_string;
  Py_ssize_t py_size;
  if (PyBytes_AsStringAndSize($input, &c_string, &py_size) == -1) {
    SWIG_fail;
  }
  $1 = c_string;
  $2 = static_cast<size_t>(py_size);
}


%ignoreall;
%unignore tensorflow::GetPythonWrappers;
%include "tensorflow/python/framework/python_op_gen.h"
