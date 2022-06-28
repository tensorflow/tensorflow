/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/python/interpreter_wrapper/python_error_reporter.h"

#include <string>

namespace tflite {
namespace interpreter_wrapper {

// Report an error message
int PythonErrorReporter::Report(const char* format, va_list args) {
  char buf[1024];
  int formatted = vsnprintf(buf, sizeof(buf), format, args);
  buffer_ << buf;
  return formatted;
}

// Set's a Python runtime exception with the last error.
PyObject* PythonErrorReporter::exception() {
  std::string last_message = message();
  PyErr_SetString(PyExc_RuntimeError, last_message.c_str());
  return nullptr;
}

// Gets the last error message and clears the buffer.
std::string PythonErrorReporter::message() {
  std::string value = buffer_.str();
  buffer_.clear();
  return value;
}
}  // namespace interpreter_wrapper
}  // namespace tflite
