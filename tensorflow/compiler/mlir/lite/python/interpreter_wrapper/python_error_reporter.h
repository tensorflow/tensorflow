/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_INTERPRETER_WRAPPER_PYTHON_ERROR_REPORTER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_INTERPRETER_WRAPPER_PYTHON_ERROR_REPORTER_H_

#include <Python.h>

#include <cstdarg>
#include <sstream>
#include <string>

#include "tensorflow/compiler/mlir/lite/stateful_error_reporter.h"

namespace tflite_migration {
namespace interpreter_wrapper {

class PythonErrorReporter : public tflite_migration::StatefulErrorReporter {
 public:
  PythonErrorReporter() = default;

  // Report an error message
  int Report(const char* format, va_list args) override;

  // Sets a Python runtime exception with the last error and
  // clears the error message buffer.
  PyObject* exception();

  // Gets the last error message and clears the buffer.
  std::string message() override;

 private:
  std::stringstream buffer_;
};

}  // namespace interpreter_wrapper
}  // namespace tflite_migration
#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_INTERPRETER_WRAPPER_PYTHON_ERROR_REPORTER_H_
