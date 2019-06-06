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
#ifndef TENSORFLOW_LITE_PYTHON_OPTIMIZE_CALIBRATION_WRAPPER_H_
#define TENSORFLOW_LITE_PYTHON_OPTIMIZE_CALIBRATION_WRAPPER_H_

#include <memory>
#include <string>
#include <vector>

// Place `<locale>` before <Python.h> to avoid build failures in macOS.
#include <locale>

// The empty line above is on purpose as otherwise clang-format will
// automatically move <Python.h> before <locale>.
#include <Python.h>

// We forward declare TFLite classes here to avoid exposing them to SWIG.
namespace tflite {
namespace ops {
namespace builtin {
class BuiltinOpResolver;
}  // namespace builtin
}  // namespace ops

class FlatBufferModel;
class Interpreter;

namespace interpreter_wrapper {
class PythonErrorReporter;
}  // namespace interpreter_wrapper

namespace optimize {
namespace calibration {
class CalibrationReader;
}  // namespace calibration
}  // namespace optimize

namespace calibration_wrapper {

class CalibrationWrapper {
 public:
  // SWIG caller takes ownership of pointer.
  static CalibrationWrapper* CreateWrapperCPPFromBuffer(PyObject* data);
  ~CalibrationWrapper();

  PyObject* Prepare();

  PyObject* FeedTensor(PyObject* input_value);

  PyObject* QuantizeModel(int input_py_type, int output_py_type,
                          bool allow_float);

 private:
  // CalibrationWrapper is not copyable or assignable. We avoid the use of
  // CalibrationWrapper() = delete here for SWIG compatibility.
  CalibrationWrapper(
      std::unique_ptr<tflite::Interpreter> interpreter,
      std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver,
      std::unique_ptr<tflite::interpreter_wrapper::PythonErrorReporter>
          error_reporter,
      std::unique_ptr<tflite::FlatBufferModel> model,
      std::unique_ptr<tflite::optimize::calibration::CalibrationReader> reader);

  CalibrationWrapper(const CalibrationWrapper& rhs);

  PyObject* SetTensor(int index, PyObject* value);

  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::interpreter_wrapper::PythonErrorReporter>
      error_reporter_;
  std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::optimize::calibration::CalibrationReader> reader_;
};

}  // namespace calibration_wrapper
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PYTHON_OPTIMIZE_CALIBRATION_WRAPPER_H_
