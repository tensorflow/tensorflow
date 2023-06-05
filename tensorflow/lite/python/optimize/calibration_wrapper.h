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

#include <functional>
#include <memory>
#include <string>
#include <vector>

// Place `<locale>` before <Python.h> to avoid build failures in macOS.
#include <locale>

// The empty line above is on purpose as otherwise clang-format will
// automatically move <Python.h> before <locale>.
#include <Python.h>

#include "tensorflow/lite/core/interpreter.h"

// We forward declare TFLite classes here to avoid exposing them to SWIG.
namespace tflite {
namespace ops {
namespace builtin {
class BuiltinOpResolver;
}  // namespace builtin
}  // namespace ops

namespace impl {
class FlatBufferModel;
}

namespace interpreter_wrapper {
class PythonErrorReporter;
}  // namespace interpreter_wrapper

namespace optimize {
namespace calibration {
class CalibrationReader;
}  // namespace calibration
}  // namespace optimize

namespace calibration_wrapper {

PyObject* AddIntermediateTensors(PyObject* data);

class CalibrationWrapper {
 public:
  // SWIG caller takes ownership of pointer.
  static CalibrationWrapper* CreateWrapperCPPFromBuffer(
      PyObject* data, const std::vector<std::string>& registerers_by_name,
      const std::vector<std::function<void(uintptr_t)>>& registerers_by_func,
      std::string* error_msg);
  ~CalibrationWrapper();

  // Allocates the primary subgraph's tensors.
  PyObject* Prepare();

  // Allocates the tensors of the given signature, defined by the signature
  // key.
  PyObject* Prepare(std::string signature_key);

  // Allocates the primary subgraph's tensors with the given input shapes.
  PyObject* Prepare(PyObject* input_shapes);

  // Allocates the tensors of the given signature with the given input
  // shapes, defined by the signature key.
  PyObject* Prepare(PyObject* input_shapes, std::string signature_key);

  // Sets the given input tensors to the primary subgraph.
  PyObject* FeedTensor(PyObject* input_value);

  // Sets the given input tensor to the given signature, defined by the
  // signature key.
  PyObject* FeedTensor(PyObject* input_value, std::string signature_key);

  // Allows quantizing only the operator that produces the tensor.
  PyObject* QuantizeModel(int input_py_type, int output_py_type,
                          bool allow_float, int activations_py_type,
                          int bias_py_type);

  // Allows quantizing only the operator that produces the tensor with name
  // operator_output_name. (This can be used to help debug.).
  // TODO(suharshs): Allow providing multiple names.
  PyObject* QuantizeModel(int input_py_type, int output_py_type,
                          bool allow_float, const char* operator_output_name);

  // Disables per-channel quantization, can be used to produce smaller
  // models but may cause accuracy issues.
  PyObject* QuantizeModel(int input_py_type, int output_py_type,
                          bool allow_float, int activations_py_type,
                          int bias_py_type, bool disable_per_channel);

  // Writes the in-memory calibration results to the model flatbuffer. The
  // produced model is as same as the original input model, but the min/max
  // in the quantization field.
  PyObject* Calibrate();

 private:
  // CalibrationWrapper is not copyable or assignable. We avoid the use of
  // CalibrationWrapper() = delete here for SWIG compatibility.
  CalibrationWrapper(
      std::unique_ptr<tflite::Interpreter> interpreter,
      std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver,
      std::unique_ptr<tflite::interpreter_wrapper::PythonErrorReporter>
          error_reporter,
      std::unique_ptr<tflite::impl::FlatBufferModel> model,
      std::unique_ptr<tflite::optimize::calibration::CalibrationReader> reader,
      std::unique_ptr<std::string> model_str_);

  CalibrationWrapper(const CalibrationWrapper& rhs);

  PyObject* SetTensor(int index, PyObject* value);
  PyObject* SetTensor(int index, PyObject* value, std::string signature_key);

  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::interpreter_wrapper::PythonErrorReporter>
      error_reporter_;
  std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver_;
  std::unique_ptr<tflite::impl::FlatBufferModel> model_;
  std::unique_ptr<tflite::optimize::calibration::CalibrationReader> reader_;
  std::unique_ptr<std::string> model_str_;
};

}  // namespace calibration_wrapper
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PYTHON_OPTIMIZE_CALIBRATION_WRAPPER_H_
