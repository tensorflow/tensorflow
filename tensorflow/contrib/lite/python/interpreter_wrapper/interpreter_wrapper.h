/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_PYTHON_INTERPRETER_WRAPPER_INTERPRETER_WRAPPER_H_
#define TENSORFLOW_CONTRIB_LITE_PYTHON_INTERPRETER_WRAPPER_INTERPRETER_WRAPPER_H_

#include <memory>
#include <string>
#include <vector>

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

class InterpreterWrapper {
 public:
  // SWIG caller takes ownership of pointer.
  static InterpreterWrapper* CreateWrapperCPP(const char* model_path);

  ~InterpreterWrapper();
  bool AllocateTensors();
  bool Invoke();

  PyObject* InputIndices() const;
  PyObject* OutputIndices() const;
  bool ResizeInputTensor(int i, PyObject* value);

  std::string TensorName(int i) const;
  PyObject* TensorType(int i) const;
  PyObject* TensorSize(int i) const;
  bool SetTensor(int i, PyObject* value);
  PyObject* GetTensor(int i) const;

 private:
  InterpreterWrapper(std::unique_ptr<tflite::FlatBufferModel> model);

  // InterpreterWrapper is not copyable or assignable. We avoid the use of
  // InterpreterWrapper() = delete here for SWIG compatibility.
  InterpreterWrapper();
  InterpreterWrapper(const InterpreterWrapper& rhs);

  const std::unique_ptr<tflite::FlatBufferModel> model_;
  const std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver_;
  const std::unique_ptr<tflite::Interpreter> interpreter_;
};

}  // namespace interpreter_wrapper
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_PYTHON_INTERPRETER_WRAPPER_INTERPRETER_WRAPPER_H_
