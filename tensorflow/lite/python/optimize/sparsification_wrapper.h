/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_PYTHON_OPTIMIZE_SPARSIFICATION_WRAPPER_H_
#define TENSORFLOW_LITE_PYTHON_OPTIMIZE_SPARSIFICATION_WRAPPER_H_

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

class FlatBufferModel;

namespace interpreter_wrapper {
class PythonErrorReporter;
}  // namespace interpreter_wrapper

namespace sparsification_wrapper {

class SparsificationWrapper {
 public:
  // SWIG caller takes ownership of pointer.
  static SparsificationWrapper* CreateWrapperCPPFromBuffer(PyObject* data);
  ~SparsificationWrapper();

  PyObject* SparsifyModel();

 private:
  // SparsificationWrapper is not copyable or assignable. We avoid the use of
  // SparsificationWrapper() = delete here for SWIG compatibility.
  SparsificationWrapper(
      std::unique_ptr<tflite::FlatBufferModel> model,
      std::unique_ptr<tflite::interpreter_wrapper::PythonErrorReporter>
          error_reporter);
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::interpreter_wrapper::PythonErrorReporter>
      error_reporter_;
};

}  // namespace sparsification_wrapper
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PYTHON_OPTIMIZE_SPARSIFICATION_WRAPPER_H_
