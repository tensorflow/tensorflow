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
#ifndef TENSORFLOW_LITE_PYTHON_INTERPRETER_WRAPPER_INTERPRETER_WRAPPER_H_
#define TENSORFLOW_LITE_PYTHON_INTERPRETER_WRAPPER_INTERPRETER_WRAPPER_H_

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

struct TfLiteDelegate;

// We forward declare TFLite classes here to avoid exposing them to SWIG.
namespace tflite {
class MutableOpResolver;
class FlatBufferModel;

namespace interpreter_wrapper {

class PythonErrorReporter;

class InterpreterWrapper {
 public:
  using Model = FlatBufferModel;

  // SWIG caller takes ownership of pointer.
  static InterpreterWrapper* CreateWrapperCPPFromFile(
      const char* model_path, int op_resolver_id,
      const std::vector<std::string>& registerers, std::string* error_msg,
      bool preserve_all_tensors);
  static InterpreterWrapper* CreateWrapperCPPFromFile(
      const char* model_path, int op_resolver_id,
      const std::vector<std::string>& registerers_by_name,
      const std::vector<std::function<void(uintptr_t)>>& registerers_by_func,
      std::string* error_msg, bool preserve_all_tensors);

  // SWIG caller takes ownership of pointer.
  static InterpreterWrapper* CreateWrapperCPPFromBuffer(
      PyObject* data, int op_resolver_id,
      const std::vector<std::string>& registerers, std::string* error_msg,
      bool preserve_all_tensors);
  static InterpreterWrapper* CreateWrapperCPPFromBuffer(
      PyObject* data, int op_resolver_id,
      const std::vector<std::string>& registerers_by_name,
      const std::vector<std::function<void(uintptr_t)>>& registerers_by_func,
      std::string* error_msg, bool preserve_all_tensors);

  ~InterpreterWrapper();
  PyObject* AllocateTensors(int subgraph_index);
  PyObject* Invoke(int subgraph_index);

  PyObject* InputIndices() const;
  PyObject* OutputIndices() const;
  PyObject* ResizeInputTensor(int i, PyObject* value, bool strict,
                              int subgraph_index);

  int NumTensors() const;
  std::string TensorName(int i) const;
  PyObject* TensorType(int i) const;
  PyObject* TensorSize(int i) const;
  PyObject* TensorSizeSignature(int i) const;
  PyObject* TensorSparsityParameters(int i) const;
  // Deprecated in favor of TensorQuantizationScales, below.
  PyObject* TensorQuantization(int i) const;
  PyObject* TensorQuantizationParameters(int i) const;
  PyObject* SetTensor(int i, PyObject* value, int subgraph_index);
  PyObject* GetTensor(int i, int subgraph_index) const;
  PyObject* GetSubgraphIndexFromSignature(const char* signature_key);
  PyObject* GetSignatureDefs() const;
  PyObject* ResetVariableTensors();

  int NumNodes() const;
  std::string NodeName(int i) const;
  PyObject* NodeInputs(int i) const;
  PyObject* NodeOutputs(int i) const;

  // Returns a reference to tensor index as a numpy array from subgraph. The
  // base_object should be the interpreter object providing the memory.
  PyObject* tensor(PyObject* base_object, int tensor_index, int subgraph_index);

  PyObject* SetNumThreads(int num_threads);

  // Adds a delegate to the interpreter.
  PyObject* ModifyGraphWithDelegate(TfLiteDelegate* delegate);

  // Experimental and subject to change.
  //
  // Returns a pointer to the underlying interpreter.
  Interpreter* interpreter() { return interpreter_.get(); }

 private:
  // Helper function to construct an `InterpreterWrapper` object.
  // It only returns InterpreterWrapper if it can construct an `Interpreter`.
  // Otherwise it returns `nullptr`.
  static InterpreterWrapper* CreateInterpreterWrapper(
      std::unique_ptr<Model> model, int op_resolver_id,
      std::unique_ptr<PythonErrorReporter> error_reporter,
      const std::vector<std::string>& registerers_by_name,
      const std::vector<std::function<void(uintptr_t)>>& registerers_by_func,
      std::string* error_msg, bool preserve_all_tensors);

  InterpreterWrapper(std::unique_ptr<Model> model,
                     std::unique_ptr<PythonErrorReporter> error_reporter,
                     std::unique_ptr<tflite::MutableOpResolver> resolver,
                     std::unique_ptr<Interpreter> interpreter);

  // InterpreterWrapper is not copyable or assignable. We avoid the use of
  // InterpreterWrapper() = delete here for SWIG compatibility.
  InterpreterWrapper();
  InterpreterWrapper(const InterpreterWrapper& rhs);

  // Helper function to resize an input tensor.
  PyObject* ResizeInputTensorImpl(int i, PyObject* value);

  // The public functions which creates `InterpreterWrapper` should ensure all
  // these member variables are initialized successfully. Otherwise it should
  // report the error and return `nullptr`.
  const std::unique_ptr<Model> model_;
  const std::unique_ptr<PythonErrorReporter> error_reporter_;
  const std::unique_ptr<tflite::MutableOpResolver> resolver_;
  const std::unique_ptr<Interpreter> interpreter_;
};

}  // namespace interpreter_wrapper
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PYTHON_INTERPRETER_WRAPPER_INTERPRETER_WRAPPER_H_
