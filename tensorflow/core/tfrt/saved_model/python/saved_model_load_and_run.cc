/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/saved_model/python/saved_model_load_and_run.h"

#include <Python.h>

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/refcount.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow::tfrt_stub {
using RefCountHandle = tsl::core::RefCountPtr<tensorflow::TensorHandle>;

absl::StatusOr<std::unique_ptr<SavedModel>> LoadSavedModel(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags) {
  auto runtime = tensorflow::tfrt_stub::Runtime::Create(
      tensorflow::tfrt_stub::WrapDefaultWorkQueue(
          tfrt::CreateMultiThreadedWorkQueue(1, 1)));
  SavedModel::Options options(runtime.get());
  options.graph_execution_options.enable_tfrt_gpu = true;
  options.graph_execution_options.enable_grappler_function_optimizer = true;
  options.graph_execution_options.compile_options.enable_grappler = true;
  options.graph_execution_options.compile_options.device_target =
      TfrtDeviceInfraTarget::kGpu;
  options.graph_execution_options.compile_options.hoist_invariant_ops = true;
  return SavedModelImpl::LoadSavedModel(options, saved_model_dir, tags);
}

// Helper function for making vector of pyobjects
std::vector<PyObject*> MakeTensorList(PyObject* tensors) {
  PyObject* seq = PySequence_Fast(tensors, "expected a sequence");
  if (seq == nullptr) {
    return {};
  }
  const int len = PySequence_Fast_GET_SIZE(seq);
  PyObject** seq_array = PySequence_Fast_ITEMS(seq);
  std::vector<PyObject*> list(seq_array, seq_array + len);
  Py_DECREF(seq);
  return list;
}

// Helper function for getting string literals from Pyobjects
std::string PyObject_ToString(PyObject* o, int length = -1) {
  auto str_o = make_safe(PyObject_Str(o));
  std::string str = PyUnicode_AsUTF8(str_o.get());
  if (length < 0 || str.size() <= length) {
    return str;
  }
  absl::string_view str_piece(str);
  return tensorflow::strings::StrCat(str_piece.substr(length), "...");
}

// Assume inputs are name, inputs, outputs
std::vector<tensorflow::Tensor> RunConvertor(PyObject* args) {
  // Create Py Objects to be converted into a TFE_Tensor Handle
  tensorflow::Safe_PyObjectPtr py_eager_tensor = nullptr;
  PyObject* lst = PyTuple_GetItem(args, 0);
  std::vector<PyObject*> input = MakeTensorList(lst);
  for (PyObject* tensor : input) {
    // Creating a C++ Tensor object on a python buffer will eat a reference to
    // the buffer, so we need to increase their reference count.
    Py_INCREF(tensor);
  }
  std::vector<tensorflow::Tensor> input_run;
  for (int i = 0; i < input.size(); ++i) {
    py_eager_tensor.reset(input[i]);
    // Create the TFE_Tensorhandle and convert into a immediateExecutionHandle
    TFE_TensorHandle* input_handle = EagerTensor_Handle(py_eager_tensor.get());
    //  std::vector<TFE_TensorHandle> output_handles;
    // output_handles.emplace_back(EagerTensor_Handle(py_eager_tensor.get()));
    ImmediateExecutionTensorHandle* handle = tensorflow::unwrap(input_handle);
    if (tensorflow::TensorHandle::classof(handle)) {
      TensorHandle* push = down_cast<TensorHandle*>(handle);
      const tensorflow::Tensor* tensor;
      push->Tensor(&tensor).IgnoreError();
      input_run.push_back(*tensor);
    }
  }
  return input_run;
}

absl::Status Run(
    SavedModel* saved_model,
    const tensorflow::tfrt_stub::GraphExecutionRunOptions& run_options,
    absl::string_view name, const std::vector<tensorflow::Tensor>& inputs,
    std::vector<tensorflow::Tensor>* outputs) {
  return saved_model->Run(run_options, name, inputs, outputs);
}
}  // namespace tensorflow::tfrt_stub
