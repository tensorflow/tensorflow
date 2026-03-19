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
#ifndef TENSORFLOW_CORE_TFRT_SAVED_MODEL_PYTHON_SAVED_MODEL_LOAD_AND_RUN_H_
#define TENSORFLOW_CORE_TFRT_SAVED_MODEL_PYTHON_SAVED_MODEL_LOAD_AND_RUN_H_

#include <Python.h>

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"

namespace tensorflow::tfrt_stub {

absl::StatusOr<std::unique_ptr<SavedModel>> LoadSavedModel(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags);

std::vector<tensorflow::Tensor> RunConvertor(PyObject* args);

absl::Status Run(
    SavedModel* saved_model,
    const tensorflow::tfrt_stub::GraphExecutionRunOptions& run_options,
    absl::string_view name, const std::vector<tensorflow::Tensor>& inputs,
    std::vector<tensorflow::Tensor>* outputs);
}  // namespace tensorflow::tfrt_stub

#endif  // TENSORFLOW_CORE_TFRT_SAVED_MODEL_PYTHON_SAVED_MODEL_LOAD_AND_RUN_H_
