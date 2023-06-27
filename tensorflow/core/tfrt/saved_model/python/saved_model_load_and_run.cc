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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"

namespace tensorflow::tfrt_stub {

tensorflow::StatusOr<std::unique_ptr<SavedModel>> LoadSavedModel(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags) {
  return SavedModelImpl::LoadSavedModel(
      tensorflow::tfrt_stub::SavedModel::Options(
          tensorflow::tfrt_stub::GetGlobalRuntime()),
      saved_model_dir, tags);
}

tensorflow::Status Run(
    SavedModel& saved_model,
    const tensorflow::tfrt_stub::GraphExecutionRunOptions& run_options,
    absl::string_view name, absl::Span<const tensorflow::Tensor> inputs,
    std::vector<tensorflow::Tensor>* outputs) {
  return saved_model.Run(run_options, name, inputs, outputs);
}

}  // namespace tensorflow::tfrt_stub
