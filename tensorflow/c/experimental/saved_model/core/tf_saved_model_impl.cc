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

#include "tensorflow/c/experimental/saved_model/core/tf_saved_model_impl.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/c/experimental/saved_model/core/concrete_function.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

Status TFSavedModelAPIImpl::GetFunction(const std::string& function_path,
                                        ConcreteFunction** function) {
  // TODO(bmzhao): Add support for retrieving a function.
  return errors::Unimplemented(
      "Retrieving functions is unimplemented currently");
}

Status TFSavedModelAPIImpl::GetSignatureDefFunction(
    const std::string& signature_def_key, ConcreteFunction** function) {
  // TODO(bmzhao): Add support for retrieving a signaturedef function.
  return errors::Unimplemented(
      "Retrieving functions is unimplemented currently");
}

std::vector<ConcreteFunction*> TFSavedModelAPIImpl::ListFunctions() {
  std::vector<ConcreteFunction*> result;
  result.reserve(functions_.size());
  for (ConcreteFunction& function : functions_) {
    result.push_back(&function);
  }
  return result;
}

Status TFSavedModelAPIImpl::Load(
    const std::string& directory,
    const absl::optional<std::unordered_set<std::string>>& tags,
    TFSavedModelAPIImpl* out) {
  // TODO(bmzhao): Add support for loading a TFSavedModelImpl.
  return errors::Unimplemented(
      "TFSavedModelAPIImpl loading is unimplemented currently");
}

}  // namespace tensorflow
