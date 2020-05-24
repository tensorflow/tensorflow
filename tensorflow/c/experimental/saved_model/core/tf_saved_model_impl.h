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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_TF_SAVED_MODEL_IMPL_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_TF_SAVED_MODEL_IMPL_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/c/experimental/saved_model/core/concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_api.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

class TFSavedModelAPIImpl : public SavedModelAPI {
 public:
  TFSavedModelAPIImpl() = default;

  Status GetFunction(const std::string& function_path,
                     ConcreteFunction** function) override;

  Status GetSignatureDefFunction(const std::string& signature_def_key,
                                 ConcreteFunction** function) override;

  static Status Load(
      const std::string& directory,
      const absl::optional<std::unordered_set<std::string>>& tags,
      TFSavedModelAPIImpl* out);

  std::vector<ConcreteFunction*> ListFunctions() override;

  ~TFSavedModelAPIImpl() override = default;

 private:
  std::vector<ConcreteFunction> functions_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_TF_SAVED_MODEL_IMPL_H_
