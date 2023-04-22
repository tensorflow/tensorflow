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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/experimental/saved_model/core/concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/revived_objects.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/variable.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_api.h"
#include "tensorflow/c/experimental/saved_model/core/signature_def_function.h"
#include "tensorflow/cc/saved_model/bundle_v2.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// An implementation of the SavedModelAPI using the TF Eager runtime. See
// https://github.com/tensorflow/community/blob/master/rfcs/20200218-tf-c-saved-model.md
// Conceptually, there are many differences between a tf.function and
// a FunctionDef is executed by the C API.
// 1. A tf.function is polymorphic, meaning it can correspond to multiple
// ConcreteFunctions (of differing shapes, python arguments, etc). A
// FunctionDef corresponds to a single ConcreteFunction.
// 2. A tf.function can take arbitrary python inputs, whereas the FunctionDef
// only accepts tensors.
// 3. A tf.function is a closure that can contain captured inputs, whereas
// FunctionDefs loaded from SavedModels are "functional" (all inputs are
// explicitly passed as arguments).
// The SavedModelAPI only supports loading tf.functions annotated with input
// signatures so that we ensure that there is a 1:1 mapping between tf.function
// -> FunctionDef, and have a guarantee that all inputs are tensors.
// (https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/eager/def_function.py#L1167-L1171),
class TFSavedModelAPI : public SavedModelAPI {
 public:
  Status GetFunction(const std::string& function_path,
                     ConcreteFunction** function) override;

  Status GetFunctions(
      int node_id,
      absl::flat_hash_map<std::string, ConcreteFunction*>* functions) override;

  Status GetSignatureDefFunction(const std::string& signature_def_key,
                                 SignatureDefFunction** function) override;

  static Status Load(
      const std::string& directory,
      const absl::optional<std::unordered_set<std::string>>& tags,
      ImmediateExecutionContext* context,
      std::unique_ptr<TFSavedModelAPI>* out);

  ~TFSavedModelAPI() override = default;

  Status GetVariable(const std::string& variable_path, Variable** variable);

  SavedModelV2Bundle* GetBundle() override;

 private:
  TFSavedModelAPI(const std::string& directory, SavedModelV2Bundle bundle,
                  RevivedObjects revived_objects);

  std::string directory_;
  SavedModelV2Bundle bundle_;
  RevivedObjects revived_objects_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_TF_SAVED_MODEL_IMPL_H_
