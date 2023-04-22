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

#ifndef TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SAVED_MODEL_API_H_
#define TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SAVED_MODEL_API_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/c/experimental/saved_model/public/saved_model_api.h"
#include "tensorflow/cc/experimental/base/public/runtime.h"
#include "tensorflow/cc/experimental/base/public/status.h"
#include "tensorflow/cc/saved_model/experimental/public/concrete_function.h"
#include "tensorflow/cc/saved_model/experimental/public/concrete_function_list.h"
#include "tensorflow/cc/saved_model/experimental/public/signature_def_function.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// SavedModelAPI offers a way to load Tensorflow Saved Models
// (https://www.tensorflow.org/guide/saved_model) and execute saved
// tf.functions or legacy SignatureDefs in a TF2-idiomatic fashion.
// See RFC 207
// (https://github.com/tensorflow/community/blob/master/rfcs/20200218-tf-c-saved-model.md)
// TODO(bmzhao): Add an e2e example here, once ConcreteFunction::Run is added.
class SavedModelAPI {
 public:
  // Load a SavedModel from `dirname`.
  //
  // Params:
  //  saved_model_path - A directory filepath that the SavedModel is at.
  //  runtime - A runtime used to load SavedModelAPI. `runtime` must outlive the
  //            returned TF_SavedModel pointer.
  //  tags - Optional set of tags. If tags = nullptr, we expect the SavedModel
  //         to contain a single Metagraph (as for those exported from TF2's
  //         `tf.saved_model.save`). If tags != nullptr, we load the metagraph
  //         matching the tags:
  //         https://github.com/tensorflow/tensorflow/blob/428cdeda09aef81e958eeb274b83d27ad635b57b/tensorflow/core/protobuf/meta_graph.proto#L50-L56
  //  status - Set to OK on success and an appropriate error on failure.
  // Returns:
  //  If status is not OK, returns nullptr.
  static std::unique_ptr<SavedModelAPI> Load(
      const std::string& saved_model_path, const Runtime& runtime,
      Status* status, const std::unordered_set<std::string>* tags = nullptr);

  // Retrieve a function from the TF2 SavedModel via function path.
  //
  // Params:
  //  function_path - A string containing the path from the root saved python
  //                  object to a tf.function method.
  //  status - Set to OK on success and an appropriate error on failure.
  // Returns:
  //  If status is not OK, returns nullptr. Otherwise, returns a
  //  tensorflow::cc::ConcreteFunction pointer. The lifetime of this pointer
  //  is bound to SavedModelAPI it was loaded from.
  ConcreteFunction* GetConcreteFunction(const std::string& function_path,
                                        Status* status);

  // Retrieve a function from the TF SavedModel via a SignatureDef key.
  //
  // Params:
  //  signature_def_key - String key of SignatureDef map of a SavedModel:
  //                      https://github.com/tensorflow/tensorflow/blob/69b08900b1e991d84bce31f3b404f5ed768f339f/tensorflow/core/protobuf/meta_graph.proto#L89
  //  status - Set to OK on success and an appropriate error on failure.
  // Returns:
  //  If status is not OK, returns nullptr. Otherwise, returns a
  //  tensorflow::cc::ConcreteFunction pointer. The lifetime of this pointer
  //  is bound to SavedModelAPI it was loaded from.
  SignatureDefFunction* GetSignatureDefFunction(
      const std::string& function_path, Status* status);

  // SavedModelAPI is movable, but not copyable.
  SavedModelAPI(SavedModelAPI&&) = default;
  SavedModelAPI& operator=(SavedModelAPI&&) = default;

 private:
  SavedModelAPI(const SavedModelAPI&) = delete;
  SavedModelAPI& operator=(const SavedModelAPI&) = delete;

  explicit SavedModelAPI(TF_SavedModel* model) : saved_model_(model) {}
  struct TFSavedModelDeleter {
    void operator()(TF_SavedModel* p) const { TF_DeleteSavedModel(p); }
  };
  std::unique_ptr<TF_SavedModel, TFSavedModelDeleter> saved_model_;
};

inline std::unique_ptr<SavedModelAPI> SavedModelAPI::Load(
    const std::string& saved_model_path, const Runtime& runtime, Status* status,
    const std::unordered_set<std::string>* tags) {
  TF_SavedModel* saved_model = nullptr;

  if (tags == nullptr) {
    saved_model =
        TF_LoadSavedModel(saved_model_path.c_str(), runtime.GetTFEContext(),
                          status->GetTFStatus());
  } else {
    std::vector<const char*> tags_vector;
    tags_vector.reserve(tags->size());
    for (const std::string& tag : *tags) {
      tags_vector.push_back(tag.c_str());
    }
    saved_model = TF_LoadSavedModelWithTags(
        saved_model_path.c_str(), runtime.GetTFEContext(), tags_vector.data(),
        tags_vector.size(), status->GetTFStatus());
  }

  if (!status->ok()) {
    return nullptr;
  }

  // We can't use std::make_unique here because of its interaction with a
  // private constructor: https://abseil.io/tips/134
  return std::unique_ptr<SavedModelAPI>(new SavedModelAPI(saved_model));
}

inline ConcreteFunction* SavedModelAPI::GetConcreteFunction(
    const std::string& function_path, Status* status) {
  TF_ConcreteFunction* function = TF_GetSavedModelConcreteFunction(
      saved_model_.get(), function_path.c_str(), status->GetTFStatus());
  if (!status->ok()) {
    return nullptr;
  }
  return ConcreteFunction::wrap(function);
}

inline SignatureDefFunction* SavedModelAPI::GetSignatureDefFunction(
    const std::string& function_path, Status* status) {
  TF_SignatureDefFunction* function = TF_GetSavedModelSignatureDefFunction(
      saved_model_.get(), function_path.c_str(), status->GetTFStatus());
  if (!status->ok()) {
    return nullptr;
  }
  return SignatureDefFunction::wrap(function);
}

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SAVED_MODEL_API_H_
