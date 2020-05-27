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

#include "tensorflow/c/experimental/saved_model/public/saved_model_api.h"

#include <memory>
#include <string>
#include <unordered_set>

#include "absl/types/optional.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_api.h"
#include "tensorflow/c/experimental/saved_model/internal/concrete_function_list_type.h"
#include "tensorflow/c/experimental/saved_model/internal/concrete_function_type.h"
#include "tensorflow/c/experimental/saved_model/internal/saved_model_api_type.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/platform/status.h"

extern "C" {

TF_SavedModel* TF_LoadSavedModel(const char* dirname, TFE_Context* ctx,
                                 TF_Status* status) {
  std::string saved_model_dir(dirname);

  std::unique_ptr<tensorflow::SavedModelAPI> result =
      tensorflow::unwrap(ctx)->LoadSavedModelAPI(dirname, absl::nullopt,
                                                 &status->status);
  if (!status->status.ok()) {
    return nullptr;
  }
  return tensorflow::wrap(result.release());
}

TF_SavedModel* TF_LoadSavedModelWithTags(const char* dirname, TFE_Context* ctx,
                                         const char* const* tags, int tags_len,
                                         TF_Status* status) {
  std::string saved_model_dir(dirname);

  std::unordered_set<std::string> tagset;
  for (int i = 0; i < tags_len; ++i) {
    tagset.insert(std::string(tags[i]));
  }

  std::unique_ptr<tensorflow::SavedModelAPI> result =
      tensorflow::unwrap(ctx)->LoadSavedModelAPI(dirname, std::move(tagset),
                                                 &status->status);
  if (!status->status.ok()) {
    return nullptr;
  }
  return tensorflow::wrap(result.release());
}

void TF_DeleteSavedModel(TF_SavedModel* model) {
  delete tensorflow::unwrap(model);
}

TF_ConcreteFunction* TF_GetSavedModelConcreteFunction(TF_SavedModel* model,
                                                      const char* function_path,
                                                      TF_Status* status) {
  tensorflow::ConcreteFunction* result = nullptr;
  tensorflow::Status get_function_status =
      tensorflow::unwrap(model)->GetFunction(function_path, &result);
  status->status.Update(get_function_status);
  if (!get_function_status.ok()) {
    return nullptr;
  }
  return tensorflow::wrap(result);
}

TF_CAPI_EXPORT extern TF_ConcreteFunction* TF_GetSavedModelSignatureDefFunction(
    TF_SavedModel* model, const char* signature_def_key, TF_Status* status) {
  tensorflow::ConcreteFunction* result = nullptr;
  tensorflow::Status get_function_status =
      tensorflow::unwrap(model)->GetSignatureDefFunction(signature_def_key,
                                                         &result);
  status->status.Update(get_function_status);
  if (!get_function_status.ok()) {
    return nullptr;
  }
  return tensorflow::wrap(result);
}

TF_ConcreteFunctionList* TF_ListSavedModelFunctions(TF_SavedModel* model) {
  return new TF_ConcreteFunctionList{
      tensorflow::unwrap(model)->ListFunctions()};
}

}  // end extern "C"
