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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_SAVED_MODEL_API_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_SAVED_MODEL_API_H_

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/experimental/saved_model/public/concrete_function.h"
#include "tensorflow/c/experimental/saved_model/public/concrete_function_list.h"
#include "tensorflow/c/experimental/saved_model/public/signature_def_function.h"
#include "tensorflow/c/tf_status.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An opaque type representing a Tensorflow "SavedModel"
// (https://www.tensorflow.org/guide/saved_model) that we always pass by pointer
// to achieve ABI stability.
typedef struct TF_SavedModel TF_SavedModel;

// Load a SavedModel from `dirname`. We expect the SavedModel to contain a
// single Metagraph (as for those exported from TF2's `tf.saved_model.save`).
//
// Params:
//  dirname - A directory filepath that the SavedModel is at.
//  ctx - A TFE_Context containing optional load/TF runtime options.
//        `ctx` must outlive the returned TF_SavedModel pointer.
//  status - Set to OK on success and an appropriate error on failure.
// Returns:
//  If status is not OK, returns nullptr. Otherwise, returns a newly created
//  TF_SavedModel instance. It must be deleted by calling TF_DeleteSavedModel.
TF_CAPI_EXPORT extern TF_SavedModel* TF_LoadSavedModel(const char* dirname,
                                                       TFE_Context* ctx,
                                                       TF_Status* status);

// Load a SavedModel from `dirname`.
//
// Params:
//  dirname - A directory filepath that the SavedModel is at.
//  ctx - A TFE_Context containing optional load/TF runtime options.
//        `ctx` must outlive the returned TF_SavedModel pointer.
//  tags - char* array of SavedModel tags. We will load the metagraph matching
//         the tags.
//  tags_len - number of elements in the `tags` array.
//  status - Set to OK on success and an appropriate error on failure.
// Returns:
//  If status is not OK, returns nullptr. Otherwise, returns a newly created
//  TF_SavedModel instance. It must be deleted by calling TF_DeleteSavedModel.
TF_CAPI_EXPORT extern TF_SavedModel* TF_LoadSavedModelWithTags(
    const char* dirname, TFE_Context* ctx, const char* const* tags,
    int tags_len, TF_Status* status);

// Deletes a TF_SavedModel, and frees any resources owned by it.
TF_CAPI_EXPORT extern void TF_DeleteSavedModel(TF_SavedModel* model);

// Retrieve a function from the TF2 SavedModel via function path.
//
// Params:
//  model - The TF2 SavedModel to load a function from.
//  function_path - A string containing the path from the root saved python
//                  object to a tf.function method.
//                  TODO(bmzhao): Add a detailed example of this with a
//                  python tf.module before moving this out of experimental.
//  status - Set to OK on success and an appropriate error on failure.
// Returns:
//  If status is not OK, returns nullptr. Otherwise, returns a
//  TF_ConcreteFunction instance. The lifetime of this instance is
//  "conceptually" bound to `model`. Once `model` is deleted, all
//  `TF_ConcreteFunctions` retrieved from it are invalid, and have been deleted.
TF_CAPI_EXPORT extern TF_ConcreteFunction* TF_GetSavedModelConcreteFunction(
    TF_SavedModel* model, const char* function_path, TF_Status* status);

// Retrieve a function from the TF SavedModel via a SignatureDef key.
//
// Params:
//  model - The SavedModel to load a function from.
//  signature_def_key - The string key of the SignatureDef map of a SavedModel:
//                      https://github.com/tensorflow/tensorflow/blob/69b08900b1e991d84bce31f3b404f5ed768f339f/tensorflow/core/protobuf/meta_graph.proto#L89
//  status - Set to OK on success and an appropriate error on failure.
// Returns:
//  If status is not OK, returns nullptr. Otherwise, returns a
//  TF_SignatureDefFunction instance. Once `model` is deleted, all
//  `TF_SignatureDefFunctions` retrieved from it are invalid, and have been
//  deleted.
TF_CAPI_EXPORT extern TF_SignatureDefFunction*
TF_GetSavedModelSignatureDefFunction(TF_SavedModel* model,
                                     const char* signature_def_key,
                                     TF_Status* status);

#ifdef __cplusplus
}  // end extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_SAVED_MODEL_API_H_
