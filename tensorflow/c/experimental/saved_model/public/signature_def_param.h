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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_SIGNATURE_DEF_PARAM_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_SIGNATURE_DEF_PARAM_H_

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/experimental/saved_model/public/tensor_spec.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An opaque type that containing metadata of an input/output of a
// TF_SignatureDefFunction loaded from a SavedModel.
typedef struct TF_SignatureDefParam TF_SignatureDefParam;

// Returns the name of the given parameter. The caller is not responsible for
// freeing the returned char*.
TF_CAPI_EXPORT extern const char* TF_SignatureDefParamName(
    const TF_SignatureDefParam* param);

// Returns the TensorSpec associated with the given parameter. The caller is
// not reponsible for freeing the returned TF_TensorSpec*.
TF_CAPI_EXPORT extern const TF_TensorSpec* TF_SignatureDefParamTensorSpec(
    const TF_SignatureDefParam* param);

#ifdef __cplusplus
}  // end extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_PUBLIC_SIGNATURE_DEF_PARAM_H_
