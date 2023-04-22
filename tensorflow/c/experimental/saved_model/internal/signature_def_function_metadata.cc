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

#include "tensorflow/c/experimental/saved_model/public/signature_def_function_metadata.h"

#include "tensorflow/c/experimental/saved_model/internal/signature_def_function_metadata_type.h"
#include "tensorflow/c/experimental/saved_model/internal/signature_def_param_list_type.h"

extern "C" {

extern const TF_SignatureDefParamList* TF_SignatureDefFunctionMetadataArgs(
    const TF_SignatureDefFunctionMetadata* list) {
  return tensorflow::wrap(&tensorflow::unwrap(list)->arguments());
}

extern const TF_SignatureDefParamList* TF_SignatureDefFunctionMetadataReturns(
    const TF_SignatureDefFunctionMetadata* list) {
  return tensorflow::wrap(&tensorflow::unwrap(list)->returns());
}

}  // end extern "C"
