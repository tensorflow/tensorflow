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

#include "tensorflow/c/experimental/saved_model/public/signature_def_param_list.h"

#include "tensorflow/c/experimental/saved_model/internal/signature_def_param_list_type.h"
#include "tensorflow/c/experimental/saved_model/internal/signature_def_param_type.h"

extern "C" {

extern size_t TF_SignatureDefParamListSize(
    const TF_SignatureDefParamList* list) {
  return tensorflow::unwrap(list)->size();
}

extern const TF_SignatureDefParam* TF_SignatureDefParamListGet(
    const TF_SignatureDefParamList* list, int i) {
  return tensorflow::wrap(&tensorflow::unwrap(list)->at(i));
}

}  // end extern "C"
