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

#include "tensorflow/c/experimental/saved_model/public/tensorhandle_list.h"

#include <stddef.h>

#include "tensorflow/c/eager/tensor_handle_interface.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/experimental/saved_model/internal/tensorhandle_list_type.h"

extern "C" {

size_t TF_TensorHandleListSize(const TF_TensorHandleList* list) {
  return tensorflow::unwrap(list)->size();
}

TFE_TensorHandle* TF_TensorHandleListGet(const TF_TensorHandleList* list,
                                         int i) {
  return tensorflow::wrap((*tensorflow::unwrap(list))[i]);
}


}  // end extern "C"
