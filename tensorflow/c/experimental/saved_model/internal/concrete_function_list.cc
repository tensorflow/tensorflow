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

#include <stddef.h>

#include "tensorflow/c/experimental/saved_model/core/concrete_function.h"
#include "tensorflow/c/experimental/saved_model/internal/concrete_function_list_type.h"
#include "tensorflow/c/experimental/saved_model/internal/concrete_function_type.h"

extern "C" {

size_t TF_ConcreteFunctionListNumOutputs(TF_ConcreteFunctionList* list) {
  return list->list.size();
}

TF_ConcreteFunction* TF_ConcreteFunctionListGet(TF_ConcreteFunctionList* list,
                                                int i) {
  return tensorflow::wrap(list->list[i]);
}

void TF_DeleteConcreteFunctionList(TF_ConcreteFunctionList* list) {
  delete list;
}

}  // end extern "C"
