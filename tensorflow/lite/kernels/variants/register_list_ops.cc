/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/variants/register_list_ops.h"

#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {
namespace variants {
namespace ops {

void RegisterListOps(MutableOpResolver* resolver) {
  resolver->AddCustom("TensorListReserve", Register_LIST_RESERVE());
  resolver->AddCustom("TensorListStack", Register_LIST_STACK());
  resolver->AddCustom("TensorListSetItem", Register_LIST_SET_ITEM());
  resolver->AddCustom("TensorListFromTensor", Register_LIST_FROM_TENSOR());
  resolver->AddCustom("TensorListGetItem", Register_LIST_GET_ITEM());
  resolver->AddCustom("TensorListLength", Register_LIST_LENGTH());
  resolver->AddCustom("TensorListElementShape", Register_LIST_ELEMENT_SHAPE());
  resolver->AddCustom("TensorListPopBack", Register_LIST_POP_BACK());
  resolver->AddCustom("TensorListPushBack", Register_LIST_PUSH_BACK());
  resolver->AddCustom("VariantAddN", Register_VARIANT_ADD_N());
  resolver->AddCustom("VariantZerosLike", Register_VARIANT_ZEROS_LIKE());
}

}  // namespace ops
}  // namespace variants
}  // namespace tflite

void RegisterSelectedOps(tflite::MutableOpResolver* resolver) {
  tflite::variants::ops::RegisterListOps(resolver);
}
