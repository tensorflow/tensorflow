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
#ifndef TENSORFLOW_LITE_KERNELS_VARIANTS_LIST_OPS_LIB_H_
#define TENSORFLOW_LITE_KERNELS_VARIANTS_LIST_OPS_LIB_H_

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace variants {
namespace detail {

struct ListReserveOptions {
  TensorType element_type;
};

}  // namespace detail
namespace ops {

TfLiteRegistration* Register_LIST_RESERVE();

TfLiteRegistration* Register_LIST_STACK();

TfLiteRegistration* Register_LIST_SET_ITEM();

TfLiteRegistration* Register_LIST_FROM_TENSOR();

TfLiteRegistration* Register_LIST_GET_ITEM();

TfLiteRegistration* Register_LIST_LENGTH();

TfLiteRegistration* Register_LIST_ELEMENT_SHAPE();

TfLiteRegistration* Register_LIST_POP_BACK();

TfLiteRegistration* Register_LIST_PUSH_BACK();

TfLiteRegistration* Register_VARIANT_ADD_N();

TfLiteRegistration* Register_VARIANT_ZEROS_LIKE();

}  // namespace ops
}  // namespace variants
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_VARIANTS_LIST_OPS_LIB_H_
