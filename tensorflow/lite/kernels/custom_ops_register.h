/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_CUSTOM_OPS_REGISTER_H_
#define TENSORFLOW_LITE_KERNELS_CUSTOM_OPS_REGISTER_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_ATAN2();
TfLiteRegistration* Register_HASHTABLE();
TfLiteRegistration* Register_HASHTABLE_FIND();
TfLiteRegistration* Register_HASHTABLE_IMPORT();
TfLiteRegistration* Register_HASHTABLE_SIZE();
TfLiteRegistration* Register_IRFFT2D();
TfLiteRegistration* Register_MULTINOMIAL();
TfLiteRegistration* Register_RANDOM_STANDARD_NORMAL();
TfLiteRegistration* Register_RANDOM_UNIFORM();
TfLiteRegistration* Register_RANDOM_UNIFORM_INT();
TfLiteRegistration* Register_SIGN();

}  // namespace custom
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CUSTOM_OPS_REGISTER_H_
