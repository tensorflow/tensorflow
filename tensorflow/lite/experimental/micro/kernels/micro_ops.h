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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_KERNELS_MICRO_OPS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_KERNELS_MICRO_OPS_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace ops {
namespace micro {

// Forward declaration of all micro op kernel registration methods. These
// registrations are included with the standard `BuiltinOpResolver`.
//
// This header is particularly useful in cases where only a subset of ops are
// needed. In such cases, the client can selectively add only the registrations
// their model requires, using a custom `(Micro)MutableOpResolver`. Selective
// registration in turn allows the linker to strip unused kernels.

TfLiteRegistration* Register_ABS();
TfLiteRegistration* Register_ADD();
TfLiteRegistration* Register_ARG_MAX();
TfLiteRegistration* Register_ARG_MIN();
TfLiteRegistration* Register_AVERAGE_POOL_2D();
TfLiteRegistration* Register_CEIL();
TfLiteRegistration* Register_CONCATENATION();
TfLiteRegistration* Register_CONV_2D();
TfLiteRegistration* Register_COS();
TfLiteRegistration* Register_DEPTHWISE_CONV_2D();
TfLiteRegistration* Register_DEQUANTIZE();
TfLiteRegistration* Register_EQUAL();
TfLiteRegistration* Register_FLOOR();
TfLiteRegistration* Register_FULLY_CONNECTED();
TfLiteRegistration* Register_GREATER();
TfLiteRegistration* Register_GREATER_EQUAL();
TfLiteRegistration* Register_LESS();
TfLiteRegistration* Register_LESS_EQUAL();
TfLiteRegistration* Register_LOG();
TfLiteRegistration* Register_LOGICAL_AND();
TfLiteRegistration* Register_LOGICAL_NOT();
TfLiteRegistration* Register_LOGICAL_OR();
TfLiteRegistration* Register_LOGISTIC();
TfLiteRegistration* Register_MAXIMUM();
TfLiteRegistration* Register_MAX_POOL_2D();
TfLiteRegistration* Register_MINIMUM();
TfLiteRegistration* Register_MUL();
TfLiteRegistration* Register_NEG();
TfLiteRegistration* Register_NOT_EQUAL();
TfLiteRegistration* Register_PACK();
TfLiteRegistration* Register_PRELU();
TfLiteRegistration* Register_QUANTIZE();
TfLiteRegistration* Register_RELU();
TfLiteRegistration* Register_RELU6();
TfLiteRegistration* Register_RESHAPE();
TfLiteRegistration* Register_ROUND();
TfLiteRegistration* Register_RSQRT();
TfLiteRegistration* Register_SIN();
TfLiteRegistration* Register_SOFTMAX();
TfLiteRegistration* Register_SPLIT();
TfLiteRegistration* Register_SQRT();
TfLiteRegistration* Register_SQUARE();
TfLiteRegistration* Register_STRIDED_SLICE();
TfLiteRegistration* Register_SVDF();
TfLiteRegistration* Register_UNPACK();

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_KERNELS_MICRO_OPS_H_
