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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_MICRO_OPS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_MICRO_OPS_H_

#include "tensorflow/lite/c/common.h"

// Forward declaration of all micro op kernel registration methods. These
// registrations are included with the standard `BuiltinOpResolver`.
//
// This header is particularly useful in cases where only a subset of ops are
// needed. In such cases, the client can selectively add only the registrations
// their model requires, using a custom `(Micro)MutableOpResolver`. Selective
// registration in turn allows the linker to strip unused kernels.

namespace tflite {

// TFLM is incrementally moving towards a flat tflite namespace
// (https://abseil.io/tips/130). Any new ops (or cleanup of existing ops should
// have their Register function declarations in the tflite namespace.

TfLiteRegistration Register_ADD_N();
TfLiteRegistration Register_BATCH_TO_SPACE_ND();
TfLiteRegistration Register_CAST();
TfLiteRegistration Register_CONV_2D();
TfLiteRegistration Register_CUMSUM();
TfLiteRegistration Register_DEPTH_TO_SPACE();
TfLiteRegistration Register_DEPTHWISE_CONV_2D();
TfLiteRegistration Register_DIV();
TfLiteRegistration Register_ELU();
TfLiteRegistration Register_EXP();
TfLiteRegistration Register_EXPAND_DIMS();
TfLiteRegistration Register_FILL();
TfLiteRegistration Register_FLOOR_DIV();
TfLiteRegistration Register_FLOOR_MOD();
TfLiteRegistration Register_GATHER();
TfLiteRegistration Register_GATHER_ND();
TfLiteRegistration Register_IF();
TfLiteRegistration Register_L2_POOL_2D();
TfLiteRegistration Register_LEAKY_RELU();
TfLiteRegistration Register_LOG_SOFTMAX();
TfLiteRegistration Register_QUANTIZE();
TfLiteRegistration Register_RESIZE_BILINEAR();
TfLiteRegistration Register_SHAPE();
TfLiteRegistration Register_SPACE_TO_BATCH_ND();
TfLiteRegistration Register_SQUEEZE();
TfLiteRegistration Register_SVDF();
TfLiteRegistration Register_TRANSPOSE();
TfLiteRegistration Register_TRANSPOSE_CONV();
TfLiteRegistration Register_ZEROS_LIKE();

namespace ops {
namespace micro {

TfLiteRegistration Register_ABS();
TfLiteRegistration Register_ADD();
TfLiteRegistration Register_ARG_MAX();
TfLiteRegistration Register_ARG_MIN();
TfLiteRegistration Register_AVERAGE_POOL_2D();
TfLiteRegistration Register_CEIL();
// TODO(b/160234179): Change custom OPs to also return by value.
TfLiteRegistration* Register_CIRCULAR_BUFFER();
TfLiteRegistration Register_CONCATENATION();
TfLiteRegistration Register_COS();
TfLiteRegistration Register_DEQUANTIZE();
TfLiteRegistration Register_EQUAL();
TfLiteRegistration Register_FLOOR();
TfLiteRegistration Register_GREATER();
TfLiteRegistration Register_GREATER_EQUAL();
TfLiteRegistration Register_HARD_SWISH();
TfLiteRegistration Register_LESS();
TfLiteRegistration Register_LESS_EQUAL();
TfLiteRegistration Register_LOG();
TfLiteRegistration Register_LOGICAL_AND();
TfLiteRegistration Register_LOGICAL_NOT();
TfLiteRegistration Register_LOGICAL_OR();
TfLiteRegistration Register_LOGISTIC();
TfLiteRegistration Register_MAXIMUM();
TfLiteRegistration Register_MAX_POOL_2D();
TfLiteRegistration Register_MEAN();
TfLiteRegistration Register_MINIMUM();
TfLiteRegistration Register_MUL();
TfLiteRegistration Register_NEG();
TfLiteRegistration Register_NOT_EQUAL();
TfLiteRegistration Register_PACK();
TfLiteRegistration Register_PAD();
TfLiteRegistration Register_PADV2();
TfLiteRegistration Register_PRELU();
TfLiteRegistration Register_REDUCE_MAX();
TfLiteRegistration Register_RELU();
TfLiteRegistration Register_RELU6();
TfLiteRegistration Register_RESHAPE();
TfLiteRegistration Register_RESIZE_NEAREST_NEIGHBOR();
TfLiteRegistration Register_ROUND();
TfLiteRegistration Register_RSQRT();
TfLiteRegistration Register_SIN();
TfLiteRegistration Register_SPLIT();
TfLiteRegistration Register_SPLIT_V();
TfLiteRegistration Register_SQRT();
TfLiteRegistration Register_SQUARE();
TfLiteRegistration Register_STRIDED_SLICE();
TfLiteRegistration Register_SUB();
TfLiteRegistration Register_UNPACK();
TfLiteRegistration Register_L2_NORMALIZATION();
TfLiteRegistration Register_TANH();

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_MICRO_OPS_H_
