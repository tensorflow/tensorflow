/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/lite/micro/all_ops_resolver.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace custom {
TfLiteRegistration* Register_ETHOSU();
const char* GetString_ETHOSU();
}  // namespace custom
}  // namespace micro
}  // namespace ops

AllOpsResolver::AllOpsResolver() {
  // Please keep this list of Builtin Operators in alphabetical order.
  AddBuiltin(BuiltinOperator_ABS, tflite::ops::micro::Register_ABS());
  AddBuiltin(BuiltinOperator_ADD, tflite::ops::micro::Register_ADD());
  AddBuiltin(BuiltinOperator_ARG_MAX, tflite::ops::micro::Register_ARG_MAX());
  AddBuiltin(BuiltinOperator_ARG_MIN, tflite::ops::micro::Register_ARG_MIN());
  AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D,
             tflite::ops::micro::Register_AVERAGE_POOL_2D());
  AddBuiltin(BuiltinOperator_CEIL, tflite::ops::micro::Register_CEIL());
  AddBuiltin(BuiltinOperator_CONCATENATION,
             tflite::ops::micro::Register_CONCATENATION());
  AddBuiltin(BuiltinOperator_CONV_2D, tflite::ops::micro::Register_CONV_2D());
  AddBuiltin(BuiltinOperator_COS, tflite::ops::micro::Register_COS());
  AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D,
             tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  AddBuiltin(BuiltinOperator_DEQUANTIZE,
             tflite::ops::micro::Register_DEQUANTIZE());
  AddBuiltin(BuiltinOperator_EQUAL, tflite::ops::micro::Register_EQUAL());
  AddBuiltin(BuiltinOperator_FLOOR, tflite::ops::micro::Register_FLOOR());
  AddBuiltin(BuiltinOperator_FULLY_CONNECTED,
             tflite::ops::micro::Register_FULLY_CONNECTED());
  AddBuiltin(BuiltinOperator_GREATER, tflite::ops::micro::Register_GREATER());
  AddBuiltin(BuiltinOperator_GREATER_EQUAL,
             tflite::ops::micro::Register_GREATER_EQUAL());
  AddBuiltin(BuiltinOperator_L2_NORMALIZATION,
             tflite::ops::micro::Register_L2_NORMALIZATION());
  AddBuiltin(BuiltinOperator_LESS, tflite::ops::micro::Register_LESS());
  AddBuiltin(BuiltinOperator_LESS_EQUAL,
             tflite::ops::micro::Register_LESS_EQUAL());
  AddBuiltin(BuiltinOperator_LOG, tflite::ops::micro::Register_LOG());
  AddBuiltin(BuiltinOperator_LOGICAL_AND,
             tflite::ops::micro::Register_LOGICAL_AND());
  AddBuiltin(BuiltinOperator_LOGICAL_NOT,
             tflite::ops::micro::Register_LOGICAL_NOT());
  AddBuiltin(BuiltinOperator_LOGICAL_OR,
             tflite::ops::micro::Register_LOGICAL_OR());
  AddBuiltin(BuiltinOperator_LOGISTIC, tflite::ops::micro::Register_LOGISTIC());
  AddBuiltin(BuiltinOperator_MAX_POOL_2D,
             tflite::ops::micro::Register_MAX_POOL_2D());
  AddBuiltin(BuiltinOperator_MAXIMUM, tflite::ops::micro::Register_MAXIMUM());
  AddBuiltin(BuiltinOperator_MEAN, tflite::ops::micro::Register_MEAN());
  AddBuiltin(BuiltinOperator_MINIMUM, tflite::ops::micro::Register_MINIMUM());
  AddBuiltin(BuiltinOperator_MUL, tflite::ops::micro::Register_MUL());
  AddBuiltin(BuiltinOperator_NEG, tflite::ops::micro::Register_NEG());
  AddBuiltin(BuiltinOperator_NOT_EQUAL,
             tflite::ops::micro::Register_NOT_EQUAL());
  AddBuiltin(BuiltinOperator_PACK, tflite::ops::micro::Register_PACK());
  AddBuiltin(BuiltinOperator_PAD, tflite::ops::micro::Register_PAD());
  AddBuiltin(BuiltinOperator_PADV2, tflite::ops::micro::Register_PADV2());
  AddBuiltin(BuiltinOperator_PRELU, tflite::ops::micro::Register_PRELU());
  AddBuiltin(BuiltinOperator_QUANTIZE, tflite::ops::micro::Register_QUANTIZE());
  AddBuiltin(BuiltinOperator_RELU, tflite::ops::micro::Register_RELU());
  AddBuiltin(BuiltinOperator_RELU6, tflite::ops::micro::Register_RELU6());
  AddBuiltin(BuiltinOperator_RESHAPE, tflite::ops::micro::Register_RESHAPE());
  AddBuiltin(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
             tflite::ops::micro::Register_RESIZE_NEAREST_NEIGHBOR());
  AddBuiltin(BuiltinOperator_ROUND, tflite::ops::micro::Register_ROUND());
  AddBuiltin(BuiltinOperator_RSQRT, tflite::ops::micro::Register_RSQRT());
  AddBuiltin(BuiltinOperator_SIN, tflite::ops::micro::Register_SIN());
  AddBuiltin(BuiltinOperator_SOFTMAX, tflite::ops::micro::Register_SOFTMAX());
  AddBuiltin(BuiltinOperator_SPLIT, tflite::ops::micro::Register_SPLIT());
  AddBuiltin(BuiltinOperator_SQRT, tflite::ops::micro::Register_SQRT());
  AddBuiltin(BuiltinOperator_SQUARE, tflite::ops::micro::Register_SQUARE());
  AddBuiltin(BuiltinOperator_STRIDED_SLICE,
             tflite::ops::micro::Register_STRIDED_SLICE());
  AddBuiltin(BuiltinOperator_SUB, tflite::ops::micro::Register_SUB());
  AddBuiltin(BuiltinOperator_SVDF, tflite::ops::micro::Register_SVDF());
  AddBuiltin(BuiltinOperator_TANH, tflite::ops::micro::Register_TANH());
  AddBuiltin(BuiltinOperator_UNPACK, tflite::ops::micro::Register_UNPACK());

  TfLiteRegistration* registration =
      tflite::ops::micro::custom::Register_ETHOSU();
  if (registration) {
    AddCustom(tflite::ops::micro::custom::GetString_ETHOSU(), registration);
  }
}

}  // namespace tflite
