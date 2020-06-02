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

#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"

namespace tflite {
namespace ops {
namespace micro {

AllOpsResolver::AllOpsResolver() {
  // Please keep this list of Builtin Operators in alphabetical order.
  AddBuiltin(BuiltinOperator_ABS, Register_ABS());
  AddBuiltin(BuiltinOperator_ADD, Register_ADD());
  AddBuiltin(BuiltinOperator_ARG_MAX, Register_ARG_MAX());
  AddBuiltin(BuiltinOperator_ARG_MIN, Register_ARG_MIN());
  AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, Register_AVERAGE_POOL_2D());
  AddBuiltin(BuiltinOperator_CEIL, Register_CEIL());
  AddBuiltin(BuiltinOperator_CONCATENATION, Register_CONCATENATION());
  AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D());
  AddBuiltin(BuiltinOperator_COS, Register_COS());
  AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D());
  AddBuiltin(BuiltinOperator_DEQUANTIZE, Register_DEQUANTIZE());
  AddBuiltin(BuiltinOperator_EQUAL, Register_EQUAL());
  AddBuiltin(BuiltinOperator_FLOOR, Register_FLOOR());
  AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Register_FULLY_CONNECTED());
  AddBuiltin(BuiltinOperator_GREATER, Register_GREATER());
  AddBuiltin(BuiltinOperator_GREATER_EQUAL, Register_GREATER_EQUAL());
  AddBuiltin(BuiltinOperator_L2_NORMALIZATION, Register_L2_NORMALIZATION());
  AddBuiltin(BuiltinOperator_LESS, Register_LESS());
  AddBuiltin(BuiltinOperator_LESS_EQUAL, Register_LESS_EQUAL());
  AddBuiltin(BuiltinOperator_LOG, Register_LOG());
  AddBuiltin(BuiltinOperator_LOGICAL_AND, Register_LOGICAL_AND());
  AddBuiltin(BuiltinOperator_LOGICAL_NOT, Register_LOGICAL_NOT());
  AddBuiltin(BuiltinOperator_LOGICAL_OR, Register_LOGICAL_OR());
  AddBuiltin(BuiltinOperator_LOGISTIC, Register_LOGISTIC());
  AddBuiltin(BuiltinOperator_MAX_POOL_2D, Register_MAX_POOL_2D());
  AddBuiltin(BuiltinOperator_MAXIMUM, Register_MAXIMUM());
  AddBuiltin(BuiltinOperator_MEAN, Register_MEAN());
  AddBuiltin(BuiltinOperator_MINIMUM, Register_MINIMUM());
  AddBuiltin(BuiltinOperator_MUL, Register_MUL());
  AddBuiltin(BuiltinOperator_NEG, Register_NEG());
  AddBuiltin(BuiltinOperator_NOT_EQUAL, Register_NOT_EQUAL());
  AddBuiltin(BuiltinOperator_PACK, Register_PACK());
  AddBuiltin(BuiltinOperator_PAD, Register_PAD());
  AddBuiltin(BuiltinOperator_PADV2, Register_PADV2());
  AddBuiltin(BuiltinOperator_PRELU, Register_PRELU());
  AddBuiltin(BuiltinOperator_QUANTIZE, Register_QUANTIZE());
  AddBuiltin(BuiltinOperator_RELU, Register_RELU());
  AddBuiltin(BuiltinOperator_RELU6, Register_RELU6());
  AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE());
  AddBuiltin(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
             Register_RESIZE_NEAREST_NEIGHBOR());
  AddBuiltin(BuiltinOperator_ROUND, Register_ROUND());
  AddBuiltin(BuiltinOperator_RSQRT, Register_RSQRT());
  AddBuiltin(BuiltinOperator_SIN, Register_SIN());
  AddBuiltin(BuiltinOperator_SOFTMAX, Register_SOFTMAX());
  AddBuiltin(BuiltinOperator_SPLIT, Register_SPLIT());
  AddBuiltin(BuiltinOperator_SQRT, Register_SQRT());
  AddBuiltin(BuiltinOperator_SQUARE, Register_SQUARE());
  AddBuiltin(BuiltinOperator_STRIDED_SLICE, Register_STRIDED_SLICE());
  AddBuiltin(BuiltinOperator_SUB, Register_SUB());
  AddBuiltin(BuiltinOperator_SVDF, Register_SVDF());
  AddBuiltin(BuiltinOperator_TANH, Register_TANH());
  AddBuiltin(BuiltinOperator_UNPACK, Register_UNPACK());
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
