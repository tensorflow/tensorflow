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
/// WARNING: Users of TensorFlow Lite should not include this file directly,
/// but should instead include
/// "third_party/tensorflow/lite/kernels/builtin_op_kernels.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
// IWYU pragma: private, include "third_party/tensorflow/lite/kernels/builtin_op_kernels.h"

#ifndef TENSORFLOW_LITE_CORE_KERNELS_BUILTIN_OP_KERNELS_H_
#define TENSORFLOW_LITE_CORE_KERNELS_BUILTIN_OP_KERNELS_H_

#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace ops {
namespace builtin {

// Forward declaration of all builtin op kernel registration methods. These
// registrations are included with the standard `BuiltinOpResolver`.
//
// This header is particularly useful in cases where only a subset of ops are
// needed. In such cases, the client can selectively add only the registrations
// their model requires, using a custom `OpResolver` or `MutableOpResolver`.
// Selective registration in turn allows the linker to strip unused kernels.
//
// TODO(b/184734878): auto-generate this header file from the BuiltinOperator
// enum in the FlatBuffer schema.

TfLiteRegistration* Register_ABS();
TfLiteRegistration* Register_ADD();
TfLiteRegistration* Register_ADD_N();
TfLiteRegistration* Register_ARG_MAX();
TfLiteRegistration* Register_ARG_MIN();
TfLiteRegistration* Register_ASSIGN_VARIABLE();
TfLiteRegistration* Register_ATAN2();
TfLiteRegistration* Register_AVERAGE_POOL_2D();
TfLiteRegistration* Register_AVERAGE_POOL_3D();
TfLiteRegistration* Register_BATCH_TO_SPACE_ND();
TfLiteRegistration* Register_BATCH_MATMUL();
TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_LSTM();
TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_RNN();
TfLiteRegistration* Register_BROADCAST_ARGS();
TfLiteRegistration* Register_BROADCAST_TO();
TfLiteRegistration* Register_BUCKETIZE();
TfLiteRegistration* Register_CALL_ONCE();
TfLiteRegistration* Register_CAST();
TfLiteRegistration* Register_CEIL();
TfLiteRegistration* Register_COMPLEX_ABS();
TfLiteRegistration* Register_CONCATENATION();
TfLiteRegistration* Register_CONV_2D();
TfLiteRegistration* Register_CONV_3D();
TfLiteRegistration* Register_CONV_3D_TRANSPOSE();
TfLiteRegistration* Register_COS();
TfLiteRegistration* Register_CUMSUM();
TfLiteRegistration* Register_DENSIFY();
TfLiteRegistration* Register_DEPTH_TO_SPACE();
TfLiteRegistration* Register_DEPTHWISE_CONV_2D();
TfLiteRegistration* Register_DEQUANTIZE();
TfLiteRegistration* Register_DIV();
TfLiteRegistration* Register_DYNAMIC_UPDATE_SLICE();
TfLiteRegistration* Register_ELU();
TfLiteRegistration* Register_EMBEDDING_LOOKUP();
TfLiteRegistration* Register_EMBEDDING_LOOKUP_SPARSE();
TfLiteRegistration* Register_EQUAL();
TfLiteRegistration* Register_EXP();
TfLiteRegistration* Register_EXPAND_DIMS();
TfLiteRegistration* Register_FAKE_QUANT();
TfLiteRegistration* Register_FILL();
TfLiteRegistration* Register_FLOOR();
TfLiteRegistration* Register_FLOOR_DIV();
TfLiteRegistration* Register_FLOOR_MOD();
TfLiteRegistration* Register_FULLY_CONNECTED();
TfLiteRegistration* Register_GATHER();
TfLiteRegistration* Register_GATHER_ND();
TfLiteRegistration* Register_GELU();
TfLiteRegistration* Register_GREATER();
TfLiteRegistration* Register_GREATER_EQUAL();
TfLiteRegistration* Register_HARD_SWISH();
TfLiteRegistration* Register_HASHTABLE();
TfLiteRegistration* Register_HASHTABLE_FIND();
TfLiteRegistration* Register_HASHTABLE_LOOKUP();
TfLiteRegistration* Register_HASHTABLE_IMPORT();
TfLiteRegistration* Register_HASHTABLE_SIZE();
TfLiteRegistration* Register_IF();
TfLiteRegistration* Register_IMAG();
TfLiteRegistration* Register_L2_NORMALIZATION();
TfLiteRegistration* Register_L2_POOL_2D();
TfLiteRegistration* Register_LEAKY_RELU();
TfLiteRegistration* Register_LESS();
TfLiteRegistration* Register_LESS_EQUAL();
TfLiteRegistration* Register_LOCAL_RESPONSE_NORMALIZATION();
TfLiteRegistration* Register_LOG();
TfLiteRegistration* Register_LOGICAL_AND();
TfLiteRegistration* Register_LOGICAL_NOT();
TfLiteRegistration* Register_LOGICAL_OR();
TfLiteRegistration* Register_LOGISTIC();
TfLiteRegistration* Register_LOG_SOFTMAX();
TfLiteRegistration* Register_LSH_PROJECTION();
TfLiteRegistration* Register_LSTM();
TfLiteRegistration* Register_MATRIX_DIAG();
TfLiteRegistration* Register_MATRIX_SET_DIAG();
TfLiteRegistration* Register_MAXIMUM();
TfLiteRegistration* Register_MAX_POOL_2D();
TfLiteRegistration* Register_MAX_POOL_3D();
TfLiteRegistration* Register_MEAN();
TfLiteRegistration* Register_MINIMUM();
TfLiteRegistration* Register_MIRROR_PAD();
TfLiteRegistration* Register_MUL();
TfLiteRegistration* Register_NEG();
TfLiteRegistration* Register_NON_MAX_SUPPRESSION_V4();
TfLiteRegistration* Register_NON_MAX_SUPPRESSION_V5();
TfLiteRegistration* Register_NOT_EQUAL();
TfLiteRegistration* Register_ONE_HOT();
TfLiteRegistration* Register_PACK();
TfLiteRegistration* Register_PAD();
TfLiteRegistration* Register_PADV2();
TfLiteRegistration* Register_POW();
TfLiteRegistration* Register_PRELU();
TfLiteRegistration* Register_QUANTIZE();
TfLiteRegistration* Register_MULTINOMIAL();
TfLiteRegistration* Register_RANDOM_STANDARD_NORMAL();
TfLiteRegistration* Register_RANDOM_UNIFORM();
TfLiteRegistration* Register_RANGE();
TfLiteRegistration* Register_RANK();
TfLiteRegistration* Register_READ_VARIABLE();
TfLiteRegistration* Register_REAL();
TfLiteRegistration* Register_REDUCE_ALL();
TfLiteRegistration* Register_REDUCE_ANY();
TfLiteRegistration* Register_REDUCE_MAX();
TfLiteRegistration* Register_REDUCE_MIN();
TfLiteRegistration* Register_REDUCE_PROD();
TfLiteRegistration* Register_RELU();
TfLiteRegistration* Register_RELU6();
TfLiteRegistration* Register_RELU_N1_TO_1();
TfLiteRegistration* Register_RELU_0_TO_1();
TfLiteRegistration* Register_RESHAPE();
TfLiteRegistration* Register_RESIZE_BILINEAR();
TfLiteRegistration* Register_RESIZE_NEAREST_NEIGHBOR();
TfLiteRegistration* Register_REVERSE_SEQUENCE();
TfLiteRegistration* Register_REVERSE_V2();
TfLiteRegistration* Register_RFFT2D();
TfLiteRegistration* Register_RNN();
TfLiteRegistration* Register_ROUND();
TfLiteRegistration* Register_RSQRT();
TfLiteRegistration* Register_SCATTER_ND();
TfLiteRegistration* Register_SEGMENT_SUM();
TfLiteRegistration* Register_SELECT();
TfLiteRegistration* Register_SELECT_V2();
TfLiteRegistration* Register_SHAPE();
TfLiteRegistration* Register_SIGN();
TfLiteRegistration* Register_SIN();
TfLiteRegistration* Register_SKIP_GRAM();
TfLiteRegistration* Register_SLICE();
TfLiteRegistration* Register_SOFTMAX();
TfLiteRegistration* Register_SPACE_TO_BATCH_ND();
TfLiteRegistration* Register_SPACE_TO_DEPTH();
TfLiteRegistration* Register_SPARSE_TO_DENSE();
TfLiteRegistration* Register_SPLIT();
TfLiteRegistration* Register_SPLIT_V();
TfLiteRegistration* Register_SQRT();
TfLiteRegistration* Register_SQUARE();
TfLiteRegistration* Register_SQUARED_DIFFERENCE();
TfLiteRegistration* Register_SQUEEZE();
TfLiteRegistration* Register_STRIDED_SLICE();
TfLiteRegistration* Register_SUB();
TfLiteRegistration* Register_SUM();
TfLiteRegistration* Register_SVDF();
TfLiteRegistration* Register_TANH();
TfLiteRegistration* Register_TILE();
TfLiteRegistration* Register_TOPK_V2();
TfLiteRegistration* Register_TRANSPOSE();
TfLiteRegistration* Register_TRANSPOSE_CONV();
TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_RNN();
TfLiteRegistration* Register_UNIQUE();
TfLiteRegistration* Register_UNPACK();
TfLiteRegistration* Register_UNSORTED_SEGMENT_MAX();
TfLiteRegistration* Register_UNSORTED_SEGMENT_MIN();
TfLiteRegistration* Register_UNSORTED_SEGMENT_PROD();
TfLiteRegistration* Register_UNSORTED_SEGMENT_SUM();
TfLiteRegistration* Register_VAR_HANDLE();
TfLiteRegistration* Register_WHERE();
TfLiteRegistration* Register_WHILE();
TfLiteRegistration* Register_ZEROS_LIKE();
TfLiteRegistration* Register_BITCAST();
TfLiteRegistration* Register_BITWISE_XOR();
TfLiteRegistration* Register_RIGHT_SHIFT();

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_KERNELS_BUILTIN_OP_KERNELS_H_
