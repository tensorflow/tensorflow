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

#include "tensorflow/lite/kernels/register_ref.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {

namespace custom {

TfLiteRegistration* Register_NUMERIC_VERIFY_REF();
TfLiteRegistration* Register_AUDIO_SPECTROGRAM();
TfLiteRegistration* Register_MFCC();
TfLiteRegistration* Register_DETECTION_POSTPROCESS();

}  // namespace custom

namespace builtin {

// TODO(yunluli): Some of the registries, e.g. Tanh(), could only invoke
// optimized kernels. Add a _REF() variant for them.
TfLiteRegistration* Register_ABS();
TfLiteRegistration* Register_RELU();
TfLiteRegistration* Register_RELU_N1_TO_1();
TfLiteRegistration* Register_RELU_0_TO_1();
TfLiteRegistration* Register_RELU6();
TfLiteRegistration* Register_TANH_REF();
TfLiteRegistration* Register_LOGISTIC_REF();
TfLiteRegistration* Register_AVERAGE_POOL_REF();
TfLiteRegistration* Register_MAX_POOL_REF();
TfLiteRegistration* Register_L2_POOL_REF();
TfLiteRegistration* Register_CONVOLUTION_REF();
TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_REF();
TfLiteRegistration* Register_SVDF();
TfLiteRegistration* Register_RNN();
TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_RNN();
TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_RNN();
TfLiteRegistration* Register_EMBEDDING_LOOKUP();
TfLiteRegistration* Register_EMBEDDING_LOOKUP_SPARSE();
TfLiteRegistration* Register_FULLY_CONNECTED_REF();
TfLiteRegistration* Register_LSH_PROJECTION();
TfLiteRegistration* Register_HASHTABLE_LOOKUP();
TfLiteRegistration* Register_SOFTMAX_REF();
TfLiteRegistration* Register_CONCATENATION_REF();
TfLiteRegistration* Register_ADD_REF();
TfLiteRegistration* Register_SPACE_TO_BATCH_ND_REF();
TfLiteRegistration* Register_DIV_REF();
TfLiteRegistration* Register_SUB_REF();
TfLiteRegistration* Register_BATCH_TO_SPACE_ND_REF();
TfLiteRegistration* Register_MUL_REF();
TfLiteRegistration* Register_L2NORM_REF();
TfLiteRegistration* Register_LOCAL_RESPONSE_NORM_REF();
TfLiteRegistration* Register_LSTM();
TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_LSTM();
TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
TfLiteRegistration* Register_PAD_REF();
TfLiteRegistration* Register_PADV2_REF();
TfLiteRegistration* Register_RESHAPE();
TfLiteRegistration* Register_RESIZE_BILINEAR_REF();
TfLiteRegistration* Register_RESIZE_NEAREST_NEIGHBOR_REF();
TfLiteRegistration* Register_SKIP_GRAM();
TfLiteRegistration* Register_SPACE_TO_DEPTH_REF();
TfLiteRegistration* Register_GATHER();
TfLiteRegistration* Register_TRANSPOSE_REF();
TfLiteRegistration* Register_MEAN_REF();
TfLiteRegistration* Register_SPLIT();
TfLiteRegistration* Register_SPLIT_V();
TfLiteRegistration* Register_SQUEEZE();
TfLiteRegistration* Register_STRIDED_SLICE_REF();
TfLiteRegistration* Register_EXP_REF();
TfLiteRegistration* Register_TOPK_V2();
TfLiteRegistration* Register_LOG();
TfLiteRegistration* Register_LOG_SOFTMAX_REF();
TfLiteRegistration* Register_CAST();
TfLiteRegistration* Register_DEQUANTIZE_REF();
TfLiteRegistration* Register_PRELU_REF();
TfLiteRegistration* Register_MAXIMUM_REF();
TfLiteRegistration* Register_MINIMUM_REF();
TfLiteRegistration* Register_ARG_MAX();
TfLiteRegistration* Register_ARG_MIN();
TfLiteRegistration* Register_GREATER();
TfLiteRegistration* Register_GREATER_EQUAL();
TfLiteRegistration* Register_LESS();
TfLiteRegistration* Register_LESS_EQUAL();
TfLiteRegistration* Register_FLOOR_REF();
TfLiteRegistration* Register_TILE();
TfLiteRegistration* Register_NEG();
TfLiteRegistration* Register_SUM_REF();
TfLiteRegistration* Register_REDUCE_PROD_REF();
TfLiteRegistration* Register_REDUCE_MAX_REF();
TfLiteRegistration* Register_REDUCE_MIN_REF();
TfLiteRegistration* Register_REDUCE_ANY_REF();
TfLiteRegistration* Register_REDUCE_ALL_REF();
TfLiteRegistration* Register_SELECT();
TfLiteRegistration* Register_SLICE_REF();
TfLiteRegistration* Register_SIN();
TfLiteRegistration* Register_COS();
TfLiteRegistration* Register_TRANSPOSECONV_REF();
TfLiteRegistration* Register_EXPAND_DIMS();
TfLiteRegistration* Register_SPARSE_TO_DENSE();
TfLiteRegistration* Register_EQUAL();
TfLiteRegistration* Register_NOT_EQUAL();
TfLiteRegistration* Register_SQRT();
TfLiteRegistration* Register_RSQRT();
TfLiteRegistration* Register_SHAPE();
TfLiteRegistration* Register_RANK();
TfLiteRegistration* Register_POW();
TfLiteRegistration* Register_FAKE_QUANT_REF();
TfLiteRegistration* Register_PACK();
TfLiteRegistration* Register_ONE_HOT();
TfLiteRegistration* Register_LOGICAL_OR();
TfLiteRegistration* Register_LOGICAL_AND();
TfLiteRegistration* Register_LOGICAL_NOT();
TfLiteRegistration* Register_UNPACK();
TfLiteRegistration* Register_FLOOR_DIV();
TfLiteRegistration* Register_SQUARE();
TfLiteRegistration* Register_ZEROS_LIKE();
TfLiteRegistration* Register_FLOOR_MOD();
TfLiteRegistration* Register_RANGE();
TfLiteRegistration* Register_LEAKY_RELU_REF();
TfLiteRegistration* Register_SQUARED_DIFFERENCE();
TfLiteRegistration* Register_FILL();
TfLiteRegistration* Register_MIRROR_PAD();
TfLiteRegistration* Register_UNIQUE();
TfLiteRegistration* Register_REVERSE_V2();
TfLiteRegistration* Register_ADD_N();
TfLiteRegistration* Register_GATHER_ND();
TfLiteRegistration* Register_WHERE();
TfLiteRegistration* Register_REVERSE_SEQUENCE();
TfLiteRegistration* Register_MATRIX_DIAG();
TfLiteRegistration* Register_QUANTIZE_REF();
TfLiteRegistration* Register_MATRIX_SET_DIAG();
TfLiteRegistration* Register_IF();
TfLiteRegistration* Register_WHILE();
TfLiteRegistration* Register_NON_MAX_SUPPRESSION_V4();
TfLiteRegistration* Register_NON_MAX_SUPPRESSION_V5();
TfLiteRegistration* Register_SCATTER_ND();
TfLiteRegistration* Register_DENSIFY();
TfLiteRegistration* Register_BATCH_MATMUL_REF();
TfLiteRegistration* Register_HARD_SWISH_REF();
TfLiteRegistration* Register_DEPTH_TO_SPACE_REF();
TfLiteRegistration* Register_SELECT_V2();
TfLiteRegistration* Register_SEGMENT_SUM();
TfLiteRegistration* Register_BROADCAST_TO();
TfLiteRegistration* Register_CONV_3D_REF();
TfLiteRegistration* Register_IMAG();
TfLiteRegistration* Register_REAL();
TfLiteRegistration* Register_COMPLEX_ABS();
TfLiteRegistration* Register_CONV_3D_TRANSPOSE_REF();
TfLiteRegistration* Register_BROADCAST_ARGS();
TfLiteRegistration* Register_RANDOM_STANDARD_NORMAL();
TfLiteRegistration* Register_BUCKETIZE();
TfLiteRegistration* Register_RANDOM_UNIFORM();
TfLiteRegistration* Register_MULTINOMIAL();
TfLiteRegistration* Register_GELU();
TfLiteRegistration* Register_DYNAMIC_UPDATE_SLICE();
TfLiteRegistration* Register_UNSORTED_SEGMENT_PROD();
TfLiteRegistration* Register_UNSORTED_SEGMENT_MAX();
TfLiteRegistration* Register_UNSORTED_SEGMENT_MIN();
TfLiteRegistration* Register_UNSORTED_SEGMENT_SUM();
TfLiteRegistration* Register_ATAN2();
TfLiteRegistration* Register_SIGN();
TfLiteRegistration* Register_CALL_ONCE();
TfLiteRegistration* Register_VAR_HANDLE();
TfLiteRegistration* Register_READ_VARIABLE();
TfLiteRegistration* Register_ASSIGN_VARIABLE();

namespace {

TfLiteStatus UnsupportedTensorFlowOp(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(
      context,
      "Regular TensorFlow ops are not supported by this interpreter. Make sure "
      "you invoke the Flex delegate before inference.");
  return kTfLiteError;
}

}  // namespace

const TfLiteRegistration* BuiltinRefOpResolver::FindOp(
    tflite::BuiltinOperator op, int version) const {
  return MutableOpResolver::FindOp(op, version);
}

const TfLiteRegistration* BuiltinRefOpResolver::FindOp(const char* op,
                                                       int version) const {
  // Return the NULL Op for all ops whose name start with "Flex", allowing
  // the interpreter to delegate their execution.
  if (IsFlexOp(op)) {
    static TfLiteRegistration null_op{
        nullptr, nullptr, &UnsupportedTensorFlowOp,
        nullptr, nullptr, BuiltinOperator_CUSTOM,
        "Flex",  1};
    return &null_op;
  }
  return MutableOpResolver::FindOp(op, version);
}

BuiltinRefOpResolver::BuiltinRefOpResolver() {
  AddBuiltin(BuiltinOperator_ABS, Register_ABS(), /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_HARD_SWISH, Register_HARD_SWISH_REF());
  AddBuiltin(BuiltinOperator_RELU, Register_RELU(), /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_RELU_N1_TO_1, Register_RELU_N1_TO_1());
  AddBuiltin(BuiltinOperator_RELU_0_TO_1, Register_RELU_0_TO_1());
  AddBuiltin(BuiltinOperator_RELU6, Register_RELU6(), /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_TANH, Register_TANH_REF(), /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_LOGISTIC, Register_LOGISTIC_REF(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, Register_AVERAGE_POOL_REF(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_MAX_POOL_2D, Register_MAX_POOL_REF(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_L2_POOL_2D, Register_L2_POOL_REF());
  AddBuiltin(BuiltinOperator_CONV_2D, Register_CONVOLUTION_REF(),
             /* min_version = */ 1,
             /* max_version = */ 6);
  AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D,
             Register_DEPTHWISE_CONVOLUTION_REF(),
             /* min_version = */ 1,
             /* max_version = */ 6);
  AddBuiltin(BuiltinOperator_SVDF, Register_SVDF(),
             /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_RNN, Register_RNN(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN,
             Register_BIDIRECTIONAL_SEQUENCE_RNN(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN,
             Register_UNIDIRECTIONAL_SEQUENCE_RNN(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP, Register_EMBEDDING_LOOKUP(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP_SPARSE,
             Register_EMBEDDING_LOOKUP_SPARSE());
  AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Register_FULLY_CONNECTED_REF(),
             /* min_version */ 1,
             /* max_version */ 9);
  AddBuiltin(BuiltinOperator_LSH_PROJECTION, Register_LSH_PROJECTION());
  AddBuiltin(BuiltinOperator_HASHTABLE_LOOKUP, Register_HASHTABLE_LOOKUP());
  AddBuiltin(BuiltinOperator_SOFTMAX, Register_SOFTMAX_REF(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_CONCATENATION, Register_CONCATENATION_REF(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_ADD, Register_ADD_REF(),
             /* min_version */ 1,
             /* max_version */ 5);
  AddBuiltin(BuiltinOperator_SPACE_TO_BATCH_ND,
             Register_SPACE_TO_BATCH_ND_REF(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_BATCH_TO_SPACE_ND,
             Register_BATCH_TO_SPACE_ND_REF(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_MUL, Register_MUL_REF(), /* min_version = */ 1,
             /* max_version = */ 6);
  AddBuiltin(BuiltinOperator_L2_NORMALIZATION, Register_L2NORM_REF(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  // The version one of broadcast to op won't be not supported since the version
  // one was rollbacked and the builtin op code number has been changed because
  // of builtin op code shortage problem.
  AddBuiltin(BuiltinOperator_BROADCAST_TO, Register_BROADCAST_TO(),
             /* min_version = */ 2,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
             Register_LOCAL_RESPONSE_NORM_REF());
  AddBuiltin(BuiltinOperator_LSTM, Register_LSTM(), /* min_version */ 1,
             /* max_version */ 4);
  AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM,
             Register_BIDIRECTIONAL_SEQUENCE_LSTM(), /* min_version */ 1,
             /* max_version */ 3);
  AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM,
             Register_UNIDIRECTIONAL_SEQUENCE_LSTM(), /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_PAD, Register_PAD_REF(), /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_PADV2, Register_PADV2_REF(), /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE());
  AddBuiltin(BuiltinOperator_RESIZE_BILINEAR, Register_RESIZE_BILINEAR_REF(),
             /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
             Register_RESIZE_NEAREST_NEIGHBOR_REF(),
             /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_SKIP_GRAM, Register_SKIP_GRAM());
  AddBuiltin(BuiltinOperator_SPACE_TO_DEPTH, Register_SPACE_TO_DEPTH_REF(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_DEPTH_TO_SPACE, Register_DEPTH_TO_SPACE_REF());
  AddBuiltin(BuiltinOperator_GATHER, Register_GATHER(),
             /* min_version = */ 1,
             /* max_version = */ 6);
  AddBuiltin(BuiltinOperator_TRANSPOSE, Register_TRANSPOSE_REF(),
             /* min_version = */ 1,
             /* max_version = */ 5);
  AddBuiltin(BuiltinOperator_MEAN, Register_MEAN_REF(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_DIV, Register_DIV_REF(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_SUB, Register_SUB_REF(),
             /* min_version = */ 1,
             /* max_version = */ 5);
  AddBuiltin(BuiltinOperator_SPLIT, Register_SPLIT(),
             /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_SPLIT_V, Register_SPLIT_V(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_SQUEEZE, Register_SQUEEZE());
  AddBuiltin(BuiltinOperator_STRIDED_SLICE, Register_STRIDED_SLICE_REF(),
             /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_EXP, Register_EXP_REF(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_TOPK_V2, Register_TOPK_V2(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_LOG, Register_LOG());
  AddBuiltin(BuiltinOperator_LOG_SOFTMAX, Register_LOG_SOFTMAX_REF(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_CAST, Register_CAST(),
             /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_DEQUANTIZE, Register_DEQUANTIZE_REF(),
             /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_PRELU, Register_PRELU_REF());
  AddBuiltin(BuiltinOperator_MAXIMUM, Register_MAXIMUM_REF(),
             /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_MINIMUM, Register_MINIMUM_REF(),
             /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_ARG_MAX, Register_ARG_MAX(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_ARG_MIN, Register_ARG_MIN(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_GREATER, Register_GREATER(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_GREATER_EQUAL, Register_GREATER_EQUAL(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_LESS, Register_LESS(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_LESS_EQUAL, Register_LESS_EQUAL(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_FLOOR, Register_FLOOR_REF());
  AddBuiltin(BuiltinOperator_NEG, Register_NEG());
  AddBuiltin(BuiltinOperator_SELECT, Register_SELECT(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_SELECT_V2, Register_SELECT_V2());
  AddBuiltin(BuiltinOperator_SLICE, Register_SLICE_REF(),
             /* min_version = */ 1,
             /* max_version = */ 5);
  AddBuiltin(BuiltinOperator_SIN, Register_SIN());
  AddBuiltin(BuiltinOperator_COS, Register_COS());
  AddBuiltin(BuiltinOperator_TRANSPOSE_CONV, Register_TRANSPOSECONV_REF(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_TILE, Register_TILE(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_SUM, Register_SUM_REF(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_REDUCE_PROD, Register_REDUCE_PROD_REF(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_REDUCE_MAX, Register_REDUCE_MAX_REF(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_REDUCE_MIN, Register_REDUCE_MIN_REF(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_REDUCE_ANY, Register_REDUCE_ANY_REF());
  AddBuiltin(BuiltinOperator_REDUCE_ALL, Register_REDUCE_ALL_REF());
  AddBuiltin(BuiltinOperator_EXPAND_DIMS, Register_EXPAND_DIMS());
  AddBuiltin(BuiltinOperator_SPARSE_TO_DENSE, Register_SPARSE_TO_DENSE(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_EQUAL, Register_EQUAL(),
             /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_NOT_EQUAL, Register_NOT_EQUAL(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_SQRT, Register_SQRT());
  AddBuiltin(BuiltinOperator_RSQRT, Register_RSQRT(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_SHAPE, Register_SHAPE());
  AddBuiltin(BuiltinOperator_RANK, Register_RANK());
  AddBuiltin(BuiltinOperator_POW, Register_POW());
  AddBuiltin(BuiltinOperator_FAKE_QUANT, Register_FAKE_QUANT_REF(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_PACK, Register_PACK(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_ONE_HOT, Register_ONE_HOT());
  AddBuiltin(BuiltinOperator_LOGICAL_OR, Register_LOGICAL_OR());
  AddBuiltin(BuiltinOperator_LOGICAL_AND, Register_LOGICAL_AND());
  AddBuiltin(BuiltinOperator_LOGICAL_NOT, Register_LOGICAL_NOT());
  AddBuiltin(BuiltinOperator_UNPACK, Register_UNPACK(),
             /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_FLOOR_DIV, Register_FLOOR_DIV(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_SQUARE, Register_SQUARE());
  AddBuiltin(BuiltinOperator_ZEROS_LIKE, Register_ZEROS_LIKE());
  AddBuiltin(BuiltinOperator_FLOOR_MOD, Register_FLOOR_MOD(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_RANGE, Register_RANGE());
  AddBuiltin(BuiltinOperator_LEAKY_RELU, Register_LEAKY_RELU_REF(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_SQUARED_DIFFERENCE, Register_SQUARED_DIFFERENCE(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_FILL, Register_FILL(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_MIRROR_PAD, Register_MIRROR_PAD(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_UNIQUE, Register_UNIQUE());
  AddBuiltin(BuiltinOperator_REVERSE_V2, Register_REVERSE_V2(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_ADD_N, Register_ADD_N());
  AddBuiltin(BuiltinOperator_GATHER_ND, Register_GATHER_ND(),
             /* min_version = */ 1,
             /* max_version = */ 4);
  AddBuiltin(BuiltinOperator_WHERE, Register_WHERE(), /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_REVERSE_SEQUENCE, Register_REVERSE_SEQUENCE());
  AddBuiltin(BuiltinOperator_MATRIX_DIAG, Register_MATRIX_DIAG());
  AddBuiltin(BuiltinOperator_QUANTIZE, Register_QUANTIZE_REF(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_MATRIX_SET_DIAG, Register_MATRIX_SET_DIAG());
  AddBuiltin(BuiltinOperator_IF, Register_IF());
  AddBuiltin(BuiltinOperator_WHILE, Register_WHILE());
  AddBuiltin(BuiltinOperator_NON_MAX_SUPPRESSION_V4,
             Register_NON_MAX_SUPPRESSION_V4());
  AddBuiltin(BuiltinOperator_NON_MAX_SUPPRESSION_V5,
             Register_NON_MAX_SUPPRESSION_V5());
  AddBuiltin(BuiltinOperator_SCATTER_ND, Register_SCATTER_ND());
  AddBuiltin(BuiltinOperator_DENSIFY, Register_DENSIFY());
  AddBuiltin(BuiltinOperator_BATCH_MATMUL, Register_BATCH_MATMUL_REF(),
             /* min_version = */ 1,
             /* max_version = */ 3);
  AddBuiltin(BuiltinOperator_CONV_3D, Register_CONV_3D_REF());
  AddBuiltin(BuiltinOperator_IMAG, Register_IMAG());
  AddBuiltin(BuiltinOperator_REAL, Register_REAL());
  AddBuiltin(BuiltinOperator_COMPLEX_ABS, Register_COMPLEX_ABS());
  AddBuiltin(BuiltinOperator_CONV_3D_TRANSPOSE,
             Register_CONV_3D_TRANSPOSE_REF());
  AddBuiltin(BuiltinOperator_BROADCAST_ARGS, Register_BROADCAST_ARGS());
  AddBuiltin(BuiltinOperator_MULTINOMIAL, Register_MULTINOMIAL());
  AddBuiltin(BuiltinOperator_RANDOM_STANDARD_NORMAL,
             Register_RANDOM_STANDARD_NORMAL());
  AddBuiltin(BuiltinOperator_BUCKETIZE, Register_BUCKETIZE());
  AddBuiltin(BuiltinOperator_RANDOM_UNIFORM, Register_RANDOM_UNIFORM());
  AddBuiltin(BuiltinOperator_GELU, Register_GELU(),
             /* min_version = */ 1,
             /* max_version = */ 2);
  AddBuiltin(BuiltinOperator_DYNAMIC_UPDATE_SLICE,
             Register_DYNAMIC_UPDATE_SLICE());
  AddBuiltin(BuiltinOperator_UNSORTED_SEGMENT_PROD,
             Register_UNSORTED_SEGMENT_PROD());
  AddBuiltin(BuiltinOperator_UNSORTED_SEGMENT_MAX,
             Register_UNSORTED_SEGMENT_MAX());
  AddBuiltin(BuiltinOperator_UNSORTED_SEGMENT_MIN,
             Register_UNSORTED_SEGMENT_MIN());
  AddBuiltin(BuiltinOperator_UNSORTED_SEGMENT_SUM,
             Register_UNSORTED_SEGMENT_SUM());
  AddBuiltin(BuiltinOperator_ATAN2, Register_ATAN2());
  AddBuiltin(BuiltinOperator_SIGN, Register_SIGN());
  AddBuiltin(BuiltinOperator_CALL_ONCE,
             tflite::ops::builtin::Register_CALL_ONCE());
  AddBuiltin(BuiltinOperator_VAR_HANDLE, Register_VAR_HANDLE());
  AddBuiltin(BuiltinOperator_READ_VARIABLE, Register_READ_VARIABLE());
  AddBuiltin(BuiltinOperator_ASSIGN_VARIABLE, Register_ASSIGN_VARIABLE());
  AddCustom("NumericVerify",
            tflite::ops::custom::Register_NUMERIC_VERIFY_REF());
  // TODO(andrewharp, ahentz): Move these somewhere more appropriate so that
  // custom ops aren't always included by default.
  AddCustom("Mfcc", tflite::ops::custom::Register_MFCC());
  AddCustom("AudioSpectrogram",
            tflite::ops::custom::Register_AUDIO_SPECTROGRAM());
  AddCustom("TFLite_Detection_PostProcess",
            tflite::ops::custom::Register_DETECTION_POSTPROCESS());
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
