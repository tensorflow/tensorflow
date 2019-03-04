/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {

namespace custom {

TfLiteRegistration* Register_AUDIO_SPECTROGRAM();
TfLiteRegistration* Register_MFCC();
TfLiteRegistration* Register_DETECTION_POSTPROCESS();
TfLiteRegistration* Register_IF();
TfLiteRegistration* Register_WHILE();

}  // namespace custom

namespace builtin {

TfLiteRegistration* Register_ABS();
TfLiteRegistration* Register_RELU();
TfLiteRegistration* Register_RELU_N1_TO_1();
TfLiteRegistration* Register_RELU6();
TfLiteRegistration* Register_TANH();
TfLiteRegistration* Register_LOGISTIC();
TfLiteRegistration* Register_AVERAGE_POOL_2D();
TfLiteRegistration* Register_MAX_POOL_2D();
TfLiteRegistration* Register_L2_POOL_2D();
TfLiteRegistration* Register_CONV_2D();
TfLiteRegistration* Register_DEPTHWISE_CONV_2D();
TfLiteRegistration* Register_SVDF();
TfLiteRegistration* Register_RNN();
TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_RNN();
TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_RNN();
TfLiteRegistration* Register_EMBEDDING_LOOKUP();
TfLiteRegistration* Register_EMBEDDING_LOOKUP_SPARSE();
TfLiteRegistration* Register_FULLY_CONNECTED();
TfLiteRegistration* Register_LSH_PROJECTION();
TfLiteRegistration* Register_HASHTABLE_LOOKUP();
TfLiteRegistration* Register_SOFTMAX();
TfLiteRegistration* Register_CONCATENATION();
TfLiteRegistration* Register_ADD();
TfLiteRegistration* Register_SPACE_TO_BATCH_ND();
TfLiteRegistration* Register_DIV();
TfLiteRegistration* Register_SUB();
TfLiteRegistration* Register_BATCH_TO_SPACE_ND();
TfLiteRegistration* Register_MUL();
TfLiteRegistration* Register_L2_NORMALIZATION();
TfLiteRegistration* Register_LOCAL_RESPONSE_NORMALIZATION();
TfLiteRegistration* Register_LSTM();
TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_LSTM();
TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
TfLiteRegistration* Register_PAD();
TfLiteRegistration* Register_PADV2();
TfLiteRegistration* Register_RESHAPE();
TfLiteRegistration* Register_RESIZE_BILINEAR();
TfLiteRegistration* Register_RESIZE_NEAREST_NEIGHBOR();
TfLiteRegistration* Register_SKIP_GRAM();
TfLiteRegistration* Register_SPACE_TO_DEPTH();
TfLiteRegistration* Register_GATHER();
TfLiteRegistration* Register_TRANSPOSE();
TfLiteRegistration* Register_MEAN();
TfLiteRegistration* Register_SPLIT();
TfLiteRegistration* Register_SPLIT_V();
TfLiteRegistration* Register_SQUEEZE();
TfLiteRegistration* Register_STRIDED_SLICE();
TfLiteRegistration* Register_EXP();
TfLiteRegistration* Register_TOPK_V2();
TfLiteRegistration* Register_LOG();
TfLiteRegistration* Register_LOG_SOFTMAX();
TfLiteRegistration* Register_CAST();
TfLiteRegistration* Register_DEQUANTIZE();
TfLiteRegistration* Register_PRELU();
TfLiteRegistration* Register_MAXIMUM();
TfLiteRegistration* Register_MINIMUM();
TfLiteRegistration* Register_ARG_MAX();
TfLiteRegistration* Register_ARG_MIN();
TfLiteRegistration* Register_GREATER();
TfLiteRegistration* Register_GREATER_EQUAL();
TfLiteRegistration* Register_LESS();
TfLiteRegistration* Register_LESS_EQUAL();
TfLiteRegistration* Register_FLOOR();
TfLiteRegistration* Register_CEIL();
TfLiteRegistration* Register_TILE();
TfLiteRegistration* Register_NEG();
TfLiteRegistration* Register_SUM();
TfLiteRegistration* Register_REDUCE_PROD();
TfLiteRegistration* Register_REDUCE_MAX();
TfLiteRegistration* Register_REDUCE_MIN();
TfLiteRegistration* Register_REDUCE_ANY();
TfLiteRegistration* Register_SELECT();
TfLiteRegistration* Register_SLICE();
TfLiteRegistration* Register_SIN();
TfLiteRegistration* Register_COS();
TfLiteRegistration* Register_TRANSPOSE_CONV();
TfLiteRegistration* Register_EXPAND_DIMS();
TfLiteRegistration* Register_SPARSE_TO_DENSE();
TfLiteRegistration* Register_EQUAL();
TfLiteRegistration* Register_NOT_EQUAL();
TfLiteRegistration* Register_SQRT();
TfLiteRegistration* Register_RSQRT();
TfLiteRegistration* Register_SHAPE();
TfLiteRegistration* Register_RANK();
TfLiteRegistration* Register_POW();
TfLiteRegistration* Register_FAKE_QUANT();
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
TfLiteRegistration* Register_LEAKY_RELU();
TfLiteRegistration* Register_SQUARED_DIFFERENCE();
TfLiteRegistration* Register_FILL();
TfLiteRegistration* Register_MIRROR_PAD();
TfLiteRegistration* Register_UNIQUE();
TfLiteRegistration* Register_REVERSE_V2();
TfLiteRegistration* Register_ADD_N();
TfLiteRegistration* Register_GATHER_ND();
TfLiteRegistration* Register_WHERE();
TfLiteRegistration* Register_ELU();
TfLiteRegistration* Register_REVERSE_SEQUENCE();

TfLiteStatus UnsupportedTensorFlowOp(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(
      context,
      "Regular TensorFlow ops are not supported by this interpreter. Make sure "
      "you invoke the Flex delegate before inference.");
  return kTfLiteError;
}

const TfLiteRegistration* BuiltinOpResolver::FindOp(tflite::BuiltinOperator op,
                                                    int version) const {
  return MutableOpResolver::FindOp(op, version);
}

const TfLiteRegistration* BuiltinOpResolver::FindOp(const char* op,
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

BuiltinOpResolver::BuiltinOpResolver() {
  AddBuiltin(BuiltinOperator_ABS, Register_ABS());
  AddBuiltin(BuiltinOperator_RELU, Register_RELU());
  AddBuiltin(BuiltinOperator_RELU_N1_TO_1, Register_RELU_N1_TO_1());
  AddBuiltin(BuiltinOperator_RELU6, Register_RELU6(), /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_TANH, Register_TANH(), /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_LOGISTIC, Register_LOGISTIC(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, Register_AVERAGE_POOL_2D(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_MAX_POOL_2D, Register_MAX_POOL_2D(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_L2_POOL_2D, Register_L2_POOL_2D());
  AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D(),
             /* min_version */ 1,
             /* max_version */ 3);
  AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D(),
             /* min_version */ 1,
             /* max_version */ 3);
  AddBuiltin(BuiltinOperator_SVDF, Register_SVDF(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_RNN, Register_RNN(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN,
             Register_BIDIRECTIONAL_SEQUENCE_RNN(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN,
             Register_UNIDIRECTIONAL_SEQUENCE_RNN(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP, Register_EMBEDDING_LOOKUP(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP_SPARSE,
             Register_EMBEDDING_LOOKUP_SPARSE());
  AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Register_FULLY_CONNECTED(),
             /* min_version */ 1,
             /* max_version */ 4);
  AddBuiltin(BuiltinOperator_LSH_PROJECTION, Register_LSH_PROJECTION());
  AddBuiltin(BuiltinOperator_HASHTABLE_LOOKUP, Register_HASHTABLE_LOOKUP());
  AddBuiltin(BuiltinOperator_SOFTMAX, Register_SOFTMAX(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_CONCATENATION, Register_CONCATENATION(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_ADD, Register_ADD(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_SPACE_TO_BATCH_ND, Register_SPACE_TO_BATCH_ND(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_BATCH_TO_SPACE_ND, Register_BATCH_TO_SPACE_ND(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_MUL, Register_MUL());
  AddBuiltin(BuiltinOperator_L2_NORMALIZATION, Register_L2_NORMALIZATION(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
             Register_LOCAL_RESPONSE_NORMALIZATION());
  AddBuiltin(BuiltinOperator_LSTM, Register_LSTM(), /* min_version */ 1,
             /* max_version */ 3);
  AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM,
             Register_BIDIRECTIONAL_SEQUENCE_LSTM(), /* min_version */ 1,
             /* max_version */ 3);
  AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM,
             Register_UNIDIRECTIONAL_SEQUENCE_LSTM(), /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_PAD, Register_PAD(), /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_PADV2, Register_PADV2(), /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE());
  AddBuiltin(BuiltinOperator_RESIZE_BILINEAR, Register_RESIZE_BILINEAR(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
             Register_RESIZE_NEAREST_NEIGHBOR(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_SKIP_GRAM, Register_SKIP_GRAM());
  AddBuiltin(BuiltinOperator_SPACE_TO_DEPTH, Register_SPACE_TO_DEPTH(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_GATHER, Register_GATHER(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_TRANSPOSE, Register_TRANSPOSE(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_MEAN, Register_MEAN());
  AddBuiltin(BuiltinOperator_DIV, Register_DIV());
  AddBuiltin(BuiltinOperator_SUB, Register_SUB(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_SPLIT, Register_SPLIT(), /* min_version */ 1,
             /* max_version */ 3);
  AddBuiltin(BuiltinOperator_SPLIT_V, Register_SPLIT_V());
  AddBuiltin(BuiltinOperator_SQUEEZE, Register_SQUEEZE());
  AddBuiltin(BuiltinOperator_STRIDED_SLICE, Register_STRIDED_SLICE(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_EXP, Register_EXP());
  AddBuiltin(BuiltinOperator_TOPK_V2, Register_TOPK_V2(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_LOG, Register_LOG());
  AddBuiltin(BuiltinOperator_LOG_SOFTMAX, Register_LOG_SOFTMAX(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_CAST, Register_CAST());
  AddBuiltin(BuiltinOperator_DEQUANTIZE, Register_DEQUANTIZE(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_PRELU, Register_PRELU());
  AddBuiltin(BuiltinOperator_MAXIMUM, Register_MAXIMUM(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_MINIMUM, Register_MINIMUM(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_ARG_MAX, Register_ARG_MAX(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_ARG_MIN, Register_ARG_MIN(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_GREATER, Register_GREATER(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_GREATER_EQUAL, Register_GREATER_EQUAL(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_LESS, Register_LESS(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_LESS_EQUAL, Register_LESS_EQUAL(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_FLOOR, Register_FLOOR());
  AddBuiltin(BuiltinOperator_CEIL, Register_CEIL());
  AddBuiltin(BuiltinOperator_NEG, Register_NEG());
  AddBuiltin(BuiltinOperator_SELECT, Register_SELECT(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_SLICE, Register_SLICE(), /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_SIN, Register_SIN());
  AddBuiltin(BuiltinOperator_COS, Register_COS());
  AddBuiltin(BuiltinOperator_TRANSPOSE_CONV, Register_TRANSPOSE_CONV());
  AddBuiltin(BuiltinOperator_TILE, Register_TILE());
  AddBuiltin(BuiltinOperator_SUM, Register_SUM());
  AddBuiltin(BuiltinOperator_REDUCE_PROD, Register_REDUCE_PROD());
  AddBuiltin(BuiltinOperator_REDUCE_MAX, Register_REDUCE_MAX(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_REDUCE_MIN, Register_REDUCE_MIN(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_REDUCE_ANY, Register_REDUCE_ANY());
  AddBuiltin(BuiltinOperator_EXPAND_DIMS, Register_EXPAND_DIMS());
  AddBuiltin(BuiltinOperator_SPARSE_TO_DENSE, Register_SPARSE_TO_DENSE());
  AddBuiltin(BuiltinOperator_EQUAL, Register_EQUAL(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_NOT_EQUAL, Register_NOT_EQUAL(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_SQRT, Register_SQRT());
  AddBuiltin(BuiltinOperator_RSQRT, Register_RSQRT());
  AddBuiltin(BuiltinOperator_SHAPE, Register_SHAPE());
  AddBuiltin(BuiltinOperator_RANK, Register_RANK());
  AddBuiltin(BuiltinOperator_POW, Register_POW());
  AddBuiltin(BuiltinOperator_FAKE_QUANT, Register_FAKE_QUANT(), 1, 2);
  AddBuiltin(BuiltinOperator_PACK, Register_PACK(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_ONE_HOT, Register_ONE_HOT());
  AddBuiltin(BuiltinOperator_LOGICAL_OR, Register_LOGICAL_OR());
  AddBuiltin(BuiltinOperator_LOGICAL_AND, Register_LOGICAL_AND());
  AddBuiltin(BuiltinOperator_LOGICAL_NOT, Register_LOGICAL_NOT());
  AddBuiltin(BuiltinOperator_UNPACK, Register_UNPACK());
  AddBuiltin(BuiltinOperator_FLOOR_DIV, Register_FLOOR_DIV());
  AddBuiltin(BuiltinOperator_SQUARE, Register_SQUARE());
  AddBuiltin(BuiltinOperator_ZEROS_LIKE, Register_ZEROS_LIKE());
  AddBuiltin(BuiltinOperator_FLOOR_MOD, Register_FLOOR_MOD());
  AddBuiltin(BuiltinOperator_RANGE, Register_RANGE());
  AddBuiltin(BuiltinOperator_LEAKY_RELU, Register_LEAKY_RELU());
  AddBuiltin(BuiltinOperator_SQUARED_DIFFERENCE, Register_SQUARED_DIFFERENCE());
  AddBuiltin(BuiltinOperator_FILL, Register_FILL());
  AddBuiltin(BuiltinOperator_MIRROR_PAD, Register_MIRROR_PAD());
  AddBuiltin(BuiltinOperator_UNIQUE, Register_UNIQUE());
  AddBuiltin(BuiltinOperator_REVERSE_V2, Register_REVERSE_V2());
  AddBuiltin(BuiltinOperator_ADD_N, Register_ADD_N());
  AddBuiltin(BuiltinOperator_GATHER_ND, Register_GATHER_ND());
  AddBuiltin(BuiltinOperator_WHERE, Register_WHERE());
  AddBuiltin(BuiltinOperator_ELU, Register_ELU());
  AddBuiltin(BuiltinOperator_REVERSE_SEQUENCE, Register_REVERSE_SEQUENCE());

  // TODO(andrewharp, ahentz): Move these somewhere more appropriate so that
  // custom ops aren't always included by default.
  AddCustom("Mfcc", tflite::ops::custom::Register_MFCC());
  AddCustom("AudioSpectrogram",
            tflite::ops::custom::Register_AUDIO_SPECTROGRAM());
  AddCustom("TFLite_Detection_PostProcess",
            tflite::ops::custom::Register_DETECTION_POSTPROCESS());

  // WARNING: Control flow ops are experimental and subject to change.
  AddCustom("Experimental_If", tflite::ops::custom::Register_IF());
  AddCustom("Experimental_While", tflite::ops::custom::Register_WHILE());
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
