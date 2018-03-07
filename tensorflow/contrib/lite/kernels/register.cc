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

#include "tensorflow/contrib/lite/kernels/register.h"

namespace tflite {
namespace ops {
namespace builtin {

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
TfLiteRegistration* Register_RESHAPE();
TfLiteRegistration* Register_RESIZE_BILINEAR();
TfLiteRegistration* Register_SKIP_GRAM();
TfLiteRegistration* Register_SPACE_TO_DEPTH();
TfLiteRegistration* Register_GATHER();
TfLiteRegistration* Register_TRANSPOSE();
TfLiteRegistration* Register_MEAN();
TfLiteRegistration* Register_SPLIT();
TfLiteRegistration* Register_SQUEEZE();
TfLiteRegistration* Register_STRIDED_SLICE();
TfLiteRegistration* Register_EXP();
TfLiteRegistration* Register_TOPK_V2();
TfLiteRegistration* Register_LOG_SOFTMAX();
TfLiteRegistration* Register_CAST();
TfLiteRegistration* Register_DEQUANTIZE();

BuiltinOpResolver::BuiltinOpResolver() {
  AddBuiltin(BuiltinOperator_RELU, Register_RELU());
  AddBuiltin(BuiltinOperator_RELU_N1_TO_1, Register_RELU_N1_TO_1());
  AddBuiltin(BuiltinOperator_RELU6, Register_RELU6());
  AddBuiltin(BuiltinOperator_TANH, Register_TANH());
  AddBuiltin(BuiltinOperator_LOGISTIC, Register_LOGISTIC());
  AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, Register_AVERAGE_POOL_2D());
  AddBuiltin(BuiltinOperator_MAX_POOL_2D, Register_MAX_POOL_2D());
  AddBuiltin(BuiltinOperator_L2_POOL_2D, Register_L2_POOL_2D());
  AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D());
  AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D());
  AddBuiltin(BuiltinOperator_SVDF, Register_SVDF());
  AddBuiltin(BuiltinOperator_RNN, Register_RNN());
  AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN,
             Register_BIDIRECTIONAL_SEQUENCE_RNN());
  AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN,
             Register_UNIDIRECTIONAL_SEQUENCE_RNN());
  AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP, Register_EMBEDDING_LOOKUP());
  AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP_SPARSE,
             Register_EMBEDDING_LOOKUP_SPARSE());
  AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Register_FULLY_CONNECTED());
  AddBuiltin(BuiltinOperator_LSH_PROJECTION, Register_LSH_PROJECTION());
  AddBuiltin(BuiltinOperator_HASHTABLE_LOOKUP, Register_HASHTABLE_LOOKUP());
  AddBuiltin(BuiltinOperator_SOFTMAX, Register_SOFTMAX());
  AddBuiltin(BuiltinOperator_CONCATENATION, Register_CONCATENATION());
  AddBuiltin(BuiltinOperator_ADD, Register_ADD());
  AddBuiltin(BuiltinOperator_SPACE_TO_BATCH_ND, Register_SPACE_TO_BATCH_ND());
  AddBuiltin(BuiltinOperator_BATCH_TO_SPACE_ND, Register_BATCH_TO_SPACE_ND());
  AddBuiltin(BuiltinOperator_MUL, Register_MUL());
  AddBuiltin(BuiltinOperator_L2_NORMALIZATION, Register_L2_NORMALIZATION());
  AddBuiltin(BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
             Register_LOCAL_RESPONSE_NORMALIZATION());
  AddBuiltin(BuiltinOperator_LSTM, Register_LSTM());
  AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM,
             Register_BIDIRECTIONAL_SEQUENCE_LSTM());
  AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM,
             Register_UNIDIRECTIONAL_SEQUENCE_LSTM());
  AddBuiltin(BuiltinOperator_PAD, Register_PAD());
  AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE());
  AddBuiltin(BuiltinOperator_RESIZE_BILINEAR, Register_RESIZE_BILINEAR());
  AddBuiltin(BuiltinOperator_SKIP_GRAM, Register_SKIP_GRAM());
  AddBuiltin(BuiltinOperator_SPACE_TO_DEPTH, Register_SPACE_TO_DEPTH());
  AddBuiltin(BuiltinOperator_GATHER, Register_GATHER());
  AddBuiltin(BuiltinOperator_TRANSPOSE, Register_TRANSPOSE());
  AddBuiltin(BuiltinOperator_MEAN, Register_MEAN());
  AddBuiltin(BuiltinOperator_DIV, Register_DIV());
  AddBuiltin(BuiltinOperator_SUB, Register_SUB());
  AddBuiltin(BuiltinOperator_SPLIT, Register_SPLIT());
  AddBuiltin(BuiltinOperator_SQUEEZE, Register_SQUEEZE());
  AddBuiltin(BuiltinOperator_STRIDED_SLICE, Register_STRIDED_SLICE());
  AddBuiltin(BuiltinOperator_EXP, Register_EXP());
  AddBuiltin(BuiltinOperator_TOPK_V2, Register_TOPK_V2());
  AddBuiltin(BuiltinOperator_LOG_SOFTMAX, Register_LOG_SOFTMAX());
  AddBuiltin(BuiltinOperator_CAST, Register_CAST());
  AddBuiltin(BuiltinOperator_DEQUANTIZE, Register_DEQUANTIZE());
}

TfLiteRegistration* BuiltinOpResolver::FindOp(
    tflite::BuiltinOperator op) const {
  auto it = builtins_.find(op);
  return it != builtins_.end() ? it->second : nullptr;
}

TfLiteRegistration* BuiltinOpResolver::FindOp(const char* op) const {
  auto it = custom_ops_.find(op);
  return it != custom_ops_.end() ? it->second : nullptr;
}

void BuiltinOpResolver::AddBuiltin(tflite::BuiltinOperator op,
                                   TfLiteRegistration* registration) {
  registration->builtin_code = op;
  builtins_.insert(std::make_pair(op, registration));
}

void BuiltinOpResolver::AddCustom(const char* name,
                                  TfLiteRegistration* registration) {
  registration->builtin_code = BuiltinOperator_CUSTOM;
  custom_ops_.insert(std::make_pair(std::string(name), registration));
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
