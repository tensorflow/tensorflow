/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/versioning/runtime_version.h"

#include <cstring>
#include <map>
#include <string>
#include <utility>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/schema/mutable/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace tflite {

bool CompareRuntimeVersion(const std::string& v1, const std::string& v2) {
  const std::vector<std::string> vec1 = absl::StrSplit(v1, '.');
  const std::vector<std::string> vec2 = absl::StrSplit(v2, '.');
  int i = 0;
  while (i < vec1.size() && i < vec2.size()) {
    int v1_val, v2_val;
    if (absl::SimpleAtoi(vec1[i], &v1_val) &&
        absl::SimpleAtoi(vec2[i], &v2_val)) {
      if (v1_val != v2_val) return v1_val < v2_val;
    }
    ++i;
  }
  // If there are remaining items in v2 not being compared, then v1 should
  // precede v2.
  return i < vec2.size();
}

std::string FindMinimumRuntimeVersionForOp(tflite::BuiltinOperator op_code,
                                           int op_version) {
  // A map from the version key of an op to its minimum runtime version.
  // For example, {{kAveragePool, 1}, "1.5.0"},  means the 1st version of
  // AveragePool requires a minimum TF Lite runtime version '1.5.0`.
  // NOTE: When adding a new op version pair, associate it with the current
  // runtime version defined in tensorflow/core/public/version.h.
  static const std::map<std::pair<BuiltinOperator, int>,
                        std::string>* op_version_map =
      new std::map<std::pair<BuiltinOperator, int>, std::string>(
          {{{BuiltinOperator_AVERAGE_POOL_2D, 1}, "1.5.0"},
           {{BuiltinOperator_AVERAGE_POOL_2D, 2}, "1.14.0"},
           {{BuiltinOperator_AVERAGE_POOL_2D, 3}, "2.3.0"},
           {{BuiltinOperator_BATCH_MATMUL, 1}, "2.3.0"},
           {{BuiltinOperator_BATCH_MATMUL, 2}, "2.3.0"},
           {{BuiltinOperator_BATCH_MATMUL, 3}, "2.4.0"},
           {{BuiltinOperator_BATCH_MATMUL, 4}, "2.5.0"},
           // The version one of broadcast to op won't be not supported since
           // the version one was rollbacked and the builtin op code number
           // has been changed because of builtin op code shortage problem.
           {{BuiltinOperator_BROADCAST_TO, 2}, "2.5.0"},
           {{BuiltinOperator_BROADCAST_TO, 3}, "2.5.0"},
           {{BuiltinOperator_CONV_2D, 1}, "1.5.0"},
           {{BuiltinOperator_CONV_2D, 2}, "1.14.0"},
           {{BuiltinOperator_CONV_2D, 3}, "1.14.0"},
           {{BuiltinOperator_CONV_2D, 4}, "2.3.0"},
           {{BuiltinOperator_CONV_2D, 5}, "2.4.0"},
           {{BuiltinOperator_CONV_2D, 6}, "2.9.0"},
           {{BuiltinOperator_CONV_2D, 7}, "2.11.0"},
           {{BuiltinOperator_DEPTHWISE_CONV_2D, 1}, "1.5.0"},
           {{BuiltinOperator_DEPTHWISE_CONV_2D, 2}, "1.12.0"},
           {{BuiltinOperator_DEPTHWISE_CONV_2D, 3}, "1.14.0"},
           {{BuiltinOperator_DEPTHWISE_CONV_2D, 4}, "2.2.0"},
           {{BuiltinOperator_DEPTHWISE_CONV_2D, 5}, "2.3.0"},
           {{BuiltinOperator_DEPTHWISE_CONV_2D, 6}, "2.3.0"},
           {{BuiltinOperator_DEPTHWISE_CONV_2D, 7}, "2.11.0"},
           {{BuiltinOperator_ADD, 1}, "1.5.0"},
           {{BuiltinOperator_ADD, 2}, "1.14.0"},
           {{BuiltinOperator_ADD, 3}, "2.4.0"},
           {{BuiltinOperator_ADD, 4}, "2.6.0"},
           {{BuiltinOperator_ADD_N, 1}, "1.14.0"},
           {{BuiltinOperator_SPACE_TO_BATCH_ND, 1}, "1.6.0"},
           {{BuiltinOperator_SPACE_TO_BATCH_ND, 2}, "1.14.0"},
           {{BuiltinOperator_SPACE_TO_BATCH_ND, 3}, "2.3.0"},
           {{BuiltinOperator_SUB, 1}, "1.6.0"},
           {{BuiltinOperator_SUB, 2}, "1.14.0"},
           {{BuiltinOperator_SUB, 3}, "2.3.0"},
           {{BuiltinOperator_SUB, 4}, "2.4.0"},
           {{BuiltinOperator_SUB, 5}, "2.4.0"},
           {{BuiltinOperator_DENSIFY, 1}, "2.2.0"},
           {{BuiltinOperator_DIV, 1}, "1.6.0"},
           {{BuiltinOperator_DIV, 2}, "2.3.0"},
           {{BuiltinOperator_BATCH_TO_SPACE_ND, 1}, "1.6.0"},
           {{BuiltinOperator_BATCH_TO_SPACE_ND, 2}, "1.14.0"},
           {{BuiltinOperator_BATCH_TO_SPACE_ND, 3}, "2.3.0"},
           {{BuiltinOperator_CAST, 1}, "1.5.0"},
           {{BuiltinOperator_CAST, 2}, "2.7.0"},
           {{BuiltinOperator_CAST, 3}, "2.8.0"},
           {{BuiltinOperator_CAST, 4}, "2.9.0"},
           {{BuiltinOperator_CAST, 5}, "2.12.0"},
           {{BuiltinOperator_CONCATENATION, 1}, "1.5.0"},
           {{BuiltinOperator_CONCATENATION, 2}, "1.14.0"},
           {{BuiltinOperator_CONCATENATION, 3}, "2.3.0"},
           {{BuiltinOperator_DEPTH_TO_SPACE, 1}, "2.1.0"},
           {{BuiltinOperator_DEPTH_TO_SPACE, 2}, "2.5.0"},
           {{BuiltinOperator_EMBEDDING_LOOKUP, 1}, "1.13.0"},
           {{BuiltinOperator_EMBEDDING_LOOKUP, 2}, "1.14.0"},
           {{BuiltinOperator_EMBEDDING_LOOKUP, 3}, "1.14.0"},
           {{BuiltinOperator_EMBEDDING_LOOKUP_SPARSE, 1}, "1.5.0"},
           {{BuiltinOperator_FAKE_QUANT, 1}, "1.5.0"},
           {{BuiltinOperator_FAKE_QUANT, 2}, "1.10.0"},
           {{BuiltinOperator_FULLY_CONNECTED, 1}, "1.5.0"},
           {{BuiltinOperator_FULLY_CONNECTED, 2}, "1.10.0"},
           {{BuiltinOperator_FULLY_CONNECTED, 3}, "1.14.0"},
           {{BuiltinOperator_FULLY_CONNECTED, 4}, "1.14.0"},
           {{BuiltinOperator_FULLY_CONNECTED, 5}, "2.0.0"},
           {{BuiltinOperator_FULLY_CONNECTED, 6}, "2.1.0"},
           {{BuiltinOperator_FULLY_CONNECTED, 7}, "2.3.0"},
           {{BuiltinOperator_FULLY_CONNECTED, 8}, "2.3.0"},
           {{BuiltinOperator_FULLY_CONNECTED, 9}, "2.3.0"},
           {{BuiltinOperator_FULLY_CONNECTED, 10}, "2.11.0"},
           {{BuiltinOperator_GATHER, 1}, "1.6.0"},
           {{BuiltinOperator_GATHER, 2}, "1.14.0"},
           {{BuiltinOperator_GATHER, 3}, "1.15.0"},
           {{BuiltinOperator_GATHER, 4}, "2.4.0"},
           {{BuiltinOperator_GATHER, 5}, "2.5.0"},
           {{BuiltinOperator_GATHER_ND, 1}, "1.14.0"},
           {{BuiltinOperator_GATHER_ND, 2}, "2.3.0"},
           {{BuiltinOperator_GATHER_ND, 3}, "2.5.0"},
           {{BuiltinOperator_HASHTABLE_LOOKUP, 1}, "1.5.0"},
           {{BuiltinOperator_SVDF, 1}, "1.5.0"},
           {{BuiltinOperator_SVDF, 2}, "1.14.0"},
           {{BuiltinOperator_SVDF, 3}, "2.2.0"},
           {{BuiltinOperator_SVDF, 4}, "2.3.0"},
           {{BuiltinOperator_L2_NORMALIZATION, 1}, "1.5.0"},
           {{BuiltinOperator_L2_NORMALIZATION, 2}, "1.14.0"},
           {{BuiltinOperator_L2_POOL_2D, 1}, "1.5.0"},
           {{BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION, 1}, "1.5.0"},
           {{BuiltinOperator_MAX_POOL_2D, 1}, "1.5.0"},
           {{BuiltinOperator_MAX_POOL_2D, 2}, "1.14.0"},
           {{BuiltinOperator_MAX_POOL_2D, 3}, "2.3.0"},
           {{BuiltinOperator_MAXIMUM, 1}, "1.14.0"},
           {{BuiltinOperator_MAXIMUM, 2}, "1.14.0"},
           {{BuiltinOperator_MAXIMUM, 3}, "2.3.0"},
           {{BuiltinOperator_MAXIMUM, 4}, "2.3.0"},
           {{BuiltinOperator_MINIMUM, 1}, "1.14.0"},
           {{BuiltinOperator_MINIMUM, 2}, "1.14.0"},
           {{BuiltinOperator_MINIMUM, 3}, "2.3.0"},
           {{BuiltinOperator_MINIMUM, 4}, "2.3.0"},
           {{BuiltinOperator_MUL, 1}, "1.5.0"},
           {{BuiltinOperator_MUL, 2}, "1.14.0"},
           {{BuiltinOperator_MUL, 3}, "1.15.0"},
           {{BuiltinOperator_MUL, 4}, "2.3.0"},
           {{BuiltinOperator_MUL, 5}, "2.6.0"},
           {{BuiltinOperator_MUL, 6}, "2.11.0"},
           {{BuiltinOperator_NON_MAX_SUPPRESSION_V4, 1}, "2.1.0"},
           {{BuiltinOperator_NON_MAX_SUPPRESSION_V5, 1}, "2.1.0"},
           {{BuiltinOperator_PAD, 1}, "1.5.0"},
           {{BuiltinOperator_PAD, 2}, "1.14.0"},
           {{BuiltinOperator_PAD, 3}, "2.4.0"},
           {{BuiltinOperator_PAD, 4}, "2.6.0"},
           {{BuiltinOperator_TILE, 1}, "1.10.1"},
           {{BuiltinOperator_TILE, 2}, "2.2.0"},
           {{BuiltinOperator_TILE, 3}, "2.8.0"},
           {{BuiltinOperator_TILE, 4}, "2.12.0"},
           {{BuiltinOperator_PADV2, 1}, "1.9.0"},
           {{BuiltinOperator_PADV2, 2}, "1.14.0"},
           {{BuiltinOperator_PADV2, 3}, "2.4.0"},
           {{BuiltinOperator_PADV2, 4}, "2.6.0"},
           {{BuiltinOperator_RESHAPE, 1}, "1.5.0"},
           {{BuiltinOperator_SOFTMAX, 1}, "1.5.0"},
           {{BuiltinOperator_SOFTMAX, 2}, "1.14.0"},
           {{BuiltinOperator_SOFTMAX, 3}, "2.3.0"},
           {{BuiltinOperator_SPACE_TO_DEPTH, 1}, "1.5.0"},
           {{BuiltinOperator_SPACE_TO_DEPTH, 2}, "1.14.0"},
           {{BuiltinOperator_TRANSPOSE, 1}, "1.6.0"},
           {{BuiltinOperator_TRANSPOSE, 2}, "1.14.0"},
           {{BuiltinOperator_TRANSPOSE, 3}, "1.15.0"},
           {{BuiltinOperator_TRANSPOSE, 4}, "2.3.0"},
           {{BuiltinOperator_TRANSPOSE, 5}, "2.4.0"},
           {{BuiltinOperator_TRANSPOSE, 6}, "2.12.0"},
           {{BuiltinOperator_LSTM, 1}, "1.7.0"},
           {{BuiltinOperator_LSTM, 2}, "1.10.0"},
           {{BuiltinOperator_LSTM, 3}, "1.14.0"},
           {{BuiltinOperator_LSTM, 4}, "2.3.0"},
           {{BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM, 1}, "1.13.1"},
           {{BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM, 2}, "1.14.0"},
           {{BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM, 3}, "2.3.0"},
           {{BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM, 4}, "2.12.0"},
           {{BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM, 1}, "1.14.0"},
           {{BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM, 2}, "1.14.0"},
           {{BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM, 3}, "1.14.0"},
           {{BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN, 1}, "1.14.0"},
           {{BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN, 2}, "1.14.0"},
           {{BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN, 3}, "2.3.0"},
           {{BuiltinOperator_MEAN, 1}, "1.6.0"},
           {{BuiltinOperator_MEAN, 2}, "1.14.0"},
           {{BuiltinOperator_MEAN, 3}, "2.4.0"},
           {{BuiltinOperator_SUM, 1}, "1.10.0"},
           {{BuiltinOperator_SUM, 2}, "1.15.0"},
           {{BuiltinOperator_REDUCE_MAX, 1}, "1.11.0"},
           {{BuiltinOperator_REDUCE_MAX, 2}, "1.14.0"},
           {{BuiltinOperator_REDUCE_MAX, 3}, "2.5.0"},
           {{BuiltinOperator_REDUCE_MIN, 1}, "1.11.0"},
           {{BuiltinOperator_REDUCE_MIN, 2}, "1.14.0"},
           {{BuiltinOperator_REDUCE_MIN, 3}, "2.5.0"},
           {{BuiltinOperator_REDUCE_PROD, 1}, "1.11.0"},
           {{BuiltinOperator_REDUCE_PROD, 2}, "2.6.0"},
           {{BuiltinOperator_REDUCE_ANY, 1}, "1.11.0"},
           {{BuiltinOperator_RELU6, 1}, "1.5.0"},
           {{BuiltinOperator_RELU6, 2}, "1.14.0"},
           {{BuiltinOperator_RELU6, 3}, "2.5.0"},
           {{BuiltinOperator_RESIZE_BILINEAR, 1}, "1.7.0"},
           {{BuiltinOperator_RESIZE_BILINEAR, 2}, "1.14.0"},
           {{BuiltinOperator_RESIZE_BILINEAR, 3}, "2.2.0"},
           {{BuiltinOperator_RESIZE_BILINEAR, 4}, "2.5.0"},
           {{BuiltinOperator_RESIZE_NEAREST_NEIGHBOR, 1}, "1.13.1"},
           {{BuiltinOperator_RESIZE_NEAREST_NEIGHBOR, 2}, "1.14.0"},
           {{BuiltinOperator_RESIZE_NEAREST_NEIGHBOR, 3}, "2.3.0"},
           {{BuiltinOperator_RESIZE_NEAREST_NEIGHBOR, 4}, "2.4.0"},
           {{BuiltinOperator_RNN, 1}, "1.5.0"},
           {{BuiltinOperator_RNN, 2}, "1.14.0"},
           {{BuiltinOperator_RNN, 3}, "2.3.0"},
           {{BuiltinOperator_SKIP_GRAM, 1}, "1.5.0"},
           {{BuiltinOperator_SQUEEZE, 1}, "1.6.0"},
           {{BuiltinOperator_SQUEEZE, 2}, "2.5.0"},
           {{BuiltinOperator_SPLIT, 1}, "1.5.0"},
           {{BuiltinOperator_SPLIT, 2}, "1.14.0"},
           {{BuiltinOperator_SPLIT, 3}, "1.14.0"},
           {{BuiltinOperator_SPLIT, 4}, "2.3.0"},
           {{BuiltinOperator_SPLIT_V, 1}, "1.13.1"},
           {{BuiltinOperator_SPLIT_V, 2}, "2.3.0"},
           {{BuiltinOperator_STRIDED_SLICE, 1}, "1.6.0"},
           {{BuiltinOperator_STRIDED_SLICE, 2}, "1.14.0"},
           {{BuiltinOperator_STRIDED_SLICE, 3}, "2.1.0"},
           {{BuiltinOperator_STRIDED_SLICE, 4}, "2.2.0"},
           {{BuiltinOperator_STRIDED_SLICE, 5}, "2.5.0"},
           {{BuiltinOperator_STRIDED_SLICE, 6}, "2.6.0"},
           {{BuiltinOperator_TOPK_V2, 1}, "1.7.0"},
           {{BuiltinOperator_TOPK_V2, 2}, "1.14.0"},
           {{BuiltinOperator_ARG_MAX, 1}, "1.9.0"},
           {{BuiltinOperator_ARG_MAX, 2}, "1.14.0"},
           {{BuiltinOperator_ARG_MAX, 3}, "2.9.0"},
           {{BuiltinOperator_ARG_MIN, 1}, "1.9.0"},
           {{BuiltinOperator_ARG_MIN, 2}, "1.14.0"},
           {{BuiltinOperator_ARG_MIN, 3}, "2.9.0"},
           {{BuiltinOperator_TRANSPOSE_CONV, 1}, "1.9.0"},
           {{BuiltinOperator_TRANSPOSE_CONV, 2}, "2.2.0"},
           {{BuiltinOperator_TRANSPOSE_CONV, 3}, "2.3.0"},
           {{BuiltinOperator_TRANSPOSE_CONV, 4}, "2.13.0"},
           {{BuiltinOperator_SPARSE_TO_DENSE, 1}, "1.9.0"},
           {{BuiltinOperator_SPARSE_TO_DENSE, 2}, "1.14.0"},
           {{BuiltinOperator_SPARSE_TO_DENSE, 3}, "1.15.0"},
           {{BuiltinOperator_EXPAND_DIMS, 1}, "1.10.0"},
           {{BuiltinOperator_PACK, 1}, "1.11.0"},
           {{BuiltinOperator_PACK, 2}, "1.14.0"},
           {{BuiltinOperator_PACK, 3}, "2.3.0"},
           {{BuiltinOperator_SHAPE, 1}, "1.10.0"},
           {{BuiltinOperator_SLICE, 1}, "1.14.0"},
           {{BuiltinOperator_SLICE, 2}, "1.14.0"},
           {{BuiltinOperator_SLICE, 3}, "1.14.0"},
           {{BuiltinOperator_SLICE, 4}, "2.4.0"},
           {{BuiltinOperator_SLICE, 5}, "2.5.0"},
           {{BuiltinOperator_TANH, 1}, "1.14.0"},
           {{BuiltinOperator_TANH, 2}, "1.14.0"},
           {{BuiltinOperator_TANH, 3}, "2.3.0"},
           {{BuiltinOperator_ONE_HOT, 1}, "1.11.0"},
           {{BuiltinOperator_UNPACK, 1}, "1.11.0"},
           {{BuiltinOperator_UNPACK, 2}, "1.14.0"},
           {{BuiltinOperator_UNPACK, 3}, "2.2.0"},
           {{BuiltinOperator_UNPACK, 4}, "2.3.0"},
           {{BuiltinOperator_LEAKY_RELU, 1}, "1.13.1"},
           {{BuiltinOperator_LEAKY_RELU, 2}, "2.3.0"},
           {{BuiltinOperator_LOGISTIC, 1}, "1.14.0"},
           {{BuiltinOperator_LOGISTIC, 2}, "1.14.0"},
           {{BuiltinOperator_LOGISTIC, 3}, "2.3.0"},
           {{BuiltinOperator_LOG_SOFTMAX, 1}, "1.14.0"},
           {{BuiltinOperator_LOG_SOFTMAX, 2}, "1.14.0"},
           {{BuiltinOperator_LSH_PROJECTION, 1}, "1.5.0"},
           {{BuiltinOperator_SQUARED_DIFFERENCE, 1}, "1.13.1"},
           {{BuiltinOperator_SQUARED_DIFFERENCE, 2}, "2.5.0"},
           {{BuiltinOperator_MIRROR_PAD, 1}, "1.13.1"},
           {{BuiltinOperator_MIRROR_PAD, 2}, "2.3.0"},
           {{BuiltinOperator_MIRROR_PAD, 3}, "2.12.0"},
           {{BuiltinOperator_UNIQUE, 1}, "1.14.0"},
           {{BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN, 1}, "1.14.0"},
           {{BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN, 2}, "1.14.0"},
           {{BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN, 3}, "2.3.0"},
           {{BuiltinOperator_WHERE, 1}, "1.14.0"},
           {{BuiltinOperator_DEQUANTIZE, 1}, "1.13.1"},
           {{BuiltinOperator_DEQUANTIZE, 2}, "1.14.0"},
           {{BuiltinOperator_DEQUANTIZE, 3}, "1.15.0"},
           {{BuiltinOperator_DEQUANTIZE, 4}, "2.2.0"},
           {{BuiltinOperator_DEQUANTIZE, 5}, "2.7.0"},
           {{BuiltinOperator_REVERSE_SEQUENCE, 1}, "1.14.0"},
           {{BuiltinOperator_EQUAL, 1}, "1.14.0"},
           {{BuiltinOperator_EQUAL, 2}, "1.14.0"},
           {{BuiltinOperator_EQUAL, 3}, "2.3.0"},
           {{BuiltinOperator_NOT_EQUAL, 1}, "1.14.0"},
           {{BuiltinOperator_NOT_EQUAL, 2}, "1.14.0"},
           {{BuiltinOperator_NOT_EQUAL, 3}, "2.3.0"},
           {{BuiltinOperator_GREATER, 1}, "1.14.0"},
           {{BuiltinOperator_GREATER, 2}, "1.14.0"},
           {{BuiltinOperator_GREATER_EQUAL, 1}, "1.14.0"},
           {{BuiltinOperator_GREATER_EQUAL, 2}, "1.14.0"},
           {{BuiltinOperator_LESS, 1}, "1.14.0"},
           {{BuiltinOperator_LESS, 2}, "1.14.0"},
           {{BuiltinOperator_LESS_EQUAL, 1}, "1.14.0"},
           {{BuiltinOperator_LESS_EQUAL, 2}, "1.14.0"},
           {{BuiltinOperator_SCATTER_ND, 1}, "2.1.0"},
           {{BuiltinOperator_SEGMENT_SUM, 1}, "2.2.0"},
           {{BuiltinOperator_SELECT, 1}, "1.14.0"},
           {{BuiltinOperator_SELECT, 2}, "1.14.0"},
           {{BuiltinOperator_SELECT, 3}, "2.12.0"},
           {{BuiltinOperator_SELECT_V2, 1}, "2.2.0"},
           {{BuiltinOperator_IF, 1}, "1.15.0"},
           {{BuiltinOperator_FLOOR_DIV, 1}, "1.14.0"},
           {{BuiltinOperator_FLOOR_DIV, 2}, "1.14.0"},
           {{BuiltinOperator_FLOOR, 1}, "1.9.0"},
           {{BuiltinOperator_CEIL, 1}, "1.14.0"},
           {{BuiltinOperator_MATRIX_DIAG, 1}, "1.14.0"},
           {{BuiltinOperator_MATRIX_SET_DIAG, 1}, "1.14.0"},
           {{BuiltinOperator_ELU, 1}, "1.14.0"},
           {{BuiltinOperator_QUANTIZE, 1}, "1.14.0"},
           {{BuiltinOperator_QUANTIZE, 2}, "1.15.0"},
           {{BuiltinOperator_QUANTIZE, 3}, "2.7.0"},
           {{BuiltinOperator_ROUND, 1}, "1.14.0"},
           {{BuiltinOperator_RELU, 1}, "1.5.0"},
           {{BuiltinOperator_RELU, 2}, "2.1.0"},
           {{BuiltinOperator_RELU, 3}, "2.5.0"},
           {{BuiltinOperator_RELU_N1_TO_1, 1}, "1.5.0"},
           {{BuiltinOperator_RELU_0_TO_1, 1}, "2.10.0"},
           {{BuiltinOperator_PRELU, 1}, "1.8.0"},
           {{BuiltinOperator_EXP, 1}, "1.7.0"},
           {{BuiltinOperator_COS, 1}, "1.14.0"},
           {{BuiltinOperator_NEG, 1}, "1.9.0"},
           {{BuiltinOperator_POW, 1}, "1.10.0"},
           {{BuiltinOperator_LOGICAL_OR, 1}, "1.11.0"},
           {{BuiltinOperator_LOGICAL_AND, 1}, "1.11.0"},
           {{BuiltinOperator_LOGICAL_NOT, 1}, "1.11.0"},
           {{BuiltinOperator_FLOOR_MOD, 1}, "1.13.0"},
           {{BuiltinOperator_RANGE, 1}, "1.13.0"},
           {{BuiltinOperator_SIN, 1}, "1.9.0"},
           {{BuiltinOperator_LOG, 1}, "1.14.0"},
           {{BuiltinOperator_SQRT, 1}, "1.10.0"},
           {{BuiltinOperator_RSQRT, 1}, "1.10.0"},
           {{BuiltinOperator_RSQRT, 2}, "2.5.0"},
           {{BuiltinOperator_SQUARE, 1}, "1.12.0"},
           {{BuiltinOperator_ZEROS_LIKE, 1}, "1.12.0"},
           {{BuiltinOperator_ABS, 1}, "1.13.0"},
           {{BuiltinOperator_ABS, 2}, "2.4.0"},
           {{BuiltinOperator_ABS, 3}, "2.5.0"},
           {{BuiltinOperator_ABS, 4}, "2.6.0"},
           {{BuiltinOperator_ABS, 5}, "2.12.0"},
           {{BuiltinOperator_HARD_SWISH, 1}, "1.15.0"},
           {{BuiltinOperator_FILL, 1}, "1.13.0"},
           {{BuiltinOperator_FILL, 2}, "2.3.0"},
           {{BuiltinOperator_FILL, 3}, "2.5.0"},
           {{BuiltinOperator_FILL, 4}, "2.12.0"},
           {{BuiltinOperator_REVERSE_V2, 1}, "1.14.0"},
           {{BuiltinOperator_REVERSE_V2, 2}, "2.2.0"},
           {{BuiltinOperator_REVERSE_V2, 3}, "2.5.0"},
           {{BuiltinOperator_RANK, 1}, "1.14.0"},
           {{BuiltinOperator_WHILE, 1}, "1.15.0"},
           {{BuiltinOperator_CUMSUM, 1}, "2.4.0"},
           {{BuiltinOperator_CALL_ONCE, 1}, "2.5.0"},
           {{BuiltinOperator_RFFT2D, 1}, "2.5.0"},
           {{BuiltinOperator_CONV_3D, 1}, "2.5.0"},
           {{BuiltinOperator_IMAG, 1}, "2.5.0"},
           {{BuiltinOperator_REAL, 1}, "2.5.0"},
           {{BuiltinOperator_COMPLEX_ABS, 1}, "2.5.0"},
           {{BuiltinOperator_HASHTABLE, 1}, "2.5.0"},
           {{BuiltinOperator_HASHTABLE_FIND, 1}, "2.5.0"},
           {{BuiltinOperator_HASHTABLE_IMPORT, 1}, "2.5.0"},
           {{BuiltinOperator_HASHTABLE_SIZE, 1}, "2.5.0"},
           {{BuiltinOperator_REDUCE_ALL, 1}, "2.6.0"},
           {{BuiltinOperator_CONV_3D_TRANSPOSE, 1}, "2.6.0"},
           {{BuiltinOperator_VAR_HANDLE, 1}, "2.6.0"},
           {{BuiltinOperator_READ_VARIABLE, 1}, "2.6.0"},
           {{BuiltinOperator_ASSIGN_VARIABLE, 1}, "2.6.0"},
           {{BuiltinOperator_BROADCAST_ARGS, 1}, "2.6.0"},
           {{BuiltinOperator_RANDOM_STANDARD_NORMAL, 1}, "2.8.0"},
           {{BuiltinOperator_BUCKETIZE, 1}, "2.8.0"},
           {{BuiltinOperator_WHERE, 2}, "2.8.0"},
           {{BuiltinOperator_RANDOM_UNIFORM, 1}, "2.8.0"},
           {{BuiltinOperator_MULTINOMIAL, 1}, "2.8.0"},
           {{BuiltinOperator_GELU, 1}, "2.9.0"},
           {{BuiltinOperator_GELU, 2}, "2.9.0"},
           {{BuiltinOperator_DYNAMIC_UPDATE_SLICE, 1}, "2.9.0"},
           {{BuiltinOperator_UNSORTED_SEGMENT_PROD, 1}, "2.10.0"},
           {{BuiltinOperator_UNSORTED_SEGMENT_MAX, 1}, "2.10.0"},
           {{BuiltinOperator_UNSORTED_SEGMENT_MIN, 1}, "2.11.0"},
           {{BuiltinOperator_UNSORTED_SEGMENT_SUM, 1}, "2.10.0"},
           {{BuiltinOperator_ATAN2, 1}, "2.10.0"},
           {{BuiltinOperator_SIGN, 1}, "2.11.0"},
           {{BuiltinOperator_SIGN, 2}, "2.12.0"}});

  std::pair<BuiltinOperator, int> version_key = {op_code, op_version};
  auto it = op_version_map->find(version_key);
  if (it == op_version_map->end()) {
    return std::string();
  }
  return it->second;
}

void UpdateMinimumRuntimeVersionForModel(uint8_t* model_buffer_pointer) {
  auto model = GetMutableModel(model_buffer_pointer);
  std::string model_min_version;
  auto subgraphs = model->subgraphs();
  for (int i = 0; i < subgraphs->Length(); ++i) {
    const SubGraph* subgraph = subgraphs->Get(i);
    for (int j = 0; j < subgraph->operators()->Length(); ++j) {
      const Operator* op = subgraph->operators()->Get(j);
      const OperatorCode* op_code =
          model->operator_codes()->Get(op->opcode_index());
      std::string runtime_version = FindMinimumRuntimeVersionForOp(
          GetBuiltinCode(op_code), op_code->version());
      // If we didn't find the current op version in the map, skip comparison.
      if (runtime_version.empty()) {
        continue;
      }
      if (CompareRuntimeVersion(model_min_version, runtime_version)) {
        // Current min model runtime version should be bumped if we see a higher
        // op version.
        model_min_version = runtime_version;
      }
    }
  }
  // The size of the `min_runtime_version` metadata buffer is 16 bytes. If the
  // generated `model_min_version` is equal or longer than 16 bytes, print a
  // warning message and return.
  if (model_min_version.size() >= 16) {
    TFLITE_LOG(TFLITE_LOG_WARNING,
               "Skip writing minimum runtime version string since it's "
               "longer than 16 bytes.");
    return;
  }
  // Copy over the bytes from `model_min_version` into the buffer.
  for (int i = 0; i < model->metadata()->size(); ++i) {
    if (model->metadata()->Get(i)->name()->str() == "min_runtime_version") {
      auto buffer = model->metadata()->Get(i)->buffer();
      auto buffer_data =
          model->mutable_buffers()->GetMutableObject(buffer)->mutable_data();
      memset(buffer_data->data(), 0, buffer_data->size());
      memcpy(buffer_data->data(), model_min_version.data(),
             model_min_version.size());
      break;
    }
  }
}

}  // namespace tflite
