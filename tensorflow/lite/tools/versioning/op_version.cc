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
#include "tensorflow/lite/tools/versioning/op_version.h"

#include <cstring>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"

namespace tflite {

int GetBuiltinOperatorVersion(const OpSignature& op_sig) {
  switch (op_sig.op) {
    case BuiltinOperator_CONV_2D:
      // If the op has signed int8 op_sig.inputs and op_sig.outputs, its
      // version 3.
      if (op_sig.input_types.at(0) == TensorType_INT8 &&
          op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_INT8) {
        return 3;
      }
      // If the op is a signed int8 hybrid operation, we need to return
      // version 2.
      if (op_sig.input_types.at(0) == TensorType_FLOAT32 &&
          op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_FLOAT32) {
        return 2;
      }
      return 1;

    case BuiltinOperator_DEPTHWISE_CONV_2D:
      // If the op has signed int8 op_sig.inputs and op_sig.outputs, its
      // version 3.
      if (op_sig.input_types.at(0) == TensorType_INT8 &&
          op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_INT8) {
        return 3;
      }
      if (op_sig.options.depthwise_conv_2d.dilation_w_factor != 1 ||
          op_sig.options.depthwise_conv_2d.dilation_h_factor != 1) {
        return 2;
      }
      return 1;

    case BuiltinOperator_FAKE_QUANT:
      if (op_sig.options.fakequant.narrow_range) {
        return 2;
      }
      return 1;

    case BuiltinOperator_FULLY_CONNECTED:
      // +-----------------+--------------------+--------------------------+
      // |                 |    Weight::Default | Weight::Shuffled4x16Int8 |
      // +-----------------+--------------------+--------------------------+
      // | Float           |                  1 |                        2 |
      // | Quantized Uint8 |                  1 |                        2 |
      // | Hybrid          |                  3 |                        3 |
      // | Quantized Int8  |                  4 |                        4 |
      // +-----------------+--------------------+--------------------------+
      // 2 op_sig.inputs (no bias) use case is supported starting from
      // version 6.
      if (op_sig.input_types.size() == 2) {
        return 6;
      }
      // `keep_num_dims` is supported at verison 5.
      if (op_sig.options.fully_connected.keep_num_dims) {
        return 5;
      }
      // Int8 fully fixed point kernel is at version 4.
      if (op_sig.input_types.at(0) == TensorType_INT8 &&
          op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_INT8) {
        return 4;
      }
      // If the op is a signed int8 hybrid operation, we need to return
      // version 3.
      if (op_sig.input_types.at(0) == TensorType_FLOAT32 &&
          op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_FLOAT32) {
        return 3;
      }
      // For float and uint8 fixed point kernels, if the weight is
      // Shuffled4x16Int8, is is version 2.
      if (op_sig.options.fully_connected.weights_format ==
          FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8) {
        return 2;
      }
      // Otherwise (weight is default), the version is 1.
      return 1;

    case BuiltinOperator_GATHER:
      // If the op takes bool input, it is version 3.
      if (op_sig.input_types.at(0) == TensorType_BOOL) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SVDF:
      // If the op is a signed int8 hybrid operation, we need to return
      // version 2.
      if (op_sig.input_types.at(0) == TensorType_FLOAT32 &&
          op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_FLOAT32) {
        return 2;
      }
      return 1;

    case BuiltinOperator_MUL:
      // Version 3 supports have a rescale value greater than or equal to 1.
      if (op_sig.options.mul.input1_scale != 0 &&
          op_sig.options.mul.input2_scale != 0 &&
          op_sig.options.mul.output_scale != 0 &&
          (op_sig.options.mul.input1_scale * op_sig.options.mul.input2_scale /
           op_sig.options.mul.output_scale) >= 1.0) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_TRANSPOSE:
      // If the op takes bool input, it is version 3.
      if (op_sig.input_types.at(0) == TensorType_BOOL) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_LSTM:
      // If the input tensor is float and a weight is int8, this is a version
      // 3 hybrid operation.
      if (op_sig.options.lstm.kernel_type == LSTMKernelType_FULL &&
          op_sig.input_types.at(0) == TensorType_FLOAT32 &&
          op_sig.input_types.at(2) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_FLOAT32) {
        return 3;
      }
      // KERNEL_BASIC was added in version 2.
      if (op_sig.options.lstm.kernel_type == LSTMKernelType_BASIC) {
        return 2;
      }
      return 1;

    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
      // If the input tensor is float and a weight is int8, this is a version
      // 2 hybrid operation.
      if (op_sig.input_types.at(0) == TensorType_FLOAT32 &&
          op_sig.input_types.at(2) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_FLOAT32) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SPLIT:
      // If the op take int8 input, it is version 2, for int32 it's version 3.
      if (op_sig.input_types.at(0) == TensorType_INT32) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SPARSE_TO_DENSE:
      // Version 3 supports Int8 and Uint8 type.
      if (op_sig.input_types.at(2) == TensorType_INT8 ||
          op_sig.input_types.at(2) == TensorType_UINT8) {
        return 3;
      }
      // Version 2 supports Int64 value type.
      if (op_sig.input_types.at(2) == TensorType_INT64) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SLICE:
      // Version 3 supports string input types.
      if (op_sig.input_types.at(0) == TensorType_STRING) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_UNPACK:
      // If the op take int8/uint8 input, it is version 2.
      if (op_sig.input_types.at(0) == TensorType_INT8 ||
          op_sig.input_types.at(0) == TensorType_UINT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_DEQUANTIZE:
      // Version 3 supports signed int16 input types.
      if (op_sig.input_types.at(0) == TensorType_INT16 ||
          op_sig.input_types.at(0) == TensorType_FLOAT16) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_FLOOR_DIV:
      if (op_sig.input_types.at(0) == TensorType_FLOAT32) {
        return 2;
      }
      return 1;

    case BuiltinOperator_L2_NORMALIZATION:
      if (op_sig.output_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_AVERAGE_POOL_2D:
    case BuiltinOperator_ADD:
    case BuiltinOperator_SPACE_TO_BATCH_ND:
    case BuiltinOperator_SUB:
    case BuiltinOperator_BATCH_TO_SPACE_ND:
    case BuiltinOperator_CONCATENATION:
    case BuiltinOperator_MAX_POOL_2D:
    case BuiltinOperator_MAXIMUM:
    case BuiltinOperator_MINIMUM:
    case BuiltinOperator_PAD:
    case BuiltinOperator_PADV2:
    case BuiltinOperator_SOFTMAX:
    case BuiltinOperator_SPACE_TO_DEPTH:
    case BuiltinOperator_MEAN:
    case BuiltinOperator_SUM:
    case BuiltinOperator_REDUCE_MAX:
    case BuiltinOperator_REDUCE_MIN:
    case BuiltinOperator_RELU6:
    case BuiltinOperator_RESIZE_BILINEAR:
    case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR:
    case BuiltinOperator_PACK:
    case BuiltinOperator_TANH:
    case BuiltinOperator_LOGISTIC:
    case BuiltinOperator_LOG_SOFTMAX:
    case BuiltinOperator_STRIDED_SLICE:
    case BuiltinOperator_TOPK_V2:
    case BuiltinOperator_ARG_MAX:
    case BuiltinOperator_ARG_MIN:
    case BuiltinOperator_EQUAL:
    case BuiltinOperator_NOT_EQUAL:
    case BuiltinOperator_GREATER:
    case BuiltinOperator_GREATER_EQUAL:
    case BuiltinOperator_LESS:
    case BuiltinOperator_LESS_EQUAL:
    case BuiltinOperator_SELECT:
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    default:
      return 1;
  }
}

}  // namespace tflite
