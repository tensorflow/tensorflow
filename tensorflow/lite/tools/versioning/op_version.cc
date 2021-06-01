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

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace tflite {

int GetBuiltinOperatorVersion(const OpSignature& op_sig) {
  switch (op_sig.op) {
    case BuiltinOperator_CONV_2D:
      // If the op has signed int16 op_sig.inputs and op_sig.outputs, its
      // version 4.
      if (op_sig.input_types.at(0) == TensorType_INT16 &&
          op_sig.input_types.at(1) == TensorType_INT16 &&
          op_sig.output_types.at(1) == TensorType_INT16) {
        return 4;
      }

      // If the op has signed int8 op_sig.inputs and op_sig.outputs, its
      // version 3.
      if (op_sig.input_types.at(0) == TensorType_INT8 &&
          op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_INT8) {
        return 3;
      }
      // If the op is a signed int8 hybrid operation, we need to return
      // version 2 or 5 if per channel.
      if (op_sig.input_types.at(0) == TensorType_FLOAT32 &&
          op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_FLOAT32) {
        if (op_sig.options.conv_2d.is_per_channel_quantized) {
          return 5;
        }
        return 2;
      }
      return 1;

    case BuiltinOperator_DEPTHWISE_CONV_2D:
      // If the op accepts int16, we return version 5.
      if (op_sig.input_types.at(0) == TensorType_INT16 &&
          op_sig.input_types.at(1) == TensorType_INT16 &&
          op_sig.output_types.at(1) == TensorType_INT16) {
        return 5;
      }

      // If the op is a signed int8 hybrid operation, we need to return
      // version 4 or 6 if per-channel.
      if (op_sig.input_types.at(0) == TensorType_FLOAT32 &&
          op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_FLOAT32) {
        if (op_sig.options.depthwise_conv_2d.is_per_channel_quantized) {
          return 6;
        }
        return 4;
      }
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

      // FullyConnected with sparse weight is supported at version 8.
      if (op_sig.options.fully_connected.sparse_weight) {
        return 8;
      }

      // Int16 fully fixed point kernel is at version 7.
      if (op_sig.input_types.at(0) == TensorType_INT16 &&
          op_sig.input_types.at(1) == TensorType_INT16 &&
          op_sig.output_types.at(0) == TensorType_INT16) {
        return 7;
      }

      // 2 op_sig.inputs (no bias) use case is supported starting from
      // version 6.
      if (op_sig.input_types.size() == 2) {
        return 6;
      }
      // `keep_num_dims` is supported at version 5.
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
        if (op_sig.options.fully_connected.asymmetric_quantize_inputs) {
          // This is to use the updated quantization scheme.
          return 9;
        }
        return 3;
      }
      // For float and uint8 fixed point kernels, if the weight is
      // Shuffled4x16Int8, it is version 2.
      if (op_sig.options.fully_connected.weights_format ==
          FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8) {
        return 2;
      }
      // Otherwise (weight is default), the version is 1.
      return 1;

    case BuiltinOperator_GATHER:
      if (op_sig.options.gather.batch_dims != 0) {
        return 5;
      }

      if (op_sig.input_types.at(0) == TensorType_INT16) {
        return 4;
      }
      // If the op takes bool input, it is version 3.
      if (op_sig.input_types.at(0) == TensorType_BOOL) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SVDF:
      // Fully integer SVDF has int8 as input and is of version 3.
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 3;
      }
      // If the op is a signed int8 hybrid operation, we need to return
      // version 2.
      if (op_sig.input_types.at(0) == TensorType_FLOAT32 &&
          op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_FLOAT32) {
        // This is to use the updated quantization scheme
        if (op_sig.options.input_quantization.asymmetric_quantize_inputs) {
          return 4;
        }
        return 2;
      }
      return 1;

    case BuiltinOperator_MUL:
      // Version 4 supports int16 inputs
      if (op_sig.input_types.at(0) == TensorType_INT16) {
        return 4;
      }
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

    case BuiltinOperator_MAX_POOL_2D:
    case BuiltinOperator_AVERAGE_POOL_2D:
      if (op_sig.input_types.at(0) == TensorType_INT16 &&
          op_sig.output_types.at(0) == TensorType_INT16) {
        return 3;
      }

      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_TRANSPOSE:
      if (op_sig.input_types.at(0) == TensorType_INT16) {
        return 5;
      }
      if (op_sig.options.single_input_op.num_dims > 4) {
        return 4;
      }
      // If the op takes bool input, it is version 3.
      if (op_sig.input_types.at(0) == TensorType_BOOL) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_TRANSPOSE_CONV: {
      if (op_sig.input_types.size() == 4 &&
          op_sig.input_types.at(3) != kTensorTypeNone) {
        return 3;
      }
      // If the op takes int8 input, it is version 2.
      if (op_sig.input_types.at(1) == TensorType_INT8) {
        return 2;
      }
      return 1;
    }

    case BuiltinOperator_LSTM:
      // If the input tensor is float and a weight is int8, this is a version
      // 3 hybrid operation.
      if (op_sig.options.lstm.kernel_type == LSTMKernelType_FULL &&
          op_sig.input_types.at(0) == TensorType_FLOAT32 &&
          op_sig.input_types.at(2) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_FLOAT32) {
        if (op_sig.options.lstm.asymmetric_quantize_inputs) {
          return 4;
        }
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
        if (op_sig.options.lstm.asymmetric_quantize_inputs) {
          return 3;
        }
        return 2;
      }
      return 1;

    case BuiltinOperator_SPLIT:
      // If the op take in16 input, it is version 4.
      if (op_sig.input_types.at(1) == TensorType_INT16) {
        return 4;
      }
      // If the op take int8 input, it is version 2, for int32 it's version 3.
      // The input tensor is at index 1 not 0, 0 is the axis.
      if (op_sig.input_types.at(1) == TensorType_INT32) {
        return 3;
      }
      if (op_sig.input_types.at(1) == TensorType_INT8) {
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
      if (op_sig.options.single_input_op.num_dims > 4) {
        return 5;
      }
      if (op_sig.input_types.at(0) == TensorType_INT16) {
        return 4;
      }
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
      // If the op take bool input, it is version 3.
      if (op_sig.input_types.at(0) == TensorType_BOOL) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT16 &&
          op_sig.output_types.at(0) == TensorType_INT16) {
        return 4;
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

    case BuiltinOperator_ABS:
      if (op_sig.input_types.at(0) == TensorType_INT16) {
        return op_sig.options.abs.input_quantized ? 3 : 4;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8 ||
          op_sig.input_types.at(0) == TensorType_UINT8) {
        return 2;
      }
      return 1;
    case BuiltinOperator_RELU:
      if (op_sig.input_types.at(0) == TensorType_INT16) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8 ||
          op_sig.input_types.at(0) == TensorType_UINT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_STRIDED_SLICE:
      if (op_sig.options.strided_slice.ellipsis_mask != 0 ||
          op_sig.options.strided_slice.new_axis_mask != 0) {
        return 6;
      }
      if (op_sig.input_types.at(0) == TensorType_STRING) {
        return 5;
      }
      if (op_sig.options.strided_slice.num_dims > 4) {
        return 4;
      }
      // If the op takes bool input, it is version 3.
      if (op_sig.input_types.at(0) == TensorType_BOOL) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;
    case BuiltinOperator_REVERSE_V2:
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_BOOL) {
        return 2;
      }
      return 1;
    case BuiltinOperator_RESIZE_BILINEAR:
      if (op_sig.input_types.at(0) == TensorType_INT16) {
        return 4;
      } else if (op_sig.options.resize.half_pixel_centers) {
        return 3;
      } else if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;
    case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR:
      if (op_sig.input_types.at(0) == TensorType_INT16) {
        return 4;
      } else if (op_sig.options.resize.half_pixel_centers ||
                 op_sig.options.resize.align_corners) {
        return 3;
      } else if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_MAXIMUM:
    case BuiltinOperator_MINIMUM:
      if (op_sig.input_types.at(0) == TensorType_INT16 &&
          op_sig.output_types.at(0) == TensorType_INT16) {
        return 4;
      }
      if (op_sig.options.broadcast.need_broadcast &&
          op_sig.options.broadcast.num_dims > 4) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_PACK:
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }

      if (op_sig.input_types.at(0) == TensorType_INT16 &&
          op_sig.output_types.at(0) == TensorType_INT16) {
        return 3;
      }
      return 1;

    case BuiltinOperator_TILE:
      if (op_sig.input_types.at(0) == TensorType_STRING) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SQUEEZE:
      if (op_sig.input_types.at(0) == TensorType_STRING) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SPACE_TO_BATCH_ND:
    case BuiltinOperator_BATCH_TO_SPACE_ND:
      if (op_sig.options.single_input_op.num_dims != 4) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_ADD:
      if (!op_sig.input_types.empty() &&
          op_sig.input_types.at(0) == TensorType_INT64) {
        return 4;
      }
      if (op_sig.input_types.at(0) == TensorType_INT16 &&
          op_sig.output_types.at(0) == TensorType_INT16) {
        if (!op_sig.options.addsub.pot_scale_int16) {
          return 3;
        }
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SUB:
      if (op_sig.input_types.at(0) == TensorType_INT16 &&
          op_sig.output_types.at(0) == TensorType_INT16) {
        if (!op_sig.options.addsub.pot_scale_int16) {
          return 5;
        }
      }
      if (!op_sig.input_types.empty() &&
          op_sig.input_types.at(0) == TensorType_INT64) {
        return 4;
      }
      if (op_sig.options.addsub.need_broadcast &&
          op_sig.options.addsub.num_dims > 4) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_GATHER_ND:
      if (!op_sig.input_types.empty() &&
          (op_sig.input_types.at(0) == TensorType_INT16)) {
        return 3;
      }
      if (!op_sig.input_types.empty() &&
          op_sig.input_types.at(0) == TensorType_STRING) {
        return 2;
      }
      return 1;

    case BuiltinOperator_DIV:
      if (op_sig.options.broadcast.need_broadcast &&
          op_sig.options.broadcast.num_dims > 4) {
        return 2;
      }
      return 1;
    case BuiltinOperator_TANH:
    case BuiltinOperator_LOGISTIC:
      if (op_sig.input_types.at(0) == TensorType_INT16 &&
          op_sig.output_types.at(0) == TensorType_INT16) {
        return 3;
      }

      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_FILL:
      if (op_sig.input_types.size() >= 2) {
        if (op_sig.input_types.at(1) == TensorType_INT8 ||
            op_sig.input_types.at(1) == TensorType_INT16) {
          return 3;
        } else if ((op_sig.input_types.at(1) == TensorType_BOOL ||
                    op_sig.input_types.at(1) == TensorType_STRING)) {
          return 2;
        }
      }
      return 1;

    case BuiltinOperator_EQUAL:
    case BuiltinOperator_NOT_EQUAL:
      if (!op_sig.input_types.empty()) {
        if (op_sig.input_types.at(0) == TensorType_STRING) {
          return 3;
        }
        if (op_sig.input_types.at(0) == TensorType_INT8) {
          return 2;
        }
      }
      return 1;

    case BuiltinOperator_LEAKY_RELU:
      if (op_sig.input_types.at(0) == TensorType_INT16) {
        return 2;
      }
      return 1;

    case BuiltinOperator_BATCH_MATMUL:
      // In case of int16 inputs, the version is 3.
      if (op_sig.input_types.at(0) == TensorType_INT16) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      if (op_sig.input_types.at(0) == TensorType_FLOAT32 &&
          op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_FLOAT32) {
        if (op_sig.options.input_quantization.asymmetric_quantize_inputs) {
          // This is to use the updated quantization scheme.
          return 4;
        }
      }
      return 1;

    case BuiltinOperator_PAD:
    case BuiltinOperator_PADV2:
      if (op_sig.options.single_input_op.num_dims > 4) {
        return 4;
      }
      if (op_sig.input_types.at(0) == TensorType_INT16) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_CONCATENATION:
    case BuiltinOperator_SOFTMAX:
    case BuiltinOperator_MEAN:
    case BuiltinOperator_REDUCE_MAX:
    case BuiltinOperator_REDUCE_MIN:
    case BuiltinOperator_RELU6:
      // In case of int16 inputs, the version is 3.
      if (op_sig.input_types.at(0) == TensorType_INT16) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_RNN:
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN:
    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN:
    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM:
      if (op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_FLOAT32) {
        if (op_sig.options.input_quantization.asymmetric_quantize_inputs) {
          return 3;
        } else {
          return 2;
        }
      }
      return 1;

    case BuiltinOperator_SPACE_TO_DEPTH:
    case BuiltinOperator_SPLIT_V:
    case BuiltinOperator_SUM:
    case BuiltinOperator_LOG_SOFTMAX:
    case BuiltinOperator_TOPK_V2:
    case BuiltinOperator_ARG_MAX:
    case BuiltinOperator_ARG_MIN:
    case BuiltinOperator_GREATER:
    case BuiltinOperator_GREATER_EQUAL:
    case BuiltinOperator_LESS:
    case BuiltinOperator_LESS_EQUAL:
    case BuiltinOperator_SELECT:
    case BuiltinOperator_RSQRT:
    case BuiltinOperator_SQUARED_DIFFERENCE:
    case BuiltinOperator_DEPTH_TO_SPACE:
    case BuiltinOperator_MIRROR_PAD:
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;
    // The version one of broadcast to op won't be not supported since the
    // version one was rollbacked and the builtin op code number has been
    // changed because of builtin op code shortage problem.
    // Quantized broadcast_to is version 3
    case BuiltinOperator_BROADCAST_TO:
      if (op_sig.input_types.at(0) == TensorType_INT8 ||
          op_sig.input_types.at(0) == TensorType_INT16) {
        return 3;
      }
      return 2;
    default:
      return 1;
  }
  // Prevent lint error about this function being too long.
  // NOLINTNEXTLINE
}

void UpdateOpVersion(uint8_t* model_buffer_pointer) {
  auto model = GetMutableModel(model_buffer_pointer);
  auto subgraphs = model->subgraphs();

  for (int i = 0; i < subgraphs->Length(); ++i) {
    const SubGraph* subgraph = subgraphs->Get(i);
    for (int j = 0; j < subgraph->operators()->Length(); ++j) {
      const Operator* op = subgraph->operators()->Get(j);
      OperatorCode* op_code =
          model->mutable_operator_codes()->GetMutableObject(op->opcode_index());

      auto builtin_code = GetBuiltinCode(op_code);
      if (builtin_code != BuiltinOperator_CUSTOM) {
        OpSignature op_sig = GetOpSignature(op_code, op, subgraph);
        // Update builtin operator version.
        int32_t op_ver = GetBuiltinOperatorVersion(op_sig);
        // Skip updating op version if the current node uses lower version.
        // TODO(b/184366869): Populate multiple versions of operator once MLIR
        // quantizer is ready.
        if (op_ver <= op_code->version()) {
          continue;
        }
        if (!op_code->mutate_version(op_ver)) {
          LOG(ERROR) << "Can't set operator "
                     << EnumNameBuiltinOperator(builtin_code) << " to version "
                     << op_ver;
        }
      }
    }
  }
}

}  // namespace tflite
