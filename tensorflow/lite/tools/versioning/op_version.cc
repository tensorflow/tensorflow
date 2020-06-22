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
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

static const auto kTensorTypeNone = static_cast<::tflite::TensorType>(-1);

// Get the number of dimensions of a tensor with idx of an operator op.
inline int GetNumDims(const SubGraph* subgraph, const Operator* op, int idx) {
  return subgraph->tensors()->Get(op->inputs()->Get(idx))->shape()->size();
}

// Compare shape of two tensors with idx1 and idx2 of an operator op, return
// true if they have the same shape.
inline bool HaveSameShapes(const SubGraph* subgraph, const Operator* op,
                           int idx1, int idx2) {
  const flatbuffers::Vector<int32_t>* shape1 =
      subgraph->tensors()->Get(op->inputs()->Get(idx1))->shape();
  const flatbuffers::Vector<int32_t>* shape2 =
      subgraph->tensors()->Get(op->inputs()->Get(idx2))->shape();
  if (shape1->size() != shape2->size()) {
    return false;
  }
  return std::equal(shape1->begin(), shape1->end(), shape2->begin());
}
}  // namespace

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
      // version 2.
      if (op_sig.input_types.at(0) == TensorType_FLOAT32 &&
          op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_FLOAT32) {
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
      // version 4.
      if (op_sig.input_types.at(0) == TensorType_FLOAT32 &&
          op_sig.input_types.at(1) == TensorType_INT8 &&
          op_sig.output_types.at(0) == TensorType_FLOAT32) {
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
      // Fully integer SVDF has int8 as input and is of version 3.
      if (op_sig.input_types.at(0) == TensorType_INT8) {
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

    case BuiltinOperator_RELU:
      if (op_sig.input_types.at(0) == TensorType_INT8 ||
          op_sig.input_types.at(0) == TensorType_UINT8) {
        return 2;
      }
      return 1;
    case BuiltinOperator_STRIDED_SLICE:
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
    case BuiltinOperator_REVERSE_V2:
      if (op_sig.input_types.at(0) == TensorType_BOOL) {
        return 2;
      }
      return 1;
    case BuiltinOperator_RESIZE_BILINEAR:
      if (op_sig.options.resize.half_pixel_centers) {
        return 3;
      } else if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;
    case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR:
      if (op_sig.options.resize.half_pixel_centers ||
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

    case BuiltinOperator_SPACE_TO_BATCH_ND:
    case BuiltinOperator_BATCH_TO_SPACE_ND:
      if (op_sig.options.single_input_op.num_dims != 4) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SUB:
      if (op_sig.options.broadcast.need_broadcast &&
          op_sig.options.broadcast.num_dims > 4) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_GATHER_ND:
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
      if (op_sig.input_types.size() >= 2 &&
          (op_sig.input_types.at(1) == TensorType_BOOL ||
           op_sig.input_types.at(1) == TensorType_STRING)) {
        return 2;
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

    case BuiltinOperator_CONCATENATION:
    case BuiltinOperator_SOFTMAX:
      // In case of int16 inputs, the version is 3.
      if (op_sig.input_types.at(0) == TensorType_INT16) {
        return 3;
      }
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_ADD:
    case BuiltinOperator_PAD:
    case BuiltinOperator_PADV2:
    case BuiltinOperator_SPACE_TO_DEPTH:
    case BuiltinOperator_SPLIT_V:
    case BuiltinOperator_MEAN:
    case BuiltinOperator_SUM:
    case BuiltinOperator_REDUCE_MAX:
    case BuiltinOperator_REDUCE_MIN:
    case BuiltinOperator_RELU6:
    case BuiltinOperator_LOG_SOFTMAX:
    case BuiltinOperator_TOPK_V2:
    case BuiltinOperator_ARG_MAX:
    case BuiltinOperator_ARG_MIN:
    case BuiltinOperator_GREATER:
    case BuiltinOperator_GREATER_EQUAL:
    case BuiltinOperator_LESS:
    case BuiltinOperator_LESS_EQUAL:
    case BuiltinOperator_SELECT:
    case BuiltinOperator_BATCH_MATMUL:
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;
    case BuiltinOperator_MIRROR_PAD:
      if (op_sig.input_types.at(0) == TensorType_INT8) {
        return 2;
      }
      return 1;
    default:
      return 1;
  }
}

TensorType GetTensorType(int32_t idx, const SubGraph* subgraph) {
  if (idx == -1)
    // For optional input/output, return none type directly.
    return kTensorTypeNone;

  // Some tests have a graph with invalid tensor index.
  TFLITE_DCHECK_GE(idx, 0);
  if (subgraph->tensors() && idx < subgraph->tensors()->Length()) {
    return subgraph->tensors()->Get(idx)->type();
  }
  LOG(ERROR) << "Can't access tenor " << idx;
  return kTensorTypeNone;
}

// Generate OpSignature with the given OperatorCode, Operator and Tensors (from
// SubGraph). The OpSignature will be used by GetBuiltinOperatorVersion() and
// mostly input and output tensor types are enough to figure out op version.
// But some ops (DEPTHWISE_CONV_2D,  FULLY_CONNECTED, ...) require to pass their
// options to decide op version.
OpSignature GetOpSignature(const OperatorCode* op_code, const Operator* op,
                           const SubGraph* subgraph) {
  OpSignature op_sig = {op_code->builtin_code()};

  switch (op_code->builtin_code()) {
    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      auto conv_option = op->builtin_options_as_DepthwiseConv2DOptions();
      if (conv_option) {
        op_sig.options.depthwise_conv_2d.dilation_w_factor =
            conv_option->dilation_w_factor();
        op_sig.options.depthwise_conv_2d.dilation_h_factor =
            conv_option->dilation_h_factor();
      }
    } break;

    case BuiltinOperator_FAKE_QUANT: {
      auto fakequant_option = op->builtin_options_as_FakeQuantOptions();
      if (fakequant_option) {
        op_sig.options.fakequant.narrow_range =
            fakequant_option->narrow_range();
      }
    } break;

    case BuiltinOperator_FULLY_CONNECTED: {
      auto fully_connected_option =
          op->builtin_options_as_FullyConnectedOptions();
      if (fully_connected_option) {
        op_sig.options.fully_connected.keep_num_dims =
            fully_connected_option->keep_num_dims();
        op_sig.options.fully_connected.weights_format =
            fully_connected_option->weights_format();
      }

      const Tensor* weight_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(1));
      op_sig.options.fully_connected.sparse_weight =
          (weight_tensor->sparsity() != nullptr);
    } break;

    case BuiltinOperator_MUL: {
      if (op->inputs()->Length() < 2 || op->outputs()->Length() < 1) {
        break;
      }
      const Tensor* input1_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(0));
      const Tensor* input2_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(1));
      const Tensor* output_tensor =
          subgraph->tensors()->Get(op->outputs()->Get(0));
      const QuantizationParameters* input1_quant =
          input1_tensor->quantization();
      const QuantizationParameters* input2_qunt = input2_tensor->quantization();
      const QuantizationParameters* output_quant =
          output_tensor->quantization();
      if (input1_quant && input1_quant->scale() &&
          input1_quant->scale()->Length() && input2_qunt &&
          input2_qunt->scale() && input2_qunt->scale()->Length() &&
          output_quant && output_quant->scale() &&
          output_quant->scale()->Length()) {
        op_sig.options.mul.input1_scale = input1_quant->scale()->Get(0);
        op_sig.options.mul.input2_scale = input2_qunt->scale()->Get(0);
        op_sig.options.mul.output_scale = output_quant->scale()->Get(0);
      }
    } break;

    case BuiltinOperator_LSTM: {
      auto lstm_option = op->builtin_options_as_LSTMOptions();
      if (lstm_option) {
        op_sig.options.lstm.kernel_type = lstm_option->kernel_type();
      }
    } break;

    case BuiltinOperator_RESIZE_BILINEAR: {
      auto resize_bilinear_option =
          op->builtin_options_as_ResizeBilinearOptions();
      if (resize_bilinear_option) {
        op_sig.options.resize.half_pixel_centers =
            resize_bilinear_option->half_pixel_centers();
        op_sig.options.resize.align_corners =
            resize_bilinear_option->align_corners();
      }
    } break;
    case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR: {
      auto resize_nn_option =
          op->builtin_options_as_ResizeNearestNeighborOptions();
      if (resize_nn_option) {
        op_sig.options.resize.half_pixel_centers =
            resize_nn_option->half_pixel_centers();
        op_sig.options.resize.align_corners = resize_nn_option->align_corners();
      }
    } break;
    // TODO(b/150176627): Add tests for GetOpSignature.
    case BuiltinOperator_STRIDED_SLICE:
    case BuiltinOperator_SPACE_TO_BATCH_ND:
    case BuiltinOperator_BATCH_TO_SPACE_ND:
    case BuiltinOperator_TRANSPOSE: {
      op_sig.options.single_input_op.num_dims = GetNumDims(subgraph, op, 0);
    } break;

    case BuiltinOperator_SUB:
    case BuiltinOperator_DIV:
    case BuiltinOperator_MAXIMUM:
    case BuiltinOperator_MINIMUM: {
      op_sig.options.broadcast.need_broadcast =
          !HaveSameShapes(subgraph, op, 0, 1);
      op_sig.options.broadcast.num_dims =
          std::max(GetNumDims(subgraph, op, 0), GetNumDims(subgraph, op, 1));
    } break;

    default:
      break;
  }

  for (int32_t i = 0; i < op->inputs()->Length(); ++i) {
    TensorType tensor_type = GetTensorType(op->inputs()->Get(i), subgraph);
    op_sig.input_types.push_back(tensor_type);
  }
  for (int32_t i = 0; i < op->outputs()->Length(); ++i) {
    TensorType tensor_type = GetTensorType(op->outputs()->Get(i), subgraph);
    op_sig.output_types.push_back(tensor_type);
  }
  return op_sig;
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

      if (op_code->builtin_code() != BuiltinOperator_CUSTOM) {
        OpSignature op_sig = GetOpSignature(op_code, op, subgraph);
        // Update builtin operator version.
        int32_t op_ver = GetBuiltinOperatorVersion(op_sig);
        if (!op_code->mutate_version(op_ver)) {
          LOG(ERROR) << "Can't set operator "
                     << EnumNameBuiltinOperator(op_code->builtin_code())
                     << " to version " << op_ver;
        }
      }
    }
  }
}

}  // namespace tflite
