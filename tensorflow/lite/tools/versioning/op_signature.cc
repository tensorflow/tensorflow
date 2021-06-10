/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/versioning/op_signature.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace tflite {
namespace {

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

TensorType GetTensorType(int32_t idx, const SubGraph* subgraph) {
  if (idx == -1)
    // For optional input/output, return none type directly.
    return kTensorTypeNone;

  // Some tests have a graph with invalid tensor index.
  TFLITE_DCHECK_GE(idx, 0);
  if (subgraph->tensors() && idx < subgraph->tensors()->Length()) {
    return subgraph->tensors()->Get(idx)->type();
  }
  LOG(ERROR) << "Can't access tensor " << idx;
  return kTensorTypeNone;
}

}  // namespace

OpSignature GetOpSignature(const OperatorCode* op_code, const Operator* op,
                           const SubGraph* subgraph) {
  auto builtin_code = GetBuiltinCode(op_code);
  OpSignature op_sig = {builtin_code};

  switch (builtin_code) {
    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      auto conv_option = op->builtin_options_as_DepthwiseConv2DOptions();
      if (conv_option) {
        op_sig.options.depthwise_conv_2d.dilation_w_factor =
            conv_option->dilation_w_factor();
        op_sig.options.depthwise_conv_2d.dilation_h_factor =
            conv_option->dilation_h_factor();
      }
      const Tensor* filter_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(1));
      const QuantizationParameters* filter_quant =
          filter_tensor->quantization();
      int num_channels = filter_tensor->shape()->Get(3);
      if (filter_quant && filter_quant->scale() &&
          filter_quant->scale()->Length() &&
          filter_quant->scale()->Length() == num_channels) {
        op_sig.options.depthwise_conv_2d.is_per_channel_quantized = true;
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
        op_sig.options.fully_connected.asymmetric_quantize_inputs =
            fully_connected_option->asymmetric_quantize_inputs();
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

    case BuiltinOperator_ADD: {
      auto add_option = op->builtin_options_as_AddOptions();
      op_sig.options.addsub.pot_scale_int16 = true;
      if (add_option) {
        op_sig.options.addsub.pot_scale_int16 = add_option->pot_scale_int16();
      }
    } break;

    case BuiltinOperator_SUB: {
      auto sub_option = op->builtin_options_as_SubOptions();
      op_sig.options.addsub.need_broadcast =
          !HaveSameShapes(subgraph, op, 0, 1);
      op_sig.options.addsub.num_dims =
          std::max(GetNumDims(subgraph, op, 0), GetNumDims(subgraph, op, 1));
      op_sig.options.addsub.pot_scale_int16 = true;
      if (sub_option) {
        op_sig.options.addsub.pot_scale_int16 = sub_option->pot_scale_int16();
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
    case BuiltinOperator_CONV_2D: {
      const Tensor* filter_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(1));
      const QuantizationParameters* filter_quant =
          filter_tensor->quantization();
      int num_channels = filter_tensor->shape()->Get(0);
      if (filter_quant && filter_quant->scale() &&
          filter_quant->scale()->Length() &&
          filter_quant->scale()->Length() == num_channels) {
        op_sig.options.conv_2d.is_per_channel_quantized = true;
      }
    } break;
    case BuiltinOperator_STRIDED_SLICE: {
      auto strided_slice_option = op->builtin_options_as_StridedSliceOptions();
      if (strided_slice_option) {
        op_sig.options.strided_slice.ellipsis_mask =
            strided_slice_option->ellipsis_mask();
        op_sig.options.strided_slice.new_axis_mask =
            strided_slice_option->new_axis_mask();
      }
      op_sig.options.strided_slice.num_dims = GetNumDims(subgraph, op, 0);
    } break;
    case BuiltinOperator_PAD:
    case BuiltinOperator_PADV2:
    case BuiltinOperator_SLICE:
    case BuiltinOperator_SPACE_TO_BATCH_ND:
    case BuiltinOperator_BATCH_TO_SPACE_ND:
    case BuiltinOperator_TRANSPOSE: {
      op_sig.options.single_input_op.num_dims = GetNumDims(subgraph, op, 0);
    } break;

    case BuiltinOperator_DIV:
    case BuiltinOperator_MAXIMUM:
    case BuiltinOperator_MINIMUM: {
      op_sig.options.broadcast.need_broadcast =
          !HaveSameShapes(subgraph, op, 0, 1);
      op_sig.options.broadcast.num_dims =
          std::max(GetNumDims(subgraph, op, 0), GetNumDims(subgraph, op, 1));
    } break;

    case BuiltinOperator_BATCH_MATMUL: {
      auto batch_matmul_option = op->builtin_options_as_BatchMatMulOptions();
      op_sig.options.input_quantization.asymmetric_quantize_inputs =
          batch_matmul_option->asymmetric_quantize_inputs();
    } break;

    case BuiltinOperator_GATHER: {
      auto gather_option = op->builtin_options_as_GatherOptions();
      op_sig.options.gather.batch_dims = gather_option->batch_dims();
    } break;

    case BuiltinOperator_ABS: {
      if (subgraph->tensors()->Get(op->inputs()->Get(0))->quantization()) {
        op_sig.options.abs.input_quantized = true;
      }
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

}  // namespace tflite
