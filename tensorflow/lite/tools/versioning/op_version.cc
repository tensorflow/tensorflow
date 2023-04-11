/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace tflite {
namespace {

bool NeedBroadcastForBinaryInputs(const OpSignature& op_sig) {
  if (op_sig.inputs.size() < 2) {
    return false;
  }
  return (op_sig.inputs.at(0).dims != op_sig.inputs.at(1).dims);
}

int GetInputMaxDims(const OpSignature& op_sig) {
  int max_dims = 0;
  for (auto& input : op_sig.inputs) {
    if (input.dims.size() > max_dims) {
      max_dims = input.dims.size();
    }
  }
  return max_dims;
}

}  // namespace

int GetBuiltinOperatorVersion(const OpSignature& op_sig) {
  switch (op_sig.op) {
    case BuiltinOperator_CONV_2D:
      if (op_sig.ext_options.conv_2d.is_grouped_convolution) {
        return 6;
      }
      // If the op has signed int16 op_sig.inputs and op_sig.outputs, its
      // version 4.
      if (op_sig.inputs.at(0).type == kTfLiteInt16 &&
          op_sig.inputs.at(1).type == kTfLiteInt16 &&
          op_sig.outputs.at(1).type == kTfLiteInt16) {
        return 4;
      }

      // If the op has signed int8 op_sig.inputs and op_sig.outputs, its
      // version 3.
      if (op_sig.inputs.at(0).type == kTfLiteInt8 &&
          op_sig.inputs.at(1).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteInt8) {
        return 3;
      }
      // If the op has signed int8 and int4 op_sig.inputs and op_sig.outputs,
      // its version 7.
      if (op_sig.inputs.at(0).type == kTfLiteInt8 &&
          op_sig.inputs.at(1).type == kTfLiteInt4 &&
          op_sig.outputs.at(0).type == kTfLiteInt8) {
        return 7;
      }
      // If the op is a signed int8 hybrid operation, we need to return
      // version 2 or 5 if per channel.
      if (op_sig.inputs.at(0).type == kTfLiteFloat32 &&
          op_sig.inputs.at(1).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteFloat32) {
        if (op_sig.ext_options.conv_2d.is_per_channel_quantized) {
          return 5;
        }
        return 2;
      }
      return 1;

    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      // If the op accepts int16, we return version 5.
      if (op_sig.inputs.at(0).type == kTfLiteInt16 &&
          op_sig.inputs.at(1).type == kTfLiteInt16 &&
          op_sig.outputs.at(1).type == kTfLiteInt16) {
        return 5;
      }

      // If the op is a signed int8 hybrid operation, we need to return
      // version 4 or 6 if per-channel.
      if (op_sig.inputs.at(0).type == kTfLiteFloat32 &&
          op_sig.inputs.at(1).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteFloat32) {
        if (op_sig.ext_options.depthwise_conv_2d.is_per_channel_quantized) {
          return 6;
        }
        return 4;
      }
      // If the op has signed int8 op_sig.inputs and op_sig.outputs, its
      // version 3.
      if (op_sig.inputs.at(0).type == kTfLiteInt8 &&
          op_sig.inputs.at(1).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteInt8) {
        return 3;
      }

      // If the op has signed int8 and int4 op_sig.inputs and op_sig.outputs,
      // its version 7.
      if (op_sig.inputs.at(0).type == kTfLiteInt8 &&
          op_sig.inputs.at(1).type == kTfLiteInt4 &&
          op_sig.outputs.at(0).type == kTfLiteInt8) {
        return 7;
      }

      auto depthwise_conv_params =
          reinterpret_cast<TfLiteDepthwiseConvParams*>(op_sig.builtin_data);
      TFLITE_DCHECK(depthwise_conv_params != nullptr);
      if (depthwise_conv_params->dilation_width_factor != 1 ||
          depthwise_conv_params->dilation_height_factor != 1) {
        return 2;
      }
      return 1;
    }

    case BuiltinOperator_FAKE_QUANT: {
      auto fake_quant_params =
          reinterpret_cast<TfLiteFakeQuantParams*>(op_sig.builtin_data);
      TFLITE_DCHECK(fake_quant_params != nullptr);
      if (fake_quant_params->narrow_range) {
        return 2;
      }
      return 1;
    }

    case BuiltinOperator_FULLY_CONNECTED: {
      // +-----------------+--------------------+--------------------------+
      // |                 |    Weight::Default | Weight::Shuffled4x16Int8 |
      // +-----------------+--------------------+--------------------------+
      // | Float           |                  1 |                        2 |
      // | Quantized Uint8 |                  1 |                        2 |
      // | Hybrid          |                  3 |                        3 |
      // | Quantized Int8  |                  4 |                        4 |
      // +-----------------+--------------------+--------------------------+

      // FullyConnected with sparse weight is supported at version 8.
      if (op_sig.ext_options.fully_connected.sparse_weight) {
        return 8;
      }

      // Int16 fully fixed point kernel is at version 7.
      if (op_sig.inputs.at(0).type == kTfLiteInt16 &&
          op_sig.inputs.at(1).type == kTfLiteInt16 &&
          op_sig.outputs.at(0).type == kTfLiteInt16) {
        return 7;
      }

      // 2 op_sig.inputs (no bias) use case is supported starting from
      // version 6.
      if (op_sig.inputs.size() == 2) {
        return 6;
      }
      auto fully_connected_params =
          reinterpret_cast<TfLiteFullyConnectedParams*>(op_sig.builtin_data);
      TFLITE_DCHECK(fully_connected_params != nullptr);
      // `keep_num_dims` is supported at version 5.
      if (fully_connected_params->keep_num_dims) {
        return 5;
      }
      // Int8 fully fixed point kernel is at version 4.
      if (op_sig.inputs.at(0).type == kTfLiteInt8 &&
          op_sig.inputs.at(1).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteInt8) {
        return 4;
      }

      // If the op has signed int8 and int4 op_sig.inputs and op_sig.outputs,
      // its version 7.
      if (op_sig.inputs.at(0).type == kTfLiteInt8 &&
          op_sig.inputs.at(1).type == kTfLiteInt4 &&
          op_sig.outputs.at(0).type == kTfLiteInt8) {
        return 10;
      }

      // If the op is a signed int8 hybrid operation, we need to return
      // version 3.
      if (op_sig.inputs.at(0).type == kTfLiteFloat32 &&
          op_sig.inputs.at(1).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteFloat32) {
        if (fully_connected_params->asymmetric_quantize_inputs) {
          // This is to use the updated quantization scheme.
          return 9;
        }
        return 3;
      }
      // For float and uint8 fixed point kernels, if the weight is
      // Shuffled4x16Int8, it is version 2.
      if (fully_connected_params->weights_format ==
          kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8) {
        return 2;
      }
      // Otherwise (weight is default), the version is 1.
      return 1;
    }

    case BuiltinOperator_GATHER: {
      if (op_sig.inputs.at(1).type == kTfLiteInt16) {
        return 6;
      }
      auto gather_params =
          reinterpret_cast<TfLiteGatherParams*>(op_sig.builtin_data);
      if (gather_params && gather_params->batch_dims != 0) {
        return 5;
      }

      if (op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 4;
      }
      // If the op takes bool input, it is version 3.
      if (op_sig.inputs.at(0).type == kTfLiteBool) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;
    }

    case BuiltinOperator_SVDF: {
      // Fully integer SVDF has int8 as input and is of version 3.
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 3;
      }
      // If the op is a signed int8 hybrid operation, we need to return
      // version 2.
      if (op_sig.inputs.at(0).type == kTfLiteFloat32 &&
          op_sig.inputs.at(1).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteFloat32) {
        auto svdf_params =
            reinterpret_cast<TfLiteSVDFParams*>(op_sig.builtin_data);
        // This is to use the updated quantization scheme
        if (svdf_params && svdf_params->asymmetric_quantize_inputs) {
          return 4;
        }
        return 2;
      }
      return 1;
    }

    case BuiltinOperator_SIGN:
      // Version 2 supports int32 inputs
      if (op_sig.inputs.at(0).type == kTfLiteInt32) {
        return 2;
      }
      return 1;

    case BuiltinOperator_MUL:
      // Version 7 supports int16 and uint32 inputs
      if ((op_sig.inputs.at(0).type == kTfLiteInt16 &&
           !op_sig.ext_options.mul.input_quantized) ||
          op_sig.inputs.at(0).type == kTfLiteUInt32) {
        return 7;
      }
      // Version 6 supports complex32 inputs
      if (op_sig.inputs.at(0).type == kTfLiteComplex64) {
        return 6;
      }
      // Version 5 supports int64 inputs
      if (op_sig.inputs.at(0).type == kTfLiteInt64) {
        return 5;
      }
      // Version 4 supports int16 inputs
      if (op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 4;
      }
      // Version 3 supports have a rescale value greater than or equal to 1.
      if (op_sig.ext_options.mul.input1_scale != 0 &&
          op_sig.ext_options.mul.input2_scale != 0 &&
          op_sig.ext_options.mul.output_scale != 0 &&
          (op_sig.ext_options.mul.input1_scale *
           op_sig.ext_options.mul.input2_scale /
           op_sig.ext_options.mul.output_scale) >= 1.0) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_MAX_POOL_2D:
    case BuiltinOperator_AVERAGE_POOL_2D:
      if (op_sig.inputs.at(0).type == kTfLiteInt16 &&
          op_sig.outputs.at(0).type == kTfLiteInt16) {
        return 3;
      }

      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_TRANSPOSE:
      if (op_sig.inputs.at(0).dims.size() > 5) {
        return 6;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 5;
      }
      if (op_sig.inputs.at(0).dims.size() > 4) {
        return 4;
      }
      // If the op takes bool input, it is version 3.
      if (op_sig.inputs.at(0).type == kTfLiteBool) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_TRANSPOSE_CONV: {
      auto transpose_conv_params =
          reinterpret_cast<TfLiteTransposeConvParams*>(op_sig.builtin_data);

      // TransposeConvOp has fused activation function from version 4.
      if (transpose_conv_params != nullptr &&
          transpose_conv_params->activation) {
        return 4;
      }

      if (op_sig.inputs.size() == 4 &&
          op_sig.inputs.at(3).type != kTfLiteNoType) {
        return 3;
      }
      // If the op takes int8 input, it is version 2.
      if (op_sig.inputs.at(1).type == kTfLiteInt8) {
        return 2;
      }
      return 1;
    }

    case BuiltinOperator_LSTM: {
      auto lstm_params =
          reinterpret_cast<TfLiteLSTMParams*>(op_sig.builtin_data);
      // If the input activation and output tensor are int16 and a weight is
      // int8, this is a version 5.
      if (lstm_params->kernel_type == kTfLiteLSTMFullKernel &&
          op_sig.inputs.at(0).type == kTfLiteInt16 &&
          op_sig.inputs.at(2).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteInt16) {
        return 5;
      }
      // If the input tensor is float and a weight is int8, this is a version
      // 3 hybrid operation.
      TFLITE_DCHECK(lstm_params != nullptr);
      if (lstm_params->kernel_type == kTfLiteLSTMFullKernel &&
          op_sig.inputs.at(0).type == kTfLiteFloat32 &&
          op_sig.inputs.at(2).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteFloat32) {
        if (lstm_params->asymmetric_quantize_inputs) {
          return 4;
        }
        return 3;
      }
      // KERNEL_BASIC was added in version 2.
      if (lstm_params->kernel_type == kTfLiteLSTMBasicKernel) {
        return 2;
      }
      return 1;
    }

    case BuiltinOperator_SPLIT:
      // If the op take in16 input, it is version 4.
      if (op_sig.inputs.at(1).type == kTfLiteInt16) {
        return 4;
      }
      // If the op take int8 input, it is version 2, for int32 it's version 3.
      // The input tensor is at index 1 not 0, 0 is the axis.
      if (op_sig.inputs.at(1).type == kTfLiteInt32) {
        return 3;
      }
      if (op_sig.inputs.at(1).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SPARSE_TO_DENSE:
      // Version 3 supports Int8 and Uint8 type.
      if (op_sig.inputs.at(2).type == kTfLiteInt8 ||
          op_sig.inputs.at(2).type == kTfLiteUInt8) {
        return 3;
      }
      // Version 2 supports Int64 value type.
      if (op_sig.inputs.at(2).type == kTfLiteInt64) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SLICE:
      if (op_sig.inputs.at(0).dims.size() > 4) {
        return 5;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 4;
      }
      // Version 3 supports string input types.
      if (op_sig.inputs.at(0).type == kTfLiteString) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_UNPACK:
      // If the op take int8/uint8 input, it is version 2.
      if (op_sig.inputs.at(0).type == kTfLiteInt8 ||
          op_sig.inputs.at(0).type == kTfLiteUInt8) {
        return 2;
      }
      // If the op take bool input, it is version 3.
      if (op_sig.inputs.at(0).type == kTfLiteBool) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt16 &&
          op_sig.outputs.at(0).type == kTfLiteInt16) {
        return 4;
      }
      return 1;

    case BuiltinOperator_DEQUANTIZE:
      if (op_sig.inputs.at(0).type == kTfLiteInt4) {
        return 6;
      }
      // Version 3 supports signed int16 input types.
      if (op_sig.inputs.at(0).type == kTfLiteInt16 ||
          op_sig.inputs.at(0).type == kTfLiteFloat16) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        if (op_sig.ext_options.dequantize.is_per_channel_quantized) {
          return 5;
        }
        return 2;
      }
      return 1;

    case BuiltinOperator_QUANTIZE:
      if (op_sig.inputs.at(0).type == kTfLiteInt4 ||
          op_sig.outputs.at(0).type == kTfLiteInt4) {
        return 4;
      }
      if (op_sig.ext_options.quantize.is_per_channel_quantized) {
        return 3;
      }
      if (op_sig.outputs.at(0).type == kTfLiteInt16) {
        return 2;
      }
      return 1;

    case BuiltinOperator_FLOOR_DIV:
      if (op_sig.inputs.at(0).type == kTfLiteInt16 ||
          op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteFloat32) {
        return 2;
      }
      return 1;

    case BuiltinOperator_FLOOR_MOD:
      if (op_sig.inputs.at(0).type == kTfLiteInt16 ||
          op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_L2_NORMALIZATION:
      if (op_sig.outputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_ABS:
      // Version 5 supports int32
      if (op_sig.inputs.at(0).type == kTfLiteInt32) {
        return 5;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt16) {
        return op_sig.ext_options.abs.input_quantized ? 3 : 4;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8 ||
          op_sig.inputs.at(0).type == kTfLiteUInt8) {
        return 2;
      }
      return 1;
    case BuiltinOperator_RELU:
      if (op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8 ||
          op_sig.inputs.at(0).type == kTfLiteUInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_STRIDED_SLICE: {
      auto strided_slice_params =
          reinterpret_cast<TfLiteStridedSliceParams*>(op_sig.builtin_data);
      TFLITE_DCHECK(strided_slice_params != nullptr);
      if (strided_slice_params->ellipsis_mask != 0 ||
          strided_slice_params->new_axis_mask != 0) {
        return 6;
      }
      if (op_sig.inputs.at(0).type == kTfLiteString) {
        return 5;
      }
      if (op_sig.ext_options.strided_slice.num_dims > 4) {
        return 4;
      }
      // If the op takes bool input, it is version 3.
      if (op_sig.inputs.at(0).type == kTfLiteBool) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;
    }
    case BuiltinOperator_REVERSE_V2:
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteBool) {
        return 2;
      }
      return 1;
    case BuiltinOperator_RESIZE_BILINEAR: {
      if (op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 4;
      }
      auto resize_bilinear_params =
          reinterpret_cast<TfLiteResizeBilinearParams*>(op_sig.builtin_data);
      TFLITE_DCHECK(resize_bilinear_params != nullptr);
      if (resize_bilinear_params->half_pixel_centers) {
        return 3;
      } else if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;
    }
    case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR: {
      if (op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 4;
      }
      auto resize_nearest_neighbor_params =
          reinterpret_cast<TfLiteResizeNearestNeighborParams*>(
              op_sig.builtin_data);
      TFLITE_DCHECK(resize_nearest_neighbor_params != nullptr);
      if (resize_nearest_neighbor_params->half_pixel_centers ||
          resize_nearest_neighbor_params->align_corners) {
        return 3;
      } else if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;
    }

    case BuiltinOperator_MAXIMUM:
    case BuiltinOperator_MINIMUM:
      if (op_sig.inputs.at(0).type == kTfLiteInt16 &&
          op_sig.outputs.at(0).type == kTfLiteInt16) {
        return 4;
      }
      if (NeedBroadcastForBinaryInputs(op_sig) && GetInputMaxDims(op_sig) > 4) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_PACK:
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }

      if (op_sig.inputs.at(0).type == kTfLiteInt16 &&
          op_sig.outputs.at(0).type == kTfLiteInt16) {
        return 3;
      }
      return 1;

    case BuiltinOperator_TILE:
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteString) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SQUEEZE:
      if (op_sig.inputs.at(0).type == kTfLiteString) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SPACE_TO_BATCH_ND:
    case BuiltinOperator_BATCH_TO_SPACE_ND:
      if (op_sig.inputs.at(0).dims.size() != 4) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_ADD: {
      if (!op_sig.inputs.empty() && op_sig.inputs.at(0).type == kTfLiteInt16 &&
          !op_sig.ext_options.add.input_quantized) {
        return 5;
      }
      if (!op_sig.inputs.empty() && op_sig.inputs.at(0).type == kTfLiteInt64) {
        return 4;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt16 &&
          op_sig.outputs.at(0).type == kTfLiteInt16) {
        auto add_params =
            reinterpret_cast<TfLiteAddParams*>(op_sig.builtin_data);
        if (add_params && !add_params->pot_scale_int16) {
          return 3;
        }
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;
    }

    case BuiltinOperator_SUB: {
      if (op_sig.inputs.at(0).type == kTfLiteInt16 &&
          op_sig.outputs.at(0).type == kTfLiteInt16) {
        auto sub_params =
            reinterpret_cast<TfLiteSubParams*>(op_sig.builtin_data);
        if (sub_params && !sub_params->pot_scale_int16) {
          return 5;
        }
      }
      if (!op_sig.inputs.empty() && op_sig.inputs.at(0).type == kTfLiteInt64) {
        return 4;
      }
      if (NeedBroadcastForBinaryInputs(op_sig) && GetInputMaxDims(op_sig) > 4) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;
    }

    case BuiltinOperator_GATHER_ND:
      if (op_sig.inputs.at(1).type == kTfLiteInt16) {
        return 4;
      }
      if (!op_sig.inputs.empty() &&
          (op_sig.inputs.at(0).type == kTfLiteInt16)) {
        return 3;
      }
      if (!op_sig.inputs.empty() && op_sig.inputs.at(0).type == kTfLiteString) {
        return 2;
      }
      return 1;

    case BuiltinOperator_DIV:
      if (NeedBroadcastForBinaryInputs(op_sig) && GetInputMaxDims(op_sig) > 4) {
        return 2;
      }
      return 1;
    case BuiltinOperator_TANH:
    case BuiltinOperator_LOGISTIC:
      if (op_sig.inputs.at(0).type == kTfLiteInt16 &&
          op_sig.outputs.at(0).type == kTfLiteInt16) {
        return 3;
      }

      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_FILL:
      if (op_sig.inputs.size() >= 2) {
        if (op_sig.inputs.at(1).type == kTfLiteFloat16) return 4;
        if (op_sig.inputs.at(1).type == kTfLiteInt8 ||
            op_sig.inputs.at(1).type == kTfLiteInt16) {
          return 3;
        } else if ((op_sig.inputs.at(1).type == kTfLiteBool ||
                    op_sig.inputs.at(1).type == kTfLiteString)) {
          return 2;
        }
      }
      return 1;

    case BuiltinOperator_EQUAL:
      if (!op_sig.inputs.empty()) {
        if (op_sig.inputs.at(0).type == kTfLiteInt16) {
          return 4;
        }
        if (op_sig.inputs.at(0).type == kTfLiteString) {
          return 3;
        }
        if (op_sig.inputs.at(0).type == kTfLiteInt8) {
          return 2;
        }
      }
      return 1;
    case BuiltinOperator_NOT_EQUAL:
      if (!op_sig.inputs.empty()) {
        if (op_sig.inputs.at(0).type == kTfLiteString) {
          return 3;
        }
        if (op_sig.inputs.at(0).type == kTfLiteInt8) {
          return 2;
        }
      }
      return 1;

    case BuiltinOperator_LEAKY_RELU:
      if (op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 2;
      }
      return 1;

    case BuiltinOperator_BATCH_MATMUL: {
      // In case of int16 inputs, the version is 3.
      if (op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      if (op_sig.inputs.at(0).type == kTfLiteFloat32 &&
          op_sig.inputs.at(1).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteFloat32) {
        auto batch_mat_mul_params =
            reinterpret_cast<TfLiteBatchMatMulParams*>(op_sig.builtin_data);
        if (batch_mat_mul_params &&
            batch_mat_mul_params->asymmetric_quantize_inputs) {
          // This is to use the updated quantization scheme.
          return 4;
        }
      }
      return 1;
    }

    case BuiltinOperator_PAD:
    case BuiltinOperator_PADV2:
      if (op_sig.inputs.at(0).dims.size() > 4) {
        return 4;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_CONCATENATION:
    case BuiltinOperator_SOFTMAX:
    case BuiltinOperator_MEAN:
    case BuiltinOperator_MIRROR_PAD:
    case BuiltinOperator_REDUCE_MAX:
    case BuiltinOperator_REDUCE_MIN:
    case BuiltinOperator_RELU6:
      // In case of int16 inputs, the version is 3.
      if (op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_RNN: {
      if (op_sig.inputs.at(1).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteFloat32) {
        auto rnn_params =
            reinterpret_cast<TfLiteRNNParams*>(op_sig.builtin_data);
        if (rnn_params && rnn_params->asymmetric_quantize_inputs) {
          return 3;
        } else {
          return 2;
        }
      }
      return 1;
    }

    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN: {
      if (op_sig.inputs.at(1).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteFloat32) {
        auto sequence_rnn_params =
            reinterpret_cast<TfLiteSequenceRNNParams*>(op_sig.builtin_data);
        if (sequence_rnn_params &&
            sequence_rnn_params->asymmetric_quantize_inputs) {
          return 3;
        } else {
          return 2;
        }
      }
      return 1;
    }

    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN: {
      if (op_sig.inputs.at(1).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteFloat32) {
        auto bidirectional_sequence_rnn_params =
            reinterpret_cast<TfLiteBidirectionalSequenceRNNParams*>(
                op_sig.builtin_data);
        if (bidirectional_sequence_rnn_params &&
            bidirectional_sequence_rnn_params->asymmetric_quantize_inputs) {
          return 3;
        } else {
          return 2;
        }
      }
      return 1;
    }

    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM: {
      if (op_sig.inputs.at(1).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteFloat32) {
        auto bidirectional_sequence_lstm_params =
            reinterpret_cast<TfLiteBidirectionalSequenceLSTMParams*>(
                op_sig.builtin_data);
        if (bidirectional_sequence_lstm_params &&
            bidirectional_sequence_lstm_params->asymmetric_quantize_inputs) {
          return 3;
        } else {
          return 2;
        }
      }
      return 1;
    }

    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM: {
      auto unidirectional_sequence_lstm_params =
          reinterpret_cast<TfLiteUnidirectionalSequenceLSTMParams*>(
              op_sig.builtin_data);
      // If the input activation and output tensor are int16 and a weight is
      // int8, this is a version 5.
      if (op_sig.inputs.at(0).type == kTfLiteInt16 &&
          op_sig.inputs.at(2).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteInt16) {
        return 5;
      }
      if (unidirectional_sequence_lstm_params &&
          unidirectional_sequence_lstm_params->diagonal_recurrent_tensors) {
        return 4;
      }
      // If the input tensor is float and a weight is int8, this is a version
      // 2 hybrid operation.
      if (op_sig.inputs.at(0).type == kTfLiteFloat32 &&
          op_sig.inputs.at(2).type == kTfLiteInt8 &&
          op_sig.outputs.at(0).type == kTfLiteFloat32) {
        if (unidirectional_sequence_lstm_params &&
            unidirectional_sequence_lstm_params->asymmetric_quantize_inputs) {
          return 3;
        }
        return 2;
      }
      return 1;
    }

    case BuiltinOperator_ARG_MAX:
    case BuiltinOperator_ARG_MIN:
      if (op_sig.inputs.at(0).type == kTfLiteBool) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_SELECT: {
      if (op_sig.inputs.at(0).dims.size() == 5 ||
          op_sig.inputs.at(1).dims.size() == 5 ||
          op_sig.inputs.at(2).dims.size() == 5)
        return 3;
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;
    }
    case BuiltinOperator_LESS:
    case BuiltinOperator_GREATER_EQUAL: {
      if (op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 3;
      }
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;
    }
    case BuiltinOperator_SPACE_TO_DEPTH:
    case BuiltinOperator_SPLIT_V:
    case BuiltinOperator_SUM:
    case BuiltinOperator_LOG_SOFTMAX:
    case BuiltinOperator_TOPK_V2:
    case BuiltinOperator_GREATER:
    case BuiltinOperator_LESS_EQUAL:
    case BuiltinOperator_RSQRT:
    case BuiltinOperator_SQUARED_DIFFERENCE:
    case BuiltinOperator_DEPTH_TO_SPACE:
      if (op_sig.inputs.at(0).type == kTfLiteInt8) {
        return 2;
      }
      return 1;

    case BuiltinOperator_EXP:
    case BuiltinOperator_REDUCE_PROD:
      if (op_sig.inputs.at(0).type == kTfLiteInt8 ||
          op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 2;
      }
      return 1;

    // The version one of broadcast to op won't be not supported since the
    // version one was rollbacked and the builtin op code number has been
    // changed because of builtin op code shortage problem.
    // Quantized broadcast_to is version 3
    case BuiltinOperator_BROADCAST_TO:
      if (op_sig.inputs.at(0).type == kTfLiteInt8 ||
          op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 3;
      }
      return 2;
    case BuiltinOperator_CAST:
      if (op_sig.inputs.at(0).type == kTfLiteFloat64 ||
          op_sig.outputs.at(0).type == kTfLiteFloat64 ||
          op_sig.inputs.at(0).type == kTfLiteFloat16 ||
          op_sig.outputs.at(0).type == kTfLiteFloat16) {
        return 5;
      } else if (op_sig.inputs.at(0).type == kTfLiteUInt16 ||
                 op_sig.outputs.at(0).type == kTfLiteUInt16) {
        return 4;
      } else if (op_sig.inputs.at(0).type == kTfLiteInt8 ||
                 op_sig.outputs.at(0).type == kTfLiteInt8) {
        return 3;
      } else if (op_sig.inputs.at(0).type == kTfLiteUInt32 ||
                 op_sig.outputs.at(0).type == kTfLiteUInt32) {
        return 2;
      }
      return 1;
    case BuiltinOperator_WHERE:
      if (op_sig.inputs.at(0).type == kTfLiteBool) return 1;
      return 2;
    case BuiltinOperator_GELU:
      if (op_sig.inputs.at(0).type == kTfLiteInt8 ||
          op_sig.inputs.at(0).type == kTfLiteUInt8) {
        return 2;
      }
      return 1;
    case BuiltinOperator_NON_MAX_SUPPRESSION_V4:
    case BuiltinOperator_NON_MAX_SUPPRESSION_V5:
      if (op_sig.inputs.at(0).type == kTfLiteInt8 ||
          op_sig.inputs.at(0).type == kTfLiteInt16) {
        return 2;
      }
      return 1;
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
        OpSignature op_sig = GetOpSignature(op_code, op, subgraph, model);
        // Update builtin operator version.
        int32_t op_ver = GetBuiltinOperatorVersion(op_sig);
        if (op_sig.builtin_data) {
          free(op_sig.builtin_data);
        }
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
