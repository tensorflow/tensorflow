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

#include "tensorflow/lite/kernels/internal/reference/conv.h"

#include <xtensa/tie/xt_hifi2.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/xtensa_hifimini_legacy/fixedpoint_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace conv {
namespace xtensa {
namespace hifimini {

void ConvPerChannel(const ConvParams& params, const int32* output_multiplier,
                    const int32* output_shift, const RuntimeShape& input_shape,
                    const int8* input_data, const RuntimeShape& filter_shape,
                    const int8* filter_data, const RuntimeShape& bias_shape,
                    const int32* bias_data, const RuntimeShape& output_shape,
                    int8* output_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32 input_offset = params.input_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;

  const int batches = input_shape.Dims(0);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int input_depth_iters = input_depth / 2;

  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_depth = filter_shape.Dims(3);

  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  ae_p24x2s input_offset_24x2 = AE_MOVPA24(input_offset);
  ae_q56s output_offset_56 = AE_CVTQ48A32S(output_offset);
  ae_q56s output_activation_min_56 = AE_CVTQ48A32S(output_activation_min);
  ae_q56s output_activation_max_56 = AE_CVTQ48A32S(output_activation_max);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          ae_q56s acc_56 = AE_ZEROQ56();

          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);
              if (is_point_inside_image) {
                // Find current input index, minus 2 for Xtensa load
                // alignments:
                // TODO(b/147322595): Consider doing these offset calculations
                // with intrinsics:
                int input_idx =
                    ((batch * input_height + in_y) * input_width + in_x) *
                        input_depth -
                    2;
                const int8_t* input_vals_offset_ptr = input_data + input_idx;
                for (int i = 0; i < input_depth_iters; ++i) {
                  // Load signed 2x 8bit values and right shift into 24bit
                  // alignment:
                  ae_p24x2s input_vals_24x2;
                  AE_LP8X2F_IU(input_vals_24x2, input_vals_offset_ptr, 2);
                  input_vals_24x2 = AE_P24X2S_SRAI(input_vals_24x2, 16);

                  // Add input offset (24bit aligned):
                  input_vals_24x2 =
                      AE_P24S_ADDS_P24X2S(input_vals_24x2, input_offset_24x2);

                  // Find current filter index, minus 2 for Xtensa load
                  // alignments:
                  int filter_idx =
                      ((out_channel * filter_height + filter_y) * filter_width +
                       filter_x) *
                          filter_depth +
                      (i * 2) - 2;
                  const int8_t* filter_vals_offset_ptr =
                      filter_data + filter_idx;

                  // Load signed 2x 8bit values and right shift into 24bit
                  // alignment:
                  ae_p24x2s filter_vals_24x2;
                  AE_LP8X2F_IU(filter_vals_24x2, filter_vals_offset_ptr, 2);
                  filter_vals_24x2 = AE_P24X2S_SRAI(filter_vals_24x2, 16);

                  // Multiply and accumulate into 48bit bit space:
                  AE_MULAAP24S_HH_LL(acc_56, filter_vals_24x2, input_vals_24x2);
                }
              }
            }
          }

          // Left shift from 48bit alignment to 32bit:
          acc_56 = AE_Q56S_SLAI(acc_56, 16);

          if (bias_data) {
            // Load and add bias at 32bit alignment:
            ae_q56s bias_56 = AE_CVTQ48A32S(bias_data[out_channel]);
            acc_56 = AE_ADDQ56(acc_56, bias_56);
          }

          // Shift from 32bit alignment to 24bit alignment and place back on
          // the PR register:
          acc_56 = AE_Q56S_SLAI(acc_56, 8);
          ae_p24x2s acc_24x2 = AE_TRUNCP24Q48(acc_56);

          // Apply quantized multiplier and accumulate result at 48bit
          // alignment:
          acc_56 = micro::xtensa::hifimini::MultiplyByQuantizedMultiplier(
              acc_24x2, output_multiplier[out_channel],
              output_shift[out_channel]);

          // Add output offset, cap activation, and assign to the output:
          acc_56 = AE_ADDQ56(acc_56, output_offset_56);
          acc_56 = AE_MINQ56S(acc_56, output_activation_max_56);
          acc_56 = AE_MAXQ56S(acc_56, output_activation_min_56);

          int output_idx =
              ((batch * output_height + out_y) * output_width + out_x) *
                  output_depth +
              out_channel;
          output_data[output_idx] = static_cast<int8_t>(AE_TRUNCA32Q48(acc_56));
        }
      }
    }
  }
}

// TODO(b/154240772): Move shared code into common methods.
inline void Conv1x32Input32x32Filter(
    const int input_offset, const int output_offset,
    const int quantized_activation_min, const int quantized_activation_max,
    const int32* output_multiplier, const int32* output_shift,
    const RuntimeShape& input_shape, const int8* input_data,
    const RuntimeShape& filter_shape, const int8* filter_data,
    const RuntimeShape& bias_shape, const int32* bias_data,
    const RuntimeShape& output_shape, int8* output_data) {
  ae_p24x2s input_offset_24x2 = AE_MOVPA24(input_offset);
  ae_q56s output_offset_56 = AE_CVTQ48A32S(output_offset);
  ae_q56s output_activation_max_56 = AE_CVTQ48A32S(quantized_activation_max);
  ae_q56s output_activation_min_56 = AE_CVTQ48A32S(quantized_activation_min);

  constexpr int kChannels = 32;
  constexpr int kFilterDepth = 32;
  for (int ch = 0; ch < kChannels; ch++) {
    ae_q56s acc_56 = AE_ZEROQ56();
    const int8_t* input_vals_ptr = input_data - 2;
    for (int i = 0; i < kFilterDepth; i += 2) {
      // Load signed 2x 8bit values and right shift into 24bit
      // alignment:
      ae_p24x2s input_vals_24x2;
      AE_LP8X2F_IU(input_vals_24x2, input_vals_ptr, 2);
      input_vals_24x2 = AE_P24X2S_SRAI(input_vals_24x2, 16);

      // Add input offset (24bit aligned):
      input_vals_24x2 = AE_P24S_ADDS_P24X2S(input_vals_24x2, input_offset_24x2);
      // Find current filter index, minus 2 for Xtensa load
      // alignments:
      const int filter_idx = ch * kFilterDepth + i - 2;
      const int8_t* filter_vals_offset_ptr = filter_data + filter_idx;

      // Load signed 2x 8bit values and right shift into 24bit
      // alignment:
      ae_p24x2s filter_vals_24x2;
      AE_LP8X2F_IU(filter_vals_24x2, filter_vals_offset_ptr, 2);
      filter_vals_24x2 = AE_P24X2S_SRAI(filter_vals_24x2, 16);

      // Multiply and accumulate into 48bit bit space:
      AE_MULAAP24S_HH_LL(acc_56, filter_vals_24x2, input_vals_24x2);
    }
    // Left shift from 48bit alignment to 32bit:
    acc_56 = AE_Q56S_SLAI(acc_56, 16);
    if (bias_data) {
      // Load and add bias at 32bit alignment:
      ae_q56s bias_56 = AE_CVTQ48A32S(bias_data[ch]);
      acc_56 = AE_ADDQ56(acc_56, bias_56);
    }

    // Shift from 32bit alignment to 24bit alignment and place back on
    // the PR register:
    acc_56 = AE_Q56S_SLAI(acc_56, 8);
    ae_p24x2s acc_24x2 = AE_TRUNCP24Q48(acc_56);

    // Apply quantized multiplier and accumulate result at 48bit alignment.
    // Convert the (unsigned) 32-bit multiplier down to a 24-bit multiplier.
    acc_56 = micro::xtensa::hifimini::MultiplyByQuantizedMultiplier(
        acc_24x2, output_multiplier[ch] >> 8, output_shift[ch]);

    // Add output offset, cap activation, and assign to the output:
    acc_56 = AE_ADDQ56(acc_56, output_offset_56);
    acc_56 = AE_MINQ56S(acc_56, output_activation_max_56);
    acc_56 = AE_MAXQ56S(acc_56, output_activation_min_56);

    output_data[ch] = static_cast<int8_t>(AE_TRUNCA32Q48(acc_56));
  }
}

}  // namespace hifimini
}  // namespace xtensa

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kConvQuantizedDimension = 0;

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteConvParams* params, int width, int height,
                             int filter_width, int filter_height, int out_width,
                             int out_height, const TfLiteType data_type,
                             OpData* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor, height,
      width, filter_height, filter_width, padding, &out_height, &out_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    int output_channels = filter->dims->data[kConvQuantizedDimension];

    return tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift),
        output_channels);
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data = nullptr;
  if (context->AllocatePersistentBuffer(context, sizeof(OpData), &data) ==
      kTfLiteError) {
    return nullptr;
  }
  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);

  auto* op_data = reinterpret_cast<OpData*>(node->user_data);

  int input_width = input->dims->data[2];
  int input_height = input->dims->data[1];
  int filter_width = filter->dims->data[2];
  int filter_height = filter->dims->data[1];
  int output_width = output->dims->data[2];
  int output_height = output->dims->data[1];

  // Per channel quantization is only needed for int8 inference. For other
  // quantized types, only a single scale and zero point is needed.
  const int num_channels = filter->dims->data[kConvQuantizedDimension];
  // Dynimically allocate per-channel quantization parameters.
  TF_LITE_ENSURE_STATUS(context->AllocatePersistentBuffer(
      context, num_channels * sizeof(int32_t),
      reinterpret_cast<void**>(&op_data->per_channel_output_multiplier)));
  TF_LITE_ENSURE_STATUS(context->AllocatePersistentBuffer(
      context, num_channels * sizeof(int32_t),
      reinterpret_cast<void**>(&op_data->per_channel_output_shift)));

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);

    TF_LITE_ENSURE(context,
                   affine_quantization->scale->size == 1 ||
                       affine_quantization->scale->size ==
                           filter->dims->data[kConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  return CalculateOpData(context, node, params, input_width, input_height,
                         filter_width, filter_height, output_width,
                         output_height, input->type, op_data);
}

void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             TfLiteConvParams* params, OpData* data,
                             const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             TfLiteTensor* im2col) {
  // TODO(b/154032858): Investigate removing extra copies.
  ConvParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  xtensa::hifimini::ConvPerChannel(
      op_params, data->per_channel_output_multiplier,
      data->per_channel_output_shift, GetTensorShape(input),
      GetTensorData<int8>(input), GetTensorShape(filter),
      GetTensorData<int8>(filter), GetTensorShape(bias),
      GetTensorData<int32>(bias), GetTensorShape(output),
      GetTensorData<int8>(output));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);

  int* input_dims = input->dims->data;
  int* filter_dims = filter->dims->data;
  if (input_dims[0] == 1 && input_dims[1] == 1 && input_dims[2] == 1 &&
      input_dims[3] == 32 && filter_dims[0] == 32 && filter_dims[1] == 1 &&
      filter_dims[2] == 1 && filter_dims[3] == 32) {
    xtensa::hifimini::Conv1x32Input32x32Filter(
        -input->params.zero_point, output->params.zero_point,
        op_data->output_activation_min, op_data->output_activation_max,
        op_data->per_channel_output_multiplier,
        op_data->per_channel_output_shift, GetTensorShape(input),
        GetTensorData<int8>(input), GetTensorShape(filter),
        GetTensorData<int8>(filter), GetTensorShape(bias),
        GetTensorData<int32>(bias), GetTensorShape(output),
        GetTensorData<int8>(output));
    return kTfLiteOk;
  }

  switch (input->type) {
    case kTfLiteInt8:
      EvalQuantizedPerChannel(context, node, params, op_data, input, filter,
                              bias, output, nullptr);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace conv

TfLiteRegistration* Register_CONV_2D() {
  static TfLiteRegistration r = {/*init=*/conv::Init,
                                 /*free=*/nullptr,
                                 /*prepare=*/conv::Prepare,
                                 /*invoke=*/conv::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
