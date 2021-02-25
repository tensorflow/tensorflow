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

#include "tensorflow/lite/micro/kernels/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/fixedpoint_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {
namespace {

struct OpData {
  OpDataConv reference_op_data;

#if defined(FUSION_F1)
  int scratch_tensor_index;
#endif  // defined(FUSION_F1)
};

#if defined(HIFIMINI)
void EvalHifiMini(const ConvParams& params, const int32_t* output_multiplier,
                  const int32_t* output_shift, const RuntimeShape& input_shape,
                  const int8_t* input_data, const RuntimeShape& filter_shape,
                  const int8_t* filter_data, const RuntimeShape& bias_shape,
                  const int32_t* bias_data, const RuntimeShape& output_shape,
                  int8_t* output_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  const int batches = input_shape.Dims(0);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);

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
            for (int filter_x = 0; filter_x < filter_width; filter_x += 2) {
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
                        input_depth * 2 -
                    2;
                const int8_t* input_vals_offset_ptr = input_data + input_idx;
                for (int i = 0; i < input_depth; i += 2) {
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
                      i - 2;
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
          // alignment. Convert the (unsigned) 32-bit multiplier down to a
          // 24-bit multiplier.
          acc_56 = MultiplyByQuantizedMultiplier(
              acc_24x2, output_multiplier[out_channel] >> 8,
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
inline void Conv1x32Input32x32FilterHifiMini(
    const int input_offset, const int output_offset,
    const int quantized_activation_min, const int quantized_activation_max,
    const int32_t* output_multiplier, const int32_t* output_shift,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
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
    acc_56 = MultiplyByQuantizedMultiplier(acc_24x2, output_multiplier[ch] >> 8,
                                           output_shift[ch]);

    // Add output offset, cap activation, and assign to the output:
    acc_56 = AE_ADDQ56(acc_56, output_offset_56);
    acc_56 = AE_MINQ56S(acc_56, output_activation_max_56);
    acc_56 = AE_MAXQ56S(acc_56, output_activation_min_56);

    output_data[ch] = static_cast<int8_t>(AE_TRUNCA32Q48(acc_56));
  }
}
#endif  // defined(HIFIMINI)

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, ConvPrepare(context, node));

#if defined(FUSION_F1)
  OpData* data = static_cast<OpData*>(node->user_data);
  const auto params = static_cast<const TfLiteConvParams*>(node->builtin_data);

  // Calculate scratch memory requirements and request scratch buffer
  TfLiteTensor* output = GetOutput(context, node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* input = GetInput(context, node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* filter = GetInput(context, node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);

  const RuntimeShape& input_shape = GetTensorShape(input);
  const RuntimeShape& filter_shape = GetTensorShape(filter);
  const RuntimeShape& output_shape = GetTensorShape(output);
  const int input_height = input_shape.Dims(1);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_channels = output_shape.Dims(3);
  const int stride_height = params->stride_height;
  const int pad_height = data->reference_op_data.padding.height;

  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);

  int required_scratch = 0;
  // Dilation is currently not supported on HiFi 4 NN Library
  if ((params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1)) {
    required_scratch = xa_nn_conv2d_std_getsize(
        input_height, input_depth, filter_height, filter_width, stride_height,
        pad_height, output_height, output_channels, PREC_ASYM8S);
    TF_LITE_ENSURE(context, required_scratch > 0);
  }
  TF_LITE_ENSURE_OK(
      context, context->RequestScratchBufferInArena(
                   context, required_scratch, &data->scratch_tensor_index));
#endif  // defined(FUSION_F1)
  return kTfLiteOk;
}

#if defined(FUSION_F1)
TfLiteStatus EvalHifi4(TfLiteContext* context, TfLiteNode* node,
                       const TfLiteConvParams& params, const OpData& data,
                       const TfLiteEvalTensor* input,
                       const TfLiteEvalTensor* filter,
                       const TfLiteEvalTensor* bias, TfLiteEvalTensor* output,
                       TfLiteEvalTensor* im2col) {
  /* Dilation is currently not supported on HiFi 4 NN Library */
  if ((params.dilation_width_factor == 1) &&
      (params.dilation_height_factor == 1)) {
    const int32_t input_offset = -data.reference_op_data.input_zero_point;
    const int32_t output_offset = data.reference_op_data.output_zero_point;
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int pad_width = data.reference_op_data.padding.width;
    const int pad_height = data.reference_op_data.padding.height;
    const int32_t output_activation_min =
        data.reference_op_data.output_activation_min;
    const int32_t output_activation_max =
        data.reference_op_data.output_activation_max;

    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
    const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
    const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
    int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

    int output_data_format = 0;
    int out_length = output_height * output_width * output_depth;

    if (filter_height == 1 && filter_width == 1) {
      for (int batch = 0; batch < batches; ++batch) {
        int8_t* p_out_temp;
        p_out_temp = &output_data[batch * out_length];

        TF_LITE_ENSURE_EQ(
            context,

            xa_nn_conv2d_pointwise_per_chan_sym8sxasym8s(
                p_out_temp, const_cast<WORD8*>(filter_data),
                const_cast<WORD8*>(&input_data[batch * input_height *
                                               input_width * input_depth]),
                const_cast<WORD32*>(bias_data), input_height, input_width,
                input_depth, output_depth, input_offset,
                data.reference_op_data.per_channel_output_multiplier,
                data.reference_op_data.per_channel_output_shift, output_offset,
                output_data_format),
            0);

        TF_LITE_ENSURE_EQ(context,
                          xa_nn_vec_activation_min_max_8_8(
                              p_out_temp, p_out_temp, output_activation_min,
                              output_activation_max, out_length),
                          0);
      }
    } else {
      void* p_scratch = static_cast<void*>(
          context->GetScratchBuffer(context, data.scratch_tensor_index));

      for (int batch = 0; batch < batches; ++batch) {
        int8_t* p_out_temp;
        p_out_temp = &output_data[batch * out_length];

        {
          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_conv2d_std_per_chan_sym8sxasym8s(
                  p_out_temp,
                  &input_data[batch * input_height * input_width * input_depth],
                  const_cast<int8_t*>(filter_data),  // filter_data,
                  bias_data, input_height, input_width, input_depth,
                  filter_height, filter_width, output_depth, stride_width,
                  stride_height, pad_width, pad_height, output_height,
                  output_width, input_offset,
                  data.reference_op_data.per_channel_output_multiplier,
                  data.reference_op_data.per_channel_output_shift,
                  output_offset, output_data_format,
                  static_cast<void*>(p_scratch)),
              0);
        }

        TF_LITE_ENSURE_EQ(context,
                          xa_nn_vec_activation_min_max_8_8(
                              p_out_temp, p_out_temp, output_activation_min,
                              output_activation_max, out_length),
                          0);
      }
    }
    return kTfLiteOk;
  }

  reference_integer_ops::ConvPerChannel(
      ConvParamsQuantized(params, data.reference_op_data),
      data.reference_op_data.per_channel_output_multiplier,
      data.reference_op_data.per_channel_output_shift,
      tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<int32_t>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}
#endif  // defined(FUSION_F1)

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  const auto& op_data = *(reinterpret_cast<OpData*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;

#if defined(HIFIMINI)
  int* input_dims = input->dims->data;
  int* filter_dims = filter->dims->data;
  if (input_dims[0] == 1 && input_dims[1] == 1 && input_dims[2] == 1 &&
      input_dims[3] == 32 && filter_dims[0] == 32 && filter_dims[1] == 1 &&
      filter_dims[2] == 1 && filter_dims[3] == 32) {
    Conv1x32Input32x32FilterHifiMini(
        -op_data.reference_op_data.input_zero_point,
        op_data.reference_op_data.output_zero_point,
        op_data.reference_op_data.output_activation_min,
        op_data.reference_op_data.output_activation_max,
        op_data.reference_op_data.per_channel_output_multiplier,
        op_data.reference_op_data.per_channel_output_shift,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetTensorData<int32_t>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
    return kTfLiteOk;
  }
#endif  // defined(HIFIMINI)

  switch (input->type) {
    case kTfLiteInt8: {
#if defined(HIFIMINI)
      EvalHifiMini(ConvParamsQuantized(params, op_data.reference_op_data),
                   op_data.reference_op_data.per_channel_output_multiplier,
                   op_data.reference_op_data.per_channel_output_shift,
                   tflite::micro::GetTensorShape(input),
                   tflite::micro::GetTensorData<int8_t>(input),
                   tflite::micro::GetTensorShape(filter),
                   tflite::micro::GetTensorData<int8_t>(filter),
                   tflite::micro::GetTensorShape(bias),
                   tflite::micro::GetTensorData<int32_t>(bias),
                   tflite::micro::GetTensorShape(output),
                   tflite::micro::GetTensorData<int8_t>(output));
#elif defined(FUSION_F1)
      EvalHifi4(context, node, params, op_data, input, filter, bias, output,
                nullptr);
#else
      reference_integer_ops::ConvPerChannel(
          ConvParamsQuantized(params, op_data.reference_op_data),
          op_data.reference_op_data.per_channel_output_multiplier,
          op_data.reference_op_data.per_channel_output_shift,
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<int8_t>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetTensorData<int32_t>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
#endif
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration Register_CONV_2D() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
