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

#if defined(FUSION_F1) || defined(HIFI5)

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_conv.h"

namespace tflite {

TfLiteStatus ConvPrepareHifi(TfLiteContext* context, TfLiteNode* node) {
  XtensaConvOpData* data = static_cast<XtensaConvOpData*>(node->user_data);
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
  return kTfLiteOk;
}

TfLiteStatus ConvEvalHifi(TfLiteContext* context, TfLiteNode* node,
                          const TfLiteConvParams& params,
                          const XtensaConvOpData& data,
                          const TfLiteEvalTensor* input,
                          const TfLiteEvalTensor* filter,
                          const TfLiteEvalTensor* bias,
                          TfLiteEvalTensor* output) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  /* Dilation is currently not supported on HiFi 4 NN Library */
  if ((params.dilation_width_factor == 1) &&
      (params.dilation_height_factor == 1) &&
      input_shape.Dims(1) >= filter_shape.Dims(1) &&
      input_shape.Dims(2) >= filter_shape.Dims(2)) {
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

}  // namespace tflite
#endif  // defined(FUSION_F1) || defined(HIFI5)
