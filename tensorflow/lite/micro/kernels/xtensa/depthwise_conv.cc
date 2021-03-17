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

#include "tensorflow/lite/micro/kernels/depthwise_conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
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
inline void EvalHifiMini(
    const DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  // TODO(b/154032858): Investigate removing extra copies.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
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
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            ae_q56s acc_56 = AE_ZEROQ56();
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                // Zero padding by omitting the areas outside the image.
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
                          input_depth +
                      (in_channel);
                  int32_t input_val = input_data[input_idx];

                  // Find current filter index, minus 2 for Xtensa load
                  // alignments:
                  int filter_idx =
                      ((filter_y)*filter_width + filter_x) * filter_depth +
                      (output_channel);
                  int32_t filter_val = filter_data[filter_idx];

                  // Load 8bit value as int32_t into a 24x24 register and right
                  // shift into 24bit space. Note: value is duplicated in the HH
                  // and LL register - but all calculations are done on the HH
                  // side.
                  ae_p24x2s input_val_24x2 = AE_MOVPA24(input_val);

                  // Add input offset (24bit aligned):
                  input_val_24x2 =
                      AE_P24S_ADDS_P24X2S(input_val_24x2, input_offset_24x2);

                  // Load filter 8bit value into 24bit alignment:
                  ae_p24x2s filter_val_24x2 = AE_MOVPA24(filter_val);

                  // Multiply and accumulate the HH side of each 24x24 PR
                  // register:
                  AE_MULAS56P24S_HH(acc_56, filter_val_24x2, input_val_24x2);
                }
              }
            }

            // Left shift from 48bit alignment to 32bit:
            acc_56 = AE_Q56S_SLAI(acc_56, 16);

            if (bias_data) {
              // Load and add bias at 32bit alignment:
              ae_q56s bias_56 = AE_CVTQ48A32S(bias_data[output_channel]);
              acc_56 = AE_ADDQ56(acc_56, bias_56);
            }

            // Shift from 32bit alignment to 24bit alignment and place back on
            // the PR register:
            acc_56 = AE_Q56S_SLAI(acc_56, 8);
            ae_p24x2s acc_24x2 = AE_TRUNCP24Q48(acc_56);

            // Apply quantized multiplier and accumulate result at 48bit
            // alignment:
            acc_56 = MultiplyByQuantizedMultiplier(
                acc_24x2, output_multiplier[output_channel],
                output_shift[output_channel]);

            // Add output offset, cap activation, and assign to the output:
            acc_56 = AE_ADDQ56(acc_56, output_offset_56);
            acc_56 = AE_MINQ56S(acc_56, output_activation_max_56);
            acc_56 = AE_MAXQ56S(acc_56, output_activation_min_56);

            int output_idx =
                ((batch * output_height + out_y) * output_width + out_x) *
                    output_depth +
                output_channel;
            output_data[output_idx] =
                static_cast<int8_t>(AE_TRUNCA32Q48(acc_56));
          }
        }
      }
    }
  }
}

constexpr int kConvolutionalKernelWidth = 4;
constexpr int kConvolutionalKernelDepth = 32;
inline void DepthwiseConv4x32MatchingInputAndFilterHifiMini(
    const int input_offset, const int output_offset,
    const int quantized_activation_min, const int quantized_activation_max,
    const int32_t* output_multiplier, const int32_t* output_shift,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  // Convert the (unsigned) 32-bit multiplier down to a 24-bit multiplier.
  const int32_t mult = output_multiplier[0] >> 8;
  const int32_t shift = output_shift[0];
  ae_p24x2s input_offset_24x2 = AE_MOVPA24(input_offset);
  ae_q56s output_offset_56 = AE_CVTQ48A32S(output_offset);
  ae_q56s output_activation_min_56 = AE_CVTQ48A32S(quantized_activation_min);
  ae_q56s output_activation_max_56 = AE_CVTQ48A32S(quantized_activation_max);

  const int num_blocks =
      kConvolutionalKernelDepth / 2;  // Based on the 24x2 register size.
  const int stride_elements =
      (kConvolutionalKernelDepth / kConvolutionalKernelWidth);

  const int8_t* input_0_ptr = (const int8_t*)(input_data - 2);
  const int8_t* weight_0_ptr = (const int8_t*)(filter_data - 2);
  // Apply the kernels in blocks of 4 for all the channels.
  const int8_t* input_1_ptr = input_0_ptr + stride_elements * 4;
  const int8_t* input_2_ptr = input_1_ptr + stride_elements * 4;
  const int8_t* input_3_ptr = input_2_ptr + stride_elements * 4;

  const int8_t* weight_1_ptr = weight_0_ptr + stride_elements * 4;
  const int8_t* weight_2_ptr = weight_1_ptr + stride_elements * 4;
  const int8_t* weight_3_ptr = weight_2_ptr + stride_elements * 4;

  for (int i = 0; i < num_blocks; ++i) {
    ae_q56s block_0_acc = AE_ZEROQ56();
    ae_q56s block_1_acc = AE_ZEROQ56();

    // Load all the weights.
    ae_p24x2s weight_0, weight_1, weight_2, weight_3;
    AE_LP8X2F_IU(weight_0, weight_0_ptr, 2);
    AE_LP8X2F_IU(weight_1, weight_1_ptr, 2);
    AE_LP8X2F_IU(weight_2, weight_2_ptr, 2);
    AE_LP8X2F_IU(weight_3, weight_3_ptr, 2);

    // Load all the inputs.
    ae_p24x2s input_0, input_1, input_2, input_3;
    AE_LP8X2F_IU(input_0, input_0_ptr, 2);
    AE_LP8X2F_IU(input_1, input_1_ptr, 2);
    AE_LP8X2F_IU(input_2, input_2_ptr, 2);
    AE_LP8X2F_IU(input_3, input_3_ptr, 2);

    // Shift inputs to 8 bit alignment and add offsets.
    input_0 = AE_P24X2S_SRAI(input_0, 16);
    input_1 = AE_P24X2S_SRAI(input_1, 16);
    input_2 = AE_P24X2S_SRAI(input_2, 16);
    input_3 = AE_P24X2S_SRAI(input_3, 16);

    input_0 = AE_P24S_ADDS_P24X2S(input_0, input_offset_24x2);
    input_1 = AE_P24S_ADDS_P24X2S(input_1, input_offset_24x2);
    input_2 = AE_P24S_ADDS_P24X2S(input_2, input_offset_24x2);
    input_3 = AE_P24S_ADDS_P24X2S(input_3, input_offset_24x2);

    // Do the multiplies across all channels.  Resulting accumulators are 32bit
    // aligned (24 bit aligned weights * 8 bit aligned inputs).
    AE_MULAS56P24S_HH(block_0_acc, input_0, weight_0);
    AE_MULAS56P24S_HH(block_0_acc, input_1, weight_1);
    AE_MULAS56P24S_HH(block_0_acc, input_2, weight_2);
    AE_MULAS56P24S_HH(block_0_acc, input_3, weight_3);

    AE_MULAS56P24S_LL(block_1_acc, input_0, weight_0);
    AE_MULAS56P24S_LL(block_1_acc, input_1, weight_1);
    AE_MULAS56P24S_LL(block_1_acc, input_2, weight_2);
    AE_MULAS56P24S_LL(block_1_acc, input_3, weight_3);

    int ch_0 = i * 2;
    int ch_1 = i * 2 + 1;

    // Load and add bias at 32bit alignment:
    ae_q56s bias_56_0 = AE_CVTQ48A32S(bias_data[ch_0]);
    ae_q56s bias_56_1 = AE_CVTQ48A32S(bias_data[ch_1]);
    block_0_acc = AE_ADDQ56(block_0_acc, bias_56_0);
    block_1_acc = AE_ADDQ56(block_1_acc, bias_56_1);

    // Shift from 32bit alignment to 24bit alignment and place back on
    // the PR register:
    block_0_acc = AE_Q56S_SLAI(block_0_acc, 8);
    block_1_acc = AE_Q56S_SLAI(block_1_acc, 8);
    ae_p24x2s acc_24x2_0 = AE_TRUNCP24Q48(block_0_acc);
    ae_p24x2s acc_24x2_1 = AE_TRUNCP24Q48(block_1_acc);

    // Apply quantized multiplier and accumulate result at 48bit
    // alignment:
    block_0_acc = MultiplyByQuantizedMultiplier(acc_24x2_0, mult, shift);
    // Apply quantized multiplier and accumulate result at 48bit
    // alignment:
    block_1_acc = MultiplyByQuantizedMultiplier(acc_24x2_1, mult, shift);

    // Add output offset, cap activation, and assign to the output:
    block_0_acc = AE_ADDQ56(block_0_acc, output_offset_56);
    block_1_acc = AE_ADDQ56(block_1_acc, output_offset_56);
    block_0_acc = AE_MINQ56S(block_0_acc, output_activation_max_56);
    block_1_acc = AE_MINQ56S(block_1_acc, output_activation_max_56);
    block_0_acc = AE_MAXQ56S(block_0_acc, output_activation_min_56);
    block_1_acc = AE_MAXQ56S(block_1_acc, output_activation_min_56);

    output_data[ch_0] = static_cast<int8_t>(AE_TRUNCA32Q48(block_0_acc));
    output_data[ch_1] = static_cast<int8_t>(AE_TRUNCA32Q48(block_1_acc));
  }
}
#endif  // defined(HIFIMINI)

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, DepthwiseConvPrepare(context, node));

#if defined(FUSION_F1)
  OpData* data = static_cast<OpData*>(node->user_data);
  const auto& params =
      *(static_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data));

  // Calculate scratch memory requirements and request scratch buffer
  TfLiteTensor* output = GetOutput(context, node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* input = GetInput(context, node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* filter = GetInput(context, node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);

  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);

  const RuntimeShape& input_shape = GetTensorShape(input);
  const RuntimeShape& filter_shape = GetTensorShape(filter);
  const RuntimeShape& output_shape = GetTensorShape(output);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int depth_multiplier = params.depth_multiplier;
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  const int pad_width = data->reference_op_data.padding.width;
  const int pad_height = data->reference_op_data.padding.height;

  int required_scratch = 0;
  // Dilation is currently not supported on HiFi 4 NN Library
  if ((params.dilation_width_factor == 1) &&
      (params.dilation_height_factor == 1)) {
    required_scratch = xa_nn_conv2d_depthwise_getsize(
        input_height, input_width, input_depth, filter_height, filter_width,
        depth_multiplier, stride_width, stride_height, pad_width, pad_height,
        output_height, output_width, PREC_ASYM8S, 0 /* NHWC */);
    TF_LITE_ENSURE(context, required_scratch > 0);
  }
  TF_LITE_ENSURE_OK(
      context, context->RequestScratchBufferInArena(
                   context, required_scratch, &data->scratch_tensor_index));
#endif  // defined(FUISON_F1)
  return kTfLiteOk;
}

#if defined(FUSION_F1)
TfLiteStatus EvalHifi4(TfLiteContext* context, TfLiteNode* node,
                       const TfLiteDepthwiseConvParams& params,
                       const OpData& data, const TfLiteEvalTensor* input,
                       const TfLiteEvalTensor* filter,
                       const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  // If dilation is not required use the optimized NN Library kernel.
  // Otherwise call the reference implementation.
  if ((params.dilation_width_factor == 1) &&
      (params.dilation_height_factor == 1)) {
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int pad_width = data.reference_op_data.padding.width;
    const int pad_height = data.reference_op_data.padding.height;
    const int depth_multiplier = params.depth_multiplier;
    const int32_t output_activation_min =
        data.reference_op_data.output_activation_min;
    const int32_t output_activation_max =
        data.reference_op_data.output_activation_max;
    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

    const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
    const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
    const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
    int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

    int32_t input_data_format = 0;
    int32_t output_data_format = 0;

    uint8_t* p_scratch = static_cast<uint8_t*>(
        context->GetScratchBuffer(context, data.scratch_tensor_index));

    for (int i = 0; i < batches; i++) {
      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s(
              &output_data[i * output_height * output_width * output_depth],
              filter_data,
              &input_data[i * input_height * input_width * input_depth],
              bias_data, input_height, input_width, input_depth, filter_height,
              filter_width, depth_multiplier, stride_width, stride_height,
              pad_width, pad_height, output_height, output_width,
              -data.reference_op_data.input_zero_point,
              data.reference_op_data.per_channel_output_multiplier,
              data.reference_op_data.per_channel_output_shift,
              data.reference_op_data.output_zero_point, input_data_format,
              output_data_format, p_scratch),
          0);
    }

    int out_length = batches * output_height * output_width * output_depth;
    TF_LITE_ENSURE_EQ(context,
                      xa_nn_vec_activation_min_max_8_8(
                          output_data, output_data, output_activation_min,
                          output_activation_max, out_length),
                      0);

    return kTfLiteOk;
  }

  reference_integer_ops::DepthwiseConvPerChannel(
      DepthwiseConvParamsQuantized(params, data.reference_op_data),
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
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
  const auto& op_data = *(reinterpret_cast<OpData*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kDepthwiseConvOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kDepthwiseConvBiasTensor)
          : nullptr;

#if defined(HIFIMINI)
  // Handle special case for streaming model.
  int* input_dims = input->dims->data;
  int* filter_dims = filter->dims->data;
  if (input_dims[0] == 1 && input_dims[1] == 4 && input_dims[2] == 1 &&
      input_dims[3] == 32 && filter_dims[0] == 1 && filter_dims[1] == 4 &&
      filter_dims[2] == 1 && filter_dims[3] == 32) {
    DepthwiseConv4x32MatchingInputAndFilterHifiMini(
        -op_data.reference_op_data.input_zero_point,
        op_data.reference_op_data.output_zero_point,
        std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max(),
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

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteInt8: {
#if defined(HIFIMINI)
      EvalHifiMini(
          DepthwiseConvParamsQuantized(params, op_data.reference_op_data),
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
      EvalHifi4(context, node, params, op_data, input, filter, bias, output);
#else
      reference_integer_ops::DepthwiseConvPerChannel(
          DepthwiseConvParamsQuantized(params, op_data.reference_op_data),
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

TfLiteRegistration Register_DEPTHWISE_CONV_2D() {
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
