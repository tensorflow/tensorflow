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

#include <xtensa/tie/xt_hifi2.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/xtensa_hifimini/fixedpoint_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa_hifimini/utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace depthwise_conv {
namespace xtensa {
namespace hifimini {

inline void DepthwiseConvPerChannel(
    const DepthwiseParams& params, const int32* output_multiplier,
    const int32* output_shift, const RuntimeShape& input_shape,
    const int8* input_data, const RuntimeShape& filter_shape,
    const int8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    int8* output_data) {
  // Get parameters.
  // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32 input_offset = params.input_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;

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

  ae_p24x2s input_offset_24x2 = AE_CONVERT_INT32_24x2(input_offset);
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
                  int32 input_val = input_data[input_idx];

                  // Find current filter index, minus 2 for Xtensa load
                  // alignments:
                  int filter_idx =
                      ((filter_y)*filter_width + filter_x) * filter_depth +
                      (output_channel);
                  int32 filter_val = filter_data[filter_idx];

                  // Load 8bit value as int32 into a 24x24 register and right
                  // shift into 24bit space. Note: value is duplicated in the HH
                  // and LL register - but all calculations are done on the HH
                  // side.
                  ae_p24x2s input_val_24x2 = AE_CONVERT_INT32_24x2(input_val);

                  // Add input offset (24bit aligned):
                  input_val_24x2 =
                      AE_P24S_ADDS_P24X2S(input_val_24x2, input_offset_24x2);

                  // Load filter 8bit value into 24bit alignment:
                  ae_p24x2s filter_val_24x2 = AE_CONVERT_INT32_24x2(filter_val);

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
            acc_56 = micro::xtensa::hifimini::MultiplyByQuantizedMultiplier(
                acc_24x2, output_multiplier[output_channel],
                output_shift[output_channel]);

            // Shift from 48bit aligned to 32bit:
            acc_56 = AE_Q56S_SLAI(acc_56, 16);

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

}  // namespace hifimini
}  // namespace xtensa

namespace {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kMaxChannels = 256;

// Depthwise conv is quantized along dimension 3:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kDepthwiseConvQuantizedDimension = 3;

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  // TODO(b/141139247): Allocate these dynamically when possible.
  int32_t per_channel_output_multiplier[kMaxChannels];
  int32_t per_channel_output_shift[kMaxChannels];

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};

// These constants represent constants specific to the music detect model.
// They exist until (b/132070898) is fixed.
static const int kMaxOpDataSize = 6;
static int kStaticOpDataCounter = 0;
static OpData kStaticOpData[kMaxOpDataSize];

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params, int width,
                             int height, int filter_width, int filter_height,
                             const TfLiteType data_type, OpData* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  int unused_output_height, unused_output_width;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, height, width,
      filter_height, filter_width, params->padding, &unused_output_height,
      &unused_output_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];

    // TODO(b/148610881): Consider calculating quantized params at int24
    // calculations:
    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift), num_channels));
  }
  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias =
      (NumInputs(node) == 3) ? GetInput(context, node, kBiasTensor) : nullptr;

  // TODO(b/132070898): Use statically slotted OpData structures until a
  // scratch memory API is ready.
  OpData* op_data = &kStaticOpData[kStaticOpDataCounter++];
  node->user_data = op_data;

  const TfLiteType data_type = input->type;
  int width = SizeOfDimension(input, 2);
  int height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);

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
    TF_LITE_ENSURE(
        context, affine_quantization->scale->size == 1 ||
                     affine_quantization->scale->size ==
                         filter->dims->data[kDepthwiseConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, node, params, width, height,
                                        filter_width, filter_height, data_type,
                                        op_data));
  return kTfLiteOk;
}

void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params, OpData* data,
                             const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output) {
  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = 0;
  op_params.output_offset = output->params.zero_point;
  // TODO(b/130439627): Use calculated value for clamping.
  op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();

  xtensa::hifimini::DepthwiseConvPerChannel(
      op_params, data->per_channel_output_multiplier,
      data->per_channel_output_shift, GetTensorShape(input),
      GetTensorData<int8>(input), GetTensorShape(filter),
      GetTensorData<int8>(filter), GetTensorShape(bias),
      GetTensorData<int32>(bias), GetTensorShape(output),
      GetTensorData<int8>(output));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias =
      (NumInputs(node) == 3) ? GetInput(context, node, kBiasTensor) : nullptr;

  // TODO(b/147710241): Consider whether float conv and quantized conv should be
  // separate ops to avoid dispatch overhead here.
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteInt8:
      EvalQuantizedPerChannel(context, node, params, op_data, input, filter,
                              bias, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace depthwise_conv

TfLiteRegistration* Register_DEPTHWISE_CONV_2D() {
  static TfLiteRegistration r = {depthwise_conv::Init, depthwise_conv::Free,
                                 depthwise_conv::Prepare, depthwise_conv::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
