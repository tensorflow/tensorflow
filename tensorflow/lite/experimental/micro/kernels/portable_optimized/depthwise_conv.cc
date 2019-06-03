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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"

namespace tflite {
namespace ops {
namespace micro {
namespace depthwise_conv {
namespace {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// Size of the cached buffer we'll be using to hold reordered weights.
constexpr int kReshapedFilterDataSize = 1 * 1024;

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params, int width,
                             int height, int filter_width, int filter_height,
                             int out_width, int out_height,
                             const TfLiteType data_type, OpData* data) {
  data->padding.height = ComputePadding(params->stride_height, 1, height,
                                        filter_height, out_height);
  data->padding.width =
      ComputePadding(params->stride_width, 1, width, filter_width, out_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    int exponent;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
    data->output_shift = -exponent;
    CalculateActivationRangeUint8(params->activation, output,
                                  &data->output_activation_min,
                                  &data->output_activation_max);
  }
  return kTfLiteOk;
}

// Specialized implementation of the depthwise convolution operation designed to
// work with the particular filter width of eight used by the default micro
// speech sample code. It uses 1KB of RAM to hold reordered weight parameters,
// converted from TFLite's NHWC format to NCHW format, and expressed as signed
// eight bit integers, rather than unsigned. Care must be taken when calling
// this not to use it for more than one node since there's only a single static
// buffer holding the weights. You should use this implementation if depthwise
// convolutions are a performance bottleneck, you have a layer that meets the
// parameter requirements, and the extra RAM usage and additional code size are
// not an issue.
static inline void DepthwiseConvOptimizedForFilterWidthEight(
    TfLiteContext* context, const DepthwiseParams& params,
    const RuntimeShape& input_shape, const uint8* input_data,
    const RuntimeShape& filter_shape, const uint8* filter_data,
    const RuntimeShape& bias_shape, const int32* bias_data,
    const RuntimeShape& output_shape, uint8* output_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
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

  static int8_t reshaped_filter_data[kReshapedFilterDataSize];
  const int needed_size =
      output_depth * filter_width * filter_height * input_depth;
  if (needed_size > kReshapedFilterDataSize) {
    context->ReportError(
        context,
        "Size too large for reshaped weight buffer (%d needed, %d available)",
        needed_size, kReshapedFilterDataSize);
    return;
  }

  RuntimeShape reshaped_filter_shape;
  reshaped_filter_shape.BuildFrom(
      {1, output_depth, filter_height, filter_width});

  // If this is the first time through, repack the weights into a cached buffer
  // so that they can be accessed sequentially.
  static bool is_reshaped_filter_initialized = false;
  if (!is_reshaped_filter_initialized) {
    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
      for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
        for (int oc = 0; oc < output_depth; ++oc) {
          const uint8* current_filter =
              filter_data + Offset(filter_shape, 0, filter_y, filter_x, oc);
          int8* reshaped_filter =
              reshaped_filter_data +
              Offset(reshaped_filter_shape, 0, oc, filter_y, filter_x);
          *reshaped_filter = (int32_t)(*current_filter) + filter_offset;
        }
      }
    }
    is_reshaped_filter_initialized = true;
  }

  for (int b = 0; b < batches; ++b) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int ic = 0; ic < input_depth; ++ic) {
          for (int m = 0; m < depth_multiplier; m++) {
            const int oc = m + ic * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32 acc = 0;
            int in_y_start = in_y_origin;
            int filter_y_start = 0;
            if (in_y_origin < 0) {
              in_y_start = 0;
              filter_y_start = 0 - in_y_origin;
            }
            int filter_y_end = filter_height;
            if ((in_y_origin + filter_height) >= input_height) {
              filter_y_end -= (in_y_origin + filter_height) - input_height;
            }
            int in_y = in_y_start;
            int in_x_start = in_x_origin;
            int filter_x_start = 0;
            bool is_out_of_x_bounds = false;
            if (in_x_origin < 0) {
              in_x_start = 0;
              filter_x_start = 0 - in_x_origin;
              is_out_of_x_bounds = true;
            }
            int filter_x_end = filter_width;
            if ((in_x_origin + filter_width) >= input_width) {
              filter_x_end -= (in_x_origin + filter_width) - input_width;
              is_out_of_x_bounds = true;
            }
            for (int filter_y = filter_y_start; filter_y < filter_y_end;
                 ++filter_y, ++in_y) {
              const uint8* current_input =
                  input_data + Offset(input_shape, b, in_y, in_x_start, ic);
              if ((filter_width == 8) && !is_out_of_x_bounds) {
                int8* current_filter =
                    reshaped_filter_data + Offset(reshaped_filter_shape, 0, oc,
                                                  filter_y, filter_x_start);
                const uint32_t input_vals0 =
                    *reinterpret_cast<const uint32_t*>(current_input);
                current_input += 4;
                const int32_t filter_vals0 =
                    *reinterpret_cast<const int32_t*>(current_filter);
                current_filter += 4;
                const uint8 input_val0 = input_vals0 & 0xff;
                const int8 filter_val0 = filter_vals0 & 0xff;
                acc += filter_val0 * input_val0;
                const uint8 input_val1 = (input_vals0 >> 8) & 0xff;
                const int8 filter_val1 = (filter_vals0 >> 8) & 0xff;
                acc += filter_val1 * input_val1;
                const uint8 input_val2 = (input_vals0 >> 16) & 0xff;
                const int8 filter_val2 = (filter_vals0 >> 16) & 0xff;
                acc += filter_val2 * input_val2;
                const uint8 input_val3 = (input_vals0 >> 24) & 0xff;
                const int8 filter_val3 = (filter_vals0 >> 24) & 0xff;
                acc += filter_val3 * input_val3;

                const uint32_t input_vals1 =
                    *reinterpret_cast<const uint32_t*>(current_input);
                const int32_t filter_vals1 =
                    *reinterpret_cast<const int32_t*>(current_filter);
                const uint8 input_val4 = input_vals1 & 0xff;
                const int8 filter_val4 = filter_vals1 & 0xff;
                acc += filter_val4 * input_val4;
                const uint8 input_val5 = (input_vals1 >> 8) & 0xff;
                const int8 filter_val5 = (filter_vals1 >> 8) & 0xff;
                acc += filter_val5 * input_val5;
                const uint8 input_val6 = (input_vals1 >> 16) & 0xff;
                const int8 filter_val6 = (filter_vals1 >> 16) & 0xff;
                acc += filter_val6 * input_val6;
                const uint8 input_val7 = (input_vals1 >> 24) & 0xff;
                const int8 filter_val7 = (filter_vals1 >> 24) & 0xff;
                acc += filter_val7 * input_val7;
              } else {
                const uint8* current_filter =
                    filter_data +
                    Offset(filter_shape, 0, filter_y, filter_x_start, oc);
                for (int filter_x = filter_x_start; filter_x < filter_x_end;
                     ++filter_x) {
                  int32 input_val = *current_input;
                  current_input += input_depth;
                  int32 filter_val = *current_filter;
                  current_filter += output_depth;
                  acc +=
                      (filter_val + filter_offset) * (input_val + input_offset);
                }
              }
            }
            if (bias_data) {
              acc += bias_data[oc];
            }
            acc = reference_ops::depthwise_conv::DepthwiseConvRound<
                DepthwiseConvOutputRounding::kAwayFromZero>(
                acc, output_multiplier, output_shift);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_shape, b, out_y, out_x, oc)] =
                static_cast<uint8>(acc);
          }
        }
      }
    }
  }
}  // namespace

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteDepthwiseConvParams* params, OpData* data,
               const TfLiteTensor* input, const TfLiteTensor* filter,
               const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = 1;
  op_params.dilation_height_factor = 1;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  tflite::reference_ops::DepthwiseConv(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(filter), GetTensorData<float>(filter),
      GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
      GetTensorData<float>(output));
}

void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   TfLiteDepthwiseConvParams* params, OpData* data,
                   const TfLiteTensor* input, const TfLiteTensor* filter,
                   const TfLiteTensor* bias, TfLiteTensor* output) {
  const int32_t input_offset = -input->params.zero_point;
  const int32_t filter_offset = -filter->params.zero_point;
  const int32_t output_offset = output->params.zero_point;

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = 1;
  op_params.dilation_height_factor = 1;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = -data->output_shift;

  // Figure out if we can use the optimized path for this set of parameters.
  const int filter_width = GetTensorShape(filter).Dims(2);
  const int input_depth = GetTensorShape(input).Dims(3);
  const int output_depth = GetTensorShape(filter).Dims(3);
  const int filter_height = GetTensorShape(filter).Dims(1);
  const int needed_size =
      output_depth * filter_width * filter_height * input_depth;
  bool use_optimized_path = false;
  if ((filter_width == 8) && (input_offset == 0) && (filter_offset == -127) &&
      (input_depth == 1) && (needed_size <= kReshapedFilterDataSize)) {
    // FIXME(petewarden) - We need a more robust way of handling this, ideally
    // with an allocation mechanism available through the context API.
    // Use the address of the node as a proxy for its identity, since we need
    // to ensure the weight values are consistent between calls, and there's
    // no easy way to do that quickly other than relying on the identity of
    // the owning node.
    static TfLiteNode* initialized_node_address = node;
    if (initialized_node_address == node) {
      use_optimized_path = true;
    } else {
      static bool has_warned = false;
      if (!has_warned) {
        context->ReportError(
            context,
            "Multiple depthwise conv ops match optimization parameters, but "
            "only the first will use the fast path, because there's only one "
            "RAM cache available");
        has_warned = true;
      }
    }
  }
  if (use_optimized_path) {
    DepthwiseConvOptimizedForFilterWidthEight(
        context, op_params, GetTensorShape(input),
        GetTensorData<uint8_t>(input), GetTensorShape(filter),
        GetTensorData<uint8_t>(filter), GetTensorShape(bias),
        GetTensorData<int32_t>(bias), GetTensorShape(output),
        GetTensorData<uint8_t>(output));
  } else {
    tflite::reference_ops::DepthwiseConv(
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(filter), GetTensorData<uint8_t>(filter),
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<uint8_t>(output));
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias =
      (NumInputs(node) == 3) ? GetInput(context, node, kBiasTensor) : nullptr;

  const TfLiteType data_type = input->type;
  int width = SizeOfDimension(input, 2);
  int height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);
  int out_width = ComputeOutSize(params->padding, width, filter_width,
                                 params->stride_width);
  int out_height = ComputeOutSize(params->padding, height, filter_height,
                                  params->stride_height);
  OpData local_data_object;
  OpData* data = &local_data_object;
  TF_LITE_ENSURE_STATUS(CalculateOpData(context, node, params, width, height,
                                        filter_width, filter_height, out_width,
                                        out_height, data_type, data));

  // TODO(aselle): Consider whether float conv and quantized conv should be
  // separate ops to avoid dispatch overhead here.
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      EvalFloat(context, node, params, data, input, filter, bias, output);
      break;
    case kTfLiteUInt8:
      EvalQuantized(context, node, params, data, input, filter, bias, output);
      break;
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           input->type);
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
