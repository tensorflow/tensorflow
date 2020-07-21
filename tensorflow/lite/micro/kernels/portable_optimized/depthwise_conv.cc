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

#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))

namespace tflite {
namespace ops {
namespace micro {
namespace depthwise_conv {
namespace {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// Depthwise conv is quantized along dimension 3:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kDepthwiseConvQuantizedDimension = 3;

// Size of the cached buffer we'll be using to hold reordered weights.
constexpr int kReshapedFilterDataSize = 1 * 1024;

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

  // The precomputed sum of filters factor
  int32* sum_of_filters_factor;
};

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

    return tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift), num_channels);
  }
  return kTfLiteOk;
}

template <typename T>
inline void PrecomputeSumOfFiltersFactor(const int32* bias,
                                         const TfLiteTensor* filters,
                                         int32_t* sum_of_filters_factor,
                                         RuntimeShape filter_shape,
                                         int32_t input_offset,
                                         int32_t filter_offset = 0) {
  if (filters->type == kTfLiteInt8) {
    // Ensure that the filter offset is 0 in the signed integer case
    TFLITE_DCHECK_EQ(filter_offset, 0);
  }
  const T* filter_data = GetTensorData<T>(filters);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int num_filters = filter_shape.Dims(3);

  int filter_size = filter_width * filter_height;

  for (int out_channel = 0; out_channel < num_filters; ++out_channel) {
    int32_t sum_of_filter_factor = filter_size * filter_offset;

    for (int filter_index = out_channel;
         filter_index < filter_size * num_filters;
         filter_index += num_filters) {
      sum_of_filter_factor += filter_data[filter_index];
    }
    sum_of_filters_factor[out_channel] = sum_of_filter_factor * input_offset;

    if (bias) {
      sum_of_filters_factor[out_channel] += bias[out_channel];
    }
  }
}

}  // namespace

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

  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);

  const TfLiteType data_type = input->type;
  int width = SizeOfDimension(input, 2);
  int height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);

  // Per channel quantization is only needed for int8 inference. For other
  // quantized types, only a single scale and zero point is needed.
  const int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];
  // Dynamically allocate per-channel quantization parameters.
  TF_LITE_ENSURE_STATUS(context->AllocatePersistentBuffer(
      context, num_channels * sizeof(int32_t),
      reinterpret_cast<void**>(&data->per_channel_output_multiplier)));
  TF_LITE_ENSURE_STATUS(context->AllocatePersistentBuffer(
      context, num_channels * sizeof(int32_t),
      reinterpret_cast<void**>(&data->per_channel_output_shift)));

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

  if (filter->type == kTfLiteInt8 || filter->type == kTfLiteUInt8) {
    const TfLiteTensor* bias = GetInput(context, node, kBiasTensor);
    const int32* bias_data = GetTensorData<int32_t>(bias);

    const int32_t filter_offset = -filter->params.zero_point;
    RuntimeShape filter_shape = GetTensorShape(filter);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);

    void* raw;
    context->AllocatePersistentBuffer(context, sizeof(int32_t) * num_channels,
                                      &raw);
    data->sum_of_filters_factor = reinterpret_cast<int32_t*>(raw);

    // Precompute the sum of filters
    const int32_t input_offset = -input->params.zero_point;
    if (filter->type == kTfLiteUInt8) {
      PrecomputeSumOfFiltersFactor<uint8_t>(bias_data, filter, data->sum_of_filters_factor,
        filter_shape, input_offset, filter_offset);
    }
    else {
      PrecomputeSumOfFiltersFactor<int8_t>(bias_data, filter, data->sum_of_filters_factor,
        filter_shape, input_offset, 0);
    }
  }

  return CalculateOpData(context, node, params, width, height, filter_width,
                         filter_height, data_type, data);
}

static inline void DepthwiseConvOptimizedForFilterWidthEight(
    TfLiteContext* context, const DepthwiseParams& params, const OpData* data,
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

  static int16_t reshaped_filter_data[kReshapedFilterDataSize];
  const int needed_size =
      output_depth * filter_width * filter_height * input_depth;
  if (needed_size > kReshapedFilterDataSize) {
    TF_LITE_KERNEL_LOG(
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
          int16_t* reshaped_filter =
              reshaped_filter_data +
              Offset(reshaped_filter_shape, 0, oc, filter_y, filter_x);
          *reshaped_filter =
              static_cast<int16_t>(*current_filter) + filter_offset;
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
                int16* current_filter =
                    reshaped_filter_data + Offset(reshaped_filter_shape, 0, oc,
                                                  filter_y, filter_x_start);
                const uint32_t input_vals0 =
                    *reinterpret_cast<const uint32_t*>(current_input);
                current_input += 4;
                const int32_t filter_vals0 =
                    *reinterpret_cast<const int32_t*>(current_filter);
                current_filter += 2;
                const uint8 input_val0 = input_vals0 & 0xff;
                const int16 filter_val0 = filter_vals0 & 0xffff;
                acc += filter_val0 * input_val0;
                const uint8 input_val1 = (input_vals0 >> 8) & 0xff;
                const int16 filter_val1 = (filter_vals0 >> 16) & 0xffff;
                acc += filter_val1 * input_val1;

                const int32_t filter_vals1 =
                    *reinterpret_cast<const int32_t*>(current_filter);
                current_filter += 2;
                const uint8 input_val2 = (input_vals0 >> 16) & 0xff;
                const int16 filter_val2 = filter_vals1 & 0xffff;
                acc += filter_val2 * input_val2;
                const uint8 input_val3 = (input_vals0 >> 24) & 0xff;
                const int16 filter_val3 = (filter_vals1 >> 16) & 0xffff;
                acc += filter_val3 * input_val3;

                const uint32_t input_vals1 =
                    *reinterpret_cast<const uint32_t*>(current_input);
                const int32_t filter_vals2 =
                    *reinterpret_cast<const int32_t*>(current_filter);
                current_filter += 2;
                const uint8 input_val4 = input_vals1 & 0xff;
                const int16 filter_val4 = filter_vals2 & 0xffff;
                acc += filter_val4 * input_val4;
                const uint8 input_val5 = (input_vals1 >> 8) & 0xff;
                const int16 filter_val5 = (filter_vals2 >> 16) & 0xffff;
                acc += filter_val5 * input_val5;

                const int32_t filter_vals3 =
                    *reinterpret_cast<const int32_t*>(current_filter);
                const uint8 input_val6 = (input_vals1 >> 16) & 0xff;
                const int16 filter_val6 = filter_vals3 & 0xffff;
                acc += filter_val6 * input_val6;
                const uint8 input_val7 = (input_vals1 >> 24) & 0xff;
                const int16 filter_val7 = (filter_vals3 >> 16) & 0xffff;
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
}

inline void DepthwiseConv(
    TfLiteContext* context, const DepthwiseParams& params, const OpData* data,
    const RuntimeShape& input_shape, const uint8* input_data,
    const RuntimeShape& filter_shape, const uint8* filter_data,
    const RuntimeShape& bias_shape, const int32* bias_data,
    const RuntimeShape& output_shape, uint8* output_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_EQ(dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(dilation_height_factor, 1);
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

  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());
  int32 acc[output_depth];
  for (int batch = 0; batch < batches; ++batch) {
    const uint32_t input_offset0 = in_dims[1] * batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      const int32_t ker_y_start = MAX(0, -in_y_origin);
      const int32_t ker_y_end = MIN(filter_height, input_height - in_y_origin);

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int32_t ker_x_start = MAX(0, -in_x_origin);
        const int32_t ker_x_end = MIN(filter_width, input_width - in_x_origin);

        for (int i = 0; i < output_depth; ++i) {
          acc[i] = 0;
        }

        for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          const uint32_t input_offset1 = in_dims[2] * (input_offset0 + in_y);
          const uint32_t filter_offset1 = fi_dims[2] * filter_y;
          for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor * filter_x;
            const uint32_t input_offset2 = in_dims[3] * (input_offset1 + in_x);
            const uint32_t filter_offset2 =
                fi_dims[3] * (filter_x + filter_offset1);

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              for (int m = 0; m < depth_multiplier; ++m) {
                const int output_channel = m + in_channel * depth_multiplier;
                int32 input_val = input_data[input_offset2 + in_channel];
                int32 filter_val = filter_data[filter_offset2 + output_channel];
                acc[output_channel] +=
                    (input_val + input_offset) * (filter_val + filter_offset);
              }
            }
          }
        }
        uint32_t out_base = Offset(output_shape, batch, out_y, out_x, 0);
        for (int i = 0; i < output_depth; i++) {
          if (bias_data) {
            acc[i] += bias_data[i];
          }
          acc[i] = MultiplyByQuantizedMultiplier(acc[i], output_multiplier,
                                                 output_shift);
          acc[i] += output_offset;
          acc[i] = std::max(acc[i], output_activation_min);
          acc[i] = std::min(acc[i], output_activation_max);
          output_data[out_base + i] = static_cast<uint8_t>(acc[i]);
        }
      }
    }
  }
}

inline void DepthwiseConvNoPadding(
    TfLiteContext* context, const DepthwiseParams& params, const OpData* data,
    const RuntimeShape& input_shape, const uint8* input_data,
    const RuntimeShape& filter_shape, const uint8* filter_data,
    const RuntimeShape& bias_shape, const int32* bias_data,
    const RuntimeShape& output_shape, uint8* output_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_EQ(dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(dilation_height_factor, 1);
  const int depth_multiplier = params.depth_multiplier;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
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

  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());
  int32 acc[output_depth];
  for (int batch = 0; batch < batches; ++batch) {
    const uint32_t input_offset0 = in_dims[1] * batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height);
      const int32_t ker_y_start = MAX(0, -in_y_origin);
      const int32_t ker_y_end = MIN(filter_height, input_height - in_y_origin);

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width);
        const int32_t ker_x_start = MAX(0, -in_x_origin);
        const int32_t ker_x_end = MIN(filter_width, input_width - in_x_origin);

        for (int i = 0; i < output_depth; ++i) {
          acc[i] = 0;
        }

        for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          const uint32_t input_offset1 = in_dims[2] * (input_offset0 + in_y);
          const uint32_t filter_offset1 = fi_dims[2] * filter_y;
          for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor * filter_x;
            const uint32_t input_offset2 = in_dims[3] * (input_offset1 + in_x);
            const uint32_t filter_offset2 =
                fi_dims[3] * (filter_x + filter_offset1);

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              for (int m = 0; m < depth_multiplier; ++m) {
                const int output_channel = m + in_channel * depth_multiplier;
                int32 input_val = input_data[input_offset2 + in_channel];
                int32 filter_val = filter_data[filter_offset2 + output_channel];
                acc[output_channel] += input_val * (filter_val + filter_offset);
              }
            }
          }
        }
        uint32_t out_base = Offset(output_shape, batch, out_y, out_x, 0);
        for (int i = 0; i < output_depth; i++) {
          acc[i] += data->sum_of_filters_factor[i];
          acc[i] = MultiplyByQuantizedMultiplier(acc[i], output_multiplier,
                                                 output_shift);
          acc[i] += output_offset;
          acc[i] = std::max(acc[i], output_activation_min);
          acc[i] = std::min(acc[i], output_activation_max);
          output_data[out_base + i] = static_cast<uint8_t>(acc[i]);
        }
      }
    }
  }
}

void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteDepthwiseConvParams* params, const OpData* data,
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
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  tflite::reference_ops::DepthwiseConv(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(filter), GetTensorData<float>(filter),
      GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
      GetTensorData<float>(output));
}

void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params,
                             const OpData* data, const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output) {
  const int32* output_multiplier = data->per_channel_output_multiplier;
  const int32* output_shift = data->per_channel_output_shift;

  const RuntimeShape& input_shape = GetTensorShape(input);
  const int8* input_data = GetTensorData<int8>(input);
  const RuntimeShape& filter_shape = GetTensorShape(filter);
  const int8* filter_data = GetTensorData<int8>(filter);
  const RuntimeShape& bias_shape = GetTensorShape(bias);
  const int32* bias_data = GetTensorData<int32>(bias);
  const RuntimeShape& output_shape = GetTensorShape(output);
  int8* output_data = GetTensorData<int8>(output);

  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;
  TFLITE_DCHECK_EQ(dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(dilation_height_factor, 1);

  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int depth_multiplier = params->depth_multiplier;
  const int32 input_offset = -input->params.zero_point;
  const int32 output_offset = output->params.zero_point;
  const int32 output_activation_min = std::numeric_limits<int8_t>::min();
  const int32 output_activation_max = std::numeric_limits<int8_t>::max();

  // Check dimensions of the tensors.
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

  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());
  int32 acc[output_depth];
  for (int batch = 0; batch < batches; ++batch) {
    const uint32_t input_offset0 = in_dims[1] * batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      const int32_t ker_y_start = MAX(0, -in_y_origin);
      const int32_t ker_y_end = MIN(filter_height, input_height - in_y_origin);

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int32_t ker_x_start = MAX(0, -in_x_origin);
        const int32_t ker_x_end = MIN(filter_width, input_width - in_x_origin);

        for (int i = 0; i < output_depth; ++i) {
          acc[i] = 0;
        }

        for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          const uint32_t input_offset1 = in_dims[2] * (input_offset0 + in_y);
          const uint32_t filter_offset1 = fi_dims[2] * filter_y;
          for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor * filter_x;
            const uint32_t input_offset2 = in_dims[3] * (input_offset1 + in_x);
            const uint32_t filter_offset2 =
                fi_dims[3] * (filter_x + filter_offset1);

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              for (int m = 0; m < depth_multiplier; ++m) {
                const int out_channel = m + in_channel * depth_multiplier;
                int32 input_val = input_data[input_offset2 + in_channel];
                int32 filter_val = filter_data[filter_offset2 + out_channel];
                acc[out_channel] += (input_val + input_offset) * filter_val;
              }
            }
          }
        }
        uint32_t out_base = Offset(output_shape, batch, out_y, out_x, 0);
        for (int i = 0; i < output_depth; i++) {
          if (bias) {
            acc[i] += bias_data[i];
          }
          acc[i] = MultiplyByQuantizedMultiplier(acc[i], output_multiplier[i],
                                                 output_shift[i]);
          acc[i] += output_offset;
          acc[i] = std::max(acc[i], output_activation_min);
          acc[i] = std::min(acc[i], output_activation_max);
          output_data[out_base + i] = static_cast<int8_t>(acc[i]);
        }
      }
    }
  }
}

void EvalQuantizedPerChannelNoPadding(
    TfLiteContext* context, TfLiteNode* node, TfLiteDepthwiseConvParams* params,
    const OpData* data, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output) {
  const int32* output_multiplier = data->per_channel_output_multiplier;
  const int32* output_shift = data->per_channel_output_shift;

  const RuntimeShape& input_shape = GetTensorShape(input);
  const int8* input_data = GetTensorData<int8>(input);
  const RuntimeShape& filter_shape = GetTensorShape(filter);
  const int8* filter_data = GetTensorData<int8>(filter);
  const RuntimeShape& bias_shape = GetTensorShape(bias);
  const RuntimeShape& output_shape = GetTensorShape(output);
  int8* output_data = GetTensorData<int8>(output);

  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;
  TFLITE_DCHECK_EQ(dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(dilation_height_factor, 1);

  const int depth_multiplier = params->depth_multiplier;
  const int32 output_offset = output->params.zero_point;
  const int32 output_activation_min = std::numeric_limits<int8_t>::min();
  const int32 output_activation_max = std::numeric_limits<int8_t>::max();

  // Check dimensions of the tensors.
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

  const int* in_dims =
      reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims =
      reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());
  int32 acc[output_depth];
  for (int batch = 0; batch < batches; ++batch) {
    const uint32_t input_offset0 = in_dims[1] * batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height);
      const int32_t ker_y_start = MAX(0, -in_y_origin);
      const int32_t ker_y_end = MIN(filter_height, input_height - in_y_origin);

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width);
        const int32_t ker_x_start = MAX(0, -in_x_origin);
        const int32_t ker_x_end = MIN(filter_width, input_width - in_x_origin);

        for (int i = 0; i < output_depth; ++i) {
          acc[i] = 0;
        }

        for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          const uint32_t input_offset1 = in_dims[2] * (input_offset0 + in_y);
          const uint32_t filter_offset1 = fi_dims[2] * filter_y;
          for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor * filter_x;
            const uint32_t input_offset2 = in_dims[3] * (input_offset1 + in_x);
            const uint32_t filter_offset2 =
                fi_dims[3] * (filter_x + filter_offset1);

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              for (int m = 0; m < depth_multiplier; ++m) {
                const int out_channel = m + in_channel * depth_multiplier;
                int32 input_val = input_data[input_offset2 + in_channel];
                int32 filter_val = filter_data[filter_offset2 + out_channel];
                acc[out_channel] += input_val * filter_val;
              }
            }
          }
        }
        uint32_t out_base = Offset(output_shape, batch, out_y, out_x, 0);
        for (int i = 0; i < output_depth; i++) {
          acc[i] += data->sum_of_filters_factor[i];
          acc[i] = MultiplyByQuantizedMultiplier(acc[i], output_multiplier[i],
                                                 output_shift[i]);
          acc[i] += output_offset;
          acc[i] = std::max(acc[i], output_activation_min);
          acc[i] = std::min(acc[i], output_activation_max);
          output_data[out_base + i] = static_cast<int8_t>(acc[i]);
        }
      }
    }
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias =
      (NumInputs(node) == 3) ? GetInput(context, node, kBiasTensor) : nullptr;

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      EvalFloat(context, node, params, &data, input, filter, bias, output);
      break;
    case kTfLiteInt8: {
      const int dilation_width_factor = params->dilation_width_factor;
      const int dilation_height_factor = params->dilation_height_factor;
      const int pad_width = data.padding.width;
      const int pad_height = data.padding.height;
      const int pad_width_offset = data.padding.width_offset;
      const int pad_height_offset = data.padding.height_offset;

      // Check if optimized filter width is used
      const bool use_optimized_filter_width =
          (GetTensorShape(filter).Dims(0) != 1);

      DepthwiseParams op_params;
      op_params.padding_type = PaddingType::kSame;
      op_params.padding_values.width = pad_width;
      op_params.padding_values.height = pad_height;
      op_params.padding_values.width_offset = pad_width_offset;
      op_params.padding_values.height_offset = pad_height_offset;
      op_params.stride_width = params->stride_width;
      op_params.stride_height = params->stride_height;
      op_params.dilation_width_factor = params->dilation_width_factor;
      op_params.dilation_height_factor = params->dilation_height_factor;
      op_params.depth_multiplier = params->depth_multiplier;
      op_params.input_offset = -input->params.zero_point;
      op_params.weights_offset = 0;
      op_params.output_offset = output->params.zero_point;
      op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
      op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();

      if ((dilation_width_factor != 1) || (dilation_height_factor != 1) ||
          use_optimized_filter_width) {
        // If dilation is used, the reference implementation
        reference_integer_ops::DepthwiseConvPerChannel(
            op_params, data.per_channel_output_multiplier,
            data.per_channel_output_shift, GetTensorShape(input),
            GetTensorData<int8>(input), GetTensorShape(filter),
            GetTensorData<int8>(filter), GetTensorShape(bias),
            GetTensorData<int32>(bias), GetTensorShape(output),
            GetTensorData<int8>(output));
      } else if (pad_width != 0 || pad_height != 0 || pad_width_offset != 0 || pad_height_offset != 0) {
        // Use the version that can handle padding
        EvalQuantizedPerChannel(context, node, params, &data, input, filter,
                                bias, output);
      } else {
        // Use the optimized version without padding
        EvalQuantizedPerChannelNoPadding(context, node, params, &data, input,
                                         filter, bias, output);
      }
      break;
    }
    case kTfLiteUInt8: {
      const int dilation_width_factor = params->dilation_width_factor;
      const int dilation_height_factor = params->dilation_height_factor;
      const int pad_width = data.padding.width;
      const int pad_height = data.padding.height;
      const int pad_width_offset = data.padding.width_offset;
      const int pad_height_offset = data.padding.height_offset;

      const int32_t input_offset = -input->params.zero_point;
      const int32_t filter_offset = -filter->params.zero_point;
      const int32_t output_offset = output->params.zero_point;

      // Check if optimized filter width is used
      const bool use_optimized_filter_width =
          (GetTensorShape(filter).Dims(0) != 1);

      tflite::DepthwiseParams op_params;
      // Padding type is ignored, but still set.
      op_params.padding_type = PaddingType::kSame;
      op_params.padding_values.width = data.padding.width;
      op_params.padding_values.height = data.padding.height;
      op_params.padding_values.width_offset = pad_width_offset;
      op_params.padding_values.height_offset = pad_height_offset;
      op_params.stride_width = params->stride_width;
      op_params.stride_height = params->stride_height;
      op_params.dilation_width_factor = params->dilation_width_factor;
      op_params.dilation_height_factor = params->dilation_height_factor;
      op_params.depth_multiplier = params->depth_multiplier;
      op_params.quantized_activation_min = data.output_activation_min;
      op_params.quantized_activation_max = data.output_activation_max;
      op_params.input_offset = input_offset;
      op_params.weights_offset = filter_offset;
      op_params.output_offset = output_offset;
      op_params.output_multiplier = data.output_multiplier;
      // Legacy ops used mixed left and right shifts. Now all are
      // +ve-means-left.
      op_params.output_shift = -data.output_shift;

      const int filter_width = GetTensorShape(filter).Dims(2);
      const int input_depth = GetTensorShape(input).Dims(3);
      const int output_depth = GetTensorShape(filter).Dims(3);
      const int filter_height = GetTensorShape(filter).Dims(1);
      const int needed_size =
          output_depth * filter_width * filter_height * input_depth;

      if ((dilation_width_factor != 1) ||
                 (dilation_height_factor != 1) || use_optimized_filter_width) {
        // If dilation is used, then the reference implementation is used
        tflite::reference_ops::DepthwiseConv(
            op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(filter), GetTensorData<uint8_t>(filter),
            GetTensorShape(bias), GetTensorData<int32_t>(bias),
            GetTensorShape(output), GetTensorData<uint8_t>(output));

      } else if ((filter_width == 8) && (input_offset == 0) &&
                 (input_depth == 1) &&
                 (needed_size <= kReshapedFilterDataSize)) {
        // Use the optimized version if possible
        DepthwiseConvOptimizedForFilterWidthEight(
            context, op_params, &data, GetTensorShape(input),
            GetTensorData<uint8_t>(input), GetTensorShape(filter),
            GetTensorData<uint8_t>(filter), GetTensorShape(bias),
            GetTensorData<int32_t>(bias), GetTensorShape(output),
            GetTensorData<uint8_t>(output));

      } else if (pad_width != 0 || pad_height != 0 || pad_width_offset != 0 || pad_height_offset != 0) {
        // Use the version(s) that can handle padding if padding is used
        DepthwiseConv(
            context, op_params, &data, GetTensorShape(input),
            GetTensorData<uint8_t>(input), GetTensorShape(filter),
            GetTensorData<uint8_t>(filter), GetTensorShape(bias),
            GetTensorData<int32_t>(bias), GetTensorShape(output),
            GetTensorData<uint8_t>(output));
      } else {
        // Use the optimized version without padding
        DepthwiseConvNoPadding(
            context, op_params, &data, GetTensorShape(input),
            GetTensorData<uint8_t>(input), GetTensorShape(filter),
            GetTensorData<uint8_t>(filter), GetTensorShape(bias),
            GetTensorData<int32_t>(bias), GetTensorShape(output),
            GetTensorData<uint8_t>(output));
      }
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace depthwise_conv

TfLiteRegistration Register_DEPTHWISE_CONV_2D() {
  return {/*init=*/depthwise_conv::Init,
          /*free=*/nullptr,
          /*prepare=*/depthwise_conv::Prepare,
          /*invoke=*/depthwise_conv::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
