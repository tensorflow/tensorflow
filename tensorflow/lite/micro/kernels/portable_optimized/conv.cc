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

// PORTABLE OPTIMIZED

#include "tensorflow/lite/kernels/internal/reference/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

#define MAX(A,B) ((A) > (B) ? (A) : (B))
#define MIN(A,B) ((A) < (B) ? (A) : (B))

namespace tflite {
namespace ops {
namespace micro {
namespace conv {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kMaxChannels = 256;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kConvQuantizedDimension = 0;

// This file has 2 implementation of Conv.

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift (allocated dynamically).
  int32_t *per_channel_output_multiplier;
  int32_t *per_channel_output_shift;

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;

  // The precomputed sum of filters factor
  int32 *sum_of_filters_factor;
};

inline PaddingType RuntimePaddingType(TfLitePadding padding) {
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

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

    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift),
        output_channels));
  }
  return kTfLiteOk;
}

template<typename T>
inline void PrecomputeSumOfFiltersFactor(const int32* bias, const TfLiteTensor* filters, int32_t *sum_of_filters_factor,
		RuntimeShape filter_shape, int32_t input_offset, int32_t filter_offset=0) {
	if (filters->type == kTfLiteInt8) {
		// Ensure that the filter offset is 0 in the signed integer case
		TFLITE_DCHECK_EQ(filter_offset, 0);
	}
	const T* filter_data = GetTensorData<T>(filters);
	const int filter_height = filter_shape.Dims(1);
	const int filter_width = filter_shape.Dims(2);
	const int input_depth = filter_shape.Dims(3);
	const int output_depth = filter_shape.Dims(0);

	int filter_size = filter_width * filter_height * input_depth;

	for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
		int32_t sum_of_filter_factor = filter_size * filter_offset;

    for (int filter_index = filter_size * out_channel; filter_index < filter_size + filter_size * out_channel; ++filter_index) {
      sum_of_filter_factor += filter_data[filter_index];
		}
		sum_of_filters_factor[out_channel] = sum_of_filter_factor * input_offset;
		if (bias) {
			sum_of_filters_factor[out_channel] += bias[out_channel];
		}
	}
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
	void* raw;
	TfLiteStatus allocation_success = context->AllocatePersistentBuffer(context, sizeof(OpData), &raw);
	TFLITE_DCHECK_EQ(allocation_success, kTfLiteOk);
	OpData* data = reinterpret_cast<OpData*>(raw);
	*data = {};
	return raw;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
	OpData* data = reinterpret_cast<OpData*>(node->user_data);
	auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);

	const TfLiteTensor* filters = GetInput(context, node, kFilterTensor);
	const TfLiteTensor* input = GetInput(context, node, kInputTensor);
	TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  int input_width = input->dims->data[2];
  int input_height = input->dims->data[1];
  int filter_width = filters->dims->data[2];
  int filter_height = filters->dims->data[1];
  int output_width = output->dims->data[2];
  int output_height = output->dims->data[1];

  if (filters->type == kTfLiteInt8 || filters->type == kTfLiteUInt8) {

    const TfLiteTensor* bias = GetInput(context, node, kBiasTensor);
    const int32* bias_data = GetTensorData<int32_t>(bias);

    const int32_t filter_offset = -filters->params.zero_point;
    RuntimeShape filter_shape = GetTensorShape(filters);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);

    const int output_depth = filter_shape.Dims(0);
    TF_LITE_ENSURE(context, output_depth <= kMaxChannels);

    void* raw;
    context->AllocatePersistentBuffer(context, sizeof(int32_t) * output_depth, &raw);
    data->sum_of_filters_factor = reinterpret_cast<int32_t*>(raw);

    context->AllocatePersistentBuffer(context, sizeof(int32_t) * output_depth, &raw);
    data->per_channel_output_multiplier = reinterpret_cast<int32_t*>(raw);

    context->AllocatePersistentBuffer(context, sizeof(int32_t) * output_depth, &raw);
    data->per_channel_output_shift = reinterpret_cast<int32_t*>(raw);

    // Precompute the sum of filters
    const int32_t input_offset = -input->params.zero_point;
    if (filters->type == kTfLiteUInt8) {
      PrecomputeSumOfFiltersFactor<uint8_t>(bias_data, filters, data->sum_of_filters_factor,
        filter_shape, input_offset, filter_offset);
    }
    else {
      PrecomputeSumOfFiltersFactor<int8_t>(bias_data, filters, data->sum_of_filters_factor,
        filter_shape, input_offset, 0);
    }
  }

  TF_LITE_ENSURE_STATUS(CalculateOpData(
              context, node, params, input_width, input_height, filter_width,
              filter_height, output_width, output_height, input->type, data));

  return kTfLiteOk;
}

void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   TfLiteConvParams* params, OpData* data,
                   const TfLiteTensor* input, const TfLiteTensor* filter,
                   const TfLiteTensor* bias, TfLiteTensor* im2col,
                   TfLiteTensor* hwcn_weights, TfLiteTensor* output) {
  const int32 filter_offset = -filter->params.zero_point;
  const int32 output_offset = output->params.zero_point;

  const RuntimeShape& input_shape = GetTensorShape(input);
  const uint8* input_data = GetTensorData<uint8_t>(input);
  const RuntimeShape& filter_shape = GetTensorShape(filter);
  const uint8* filter_data = GetTensorData<uint8_t>(filter);
  const RuntimeShape& bias_shape = GetTensorShape(bias);
  const int32* bias_data = GetTensorData<int32_t>(bias);
  const RuntimeShape& output_shape = GetTensorShape(output);
  uint8* output_data = GetTensorData<uint8_t>(output);

  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int dilation_width_factor = 1;
  const int dilation_height_factor = 1;

  const int32 output_multiplier = data->output_multiplier;
  const int output_shift = -data->output_shift;
  const int32 output_activation_min = data->output_activation_min;
  const int32 output_activation_max = data->output_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int* in_dims = reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    uint32 offset_input0 = batch * in_dims[1];
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        uint32_t filter_index = 0;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = out_x * stride_width;
          const int in_y_origin = out_y * stride_height;
          int32 acc = 0;

          const int32_t ker_y_start = MAX(0, -in_y_origin);
          const int32_t ker_x_start = MAX(0, -in_x_origin);

          const int32_t ker_y_end = MIN(filter_height, input_height - in_y_origin);
          const int32_t ker_x_end = MIN(filter_width, input_width - in_x_origin);

          for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            uint32 offset_input1 = (offset_input0 + in_y) * in_dims[2];
            for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              uint32 offset_input2 = (offset_input1 + in_x) * in_dims[3];
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {

                int32 input_val = input_data[offset_input2 + in_channel];
                int32 filter_val = filter_data[filter_index++];

                acc += (filter_val + filter_offset) * input_val;
              }
            }
          }

          acc += data->sum_of_filters_factor[out_channel];
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<uint8>(acc);
        }
      }
    }
  }
}

void EvalQuantizedWithPadding(TfLiteContext* context, TfLiteNode* node,
                   TfLiteConvParams* params, OpData* data,
                   const TfLiteTensor* input, const TfLiteTensor* filter,
                   const TfLiteTensor* bias, TfLiteTensor* im2col,
                   TfLiteTensor* hwcn_weights, TfLiteTensor* output) {
  const int32 input_offset = -input->params.zero_point;
  const int32 filter_offset = -filter->params.zero_point;
  const int32 output_offset = output->params.zero_point;

  const RuntimeShape& input_shape = GetTensorShape(input);
  const uint8* input_data = GetTensorData<uint8_t>(input);
  const RuntimeShape& filter_shape = GetTensorShape(filter);
  const uint8* filter_data = GetTensorData<uint8_t>(filter);
  const RuntimeShape& bias_shape = GetTensorShape(bias);
  const int32* bias_data = GetTensorData<int32_t>(bias);
  const RuntimeShape& output_shape = GetTensorShape(output);
  uint8* output_data = GetTensorData<uint8_t>(output);

  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int dilation_width_factor = 1;
  const int dilation_height_factor = 1;

  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int32 output_multiplier = data->output_multiplier;
  const int output_shift = -data->output_shift;
  const int32 output_activation_min = data->output_activation_min;
  const int32 output_activation_max = data->output_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int* in_dims = reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims = reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    uint32 offset_input0 = batch * in_dims[1];
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          uint32 offset_filter0 = out_channel * fi_dims[1];

          int32 acc = 0;

          const int32_t ker_y_start = MAX(0, -in_y_origin);
          const int32_t ker_x_start = MAX(0, -in_x_origin);

          const int32_t ker_y_end = MIN(filter_height, input_height - in_y_origin);
          const int32_t ker_x_end = MIN(filter_width, input_width - in_x_origin);

          for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            uint32 offset_filter1 = (offset_filter0 + filter_y) * fi_dims[2];
            uint32 offset_input1 = (offset_input0 + in_y) * in_dims[2];
            for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              uint32 offset_filter2 = (offset_filter1 + filter_x) * fi_dims[3];
              uint32 offset_input2 = (offset_input1 + in_x) * in_dims[3];
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {

                int32 input_val = input_data[offset_input2 + in_channel];
                int32 filter_val = filter_data[offset_filter2 + in_channel];

                acc += (filter_val + filter_offset) * (input_val + input_offset);
              }
            }
          }
          if (bias) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<uint8>(acc);
        }
      }
    }
  }
}

void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             TfLiteConvParams* params, OpData* data,
                             const int32* output_multiplier, const int32* output_shift,
                             const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             TfLiteTensor* im2col) {
  const int32 output_offset = output->params.zero_point;

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
  const int dilation_width_factor = 1;
  const int dilation_height_factor = 1;
  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int32 output_activation_min = data->output_activation_min;
  const int32 output_activation_max = data->output_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int* in_dims = reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims = reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    uint32 offset_input0 = batch * in_dims[1];
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          uint32 offset_filter0 = out_channel * fi_dims[1];

          const int32_t ker_y_start = MAX(0, -in_y_origin);
          const int32_t ker_x_start = MAX(0, -in_x_origin);

          const int32_t ker_y_end = MIN(filter_height, input_height - in_y_origin);
          const int32_t ker_x_end = MIN(filter_width, input_width - in_x_origin);

          int32 acc = 0;

          for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            uint32 offset_filter1 = (offset_filter0 + filter_y) * fi_dims[2];
            uint32 offset_input1 = (offset_input0 + in_y) * in_dims[2];

            for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              uint32 offset_filter2 = (offset_filter1 + filter_x) * fi_dims[3];
              uint32 offset_input2 = (offset_input1 + in_x) * in_dims[3];

              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                int32 input_val = input_data[offset_input2 + in_channel];
                int32 filter_val = filter_data[offset_filter2 + in_channel];
                acc += filter_val * input_val;
              }
            }
          }
          acc += data->sum_of_filters_factor[out_channel];
          acc = MultiplyByQuantizedMultiplier(
                        acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<int8>(acc);
        }
      }
    }
  }
}

void EvalQuantizedPerChannelWithPadding(TfLiteContext* context, TfLiteNode* node,
                             TfLiteConvParams* params, OpData* data,
                             const int32* output_multiplier, const int32* output_shift,
                             const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             TfLiteTensor* im2col) {
  const int32 input_offset = -input->params.zero_point;
  const int32 output_offset = output->params.zero_point;

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
  const int dilation_width_factor = 1;
  const int dilation_height_factor = 1;
  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int32 output_activation_min = data->output_activation_min;
  const int32 output_activation_max = data->output_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int* in_dims = reinterpret_cast<const int*>(input_shape.DimsDataUpTo5D());
  const int* fi_dims = reinterpret_cast<const int*>(filter_shape.DimsDataUpTo5D());

  for (int batch = 0; batch < batches; ++batch) {
    uint32 offset_input0 = batch * in_dims[1];
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          uint32 offset_filter0 = out_channel * fi_dims[1];

          const int32_t ker_y_start = MAX(0, -in_y_origin);
          const int32_t ker_x_start = MAX(0, -in_x_origin);

          const int32_t ker_y_end = MIN(filter_height, input_height - in_y_origin);
          const int32_t ker_x_end = MIN(filter_width, input_width - in_x_origin);

          int32 acc = 0;

          for (int filter_y = ker_y_start; filter_y < ker_y_end; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            uint32 offset_filter1 = (offset_filter0 + filter_y) * fi_dims[2];
            uint32 offset_input1 = (offset_input0 + in_y) * in_dims[2];

            for (int filter_x = ker_x_start; filter_x < ker_x_end; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              uint32 offset_filter2 = (offset_filter1 + filter_x) * fi_dims[3];
              uint32 offset_input2 = (offset_input1 + in_x) * in_dims[3];

              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                int32 input_val = input_data[offset_input2 + in_channel];
                int32 filter_val = filter_data[offset_filter2 + in_channel];
                acc += filter_val * (input_val + input_offset);
              }
            }
          }
          if (bias) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(
                        acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<int8>(acc);
        }
      }
    }
  }
}

void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteConvParams* params, OpData* data,
               const TfLiteTensor* input, const TfLiteTensor* filter,
               const TfLiteTensor* bias, TfLiteTensor* im2col,
               TfLiteTensor* hwcn_weights, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  reference_ops::Conv(op_params, GetTensorShape(input),
                      GetTensorData<float>(input), GetTensorShape(filter),
                      GetTensorData<float>(filter), GetTensorShape(bias),
                      GetTensorData<float>(bias), GetTensorShape(output),
                      GetTensorData<float>(output), GetTensorShape(im2col),
                      GetTensorData<float>(im2col));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);

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

  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = -filter->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  TfLiteTensor* im2col;
#define TF_LITE_CONV_2D_PER_CHANNEL(func_c, data_type)                                  \
  func_c(                                                                               \
      op_params, data->per_channel_output_multiplier, data->per_channel_output_shift, \
      GetTensorShape(input), GetTensorData<data_type>(input),                         \
      GetTensorShape(filter), GetTensorData<data_type>(filter),                       \
      GetTensorShape(bias), GetTensorData<int32_t>(bias),                             \
      GetTensorShape(output), GetTensorData<data_type>(output))

#define TF_LITE_CONV_2D_PER_LAYER(func_l, data_type)                                    \
  func_l(                                                                               \
      op_params, GetTensorShape(input), GetTensorData<data_type>(input),              \
      GetTensorShape(filter), GetTensorData<data_type>(filter),                       \
      GetTensorShape(bias), GetTensorData<int32_t>(bias),                             \
      GetTensorShape(output), GetTensorData<data_type>(output),                       \
      GetTensorShape(im2col), GetTensorData<data_type>(im2col), nullptr)

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
    {
      EvalFloat(context, node, params, data, input, filter, bias, nullptr,
                nullptr, output);
      break;
    }
    case kTfLiteInt8:
    {
      const int dilation_width_factor = params->dilation_width_factor;
      const int dilation_height_factor = params->dilation_height_factor;
      if ((dilation_width_factor != 1) || (dilation_height_factor != 1)) {
        TF_LITE_CONV_2D_PER_CHANNEL(reference_integer_ops::ConvPerChannel, int8_t);
      }
      else if (data->padding.height != 0 || data->padding.width != 0 ||
          data->padding.height_offset != 0 || data->padding.width_offset != 0) {
        EvalQuantizedPerChannelWithPadding(context, node, params, data, data->per_channel_output_multiplier, data->per_channel_output_shift,
                                               input, filter, bias, output, nullptr);
      }
      else {
        EvalQuantizedPerChannel(context, node, params, data, data->per_channel_output_multiplier, data->per_channel_output_shift,
                                       input, filter, bias, output, nullptr);
      }
      break;
    }
    case kTfLiteUInt8:
    {
      const int dilation_width_factor = params->dilation_width_factor;
      const int dilation_height_factor = params->dilation_height_factor;

      if ((dilation_width_factor != 1) || (dilation_height_factor != 1)) {
        TF_LITE_CONV_2D_PER_LAYER(reference_ops::Conv, uint8_t);
      }
      else if (data->padding.height != 0 || data->padding.width != 0 ||
          data->padding.height_offset != 0 || data->padding.width_offset != 0) {
        EvalQuantizedWithPadding(context, node, params, data, input, filter, bias, nullptr, nullptr, output);
      }
      else {
        EvalQuantized(context, node, params, data, input, filter, bias, nullptr, nullptr, output);
      }
      break;
    }
    default:
    {
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace conv

TfLiteRegistration Register_CONV_2D() {
  return {/*init=*/conv::Init,
          /*free=*/conv::Free,
          /*prepare=*/conv::Prepare,
          /*invoke=*/conv::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
