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
#include <sys/types.h>

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace {

void PickOutputMultiplier(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const int16* input_data, const RuntimeShape& filter_shape,
    const int8* filter_data, const RuntimeShape& bias_shape,
    const std::int64_t* bias_data, const RuntimeShape& output_shape,
    float* output_multiplier) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32 input_offset = params.input_offset;

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  std::int64_t output_accu_min = std::numeric_limits<std::int64_t>::max();
  std::int64_t output_accu_max = std::numeric_limits<std::int64_t>::min();

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            std::int64_t acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                if (is_point_inside_image) {
                  int32 input_val = input_data[Offset(input_shape, batch, in_y,
                                                      in_x, in_channel)];
                  int32 filter_val = filter_data[Offset(
                      filter_shape, 0, filter_y, filter_x, output_channel)];
                  acc += static_cast<int64_t>(filter_val) *
                         static_cast<int64_t>(input_val + input_offset);
                }
              }
            }
            if (bias_data) {
              acc += bias_data[output_channel];
            }
            output_accu_max = std::max(acc, output_accu_max);
            output_accu_min = std::min(acc, output_accu_min);
          }
        }
      }
    }
  }

  // Since int16 ranges from -32768 to 32767, we need to squeeze the accumulator
  // min/max fit in those ranges correspondingly as much as possible.
  if (std::abs(output_accu_max) > std::abs(output_accu_min)) {
    *output_multiplier = 32767.0f / std::abs(output_accu_max);
  } else {
    *output_multiplier = 32768.0f / std::abs(output_accu_min);
  }
}

void PickReasonableMultiplier(
    const DepthwiseParams& params, int output_activation_min,
    int output_activation_max, int output_depth,
    const RuntimeShape& input_shape_inference, const std::int16_t* input_data,
    const RuntimeShape& filter_shape_inference, const std::int8_t* filter_data,
    const RuntimeShape& bias_shape_inference, const std::int64_t* bias_data,
    const RuntimeShape& output_shape_inference,
    std::int32_t* output_multiplier_ptr, std::int32_t* output_shift_ptr,
    std::int16_t* output_data) {
  float output_multiplier;
  PickOutputMultiplier(params, input_shape_inference, input_data,
                       filter_shape_inference, filter_data,
                       bias_shape_inference, bias_data, output_shape_inference,
                       &output_multiplier);

  int base_multiplier;
  int base_shift;
  QuantizeMultiplier(output_multiplier, &base_multiplier, &base_shift);
  for (int i = 0; i < output_depth; ++i) {
    // multipliers typically range in [2^30 ; 2^31 - 1].
    // Values in [0, 2^30 - 1] are normally unused, but harmless.
    // Thus a good way to randomize multipliers is to subtract from them
    // a random value smaller than 2^30 but still significant compared to it.
    output_multiplier_ptr[i] = base_multiplier - (std::rand() % (1 << 26));
    output_shift_ptr[i] = base_shift - 1 + (std::rand() % 4);
  }
}

bool GenerateValidShapeConfigurations(
    int filter_width, int filter_height, int depth_multiplier,
    int dilation_width_factor, int dilation_height_factor,
    RuntimeShape* input_shape_inference, RuntimeShape* filter_shape_inference,
    RuntimeShape* output_shape_inference, int* pad_width, int* pad_height,
    int* stride) {
  const int batch = UniformRandomInt(1, 3);
  const int input_depth = 8 * ExponentialRandomPositiveInt(0.9f, 10, 50);
  const int input_width = UniformRandomInt(5, 50);
  const int input_height = UniformRandomInt(5, 50);
  *stride = UniformRandomInt(1, 2);
  const bool test_pad = UniformRandomInt(0, 1);
  const auto padding_type = test_pad ? PaddingType::kValid : PaddingType::kSame;

  const int output_depth = input_depth * depth_multiplier;

  input_shape_inference->BuildFrom(
      {batch, input_height, input_width, input_depth});

  filter_shape_inference->BuildFrom(
      {1, filter_height, filter_width, output_depth});

  EXPECT_TRUE(ComputeConvSizes(
      *input_shape_inference, output_depth, filter_width, filter_height,
      *stride, dilation_width_factor, dilation_height_factor, padding_type,
      output_shape_inference, pad_width, pad_height));

  return true;
}

void IntToFloat(std::vector<float>* d, std::vector<std::int8_t>* s) {
  for (unsigned int i = 0; i < s->size(); i++) {
    d->data()[i] = (float)s->data()[i];
  }
}

void IntToFloat(std::vector<float>* d, std::vector<std::int64_t>* s) {
  for (unsigned int i = 0; i < s->size(); i++) {
    d->data()[i] = (float)s->data()[i];
  }
}

void TryTestOneDepthwiseConv3x3Filter() {
  const int filter_width = 3;
  const int filter_height = 3;
  const int depth_multiplier = 1;
  // We don't support dilations in the 3x3 filter.
  const int dilation_width_factor = 1;
  const int dilation_height_factor = 1;

  const int output_activation_min = -32768;
  const int output_activation_max = 32767;

  const std::int32_t input_offset = UniformRandomInt(-25, 25);
  const std::int32_t output_offset = UniformRandomInt(-25, 25);

  RuntimeShape input_shape_inference;
  RuntimeShape filter_shape_inference;
  RuntimeShape output_shape_inference;
  int pad_width, pad_height;
  int stride;

  // Keeps trying until we get valid shape/configurations for 3x3 filter case.
  bool generated_valid_configurations_for_3x3_kernel = false;
  while (!generated_valid_configurations_for_3x3_kernel) {
    generated_valid_configurations_for_3x3_kernel =
        GenerateValidShapeConfigurations(
            filter_width, filter_height, depth_multiplier,
            dilation_width_factor, dilation_height_factor,
            &input_shape_inference, &filter_shape_inference,
            &output_shape_inference, &pad_width, &pad_height, &stride);
  }

  const int output_depth = output_shape_inference.Dims(3);

  RuntimeShape bias_shape_inference({1, 1, 1, output_depth});
  const int input_buffer_size = input_shape_inference.FlatSize();
  const int filter_buffer_size = filter_shape_inference.FlatSize();
  const int output_buffer_size = output_shape_inference.FlatSize();
  std::vector<std::int16_t> input_data(input_buffer_size);
  std::vector<std::int8_t> filter_data(filter_buffer_size);
  std::vector<std::int64_t> bias_data(output_depth);

  FillRandom(&input_data);
  FillRandom(&filter_data);
  for (int i = 0; i < output_depth; i++) {
    bias_data.data()[i] = 0;
  }

  DepthwiseParams params;
  params.stride_width = stride;
  params.stride_height = stride;
  params.dilation_height_factor = dilation_height_factor;
  params.dilation_width_factor = dilation_width_factor;
  params.padding_values.width = pad_width;
  params.padding_values.height = pad_height;
  params.depth_multiplier = depth_multiplier;
  params.input_offset = input_offset;
  params.output_offset = output_offset;
  params.weights_offset = 0;
  params.quantized_activation_min = output_activation_min;
  params.quantized_activation_max = output_activation_max;
  params.float_activation_max = (float)(1LL << 40);
  params.float_activation_min = -params.float_activation_max;

  std::vector<std::int16_t> reference_output_data(output_buffer_size);
  std::vector<std::int16_t> neon_output_data(output_buffer_size);

  std::vector<std::int32_t> output_multiplier(output_depth);
  std::vector<std::int32_t> output_shift(output_depth);

  // It's hard to come up with a right multiplier, random guess basically makes
  // all the results saturated and becomes meaningfulless, so we first use
  // reference impl to poke the min/max value of the accumulation, then use that
  // value as a guided suggestion for us to populate meaningful mulitplier &
  // shift.
  PickReasonableMultiplier(
      params, output_activation_min, output_activation_max, output_depth,
      input_shape_inference, input_data.data(), filter_shape_inference,
      filter_data.data(), bias_shape_inference, bias_data.data(),
      output_shape_inference, output_multiplier.data(), output_shift.data(),
      reference_output_data.data());

  // The following tests compare referene impl and Neon general impl agrees,
  // and reference impl loosely agrees with fast kernel since they use different
  // rounding strategy.
  reference_integer_ops::DepthwiseConvPerChannel(
      params, output_multiplier.data(), output_shift.data(),
      input_shape_inference, input_data.data(), filter_shape_inference,
      filter_data.data(), bias_shape_inference, bias_data.data(),
      output_shape_inference, reference_output_data.data());

  std::vector<float> input_data_float(input_buffer_size);
  std::vector<float> filter_data_float(filter_buffer_size);
  std::vector<float> bias_data_float(output_depth);
  std::vector<float> output_data_float(output_buffer_size);

  for (int i = 0; i < input_buffer_size; i++) {
    input_data_float.data()[i] = (float)(input_data.data()[i] + input_offset);
  }
  IntToFloat(&filter_data_float, &filter_data);
  IntToFloat(&bias_data_float, &bias_data);

  reference_ops::DepthwiseConv(
      params, input_shape_inference, input_data_float.data(),
      filter_shape_inference, filter_data_float.data(), bias_shape_inference,
      bias_data_float.data(), output_shape_inference, output_data_float.data());

  for (int n = 0; n < output_shape_inference.Dims(0); n++) {
    for (int h = 0; h < output_shape_inference.Dims(1); h++) {
      for (int w = 0; w < output_shape_inference.Dims(2); w++) {
        for (int c = 0; c < output_shape_inference.Dims(3); c++) {
          int offset = Offset(output_shape_inference, n, h, w, c);
          float float_res = output_data_float.data()[offset];
          int16 int16_res = reference_output_data.data()[offset];
          int32 output_mul = output_multiplier.data()[c];
          int shift = output_shift.data()[c];
          float scale = (float)output_mul / (float)(1ULL << 31);
          if (shift > 0) scale = scale * (float)(1 << shift);
          if (shift < 0) scale = scale / (float)(1 << -shift);
          int ref_res = floor(float_res * scale + 0.5) + output_offset;
          if (ref_res < output_activation_min) ref_res = output_activation_min;
          if (ref_res > output_activation_max) ref_res = output_activation_max;
          int e = (ref_res - int16_res);
          if (e < 0) e = -e;
          if (e > 1) {
            printf(
                "(%d,%d,%d,%d) scale=%08x shift=%d res=%d float=%f (%f,%f)\n",
                n, h, w, c, output_mul, shift, int16_res,
                float_res * scale + (float)output_offset, float_res, scale);
            EXPECT_TRUE(false);
          }
        }
      }
    }
  }
}

TEST(QuantizedDepthwiseConvPerChannelTest, FastKernelTest) {
  for (int i = 0; i < 30; ++i) {
    TryTestOneDepthwiseConv3x3Filter();
  }
}

}  // namespace
}  // namespace tflite
