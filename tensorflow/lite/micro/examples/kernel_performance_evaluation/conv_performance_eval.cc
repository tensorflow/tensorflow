/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

/*
 * This file contains tests which are specifically made for
 * evaluating the performance of the convolutional layer implementation.
 */

#include <chrono>
#include <iostream>
#include <fstream>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

namespace tflite {
namespace testing {
namespace {

static const int batches = 4;
static const int input_size = 32;
static const int input_channels = 3;
static const int input_zero_point = 0;
static const int kInputElements = batches*input_size*input_size*input_channels;
static const int kInputShape[] = {4, batches, input_size, input_size, input_channels};
static float kInputData[kInputElements];

static const int filter_size = 6;
static const int number_filters = 16;
static const int kFilterElements = filter_size * filter_size * input_channels * number_filters;
static const int kFilterShape[] = {4, number_filters, filter_size, filter_size, input_channels};
static float kFilterData[kFilterElements];

static const int kBiasElements = number_filters;
static const int kBiasShape[] = {1, number_filters};
static float kBiasData[kBiasElements];

static const int stride = 1;
static const int output_zero_point = 0;
static const int output_size = ((input_size - filter_size) / stride) + 1;
static const int kOutputElements = batches*output_size*output_size*number_filters;
static const int kOutputShape[] = {4, batches, output_size, output_size, number_filters};
static float kGoldenData[kOutputElements];

static const float input_scale = 0.5f;
static const float filter_scale = 0.5f;
static const float output_scale = 1.0f;

static int zero_points[kBiasElements + 1];
static float scales[kBiasElements + 1];

static const int benchmarking_iterations = 1;
static const int number_of_invocations = 1000;

static TfLiteConvParams common_conv_params = {
    kTfLitePaddingValid,  // padding
    1,                    // stride_width
    1,                    // stride_height
    kTfLiteActNone,       // activation
    1,                    // dilation_width_factor
    1,                    // dilation_height_factor
};

void InitConvTestDataQuantized(const int filter_size, const int output_channels,
                           const int input_channels, const int batches, const int input_size,
                           float* input_data,
                           float* filters_data,
                           float* bias_data) {
  const RuntimeShape input_shape = {batches, input_size, input_size, input_channels};
  const RuntimeShape filter_shape = {output_channels, filter_size, filter_size, input_channels};

  // Initialize input
  for (int b = 0; b < batches; b++) {
    for (int in_x = 0; in_x < input_size; in_x++) {
      for (int in_y = 0; in_y < input_size; in_y++) {
        for (int in_channel = 0; in_channel < input_channels; in_channel++) {
          input_data[Offset(input_shape, b, in_x, in_y, in_channel)] = in_x + in_y;
        }
      }
    }
  }
  // Initialize filters
  for (int out_channel = 0; out_channel < output_channels; out_channel++) {
    bias_data[out_channel] = out_channel;
    for (int filter_x = 0; filter_x < filter_size; filter_x++) {
      for (int filter_y = 0; filter_y < filter_size; filter_y++) {
        for (int in_channel = 0; in_channel < input_channels; in_channel++) {
          filters_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)] = filter_x + filter_y;
        }
      }
    }
  }
}

void InitGoldenData(const int filter_size, const int output_channels,
                    const int input_channels, const int batches, const int input_size,
                    float* input_data,
                    float* filters_data,
                    float* bias_data,
                    float* expected_output,
                    int stride, int output_size) {
  const RuntimeShape input_shape = {batches, input_size, input_size, input_channels};
  const RuntimeShape filter_shape = {output_channels, filter_size, filter_size, input_channels};
  const RuntimeShape output_shape = {batches, output_size, output_size, output_channels};

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_size; ++out_y) {
      for (int out_x = 0; out_x < output_size; ++out_x) {
        for (int out_channel = 0; out_channel < output_channels; ++out_channel) {
          const int in_x_origin = (out_x * stride);
          const int in_y_origin = (out_y * stride);
          float total = 0.f;
          for (int filter_y = 0; filter_y < filter_size; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_size; ++filter_x) {
              for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                float input_value = input_data[Offset(
                    input_shape, batch, in_y, in_x, in_channel)];
                float filter_value =
                    filters_data[Offset(filter_shape, out_channel, filter_y,
                                        filter_x, in_channel)];
                total += (input_value * filter_value);
              }
            }
          }
          float bias_value = bias_data[out_channel];
          expected_output[Offset(output_shape, batch, out_y, out_x, out_channel)] = total + bias_value;
        }
      }
    }
  }
}

void InitGoldenDataPadding(const int filter_size, const int output_channels,
                    const int input_channels, const int batches, const int input_size,
                    float* input_data,
                    float* filters_data,
                    float* bias_data,
                    float* expected_output,
                    int stride, int output_size, int padding) {
  const RuntimeShape input_shape = {batches, input_size, input_size, input_channels};
  const RuntimeShape filter_shape = {output_channels, filter_size, filter_size, input_channels};
  const RuntimeShape output_shape = {batches, output_size, output_size, output_channels};

  for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_size; ++out_y) {
        for (int out_x = 0; out_x < output_size; ++out_x) {
          for (int out_channel = 0; out_channel < output_channels; ++out_channel) {
            const int in_x_origin = (out_x * stride) - padding;
            const int in_y_origin = (out_y * stride) - padding;
            float total = 0.f;
            for (int filter_y = 0; filter_y < filter_size; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_size; ++filter_x) {
                for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
                  const int in_x = in_x_origin + 1 * filter_x;
                  const int in_y = in_y_origin + 1 * filter_y;
                  // If the location is outside the bounds of the input image,
                  // use zero as a default value.
                  if ((in_x >= 0) && (in_x < input_size) && (in_y >= 0) &&
                      (in_y < input_size)) {
                    float input_value = input_data[Offset(
                        input_shape, batch, in_y, in_x, in_channel)];
                    float filter_value =
                        filters_data[Offset(filter_shape, out_channel, filter_y,
                                           filter_x, in_channel)];
                    total += (input_value * filter_value);
                  }
                }
              }
            }
            float bias_value = 0.0f;
            if (bias_data) {
              bias_value = bias_data[out_channel];
            }
            expected_output[Offset(output_shape, batch, out_y, out_x, out_channel)] = total + bias_value;
          }
        }
      }
    }
}

template <typename T>
TfLiteStatus ValidateConvGoldensPerformance(TfLiteTensor* tensors, int tensors_size,
                                 const T* expected_output_data, T* output_data,
                                 int output_length,
                                 TfLiteConvParams* conv_params,
                                 float tolerance = 1e-5) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;

  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_CONV_2D);

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = reinterpret_cast<const char*>(conv_params);
  size_t init_data_size = 0;
  void* user_data = nullptr;

  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(conv_params);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }

  // Start main benchmarking loop
  // Increase the variable benchmarking_iterations to make result representative
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < benchmarking_iterations; i++) {
    TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
    for (int j = 0; j < number_of_invocations; j++) {
      TfLiteStatus return_val = registration->invoke(&context, &node);
      if (return_val != kTfLiteOk) {
        return return_val;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  micro_test::reporter->Report("%d Avg Invoke run time =  %d us", number_of_invocations, duration/benchmarking_iterations);
  if (registration->free) {
    registration->free(&context, user_data);
  }

  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }
  return kTfLiteOk;
}

template<typename T>
void TestConvQuantizedPerformance(
    const int* input_dims_data, const float* input_data,
    T* input_quantized, float input_scale, const int* filter_dims_data,
    const float* filter_data, T* filter_quantized, float filter_scale,
    const int* bias_dims_data, const float* bias_data, int32_t* bias_quantized,
    const int* output_dims_data, const float* expected_output_data,
    T* expected_output_quantized, T* output_data,
    float output_scale, TfLiteConvParams* conv_params) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  tflite::AsymmetricQuantize(expected_output_data, expected_output_quantized,
                             output_dims_count, output_scale, 128);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, 128),
      CreateQuantizedTensor(filter_data, filter_quantized, filter_dims,
                            filter_scale, 128),
      CreateQuantizedBiasTensor(bias_data, bias_quantized, bias_dims,
                                input_scale, filter_scale),
      CreateQuantizedTensor(output_data, output_dims, output_scale, 128)};

  float filter_scales[] = {1, filter_scale};
  int filter_zero_points[] = {1, 128};
  TfLiteAffineQuantization filter_quant = {
      FloatArrayFromFloats(filter_scales),
      IntArrayFromInts(filter_zero_points)};
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant};

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      ValidateConvGoldensPerformance(tensors, tensors_size, expected_output_quantized,
                          output_data, output_dims_count, conv_params, 1e-5));
}

void TestConvQuantizedPerChannelPerformance(
    const int* input_dims_data, const float* input_data,
    int8_t* input_quantized, float input_scale, int input_zero_point,
    const int* filter_dims_data, const float* filter_data,
    int8_t* filter_data_quantized, const int* bias_dims_data,
    const float* bias_data, int32_t* bias_data_quantized, float* bias_scales,
    int* bias_zero_points, const int* output_dims_data,
    const float* expected_output_data, int8_t* expected_output_data_quantized,
    int8_t* output_data, float output_scale, int output_zero_point,
    TfLiteConvParams* conv_params)
{
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  int filter_zero_points[tflite::testing::number_filters + 1];
  float filter_scales[tflite::testing::number_filters + 1];
  TfLiteAffineQuantization filter_quant;

  TfLiteAffineQuantization bias_quant;
  TfLiteTensor input_tensor =
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point);
  TfLiteTensor filter_tensor = CreateSymmetricPerChannelQuantizedTensor(
      filter_data, filter_data_quantized, filter_dims, filter_scales,
      filter_zero_points, &filter_quant, 0 /* quantized dimension */);
  TfLiteTensor bias_tensor = CreatePerChannelQuantizedBiasTensor(
      bias_data, bias_data_quantized, bias_dims, input_scale, &filter_scales[1],
      bias_scales, bias_zero_points, &bias_quant, 0 /* quantized dimension */);
  TfLiteTensor output_tensor =
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point);

  float input_scales[] = {1, input_scale};
  int input_zero_points[] = {1, input_zero_point};
  TfLiteAffineQuantization input_quant = {FloatArrayFromFloats(input_scales),
                                          IntArrayFromInts(input_zero_points)};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  float output_scales[] = {1, output_scale};
  int output_zero_points[] = {1, output_zero_point};
  TfLiteAffineQuantization output_quant = {
      FloatArrayFromFloats(output_scales),
      IntArrayFromInts(output_zero_points)};
  output_tensor.quantization = {kTfLiteAffineQuantization, &output_quant};

  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  tflite::AsymmetricQuantize(expected_output_data,
                             expected_output_data_quantized, output_dims_count,
                             output_scale, output_zero_point);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      ValidateConvGoldensPerformance(tensors, tensors_size, expected_output_data_quantized,
                          output_data, output_dims_count, conv_params,
                          1.0 /* tolerance */));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(ConvPerformanceTestQuantizedUint8) {

  tflite::testing::InitConvTestDataQuantized(tflite::testing::filter_size, tflite::testing::number_filters,
                                             tflite::testing::input_channels, tflite::testing::batches, tflite::testing::input_size,
                                             tflite::testing::kInputData,
                                             tflite::testing::kFilterData,
                                             tflite::testing::kBiasData);

  tflite::testing::InitGoldenData(tflite::testing::filter_size, tflite::testing::number_filters,
                                  tflite::testing::input_channels, tflite::testing::batches, tflite::testing::input_size,
                                  tflite::testing::kInputData,
                                  tflite::testing::kFilterData,
                                  tflite::testing::kBiasData,
                                  tflite::testing::kGoldenData,
                                  tflite::testing::stride, tflite::testing::output_size);

  uint8_t input_quantized[tflite::testing::kInputElements];
  uint8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  uint8_t golden_quantized[tflite::testing::kOutputElements];
  uint8_t output_data[tflite::testing::kOutputElements];

  tflite::testing::TestConvQuantizedPerformance(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      input_quantized, tflite::testing::input_scale, tflite::testing::kFilterShape,
      tflite::testing::kFilterData, filter_quantized, tflite::testing::filter_scale,
      tflite::testing::kBiasShape, tflite::testing::kBiasData, bias_quantized,
      tflite::testing::kOutputShape, tflite::testing::kGoldenData,
      golden_quantized, output_data, tflite::testing::output_scale,
      &tflite::testing::common_conv_params);
}

TF_LITE_MICRO_TEST(ConvPerformanceTestQuantizedInt8) {

  tflite::testing::InitConvTestDataQuantized(tflite::testing::filter_size, tflite::testing::number_filters,
                                             tflite::testing::input_channels, tflite::testing::batches, tflite::testing::input_size,
                                             tflite::testing::kInputData,
                                             tflite::testing::kFilterData,
                                             tflite::testing::kBiasData);

  tflite::testing::InitGoldenData(tflite::testing::filter_size, tflite::testing::number_filters,
                                  tflite::testing::input_channels, tflite::testing::batches, tflite::testing::input_size,
                                  tflite::testing::kInputData,
                                  tflite::testing::kFilterData,
                                  tflite::testing::kBiasData,
                                  tflite::testing::kGoldenData,
                                  tflite::testing::stride, tflite::testing::output_size);

  int8_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t golden_quantized[tflite::testing::kOutputElements];
  int8_t output_data[tflite::testing::kOutputElements];

  tflite::testing::TestConvQuantizedPerChannelPerformance(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      input_quantized, tflite::testing::input_scale, tflite::testing::input_zero_point, tflite::testing::kFilterShape,
      tflite::testing::kFilterData, filter_quantized,
      tflite::testing::kBiasShape, tflite::testing::kBiasData, bias_quantized, tflite::testing::scales, tflite::testing::zero_points,
      tflite::testing::kOutputShape, tflite::testing::kGoldenData,
      golden_quantized, output_data, tflite::testing::output_scale, tflite::testing::output_zero_point,
      &tflite::testing::common_conv_params);
}

TF_LITE_MICRO_TEST(ConvPerformanceTestUint8Padding) {
  static const int batches = 4;
  static const int input_size = 32;
  static const int input_channels = 3;
  static const int kInputElements = batches*input_size*input_size*input_channels;
  static const int kInputShape[] = {4, batches, input_size, input_size, input_channels};
  static float kInputData[kInputElements];

  static const int filter_size = 5;
  static const int number_filters = 16;
  static const int kFilterElements = filter_size * filter_size * input_channels * number_filters;
  static const int kFilterShape[] = {4, number_filters, filter_size, filter_size, input_channels};
  static float kFilterData[kFilterElements];

  static const int kBiasElements = number_filters;
  static const int kBiasShape[] = {1, number_filters};
  static float kBiasData[kBiasElements];

  static const int stride = 1;
  static const int output_size = 32;
  static const int kOutputElements = batches*output_size*output_size*number_filters;
  static const int kOutputShape[] = {4, batches, output_size, output_size, number_filters};
  static float kGoldenData[kOutputElements];

  static const float input_scale = 0.5f;
  static const float filter_scale = 0.5f;
  static const float output_scale = 1.0f;

  static TfLiteConvParams padding_conv_params = {
      kTfLitePaddingSame,  // padding
      1,                    // stride_width
      1,                    // stride_height
      kTfLiteActNone,       // activation
      1,                    // dilation_width_factor
      1,                    // dilation_height_factor
  };

    tflite::testing::InitConvTestDataQuantized(filter_size, number_filters,
                                               input_channels, batches, input_size,
                                               kInputData,
                                               kFilterData,
                                               kBiasData);

    tflite::testing::InitGoldenDataPadding(filter_size, number_filters,
                                    input_channels, batches, input_size,
                                    kInputData,
                                    kFilterData,
                                    kBiasData,
                                    kGoldenData,
                                    stride, output_size, 2);

    uint8_t input_quantized[kInputElements];
    uint8_t filter_quantized[kFilterElements];
    int32_t bias_quantized[kBiasElements];
    uint8_t golden_quantized[kOutputElements];
    uint8_t output_data[kOutputElements];

    tflite::testing::TestConvQuantizedPerformance(
        kInputShape, kInputData,
        input_quantized, input_scale, kFilterShape,
        kFilterData, filter_quantized, filter_scale,
        kBiasShape, kBiasData, bias_quantized,
        kOutputShape, kGoldenData,
        golden_quantized, output_data, output_scale,
        &padding_conv_params);
}

TF_LITE_MICRO_TEST(ConvPerformanceTestInt8Padding) {
  static const int batches = 4;
  static const int input_size = 32;
  static const int input_channels = 3;
  static const int input_zero_point = 0;
  static const int kInputElements = batches*input_size*input_size*input_channels;
  static const int kInputShape[] = {4, batches, input_size, input_size, input_channels};
  static float kInputData[kInputElements];

  static const int filter_size = 5;
  static const int number_filters = 16;
  static const int kFilterElements = filter_size * filter_size * input_channels * number_filters;
  static const int kFilterShape[] = {4, number_filters, filter_size, filter_size, input_channels};
  static float kFilterData[kFilterElements];

  static const int kBiasElements = number_filters;
  static const int kBiasShape[] = {1, number_filters};
  static float kBiasData[kBiasElements];

  static const int stride = 1;
  static const int output_zero_point = 0;
  static const int output_size = 32;
  static const int kOutputElements = batches*output_size*output_size*number_filters;
  static const int kOutputShape[] = {4, batches, output_size, output_size, number_filters};
  static float kGoldenData[kOutputElements];

  static const float input_scale = 0.5f;
  static const float output_scale = 1.0f;

  static int zero_points[kBiasElements + 1];
  static float scales[kBiasElements + 1];


  static TfLiteConvParams padding_conv_params = {
      kTfLitePaddingSame,  // padding
      1,                    // stride_width
      1,                    // stride_height
      kTfLiteActNone,       // activation
      1,                    // dilation_width_factor
      1,                    // dilation_height_factor
  };

    tflite::testing::InitConvTestDataQuantized(filter_size, number_filters,
                                               input_channels, batches, input_size,
                                               kInputData,
                                               kFilterData,
                                               kBiasData);

    tflite::testing::InitGoldenDataPadding(filter_size, number_filters,
                                    input_channels, batches, input_size,
                                    kInputData,
                                    kFilterData,
                                    kBiasData,
                                    kGoldenData,
                                    stride, output_size, 2);

    int8_t input_quantized[kInputElements];
    int8_t filter_quantized[kFilterElements];
    int32_t bias_quantized[kBiasElements];
    int8_t golden_quantized[kOutputElements];
    int8_t output_data[kOutputElements];

    tflite::testing::TestConvQuantizedPerChannelPerformance(
        kInputShape, kInputData,
        input_quantized, input_scale, input_zero_point, kFilterShape,
        kFilterData, filter_quantized,
        kBiasShape, kBiasData, bias_quantized, scales, zero_points,
        kOutputShape, kGoldenData,
        golden_quantized, output_data, output_scale, output_zero_point,
        &padding_conv_params);
}

TF_LITE_MICRO_TESTS_END
