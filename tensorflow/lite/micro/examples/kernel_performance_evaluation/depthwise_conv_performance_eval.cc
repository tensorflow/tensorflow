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

constexpr int kMaxFilterChannels = 64;
constexpr int kMaxBiasChannels = 64;

// Index of the output tensor in context->tensors, specific to
// DepthwiseConv.
constexpr int kOutputTensorIndex = 3;

static const int benchmarking_iterations = 1;
static const int number_of_invocations = 1000;

static const int batches = 4;
static const int input_size = 32;
static const int input_channels = 6;
static const int kInputElements = batches*input_size*input_size*input_channels;
static const int input_shape[] = {4, batches, input_size, input_size, input_channels};
static float kInputData[kInputElements];

static const int filter_size = 7;
static const int depth_multiplier = 2;
static const int output_depth = input_channels*depth_multiplier;
static const int kFilterElements = 1*filter_size*filter_size*output_depth;
static const int filter_shape[] = {4, 1, filter_size, filter_size, output_depth};
static float kFilterData[kFilterElements];

static const int bias_shape[] = {4, 1, 1, 1, output_depth};
static const int kBiasElements = output_depth;
static float kBiasData[output_depth];

static const int stride = 1;
static const int output_size = ((input_size - filter_size) / stride) + 1;

const int output_shape[] = {4, batches, output_size, output_size, output_depth};
const int kOutputElements = batches*output_size*output_size*output_depth;
static float kGoldenData[kOutputElements];


void InitDepthwiseConvTestDataQuantized(
    const int filter_size, const int output_channels,
    const int input_channels, const int batches, const int input_size,
    float* input_data,
    float* filters_data,
    float* bias_data) {
  const RuntimeShape input_shape = {batches, input_size, input_size, input_channels};
  const RuntimeShape filter_shape = {1, filter_size, filter_size, output_channels};

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
        filters_data[Offset(filter_shape, 0, filter_y, filter_x, out_channel)] = filter_x + filter_y;
      }
    }
  }
}

void InitGoldenData(const int filter_size, const int output_channels,
                    const int input_channels, const int batches, const int input_size,
                    float* input_data,
                    float* filter_data,
                    float* bias_data,
                    float* expected_output,
                    int stride, int output_size, int padding) {
  const RuntimeShape input_shape = {batches, input_size, input_size, input_channels};
  const RuntimeShape filter_shape = {1, filter_size, filter_size, output_channels};
  const RuntimeShape output_shape = {batches, output_size, output_size, output_channels};
  const int dilation = 1;

  for (int b = 0; b < batches; ++b) {
    for (int out_y = 0; out_y < output_size; ++out_y) {
      for (int out_x = 0; out_x < output_size; ++out_x) {
        for (int ic = 0; ic < input_channels; ++ic) {
          for (int m = 0; m < depth_multiplier; m++) {
            const int oc = m + ic * depth_multiplier;
            const int in_x_origin = (out_x * stride) - padding;
            const int in_y_origin = (out_y * stride) - padding;
            float total = 0.f;
            for (int filter_y = 0; filter_y < filter_size; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_size; ++filter_x) {
                const int in_x = in_x_origin + dilation * filter_x;
                const int in_y =
                    in_y_origin + dilation * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_size) && (in_y >= 0) &&
                    (in_y < input_size)) {
                  float input_value =
                      input_data[Offset(input_shape, b, in_y, in_x, ic)];
                  float filter_value = filter_data[Offset(
                      filter_shape, 0, filter_y, filter_x, oc)];
                  total += (input_value * filter_value);
                }
              }
            }
            float bias_value = 0.0f;
            if (bias_data) {
              bias_value = bias_data[oc];
            }
            expected_output[Offset(output_shape, b, out_y, out_x, oc)] = total + bias_value;
          }
        }
      }
    }
  }
}


template <typename T>
TfLiteStatus ValidateDepthwiseConvGoldens(const T* expected_output_data,
                                          int output_length,
                                          float tolerance, int tensors_size,
                                          TfLiteTensor* tensors, TfLiteDepthwiseConvParams& builtin_data,
                                          const int number_of_invocations) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
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
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_ENSURE_OK(context, registration->prepare(&context, &node));
  }

  // Start main benchmarking loop
  // Increase benchmarking iterations to make result representative
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
  micro_test::reporter->Report("Avg Invoke run time %d invocations =  %d us", number_of_invocations, duration);

  if (registration->free) {
    registration->free(&context, user_data);
  }

  const T* output_data = tflite::GetTensorData<T>(&tensors[kOutputTensorIndex]);
  int fails = 0;
  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
    if( micro_test::did_test_fail && ++fails > 5)
      return kTfLiteOk;
  }

  
  if (registration->free) {
    registration->free(&context, user_data);
  }

  return kTfLiteOk;
}

void TestDepthwiseConvQuantizedPerLayer(
    const int* input_dims_data, const float* input_data,
    uint8_t* input_quantized, float input_scale, int input_zero_point,
    const int* filter_dims_data, const float* filter_data,
    uint8_t* filter_quantized, float filter_scale, int filter_zero_point,
    const int* bias_dims_data, const float* bias_data, int32_t* bias_quantized,
    const float* golden, uint8_t* golden_quantized, const int* output_dims_data,
    uint8_t* output_data, float output_scale, int output_zero_point,
    TfLiteDepthwiseConvParams& builtin_data, const int number_of_invocations) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      tflite::testing::CreateQuantizedTensor(input_data, input_quantized,
                                             input_dims, input_scale,
                                             input_zero_point),
      tflite::testing::CreateQuantizedTensor(
          filter_data, filter_quantized, filter_dims, filter_scale,
          filter_zero_point),
      tflite::testing::CreateQuantizedBiasTensor(bias_data, bias_quantized,
                                                 bias_dims, input_scale,
                                                 filter_scale),
      tflite::testing::CreateQuantizedTensor(output_data, output_dims,
                                             output_scale, output_zero_point),
  };

  // TODO(njeff): Affine Quantization Params should be set on tensor creation.
  float filter_scales[] = {1, filter_scale};
  int filter_zero_points[] = {1, 128};
  TfLiteAffineQuantization filter_quant = {
      FloatArrayFromFloats(filter_scales),
      IntArrayFromInts(filter_zero_points)};
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant};

  float bias_scales[] = {1, filter_scale * input_scale};
  int bias_zero_points[] = {1, 128};
  TfLiteAffineQuantization bias_quant = {FloatArrayFromFloats(bias_scales),
                                         IntArrayFromInts(bias_zero_points)};
  tensors[2].quantization = {kTfLiteAffineQuantization, &bias_quant};

  AsymmetricQuantize(golden, golden_quantized, output_dims_count, output_scale,
                     output_zero_point);
  ValidateDepthwiseConvGoldens(golden_quantized, output_dims_count,
                               1.0, tensors_size, tensors, builtin_data, number_of_invocations);
}

void TestDepthwiseConvQuantizedPerChannel(
    const int* input_dims_data, const float* input_data,
    int8_t* input_quantized, float input_scale, int input_zero_point,
    const int* filter_dims_data, const float* filter_data,
    int8_t* filter_data_quantized, const int* bias_dims_data,
    const float* bias_data, int32_t* bias_data_quantized,
    const int* output_dims_data, const float* expected_output_data,
    int8_t* expected_output_data_quantized, int8_t* output_data,
    float output_scale, int output_zero_point,
    TfLiteDepthwiseConvParams& builtin_data, const int number_of_invocations) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  int filter_zero_points[kMaxFilterChannels];
  float filter_scales[kMaxFilterChannels];
  int bias_zero_points[kMaxBiasChannels];
  float bias_scales[kMaxBiasChannels];
  TfLiteAffineQuantization filter_quant;
  TfLiteAffineQuantization bias_quant;
  TfLiteTensor input_tensor =
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point);
  TfLiteTensor filter_tensor = CreateSymmetricPerChannelQuantizedTensor(
      filter_data, filter_data_quantized, filter_dims, filter_scales,
      filter_zero_points, &filter_quant, 3 /* quantized dimension */);
  TfLiteTensor bias_tensor = CreatePerChannelQuantizedBiasTensor(
      bias_data, bias_data_quantized, bias_dims, input_scale, &filter_scales[1],
      bias_scales, bias_zero_points, &bias_quant, 3 /* quantized dimension */);
  TfLiteTensor output_tensor =
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            input_zero_point);

  // TODO(njeff): Affine Quantization Params should be set on tensor creation.
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

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  AsymmetricQuantize(expected_output_data, expected_output_data_quantized,
                     output_dims_count, output_scale, output_zero_point);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, ValidateDepthwiseConvGoldens(expected_output_data_quantized,
                                              output_dims_count,
                                              1.0, tensors_size, tensors, builtin_data, number_of_invocations));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(Int8Performance) {

  tflite::testing::InitDepthwiseConvTestDataQuantized(tflite::testing::filter_size, tflite::testing::output_depth,
                                               tflite::testing::input_channels, tflite::testing::batches, tflite::testing::input_size,
                                               tflite::testing::kInputData,
                                               tflite::testing::kFilterData,
                                               tflite::testing::kBiasData);

  tflite::testing::InitGoldenData(tflite::testing::filter_size, tflite::testing::output_depth,
                                  tflite::testing::input_channels, tflite::testing::batches, tflite::testing::input_size,
                                  tflite::testing::kInputData,
                                  tflite::testing::kFilterData,
                                  tflite::testing::kBiasData,
                                  tflite::testing::kGoldenData,
                                  tflite::testing::stride, tflite::testing::output_size, 0);

  int8_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t golden_quantized[tflite::testing::kOutputElements];
  int8_t output_data[tflite::testing::kOutputElements];

  TfLiteDepthwiseConvParams builtin_data;
  builtin_data.padding = kTfLitePaddingValid;
  builtin_data.activation = kTfLiteActNone;
  builtin_data.stride_height = 1;
  builtin_data.stride_width = 1;
  builtin_data.dilation_height_factor = 1;
  builtin_data.dilation_width_factor = 1;
  builtin_data.depth_multiplier = tflite::testing::depth_multiplier;

  const float input_scale = 0.5;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      tflite::testing::input_shape, tflite::testing::kInputData, input_quantized, input_scale, input_zero_point,
      tflite::testing::filter_shape, tflite::testing::kFilterData, filter_quantized, tflite::testing::bias_shape, tflite::testing::kBiasData,
      bias_quantized, tflite::testing::output_shape, tflite::testing::kGoldenData, golden_quantized, output_data,
      output_scale, output_zero_point, builtin_data, tflite::testing::number_of_invocations);
}

TF_LITE_MICRO_TEST(Uint8Performance) {

  tflite::testing::InitDepthwiseConvTestDataQuantized(tflite::testing::filter_size, tflite::testing::output_depth,
                                               tflite::testing::input_channels, tflite::testing::batches, tflite::testing::input_size,
                                               tflite::testing::kInputData,
                                               tflite::testing::kFilterData,
                                               tflite::testing::kBiasData);

  tflite::testing::InitGoldenData(tflite::testing::filter_size, tflite::testing::output_depth,
                                  tflite::testing::input_channels, tflite::testing::batches, tflite::testing::input_size,
                                  tflite::testing::kInputData,
                                  tflite::testing::kFilterData,
                                  tflite::testing::kBiasData,
                                  tflite::testing::kGoldenData,
                                  tflite::testing::stride, tflite::testing::output_size, 0);

  const float input_scale = 0.5f;
  const int input_zero_point = 128;
  const float filter_scale = 0.5f;
  const int filter_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;

  uint8_t input_quantized[tflite::testing::kInputElements];
  uint8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  uint8_t golden_quantized[tflite::testing::kOutputElements];
  uint8_t output_data[tflite::testing::kOutputElements];

  TfLiteDepthwiseConvParams builtin_data;
  builtin_data.padding = kTfLitePaddingValid;
  builtin_data.activation = kTfLiteActNone;
  builtin_data.stride_height = 1;
  builtin_data.stride_width = 1;
  builtin_data.dilation_height_factor = 1;
  builtin_data.dilation_width_factor = 1;
  builtin_data.depth_multiplier = tflite::testing::depth_multiplier;

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      tflite::testing::input_shape, tflite::testing::kInputData, input_quantized, input_scale, input_zero_point,
      tflite::testing::filter_shape, tflite::testing::kFilterData, filter_quantized, filter_scale,
      filter_zero_point, tflite::testing::bias_shape, tflite::testing::kBiasData, bias_quantized, tflite::testing::kGoldenData,
      golden_quantized, tflite::testing::output_shape, output_data, output_scale,
      output_zero_point, builtin_data, tflite::testing::number_of_invocations);
}

TF_LITE_MICRO_TEST(Uint8PerformanceSpecialImplementation) {

  const int batches = 4;
  const int input_size = 64;
  const int input_channels = 1;
  const int kInputElements = batches*input_size*input_size*input_channels;
  const int input_shape[] = {4, batches, input_size, input_size, input_channels};
  float kInputData[kInputElements];

  const int filter_size = 8;
  const int depth_multiplier = 2;
  const int output_depth = input_channels*depth_multiplier;
  const int kFilterElements = 1*filter_size*filter_size*output_depth;
  const int filter_shape[] = {4, 1, filter_size, filter_size, output_depth};
  float kFilterData[kFilterElements];

  const int bias_shape[] = {4, 1, 1, 1, output_depth};
  const int kBiasElements = output_depth;
  float kBiasData[output_depth];

  const int stride = 1;
  const int output_size = ((input_size - filter_size) / stride) + 1;

  int output_shape[] = {4, batches, output_size, output_size, output_depth};
  const int kOutputElements = batches*output_size*output_size*output_depth;
  float kGoldenData[kOutputElements];

  TF_LITE_MICRO_EXPECT(output_depth * filter_size * filter_size * input_channels <= 1024);

  tflite::testing::InitDepthwiseConvTestDataQuantized(filter_size, output_depth,
                                               input_channels, batches, input_size,
                                               kInputData,
                                               kFilterData,
                                               kBiasData);

  tflite::testing::InitGoldenData(filter_size, output_depth,
                                  input_channels, batches, input_size,
                                  kInputData,
                                  kFilterData,
                                  kBiasData,
                                  kGoldenData,
                                  stride, output_size, 0);

  const float input_scale = 0.5f;
  const int input_zero_point = 0;
  const float filter_scale = 0.5f;
  const int filter_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;

  uint8_t input_quantized[kInputElements];
  uint8_t filter_quantized[kFilterElements];
  int32_t bias_quantized[kBiasElements];
  uint8_t golden_quantized[kOutputElements];
  uint8_t output_data[kOutputElements];

  TfLiteDepthwiseConvParams builtin_data;
  builtin_data.padding = kTfLitePaddingValid;
  builtin_data.activation = kTfLiteActNone;
  builtin_data.stride_height = 1;
  builtin_data.stride_width = 1;
  builtin_data.dilation_height_factor = 1;
  builtin_data.dilation_width_factor = 1;
  builtin_data.depth_multiplier = tflite::testing::depth_multiplier;

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, kInputData, input_quantized, input_scale, input_zero_point,
      filter_shape, kFilterData, filter_quantized, filter_scale,
      filter_zero_point, bias_shape, kBiasData, bias_quantized, kGoldenData,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, builtin_data, tflite::testing::number_of_invocations);
}

TF_LITE_MICRO_TEST(Int8PaddingPerformance) {

  tflite::testing::InitDepthwiseConvTestDataQuantized(tflite::testing::filter_size, tflite::testing::output_depth,
                                               tflite::testing::input_channels, tflite::testing::batches, tflite::testing::input_size,
                                               tflite::testing::kInputData,
                                               tflite::testing::kFilterData,
                                               tflite::testing::kBiasData);

  const int output_size = tflite::testing::input_size;

  const int output_shape[] = {4, tflite::testing::batches, output_size, output_size, tflite::testing::output_depth};
  const int kOutputElements = tflite::testing::batches*output_size*output_size*tflite::testing::output_depth;
  float kGoldenData[kOutputElements];

  tflite::testing::InitGoldenData(tflite::testing::filter_size, tflite::testing::output_depth,
                                  tflite::testing::input_channels, tflite::testing::batches, tflite::testing::input_size,
                                  tflite::testing::kInputData,
                                  tflite::testing::kFilterData,
                                  tflite::testing::kBiasData,
                                  kGoldenData,
                                  tflite::testing::stride, output_size, (tflite::testing::filter_size-1)/2);

  int8_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t golden_quantized[kOutputElements];
  int8_t output_data[kOutputElements];

  TfLiteDepthwiseConvParams builtin_data;
  builtin_data.padding = kTfLitePaddingSame;
  builtin_data.activation = kTfLiteActNone;
  builtin_data.stride_height = 1;
  builtin_data.stride_width = 1;
  builtin_data.dilation_height_factor = 1;
  builtin_data.dilation_width_factor = 1;
  builtin_data.depth_multiplier = tflite::testing::depth_multiplier;

  const float input_scale = 0.5;
  const float output_scale = 1.0f;
  const int input_zero_point = -2;
  const int output_zero_point = -2;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      tflite::testing::input_shape, tflite::testing::kInputData, input_quantized, input_scale, input_zero_point,
      tflite::testing::filter_shape, tflite::testing::kFilterData, filter_quantized, tflite::testing::bias_shape, tflite::testing::kBiasData,
      bias_quantized, output_shape, kGoldenData, golden_quantized, output_data,
      output_scale, output_zero_point, builtin_data, tflite::testing::number_of_invocations);
}

TF_LITE_MICRO_TEST(Uint8PaddingPerformance) {

  tflite::testing::InitDepthwiseConvTestDataQuantized(tflite::testing::filter_size, tflite::testing::output_depth,
                                               tflite::testing::input_channels, tflite::testing::batches, tflite::testing::input_size,
                                               tflite::testing::kInputData,
                                               tflite::testing::kFilterData,
                                               tflite::testing::kBiasData);

  const int output_size = tflite::testing::input_size;
  const int output_shape[] = {4, tflite::testing::batches, output_size, output_size, tflite::testing::output_depth};
  const int kOutputElements = tflite::testing::batches*output_size*output_size*tflite::testing::output_depth;
  float kGoldenData[kOutputElements];

  tflite::testing::InitGoldenData(tflite::testing::filter_size, tflite::testing::output_depth,
                                  tflite::testing::input_channels, tflite::testing::batches, tflite::testing::input_size,
                                  tflite::testing::kInputData,
                                  tflite::testing::kFilterData,
                                  tflite::testing::kBiasData,
                                  kGoldenData,
                                  tflite::testing::stride, output_size, (tflite::testing::filter_size-1)/2);

  const float input_scale = 0.5f;
  const int input_zero_point = 128;
  const float filter_scale = 0.5f;
  const int filter_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;

  uint8_t input_quantized[tflite::testing::kInputElements];
  uint8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  uint8_t golden_quantized[kOutputElements];
  uint8_t output_data[kOutputElements];

  TfLiteDepthwiseConvParams builtin_data;
  builtin_data.padding = kTfLitePaddingSame;
  builtin_data.activation = kTfLiteActNone;
  builtin_data.stride_height = 1;
  builtin_data.stride_width = 1;
  builtin_data.dilation_height_factor = 1;
  builtin_data.dilation_width_factor = 1;
  builtin_data.depth_multiplier = tflite::testing::depth_multiplier;

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      tflite::testing::input_shape, tflite::testing::kInputData, input_quantized, input_scale, input_zero_point,
      tflite::testing::filter_shape, tflite::testing::kFilterData, filter_quantized, filter_scale,
      filter_zero_point, tflite::testing::bias_shape, tflite::testing::kBiasData, bias_quantized, kGoldenData,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, builtin_data, tflite::testing::number_of_invocations);
}

TF_LITE_MICRO_TESTS_END
