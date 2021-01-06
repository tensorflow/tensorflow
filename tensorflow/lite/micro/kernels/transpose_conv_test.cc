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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/conv_test.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// Common inputs and outputs.
constexpr int kInputElements = 32;
static const int kInputShape[] = {4, 1, 4, 4, 2};
static const float kInputData[kInputElements] = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

constexpr int kFilterElements = 18;
static const int kFilterShape[] = {4, 1, 3, 3, 2};
static const float kFilterData[kFilterElements] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

constexpr int kBiasElements = 1;
static const int kBiasShape[] = {4, 1, 1, 1, 1};
static const float kBiasData[kBiasElements] = {0};

constexpr int kOutputElements = 16;
static const int kOutputShape[] = {4, 1, 4, 4, 1};
static const float kGoldenData[kOutputElements] = {
    184,  412,  568,  528,  678,  1347, 1689, 1434,
    1494, 2715, 3057, 2442, 1968, 3352, 3652, 2760};

// Transpose conv uses TfLiteConvParams.
static TfLiteConvParams common_conv_params = {kTfLitePaddingSame,  // padding
                                              1,  // stride_width
                                              1,  // stride_height
                                              kTfLiteActNone,
                                              1,
                                              1};

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleTestFloat) {
  float output_data[tflite::testing::kOutputElements];

  tflite::testing::TestConvFloat(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      tflite::testing::kFilterShape, tflite::testing::kFilterData,
      tflite::testing::kBiasShape, tflite::testing::kBiasData,
      tflite::testing::kOutputShape, tflite::testing::kGoldenData,
      &tflite::testing::common_conv_params,
      tflite::Register_TRANSPOSE_CONV_2D(), output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannel) {
  int8_t output_data[tflite::testing::kOutputElements];

  const float input_scale = 0.5f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t golden_quantized[tflite::testing::kOutputElements];
  int zero_points[tflite::testing::kBiasElements + 1];
  float scales[tflite::testing::kBiasElements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilterShape, tflite::testing::kFilterData,
      filter_quantized, tflite::testing::kBiasShape, tflite::testing::kBiasData,
      bias_quantized, scales, zero_points, tflite::testing::kOutputShape,
      tflite::testing::kGoldenData, golden_quantized, output_scale,
      output_zero_point, &tflite::testing::common_conv_params,
      tflite::Register_TRANSPOSE_CONV_2D(), output_data);
}

TF_LITE_MICRO_TEST(InputOutputDifferentTypeIsError) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  TfLiteIntArray* input_dims = IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims = IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* bias_dims = IntArrayFromInts(tflite::testing::kBiasShape);
  TfLiteIntArray* output_dims = IntArrayFromInts(tflite::testing::kOutputShape);
  const int output_dims_count = tflite::ElementCount(*output_dims);
  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;

  int8_t output_data[tflite::testing::kOutputElements];
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(tflite::testing::kInputData, input_dims),
      CreateTensor(tflite::testing::kFilterData, filter_dims),
      CreateTensor(tflite::testing::kBiasData, bias_dims),
      CreateQuantizedTensor(output_data, output_dims, /*scale=*/1.0f,
                            /*zero_point=*/0),
  };
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError, tflite::testing::InvokeConv(
                        tensors, tensors_size, output_dims_count,
                        &tflite::testing::common_conv_params,
                        tflite::Register_TRANSPOSE_CONV_2D(), output_data));
}

TF_LITE_MICRO_TEST(HybridModeIsError) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  TfLiteIntArray* input_dims = IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims = IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* bias_dims = IntArrayFromInts(tflite::testing::kBiasShape);
  TfLiteIntArray* output_dims = IntArrayFromInts(tflite::testing::kOutputShape);
  const int output_dims_count = tflite::ElementCount(*output_dims);
  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;

  int8_t filter_data[tflite::testing::kFilterElements] = {};
  float output_data[tflite::testing::kOutputElements];
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(tflite::testing::kInputData, input_dims),
      CreateQuantizedTensor(filter_data, filter_dims,
                            /*scale=*/1.0f,
                            /*zero_point=*/0),
      CreateTensor(tflite::testing::kBiasData, bias_dims),
      CreateTensor(output_data, output_dims),
  };
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError, tflite::testing::InvokeConv(
                        tensors, tensors_size, output_dims_count,
                        &tflite::testing::common_conv_params,
                        tflite::Register_TRANSPOSE_CONV_2D(), output_data));
}

TF_LITE_MICRO_TESTS_END
