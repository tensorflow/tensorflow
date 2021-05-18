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

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
void ValidatePoolingGoldens(TfLiteTensor* tensors, int tensors_size,
                            const TfLiteRegistration registration,
                            const int filter_height, const int filter_width,
                            const int stride_height, const int stride_width,
                            const T* golden, const int output_length,
                            TfLitePadding padding,
                            TfLiteFusedActivation activation, T* output_data) {
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLitePoolParams builtin_data = {padding,
                                   stride_width,
                                   stride_height,
                                   filter_width,
                                   filter_height,
                                   activation,
                                   {}};

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], 1e-5f);
  }
}

void TestAveragePoolFloat(int* input_dims_data, const float* input_data,
                          const int filter_height, const int filter_width,
                          const int stride_height, const int stride_width,
                          const float* expected_output_data,
                          int* output_dims_data, TfLitePadding padding,
                          TfLiteFusedActivation activation,
                          float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_AVERAGE_POOL_2D();

  ValidatePoolingGoldens(tensors, tensors_size, registration, filter_height,
                         filter_width, stride_height, stride_width,
                         expected_output_data, output_dims_count, padding,
                         activation, output_data);
}

template <typename T>
void TestAveragePoolQuantized(
    int* input_dims_data, const T* input_data, const float input_scale,
    const int input_zero_point, const int filter_height, const int filter_width,
    const int stride_height, const int stride_width,
    const T* expected_output_data, int* output_dims_data,
    const float output_scale, const int output_zero_point,
    TfLitePadding padding, TfLiteFusedActivation activation, T* output_data) {
  static_assert(sizeof(T) == 1, "Only int8_t/uint8_t data types allowed.");

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, input_scale,
                            input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_AVERAGE_POOL_2D();
  ValidatePoolingGoldens(tensors, tensors_size, registration, filter_height,
                         filter_width, stride_height, stride_width,
                         expected_output_data, output_dims_count, padding,
                         activation, output_data);
}

void TestMaxPoolFloat(int* input_dims_data, const float* input_data,
                      int filter_width, int filter_height, int stride_width,
                      int stride_height, const float* expected_output_data,
                      int* output_dims_data, TfLitePadding padding,
                      TfLiteFusedActivation activation, float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_MAX_POOL_2D();
  ValidatePoolingGoldens(tensors, tensors_size, registration, filter_height,
                         filter_width, stride_height, stride_width,
                         expected_output_data, output_dims_count, padding,
                         activation, output_data);
}

template <typename T>
void TestMaxPoolQuantized(int* input_dims_data, const T* input_data,
                          const float input_scale, const int input_zero_point,
                          const int filter_height, const int filter_width,
                          const int stride_height, const int stride_width,
                          const T* expected_output_data, int* output_dims_data,
                          const float output_scale, const int output_zero_point,
                          TfLitePadding padding,
                          TfLiteFusedActivation activation, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, input_scale,
                            input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_MAX_POOL_2D();
  ValidatePoolingGoldens(tensors, tensors_size, registration, filter_height,
                         filter_width, stride_height, stride_width,
                         expected_output_data, output_dims_count, padding,
                         activation, output_data);
}

}  // namespace

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleAveragePoolTestFloat) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const float input_values[] = {0, 6, 2, 4, 3, 2, 10, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const float golden[] = {2.75, 5.75};
  int output_shape[] = {4, 1, 1, 2, 1};
  float output_data[2];
  tflite::testing::TestAveragePoolFloat(
      input_shape, input_values, filter_height, filter_width, stride_height,
      stride_width, golden, output_shape, kTfLitePaddingValid, kTfLiteActNone,
      output_data);
}

TF_LITE_MICRO_TEST(SimpleAveragePoolTestUint8) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const uint8_t input_values[] = {0, 24, 8, 16, 12, 8, 40, 28};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const uint8_t golden[] = {11, 23};
  int output_shape[] = {4, 1, 1, 2, 1};
  uint8_t output_data[2];

  const float input_scale = 0.25;
  const int input_zero_point = 0;
  const float output_scale = .25;
  const int output_zero_point = 0;
  tflite::testing::TestAveragePoolQuantized(
      input_shape, input_values, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActNone,
      output_data);
}

TF_LITE_MICRO_TEST(SimpleAveragePoolTestInt8PaddingValidStride2ActNone) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const int8_t input_values[] = {0, -24, 8, 16, 12, 8, -40, 28};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const int8_t golden[] = {-1, 3};
  int output_shape[] = {4, 1, 1, 2, 1};
  int8_t output_data[2];

  const float input_scale = .25;
  const int input_zero_point = 0;
  const float output_scale = .25;
  const int output_zero_point = 0;
  tflite::testing::TestAveragePoolQuantized(
      input_shape, input_values, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActNone,
      output_data);
}

TF_LITE_MICRO_TEST(SimpleAveragePoolTestInt8PaddingValidStride1Stride2Relu) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const int8_t input_values[] = {0, -24, 8, 16, 12, 8, -40, 28};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 1;
  const int stride_height = 2;
  const int8_t golden[] = {0, 0, 3};
  int output_shape[] = {4, 1, 1, 3, 1};
  int8_t output_data[3];

  const float input_scale = .25;
  const int input_zero_point = 0;
  const float output_scale = .25;
  const int output_zero_point = 0;
  tflite::testing::TestAveragePoolQuantized(
      input_shape, input_values, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActRelu,
      output_data);
}

TF_LITE_MICRO_TEST(
    SimpleAveragePoolTestInt8PaddingValidStride2Stride1ReluN1To1) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const int8_t input_values[] = {0, -24, 8, 16, 12, 8, -40, 28};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 1;
  const int8_t golden[] = {-1, 3};
  int output_shape[] = {4, 1, 1, 2, 1};
  int8_t output_data[2];

  const float input_scale = .25;
  const int input_zero_point = 0;
  const float output_scale = .25;
  const int output_zero_point = 0;
  tflite::testing::TestAveragePoolQuantized(
      input_shape, input_values, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActReluN1To1,
      output_data);
}

TF_LITE_MICRO_TEST(SimpleAveragePoolTestInt8PaddingValidStride2Relu6) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const int8_t input_values[] = {12, -24, 32, 16, 12, 8, 40, 28};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const int8_t golden[] = {2, 24};
  int output_shape[] = {4, 1, 1, 2, 1};
  int8_t output_data[2];

  const float input_scale = .25;
  const int input_zero_point = 0;
  const float output_scale = .25;
  const int output_zero_point = 0;
  tflite::testing::TestAveragePoolQuantized(
      input_shape, input_values, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActRelu6,
      output_data);
}

TF_LITE_MICRO_TEST(SimpleAveragePoolTestInt8PaddingSameStride1ActNone) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const int8_t input_values[] = {12, -24, 32, 16, 12, 8, 40, 28};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 1;
  const int stride_height = 1;
  const int8_t golden[] = {2, 14, 29, 22, 10, 24, 34, 28};
  int output_shape[] = {4, 1, 2, 4, 1};
  int8_t output_data[8];

  const float input_scale = .25;
  const int input_zero_point = 0;
  const float output_scale = .25;
  const int output_zero_point = 0;
  tflite::testing::TestAveragePoolQuantized(
      input_shape, input_values, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActNone,
      output_data);
}

TF_LITE_MICRO_TEST(SimpleMaxPoolTestFloat) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const float input_values[] = {0, 6, 2, 4, 3, 2, 10, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const float golden[] = {6, 10};
  int output_shape[] = {4, 1, 1, 2, 1};
  float output_data[2];
  tflite::testing::TestMaxPoolFloat(input_shape, input_values, filter_height,
                                    filter_width, stride_height, stride_width,
                                    golden, output_shape, kTfLitePaddingValid,
                                    kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleMaxPoolTestFloatRelu) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const float input_values[] = {-1, -6, 2, 4, -3, -2, 10.5, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const float golden[] = {0, 10.5};
  int output_shape[] = {4, 1, 1, 2, 1};
  float output_data[2];
  tflite::testing::TestMaxPoolFloat(input_shape, input_values, filter_height,
                                    filter_width, stride_height, stride_width,
                                    golden, output_shape, kTfLitePaddingValid,
                                    kTfLiteActRelu, output_data);
}

TF_LITE_MICRO_TEST(SimpleMaxPoolTestFloatReluN1To1) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const float input_values1[] = {-2.75, -6, 0.2, 0.4, -3, -2, -0.3, 0.7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const float golden1[] = {-1.0, 0.7};
  int output_shape[] = {4, 1, 1, 2, 1};
  float output_data[2];
  tflite::testing::TestMaxPoolFloat(input_shape, input_values1, filter_height,
                                    filter_width, stride_height, stride_width,
                                    golden1, output_shape, kTfLitePaddingValid,
                                    kTfLiteActReluN1To1, output_data);

  const float input_values2[] = {-2.75, -6, -2, -4, -3, -2, 10, -7};
  const float golden2[] = {-1.0, 1.0};
  tflite::testing::TestMaxPoolFloat(input_shape, input_values2, filter_height,
                                    filter_width, stride_height, stride_width,
                                    golden2, output_shape, kTfLitePaddingValid,
                                    kTfLiteActReluN1To1, output_data);
}

TF_LITE_MICRO_TEST(SimpleMaxPoolTestFloatRelu6) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const float input_values1[] = {-1.5, -6, 12, 4, -3, -2, 10, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const float golden1[] = {0, 6};
  int output_shape[] = {4, 1, 1, 2, 1};
  float output_data[2];
  tflite::testing::TestMaxPoolFloat(input_shape, input_values1, filter_height,
                                    filter_width, stride_height, stride_width,
                                    golden1, output_shape, kTfLitePaddingValid,
                                    kTfLiteActRelu6, output_data);

  const float input_values2[] = {0, 4.5, 12, 4, 3, 2, 10, 7};
  const float golden2[] = {4.5, 6};
  tflite::testing::TestMaxPoolFloat(input_shape, input_values2, filter_height,
                                    filter_width, stride_height, stride_width,
                                    golden2, output_shape, kTfLitePaddingValid,
                                    kTfLiteActRelu6, output_data);
}

TF_LITE_MICRO_TEST(SimpleMaxPoolTestPaddingSameStride1) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const float input_values[] = {0, 6, 2, 4, 3, 2, 10, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 1;
  const int stride_height = 1;
  const float golden[] = {6, 10, 10, 7, 3, 10, 10, 7};
  int output_shape[] = {4, 1, 2, 4, 1};
  float output_data[8];
  tflite::testing::TestMaxPoolFloat(input_shape, input_values, filter_height,
                                    filter_width, stride_height, stride_width,
                                    golden, output_shape, kTfLitePaddingSame,
                                    kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleMaxPoolTestPaddingValidStride1) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const float input_values[] = {0, 6, 2, 4, 3, 2, 10, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 1;
  const int stride_height = 1;
  const float golden[] = {6, 10, 10};
  int output_shape[] = {4, 1, 1, 3, 1};
  float output_data[8];
  tflite::testing::TestMaxPoolFloat(input_shape, input_values, filter_height,
                                    filter_width, stride_height, stride_width,
                                    golden, output_shape, kTfLitePaddingValid,
                                    kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleMaxPoolTestUInt8ActNone) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const uint8_t input_values[] = {0, 12, 4, 8, 6, 4, 20, 14};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const uint8_t golden[] = {12, 20};
  int output_shape[] = {4, 1, 1, 2, 1};
  uint8_t output_data[2];

  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;
  tflite::testing::TestMaxPoolQuantized(
      input_shape, input_values, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActNone,
      output_data);
}

TF_LITE_MICRO_TEST(MaxPoolTestUInt8ActRelu) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const uint8_t input_values[] = {0, 4, 2, 4, 3, 2, 14, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const uint8_t golden[] = {4, 14};
  int output_shape[] = {4, 1, 1, 2, 1};
  uint8_t output_data[2];

  const float input_scale = 1.0;
  const int input_zero_point = 4;
  const float output_scale = 1.0;
  const int output_zero_point = 4;
  tflite::testing::TestMaxPoolQuantized(
      input_shape, input_values, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActRelu,
      output_data);
}

TF_LITE_MICRO_TEST(MaxPoolTestUInt8ActReluN1To1) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const uint8_t input_values[] = {0, 4, 2, 4, 3, 2, 14, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const uint8_t golden[] = {3, 5};
  int output_shape[] = {4, 1, 1, 2, 1};
  uint8_t output_data[2];

  const float input_scale = 1.0;
  const int input_zero_point = 4;
  const float output_scale = 1.0;
  const int output_zero_point = 4;
  tflite::testing::TestAveragePoolQuantized(
      input_shape, input_values, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActReluN1To1,
      output_data);
}

TF_LITE_MICRO_TEST(MaxPoolTestUInt8ActRelu6) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const uint8_t input_values1[] = {12, 0, 36, 20, 6, 8, 32, 26};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const uint8_t golden1[] = {12, 24};
  int output_shape[] = {4, 1, 1, 2, 1};
  uint8_t output_data[8];

  const float input_scale = 0.5;
  const int input_zero_point = 12;
  const float output_scale = 0.5;
  const int output_zero_point = 12;
  tflite::testing::TestMaxPoolQuantized(
      input_shape, input_values1, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden1, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActRelu6,
      output_data);

  const uint8_t input_values2[] = {12, 21, 36, 16, 18, 16, 32, 26};

  const uint8_t golden2[] = {21, 24};
  tflite::testing::TestMaxPoolQuantized(
      input_shape, input_values2, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden2, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActRelu6,
      output_data);
}

TF_LITE_MICRO_TEST(MaxPoolTestUInt8PaddingSameStride1) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const uint8_t input_values1[] = {0, 6, 2, 4, 3, 2, 10, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 1;
  const int stride_height = 1;
  const uint8_t golden1[] = {6, 10, 10, 7, 3, 10, 10, 7};
  int output_shape[] = {4, 1, 2, 4, 1};
  uint8_t output_data[8];

  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;
  tflite::testing::TestMaxPoolQuantized(
      input_shape, input_values1, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden1, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActNone,
      output_data);
}

TF_LITE_MICRO_TEST(MaxPoolTestUInt8PaddingValidStride1) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const uint8_t input_values1[] = {0, 6, 2, 4, 3, 2, 10, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 1;
  const int stride_height = 1;
  const uint8_t golden1[] = {6, 10, 10};
  int output_shape[] = {4, 1, 1, 3, 1};
  uint8_t output_data[3];

  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;
  tflite::testing::TestMaxPoolQuantized(
      input_shape, input_values1, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden1, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActNone,
      output_data);
}

TF_LITE_MICRO_TEST(SimpleMaxPoolTestInt8ActNone) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const int8_t input_values1[] = {0, 6, 2, 4, 3, 2, 10, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const int8_t golden1[] = {6, 10};
  int output_shape[] = {4, 1, 1, 2, 1};
  int8_t output_data[2];

  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;
  tflite::testing::TestMaxPoolQuantized(
      input_shape, input_values1, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden1, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActNone,
      output_data);
}

TF_LITE_MICRO_TEST(MaxPoolTestInt8ActRelu) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const int8_t input_values1[] = {-3, -12, 4, 8, -6, -4, 20, 14};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const int8_t golden1[] = {0, 20};
  int output_shape[] = {4, 1, 1, 2, 1};
  int8_t output_data[2];

  const float input_scale = 0.5;
  const int input_zero_point = 0;
  const float output_scale = 0.5;
  const int output_zero_point = 0;
  tflite::testing::TestMaxPoolQuantized(
      input_shape, input_values1, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden1, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActRelu,
      output_data);
}

TF_LITE_MICRO_TEST(MaxPoolTestInt8ActReluN1To1) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const int8_t input_values1[] = {-2, -6, -2, -4, -3, -2, 10, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const int8_t golden1[] = {-1, 1};
  int output_shape[] = {4, 1, 1, 2, 1};
  int8_t output_data[2];

  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;
  tflite::testing::TestMaxPoolQuantized(
      input_shape, input_values1, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden1, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActReluN1To1,
      output_data);
}

TF_LITE_MICRO_TEST(MaxPoolTestInt8ActRelu6) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const int8_t input_values1[] = {0, -6, 12, 4, -3, -2, 10, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 2;
  const int stride_height = 2;
  const int8_t golden1[] = {0, 6};
  int output_shape[] = {4, 1, 1, 2, 1};
  int8_t output_data[2];

  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;
  tflite::testing::TestMaxPoolQuantized(
      input_shape, input_values1, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden1, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActRelu6,
      output_data);
}

TF_LITE_MICRO_TEST(MaxPoolTestUInt8PaddingSameStride1) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const uint8_t input_values1[] = {0, 6, 2, 4, 3, 2, 10, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 1;
  const int stride_height = 1;
  const uint8_t golden1[] = {6, 10, 10, 7, 3, 10, 10, 7};
  int output_shape[] = {4, 1, 2, 4, 1};
  uint8_t output_data[8];

  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;
  tflite::testing::TestMaxPoolQuantized(
      input_shape, input_values1, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden1, output_shape,
      output_scale, output_zero_point, kTfLitePaddingSame, kTfLiteActNone,
      output_data);
}

TF_LITE_MICRO_TEST(MaxPoolTestUInt8PaddingValidStride1) {
  int input_shape[] = {4, 1, 2, 4, 1};
  const uint8_t input_values1[] = {0, 6, 2, 4, 3, 2, 10, 7};
  const int filter_width = 2;
  const int filter_height = 2;
  const int stride_width = 1;
  const int stride_height = 1;
  const uint8_t golden1[] = {6, 10, 10};
  int output_shape[] = {4, 1, 1, 3, 1};
  uint8_t output_data[3];

  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;
  tflite::testing::TestMaxPoolQuantized(
      input_shape, input_values1, input_scale, input_zero_point, filter_height,
      filter_width, stride_height, stride_width, golden1, output_shape,
      output_scale, output_zero_point, kTfLitePaddingValid, kTfLiteActNone,
      output_data);
}

TF_LITE_MICRO_TESTS_END
