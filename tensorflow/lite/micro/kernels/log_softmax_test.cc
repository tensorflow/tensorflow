/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

void ExecuteLogSoftmaxTest(int tensors_count, TfLiteTensor* tensors) {
  int kInputArrayData[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  int kOutputArrayData[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  const TfLiteRegistration registration = tflite::Register_LOG_SOFTMAX();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void TestLogSoftmax(const float tolerance, int* input_dims_data,
                    const T* input_data, int* expected_dims,
                    const T* expected_data, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  TfLiteTensor tensors[] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };
  constexpr int kTensorsCount = std::extent<decltype(tensors)>::value;
  ExecuteLogSoftmaxTest(kTensorsCount, tensors);

  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], tolerance);
  }
}

// min/max are used to compute scale, zero-point
template <typename T>
struct TestLogSoftmaxParams {
  // quantization parameters
  float data_min;   // input and output data minimum value
  float data_max;   // input and output data maximum value
  T* input_data;    // quantized input storage
  T* output_data;   // quantized output storage
  float tolerance;  // maximum compare difference
};

template <typename T>
void TestLogSoftmaxQuantized(const TestLogSoftmaxParams<T>& params,
                             int* input_dims_data, const float* input_data,
                             int* expected_dims, const float* expected_data,
                             const T* expected_data_quantized,
                             float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  constexpr float kOutputScale = 16.0 / 256;
  constexpr int kOutputZeroPoint = 127;
  const float scale = ScaleFromMinMax<T>(params.data_min, params.data_max);
  const int zero_point =
      ZeroPointFromMinMax<T>(params.data_min, params.data_max);

  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input_data, params.input_data, input_dims, scale,
                            zero_point),
      CreateQuantizedTensor(params.output_data, output_dims, kOutputScale,
                            kOutputZeroPoint),
  };
  constexpr int kTensorsCount = std::extent<decltype(tensors)>::value;

  ExecuteLogSoftmaxTest(kTensorsCount, tensors);

  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_EQ(expected_data_quantized[i], params.output_data[i]);
  }
  Dequantize(params.output_data, output_count, kOutputScale, kOutputZeroPoint,
             output_data);
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i],
                              params.tolerance);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

// This contains the same test values as the Softmax test, but reference answer
// generated via the following snippet of python:
//   logits1 = tf.constant([[0, -6, 2, 4],[3, -2, 10, 1]], dtype=tf.float32)
//   logits2 = tf.constant([[0,-6],[2,4],[3,-2],[10,1]], dtype=tf.float32)
//   lsm1 = tf.nn.log_softmax(logits1)
//   lsm2 = tf.nn.log_softmax(logits2)
//   with tf.Session() as sess:
//     print('lsm1', sess.run(lsm1))
//     print('lsm2', sess.run(lsm2))
TF_LITE_MICRO_TEST(FloatActivationsOpTestLogSoftmax) {
  int kDims1[] = {2, 2, 4};
  constexpr float kInput[] = {
      0, -6, 2, 4, 3, -2, 10, 1,
  };
  constexpr float kExpect1[] = {
      -4.14297, -10.14297, -2.14297,   -.142971,  //
      -7.00104, -12.00104, -.00104087, -9.00104,  //
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect1)>::value;
  float output_data[kOutputCount];

  constexpr float kTolerance = 1e-5;

  tflite::testing::TestLogSoftmax(kTolerance, kDims1, kInput, kDims1, kExpect1,
                                  output_data);

  // Same input, but a different shape.
  int kDims2[] = {2, 4, 2};
  constexpr float kExpect2[] = {
      -.00247565, -6.00247, -2.12692,    -.126928,
      -.00671534, -5.00671, -.000123374, -9.00012,
  };

  tflite::testing::TestLogSoftmax(kTolerance, kDims2, kInput, kDims2, kExpect2,
                                  output_data);
}

TF_LITE_MICRO_TEST(LogSoftmaxOpTestSimpleTest) {
  int kDims[] = {2, 2, 5};
  constexpr float kInput[] = {
      1.0,  2.0,  3.0,  4.0,  5.0,   //
      -1.0, -2.0, -3.0, -4.0, -5.0,  //
  };
  constexpr float kExpect[] = {
      -4.45191431, -3.45191431, -2.45191431, -1.45191443, -0.4519144,  //
      -0.4519144,  -1.45191443, -2.45191431, -3.45191431, -4.45191431  //
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr float kTolerance = 1e-6;

  tflite::testing::TestLogSoftmax(kTolerance, kDims, kInput, kDims, kExpect,
                                  output_data);
}

TF_LITE_MICRO_TEST(QuantizedActivationsOpTestLogSoftmaxInt8) {
  int kDims[] = {2, 2, 4};
  constexpr float kInput[] = {
      0, -6, 2, 4, 3, -2, 10, 1,
  };
  constexpr float kExpect[] = {
      -4.14297, -10.14297, -2.14297,   -.142971,
      -7.00104, -12.00104, -.00104087, -9.00104,
  };
  constexpr int8_t kExpectQuantized[] = {
      61, -36, 93, 125, 15, -65, 127, -16,
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  // setup quantization storage and parameters
  int8_t q_output_data[kOutputCount];
  int8_t q_input_data[kOutputCount];
  constexpr float kMin = -10;
  constexpr float kMax = 10;
  constexpr float kLogSoftmaxQuantizedTolerance = 0.06355;
  tflite::testing::TestLogSoftmaxParams<int8_t> params = {};
  params.data_min = kMin;
  params.data_max = kMax;
  params.input_data = q_input_data;
  params.output_data = q_output_data;
  params.tolerance = kLogSoftmaxQuantizedTolerance;

  tflite::testing::TestLogSoftmaxQuantized(
      params, kDims, kInput, kDims, kExpect, kExpectQuantized, output_data);
}

TF_LITE_MICRO_TEST(ExtraTestLogSoftmaxInt8) {
  int kDims[] = {2, 3, 1};
  constexpr float kInput[] = {0, -1, 1};
  constexpr float kExpect[] = {0, 0, 0};
  constexpr int8_t kExpectQuantized[] = {127, 127, 127};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  // setup quantization storage and parameters
  int8_t q_output_data[kOutputCount];
  int8_t q_input_data[kOutputCount];
  constexpr float kMin = -1;
  constexpr float kMax = 1;
  constexpr float kLogSoftmaxQuantizedTolerance = 0.06355;
  tflite::testing::TestLogSoftmaxParams<int8_t> params = {};
  params.data_min = kMin;
  params.data_max = kMax;
  params.input_data = q_input_data;
  params.output_data = q_output_data;
  params.tolerance = kLogSoftmaxQuantizedTolerance;

  tflite::testing::TestLogSoftmaxQuantized(
      params, kDims, kInput, kDims, kExpect, kExpectQuantized, output_data);
}

TF_LITE_MICRO_TESTS_END
