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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

const int flat_size_simple = 4;
const float scale_simple = 0.01;
const int dims_simple[] = {4, 1, 2, 2, 1};
const float input1_simple[] = {-0.8, 0.2, 0.9, 0.7};
const float input2_simple[] = {0.6, 0.4, 0.9, 0.8};
const float golden_simple[] = {-0.48, 0.08, 0.81, 0.56};
const float golden_simple_relu[] = {0.0, 0.08, 0.81, 0.56};

const int flat_size_broadcast = 6;
const float input_scale_broadcast = 0.05f;
const float output_scale_broadcast = 0.01f;
const int dims_broadcast[] = {4, 1, 3, 1, 2};
const int dims_scalar_broadcast[] = {1, 1};
const float input1_broadcast[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
const float input2_broadcast[] = {0.1};
const float golden_broadcast[] = {-0.2, 0.02, 0.07, 0.08, 0.11, 0.2};
const float golden_broadcast_relu[] = {0, 0.02, 0.07, 0.08, 0.11, 0.2};

template <typename T>
void ValidateMulGoldens(TfLiteTensor* tensors, int tensors_size,
                        TfLiteFusedActivation activation, const T* golden,
                        int output_len, float tolerance, T* output) {
  TfLiteMulParams builtin_data = {
      .activation = activation,
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = tflite::ops::micro::Register_MUL();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_len; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output[i], tolerance);
  }
}

void TestMulFloat(const int* input1_dims_data, const float* input1_data,
                  const int* input2_dims_data, const float* input2_data,
                  const int* output_dims_data, const float* golden,
                  float* output_data, TfLiteFusedActivation activation) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  ValidateMulGoldens(tensors, tensors_size, activation, golden,
                     output_dims_count, 1e-5, output_data);
}

template <typename T>
void TestMulQuantized(const int* input1_dims_data, const float* input1_data,
                      T* input1_quantized, const int* input2_dims_data,
                      const float* input2_data, T* input2_quantized,
                      const float input_scale, const int input_zero_point,
                      const int* output_dims_data, const float* golden,
                      T* golden_quantized, const float output_scale,
                      const int output_zero_point, T* output_data,
                      TfLiteFusedActivation activation) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_quantized, input1_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(input2_data, input2_quantized, input2_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point)};

  Quantize(golden, golden_quantized, output_dims_count, output_scale,
           output_zero_point);

  ValidateMulGoldens(tensors, tensors_size, activation, golden_quantized,
                     output_dims_count, 1.0f, output_data);
}

}  // namespace

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleFloatNoAcativationShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_simple];

  tflite::testing::TestMulFloat(
      tflite::testing::dims_simple, tflite::testing::input1_simple,
      tflite::testing::dims_simple, tflite::testing::input2_simple,
      tflite::testing::dims_simple, tflite::testing::golden_simple, output_data,
      kTfLiteActNone);
}

TF_LITE_MICRO_TEST(SimpleFloatReluShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_simple];

  tflite::testing::TestMulFloat(
      tflite::testing::dims_simple, tflite::testing::input1_simple,
      tflite::testing::dims_simple, tflite::testing::input2_simple,
      tflite::testing::dims_simple, tflite::testing::golden_simple_relu,
      output_data, kTfLiteActRelu);
}

TF_LITE_MICRO_TEST(SimpleInt8NoAcativationShouldMatchGolden) {
  int8_t input1_quantized[tflite::testing::flat_size_simple];
  int8_t input2_quantized[tflite::testing::flat_size_simple];
  int8_t golden_quantized[tflite::testing::flat_size_simple];
  int8_t output_data[tflite::testing::flat_size_simple];

  tflite::testing::TestMulQuantized(
      tflite::testing::dims_simple, tflite::testing::input1_simple,
      input1_quantized, tflite::testing::dims_simple,
      tflite::testing::input2_simple, input2_quantized,
      tflite::testing::scale_simple, 0, tflite::testing::dims_simple,
      tflite::testing::golden_simple, golden_quantized,
      tflite::testing::scale_simple, 0, output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(SimpleUInt8NoAcativationShouldMatchGolden) {
  uint8_t input1_quantized[tflite::testing::flat_size_simple];
  uint8_t input2_quantized[tflite::testing::flat_size_simple];
  uint8_t golden_quantized[tflite::testing::flat_size_simple];
  uint8_t output_data[tflite::testing::flat_size_simple];

  tflite::testing::TestMulQuantized(
      tflite::testing::dims_simple, tflite::testing::input1_simple,
      input1_quantized, tflite::testing::dims_simple,
      tflite::testing::input2_simple, input2_quantized,
      tflite::testing::scale_simple, 128, tflite::testing::dims_simple,
      tflite::testing::golden_simple, golden_quantized,
      tflite::testing::scale_simple, 128, output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(BroadcastFloatNoActivationShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_broadcast];

  tflite::testing::TestMulFloat(
      tflite::testing::dims_broadcast, tflite::testing::input1_broadcast,
      tflite::testing::dims_scalar_broadcast, tflite::testing::input2_broadcast,
      tflite::testing::dims_broadcast, tflite::testing::golden_broadcast,
      output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(BroadcastFloatReluShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_broadcast];

  tflite::testing::TestMulFloat(
      tflite::testing::dims_broadcast, tflite::testing::input1_broadcast,
      tflite::testing::dims_scalar_broadcast, tflite::testing::input2_broadcast,
      tflite::testing::dims_broadcast, tflite::testing::golden_broadcast_relu,
      output_data, kTfLiteActRelu);
}

TF_LITE_MICRO_TEST(BroadcastInt8NoAcativationShouldMatchGolden) {
  int8_t input1_quantized[tflite::testing::flat_size_broadcast];
  int8_t input2_quantized[tflite::testing::flat_size_broadcast];
  int8_t golden_quantized[tflite::testing::flat_size_broadcast];
  int8_t output_data[tflite::testing::flat_size_broadcast];

  tflite::testing::TestMulQuantized(
      tflite::testing::dims_broadcast, tflite::testing::input1_broadcast,
      input1_quantized, tflite::testing::dims_scalar_broadcast,
      tflite::testing::input2_broadcast, input2_quantized,
      tflite::testing::input_scale_broadcast, 0,
      tflite::testing::dims_broadcast, tflite::testing::golden_broadcast,
      golden_quantized, tflite::testing::output_scale_broadcast, 0, output_data,
      kTfLiteActNone);
}

TF_LITE_MICRO_TEST(BroadcastUInt8NoAcativationShouldMatchGolden) {
  uint8_t input1_quantized[tflite::testing::flat_size_broadcast];
  uint8_t input2_quantized[1];
  uint8_t golden_quantized[tflite::testing::flat_size_broadcast];
  uint8_t output_data[tflite::testing::flat_size_broadcast];

  tflite::testing::TestMulQuantized(
      tflite::testing::dims_broadcast, tflite::testing::input1_broadcast,
      input1_quantized, tflite::testing::dims_scalar_broadcast,
      tflite::testing::input2_broadcast, input2_quantized,
      tflite::testing::input_scale_broadcast, 128,
      tflite::testing::dims_broadcast, tflite::testing::golden_broadcast,
      golden_quantized, tflite::testing::output_scale_broadcast, 128,
      output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TESTS_END
