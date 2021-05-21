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
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// The Softmax kernel assumes an output in the range [0, 1.0], leading to these
// quantization parameters.
const float output_scale_int8 = 1.0f / 256.0f;
const float output_scale_int16 = 1.0f / 32768.0f;
const int output_zero_point_int8 = -128;
const int output_zero_point_int16 = 0;

// Empirical tolerance in quantization space
const float tolerance_int16 = 7.0;

// 1-dimensional test data.
const int flat_size_1d = 5;
int shape_1d[] = {1, 5};
const float input_data_1d[] = {1.0, 2.0, 3.0, 4.0, 5.0};
const float golden_1d[] = {0.011656231, 0.031684921, 0.086128544, 0.234121657,
                           0.636408647};

// 2-dimensional test data.
const int flat_size_2d = 10;
int shape_2d[] = {2, 2, 5};
const float input_data_2d[] = {1.0,  2.0,  3.0,  4.0,  5.0,
                               -1.0, -2.0, -3.0, -4.0, -5.0};
const float golden_2d[] = {0.011656231, 0.031684921, 0.086128544, 0.234121657,
                           0.636408647, 0.636408647, 0.234121657, 0.086128544,
                           0.031684921, 0.011656231};

// 3-dimensional test data.
const int flat_size_3d = 60;
int shape_3d[] = {3, 3, 4, 5};
const float input_data_3d[] = {
    // c = 0
    // h = 0
    3.00, 6.00, -5.00, 4.00, -9.00,
    // h = 1
    -10.00, -10.00, -8.00, 2.00, 2.00,
    // h = 2
    8.00, -5.00, -8.00, 5.00, -6.00,
    // h = 3
    -8.00, 6.00, 1.00, -10.00, -8.00,

    // c = 1
    // h = 0
    7.00, 6.00, -10.00, -4.00, -5.00,
    // h = 1
    2.00, 7.00, 9.00, -9.00, 7.00,
    // h = 2
    -4.00, -2.00, 8.00, 2.00, 2.00,
    // h = 3
    3.00, 6.00, 6.00, 2.00, 4.00,

    // c = 2
    // h = 0
    9.00, 7.00, -7.00, 0.00, 4.00,
    // h = 1
    -3.00, 8.00, 8.00, -3.00, -4.00,
    // h = 2
    -9.00, -9.00, 4.00, -8.00, -1.00,
    // h = 3
    -10.00, -2.00, 6.00, -7.00, 0.00};

float golden_3d[] = {
    // c = 0
    // h = 0
    0.042009463, 0.843782625, 0.000014093, 0.114193561, 0.000000258,
    // h = 1
    0.000003072, 0.000003072, 0.000022699, 0.499985578, 0.499985578,
    // h = 2
    0.952571219, 0.000002153, 0.000000107, 0.047425728, 0.000000792,
    // h = 3
    0.000000826, 0.993305397, 0.006692839, 0.000000112, 0.000000826,

    // c = 1
    // h = 0
    0.731046347, 0.268936922, 0.000000030, 0.000012210, 0.000004492,
    // h = 1
    0.000717124, 0.106430599, 0.786421666, 0.000000012, 0.106430599,
    // h = 2
    0.000006114, 0.000045174, 0.995015917, 0.002466398, 0.002466398,
    // h = 3
    0.022595176, 0.453836234, 0.453836234, 0.008312301, 0.061420055,

    // c = 2
    // h = 0
    0.875505904, 0.118486839, 0.000000099, 0.000108046, 0.005899112,
    // h = 1
    0.000008351, 0.499990113, 0.499990113, 0.000008351, 0.000003072,
    // h = 2
    0.000002245, 0.000002245, 0.993296627, 0.000006103, 0.006692780,
    // h = 3
    0.000000112, 0.000334520, 0.997191323, 0.000002254, 0.002471790};

// 4-dimensional test data.
const int flat_size_4d = 120;
int shape_4d[] = {4, 2, 3, 4, 5};
const float input_data_4d[] = {
    // n = 0
    // c = 0
    // h = 0
    3.00, 6.00, -5.00, 4.00, -9.00,
    // h = 1
    -10.00, -10.00, -8.00, 2.00, 2.00,
    // h = 2
    8.00, -5.00, -8.00, 5.00, -6.00,
    // h = 3
    -8.00, 6.00, 1.00, -10.00, -8.00,

    // c = 1
    // h = 0
    7.00, 6.00, -10.00, -4.00, -5.00,
    // h = 1
    2.00, 7.00, 9.00, -9.00, 7.00,
    // h = 2
    -4.00, -2.00, 8.00, 2.00, 2.00,
    // h = 3
    3.00, 6.00, 6.00, 2.00, 4.00,

    // c = 2
    // h = 0
    9.00, 7.00, -7.00, 0.00, 4.00,
    // h = 1
    -3.00, 8.00, 8.00, -3.00, -4.00,
    // h = 2
    -9.00, -9.00, 4.00, -8.00, -1.00,
    // h = 3
    -10.00, -2.00, 6.00, -7.00, 0.00,

    // n = 1
    // c = 0
    // h = 0
    -9.00, -8.00, 6.00, -1.00, -5.00,
    // h = 1
    -10.00, -5.00, -10.00, 7.00, -2.00,
    // h = 2
    -5.00, -4.00, 1.00, 2.00, 2.00,
    // h = 3
    -2.00, -2.00, 1.00, 1.00, -4.00,

    // c = 1
    // h = 0
    -8.00, -3.00, 1.00, 1.00, -1.00,
    // h = 1
    -2.00, 6.00, -1.00, -5.00, 6.00,
    // h = 2
    -7.00, 8.00, 9.00, 0.00, 9.00,
    // h = 3
    -9.00, -5.00, -2.00, 0.00, 8.00,

    // c = 2
    // h = 0
    4.00, 2.00, -3.00, 5.00, 8.00,
    // h = 1
    -1.00, 1.00, -4.00, -9.00, 7.00,
    // h = 2
    3.00, -8.00, 0.00, 9.00, -4.00,
    // h = 3
    8.00, -1.00, 9.00, -9.00, 1.00};

const float golden_4d[] = {
    // n = 0
    // c = 0
    // h = 0
    0.042009463, 0.843782625, 0.000014093, 0.114193561, 0.000000258,
    // h = 1
    0.000003072, 0.000003072, 0.000022699, 0.499985578, 0.499985578,
    // h = 2
    0.952571219, 0.000002153, 0.000000107, 0.047425728, 0.000000792,
    // h = 3
    0.000000826, 0.993305397, 0.006692839, 0.000000112, 0.000000826,

    // c = 1
    // h = 0
    0.731046347, 0.268936922, 0.000000030, 0.000012210, 0.000004492,
    // h = 1
    0.000717124, 0.106430599, 0.786421666, 0.000000012, 0.106430599,
    // h = 2
    0.000006114, 0.000045174, 0.995015917, 0.002466398, 0.002466398,
    // h = 3
    0.022595176, 0.453836234, 0.453836234, 0.008312301, 0.061420055,

    // c = 2
    // h = 0
    0.875505904, 0.118486839, 0.000000099, 0.000108046, 0.005899112,
    // h = 1
    0.000008351, 0.499990113, 0.499990113, 0.000008351, 0.000003072,
    // h = 2
    0.000002245, 0.000002245, 0.993296627, 0.000006103, 0.006692780,
    // h = 3
    0.000000112, 0.000334520, 0.997191323, 0.000002254, 0.002471790,

    // n = 1
    // c = 0
    // h = 0
    0.000000306, 0.000000831, 0.999071142, 0.000911035, 0.000016686,
    // h = 1
    0.000000041, 0.000006143, 0.000000041, 0.999870380, 0.000123394,
    // h = 2
    0.000384554, 0.001045327, 0.155140254, 0.421714933, 0.421714933,
    // h = 3
    0.023637081, 0.023637081, 0.474763454, 0.474763454, 0.003198931,

    // c = 1
    // h = 0
    0.000057299, 0.008503973, 0.464301197, 0.464301197, 0.062836334,
    // h = 1
    0.000167625, 0.499684188, 0.000455653, 0.000008346, 0.499684188,
    // h = 2
    0.000000048, 0.155354299, 0.422296769, 0.000052116, 0.422296769,
    // h = 3
    0.000000041, 0.000002259, 0.000045383, 0.000335334, 0.999616982,

    // c = 2
    // h = 0
    0.017107856, 0.002315297, 0.000015600, 0.046503973, 0.934057274,
    // h = 1
    0.000334516, 0.002471755, 0.000016655, 0.000000112, 0.997176963,
    // h = 2
    0.002472313, 0.000000041, 0.000123089, 0.997402302, 0.000002254,
    // h = 3
    0.268866557, 0.000033181, 0.730855076, 0.000000011, 0.000245175};

template <typename T>
void ValidateSoftmaxGoldens(TfLiteTensor* tensors, const int tensor_count,
                            T* output_data, const T* expected_output,
                            int output_dims_count, float tolerance) {
  TfLiteSoftmaxParams builtin_data = {1.0f};

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = Register_SOFTMAX();
  micro::KernelRunner runner(registration, tensors, tensor_count, inputs_array,
                             outputs_array, &builtin_data);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output[i], output_data[i], tolerance);
  }
}

void TestSoftmaxFloat(int* input_dims_data, const float* input_data,
                      int* output_dims_data, const float* expected_output_data,
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

  ValidateSoftmaxGoldens(tensors, tensors_size, output_data,
                         expected_output_data, output_dims_count, 1e-5);
}

template <typename inputT, typename outputT>
void TestSoftmaxQuantized(int* input_dims_data, const float* input_data,
                          inputT* input_quantized, float input_scale,
                          int input_zero_point, int* output_dims_data,
                          const float* golden, outputT* golden_quantized,
                          float output_scale, int output_zero_point,
                          outputT* output_data, float tolerance = 1.0) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  Quantize(golden, golden_quantized, output_dims_count, output_scale,
           output_zero_point);

  ValidateSoftmaxGoldens(tensors, tensors_size, output_data, golden_quantized,
                         output_dims_count, tolerance);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(Softmax1DFloatShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_1d];
  tflite::testing::TestSoftmaxFloat(
      tflite::testing ::shape_1d, tflite::testing::input_data_1d,
      tflite::testing::shape_1d, tflite::testing::golden_1d, output_data);
}

TF_LITE_MICRO_TEST(Softmax1DQuantizedInt8ShouldMatchGolden) {
  const float input_scale = 0.1f;
  const int input_zero_point = 0;

  int8_t input_quantized[tflite::testing::flat_size_1d];
  int8_t golden_quantized[tflite::testing::flat_size_1d];
  int8_t output_data[tflite::testing::flat_size_1d];
  tflite::testing::TestSoftmaxQuantized(
      tflite::testing::shape_1d, tflite::testing::input_data_1d,
      input_quantized, input_scale, input_zero_point, tflite::testing::shape_1d,
      tflite::testing::golden_1d, golden_quantized,
      tflite::testing::output_scale_int8,
      tflite::testing::output_zero_point_int8, output_data);
}

TF_LITE_MICRO_TEST(Softmax1DQuantizedInt16ShouldMatchGolden) {
  const float input_scale = 0.1f;
  const int input_zero_point = 0;

  int16_t input_quantized[tflite::testing::flat_size_1d];
  int16_t golden_quantized[tflite::testing::flat_size_1d];
  int16_t output_data[tflite::testing::flat_size_1d];
  tflite::testing::TestSoftmaxQuantized(
      tflite::testing::shape_1d, tflite::testing::input_data_1d,
      input_quantized, input_scale, input_zero_point, tflite::testing::shape_1d,
      tflite::testing::golden_1d, golden_quantized,
      tflite::testing::output_scale_int16,
      tflite::testing::output_zero_point_int16, output_data);
}

TF_LITE_MICRO_TEST(Softmax2DFloatShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_2d];
  tflite::testing::TestSoftmaxFloat(
      tflite::testing ::shape_2d, tflite::testing::input_data_2d,
      tflite::testing::shape_2d, tflite::testing::golden_2d, output_data);
}

TF_LITE_MICRO_TEST(Softmax2DQuantizedInt8ShouldMatchGolden) {
  const float input_scale = 0.1f;
  const int input_zero_point = 0;

  int8_t input_quantized[tflite::testing::flat_size_2d];
  int8_t golden_quantized[tflite::testing::flat_size_2d];
  int8_t output_data[tflite::testing::flat_size_2d];
  tflite::testing::TestSoftmaxQuantized(
      tflite::testing::shape_2d, tflite::testing::input_data_2d,
      input_quantized, input_scale, input_zero_point, tflite::testing::shape_2d,
      tflite::testing::golden_2d, golden_quantized,
      tflite::testing::output_scale_int8,
      tflite::testing::output_zero_point_int8, output_data);
}

TF_LITE_MICRO_TEST(Softmax2DQuantizedInt16ShouldMatchGolden) {
  const float input_scale = 0.1f;
  const int input_zero_point = 0;

  int16_t input_quantized[tflite::testing::flat_size_2d];
  int16_t golden_quantized[tflite::testing::flat_size_2d];
  int16_t output_data[tflite::testing::flat_size_2d];
  tflite::testing::TestSoftmaxQuantized(
      tflite::testing::shape_2d, tflite::testing::input_data_2d,
      input_quantized, input_scale, input_zero_point, tflite::testing::shape_2d,
      tflite::testing::golden_2d, golden_quantized,
      tflite::testing::output_scale_int16,
      tflite::testing::output_zero_point_int16, output_data);
}

TF_LITE_MICRO_TEST(Softmax3DFloatShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_3d];
  tflite::testing::TestSoftmaxFloat(
      tflite::testing ::shape_3d, tflite::testing::input_data_3d,
      tflite::testing::shape_3d, tflite::testing::golden_3d, output_data);
}

TF_LITE_MICRO_TEST(Softmax3DQuantizedInt8ShouldMatchGolden) {
  const float input_scale = 0.1f;
  const int input_zero_point = 0;

  int8_t input_quantized[tflite::testing::flat_size_3d];
  int8_t golden_quantized[tflite::testing::flat_size_3d];
  int8_t output_data[tflite::testing::flat_size_3d];
  tflite::testing::TestSoftmaxQuantized(
      tflite::testing::shape_3d, tflite::testing::input_data_3d,
      input_quantized, input_scale, input_zero_point, tflite::testing::shape_3d,
      tflite::testing::golden_3d, golden_quantized,
      tflite::testing::output_scale_int8,
      tflite::testing::output_zero_point_int8, output_data);
}

TF_LITE_MICRO_TEST(Softmax3DQuantizedInt16ShouldMatchGolden) {
  const float input_scale = 0.1f;
  const int input_zero_point = 0;

  int16_t input_quantized[tflite::testing::flat_size_3d];
  int16_t golden_quantized[tflite::testing::flat_size_3d];
  int16_t output_data[tflite::testing::flat_size_3d];
  tflite::testing::TestSoftmaxQuantized(
      tflite::testing::shape_3d, tflite::testing::input_data_3d,
      input_quantized, input_scale, input_zero_point, tflite::testing::shape_3d,
      tflite::testing::golden_3d, golden_quantized,
      tflite::testing::output_scale_int16,
      tflite::testing::output_zero_point_int16, output_data,
      tflite::testing::tolerance_int16);
}

TF_LITE_MICRO_TEST(Softmax4DFloatShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_4d];
  tflite::testing::TestSoftmaxFloat(
      tflite::testing ::shape_4d, tflite::testing::input_data_4d,
      tflite::testing::shape_4d, tflite::testing::golden_4d, output_data);
}

TF_LITE_MICRO_TEST(Softmax4DQuantizedInt8ShouldMatchGolden) {
  const float input_scale = 0.1f;
  const int input_zero_point = 0;

  int8_t input_quantized[tflite::testing::flat_size_4d];
  int8_t golden_quantized[tflite::testing::flat_size_4d];
  int8_t output_data[tflite::testing::flat_size_4d];
  tflite::testing::TestSoftmaxQuantized(
      tflite::testing::shape_4d, tflite::testing::input_data_4d,
      input_quantized, input_scale, input_zero_point, tflite::testing::shape_4d,
      tflite::testing::golden_4d, golden_quantized,
      tflite::testing::output_scale_int8,
      tflite::testing::output_zero_point_int8, output_data);
}

TF_LITE_MICRO_TEST(Softmax4DQuantizedInt16ShouldMatchGolden) {
  const float input_scale = 0.1f;
  const int input_zero_point = 0;

  int16_t input_quantized[tflite::testing::flat_size_4d];
  int16_t golden_quantized[tflite::testing::flat_size_4d];
  int16_t output_data[tflite::testing::flat_size_4d];
  tflite::testing::TestSoftmaxQuantized(
      tflite::testing::shape_4d, tflite::testing::input_data_4d,
      input_quantized, input_scale, input_zero_point, tflite::testing::shape_4d,
      tflite::testing::golden_4d, golden_quantized,
      tflite::testing::output_scale_int16,
      tflite::testing::output_zero_point_int16, output_data,
      tflite::testing::tolerance_int16);
}

TF_LITE_MICRO_TEST(Softmax2DQuantizedInt8InputInt16OutputShouldMatchGolden) {
  const float input_scale = 0.1f;
  const int input_zero_point = 0;
  const float output_scale = 1.0f / 65536.0f;
  const int output_zero_point = -32768;

  int8_t input_quantized[tflite::testing::flat_size_2d];
  int16_t golden_quantized[tflite::testing::flat_size_2d];
  int16_t output_data[tflite::testing::flat_size_2d];
  tflite::testing::TestSoftmaxQuantized(
      tflite::testing::shape_2d, tflite::testing::input_data_2d,
      input_quantized, input_scale, input_zero_point, tflite::testing::shape_2d,
      tflite::testing::golden_2d, golden_quantized, output_scale,
      output_zero_point, output_data);
}

TF_LITE_MICRO_TESTS_END
