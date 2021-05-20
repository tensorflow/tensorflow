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

template <typename T>
void ValidateQuantizeGoldens(TfLiteTensor* tensors, int tensors_size,
                             const float* golden, T* golden_quantized,
                             float scale, int zero_point, int output_len,
                             T* output_data) {
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  // Version 1 of quantize supports int8_t and uint8_t quantization.
  const TfLiteRegistration registration = Register_QUANTIZE();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  // Use reference quantization from test utils to compare against op output.
  Quantize(golden, golden_quantized, output_len, scale, zero_point);
  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

#if !defined(XTENSA)
template <typename T>
void TestQuantizeFloat(int* input_dims_data, const float* input_data,
                       int* output_dims_data, const float* golden,
                       T* golden_quantized, const float scale,
                       const int zero_point, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  TfLiteTensor output_tensor =
      CreateQuantizedTensor(output_data, output_dims, scale, zero_point);

  TfLiteAffineQuantization quant;
  float scales[] = {1, scale};
  int zero_points[] = {1, zero_point};
  quant.scale = FloatArrayFromFloats(scales);
  quant.zero_point = IntArrayFromInts(zero_points);
  output_tensor.quantization = {kTfLiteAffineQuantization, &quant};

  // 1 input, 1 output.
  constexpr int tensors_size = 2;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      output_tensor,
  };

  ValidateQuantizeGoldens(tensors, tensors_size, golden, golden_quantized,
                          scale, zero_point, output_dims_count, output_data);
}
#endif  // defined(XTENSA)

template <typename InputType, typename OutputType>
void TestRequantize(int* input_dims_data, const float* input_data,
                    InputType* input_quantized, const float input_scale,
                    const int input_zero_point, int* output_dims_data,
                    const float* golden, OutputType* golden_quantized,
                    const float output_scale, const int output_zero_point,
                    OutputType* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  TfLiteTensor output_tensor = CreateQuantizedTensor(
      output_data, output_dims, output_scale, output_zero_point);

  TfLiteAffineQuantization quant;
  float scales[] = {1, output_scale};
  int zero_points[] = {1, output_zero_point};
  quant.scale = FloatArrayFromFloats(scales);
  quant.zero_point = IntArrayFromInts(zero_points);
  output_tensor.quantization = {kTfLiteAffineQuantization, &quant};

  // 1 input, 1 output.
  constexpr int tensors_size = 2;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      output_tensor,
  };

  ValidateQuantizeGoldens(tensors, tensors_size, golden, golden_quantized,
                          output_scale, output_zero_point, output_dims_count,
                          output_data);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

#if !defined(XTENSA)
TF_LITE_MICRO_TEST(QuantizeOpTestInt16) {
  const int length = 10;
  int dims[] = {2, 2, 5};
  const float values[] = {-63.5, -63,  -62.5, -62,  -61.5,
                          62,    62.5, 63,    63.5, 64};
  const float scale = 0.5;
  const int zero_point = -1;
  int16_t output[length];
  int16_t values_quantized[length];
  tflite::testing::TestQuantizeFloat(
      dims, values, dims, values, values_quantized, scale, zero_point, output);
}

TF_LITE_MICRO_TEST(QuantizeOpTestInt16NoScale) {
  const int length = 10;
  int dims[] = {2, 2, 5};
  const float values[] = {-128, -127, -126, -125, -124,
                          123,  124,  125,  126,  127};
  const float scale = 1.0;
  const int zero_point = 0;
  int16_t output[length];
  int16_t values_quantized[length];
  tflite::testing::TestQuantizeFloat(
      dims, values, dims, values, values_quantized, scale, zero_point, output);
}

TF_LITE_MICRO_TEST(QuantizeOpTestInt16toInt16) {
  const int length = 10;
  int dims[] = {2, 2, 5};
  const float values[] = {-64, -62, -60, -58, -56, 54, 56, 58, 60, 62};
  const float input_scale = 2.f;
  const int input_zero_point = 0;
  const float output_scale = 0.5;
  const int output_zero_point = 32;
  int16_t output_quantized[length];
  int16_t values_quantized[length];
  int16_t input_quantized[length];
  tflite::testing::TestRequantize(dims, values, input_quantized, input_scale,
                                  input_zero_point, dims, values,
                                  values_quantized, output_scale,
                                  output_zero_point, output_quantized);
}

TF_LITE_MICRO_TEST(QuantizeOpTestInt16toInt16NoZeroPoint) {
  const int length = 10;
  int dims[] = {2, 2, 5};
  const float values[] = {-32, -31, -30, -29, -28, 27, 28, 29, 30, 31};
  const float input_scale = 1.f;
  const int input_zero_point = 0;
  const float output_scale = 0.5;
  const int output_zero_point = 0;
  int16_t output_quantized[length];
  int16_t values_quantized[length];
  int16_t input_quantized[length];
  tflite::testing::TestRequantize(dims, values, input_quantized, input_scale,
                                  input_zero_point, dims, values,
                                  values_quantized, output_scale,
                                  output_zero_point, output_quantized);
}

TF_LITE_MICRO_TEST(QuantizeOpTestInt8toInt8) {
  const int length = 10;
  int dims[] = {2, 2, 5};
  const float values[] = {-64, -62, -60, -58, -56, 54, 56, 58, 60, 62};
  const float input_scale = 2.f;
  const int input_zero_point = 0;
  const float output_scale = 0.5;
  const int output_zero_point = 32;
  int8_t output_quantized[length];
  int8_t values_quantized[length];
  int8_t input_quantized[length];
  tflite::testing::TestRequantize(dims, values, input_quantized, input_scale,
                                  input_zero_point, dims, values,
                                  values_quantized, output_scale,
                                  output_zero_point, output_quantized);
}

TF_LITE_MICRO_TEST(QuantizeOpTestInt8toInt8NoZeroPoint) {
  const int length = 10;
  int dims[] = {2, 2, 5};
  const float values[] = {-32, -31, -30, -29, -28, 27, 28, 29, 30, 31};
  const float input_scale = 1.f;
  const int input_zero_point = 0;
  const float output_scale = 0.5;
  const int output_zero_point = 0;
  int8_t output_quantized[length];
  int8_t values_quantized[length];
  int8_t input_quantized[length];
  tflite::testing::TestRequantize(dims, values, input_quantized, input_scale,
                                  input_zero_point, dims, values,
                                  values_quantized, output_scale,
                                  output_zero_point, output_quantized);
}
#endif  // defined(XTENSA)

#if !defined(XTENSA)
// TODO(b/155682734): Hifimini optimized quantize requires input scale to be
// smaller then output scale.
TF_LITE_MICRO_TEST(QuantizeOpTestInt16toInt8) {
  const int length = 10;
  int dims[] = {2, 2, 5};
  const float values[] = {-64, -62, -60, -58, -56, 54, 56, 58, 60, 62};
  const float input_scale = 2.f;
  const int input_zero_point = 0;
  const float output_scale = 0.5;
  const int output_zero_point = 0;
  int8_t output_quantized[length];
  int8_t values_quantized[length];
  int16_t input_quantized[length];
  tflite::testing::TestRequantize(dims, values, input_quantized, input_scale,
                                  input_zero_point, dims, values,
                                  values_quantized, output_scale,
                                  output_zero_point, output_quantized);
}
#endif  // defined(XTENSA)

TF_LITE_MICRO_TEST(QuantizeOpTestInt8toInt32) {
  const int length = 10;
  int dims[] = {2, 2, 5};
  const float values[] = {-32, -31, -30, -29, -28, 27, 28, 29, 30, 31};
  const float input_scale = 1.f;
  const int input_zero_point = 0;
  const float output_scale = 0.5;
  const int output_zero_point = 0;
  int32_t output_quantized[length];
  int32_t values_quantized[length];
  int8_t input_quantized[length];
  tflite::testing::TestRequantize(dims, values, input_quantized, input_scale,
                                  input_zero_point, dims, values,
                                  values_quantized, output_scale,
                                  output_zero_point, output_quantized);
}

TF_LITE_MICRO_TEST(QuantizeOpTestInt16toInt32) {
  const int length = 10;
  int dims[] = {2, 2, 5};
  const float values[] = {-32, -31, -30, -29, -28, 27, 28, 29, 30, 31};
  const float input_scale = 1.f;
  const int input_zero_point = 0;
  const float output_scale = 0.5;
  const int output_zero_point = 0;
  int32_t output_quantized[length];
  int32_t values_quantized[length];
  int16_t input_quantized[length];
  tflite::testing::TestRequantize(dims, values, input_quantized, input_scale,
                                  input_zero_point, dims, values,
                                  values_quantized, output_scale,
                                  output_zero_point, output_quantized);
}

TF_LITE_MICRO_TEST(QuantizeOpTestInt16toInt8) {
  constexpr int length = 10;
  int dims[] = {2, 2, 5};
  const float values[] = {-32, -31, -30, -29, -28, 27, 28, 29, 30, 31};
  // TODO(b/155682734): Input scale must be smaller than output scale for
  // xtensa.
  const float input_scale = 0.4f;
  const int input_zero_point = 0;
  const float output_scale = 1.0f;
  const int output_zero_point = 0;
  int8_t output_quantized[length];
  int8_t values_quantized[length];
  int16_t input_quantized[length];
  tflite::testing::TestRequantize(dims, values, input_quantized, input_scale,
                                  input_zero_point, dims, values,
                                  values_quantized, output_scale,
                                  output_zero_point, output_quantized);
}

TF_LITE_MICRO_TESTS_END
