/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
void ValidateDequantizeGoldens(TfLiteTensor* tensors, int tensors_size,
                               const T* expected_output_data, T* output_data,
                               int output_length, float tolerance = 1e-5) {
  TfLiteContext context;
  ::tflite::ops::micro::AllOpsResolver resolver;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  // Version 2 of dequantize supports int8 quantization.
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_DEQUANTIZE, 2);

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = nullptr;
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));

  if (registration->free) {
    registration->free(&context, user_data);
  }

  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i], 0.001);
  }
}

template <typename T>
void TestDequantizeToFloat(const int* input_dims_data, const float* input_data,
                           T* input_data_quantized, float scale, int zero_point,
                           const int* output_dims_data,
                           const float* expected_output_data,
                           float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_length = ElementCount(*output_dims);

  // 1 input, 1 output.
  const int tensors_size = 2;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims, scale,
                            zero_point, "input_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };

  ValidateDequantizeGoldens(tensors, tensors_size, expected_output_data,
                            output_data, output_length);
}

template <typename T>
void TestDequantizeToInt32(const int* input_dims_data, const float* input_data,
                           T* input_data_quantized, float input_scale,
                           int input_zero_point, const int* output_dims_data,
                           const int32_t* expected_output_data,
                           float output_scale, int output_zero_point,
                           int32_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_length = ElementCount(*output_dims);

  // 1 input, 1 output.
  const int tensors_size = 2;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point, "input_tensor"),
      CreateInt32Tensor(output_data, output_dims, "output_tensor"),
  };

  TfLiteQuantizationParams output_quant;
  tensors[1].params.scale = output_scale;
  tensors[1].params.zero_point = output_zero_point;

  ValidateDequantizeGoldens(tensors, tensors_size, expected_output_data,
                            output_data, output_length);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(DequantizeOpTestUint8) {
  const int length = 10;
  const int dims[] = {2, 5, 2};
  const float values[] = {-63.5, -63,  -62.5, -62,  -61.5,
                          62,    62.5, 63,    63.5, 64};
  const float scale = 0.5;
  const int zero_point = 127;
  uint8_t input_quantized[length];
  float output[length];
  tflite::testing::TestDequantizeToFloat(dims, values, input_quantized, scale,
                                         zero_point, dims, values, output);
}

TF_LITE_MICRO_TEST(DequantizeOpTestInt8) {
  const int length = 10;
  const int dims[] = {2, 5, 2};
  const float values[] = {-63.5, -63,  -62.5, -62,  -61.5,
                          62,    62.5, 63,    63.5, 64};
  const float scale = 0.5;
  const int zero_point = -1;
  int8_t input_quantized[length];
  float output[length];
  tflite::testing::TestDequantizeToFloat(dims, values, input_quantized, scale,
                                         zero_point, dims, values, output);
}

TF_LITE_MICRO_TEST(DequantizeOpTestInt16) {
  const int length = 10;
  const int dims[] = {2, 5, 2};
  const float values[] = {-63.5, -63,  -62.5, -62,  -61.5,
                          62,    62.5, 63,    63.5, 64};
  const float scale = 0.5;
  const int zero_point = -1;
  int16_t input_quantized[length];
  float output[length];
  tflite::testing::TestDequantizeToFloat(dims, values, input_quantized, scale,
                                         zero_point, dims, values, output);
}

TF_LITE_MICRO_TEST(DequantizeOpTestInt8ToInt32) {
  const int length = 10;
  const int dims[] = {2, 5, 2};
  const float input_float[] = {-63.5, -63,  -62.5, -62,  -61.5,
                               62,    62.5, 63,    63.5, 64};
  const int32_t golden[] = {-630, -625, -620, -615, -610,
                            625,  630,  635,  640,  645};
  const float input_scale = 0.5f;
  const int input_zero_point = -1;
  const float output_scale = 0.1f;
  const int output_zero_point = 5;
  int8_t input_quantized[length];
  int32_t output[length];
  tflite::testing::TestDequantizeToInt32(
      dims, input_float, input_quantized, input_scale, input_zero_point, dims,
      golden, output_scale, output_zero_point, output);
}

TF_LITE_MICRO_TESTS_END
