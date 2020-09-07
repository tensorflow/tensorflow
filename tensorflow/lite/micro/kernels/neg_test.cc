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
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestNegFloat(const int* input_dims_data, const float* input_data,
                  const float* expected_output_data,
                  const int* output_dims_data, float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims),
      CreateFloatTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = tflite::ops::micro::Register_NEG();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr, micro_test::reporter);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  TF_LITE_MICRO_EXPECT_EQ(expected_output_data[0], output_data[0]);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

// input_quantization_buffer: buffer used for the quantization data
void TestNegQuantizedInt8(float* input, int8_t* quantized_input_data,
                          float input_min, float input_max,
                          float* expected_output,
                          int8_t* quantized_expected_output_data,
                          int8_t* quantized_output_data, float output_min,
                          float output_max, int* dimension_data) {
  TfLiteIntArray* tensor_dims = IntArrayFromInts(dimension_data);
  const int element_count = ElementCount(*tensor_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;

  // quantize input
  std::transform(input, input + element_count, quantized_input_data,
                 [=](float value) -> float {
                   return tflite::testing::F2QS(value, input_min, input_max);
                 });

  // quantize expected_output since there'll be loss of precision and
  // comparision in floating point will be inaccurate
  std::transform(expected_output, expected_output + element_count,
                 quantized_expected_output_data, [=](float value) -> float {
                   return tflite::testing::F2QS(value, output_min, output_max);
                 });

  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(quantized_input_data, tensor_dims, input_min,
                            input_max),
      CreateQuantizedTensor(quantized_output_data, tensor_dims, output_min,
                            output_max)};

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = nullptr;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));

  TF_LITE_MICRO_EXPECT_EQ(quantized_expected_output_data[0],
                          quantized_output_data[0]);
  for (int i = 0; i < element_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(quantized_expected_output_data[i],
                            quantized_output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(NegOpSingleFloat) {
  const int dims[] = {1, 2};
  const float input_data[] = {8.5, 0.0};
  const float golden[] = {-8.5, 0.0};
  float output_data[2];

  tflite::testing::TestNegFloat(dims, input_data, golden, dims, output_data);
}

TF_LITE_MICRO_TEST(NegOpFloat) {
  const int dims[] = {2, 2, 3};
  const float input_data[] = {-2.0f, -1.0f, 0.f, 1.0f, 2.0f, 3.0f};
  const float golden[] = {2.0f, 1.0f, -0.f, -1.0f, -2.0f, -3.0f};
  float output_data[6];

  tflite::testing::TestNegFloat(dims, input_data, golden, dims, output_data);
}

TF_LITE_MICRO_TEST(NegOpQuantizedInt8DifferentZeroPoints) {
  // setup input and output data

  // same scale, different zero points
  constexpr auto input_min = -12.f;
  constexpr auto input_max = 24.f;
  constexpr auto output_min = -20.f;
  constexpr auto output_max = 16.f;

  float input[] = {-2.f, -1.f, 0.f, 1.f, 2.f, 3.f};

  constexpr auto kTensorSize = sizeof(input) / sizeof(float);

  float expected_output[kTensorSize];
  // direct negation
  std::transform(input, input + kTensorSize, expected_output,
                 [](float value) { return -value; });

  int dimension_data[] = {
      // number of elements in shape
      2,
      // shape data
      2, 3};  // multiply reduce of shape data should yield kTensorSize

  // used by helper
  int8_t quantized_input[kTensorSize] = {};
  int8_t quantized_expected_output[kTensorSize] = {};
  int8_t quantized_output[kTensorSize] = {};

  tflite::testing::TestNegQuantizedInt8(
      input, quantized_input, input_min, input_max, expected_output,
      quantized_expected_output, quantized_output, output_min, output_max,
      dimension_data);
}

TF_LITE_MICRO_TEST(NegOpQuantizedInt8SameZeroPoints) {
  // setup input and output data

  // same scale, different zero points
  constexpr auto input_min = -12.f;
  constexpr auto input_max = 24.f;
  constexpr auto output_min = -12.f;
  constexpr auto output_max = 24.f;

  float input[] = {-2.f, -1.f, 0.f, 1.f, 2.f, 3.f};

  constexpr auto kTensorSize = sizeof(input) / sizeof(float);

  float expected_output[kTensorSize];
  // direct negation
  std::transform(input, input + kTensorSize, expected_output,
                 [](float value) { return -value; });

  int dimension_data[] = {
      // number of elements in shape
      2,
      // shape data
      2, 3};  // multiply reduce of shape data should yield kTensorSize

  // used by helper
  int8_t quantized_input[kTensorSize] = {};
  int8_t quantized_expected_output[kTensorSize] = {};
  int8_t quantized_output[kTensorSize] = {};

  tflite::testing::TestNegQuantizedInt8(
      input, quantized_input, input_min, input_max, expected_output,
      quantized_expected_output, quantized_output, output_min, output_max,
      dimension_data);
}

TF_LITE_MICRO_TESTS_END
