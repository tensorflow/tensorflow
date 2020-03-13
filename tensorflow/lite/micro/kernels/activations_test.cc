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

#include <random>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestReluFloat(const int* input_dims_data, const float* input_data,
                   const int* output_dims_data, const float* golden,
                   float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims),
      CreateFloatTensor(output_data, output_dims),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU);
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

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
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
  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], 1e-5f);
  }
}

void TestRelu6Float(const int* input_dims_data, const float* input_data,
                    const int* output_dims_data, const float* golden,
                    float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims),
      CreateFloatTensor(output_data, output_dims),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU6);
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

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
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
  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], 1e-5f);
  }
}

void GenerateUniformRandomVector(int size, float min, float max,
                                 std::minstd_rand* random_engine,
                                 std::vector<float>* result) {
  // Never use std::uniform_*_distribution in tests, it's
  // implementation-defined. Likewise, don't use std::default_random_engine,
  // implementation-defined. Implementation-defined is bad because it means that
  // any toolchain update or new platform may run into test failures.
  // std::minstd_rand is a standard instantiation of
  // std::linear_congruential_engine, the cheapest generator in c++11 stdlib,
  // it's good enough here.
  result->resize(size);
  for (int i = 0; i < size; i++) {
    // We don't care whether the `max` value may ever be produced exactly.
    // It may actually be thanks to rounding, as std::minstd_rand::modulus
    // is 2^31 - 1 is greater than the inverse float epsilon.
    float random_value_scaled_0_1 =
        (*random_engine)() *
        (1.0f / static_cast<float>(std::minstd_rand::modulus));
    (*result)[i] = min + (max - min) * random_value_scaled_0_1;
  }
}

void EvalTestReferenceHardSwish(int size, const std::vector<float>& input,
                                std::vector<float>* result) {
  result->resize(size);
  for (int i = 0; i < size; i++) {
    const float in = input[i];
    (*result)[i] = in * std::min(6.0f, std::max(0.0f, in + 3)) * (1.0f / 6.0f);
  }
}

template <typename T>
void TestHardSwishQuantized(int size, float input_min,
                            float input_max, float output_min,
                            float output_max, std::minstd_rand* random_engine) {
  T output_data[size];
  T input_data_quantized[size];
  const int input_dims_data[] = {2, 1, size};
  const int output_dims_data[] = {2, 1, size};
  const float input_scale = ScaleFromMinMax<T>(input_min, input_max);
  const int input_zero_point = ZeroPointFromMinMax<T>(input_min, input_max);
  const float output_scale = ScaleFromMinMax<T>(output_min, output_max);
  const int output_zero_point = ZeroPointFromMinMax<T>(output_min, output_max);

  // The numerical error for any 8bit quantized function is at least one half
  // times the quantization step: 0.5 * (kOutMax - kOutMin) / 256.
  // To that we add again the quantization step (kOutMax - kOutMin) / 256
  // to allow for an off-by-one rounding error.
  const float kTolerance = std::max(input_max - input_min, output_max - output_min) * (1.5f / 256.f);

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  TF_LITE_MICRO_EXPECT_EQ(output_elements_count, size);

  float dequantized_output[output_elements_count];

  std::vector<float> float_input_values;
  std::vector<float> float_ref_output_values;
  GenerateUniformRandomVector(size, input_min, input_max, random_engine,
                              &float_input_values);
  EvalTestReferenceHardSwish(size, float_input_values,
                             &float_ref_output_values);
  for (float& val : float_ref_output_values) {
    val = std::min(output_max, std::max(output_min, val));
  }

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(float_input_values.data(), input_data_quantized, input_dims,
                            input_scale, input_zero_point, "input_tensor"),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point, "output_tensor"),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_HARD_SWISH);
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

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
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

  AsymmetricDequantize<T>(output_data, output_elements_count, output_scale, output_zero_point, dequantized_output);

  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(float_ref_output_values.data()[i], dequantized_output[i], kTolerance);
  }
}

template <typename T>
void TestHardSwishQuantizedBias(float input_min, float input_max, float output_min,
                                float output_max, float tolerated_bias) {
  const float quantized_type_range =
      static_cast<float>(std::numeric_limits<T>::max()) -
      static_cast<float>(std::numeric_limits<T>::min());


  const float input_scale = ScaleFromMinMax<T>(input_min, input_max);
  const float output_scale = ScaleFromMinMax<T>(output_min, output_max);

  const int input_zero_point = ZeroPointFromMinMax<T>(input_min, input_max);
  const int output_zero_point = ZeroPointFromMinMax<T>(output_min, output_max);

  const float max_scale = std::max(output_scale, input_scale);

  // In this bias-focused test case, no need for randomly generated input
  // values.
  TF_LITE_MICRO_EXPECT_LE(input_min, -3.0f);
  TF_LITE_MICRO_EXPECT_GE(input_max, 3.0f);
  const int quantized_input_negative_three =
      std::round(std::numeric_limits<T>::min() +
                 (-3.0f - input_min) / input_scale);
  const int quantized_input_positive_three =
      std::round(std::numeric_limits<T>::min() +
                 (3.0f - input_min) / input_scale);
  std::vector<float> float_input_values;
  for (int i = quantized_input_negative_three;
       i <= quantized_input_positive_three; i++) {
    float_input_values.push_back(
        input_min +
        (i - std::numeric_limits<T>::min()) * input_scale);
  }
  const int size = float_input_values.size();
  std::vector<float> float_ref_output_values;
  EvalTestReferenceHardSwish(size, float_input_values,
                             &float_ref_output_values);
  for (float& val : float_ref_output_values) {
    val = std::min(output_max, std::max(output_min, val));
  }

  T output_data[size];
  T input_data_quantized[size];
  const int input_dims_data[] = {2, 1, size};
  const int output_dims_data[] = {2, 1, size};

  // The numerical error for any 8bit quantized function is at least one half
  // times the quantization step: 0.5 * (kOutMax - kOutMin) / 256.
  // To that we add again the quantization step (kOutMax - kOutMin) / 256
  // to allow for an off-by-one rounding error.
  const float kTolerance = std::max(input_max - input_min, output_max - output_min) * (1.5f / 256.f);

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  TF_LITE_MICRO_EXPECT_EQ(output_elements_count, size);

  float dequantized_output[output_elements_count];

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(float_input_values.data(), input_data_quantized, input_dims,
                            input_scale, input_zero_point, "input_tensor"),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point, "output_tensor"),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_HARD_SWISH);
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

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
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

  AsymmetricDequantize<T>(output_data, output_elements_count, output_scale, output_zero_point, dequantized_output);

  float sum_diff = 0;
  for (int i = 0; i < size; i++) {
    sum_diff += dequantized_output[i] - float_ref_output_values[i];
  }
  const float bias = sum_diff / (size * max_scale);
  TF_LITE_MICRO_EXPECT_LE(std::abs(bias), tolerated_bias);
}

void TestHardSwishFloat(int size, std::minstd_rand* random_engine) {
  std::vector<float> float_input_values;
  const float kMin = -10.0f;
  const float kMax = 10.0f;
  GenerateUniformRandomVector(size, kMin, kMax, random_engine,
                              &float_input_values);
  std::vector<float> float_ref_output_values;
  EvalTestReferenceHardSwish(size, float_input_values,
                             &float_ref_output_values);

  float output_data[size];
  const int input_dims_data[] = {1, size};
  const int output_dims_data[] = {1, size};

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  TF_LITE_MICRO_EXPECT_EQ(output_elements_count, size);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(float_input_values.data(), input_dims, "input_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_HARD_SWISH);
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
  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
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

  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(float_ref_output_values.data()[i], output_data[i], 1e-5f);
  }
}

void TestReluUint8(const int* input_dims_data, const float* input_data,
                   uint8_t* input_data_quantized, const float input_scale,
                   const int input_zero_point, const float* golden,
                   uint8_t* golden_quantized, const int* output_dims_data,
                   const float output_scale, const int output_zero_point,
                   uint8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU);
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

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
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

  AsymmetricQuantize(golden, golden_quantized, output_elements_count,
                     output_scale, output_zero_point);

  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

void TestRelu6Uint8(const int* input_dims_data, const float* input_data,
                    uint8_t* input_data_quantized, const float input_scale,
                    const int input_zero_point, const float* golden,
                    uint8_t* golden_quantized, const int* output_dims_data,
                    const float output_scale, const int output_zero_point,
                    uint8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU6);
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

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
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

  AsymmetricQuantize(golden, golden_quantized, output_elements_count,
                     output_scale, output_zero_point);

  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

void TestReluInt8(const int* input_dims_data, const float* input_data,
                  int8_t* input_data_quantized, const float input_scale,
                  const int input_zero_point, const float* golden,
                  int8_t* golden_quantized, const int* output_dims_data,
                  const float output_scale, const int output_zero_point,
                  int8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU);
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

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
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

  AsymmetricQuantize(golden, golden_quantized, output_elements_count,
                     output_scale, output_zero_point);

  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

void TestRelu6Int8(const int* input_dims_data, const float* input_data,
                   int8_t* input_data_quantized, const float input_scale,
                   const int input_zero_point, const float* golden,
                   int8_t* golden_quantized, const int* output_dims_data,
                   const float output_scale, const int output_zero_point,
                   int8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU6);
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

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
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

  AsymmetricQuantize(golden, golden_quantized, output_elements_count,
                     output_scale, output_zero_point);

  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleReluTestFloat) {
  const int output_elements_count = 10;
  const int input_shape[] = {2, 1, 5};
  const float input_data[] = {
      1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0,
  };
  const float golden[] = {1.0, 2.0, 3.0, 4.0, 5.0, 0, 0, 0, 0, 0};
  const int output_shape[] = {2, 1, 5};
  float output_data[output_elements_count];
  tflite::testing::TestReluFloat(input_shape, input_data, output_shape, golden,
                                 output_data);
}

TF_LITE_MICRO_TEST(SimpleRelu6TestFloat) {
  const int output_elements_count = 10;
  float output_data[output_elements_count];
  const int input_shape[] = {2, 1, 5};
  const float input_data[] = {4.0,  5.0,  6.0,  7.0,  8.0,
                              -4.0, -5.0, -6.0, -7.0, -8.0};
  const int output_shape[] = {2, 1, 5};
  const float golden[] = {
      4.0, 5.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  };

  tflite::testing::TestRelu6Float(input_shape, input_data, output_shape, golden,
                                  output_data);
}

TF_LITE_MICRO_TEST(SimpleHardSwishTestFloat) {
  std::minstd_rand random_engine;
  for (int size : {1, 2, 3, 4, 10, 20, 30, 40, 100}) {
    tflite::testing::TestHardSwishFloat(size, &random_engine);
  }
}

TF_LITE_MICRO_TEST(SimpleHardSwishTestQuantized) {
  std::minstd_rand random_engine;
  std::vector<std::pair<float, float>> minmax_pairs{
      {0.f, 1.f}, {-2.f, 1.f}, {-5.f, 10.f}, {-40.f, 60.f}};
  for (const auto& input_minmax : minmax_pairs) {
    for (const auto& output_minmax : minmax_pairs) {
      float input_min = input_minmax.first;
      float input_max = input_minmax.second;
      float output_min = output_minmax.first;
      float output_max = output_minmax.second;
      for (int size : {1, 3, 10, 100}) {
        tflite::testing::TestHardSwishQuantized<int8_t>(size, input_min, input_max,
                                                        output_min, output_max,
                                                        &random_engine);
        tflite::testing::TestHardSwishQuantized<uint8_t>(size, input_min, input_max,
                                                         output_min, output_max,
                                                         &random_engine);
      }
    }
  }
}

// See the comment in the reference implementation of quantized HardSwish:
// A numerical issue significantly affecting ImageNet classification accuracy
// with MobileNet v3 is only observable at the scale of HardSwish unit tests
// if we monitor specifically bias. This testcase is extracted from one of the
// HardSwish nodes in that MobileNet v3 that exhibited this issue.
TF_LITE_MICRO_TEST(SimpleHardSwishTestQuantizedBias) {
  tflite::testing::TestHardSwishQuantizedBias<uint8_t>(-11.654928f, 25.036512f,
                                                       -0.3905796f, 24.50887f, 0.035);
}


TF_LITE_MICRO_TEST(SimpleReluTestUint8) {
  const int elements_count = 10;

  const int input_shape[] = {2, 1, 5};
  const float input_data[] = {1, 2, 3, 4, 5, -1, -2, -3, -4, -5};
  uint8_t input_quantized[elements_count];
  const int output_shape[] = {2, 1, 5};
  const float golden[] = {1, 2, 3, 4, 5, 0, 0, 0, 0, 0};
  uint8_t golden_quantized[elements_count];
  uint8_t output_data[elements_count];

  const float input_scale = 0.5f;
  const int input_zero_point = 127;
  const float output_scale = 0.5f;
  const int output_zero_point = 127;

  tflite::testing::TestReluUint8(input_shape, input_data, input_quantized,
                                 input_scale, input_zero_point, golden,
                                 golden_quantized, output_shape, output_scale,
                                 output_zero_point, output_data);
}

TF_LITE_MICRO_TEST(SimpleRelu6TestUint8) {
  const int elements_count = 10;

  const int input_shape[] = {2, 1, 5};
  const float input_data[] = {4, 5, 6, 7, 8, -1, -2, -3, -4, -5};
  uint8_t input_quantized[elements_count];
  const int output_shape[] = {2, 1, 5};
  const float golden[] = {4, 5, 6, 6, 6, 0, 0, 0, 0, 0};
  uint8_t golden_quantized[elements_count];
  uint8_t output_data[elements_count];

  const float input_scale = 0.5f;
  const int input_zero_point = 127;
  const float output_scale = 0.5f;
  const int output_zero_point = 127;

  tflite::testing::TestRelu6Uint8(input_shape, input_data, input_quantized,
                                  input_scale, input_zero_point, golden,
                                  golden_quantized, output_shape, output_scale,
                                  output_zero_point, output_data);
}

TF_LITE_MICRO_TEST(SimpleReluTestInt8) {
  const int elements_count = 10;

  const int input_shape[] = {2, 1, 5};
  const float input_data[] = {1, 2, 3, 4, 5, -1, -2, -3, -4, -5};
  int8_t input_quantized[elements_count];
  const int output_shape[] = {2, 1, 5};
  const float golden[] = {1, 2, 3, 4, 5, 0, 0, 0, 0, 0};
  int8_t golden_quantized[elements_count];
  int8_t output_data[elements_count];

  const float input_scale = 0.5f;
  const int input_zero_point = 0;
  const float output_scale = 0.5f;
  const int output_zero_point = 0;

  tflite::testing::TestReluInt8(input_shape, input_data, input_quantized,
                                input_scale, input_zero_point, golden,
                                golden_quantized, output_shape, output_scale,
                                output_zero_point, output_data);
}

TF_LITE_MICRO_TEST(SimpleRelu6TestInt8) {
  const int elements_count = 10;

  const int input_shape[] = {2, 1, 5};
  const float input_data[] = {4, 5, 6, 7, 8, -1, -2, -3, -4, -5};
  int8_t input_quantized[elements_count];
  const int output_shape[] = {2, 1, 5};
  const float golden[] = {4, 5, 6, 6, 6, 0, 0, 0, 0, 0};
  int8_t golden_quantized[elements_count];
  int8_t output_data[elements_count];

  const float input_scale = 0.5f;
  const int input_zero_point = 127;
  const float output_scale = 0.5f;
  const int output_zero_point = 127;

  tflite::testing::TestRelu6Int8(input_shape, input_data, input_quantized,
                                 input_scale, input_zero_point, golden,
                                 golden_quantized, output_shape, output_scale,
                                 output_zero_point, output_data);
}

TF_LITE_MICRO_TESTS_END
