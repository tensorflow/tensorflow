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

const float tanh_input_vec_fp[] = {
    -8.0000000000, -7.8181818182, -7.6363636364, -7.4545454545, -7.2727272727,
    -7.0909090909, -6.9090909091, -6.7272727273, -6.5454545455, -6.3636363636,
    -6.1818181818, -6.0000000000, -5.8181818182, -5.6363636364, -5.4545454545,
    -5.2727272727, -5.0909090909, -4.9090909091, -4.7272727273, -4.5454545455,
    -4.3636363636, -4.1818181818, -4.0000000000, -3.8181818182, -3.6363636364,
    -3.4545454545, -3.2727272727, -3.0909090909, -2.9090909091, -2.7272727273,
    -2.5454545455, -2.3636363636, -2.1818181818, -2.0000000000, -1.8181818182,
    -1.6363636364, -1.4545454545, -1.2727272727, -1.0909090909, -0.9090909091,
    -0.7272727273, -0.5454545455, -0.3636363636, -0.1818181818, 0.0000000000,
    0.1818181818,  0.3636363636,  0.5454545455,  0.7272727273,  0.9090909091,
    1.0909090909,  1.2727272727,  1.4545454545,  1.6363636364,  1.8181818182,
    2.0000000000,  2.1818181818,  2.3636363636,  2.5454545455,  2.7272727273,
    2.9090909091,  3.0909090909,  3.2727272727,  3.4545454545,  3.6363636364,
    3.8181818182,  4.0000000000,  4.1818181818,  4.3636363636,  4.5454545455,
    4.7272727273,  4.9090909091,  5.0909090909,  5.2727272727,  5.4545454545,
    5.6363636364,  5.8181818182,  6.0000000000,  6.1818181818,  6.3636363636,
    6.5454545455,  6.7272727273,  6.9090909091,  7.0909090909,  7.2727272727,
    7.4545454545,  7.6363636364,  7.8181818182,  8.0000000000};

const float tanh_output_vec_fp[] = {
    -0.9999997749, -0.9999996762, -0.9999995342, -0.9999993300, -0.9999990361,
    -0.9999986134, -0.9999980053, -0.9999971306, -0.9999958722, -0.9999940619,
    -0.9999914578, -0.9999877117, -0.9999823226, -0.9999745703, -0.9999634183,
    -0.9999473758, -0.9999242982, -0.9998911009, -0.9998433469, -0.9997746542,
    -0.9996758446, -0.9995337191, -0.9993292997, -0.9990353053, -0.9986125310,
    -0.9980046622, -0.9971308601, -0.9958751909, -0.9940716137, -0.9914827859,
    -0.9877703933, -0.9824541388, -0.9748561217, -0.9640275801, -0.9486568273,
    -0.9269625051, -0.8965880154, -0.8545351057, -0.7972097087, -0.7206956332,
    -0.6213939966, -0.4971057414, -0.3484130125, -0.1798408185, 0.0000000000,
    0.1798408185,  0.3484130125,  0.4971057414,  0.6213939966,  0.7206956332,
    0.7972097087,  0.8545351057,  0.8965880154,  0.9269625051,  0.9486568273,
    0.9640275801,  0.9748561217,  0.9824541388,  0.9877703933,  0.9914827859,
    0.9940716137,  0.9958751909,  0.9971308601,  0.9980046622,  0.9986125310,
    0.9990353053,  0.9993292997,  0.9995337191,  0.9996758446,  0.9997746542,
    0.9998433469,  0.9998911009,  0.9999242982,  0.9999473758,  0.9999634183,
    0.9999745703,  0.9999823226,  0.9999877117,  0.9999914578,  0.9999940619,
    0.9999958722,  0.9999971306,  0.9999980053,  0.9999986134,  0.9999990361,
    0.9999993300,  0.9999995342,  0.9999996762,  0.9999997749};

constexpr int tanh_vec_size = 90;

void TestTanhFloat(std::initializer_list<int> input_dims_data,
                   const float* input_data, const float* expected_output_data,
                   std::initializer_list<int> output_dims_data,
                   float* output_data, const float tolerance) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_TANH, 1);
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
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }
}

template <typename T>
void TestTanhQuantized(std::initializer_list<int> input_dims_data,
                       const float* input_data, T* input_quantized,
                       float input_scale, int input_zero_point,
                       const float* expected_output_data,
                       T* expected_output_quantized,
                       std::initializer_list<int> output_dims_data,
                       float output_scale, int output_zero_point,
                       T* output_quantized, const int tolerance) {
  static_assert(sizeof(T) == 1, "Valid only for 8bit data types");
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  tflite::AsymmetricQuantize(expected_output_data, expected_output_quantized,
                             output_elements_count, output_scale,
                             output_zero_point);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point, "input_tensor"),
      CreateQuantizedTensor(output_quantized, output_dims, output_scale,
                            output_zero_point, "output_tensor")};

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_TANH, 1);
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
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_quantized[i], output_quantized[i],
                              tolerance);
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

TF_LITE_MICRO_TEST(SimpleTestTanhFloat) {
  using tflite::testing::tanh_input_vec_fp;
  using tflite::testing::tanh_output_vec_fp;
  using tflite::testing::tanh_vec_size;

  float output_data[tanh_vec_size];
  tflite::testing::TestTanhFloat(  //
      {2, 1, tanh_vec_size},       // Input shape.
      tanh_input_vec_fp,           // Input data
      tanh_output_vec_fp,          // Expected results.
      {2, 1, tanh_vec_size},       // Output shape.
      output_data, 1e-7 /* tolerance */);
}

TF_LITE_MICRO_TEST(SimpleTestTanhUInt8) {
  using tflite::testing::tanh_input_vec_fp;
  using tflite::testing::tanh_output_vec_fp;
  using tflite::testing::tanh_vec_size;

  const float input_scale = 16 / 256.f;
  const int input_zero_point = 128;
  const float output_scale = 1.99999955 / 256.f;
  const int output_zero_point = 128;

  uint8_t input_quantized[tanh_vec_size];
  uint8_t expected_output_quantized[tanh_vec_size];
  uint8_t output_quantized[tanh_vec_size];
  tflite::testing::TestTanhQuantized<uint8_t>(        //
      {2, 1, tanh_vec_size},                          // Input shape.
      tanh_input_vec_fp, input_quantized,             // Input data.
      input_scale, input_zero_point,                  // Input quantized info.
      tanh_output_vec_fp, expected_output_quantized,  // Expected results.
      {2, 1, tanh_vec_size},                          // Output shape.
      output_scale, output_zero_point,                // Output quantized info.
      output_quantized,                               // Operation results
      2                                               // Tolerance.
  );
}

TF_LITE_MICRO_TEST(SimpleTestTanhUInt8) {
  using tflite::testing::tanh_input_vec_fp;
  using tflite::testing::tanh_output_vec_fp;
  using tflite::testing::tanh_vec_size;

  const float input_scale = 16 / 256.f;
  const int input_zero_point = 0;
  const float output_scale = 1.99999955 / 256.f;
  const int output_zero_point = 0;

  int8_t input_quantized[tanh_vec_size];
  int8_t expected_output_quantized[tanh_vec_size];
  int8_t output_quantized[tanh_vec_size];
  tflite::testing::TestTanhQuantized<int8_t>(         //
      {2, 1, tanh_vec_size},                          // Input shape.
      tanh_input_vec_fp, input_quantized,             // Input data.
      input_scale, input_zero_point,                  // Input quantized info.
      tanh_output_vec_fp, expected_output_quantized,  // Expected results.
      {2, 1, tanh_vec_size},                          // Output shape.
      output_scale, output_zero_point,                // Output quantized info.
      output_quantized,                               // Operation results
      2                                               // Tolerance.
  );
}

TF_LITE_MICRO_TESTS_END
