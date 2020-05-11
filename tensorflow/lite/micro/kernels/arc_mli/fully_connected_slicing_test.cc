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

// This test checks that slicing logic doesn`t affect result of fully
// connected kernel
//
// This test doesn`t replace default fully connected test
// (tensorflow/lite/micro/kernels/fully_connected_test.cc). It is added to the
// whole testset only in case MLI for ARC platform is used during generation
// (which is handled in arc_mli.inc). So such tests won`t be generated for other
// platforms.

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
void TestFullyConnectedQuantized(
    const int* input_dims_data, const T* input_data, const float input_min,
    const float input_max, const int* weights_dims_data, const T* weights_data,
    const float weights_min, const float weights_max, const int* bias_dims_data,
    const int32_t* bias_data, const float bias_scale,
    const T* expected_output_data, const int* output_dims_data,
    const float output_min, const float output_max,
    TfLiteFusedActivation activation, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, "input_tensor", input_min,
                            input_max),
      CreateQuantizedTensor(weights_data, weights_dims, "weights_tensor",
                            weights_min, weights_max),
      CreateQuantized32Tensor(bias_data, bias_dims, "bias_tensor", bias_scale),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };

  tensors[0].params.zero_point = 0;
  tensors[1].params.zero_point = 0;
  tensors[3].params.zero_point = 0;

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_FULLY_CONNECTED, 4);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteFullyConnectedParams builtin_data = {
      activation,
      kTfLiteFullyConnectedWeightsFormatDefault,
  };
  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
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
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

// Test group 1
TF_LITE_MICRO_TEST(SystemSimpleTestQuantized1) {
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const float weights_min = -128.0f;
  const float weights_max = 127.0f;
  const float bias_scale = 1.0f;
  const float output_min = -128.0f;
  const float output_max = 127.0f;

  const int input_dims_data[] = {2, 2, 10};
  const int8_t input_data[] = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
  const int weights_dims_data[] = {2, 3, 10};
  const int8_t weights_data[] = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {1,1,1};
  const int8_t expected_output_data[] = {41,41,41,41,41,41};
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  int8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      weights_data, weights_min, weights_max, bias_dims_data, bias_data,
      bias_scale, expected_output_data, output_dims_data, output_min,
      output_max, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(LocalSimpleTestQuantized1) {
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const float weights_min = -128.0f;
  const float weights_max = 127.0f;
  const float bias_scale = 1.0f;
  const float output_min = -128.0f;
  const float output_max = 127.0f;

  const int input_dims_data_local[] = {2, 2, 10};
  const int weights_dims_data_local[] = {2, 3, 10};
  const int bias_dims_data_local[] = {1, 3};
  const int output_dims_data_local[] = {2, 2, 3};

  const int output_dims_count = 6;

#pragma Bss(".Zdata")  
  const int8_t input_data_local[] = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
  const int8_t weights_data_local[] = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
  const int32_t bias_data_local[] = {1,1,1};
  int8_t output_data_local[output_dims_count];
#pragma Bss()

  const int8_t expected_output_data[] = {41,41,41,41,41,41};

  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data_local, input_data_local, input_min, input_max, weights_dims_data_local,
      weights_data_local, weights_min, weights_max, bias_dims_data_local, bias_data_local,
      bias_scale, expected_output_data, output_dims_data_local, output_min,
      output_max, kTfLiteActNone, output_data_local);
}

// Test group 2
TF_LITE_MICRO_TEST(SystemSimpleTestQuantized2) {
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const float weights_min = -128.0f;
  const float weights_max = 127.0f;
  const float bias_scale = 1.0f;
  const float output_min = -128.0f;
  const float output_max = 127.0f;

  const int input_dims_data_2[] = {2, 10, 4};
  const int8_t input_data_2[] = {2,2,2,2,2,2,2,2,2,2,
                               2,2,2,2,2,2,2,2,2,2,
                               2,2,2,2,2,2,2,2,2,2,
                               2,2,2,2,2,2,2,2,2,2};
  const int weights_dims_data_2[] = {2, 6, 4};
  const int8_t weights_data_2[] = {2,2,2,2,2,2,2,2,2,2,
                                 2,2,2,2,2,2,2,2,2,2,
                                 2,2,2,2};
  const int bias_dims_data_2[] = {1, 6};
  const int32_t bias_data_2[] = {1,1,1,1,1,1};
  const int8_t expected_output_data_2[] = {17,17,17,17,17,17,17,17,17,17,
                                         17,17,17,17,17,17,17,17,17,17,
                                         17,17,17,17,17,17,17,17,17,17,
                                         17,17,17,17,17,17,17,17,17,17,
                                         17,17,17,17,17,17,17,17,17,17,
                                         17,17,17,17,17,17,17,17,17,17};
  const int output_dims_data_2[] = {2, 10, 6};

  const int output_dims_count_2 = 60;
  int8_t output_data_2[output_dims_count_2];
  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data_2, input_data_2, input_min, input_max, weights_dims_data_2,
      weights_data_2, weights_min, weights_max, bias_dims_data_2, bias_data_2,
      bias_scale, expected_output_data_2, output_dims_data_2, output_min,
      output_max, kTfLiteActNone, output_data_2);
}

TF_LITE_MICRO_TEST(LocalSimpleTestQuantized2) {
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const float weights_min = -128.0f;
  const float weights_max = 127.0f;
  const float bias_scale = 1.0f;
  const float output_min = -128.0f;
  const float output_max = 127.0f;

  const int input_dims_data_local_2[] = {2, 10, 4};
  const int weights_dims_data_local_2[] = {2, 6, 4};
  const int bias_dims_data_local_2[] = {1, 6};
  const int output_dims_data_local_2[] = {2, 10, 6};

  const int output_dims_count_local_2 = 60;

#pragma Bss(".Zdata")  
  const int8_t input_data_local_2[] = {2,2,2,2,2,2,2,2,2,2,
                               2,2,2,2,2,2,2,2,2,2,
                               2,2,2,2,2,2,2,2,2,2,
                               2,2,2,2,2,2,2,2,2,2};
  const int8_t weights_data_local_2[] = {2,2,2,2,2,2,2,2,2,2,
                                 2,2,2,2,2,2,2,2,2,2,
                                 2,2,2,2};
  const int32_t bias_data_local_2[] = {1,1,1,1,1,1};
  int8_t output_data_local_2[output_dims_count_local_2];
#pragma Bss()

  const int8_t expected_output_data_local_2[] = {41,41,41,41,41,41};

  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data_local_2, input_data_local_2, input_min, input_max, weights_dims_data_local_2,
      weights_data_local_2, weights_min, weights_max, bias_dims_data_local_2, bias_data_local_2,
      bias_scale, expected_output_data_local_2, output_dims_data_local_2, output_min,
      output_max, kTfLiteActNone, output_data_local_2);
}

// Test group 3
TF_LITE_MICRO_TEST(SystemSimpleTestQuantized3) {
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const float weights_min = -128.0f;
  const float weights_max = 127.0f;
  const float bias_scale = 1.0f;
  const float output_min = -128.0f;
  const float output_max = 127.0f;

  const int input_dims_data_3[] = {2, 2, 5};
  const int8_t input_data_3[] = {2,2,2,2,2,2,2,2,2,2};
  const int weights_dims_data_3[] = {2, 10, 5};
  const int8_t weights_data_3[] = {2,2,2,2,2,2,2,2,2,2,
                                   2,2,2,2,2,2,2,2,2,2,
                                   2,2,2,2,2,2,2,2,2,2,
                                   2,2,2,2,2,2,2,2,2,2,
                                   2,2,2,2,2,2,2,2,2,2};
  const int bias_dims_data_3[] = {1, 10};
  const int32_t bias_data_3[] = {1,1,1,1,1,1,1,1,1,1};
  const int8_t expected_output_data_3[] = {21,21,21,21,21,21,21,21,21,21,
                                           21,21,21,21,21,21,21,21,21,21};
  const int output_dims_data_3[] = {2, 2, 10};

  const int output_dims_count_3 = 20;
  int8_t output_data_3[output_dims_count_3];
  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data_3, input_data_3, input_min, input_max, weights_dims_data_3,
      weights_data_3, weights_min, weights_max, bias_dims_data_3, bias_data_3,
      bias_scale, expected_output_data_3, output_dims_data_3, output_min,
      output_max, kTfLiteActNone, output_data_3);
}

TF_LITE_MICRO_TEST(LocalSimpleTestQuantized3) {
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const float weights_min = -128.0f;
  const float weights_max = 127.0f;
  const float bias_scale = 1.0f;
  const float output_min = -128.0f;
  const float output_max = 127.0f;

  const int input_dims_data_local_3[] = {2, 2, 5};
  const int weights_dims_data_local_3[] = {2, 10, 5};
  const int bias_dims_data_local_3[] = {1, 10};
  const int output_dims_data_local_3[] = {2, 2, 10};

  const int output_dims_count_local_3 = 20;

#pragma Bss(".Zdata")  
  static int8_t input_data_local_3[10];
  static int8_t weights_data_local_3[50];
  static int32_t bias_data_local_3[10];
  static int8_t output_data_local_3[output_dims_count_local_3];
#pragma Bss()

  for(int i = 0; i < 10; ++i) {
    input_data_local_3[i] = 2;  
  }

  for(int i = 0; i < 50; ++i) {
    weights_data_local_3[i] = 2;  
  }

  for(int i = 0; i < 10; ++i) {
    bias_data_local_3[i] = 1;  
  }

  for(int i = 0; i < 20; ++i) {
    output_data_local_3[i] = 0;  
  }

  const int8_t expected_output_data_local_3[] = {21,21,21,21,21,21,21,21,21,21,
                                                 21,21,21,21,21,21,21,21,21,21};

  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data_local_3, input_data_local_3, input_min, input_max, weights_dims_data_local_3,
      weights_data_local_3, weights_min, weights_max, bias_dims_data_local_3, bias_data_local_3,
      bias_scale, expected_output_data_local_3, output_dims_data_local_3, output_min,
      output_max, kTfLiteActNone, output_data_local_3);
}

// Test group 4
TF_LITE_MICRO_TEST(SystemSimpleTestQuantized4) {
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const float weights_min = -128.0f;
  const float weights_max = 127.0f;
  const float bias_scale = 1.0f;
  const float output_min = -128.0f;
  const float output_max = 127.0f;

  const int input_dims_data_4[] = {2, 5, 10};
  const int8_t input_data_4[] = {2,2,2,2,2,2,2,2,2,2,
                                 2,2,2,2,2,2,2,2,2,2,
                                 2,2,2,2,2,2,2,2,2,2,
                                 2,2,2,2,2,2,2,2,2,2,
                                 2,2,2,2,2,2,2,2,2,2};
  const int weights_dims_data_4[] = {2, 5, 10};
  const int8_t weights_data_4[] = {2,2,2,2,2,2,2,2,2,2,
                                   2,2,2,2,2,2,2,2,2,2,
                                   2,2,2,2,2,2,2,2,2,2,
                                   2,2,2,2,2,2,2,2,2,2,
                                   2,2,2,2,2,2,2,2,2,2};
  const int bias_dims_data_4[] = {1, 5};
  const int32_t bias_data_4[] = {1,1,1,1,1};
  const int8_t expected_output_data_4[] = {41,41,41,41,41,41,41,41,41,41,
                                           41,41,41,41,41,41,41,41,41,41,
                                           41,41,41,41,41};
  const int output_dims_data_4[] = {2, 5, 5};

  const int output_dims_count_4 = 25;
  int8_t output_data_4[output_dims_count_4];
  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data_4, input_data_4, input_min, input_max, weights_dims_data_4,
      weights_data_4, weights_min, weights_max, bias_dims_data_4, bias_data_4,
      bias_scale, expected_output_data_4, output_dims_data_4, output_min,
      output_max, kTfLiteActNone, output_data_4);
}

TF_LITE_MICRO_TEST(LocalSimpleTestQuantized4) {
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const float weights_min = -128.0f;
  const float weights_max = 127.0f;
  const float bias_scale = 1.0f;
  const float output_min = -128.0f;
  const float output_max = 127.0f;

  const int input_dims_data_local_4[] = {2, 5, 10};
  const int weights_dims_data_local_4[] = {2, 5, 10};
  const int bias_dims_data_local_4[] = {1, 5};
  const int output_dims_data_local_4[] = {2, 5, 5};

  const int output_dims_count_local_4 = 25;

#pragma Bss(".Zdata")  
  const int8_t input_data_local_4[] = {2,2,2,2,2,2,2,2,2,2,
                                       2,2,2,2,2,2,2,2,2,2,
                                       2,2,2,2,2,2,2,2,2,2,
                                       2,2,2,2,2,2,2,2,2,2,
                                       2,2,2,2,2,2,2,2,2,2};
  const int8_t weights_data_local_4[] = {2,2,2,2,2,2,2,2,2,2,
                                         2,2,2,2,2,2,2,2,2,2,
                                         2,2,2,2,2,2,2,2,2,2,
                                         2,2,2,2,2,2,2,2,2,2,
                                         2,2,2,2,2,2,2,2,2,2};
  const int32_t bias_data_local_4[] = {1,1,1,1,1};
  int8_t output_data_local_4[output_dims_count_local_4];
#pragma Bss()

  const int8_t expected_output_data_local_4[] = {41,41,41,41,41,41,41,41,41,41,
                                                 41,41,41,41,41,41,41,41,41,41,
                                                 41,41,41,41,41};

  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data_local_4, input_data_local_4, input_min, input_max, weights_dims_data_local_4,
      weights_data_local_4, weights_min, weights_max, bias_dims_data_local_4, bias_data_local_4,
      bias_scale, expected_output_data_local_4, output_dims_data_local_4, output_min,
      output_max, kTfLiteActNone, output_data_local_4);
}

TF_LITE_MICRO_TESTS_END
