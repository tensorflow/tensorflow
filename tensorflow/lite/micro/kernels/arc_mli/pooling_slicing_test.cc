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

// This test checks that slicing logic doesn`t affect result of pooling kernels
//
// This test doesn`t replace default pooling test
// (tensorflow/lite/micro/kernels/pooling.cc). It is added to the
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
void TestAveragePoolingQuantized(
    const int* input_dims_data, const T* input_data, const float input_min,
    const float input_max, const int filter_height, const int filter_width,
    const int stride_height, const int stride_width,
    const T* expected_output_data, const int* output_dims_data,
    float output_min, float output_max, TfLitePadding padding,
    TfLiteFusedActivation activation, T* output_data) {
  static_assert(sizeof(T) == 1, "Only int8/uint8 data types allowed.");

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, "input_tensor", input_min,
                            input_max),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_AVERAGE_POOL_2D, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLitePoolParams builtin_data = {padding,      stride_width,  stride_height,
                                   filter_width, filter_height, activation};
  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
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
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i], 1e-5f);
  }
}

template <typename T>
void TestMaxPoolQuantized(const int* input_dims_data, const T* input_data,
                          float input_min, float input_max, int filter_width,
                          int filter_height, int stride_width,
                          int stride_height, const T* expected_output_data,
                          float output_min, float output_max,
                          const int* output_dims_data, TfLitePadding padding,
                          TfLiteFusedActivation activation, T* output_data) {
  static_assert(sizeof(T) == 1, "Only int8/uint8 data types allowed.");

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, "input_tensor", input_min,
                            input_max),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_MAX_POOL_2D, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLitePoolParams builtin_data = {
      padding,      stride_width,  stride_height,
      filter_width, filter_height, activation,
  };

  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
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

TF_LITE_MICRO_TEST(SystemAveragePoolTestInt1) {
  using tflite::testing::F2QS;

  const float input_min = -128;
  const float input_max = 127;
  const float output_min = -128;
  const float output_max = 127;
  int8_t output_data[3];

  const int kInput1Shape[] = {4, 1, 2, 4, 1};
  const int8_t kInput1Data[] = {1, 1, 1, 1, 1, 1, 1, 1};
  const int kOutput1Shape[] = {4, 1, 1, 3, 1};
  const int8_t kGolden1Data[] = {1, 1, 1};

  tflite::testing::TestAveragePoolingQuantized(
      kInput1Shape,                       // Input shape
      kInput1Data, input_min, input_max,  // input quantization range
      2, 2,                               // filter height, filter width
      1, 1,                               // stride height, stride width
      kGolden1Data,
      kOutput1Shape,           // Output shape
      output_min, output_max,  // output quantization range
      kTfLitePaddingValid, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(LocalAveragePoolTestInt1) {
  using tflite::testing::F2QS;

  const float input_min = -128;
  const float input_max = 127;
  const float output_min = -128;
  const float output_max = 127;
  int8_t output_data[3];

#pragma Bss(".Zdata")
  const int kInput1Shape[] = {4, 1, 2, 4, 1};
  const int8_t kInput1Data[] = {1, 1, 1, 1, 1, 1, 1, 1};
  const int kOutput1Shape[] = {4, 1, 1, 3, 1};
  const int8_t kGolden1Data[] = {1, 1, 1};
#pragma Bss()

  tflite::testing::TestAveragePoolingQuantized(
      kInput1Shape,                       // Input shape
      kInput1Data, input_min, input_max,  // input quantization range
      2, 2,                               // filter height, filter width
      1, 1,                               // stride height, stride width
      kGolden1Data,
      kOutput1Shape,           // Output shape
      output_min, output_max,  // output quantization range
      kTfLitePaddingValid, kTfLiteActNone, output_data);
}

// Test group AVG 2
TF_LITE_MICRO_TEST(SystemAveragePoolTestInt2) {
  using tflite::testing::F2QS;

  const float input_min = -128;
  const float input_max = 127;
  const float output_min = -128;
  const float output_max = 127;
  int8_t output_data[45];

  const int kInput2Shape[] = {4, 1, 6, 10, 1};
  const int8_t kInput2Data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const int kOutput2Shape[] = {4, 1, 5, 9, 1};
  const int8_t kGolden2Data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  tflite::testing::TestAveragePoolingQuantized(
      kInput2Shape,                       // Input shape
      kInput2Data, input_min, input_max,  // input quantization range
      2, 2,                               // filter height, filter width
      1, 1,                               // stride height, stride width
      kGolden2Data,
      kOutput2Shape,           // Output shape
      output_min, output_max,  // output quantization range
      kTfLitePaddingValid, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(LocalAveragePoolTestInt2) {
  using tflite::testing::F2QS;

  const float input_min = -128;
  const float input_max = 127;
  const float output_min = -128;
  const float output_max = 127;
  int8_t output_data[45];

#pragma Bss(".Zdata")
  const int kInput2Shape[] = {4, 1, 6, 10, 1};
  const int8_t kInput2Data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const int kOutput2Shape[] = {4, 1, 5, 9, 1};
  const int8_t kGolden2Data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
#pragma Bss()

  tflite::testing::TestAveragePoolingQuantized(
      kInput2Shape,                       // Input shape
      kInput2Data, input_min, input_max,  // input quantization range
      2, 2,                               // filter height, filter width
      1, 1,                               // stride height, stride width
      kGolden2Data,
      kOutput2Shape,           // Output shape
      output_min, output_max,  // output quantization range
      kTfLitePaddingValid, kTfLiteActNone, output_data);
}

// Test group MAX 1
TF_LITE_MICRO_TEST(SystemMaxPoolTestInt1) {
  using tflite::testing::F2QS;

  int8_t output_data[3];
  const float input_min = -128;
  const float input_max = 127;
  const float output_min = -128;
  const float output_max = 127;
  int filter_width = 2;
  int filter_height = 2;
  int stride_width = 1;
  int stride_height = 1;

  const int kInput1Shape[] = {4, 1, 2, 4, 1};
  const int8_t kInput1Data[] = {1, 1, 1, 1, 1, 1, 1, 1};
  const int kOutput1Shape[] = {4, 1, 1, 3, 1};
  const int8_t kGolden1Data[] = {1, 1, 1};

  tflite::testing::TestMaxPoolQuantized(
      kInput1Shape,  // Input shape
      kInput1Data, input_min, input_max, filter_width, filter_height,
      stride_width, stride_height, kGolden1Data, output_min, output_max,
      kOutput1Shape,  // Output shape
      kTfLitePaddingValid, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(LocalMaxPoolTestInt1) {
  using tflite::testing::F2QS;

  int8_t output_data[3];
  const float input_min = -128;
  const float input_max = 127;
  const float output_min = -128;
  const float output_max = 127;
  int filter_width = 2;
  int filter_height = 2;
  int stride_width = 1;
  int stride_height = 1;

#pragma Bss(".Zdata")
  const int kInput1Shape[] = {4, 1, 2, 4, 1};
  const int8_t kInput1Data[] = {1, 1, 1, 1, 1, 1, 1, 1};
  const int kOutput1Shape[] = {4, 1, 1, 3, 1};
  const int8_t kGolden1Data[] = {1, 1, 1};
#pragma Bss()

  tflite::testing::TestMaxPoolQuantized(
      kInput1Shape,  // Input shape
      kInput1Data, input_min, input_max, filter_width, filter_height,
      stride_width, stride_height, kGolden1Data, output_min, output_max,
      kOutput1Shape,  // Output shape
      kTfLitePaddingValid, kTfLiteActNone, output_data);
}

// Test group MAX 2
TF_LITE_MICRO_TEST(SystemMaxPoolTestInt2) {
  using tflite::testing::F2QS;

  int8_t output_data[45];
  const float input_min = -128;
  const float input_max = 127;
  const float output_min = -128;
  const float output_max = 127;
  int filter_width = 2;
  int filter_height = 2;
  int stride_width = 1;
  int stride_height = 1;

  const int kInput2Shape[] = {4, 1, 6, 10, 1};
  const int8_t kInput2Data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const int kOutput2Shape[] = {4, 1, 5, 9, 1};
  const int8_t kGolden2Data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  tflite::testing::TestMaxPoolQuantized(
      kInput2Shape,  // Input shape
      kInput2Data, input_min, input_max, filter_width, filter_height,
      stride_width, stride_height, kGolden2Data, output_min, output_max,
      kOutput2Shape,  // Output shape
      kTfLitePaddingValid, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(LocalMaxPoolTestInt2) {
  using tflite::testing::F2QS;

  int8_t output_data[45];
  const float input_min = -128;
  const float input_max = 127;
  const float output_min = -128;
  const float output_max = 127;
  int filter_width = 2;
  int filter_height = 2;
  int stride_width = 1;
  int stride_height = 1;

#pragma Bss(".Zdata")
  const int kInput2Shape[] = {4, 1, 6, 10, 1};
  const int8_t kInput2Data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const int kOutput2Shape[] = {4, 1, 5, 9, 1};
  const int8_t kGolden2Data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
#pragma Bss()

  tflite::testing::TestMaxPoolQuantized(
      kInput2Shape,  // Input shape
      kInput2Data, input_min, input_max, filter_width, filter_height,
      stride_width, stride_height, kGolden2Data, output_min, output_max,
      kOutput2Shape,  // Output shape
      kTfLitePaddingValid, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TESTS_END
