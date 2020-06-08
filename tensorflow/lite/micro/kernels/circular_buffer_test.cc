/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kRunPeriod = 2;

// TODO(b/149795762): Add this to TfLiteStatus enum.
constexpr int kTfLiteAbort = -9;

TfLiteNode PrepareCircularBufferInt8(const int* input_dims_data,
                                     const int8_t* input_data,
                                     const int* output_dims_data,
                                     const int8_t* expected_output_data,
                                     int8_t* output_data) {
  const TfLiteRegistration* registration =
      ops::micro::Register_CIRCULAR_BUFFER();

  TfLiteNode node;
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, 1, 0, "input_tensor"),
      CreateQuantizedTensor(output_data, output_dims, 1, 0, "output_tensor"),
  };
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  // There is one input - tensor 0.
  const int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  // There is one output - tensor 1.
  const int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  // There are no intermediates.
  const int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->prepare);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  return node;
}

// Run invoke cycles_until_output times with the supplied input, expecting
// invoke to return kTfLiteAbort until the last iteration, at which point the
// output should match expected_output_data.
TfLiteStatus InvokeCircularBufferInt8(const int* input_dims_data,
                                      const int8_t* input_data,
                                      const int* output_dims_data,
                                      const int8_t* expected_output_data,
                                      int8_t* output_data, TfLiteNode* node) {
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);

  const int output_dims_count = ElementCount(*output_dims);
  const TfLiteRegistration* registration =
      ops::micro::Register_CIRCULAR_BUFFER();

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, 1, 0, "input_tensor"),
      CreateQuantizedTensor(output_data, output_dims, 1, 0, "output_tensor"),
  };
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  // There is one input - tensor 0.
  const int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  // There is one output - tensor 1.
  const int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  // There are no intermediates.
  const int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  node->inputs = inputs_array;
  node->outputs = outputs_array;
  node->temporaries = temporaries_array;
  node->builtin_data = nullptr;
  node->custom_initial_data = nullptr;
  node->custom_initial_data_size = 0;
  node->delegate = nullptr;

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);

  TfLiteStatus status = registration->invoke(&context, node);

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
  return status;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(OutputTensorLength4) {
  constexpr int depth = 3;
  constexpr int num_slots = 4;
  int8_t output_data[depth * num_slots];

  memset(output_data, 0, sizeof(output_data));
  // There are four input dimensions - [1, 1, 1, depth].
  const int input_dims[] = {4, 1, 1, 1, depth};
  // There are four output dimensions - [1, num_slots, 1, depth].
  const int output_dims[] = {4, 1, num_slots, 1, depth};

  const int8_t goldens[5][16] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3},
                                 {0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6},
                                 {0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                 {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};

  int8_t input[depth];
  TfLiteNode node = tflite::testing::PrepareCircularBufferInt8(
      input_dims, input, output_dims, goldens[0], output_data);
  // Expect the circular buffer to run every other invoke for 4xN output.
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < depth; j++) {
      input[j] = i * depth + j + 1;
    }
    TfLiteStatus status = tflite::testing::InvokeCircularBufferInt8(
        input_dims, input, output_dims, goldens[i], output_data, &node);

    // Every kRunPeriod iterations, the circular buffer should return kTfLiteOk.
    if (i % tflite::testing::kRunPeriod == tflite::testing::kRunPeriod - 1) {
      TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, status);
    } else {
      TF_LITE_MICRO_EXPECT_EQ(tflite::testing::kTfLiteAbort, status);
    }
  }
}

TF_LITE_MICRO_TEST(OutputTensorLength5) {
  constexpr int depth = 4;
  constexpr int num_slots = 5;
  int8_t output_data[depth * num_slots];

  memset(output_data, 0, sizeof(output_data));
  const int input_dims[] = {4, 1, 1, 1, depth};
  const int output_dims[] = {4, 1, num_slots, 1, depth};

  const int8_t goldens[6][20] = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8},
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
      {0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
      {5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
       15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};

  int8_t input[depth];
  TfLiteNode node = tflite::testing::PrepareCircularBufferInt8(
      input_dims, input, output_dims, goldens[0], output_data);
  // Expect circular buffer to run every cycle for 5xN output.
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < depth; j++) {
      input[j] = i * depth + j + 1;
    }
    TF_LITE_MICRO_EXPECT_EQ(
        kTfLiteOk,
        tflite::testing::InvokeCircularBufferInt8(
            input_dims, input, output_dims, goldens[i], output_data, &node));
  }
}

TF_LITE_MICRO_TESTS_END
