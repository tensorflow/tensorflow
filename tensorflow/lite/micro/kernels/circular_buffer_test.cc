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
#include "tensorflow/lite/micro/kernels/circular_buffer_flexbuffers_generated_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kRunPeriod = 2;

// TODO(b/149795762): Add this to TfLiteStatus enum.
constexpr TfLiteStatus kTfLiteAbort = static_cast<TfLiteStatus>(-9);

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(OutputTensorLength4) {
  constexpr int depth = 3;
  constexpr int num_slots = 4;
  int8_t input_data[depth];
  int8_t output_data[depth * num_slots];

  memset(output_data, 0, sizeof(output_data));

  // There are four input dimensions - [1, 1, 1, depth].
  int input_dims[] = {4, 1, 1, 1, depth};
  // There are four output dimensions - [1, num_slots, 1, depth].
  int output_dims[] = {4, 1, num_slots, 1, depth};

  TfLiteIntArray* input_tensor_dims =
      tflite::testing::IntArrayFromInts(input_dims);
  TfLiteIntArray* output_tensor_dims =
      tflite::testing::IntArrayFromInts(output_dims);

  const int output_dims_count = tflite::ElementCount(*output_tensor_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      tflite::testing::CreateQuantizedTensor(input_data, input_tensor_dims, 1,
                                             0),
      tflite::testing::CreateQuantizedTensor(output_data, output_tensor_dims, 1,
                                             0),
  };

  // There is one input - tensor 0.
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array =
      tflite::testing::IntArrayFromInts(inputs_array_data);
  // There is one output - tensor 1.
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array =
      tflite::testing::IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration* registration =
      tflite::ops::micro::Register_CIRCULAR_BUFFER();
  tflite::micro::KernelRunner runner = tflite::micro::KernelRunner(
      *registration, tensors, tensors_size, inputs_array, outputs_array,
      /*builtin_data=*/nullptr);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());

  const int8_t goldens[5][16] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3},
                                 {0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6},
                                 {0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                 {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};

  // Expect the circular buffer to run every other invoke for 4xN output.
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < depth; j++) {
      input_data[j] = i * depth + j + 1;
    }
    TfLiteStatus status = runner.Invoke();

    for (int j = 0; j < output_dims_count; ++j) {
      TF_LITE_MICRO_EXPECT_EQ(goldens[i][j], output_data[j]);
    }

    // Every kRunPeriod iterations, the circular buffer should return kTfLiteOk.
    if (i % tflite::testing::kRunPeriod == tflite::testing::kRunPeriod - 1) {
      TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, status);
    } else {
      TF_LITE_MICRO_EXPECT_EQ(tflite::testing::kTfLiteAbort, status);
    }
  }
}

TF_LITE_MICRO_TEST(OutputTensorOnEveryIterationLength4) {
  constexpr int depth = 3;
  constexpr int num_slots = 4;
  int8_t input_data[depth];
  int8_t output_data[depth * num_slots];

  memset(output_data, 0, sizeof(output_data));

  // There are four input dimensions - [1, 1, 1, depth].
  int input_dims[] = {4, 1, 1, 1, depth};
  // There are four output dimensions - [1, num_slots, 1, depth].
  int output_dims[] = {4, 1, num_slots, 1, depth};

  TfLiteIntArray* input_tensor_dims =
      tflite::testing::IntArrayFromInts(input_dims);
  TfLiteIntArray* output_tensor_dims =
      tflite::testing::IntArrayFromInts(output_dims);

  const int output_dims_count = tflite::ElementCount(*output_tensor_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      tflite::testing::CreateQuantizedTensor(input_data, input_tensor_dims, 1,
                                             0),
      tflite::testing::CreateQuantizedTensor(output_data, output_tensor_dims, 1,
                                             0),
  };

  // There is one input - tensor 0.
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array =
      tflite::testing::IntArrayFromInts(inputs_array_data);
  // There is one output - tensor 1.
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array =
      tflite::testing::IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration* registration =
      tflite::ops::micro::Register_CIRCULAR_BUFFER();
  tflite::micro::KernelRunner runner = tflite::micro::KernelRunner(
      *registration, tensors, tensors_size, inputs_array, outputs_array,
      /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, runner.InitAndPrepare(reinterpret_cast<const char*>(
                                           g_gen_data_circular_buffer_config),
                                       g_gen_data_size_circular_buffer_config));

  const int8_t goldens[5][16] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3},
                                 {0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6},
                                 {0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                 {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};

  // Expect the circular buffer to run every other invoke for 4xN output.
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < depth; j++) {
      input_data[j] = i * depth + j + 1;
    }
    TfLiteStatus status = runner.Invoke();
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, status);

    for (int j = 0; j < output_dims_count; ++j) {
      TF_LITE_MICRO_EXPECT_EQ(goldens[i][j], output_data[j]);
    }
  }
}

TF_LITE_MICRO_TEST(OutputTensorLength5) {
  constexpr int depth = 4;
  constexpr int num_slots = 5;
  int8_t input_data[depth];
  int8_t output_data[depth * num_slots];

  memset(output_data, 0, sizeof(output_data));
  int input_dims[] = {4, 1, 1, 1, depth};
  int output_dims[] = {4, 1, num_slots, 1, depth};
  TfLiteIntArray* input_tensor_dims =
      tflite::testing::IntArrayFromInts(input_dims);
  TfLiteIntArray* output_tensor_dims =
      tflite::testing::IntArrayFromInts(output_dims);

  const int output_dims_count = tflite::ElementCount(*output_tensor_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      tflite::testing::CreateQuantizedTensor(input_data, input_tensor_dims, 1,
                                             0),
      tflite::testing::CreateQuantizedTensor(output_data, output_tensor_dims, 1,
                                             0),
  };

  // There is one input - tensor 0.
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array =
      tflite::testing::IntArrayFromInts(inputs_array_data);
  // There is one output - tensor 1.
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array =
      tflite::testing::IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration* registration =
      tflite::ops::micro::Register_CIRCULAR_BUFFER();
  tflite::micro::KernelRunner runner = tflite::micro::KernelRunner(
      *registration, tensors, tensors_size, inputs_array, outputs_array,
      /*builtin_data=*/nullptr);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());

  const int8_t goldens[6][20] = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8},
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
      {0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
      {5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
       15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};

  // Expect circular buffer to run every cycle for 5xN output.
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < depth; j++) {
      input_data[j] = i * depth + j + 1;
    }
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

    for (int j = 0; j < output_dims_count; ++j) {
      TF_LITE_MICRO_EXPECT_EQ(goldens[i][j], output_data[j]);
    }
  }
}

TF_LITE_MICRO_TESTS_END
