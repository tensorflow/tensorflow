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
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

int input_dims_data_common[] = {3, 1, 24, 1};
int output_dims_data_common[] = {1, 24};
const int input_data_common[] = {1,  2,  3,  4,  5,  6,  7,  8,
                                 9,  10, 11, 12, 13, 14, 15, 16,
                                 17, 18, 19, 20, 21, 22, 23, 24};
const int golden_common[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
const int expected_output_size_common = 24;

void TestSqueezeOp(int* input_dims_data, const int* input_data,
                   int* output_dims_data, int* output_data, const int* golden,
                   int expected_output_size,
                   TfLiteSqueezeParams* squeeze_params) {
  TfLiteIntArray* input_dims1 = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims1 = IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;

  TfLiteTensor tensors[tensors_size];
  tensors[0] = CreateTensor(input_data, input_dims1);
  tensors[1] = CreateTensor(output_data, output_dims1);

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = Register_SQUEEZE();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(squeeze_params));

  const char* init_data = reinterpret_cast<const char*>(squeeze_params);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare(init_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < expected_output_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden[i], output_data[i]);
  }
}
}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SqueezeAll) {
  int output_data[24];
  TfLiteSqueezeParams squeeze_params = {{}, 0};

  tflite::testing::TestSqueezeOp(tflite::testing::input_dims_data_common,
                                 tflite::testing::input_data_common,
                                 tflite::testing::output_dims_data_common,
                                 output_data, tflite::testing::golden_common,
                                 tflite::testing::expected_output_size_common,
                                 &squeeze_params);
}

TF_LITE_MICRO_TEST(SqueezeSelectedAxis) {
  int output_data[24];
  TfLiteSqueezeParams squeeze_params = {{2}, 1};
  int output_dims_data_common[] = {2, 1, 24};

  tflite::testing::TestSqueezeOp(
      tflite::testing::input_dims_data_common,
      tflite::testing::input_data_common, output_dims_data_common, output_data,
      tflite::testing::golden_common,
      tflite::testing::expected_output_size_common, &squeeze_params);
}

TF_LITE_MICRO_TEST(SqueezeNegativeAxis) {
  int output_data[24];
  TfLiteSqueezeParams squeeze_params = {{-1, 0}, 2};

  tflite::testing::TestSqueezeOp(tflite::testing::input_dims_data_common,
                                 tflite::testing::input_data_common,
                                 tflite::testing::output_dims_data_common,
                                 output_data, tflite::testing::golden_common,
                                 tflite::testing::expected_output_size_common,
                                 &squeeze_params);
}

TF_LITE_MICRO_TEST(SqueezeAllDims) {
  int input_dims_data[] = {7, 1, 1, 1, 1, 1, 1, 1};
  int output_dims_data[] = {1, 1};
  const int input_data[] = {3};
  const int golden[] = {3};
  const int expected_output_size = 1;

  int output_data[24];
  TfLiteSqueezeParams squeeze_params = {{}, 0};

  tflite::testing::TestSqueezeOp(input_dims_data, input_data, output_dims_data,
                                 output_data, golden, expected_output_size,
                                 &squeeze_params);
}

TF_LITE_MICRO_TESTS_END
