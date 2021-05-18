/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

template <typename ParamType, typename IndexType>
void TestGatherNd(int* param_dims, const ParamType* param_data, int* index_dims,
                  const IndexType* index_data, int* output_dims,
                  ParamType* output_data,
                  const ParamType* expected_output_data) {
  TfLiteIntArray* pdims = IntArrayFromInts(param_dims);
  TfLiteIntArray* idims = IntArrayFromInts(index_dims);
  TfLiteIntArray* odims = IntArrayFromInts(output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(param_data, pdims),
      CreateTensor(index_data, idims),
      CreateTensor(output_data, odims),
  };
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = Register_GATHER_ND();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, /*builtin_data=*/nullptr);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  // The output tensor's data and shape have been updated by the kernel.
  TfLiteTensor* actual_output_tensor = &tensors[2];
  TfLiteIntArray* actual_output_dims = actual_output_tensor->dims;
  const int output_size = ElementCount(*actual_output_dims);
  for (int i = 0; i < output_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(GatherNd_ElementIndexingIntoMatrix) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {2, 2, 2};
  int index_dims[] = {2, 2, 2};
  const int32_t index_data[] = {0, 0, 1, 1};
  const float input_data[] = {1.1, 1.2, 2.1, 2.2};
  const float golden_data[] = {1.1, 2.2};
  float output_data[2];
  int output_dims[] = {1, 0};
  tflite::testing::TestGatherNd<float, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TEST(GatherNd_SliceIndexingIntoMatrix) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {2, 2, 2};
  int index_dims[] = {2, 2, 1};
  const int32_t index_data[] = {1, 0};
  const float input_data[] = {1.1, 1.2, 2.1, 2.2};
  const float golden_data[] = {2.1, 2.2, 1.1, 1.2};
  float output_data[4];
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGatherNd<float, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TEST(GatherNd_BatchedIndexingIntoMatrix1) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {2, 2, 2};
  int index_dims[] = {3, 2, 1, 1};
  const int32_t index_data[] = {1, 0};
  const float input_data[] = {1.1, 1.2, 2.1, 2.2};
  const float golden_data[] = {2.1, 2.2, 1.1, 1.2};
  float output_data[4];
  int output_dims[] = {3, 0, 0, 0};
  tflite::testing::TestGatherNd<float, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TEST(GatherNd_BatchedIndexingIntoMatrix2) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {2, 2, 2};
  int index_dims[] = {3, 2, 1, 2};
  const int32_t index_data[] = {0, 0, 1, 1};
  const float input_data[] = {1.1, 1.2, 2.1, 2.2};
  const float golden_data[] = {1.1, 2.2};
  float output_data[2];
  int output_dims[] = {3, 0, 0, 0};
  tflite::testing::TestGatherNd<float, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TEST(GatherNd_DuplicateIndexingIntoMatrix) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {2, 2, 2};
  int index_dims[] = {2, 2, 2};
  const int32_t index_data[] = {0, 0, 0, 0};
  const float input_data[] = {1.1, 1.2, 2.1, 2.2};
  const float golden_data[] = {1.1, 1.1};
  float output_data[2];
  int output_dims[] = {1, 0};
  tflite::testing::TestGatherNd<float, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TEST(GatherNd_ElementIndexingIntoRank3Tensor) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {3, 3, 2, 3};
  int index_dims[] = {3, 1, 2, 3};
  const int32_t index_data[] = {0, 0, 1, 1, 1, 0};
  const float input_data[] = {1.1, -1.2, 1.3,  -2.1, 2.2,  2.3,  //
                              3.1, 3.2,  -3.3, -4.1, -4.2, 4.3,  //
                              5.1, -5.2, 5.3,  6.1,  -6.2, 6.3};
  const float golden_data[] = {-1.2, -4.1};
  float output_data[2];
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGatherNd<float, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TEST(GatherNd_SliceIndexingIntoRank3Tensor) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {3, 3, 2, 3};
  int index_dims[] = {2, 2, 1};
  const int32_t index_data[] = {0, 2};
  const float input_data[] = {1.1, -1.2, 1.3,  -2.1, 2.2,  2.3,  //
                              3.1, 3.2,  -3.3, -4.1, -4.2, 4.3,  //
                              5.1, -5.2, 5.3,  6.1,  -6.2, 6.3};
  const float golden_data[] = {1.1, -1.2, 1.3, -2.1, 2.2,  2.3,
                               5.1, -5.2, 5.3, 6.1,  -6.2, 6.3};
  float output_data[12];
  int output_dims[] = {3, 0, 0, 0};
  tflite::testing::TestGatherNd<float, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TEST(GatherNd_BatchedIndexingIntoRank3Tensor1) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {3, 3, 2, 3};
  int index_dims[] = {3, 2, 1, 3};
  const int32_t index_data[] = {0, 0, 1, 1, 1, 0};
  const float input_data[] = {1.1, -1.2, 1.3,  -2.1, 2.2,  2.3,  //
                              3.1, 3.2,  -3.3, -4.1, -4.2, 4.3,  //
                              5.1, -5.2, 5.3,  6.1,  -6.2, 6.3};
  const float golden_data[] = {-1.2, -4.1};
  float output_data[2];
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGatherNd<float, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TEST(GatherNd_BatchedIndexingIntoRank3Tensor2) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {3, 3, 2, 3};
  int index_dims[] = {3, 3, 1, 1};
  const int32_t index_data[] = {1, 2, 0};
  const float input_data[] = {1.1, -1.2, 1.3,  -2.1, 2.2,  2.3,  //
                              3.1, 3.2,  -3.3, -4.1, -4.2, 4.3,  //
                              5.1, -5.2, 5.3,  6.1,  -6.2, 6.3};
  const float golden_data[] = {3.1, 3.2,  -3.3, -4.1, -4.2, 4.3,
                               5.1, -5.2, 5.3,  6.1,  -6.2, 6.3,
                               1.1, -1.2, 1.3,  -2.1, 2.2,  2.3};
  float output_data[18];
  int output_dims[] = {4, 0, 0, 0, 0};
  tflite::testing::TestGatherNd<float, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TEST(GatherNd_BatchedIndexingIntoRank3Tensor3) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {3, 3, 2, 3};
  int index_dims[] = {3, 2, 2, 2};
  const int32_t index_data[] = {0, 1, 1, 0, 0, 0, 2, 1};
  const float input_data[] = {1.1, -1.2, 1.3,  -2.1, 2.2,  2.3,  //
                              3.1, 3.2,  -3.3, -4.1, -4.2, 4.3,  //
                              5.1, -5.2, 5.3,  6.1,  -6.2, 6.3};
  const float golden_data[] = {-2.1, 2.2,  2.3, 3.1, 3.2,  -3.3,
                               1.1,  -1.2, 1.3, 6.1, -6.2, 6.3};
  float output_data[12];
  int output_dims[] = {3, 0, 0, 0};
  tflite::testing::TestGatherNd<float, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TEST(GatherNd_BatchedIndexingIntoRank3Tensor4) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {3, 3, 2, 3};
  int index_dims[] = {3, 2, 2, 3};
  const int32_t index_data[] = {0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 1, 2};
  const float input_data[] = {1.1, -1.2, 1.3,  -2.1, 2.2,  2.3,  //
                              3.1, 3.2,  -3.3, -4.1, -4.2, 4.3,  //
                              5.1, -5.2, 5.3,  6.1,  -6.2, 6.3};
  const float golden_data[] = {-1.2, 3.2, 4.3, 6.3};
  float output_data[4];
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGatherNd<float, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TEST(GatherNd_DuplicateIndexingIntoRank3Tensor) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {3, 3, 2, 3};
  int index_dims[] = {2, 2, 2};
  const int32_t index_data[] = {0, 1, 0, 1};
  const float input_data[] = {1.1, -1.2, 1.3,  -2.1, 2.2,  2.3,  //
                              3.1, 3.2,  -3.3, -4.1, -4.2, 4.3,  //
                              5.1, -5.2, 5.3,  6.1,  -6.2, 6.3};
  const float golden_data[] = {-2.1, 2.2, 2.3, -2.1, 2.2, 2.3};
  float output_data[6];
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGatherNd<float, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TEST(GatherNd_Float32Int32) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {3, 3, 2, 3};
  int index_dims[] = {2, 2, 2};
  const int32_t index_data[] = {0, 1, 1, 0};
  const float input_data[] = {1.1, -1.2, 1.3,  -2.1, 2.2,  2.3,  //
                              3.1, 3.2,  -3.3, -4.1, -4.2, 4.3,  //
                              5.1, -5.2, 5.3,  6.1,  -6.2, 6.3};
  const float golden_data[] = {-2.1, 2.2, 2.3, 3.1, 3.2, -3.3};
  float output_data[6];
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGatherNd<float, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TEST(GatherNd_Int8Int32) {
  // For input_dims[], index_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {3, 3, 2, 3};
  int index_dims[] = {2, 2, 2};
  const int32_t index_data[] = {0, 1, 1, 0};
  const int8_t input_data[] = {1, -1, 1,  -2, 2,  2,  //
                               3, 3,  -3, -4, -4, 4,  //
                               5, -5, 5,  6,  -6, 6};
  const int8_t golden_data[] = {-2, 2, 2, 3, 3, -3};
  int8_t output_data[6];
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGatherNd<int8_t, int32_t>(
      input_dims, input_data, index_dims, index_data, output_dims, output_data,
      golden_data);
}

TF_LITE_MICRO_TESTS_END
