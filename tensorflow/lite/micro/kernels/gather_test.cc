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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

template <typename InType, typename PosType>
void TestGather(int* input_dims, const InType* input_data, int* positions_dims,
                const PosType* positions_data, int* output_dims,
                InType* output_data, const int* expected_output_dims,
                const InType* expected_output_data, const int axis = 0,
                const int batch_dims = 0) {
  TfLiteIntArray* in_dims = IntArrayFromInts(input_dims);
  TfLiteIntArray* pos_dims = IntArrayFromInts(positions_dims);
  TfLiteIntArray* out_dims = IntArrayFromInts(output_dims);
  TfLiteGatherParams params = {axis, batch_dims};

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, in_dims),
      CreateTensor(positions_data, pos_dims),
      CreateTensor(output_data, out_dims, true),
  };
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = Register_GATHER();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, &params);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  // The output tensor's data and shape have been updated by the kernel.
  TfLiteTensor* actual_output_tensor = &tensors[2];
  TfLiteIntArray* actual_output_dims = actual_output_tensor->dims;
  const int actual_output_dims_size = actual_output_dims->size;
  const int output_size = ElementCount(*actual_output_dims);
  for (int i = 0; i < output_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }

  // Compare output tensor's shape if expected_output_dims[] is provided.
  for (int i = 0; i < actual_output_dims_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_dims[i],
                            actual_output_dims->data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

// For all test functions below, dims[0] is the dimension count.
TF_LITE_MICRO_TEST(GatherOp_Shuffle) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {2, 2, 2};
  int positions_dims[] = {1, 2};
  const int32_t positions_data[] = {1, 0};
  const float input_data[] = {-2.0, 0.2, 0.7, 0.8};
  const float golden_data[] = {0.7, 0.8, -2, 0.2};
  float output_data[4];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {2, 2};
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOp_Test0DIndex) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {2, 2, 2};
  int positions_dims[] = {0};
  const int32_t positions_data[] = {1};
  const float input_data[] = {-2.0, 0.2, 0.7, 0.8};
  const float golden_data[] = {0.7, 0.8};
  float output_data[2];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {2};
  int output_dims[] = {1, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOp_Test0DIndexWith0DResult) {
  // 0D tensor is special case in current TFLite. Test it once to make sure
  // existing workarounds are fine with it.
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {1, 3};
  int positions_dims[] = {0};
  const int32_t positions_data[] = {1};
  const float input_data[] = {1.0, 2.0, 3.0};
  const float golden_data[] = {2.0};
  float output_data[1];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {0};
  int output_dims[] = {1, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOp_Test1DInput1DIndex) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {1, 3};
  int positions_dims[] = {1, 1};
  const int32_t positions_data[] = {1};
  const float input_data[] = {1.0, 3.0, 5.0};
  const float golden_data[] = {3.0};
  float output_data[1];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {1};
  int output_dims[] = {1, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOp_Test2DIndexWith2DResult) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {1, 3};
  int positions_dims[] = {2, 1, 2};
  const int32_t positions_data[] = {1, 0};
  const float input_data[] = {1.0, 2.0, 3.0};
  const float golden_data[] = {2.0, 1.0};
  float output_data[2];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {1, 2};
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOp_Duplicate) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {3, 1, 2, 2};
  int positions_dims[] = {1, 2};
  const int32_t positions_data[] = {0, 0};
  const float input_data[] = {-2.0, 0.2, 0.7, 0.8};
  const float golden_data[] = {-2, 0.2, 0.7, 0.8, -2, 0.2, 0.7, 0.8};
  float output_data[8];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {2, 2, 2};
  int output_dims[] = {3, 0, 0, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOp_Slice) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {2, 4, 1};
  int positions_dims[] = {1, 2};
  const int32_t positions_data[] = {1, 3};
  const float input_data[] = {-2.0, 0.2, 0.7, 0.8};
  const float golden_data[] = {0.2, 0.8};
  float output_data[2];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {2, 1};
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOp_Axis1) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  const int axis = 1;
  int input_dims[] = {3, 1, 2, 3};
  int positions_dims[] = {1, 2};
  const int32_t positions_data[] = {1, 0};
  const float input_data[] = {1, 2, 3, 4, 5, 6};
  const float golden_data[] = {4, 5, 6, 1, 2, 3};
  float output_data[6];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {1, 2, 3};
  int output_dims[] = {3, 0, 0, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis);
}

TF_LITE_MICRO_TEST(GatherOp_Axis1_0DIndex) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  const int axis = 1;
  int input_dims[] = {3, 1, 3, 2};
  int positions_dims[] = {0};
  const int32_t positions_data[] = {1};
  const float input_data[] = {1, 2, 3, 4, 5, 6};
  const float golden_data[] = {3, 4};
  float output_data[2];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {1, 2};
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis);
}

TF_LITE_MICRO_TEST(GatherOp_Axis1Slice) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  const int axis = 1;
  int input_dims[] = {3, 1, 4, 2};
  int positions_dims[] = {1, 2};
  const int32_t positions_data[] = {3, 1};
  const float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const float golden_data[] = {7, 8, 3, 4};
  float output_data[4];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {1, 2, 2};
  int output_dims[] = {3, 0, 0, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis);
}

TF_LITE_MICRO_TEST(GatherOp_LastAxis) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  const int axis = -1;
  int input_dims[] = {3, 1, 2, 3};
  int positions_dims[] = {1, 2};
  const int32_t positions_data[] = {2, 0};
  const float input_data[] = {1, 2, 3, 4, 5, 6};
  const float golden_data[] = {3, 1, 6, 4};
  float output_data[4];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {1, 2, 2};
  int output_dims[] = {3, 0, 0, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis);
}

TF_LITE_MICRO_TEST(GatherOp_LastAxis0DIndex) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  const int axis = -1;
  int input_dims[] = {3, 1, 2, 3};
  int positions_dims[] = {0};
  const int32_t positions_data[] = {2};
  const float input_data[] = {1, 2, 3, 4, 5, 6};
  const float golden_data[] = {3, 6};
  float output_data[2];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {1, 2};
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis);
}

TF_LITE_MICRO_TEST(GatherOp_Float32Int32) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {2, 2, 2};
  int positions_dims[] = {1, 2};
  const int32_t positions_data[] = {1, 0};
  const float input_data[] = {13.3, -13.4, -1.4, 1.5};
  const float golden_data[] = {-1.4, 1.5, 13.3, -13.4};
  float output_data[4];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {2, 2};
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOp_Int8Int32) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  int input_dims[] = {2, 2, 2};
  int positions_dims[] = {1, 2};
  const int32_t positions_data[] = {1, 0};
  const int8_t input_data[] = {-13, -120, 14, 15};
  const int8_t golden_data[] = {14, 15, -13, -120};
  int8_t output_data[4];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {2, 2};
  int output_dims[] = {2, 0, 0};
  tflite::testing::TestGather<int8_t, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOp_BatchDims2) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  const int axis = 2;
  const int batch_dims = 2;
  int input_dims[] = {4, 2, 2, 3, 5};
  int positions_dims[] = {3, 2, 2, 2};
  const int32_t positions_data[] = {1, 0, 0, 1, 1, 0, 0, 1};
  const float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                              24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                              36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                              48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
  const float golden_data[] = {5,  6,  7,  8,  9,  0,  1,  2,  3,  4,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                               35, 36, 37, 38, 39, 30, 31, 32, 33, 34,
                               45, 46, 47, 48, 49, 50, 51, 52, 53, 54};
  float output_data[40];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {2, 2, 2, 5};
  int output_dims[] = {4, 0, 0, 0, 0};
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis, batch_dims);
}

TF_LITE_MICRO_TEST(GatherOp_BatchDims1) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  const int axis = 2;
  const int batch_dims = 1;
  int input_dims[] = {4, 2, 2, 3, 5};
  int positions_dims[] = {3, 2, 2, 2};
  const int32_t positions_data[] = {1, 0, 0, 1, 1, 0, 0, 1};
  const int8_t input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                               12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                               36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
  const int8_t golden_data[] = {
      5,  6,  7,  8,  9,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  5,
      6,  7,  8,  9,  20, 21, 22, 23, 24, 15, 16, 17, 18, 19, 15, 16,
      17, 18, 19, 20, 21, 22, 23, 24, 35, 36, 37, 38, 39, 30, 31, 32,
      33, 34, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53,
      54, 45, 46, 47, 48, 49, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54};
  int8_t output_data[80];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {2, 2, 2, 2, 5};
  int output_dims[] = {5, 0, 0, 0, 0, 0};
  tflite::testing::TestGather<int8_t, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis, batch_dims);
}

TF_LITE_MICRO_TEST(GatherOp_NegativeBatchDims) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  const int axis = 2;
  const int batch_dims = -2;
  int input_dims[] = {4, 2, 2, 3, 5};
  int positions_dims[] = {3, 2, 2, 2};
  const int32_t positions_data[] = {1, 0, 0, 1, 1, 0, 0, 1};
  const int8_t input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                               12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                               36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
  const int8_t golden_data[] = {
      5,  6,  7,  8,  9,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  5,
      6,  7,  8,  9,  20, 21, 22, 23, 24, 15, 16, 17, 18, 19, 15, 16,
      17, 18, 19, 20, 21, 22, 23, 24, 35, 36, 37, 38, 39, 30, 31, 32,
      33, 34, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53,
      54, 45, 46, 47, 48, 49, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54};
  int8_t output_data[80];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {2, 2, 2, 2, 5};
  int output_dims[] = {5, 0, 0, 0, 0, 0};
  tflite::testing::TestGather<int8_t, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis, batch_dims);
}

TF_LITE_MICRO_TEST(GatherOp_BatchDimsEqualIndiceDims) {
  // For input_dims[], positions_dims[], or output_dims[], element 0 is the
  // number of dimensions in that array, not the actual dimension data.
  const int axis = 3;
  const int batch_dims = 3;
  int input_dims[] = {4, 2, 2, 2, 5};
  int positions_dims[] = {3, 2, 2, 2};
  const int32_t positions_data[] = {1, 0, 0, 1, 1, 0, 0, 1};
  const int8_t input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                               10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  const int8_t golden_data[] = {1, 5, 10, 16, 21, 25, 30, 36};
  int8_t output_data[8];

  // The kernel under test will fill output_dims[1] onward, to be compared
  // against golden_dims[0] onward.
  const int golden_dims[] = {2, 2, 2};
  int output_dims[] = {3, 0, 0, 0};
  tflite::testing::TestGather<int8_t, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis, batch_dims);
}

TF_LITE_MICRO_TESTS_END
