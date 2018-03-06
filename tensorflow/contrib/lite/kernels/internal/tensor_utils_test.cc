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
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"
#include <gmock/gmock.h>
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"

namespace tflite {
namespace tensor_utils {

TEST(uKernels, ClipTest) {
  constexpr int kVectorSize = 10;
  constexpr float kAbsLimit = 2.0;
  static float input[kVectorSize] = {0.0,  -0.5, 1.0,  -1.5, 2.0,
                                     -2.5, 3.0,  -3.5, 4.0,  -4.5};
  std::vector<float> output(kVectorSize);
  ClipVector(input, kVectorSize, kAbsLimit, output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear(
                  {0.0, -0.5, 1.0, -1.5, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0})));
}

TEST(uKernels, MatrixBatchVectorMultiplyAccumulateTest) {
  constexpr int kRow = 3;
  constexpr int kCol = 4;
  constexpr int kBatch = 2;
  static float matrix[kRow * kCol] = {1.0,  2.0,  3.0,  4.0,   //
                                      -1.0, -2.0, -3.0, -4.0,  //
                                      1.0,  -2.0, 3.0,  -4.0};
  static float vector[kCol * kBatch] = {1.0, -1.0, 1.0, -1.0,  //
                                        2.0, -2.0, 2.0, -2.0};
  std::vector<float> output(kRow * kBatch);
  std::fill(output.begin(), output.end(), 3.0);
  MatrixBatchVectorMultiplyAccumulate(matrix, kRow, kCol, vector, kBatch,
                                      output.data(), /*result_stride=*/1);
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear({1., 5., 13.,  //
                                                       -1., 7., 23.})));

  std::vector<float> output_with_stride2(kRow * kBatch * 2);
  std::fill(output_with_stride2.begin(), output_with_stride2.end(), 3.0);
  MatrixBatchVectorMultiplyAccumulate(matrix, kRow, kCol, vector, kBatch,
                                      output_with_stride2.data(),
                                      /*result_stride=*/2);
  EXPECT_THAT(output_with_stride2,
              ElementsAreArray(ArrayFloatNear({1., 3., 5., 3., 13., 3.,  //
                                               -1., 3., 7., 3., 23., 3.})));
}

TEST(uKernels, VectorVectorCwiseProductTest) {
  constexpr int kVectorSize = 10;
  static float input1[kVectorSize] = {0.0,  -0.5, 1.0,  -1.5, 2.0,
                                      -2.5, 3.0,  -3.5, 4.0,  -4.5};
  static float input2[kVectorSize] = {0.1,  -0.1, 0.1,  -0.1, 0.1,
                                      -0.1, 0.1,  -0.1, 0.1,  -0.1};
  std::vector<float> output(kVectorSize);
  VectorVectorCwiseProduct(input1, input2, kVectorSize, output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear(
                  {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45})));
}

TEST(uKernels, VectorVectorCwiseProductAccumulateTest) {
  constexpr int kVectorSize = 10;
  static float input1[kVectorSize] = {0.0,  -0.5, 1.0,  -1.5, 2.0,
                                      -2.5, 3.0,  -3.5, 4.0,  -4.5};
  static float input2[kVectorSize] = {0.1,  -0.1, 0.1,  -0.1, 0.1,
                                      -0.1, 0.1,  -0.1, 0.1,  -0.1};
  std::vector<float> output(kVectorSize);
  std::fill(output.begin(), output.end(), 1.0);
  VectorVectorCwiseProductAccumulate(input1, input2, kVectorSize,
                                     output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear(
                  {1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45})));
}

TEST(uKernels, VectorBatchVectorAssignTest) {
  constexpr int kVectorSize = 5;
  constexpr int kBatchSize = 3;
  static float input[kVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0};
  std::vector<float> output(kVectorSize * kBatchSize);
  VectorBatchVectorAssign(input, kVectorSize, kBatchSize, output.data());
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear(
                          {0.0, -0.5, 1.0, -1.5, 2.0, 0.0, -0.5, 1.0, -1.5, 2.0,
                           0.0, -0.5, 1.0, -1.5, 2.0})));
}

TEST(uKernels, ApplySigmoidToVectorTest) {
  constexpr int kVectorSize = 5;
  static float input[kVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0};
  std::vector<float> output(kVectorSize);
  ApplySigmoidToVector(input, kVectorSize, output.data());
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear(
                          {0.5, 0.377541, 0.731059, 0.182426, 0.880797})));
}

TEST(uKernels, ApplyActivationToVectorTest) {
  constexpr int kVectorSize = 5;
  static float input[kVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0};
  std::vector<float> output(kVectorSize);
  ApplyActivationToVector(input, kVectorSize, kTfLiteActRelu, output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear({0.0, 0.0, 1.0, 0.0, 2.0})));

  ApplyActivationToVector(input, kVectorSize, kTfLiteActTanh, output.data());
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear(
                          {0.0, -0.462117, 0.761594, -0.905148, 0.964028})));
}

TEST(uKernels, CopyVectorTest) {
  constexpr int kVectorSize = 5;
  static float input[kVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0};
  std::vector<float> output(kVectorSize);
  CopyVector(input, kVectorSize, output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear({0.0, -0.5, 1.0, -1.5, 2.0})));
}

TEST(uKernels, Sub1VectorTest) {
  constexpr int kVectorSize = 5;
  static float input[kVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0};
  std::vector<float> output(kVectorSize);
  Sub1Vector(input, kVectorSize, output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear({1.0, 1.5, 0.0, 2.5, -1.0})));
}

TEST(uKernels, ZeroVectorTest) {
  constexpr int kVectorSize = 5;
  std::vector<float> output(kVectorSize);
  ZeroVector(output.data(), kVectorSize);
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear({0.0, 0.0, 0.0, 0.0, 0.0})));
}

TEST(uKernels, BatchVectorBatchVectorDotProductTest) {
  constexpr int kVectorSize = 5;
  constexpr int kBatch = 2;
  static float input1[kVectorSize * kBatch] = {0.0,  -0.5, 1.0,  -1.5, 2.0,
                                               -2.5, 3.0,  -3.5, 4.0,  -4.5};
  static float input2[kVectorSize * kBatch] = {0.1,  -0.1, 0.1,  -0.1, 0.1,
                                               -0.1, 0.1,  -0.1, 0.1,  -0.1};
  std::vector<float> output(kBatch);
  BatchVectorBatchVectorDotProduct(input1, input2, kVectorSize, kBatch,
                                   output.data(), /*result_stride=*/1);
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear({0.5, 1.75})));
}

TEST(uKernels, VectorShiftLeftTest) {
  constexpr int kVectorSize = 5;
  static float input[kVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0};
  std::vector<float> result(kVectorSize);
  VectorShiftLeft(input, kVectorSize, 3.0);
  result.assign(input, input + kVectorSize);
  EXPECT_THAT(result,
              ElementsAreArray(ArrayFloatNear({-0.5, 1.0, -1.5, 2.0, 3.0})));
}

TEST(uKernels, ReductionSumVectorTest) {
  constexpr int kInputVectorSize = 10;
  constexpr int kOutputVectorSize1 = 5;
  constexpr int kReductionSize1 = 2;
  static float input[kInputVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0,
                                          0.0, -0.5, 1.0, 1.0,  2.0};
  std::vector<float> result1(kOutputVectorSize1);
  ReductionSumVector(input, result1.data(), kOutputVectorSize1,
                     kReductionSize1);
  EXPECT_THAT(result1,
              ElementsAreArray(ArrayFloatNear({-0.5, -0.5, 2.0, 0.5, 3.0})));

  constexpr int kOutputVectorSize2 = 2;
  constexpr int kReductionSize2 = 5;
  std::vector<float> result2(kOutputVectorSize2);
  ReductionSumVector(input, result2.data(), kOutputVectorSize2,
                     kReductionSize2);
  EXPECT_THAT(result2, ElementsAreArray(ArrayFloatNear({1.0, 3.5})));
}

}  // namespace tensor_utils
}  // namespace tflite
