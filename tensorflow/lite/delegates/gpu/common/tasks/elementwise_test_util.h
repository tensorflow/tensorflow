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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_ELEMENTWISE_TEST_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_ELEMENTWISE_TEST_UTIL_H_

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"

namespace tflite {
namespace gpu {

absl::Status AbsTest(TestExecutionEnvironment* env);
absl::Status CosTest(TestExecutionEnvironment* env);
absl::Status CopyTest(TestExecutionEnvironment* env);
absl::Status EluTest(TestExecutionEnvironment* env);
absl::Status ExpTest(TestExecutionEnvironment* env);
absl::Status FloorTest(TestExecutionEnvironment* env);
absl::Status FloorDivTest(TestExecutionEnvironment* env);
absl::Status FloorModTest(TestExecutionEnvironment* env);
absl::Status HardSwishTest(TestExecutionEnvironment* env);
absl::Status LogTest(TestExecutionEnvironment* env);
absl::Status NegTest(TestExecutionEnvironment* env);
absl::Status RsqrtTest(TestExecutionEnvironment* env);
absl::Status SigmoidTest(TestExecutionEnvironment* env);
absl::Status SinTest(TestExecutionEnvironment* env);
absl::Status SqrtTest(TestExecutionEnvironment* env);
absl::Status SquareTest(TestExecutionEnvironment* env);
absl::Status TanhTest(TestExecutionEnvironment* env);
absl::Status SubTest(TestExecutionEnvironment* env);
absl::Status SquaredDiffTest(TestExecutionEnvironment* env);
absl::Status DivTest(TestExecutionEnvironment* env);
absl::Status PowTest(TestExecutionEnvironment* env);
absl::Status AddTest(TestExecutionEnvironment* env);
absl::Status MaximumTest(TestExecutionEnvironment* env);
absl::Status MaximumWithScalarTest(TestExecutionEnvironment* env);
absl::Status MaximumWithConstantLinearTensorTest(TestExecutionEnvironment* env);
absl::Status MaximumWithConstantHWCTensorTest(TestExecutionEnvironment* env);
absl::Status MaximumWithConstantHWCTensorBroadcastChannelsTest(
    TestExecutionEnvironment* env);
absl::Status MinimumTest(TestExecutionEnvironment* env);
absl::Status MinimumWithScalarTest(TestExecutionEnvironment* env);
absl::Status MulTest(TestExecutionEnvironment* env);
absl::Status MulBroadcastHWTest(TestExecutionEnvironment* env);
absl::Status MulBroadcastChannelsTest(TestExecutionEnvironment* env);
absl::Status SubWithScalarAtFirstPositionTest(TestExecutionEnvironment* env);
absl::Status LessTest(TestExecutionEnvironment* env);
absl::Status LessEqualTest(TestExecutionEnvironment* env);
absl::Status GreaterTest(TestExecutionEnvironment* env);
absl::Status GreaterEqualTest(TestExecutionEnvironment* env);
absl::Status EqualTest(TestExecutionEnvironment* env);
absl::Status NotEqualTest(TestExecutionEnvironment* env);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_ELEMENTWISE_TEST_UTIL_H_
