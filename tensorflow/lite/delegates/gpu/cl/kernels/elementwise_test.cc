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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, Abs) { ASSERT_OK(AbsTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Cos) { ASSERT_OK(CosTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Copy) { ASSERT_OK(CopyTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Elu) { ASSERT_OK(EluTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Exp) { ASSERT_OK(ExpTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Floor) { ASSERT_OK(FloorTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, FloorDiv) { ASSERT_OK(FloorDivTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, FloorMod) { ASSERT_OK(FloorModTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, HardSwish) { ASSERT_OK(HardSwishTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Log) { ASSERT_OK(LogTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Neg) { ASSERT_OK(NegTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Rsqrt) { ASSERT_OK(RsqrtTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Sigmoid) { ASSERT_OK(SigmoidTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Sin) { ASSERT_OK(SinTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Sqrt) { ASSERT_OK(SqrtTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Square) { ASSERT_OK(SquareTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Tanh) { ASSERT_OK(TanhTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Sub) { ASSERT_OK(SubTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, SquaredDiff) {
  ASSERT_OK(SquaredDiffTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, Div) { ASSERT_OK(DivTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Pow) { ASSERT_OK(PowTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Add) { ASSERT_OK(AddTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Maximum) { ASSERT_OK(MaximumTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, MaximumWithScalar) {
  ASSERT_OK(MaximumWithScalarTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, MaximumWithConstantLinearTensor) {
  ASSERT_OK(MaximumWithConstantLinearTensorTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, MaximumWithConstantHWCTensor) {
  ASSERT_OK(MaximumWithConstantHWCTensorTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, MaximumWithConstantHWCTensorBroadcastChannels) {
  ASSERT_OK(MaximumWithConstantHWCTensorBroadcastChannelsTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, Minimum) { ASSERT_OK(MinimumTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, MinimumWithScalar) {
  ASSERT_OK(MinimumWithScalarTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, Mul) { ASSERT_OK(MulTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, MulBroadcastHW) {
  ASSERT_OK(MulBroadcastHWTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, MulBroadcastChannels) {
  ASSERT_OK(MulBroadcastChannelsTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, SubWithScalarAtFirstPosition) {
  ASSERT_OK(SubWithScalarAtFirstPositionTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, Less) { ASSERT_OK(LessTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, LessEqual) { ASSERT_OK(LessEqualTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, Greater) { ASSERT_OK(GreaterTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, GreaterEqual) {
  ASSERT_OK(GreaterEqualTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, Equal) { ASSERT_OK(EqualTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, NotEqual) { ASSERT_OK(NotEqualTest(&exec_env_)); }

TEST_F(OpenCLOperationTest, CosBroadcast) {
  ASSERT_OK(CosBroadcastTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, MaximumScalarBroadcastInput) {
  ASSERT_OK(MaximumScalarBroadcastInputTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, MulLinearBroadcastInput) {
  ASSERT_OK(MulLinearBroadcastInputTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, MulBroadcastBothInputs) {
  ASSERT_OK(MulBroadcastBothInputsTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, LogicalAndTest) {
  ASSERT_OK(LogicalAndTest(&exec_env_));
}

TEST_F(OpenCLOperationTest, LogicalAndWithConstantTest) {
  ASSERT_OK(LogicalAndWithConstantTest(&exec_env_));
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
