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
#include "tensorflow/lite/delegates/gpu/common/tasks/padding_test_util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, PaddingAppendWidth) {
  auto status = PaddingAppendWidthTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, PaddingAppendWidthConstValues) {
  auto status = PaddingAppendWidthConstValuesTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, PaddingPrependWidth) {
  auto status = PaddingPrependWidthTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, PaddingAppendHeight) {
  auto status = PaddingAppendHeightTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, PaddingPrependHeight) {
  auto status = PaddingPrependHeightTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, PaddingAppendChannels) {
  auto status = PaddingAppendChannelsTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, PaddingPrependChannels) {
  auto status = PaddingPrependChannelsTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, PaddingPrependChannelsX4) {
  auto status = PaddingPrependChannelsX4Test(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, PaddingComplex) {
  auto status = PaddingComplexTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, PaddingReflectWidth) {
  auto status = PaddingReflectWidthTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, PaddingReflectChannels) {
  auto status = PaddingReflectChannelsTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
