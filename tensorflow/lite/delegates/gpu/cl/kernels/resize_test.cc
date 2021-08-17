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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/resize_test_util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, ResizeBilinearAligned) {
  auto status = ResizeBilinearAlignedTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, ResizeBilinearNonAligned) {
  auto status = ResizeBilinearNonAlignedTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, ResizeBilinearWithoutHalfPixel) {
  auto status = ResizeBilinearWithoutHalfPixelTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, ResizeBilinearWithHalfPixel) {
  auto status = ResizeBilinearWithHalfPixelTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, ResizeNearest) {
  auto status = ResizeNearestTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, ResizeNearestAlignCorners) {
  auto status = ResizeNearestAlignCornersTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, ResizeNearestHalfPixelCenters) {
  auto status = ResizeNearestHalfPixelCentersTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
