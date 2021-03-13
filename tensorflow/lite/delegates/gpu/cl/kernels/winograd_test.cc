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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/winograd_test_util.h"

namespace tflite {
namespace gpu {
namespace cl {

TEST_F(OpenCLOperationTest, Winograd4x4To36TileX6) {
  auto status = Winograd4x4To36TileX6Test(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, Winograd36To4x4Tile4x1) {
  auto status = Winograd36To4x4Tile4x1Test(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, Winograd4x4To36) {
  auto status = Winograd4x4To36Test(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, Winograd36To4x4) {
  auto status = Winograd36To4x4Test(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
