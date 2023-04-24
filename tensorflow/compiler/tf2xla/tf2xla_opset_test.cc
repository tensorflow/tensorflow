/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/tf2xla_opset.h"

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(GeXlaOpsForDeviceTest, InvalidDeviceToRegister) {
  absl::StatusOr<std::vector<std::string>> result =
      GetRegisteredXlaOpsForDevice("Invalid_Device");
  EXPECT_FALSE(result.ok());
}
TEST(GeXlaOpsForDeviceTest, GetGpuNames) {
  absl::StatusOr<std::vector<std::string>> result =
      GetRegisteredXlaOpsForDevice("XLA_GPU_JIT");
  EXPECT_GT(result.value().size(), 0);
  auto matmul =
      std::find(result.value().begin(), result.value().end(), "MatMul");
  auto max = std::find(result.value().begin(), result.value().end(), "Max");
  auto min = std::find(result.value().begin(), result.value().end(), "Min");
  EXPECT_TRUE((matmul != result.value().end()));
  EXPECT_TRUE((max != result.value().end()));
  EXPECT_TRUE((min != result.value().end()));
  EXPECT_LT(matmul, max);
  EXPECT_LT(max, min);
}
TEST(GeXlaOpsForDeviceTest, GetCpuNames) {
  absl::StatusOr<std::vector<std::string>> result =
      GetRegisteredXlaOpsForDevice("XLA_CPU_JIT");
  EXPECT_GT(result.value().size(), 0);
  auto matmul =
      std::find(result.value().begin(), result.value().end(), "MatMul");
  auto max = std::find(result.value().begin(), result.value().end(), "Max");
  auto min = std::find(result.value().begin(), result.value().end(), "Min");
  EXPECT_TRUE((matmul != result.value().end()));
  EXPECT_TRUE((max != result.value().end()));
  EXPECT_TRUE((min != result.value().end()));
  EXPECT_LT(matmul, max);
  EXPECT_LT(max, min);
}

}  // namespace
}  // namespace tensorflow
