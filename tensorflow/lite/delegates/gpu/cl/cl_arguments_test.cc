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
#include "tensorflow/lite/delegates/gpu/cl/cl_arguments.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_test.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"

namespace tflite {
namespace gpu {
namespace cl {
TEST(CLArgumentsTest, TestSelectorResolve) {
  BufferDescriptor desc;
  desc.element_type = DataType::FLOAT32;
  desc.element_size = 4;
  desc.memory_type = MemoryType::GLOBAL;

  Arguments args;
  args.AddObjectRef("weights", AccessType::READ,
                    std::make_unique<BufferDescriptor>(std::move(desc)));
  std::string sample_code = R"(
__kernel void main_function($0) {
  if (a < 3) {
    value = args.weights.Read(id);
  }
})";

  CLArguments cl_args;
  GpuInfo gpu_info;
  ASSERT_OK(cl_args.Init(gpu_info, nullptr, &args, &sample_code));
  EXPECT_TRUE(absl::StrContains(sample_code, "value = weights_buffer[id];"));
  EXPECT_TRUE(
      absl::StrContains(sample_code, "__global float4* weights_buffer"));
}

TEST(CLArgumentsTest, TestNoSelector) {
  BufferDescriptor desc;
  desc.element_type = DataType::FLOAT32;
  desc.element_size = 4;
  desc.memory_type = MemoryType::GLOBAL;

  Arguments args;
  args.AddObjectRef("weights", AccessType::READ,
                    std::make_unique<BufferDescriptor>(std::move(desc)));
  std::string sample_code = R"(
  if (a < 3) {
    value = args.weights.UnknownSelector(id);
  }
)";
  CLArguments cl_args;
  GpuInfo gpu_info;
  EXPECT_FALSE(cl_args.Init(gpu_info, nullptr, &args, &sample_code).ok());
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
