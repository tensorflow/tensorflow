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
#include "tensorflow/lite/delegates/gpu/cl/arguments.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/device_info.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"

namespace tflite {
namespace gpu {
namespace cl {
TEST(ArgumentsTest, TestSelectorResolve) {
  BufferDescriptor desc;
  desc.element_type = DataType::FLOAT32;
  desc.element_size = 4;
  desc.memory_type = MemoryType::GLOBAL;

  Arguments args;
  args.AddObjectRef("weights", AccessType::READ,
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  std::string sample_code = R"(
__kernel void main_function($0) {
  if (a < 3) {
    value = args.weights.Read(id);
  }
})";

  DeviceInfo device_info;
  ASSERT_OK(args.TransformToCLCode(device_info, {}, &sample_code));
  EXPECT_TRUE(absl::StrContains(sample_code, "value = weights_buffer[id];"));
  EXPECT_TRUE(
      absl::StrContains(sample_code, "__global float4* weights_buffer"));
}

TEST(ArgumentsTest, TestNoSelector) {
  BufferDescriptor desc;
  desc.element_type = DataType::FLOAT32;
  desc.element_size = 4;
  desc.memory_type = MemoryType::GLOBAL;

  Arguments args;
  args.AddObjectRef("weights", AccessType::READ,
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  std::string sample_code = R"(
  if (a < 3) {
    value = args.weights.UnknownSelector(id);
  }
)";
  DeviceInfo device_info;
  EXPECT_FALSE(args.TransformToCLCode(device_info, {}, &sample_code).ok());
}

TEST(ArgumentsTest, TestRenameArgs) {
  Arguments linkable_args;
  linkable_args.AddFloat("alpha", 0.5f);
  std::string linkable_code = "in_out_value += args.alpha;\n";
  linkable_args.RenameArgs("_link0", &linkable_code);
  EXPECT_EQ(linkable_code, "in_out_value += args.alpha_link0;\n");
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
