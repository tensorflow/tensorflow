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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
struct TestDescriptor : public GPUObjectDescriptor {
  absl::Status PerformSelector(const std::string& selector,
                               const std::vector<std::string>& args,
                               std::string* result) const override {
    if (selector == "Length") {
      *result = "length";
      return absl::OkStatus();
    } else if (selector == "Read") {
      if (args.size() != 1) {
        return absl::NotFoundError(
            absl::StrCat("TestDescriptor Read require one argument, but ",
                         args.size(), " was passed"));
      }
      *result = absl::StrCat("buffer[", args[0], "]");
      return absl::OkStatus();
    } else {
      return absl::NotFoundError(absl::StrCat(
          "TestDescriptor don't have selector with name - ", selector));
    }
  }

  GPUResources GetGPUResources() const override {
    GPUResources resources;
    resources.ints.push_back("length");
    GPUBufferDescriptor desc;
    desc.data_type = DataType::FLOAT32;
    desc.element_size = 4;
    resources.buffers.push_back({"buffer", desc});
    return resources;
  }
};
}  // namespace

TEST(ArgumentsTest, TestSelectorResolve) {
  TestDescriptor descriptor;
  Arguments args;
  args.AddObjectRef("object", absl::make_unique<TestDescriptor>(descriptor));
  std::string sample_code = R"(
  if (a < 3) {
    value = args.object.Read(id);
  }
)";
  const std::string expected_result = R"(
  if (a < 3) {
    value = object_buffer[id];
  }
)";
  ASSERT_OK(args.TransformToCLCode(&sample_code));
  EXPECT_EQ(sample_code, expected_result);

  std::string cl_arguments = args.GetListOfArgs();
  EXPECT_TRUE(cl_arguments.find("__global float4* object_buffer") !=
              std::string::npos);
}

TEST(ArgumentsTest, TestNoSelector) {
  TestDescriptor descriptor;
  Arguments args;
  args.AddObjectRef("object", absl::make_unique<TestDescriptor>(descriptor));
  std::string sample_code = R"(
  if (a < 3) {
    value = args.object.Write(id);
  }
)";
  EXPECT_FALSE(args.TransformToCLCode(&sample_code).ok());
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
