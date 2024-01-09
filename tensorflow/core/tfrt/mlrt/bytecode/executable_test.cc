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
#include "tensorflow/core/tfrt/mlrt/bytecode/executable.h"

#include <cstring>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlrt {
namespace bc {
namespace {

TEST(ExecutableTest, Executable) {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  Executable::Constructor executable_ctor = bc::New<bc::Executable>(&allocator);

  Vector<String>::Constructor kernel_names_ctor =
      executable_ctor.construct_kernel_names(2);
  kernel_names_ctor.ConstructAt(0, "add");
  kernel_names_ctor.ConstructAt(1, "return");

  auto attributes_ctor = executable_ctor.construct_attributes(1);

  int32_t constant = 1;
  std::string constant_str(sizeof(int32_t), '\0');
  std::memcpy(constant_str.data(), &constant, sizeof(int32_t));
  attributes_ctor.ConstructAt(0, constant_str);

  executable_ctor.construct_functions(1);

  Executable executable(buffer.Get(executable_ctor.address()));

  EXPECT_THAT(executable.kernel_names(),
              ::testing::ElementsAreArray({"add", "return"}));
  EXPECT_EQ(executable.attributes().size(), 1);

  int32_t value;
  ASSERT_EQ(executable.attributes()[0].size(), sizeof(value));
  std::memcpy(&value, executable.attributes()[0].data(), sizeof(int32_t));

  EXPECT_EQ(value, constant);

  EXPECT_EQ(executable.functions().size(), 1);
}

}  // namespace
}  // namespace bc
}  // namespace mlrt
