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
#include "tensorflow/core/tfrt/mlrt/bytecode/function.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlrt {
namespace bc {
namespace {

TEST(FunctionTest, Function) {
  Buffer buffer;
  Allocator allocator(&buffer);

  Function::Constructor ctor = New<Function>(&allocator);

  ctor.construct_name("main");

  ctor.set_num_regs(10);

  ctor.construct_input_regs(/*size=*/2).Assign({0, 1});
  ctor.construct_output_regs(/*size=*/1).Assign({9});
  ctor.construct_output_last_uses(/*size=*/1).Assign({true});
  ctor.construct_kernels(/*size=*/3);

  Function function(buffer.Get(ctor.address()));

  EXPECT_EQ(function.name().Get(), "main");
  EXPECT_EQ(function.num_regs(), 10);

  EXPECT_THAT(function.input_regs(), ::testing::ElementsAreArray({0, 1}));
  EXPECT_THAT(function.output_regs(), ::testing::ElementsAreArray({9}));
  EXPECT_THAT(function.output_last_uses(), ::testing::ElementsAreArray({true}));

  Vector<Kernel> kernels = function.kernels();

  EXPECT_EQ(kernels.size(), 3);
}

}  // namespace
}  // namespace bc
}  // namespace mlrt
