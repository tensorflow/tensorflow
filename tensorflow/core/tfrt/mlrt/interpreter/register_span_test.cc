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
#include "tensorflow/core/tfrt/mlrt/interpreter/register_span.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/value.h"

namespace mlrt {
namespace {

TEST(RegisterSpan, RegisterSpan) {
  std::vector<Value> regs(4);
  regs[0].Set<int>(0);
  regs[1].Set<int>(1);
  regs[2].Set<int>(2);
  regs[3].Set<int>(3);

  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto reg_indices_ctor =
      bc::New<bc::Vector<uint32_t>>(&allocator, std::vector<uint32_t>{1, 2});

  bc::Vector<uint32_t> reg_indices(buffer.Get(reg_indices_ctor.address()));

  RegisterSpan reg_span(reg_indices, absl::MakeSpan(regs));

  ASSERT_EQ(reg_span.size(), 2);

  EXPECT_EQ(reg_span[0].Get<int>(), 1);
  EXPECT_EQ(reg_span[1].Get<int>(), 2);

  EXPECT_THAT(RegisterValueSpan<int>(reg_span),
              ::testing::ElementsAreArray({1, 2}));
}

TEST(RegisterSpan, RegisterSpanToStdVector) {
  std::vector<Value> regs(4);
  regs[0].Set<int>(0);
  regs[1].Set<int>(1);
  regs[2].Set<int>(2);
  regs[3].Set<int>(3);

  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto reg_indices_ctor =
      bc::New<bc::Vector<uint32_t>>(&allocator, std::vector<uint32_t>{1, 2});

  bc::Vector<uint32_t> reg_indices(buffer.Get(reg_indices_ctor.address()));

  RegisterSpan reg_span(reg_indices, absl::MakeSpan(regs));

  std::vector<Value> subset(reg_span.begin(), reg_span.end());

  ASSERT_EQ(subset.size(), 2);

  EXPECT_EQ(subset[0].Get<int>(), 1);
  EXPECT_EQ(subset[1].Get<int>(), 2);
}

TEST(RegisterSpan, RegisterValueSpan) {
  std::vector<Value> regs(4);
  regs[0].Set<int>(0);
  regs[1].Set<int>(1);
  regs[2].Set<int>(2);
  regs[3].Set<int>(3);

  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto reg_indices_ctor =
      bc::New<bc::Vector<uint32_t>>(&allocator, std::vector<uint32_t>{1, 3});

  bc::Vector<uint32_t> reg_indices(buffer.Get(reg_indices_ctor.address()));

  RegisterValueSpan<int> reg_span(reg_indices, absl::MakeSpan(regs));

  ASSERT_EQ(reg_span.size(), 2);

  EXPECT_EQ(reg_span[0], 1);
  EXPECT_EQ(reg_span[1], 3);

  EXPECT_THAT(reg_span, ::testing::ElementsAreArray({1, 3}));
}

TEST(RegisterSpan, Modifiers) {
  std::vector<Value> regs(4);
  regs[0].Set<int>(0);
  regs[1].Set<int>(1);
  regs[2].Set<int>(2);
  regs[3].Set<int>(3);

  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto reg_indices_ctor = bc::New<bc::Vector<uint32_t>>(
      &allocator, std::vector<uint32_t>{0, 2, 1, 3});

  bc::Vector<uint32_t> reg_indices(buffer.Get(reg_indices_ctor.address()));

  RegisterSpan reg_span(reg_indices, absl::MakeSpan(regs));

  RegisterValueSpan<int> reg_value_span(reg_span);

  EXPECT_THAT(RegisterValueSpan<int>(reg_span.drop_back(2)),
              ::testing::ElementsAreArray({0, 2}));
  EXPECT_THAT(reg_value_span.drop_front(2),
              ::testing::ElementsAreArray({1, 3}));

  reg_value_span.Destroy(1);
  EXPECT_FALSE(regs[2].HasValue());
}

}  // namespace
}  // namespace mlrt
