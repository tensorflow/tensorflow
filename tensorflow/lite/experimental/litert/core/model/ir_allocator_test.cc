// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/model/ir_allocator.h"

#include <cstddef>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAreArray;

static constexpr auto kCustomOpCode = kLiteRtOpCodeTflCustom;
static constexpr auto kNonCustomOpCode = kLiteRtOpCodeTflSoftmax;

TEST(IrAllocatorTest, EmplaceBack) {
  IrAllocator<LiteRtOpT> ops;

  LiteRtOpT my_op;
  my_op.SetOpCode(kCustomOpCode);

  ops.EmplaceBack(std::move(my_op));
  ASSERT_EQ(ops.Elements().size(), 1);
  EXPECT_EQ(ops.Elements().at(0)->OpCode(), kCustomOpCode);
}

TEST(IrAllocatorTest, RemoveIf) {
  IrAllocator<LiteRtOpT> ops;

  LiteRtOpT my_op;
  my_op.SetOpCode(kNonCustomOpCode);
  ops.EmplaceBack(std::move(my_op));

  LiteRtOpT my_op2;
  my_op2.SetOpCode(kCustomOpCode);
  ops.EmplaceBack(std::move(my_op2));

  LiteRtOpT my_op3;
  my_op3.SetOpCode(kCustomOpCode);
  ops.EmplaceBack(std::move(my_op3));

  LiteRtOpT my_op4;
  my_op4.SetOpCode(kNonCustomOpCode);
  ops.EmplaceBack(std::move(my_op4));

  auto pred = [](const auto& op) { return op.OpCode() != kCustomOpCode; };
  ASSERT_EQ(ops.RemoveIf(pred), 2);

  ASSERT_EQ(ops.Elements().size(), 2);
  ASSERT_EQ(ops.Elements().at(0)->OpCode(), kCustomOpCode);
  ASSERT_EQ(ops.Elements().at(1)->OpCode(), kCustomOpCode);
}

TEST(IrAllocatorTest, ResizeDown) {
  IrAllocator<LiteRtOpT> ops;

  LiteRtOp op1 = nullptr;
  {
    LiteRtOpT my_op;
    my_op.SetOpCode(kNonCustomOpCode);
    op1 = &ops.EmplaceBack(std::move(my_op));
  }

  {
    LiteRtOpT my_op2;
    my_op2.SetOpCode(kCustomOpCode);
    ops.EmplaceBack(std::move(my_op2));
  }

  ops.ResizeDown(1);

  ASSERT_EQ(ops.Size(), 1);
  EXPECT_EQ(ops.Elements().at(0), op1);
}

TEST(IrAllocatorTest, Transfer) {
  IrAllocator<LiteRtOpT> ops;
  auto& op1 = ops.EmplaceBack();
  auto& op2 = ops.EmplaceBack();

  IrAllocator<LiteRtOpT> other_ops;
  auto& other_op1 = other_ops.EmplaceBack();
  auto& other_op2 = other_ops.EmplaceBack();

  ops.TransferFrom(std::move(other_ops));

  EXPECT_THAT(ops.Elements(),
              ElementsAreArray({&op1, &op2, &other_op1, &other_op2}));
}

TEST(IrAllocatorTest, TransferWithIndices) {
  IrAllocator<LiteRtOpT> ops;
  auto& op1 = ops.EmplaceBack();
  auto& op2 = ops.EmplaceBack();

  IrAllocator<LiteRtOpT> other_ops;
  auto& other_op1 = other_ops.EmplaceBack();
  auto& other_op2 = other_ops.EmplaceBack();
  auto& other_op3 = other_ops.EmplaceBack();
  auto& other_op4 = other_ops.EmplaceBack();

  std::vector<size_t> indices = {1, 3};
  ops.TransferFrom(other_ops, std::move(indices));

  EXPECT_THAT(other_ops.Elements(), ElementsAreArray({&other_op1, &other_op3}));
  EXPECT_THAT(ops.Elements(),
              ElementsAreArray({&op1, &op2, &other_op2, &other_op4}));
}

}  // namespace
}  // namespace litert::internal
