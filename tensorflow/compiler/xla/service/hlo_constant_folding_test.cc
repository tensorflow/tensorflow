/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/types.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using HloConstantFoldingTest = HloTestBase;

TEST_F(HloConstantFoldingTest, ConvertF32ToS64) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(S64, {}), input));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Convert(input));

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  EXPECT_THAT(computation->root_instruction(), op::Constant());
  EXPECT_EQ(computation->root_instruction()->literal().GetFirstElement<int64>(),
            42);
}

TEST_F(HloConstantFoldingTest, ConvertS64ToF32) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64>(42)));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(F32, {}), input));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Convert(input));

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  EXPECT_THAT(computation->root_instruction(), op::Constant());
  EXPECT_EQ(computation->root_instruction()->literal().GetFirstElement<float>(),
            42.0f);
}

TEST_F(HloConstantFoldingTest, ConvertF32ArrayToS64Array) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({42.0f, 19.0f})));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(S64, {2}), input));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Convert(input));

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  EXPECT_THAT(computation->root_instruction(), op::Constant());
  EXPECT_EQ(computation->root_instruction()->literal().Get<int64>({0}), 42);
  EXPECT_EQ(computation->root_instruction()->literal().Get<int64>({1}), 19);
}

TEST_F(HloConstantFoldingTest, Concatenate) {
  const struct TestConfig {
    int concat_dimension;
    tensorflow::gtl::ArraySlice<int64> dimensions;
    tensorflow::gtl::ArraySlice<int64> concat_sizes;
  } test_configs[] = {
      {1, {11, 0, 7, 5, 9}, {2, 5, 7, 11}},
      {3, {1, 4, 17, 0, 8}, {1, 3, 9, 12}},
  };

  for (auto& test_config : test_configs) {
    HloComputation::Builder builder(TestName());
    std::vector<int64> dimensions(test_config.dimensions.begin(),
                                  test_config.dimensions.end());
    int64 concat_size = 0;
    std::vector<HloInstruction*> operands;
    for (auto csize : test_config.concat_sizes) {
      dimensions[test_config.concat_dimension] = csize;
      concat_size += csize;
      auto literal = LiteralUtil::CreateFromDimensions(F32, dimensions);
      HloInstruction* insn = builder.AddInstruction(
          HloInstruction::CreateConstant(std::move(literal)));
      operands.push_back(insn);
    }
    dimensions[test_config.concat_dimension] = concat_size;
    Shape shape = ShapeUtil::MakeShape(F32, dimensions);
    builder.AddInstruction(HloInstruction::CreateConcatenate(
        shape, operands, test_config.concat_dimension));
    auto module = CreateNewModule();
    auto computation = module->AddEntryComputation(builder.Build());

    HloConstantFolding const_folder;
    TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
    EXPECT_TRUE(result);

    HloInstruction* root = computation->root_instruction();
    EXPECT_THAT(root, op::Constant());
    EXPECT_TRUE(ShapeUtil::Equal(root->shape(), shape));
  }
}

TEST_F(HloConstantFoldingTest, Slice) {
  HloComputation::Builder builder(TestName());
  const int64 dimensions[] = {11, 8, 7, 5, 9};
  const int64 slice_start[] = {4, 2, 3, 1, 5};
  const int64 slice_limits[] = {10, 8, 6, 5, 9};
  const int64 slice_strides[] = {1, 1, 1, 1, 1};
  TF_ASSERT_OK_AND_ASSIGN(auto literal,
                          LiteralUtil::CreateRandomLiteral<F32>(
                              ShapeUtil::MakeShape(F32, dimensions), 0.0, 1.0));
  HloInstruction* literal_instruction = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));
  Shape shape = ShapeUtil::MakeShape(F32, {6, 6, 3, 4, 4});
  builder.AddInstruction(HloInstruction::CreateSlice(
      shape, literal_instruction, slice_start, slice_limits, slice_strides));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Constant());
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), shape));
}

TEST_F(HloConstantFoldingTest, TransposeConstantFold) {
  HloComputation::Builder builder(TestName());
  const int64 dimensions[] = {11, 8, 7, 5, 9};
  TF_ASSERT_OK_AND_ASSIGN(auto literal,
                          LiteralUtil::CreateRandomLiteral<F32>(
                              ShapeUtil::MakeShape(F32, dimensions), 0.0, 1.0));
  auto literal_clone = literal->Literal::CloneToUnique();
  HloInstruction* literal_instruction = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));
  Shape shape = ShapeUtil::MakeShape(F32, {8, 7, 11, 9, 5});
  const int64 permutation[] = {1, 2, 0, 4, 3};
  builder.AddInstruction(
      HloInstruction::CreateTranspose(shape, literal_instruction, permutation));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module.get()));
  EXPECT_TRUE(result);

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Constant());
  EXPECT_TRUE(ShapeUtil::Compatible(root->shape(), shape));

  using NativeT = typename primitive_util::PrimitiveTypeToNative<F32>::type;
  bool matched = true;
  root->literal().EachCell<NativeT>(
      [&](tensorflow::gtl::ArraySlice<int64> indices, NativeT value) {
        std::vector<int64> rindexes = Permute(permutation, indices);
        matched = matched && (value == literal_clone->Get<NativeT>(rindexes));
      });
  EXPECT_TRUE(matched);
}

}  // namespace
}  // namespace xla
