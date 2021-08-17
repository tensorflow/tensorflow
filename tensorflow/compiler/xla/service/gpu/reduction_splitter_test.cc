/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/reduction_splitter.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;

class ReductionSplitterTest : public HloTestBase {};

TEST_F(ReductionSplitterTest, SplitReductionAtDimensionTwo) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test

  add_computation {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  ENTRY entry_computation {
    param_0 = f16[6,16,512,64]{3,2,1,0} parameter(0)
    transpose.1781 = f16[6,512,16,64]{3,1,2,0} transpose(param_0), dimensions={0,2,1,3}
    convert.6986 = f32[6,512,16,64]{3,1,2,0} convert(transpose.1781)
    bitcast.2136 = f32[6,16,512,64]{3,2,1,0} bitcast(convert.6986)
    constant_11111 = f32[] constant(0)
    ROOT reduce.982 = f32[16,64]{1,0} reduce(bitcast.2136, constant_11111), dimensions={0,2}, to_apply=add_computation
  }
  )")
                    .ValueOrDie();
  ASSERT_TRUE(ReductionSplitter().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root_reduction =
      module->entry_computation()->root_instruction();
  ASSERT_THAT(root_reduction, op::Reduce(op::Reduce(), op::Constant()));

  auto* pre_reduction = root_reduction->operand(0);
  EXPECT_THAT(pre_reduction->dimensions(), std::vector<int64_t>({2}));
  EXPECT_THAT(pre_reduction->shape(), ShapeUtil::MakeShape(F32, {6, 16, 64}));
  EXPECT_THAT(root_reduction->dimensions(), std::vector<int64_t>({0}));
  EXPECT_THAT(root_reduction->shape(), ShapeUtil::MakeShape(F32, {16, 64}));
}

TEST_F(ReductionSplitterTest, SplitReductionAtDimensionZero) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test

  add_computation {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  ENTRY entry_computation {
    param_0 = f32[1024,16,512,64,128]{4,3,2,1,0} parameter(0)
    constant_11111 = f32[] constant(0)
    ROOT reduce.982 = f32[16,64]{1,0} reduce(param_0, constant_11111), dimensions={2,0,4}, to_apply=add_computation
  }
  )")
                    .ValueOrDie();
  ASSERT_TRUE(ReductionSplitter().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root_reduction =
      module->entry_computation()->root_instruction();
  ASSERT_THAT(root_reduction, op::Reduce(op::Reduce(), op::Constant()));

  auto* pre_reduction = root_reduction->operand(0);
  EXPECT_THAT(pre_reduction->dimensions(), std::vector<int64_t>({0}));
  EXPECT_THAT(pre_reduction->shape(),
              ShapeUtil::MakeShape(F32, {16, 512, 64, 128}));
  EXPECT_THAT(root_reduction->dimensions(), std::vector<int64_t>({1, 3}));
  EXPECT_THAT(root_reduction->shape(), ShapeUtil::MakeShape(F32, {16, 64}));
}

TEST_F(ReductionSplitterTest, DontSplitReductionWithSmallDimensions) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test

  add_computation {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  ENTRY entry_computation {
    param_0 = f32[8,1024,8]{2,1,0} parameter(0)
    constant_11111 = f32[] constant(0)
    ROOT reduce.982 = f32[1024]{0} reduce(param_0, constant_11111), dimensions={2,0}, to_apply=add_computation
  }
  )")
                    .ValueOrDie();
  EXPECT_FALSE(ReductionSplitter().Run(module.get()).ValueOrDie());
}

TEST_F(ReductionSplitterTest, DontSplitReductionsWithContiguousDimensions) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test

  add_computation {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  ENTRY entry_computation {
    param_0 = f32[128,128,64,128]{3,2,1,0} parameter(0)
    constant_11111 = f32[] constant(0)
    // The dimenstions to keep (1 and 2) are contiguous.
    ROOT reduce.982 = f32[128,64]{1,0} reduce(param_0, constant_11111), dimensions={3,0}, to_apply=add_computation
  }
  )")
                    .ValueOrDie();
  EXPECT_FALSE(ReductionSplitter().Run(module.get()).ValueOrDie());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
