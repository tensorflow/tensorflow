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

#include "tensorflow/compiler/xla/service/while_loop_all_reduce_code_motion.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;
using ::testing::Ne;
using ::testing::NotNull;
using ::testing::Property;
using ::testing::SizeIs;

class WhileLoopAllReduceCodeMotionTest : public HloTestBase {};

TEST_F(WhileLoopAllReduceCodeMotionTest, AllReduceAccumulate) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %reduction {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(f32[] %x, f32[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %all-reduce = f32[1024, 1024] all-reduce(f32[1024, 1024] %gte.2), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %all-reduce, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[1024, 1024] parameter(1)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      ROOT %while = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopAllReduceCodeMotion{}.Run(module.get()));
  ASSERT_TRUE(simplified_loop);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  HloInstruction* transformed_while =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) -> bool {
                       return Value(instruction, op::While());
                     }));

  ASSERT_THAT(transformed_while, NotNull());
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::AllReduce())));
  HloInstruction* accumulation_buffer =
      transformed_while->mutable_operand(0)->mutable_operand(3);
  EXPECT_THAT(accumulation_buffer, op::Constant());
  HloAllReduceInstruction* moved_all_reduce = DynCast<HloAllReduceInstruction>(
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) {
                       return Value(instruction, op::AllReduce());
                     })));
  ASSERT_THAT(moved_all_reduce, NotNull());
  EXPECT_THAT(moved_all_reduce->operand(0), op::GetTupleElement());
  EXPECT_EQ(DynCast<HloGetTupleElementInstruction>(
                moved_all_reduce->mutable_operand(0))
                ->tuple_index(),
            3);
  EXPECT_THAT(moved_all_reduce->replica_groups(), SizeIs(1));
  EXPECT_TRUE(
      std::equal(moved_all_reduce->replica_groups()[0].replica_ids().begin(),
                 moved_all_reduce->replica_groups()[0].replica_ids().end(),
                 std::vector<int>{0, 1, 2, 3}.begin()));
  EXPECT_FALSE(moved_all_reduce->constrain_layout());
  EXPECT_TRUE(moved_all_reduce->use_global_device_ids());
  HloComputation* reduction_computation =
      module->GetComputationWithName("reduction");
  ASSERT_THAT(reduction_computation, NotNull());
  EXPECT_EQ(moved_all_reduce->called_computations()[0], reduction_computation);
}

TEST_F(WhileLoopAllReduceCodeMotionTest, ReduceScatterAccumulate) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_reduce_scatter

    %reduction {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(f32[] %x, f32[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[4096, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[4096, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[4096, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %reduce-scatter = f32[1024, 1024] reduce-scatter(f32[4096, 1024] %gte.2), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction, dimensions={0}
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %reduce-scatter, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[4096, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[4096, 1024] parameter(1)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[4096, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[4096, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      ROOT %while = (s32[], s32[], f32[4096, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopAllReduceCodeMotion{/*enable_reduce_scatter=*/true}.Run(
          module.get()));
  ASSERT_TRUE(simplified_loop);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  HloInstruction* transformed_while =
      *std::find_if(module->entry_computation()->instructions().begin(),
                    module->entry_computation()->instructions().end(),
                    HloPredicateIsOp<HloOpcode::kWhile>);

  ASSERT_THAT(transformed_while, NotNull());
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::ReduceScatter())));
  HloInstruction* accumulation_buffer =
      transformed_while->mutable_operand(0)->mutable_operand(3);
  EXPECT_THAT(accumulation_buffer, op::Constant());
  // Verify that the accumulation buffer's shape changed.
  EXPECT_THAT(accumulation_buffer, op::Shape("f32[4096, 1024]"));
  auto* moved_reduce_scatter = DynCast<HloReduceScatterInstruction>(
      *std::find_if(module->entry_computation()->instructions().begin(),
                    module->entry_computation()->instructions().end(),
                    HloPredicateIsOp<HloOpcode::kReduceScatter>));
  ASSERT_THAT(moved_reduce_scatter, NotNull());
  EXPECT_THAT(moved_reduce_scatter->operand(0), op::GetTupleElement());
  EXPECT_EQ(DynCast<HloGetTupleElementInstruction>(
                moved_reduce_scatter->mutable_operand(0))
                ->tuple_index(),
            3);
  EXPECT_THAT(moved_reduce_scatter, op::ReplicaGroups({{0, 1, 2, 3}}));
  EXPECT_FALSE(moved_reduce_scatter->constrain_layout());
  EXPECT_TRUE(moved_reduce_scatter->use_global_device_ids());
  HloComputation* reduction_computation =
      module->GetComputationWithName("reduction");
  ASSERT_THAT(reduction_computation, NotNull());
  EXPECT_EQ(moved_reduce_scatter->to_apply(), reduction_computation);
}

TEST_F(WhileLoopAllReduceCodeMotionTest,
       ReduceScatterAccumulateDisabledByDefault) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_reduce_scatter

    %reduction {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(f32[] %x, f32[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[4096, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[4096, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[4096, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %reduce-scatter = f32[1024, 1024] reduce-scatter(f32[4096, 1024] %gte.2), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction, dimensions={0}
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %reduce-scatter, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[4096, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[4096, 1024] parameter(1)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[4096, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[4096, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      ROOT %while = (s32[], s32[], f32[4096, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopAllReduceCodeMotion{}.Run(module.get()));
  EXPECT_FALSE(simplified_loop);
}

TEST_F(WhileLoopAllReduceCodeMotionTest, AllReduceSliceAccumulate) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %reduction {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(f32[] %x, f32[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[3, 1024, 1024], f32[1024, 1024], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[3, 1024, 1024], f32[1024, 1024], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[3, 1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %gte.4 = f32[1024, 1024] get-tuple-element(%param), index=4
      %gte.5 = f32[1024, 1024] get-tuple-element(%param), index=5
      %all-reduce = f32[3, 1024, 1024] all-reduce(f32[3, 1024, 1024] %gte.2), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction
      %slice.0 = f32[1, 1024, 1024] slice(f32[3, 1024, 1024] %all-reduce), slice={[0:1], [0:1024], [0:1024]}
      %reshape.0 = f32[1024, 1024] reshape(f32[1, 1024, 1024] %slice.0)
      %slice.1 = f32[1, 1024, 1024] slice(f32[3, 1024, 1024] %all-reduce), slice={[1:2], [0:1024], [0:1024]}
      %reshape.1 = f32[1024, 1024] reshape(f32[1, 1024, 1024] %slice.1)
      %slice.2 = f32[1, 1024, 1024] slice(f32[3, 1024, 1024] %all-reduce), slice={[2:3], [0:1024], [0:1024]}
      %reshape.2 = f32[1024, 1024] reshape(f32[1, 1024, 1024] %slice.2)
      %accumulation.0 = f32[1024, 1024] add(f32[1024, 1024] %reshape.0, f32[1024, 1024] %gte.3)
      %accumulation.1 = f32[1024, 1024] add(f32[1024, 1024] %reshape.1, f32[1024, 1024] %gte.4)
      %accumulation.2 = f32[1024, 1024] add(f32[1024, 1024] %reshape.2, f32[1024, 1024] %gte.5)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[3, 1024, 1024], f32[1024, 1024], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation.0, %accumulation.1, %accumulation.2)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[3, 1024, 1024] parameter(1)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer.0 = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %accumulation_buffer.1 = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %accumulation_buffer.2 = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[3, 1024, 1024], f32[1024, 1024], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[3, 1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer.0, f32[1024, 1024] %accumulation_buffer.1, f32[1024, 1024] %accumulation_buffer.2)
      ROOT %while = (s32[], s32[], f32[3, 1024, 1024], f32[1024, 1024], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopAllReduceCodeMotion{}.Run(module.get()));
  ASSERT_TRUE(simplified_loop);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  HloInstruction* transformed_while =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) -> bool {
                       return Value(instruction, op::While());
                     }));

  ASSERT_THAT(transformed_while, NotNull());
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::AllReduce())));
  std::vector<HloInstruction*> hoisted_all_reduces;
  absl::c_copy_if(module->entry_computation()->instructions(),
                  std::back_inserter(hoisted_all_reduces),
                  [](HloInstruction* instruction) {
                    return Value(instruction, op::AllReduce());
                  });
  EXPECT_THAT(hoisted_all_reduces, SizeIs(3));
  ASSERT_THAT(
      hoisted_all_reduces,
      Each(Pointee(Property(&HloInstruction::channel_id, Ne(std::nullopt)))));
  // Check if added all-reduces have distinct channel IDs.
  absl::flat_hash_set<int> unique_channel_ids = {
      hoisted_all_reduces[0]->channel_id().value(),
      hoisted_all_reduces[1]->channel_id().value(),
      hoisted_all_reduces[2]->channel_id().value()};
  EXPECT_THAT(unique_channel_ids, SizeIs(3));
}

TEST_F(WhileLoopAllReduceCodeMotionTest, AllReduceAccumulateUse) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %reduction {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(f32[] %x, f32[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %all-reduce = f32[1024, 1024] all-reduce(f32[1024, 1024] %gte.2), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %all-reduce, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[1024, 1024] parameter(1)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
      %gte_while = f32[1024, 1024] get-tuple-element((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %while), index=3
      ROOT %multiply = f32[1024, 1024] multiply(f32[1024, 1024] %gte_while, f32[1024, 1024] %param.1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopAllReduceCodeMotion{}.Run(module.get()));
  ASSERT_TRUE(simplified_loop);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  HloInstruction* transformed_while =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) -> bool {
                       return Value(instruction, op::While());
                     }));

  ASSERT_THAT(transformed_while, NotNull());
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::AllReduce())));
  HloInstruction* new_root = module->entry_computation()->root_instruction();
  ASSERT_THAT(new_root, op::Multiply());
  ASSERT_THAT(new_root->operand(0), op::GetTupleElement());
  ASSERT_THAT(new_root->operand(0)->operand(0), op::Tuple());
  EXPECT_THAT(new_root->operand(0)->operand(0)->operand(3), op::Add());
}

TEST_F(WhileLoopAllReduceCodeMotionTest, RepeatedlyAccumulatedAllReduce) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %reduction {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(f32[] %x, f32[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %all-reduce = f32[1024, 1024] all-reduce(f32[1024, 1024] %gte.2), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %all-reduce, f32[1024, 1024] %gte.3)
      %add.0 = f32[1024, 1024] add(f32[1024, 1024] %all-reduce, f32[1024, 1024] %accumulation)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %add.0)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[1024, 1024] parameter(1)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      ROOT %while = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopAllReduceCodeMotion{}.Run(module.get()));
  EXPECT_FALSE(simplified_loop);
}

TEST_F(WhileLoopAllReduceCodeMotionTest, TypeCastAllReduceAccumulate) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %reduction {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add = bf16[] add(bf16[] %x, bf16[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %convert.0 = bf16[1024, 1024] convert(f32[1024, 1024] %gte.2)
      %all-reduce = bf16[1024, 1024] all-reduce(bf16[1024, 1024] %convert.0), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction
      %convert.1 = f32[1024, 1024] convert(bf16[1024, 1024] %all-reduce)
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %convert.1, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[1024, 1024] parameter(1)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      ROOT %while = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopAllReduceCodeMotion{}.Run(module.get()));
  ASSERT_TRUE(simplified_loop);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  HloInstruction* transformed_while =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) -> bool {
                       return Value(instruction, op::While());
                     }));

  ASSERT_THAT(transformed_while, NotNull());
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::AllReduce())));
  HloInstruction* accumulation_buffer =
      transformed_while->mutable_operand(0)->mutable_operand(3);
  EXPECT_THAT(accumulation_buffer, op::Constant());
  HloAllReduceInstruction* moved_all_reduce = DynCast<HloAllReduceInstruction>(
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) {
                       return Value(instruction, op::AllReduce());
                     })));
  EXPECT_TRUE(ShapeUtil::Equal(moved_all_reduce->shape(),
                               ShapeUtil::MakeShape(BF16, {1024, 1024})));

  HloInstruction* add_delta_to_old_buffer =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) -> bool {
                       return Value(instruction, op::Add());
                     }));
  ASSERT_THAT(add_delta_to_old_buffer, NotNull());
  EXPECT_TRUE(ShapeUtil::Equal(add_delta_to_old_buffer->shape(),
                               ShapeUtil::MakeShape(F32, {1024, 1024})));
  EXPECT_TRUE(ShapeUtil::Equal(add_delta_to_old_buffer->operand(0)->shape(),
                               ShapeUtil::MakeShape(F32, {1024, 1024})));
  EXPECT_TRUE(ShapeUtil::Equal(add_delta_to_old_buffer->operand(1)->shape(),
                               ShapeUtil::MakeShape(F32, {1024, 1024})));
}

TEST_F(WhileLoopAllReduceCodeMotionTest, SelectAllReduceAccumulate) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %reduction {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add = bf16[] add(bf16[] %x, bf16[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[1024,1024], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024,1024], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024,1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024,1024] get-tuple-element(%param), index=3
      %all-reduce = f32[1024,1024] all-reduce(%gte.2), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction
      %const.0 = f32[] constant(0)
      %zeros = f32[1024,1024] broadcast(%const.0), dimensions={}
      %predicates = pred[1024,1024] custom-call(), custom_call_target="something"
      %select = f32[1024,1024] select(%predicates, %zeros, %all-reduce)
      %accumulation = f32[1024,1024] add(%select, %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024,1024], f32[1024,1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[1024,1024] parameter(1)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024,1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 1024], f32[1024,1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      ROOT %while = (s32[], s32[], f32[1024, 1024], f32[1024,1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopAllReduceCodeMotion{}.Run(module.get()));
  ASSERT_TRUE(simplified_loop);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  HloInstruction* transformed_while =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) -> bool {
                       return Value(instruction, op::While());
                     }));

  ASSERT_THAT(transformed_while, NotNull());
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::AllReduce())));
  HloInstruction* accumulation_buffer =
      transformed_while->mutable_operand(0)->mutable_operand(3);
  EXPECT_THAT(accumulation_buffer, op::Constant());
  HloAllReduceInstruction* moved_all_reduce = DynCast<HloAllReduceInstruction>(
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) {
                       return Value(instruction, op::AllReduce());
                     })));
  EXPECT_TRUE(ShapeUtil::Equal(moved_all_reduce->shape(),
                               ShapeUtil::MakeShape(F32, {1024, 1024})));

  HloInstruction* add_delta_to_old_buffer =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) -> bool {
                       return Value(instruction, op::Add());
                     }));
  ASSERT_THAT(add_delta_to_old_buffer, NotNull());
  EXPECT_TRUE(ShapeUtil::Equal(add_delta_to_old_buffer->shape(),
                               ShapeUtil::MakeShape(F32, {1024, 1024})));
  EXPECT_TRUE(ShapeUtil::Equal(add_delta_to_old_buffer->operand(0)->shape(),
                               ShapeUtil::MakeShape(F32, {1024, 1024})));
  EXPECT_TRUE(ShapeUtil::Equal(add_delta_to_old_buffer->operand(1)->shape(),
                               ShapeUtil::MakeShape(F32, {1024, 1024})));
}

TEST_F(WhileLoopAllReduceCodeMotionTest, SelectReduceScatterAccumulate) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_reduce_scatter

    %reduction {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add = bf16[] add(bf16[] %x, bf16[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[1024,4096], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024,4096], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024,4096] get-tuple-element(%param), index=2
      %gte.3 = f32[1024,1024] get-tuple-element(%param), index=3
      %reduce-scatter = f32[1024,1024] reduce-scatter(%gte.2), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction, dimensions={1}
      %const.0 = f32[] constant(0)
      %zeros = f32[1024,1024] broadcast(%const.0), dimensions={}
      // effectively scalar predicate
      %scalarp = pred[] custom-call(), custom_call_target="something"
      %predicates = pred[1024,1024] broadcast(%scalarp), dimensions={}
      %select = f32[1024,1024] select(%predicates, %zeros, %reduce-scatter)
      %accumulation = f32[1024,1024] add(%select, %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024,4096], f32[1024,1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[1024,4096] parameter(1)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024,1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 4096], f32[1024,1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[1024, 4096] %param.1, f32[1024, 1024] %accumulation_buffer)
      ROOT %while = (s32[], s32[], f32[1024, 4096], f32[1024,1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopAllReduceCodeMotion{/*enable_reduce_scatter=*/true}.Run(
          module.get()));
  ASSERT_TRUE(simplified_loop);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  HloInstruction* transformed_while =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     HloPredicateIsOp<HloOpcode::kWhile>));

  ASSERT_THAT(transformed_while, NotNull());
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::ReduceScatter())));
  HloInstruction* accumulation_buffer =
      transformed_while->mutable_operand(0)->mutable_operand(3);
  EXPECT_THAT(accumulation_buffer, op::Constant());
  EXPECT_THAT(accumulation_buffer, op::Shape("f32[1024,4096]"));
  auto* moved_reduce_scatter = DynCast<HloReduceScatterInstruction>(
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     HloPredicateIsOp<HloOpcode::kReduceScatter>)));
  EXPECT_THAT(moved_reduce_scatter, op::Shape("f32[1024,1024]"));
  HloInstruction* add_delta_to_old_buffer =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     HloPredicateIsOp<HloOpcode::kAdd>));
  ASSERT_THAT(add_delta_to_old_buffer, NotNull());
  EXPECT_THAT(add_delta_to_old_buffer, op::Shape("f32[1024,1024]"));
}

TEST_F(WhileLoopAllReduceCodeMotionTest,
       SelectReduceScatterAccumulateNotScalarPredicate) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_reduce_scatter

    %reduction {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add = bf16[] add(bf16[] %x, bf16[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[1024,4096], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024,4096], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024,4096] get-tuple-element(%param), index=2
      %gte.3 = f32[1024,1024] get-tuple-element(%param), index=3
      %reduce-scatter = f32[1024,1024] reduce-scatter(%gte.2), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction, dimensions={1}
      %const.0 = f32[] constant(0)
      %zeros = f32[1024,1024] broadcast(%const.0), dimensions={}
      %predicates = pred[1024,1024] custom-call(), custom_call_target="something"
      %select = f32[1024,1024] select(%predicates, %zeros, %reduce-scatter)
      %accumulation = f32[1024,1024] add(%select, %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024,4096], f32[1024,1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[1024,4096] parameter(1)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024,1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 4096], f32[1024,1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[1024, 4096] %param.1, f32[1024, 1024] %accumulation_buffer)
      ROOT %while = (s32[], s32[], f32[1024, 4096], f32[1024,1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopAllReduceCodeMotion{/*enable_reduce_scatter=*/true}.Run(
          module.get()));
  EXPECT_FALSE(simplified_loop);
}

TEST_F(WhileLoopAllReduceCodeMotionTest, MultipleLoopCalls) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %reduction {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add = bf16[] add(bf16[] %x, bf16[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %convert.0 = bf16[1024, 1024] convert(f32[1024, 1024] %gte.2)
      %all-reduce = bf16[1024, 1024] all-reduce(bf16[1024, 1024] %convert.0), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction
      %convert.1 = f32[1024, 1024] convert(bf16[1024, 1024] %all-reduce)
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %convert.1, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[1024, 1024] parameter(1)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init.0 = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while.0 = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init.0), condition=%while_condition, body=%while_body
      %gte.3 = f32[1024, 1024] get-tuple-element(%while.0), index=3
      %while_init.1 = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[1024, 1024] %param.1, f32[1024, 1024] %gte.3)
      %while.1 = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init.0), condition=%while_condition, body=%while_body
      ROOT %gte.4 = f32[1024, 1024] get-tuple-element((s32[], s32[], f32[1024, 1024], f32[1024, 1024])%while.1), index=3
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopAllReduceCodeMotion{}.Run(module.get()));
  ASSERT_TRUE(simplified_loop);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  EXPECT_EQ(absl::c_count_if(module->entry_computation()->instructions(),
                             Matches(op::While())),
            2);
  EXPECT_EQ(absl::c_count_if(module->entry_computation()->instructions(),
                             Matches(op::AllReduce())),
            2);
  HloInstruction* transformed_while =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) -> bool {
                       return Value(instruction, op::While());
                     }));

  ASSERT_THAT(transformed_while, NotNull());
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::AllReduce())));
}

TEST_F(WhileLoopAllReduceCodeMotionTest, MultipleAllReduceAccumulate) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %reduction.0 {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(f32[] %x, f32[] %y)
    }

    %reduction.1 {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add = bf16[] add(bf16[] %x, bf16[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024], bf16[1024, 1024], bf16[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024], bf16[1024, 1024], bf16[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %gte.4 = bf16[1024, 1024] get-tuple-element(%param), index=4
      %gte.5 = bf16[1024, 1024] get-tuple-element(%param), index=5
      %all-reduce.0 = f32[1024, 1024] all-reduce(f32[1024, 1024] %gte.2), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction.0
      %accumulation.0 = f32[1024, 1024] add(f32[1024, 1024] %all-reduce.0, f32[1024, 1024] %gte.3)
      %all-reduce.1 = bf16[1024, 1024] all-reduce(bf16[1024, 1024] %gte.4), channel_id=2, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction.1
      %accumulation.1 = bf16[1024, 1024] add(bf16[1024, 1024] %all-reduce.1, bf16[1024, 1024] %gte.5)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024], bf16[1024, 1024], bf16[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation.0, %gte.4, %accumulation.1)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[1024, 1024] parameter(1)
      %param.2 = bf16[1024, 1024] parameter(2)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer.0 = f32[1024, 1024] constant({...})
      %accumulation_buffer.1 = bf16[1024, 1024] constant({...})
      %while_init = (s32[], s32[], f32[1024, 1024], f32[1024, 1024], bf16[1024, 1024], bf16[1024, 1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer.0, bf16[1024, 1024] %param.2, bf16[1024, 1024] %accumulation_buffer.1)
      ROOT %while = (s32[], s32[], f32[1024, 1024], f32[1024, 1024], bf16[1024, 1024], bf16[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopAllReduceCodeMotion{}.Run(module.get()));
  ASSERT_TRUE(simplified_loop);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  HloInstruction* transformed_while =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) -> bool {
                       return Value(instruction, op::While());
                     }));

  ASSERT_THAT(transformed_while, NotNull());
  // Both all-reduces should have been sinked.
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::AllReduce())));
  HloInstruction* accumulation_buffer =
      transformed_while->mutable_operand(0)->mutable_operand(3);
  EXPECT_THAT(accumulation_buffer, op::Constant());
  EXPECT_EQ(absl::c_count_if(module->entry_computation()->instructions(),
                             Matches(op::AllReduce())),
            2);
}

TEST_F(WhileLoopAllReduceCodeMotionTest, MultipleReduceScatterAccumulate) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_reduce_scatter

    %reduction.0 {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(f32[] %x, f32[] %y)
    }

    %reduction.1 {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add = bf16[] add(bf16[] %x, bf16[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[4096, 1024], f32[1024, 1024], bf16[4096, 1024], bf16[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[4096, 1024], f32[1024, 1024], bf16[4096, 1024], bf16[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[4096, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %gte.4 = bf16[4096, 1024] get-tuple-element(%param), index=4
      %gte.5 = bf16[1024, 1024] get-tuple-element(%param), index=5
      %reduce-scatter.0 = f32[1024, 1024] reduce-scatter(f32[4096, 1024] %gte.2), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction.0, dimensions={0}
      %accumulation.0 = f32[1024, 1024] add(f32[1024, 1024] %reduce-scatter.0, f32[1024, 1024] %gte.3)
      %reduce-scatter.1 = bf16[1024, 1024] reduce-scatter(bf16[4096, 1024] %gte.4), channel_id=2, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction.1, dimensions={0}
      %accumulation.1 = bf16[1024, 1024] add(bf16[1024, 1024] %reduce-scatter.1, bf16[1024, 1024] %gte.5)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[4096, 1024], f32[1024, 1024], bf16[4096, 1024], bf16[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation.0, %gte.4, %accumulation.1)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[4096, 1024] parameter(1)
      %param.2 = bf16[4096, 1024] parameter(2)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer.0 = f32[1024, 1024] constant({...})
      %accumulation_buffer.1 = bf16[1024, 1024] constant({...})
      %while_init = (s32[], s32[], f32[4096, 1024], f32[1024, 1024], bf16[4096, 1024], bf16[1024, 1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[4096, 1024] %param.1, f32[1024, 1024] %accumulation_buffer.0, bf16[4096, 1024] %param.2, bf16[1024, 1024] %accumulation_buffer.1)
      ROOT %while = (s32[], s32[], f32[4096, 1024], f32[1024, 1024], bf16[4096, 1024], bf16[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopAllReduceCodeMotion{/*enable_reduce_scatter=*/true}.Run(
          module.get()));
  ASSERT_TRUE(simplified_loop);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  HloInstruction* transformed_while =
      *std::find_if(module->entry_computation()->instructions().begin(),
                    module->entry_computation()->instructions().end(),
                    HloPredicateIsOp<HloOpcode::kWhile>);

  ASSERT_THAT(transformed_while, NotNull());
  // Both reduce-scatters should have been sinked.
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::ReduceScatter())));

  // Verify both accumulation buffers' shape has changed.
  constexpr std::array<std::pair<int64_t, absl::string_view>, 2> accum_buffers =
      {{
          {3, "f32[4096, 1024]"},
          {5, "bf16[4096, 1024]"},
      }};

  for (auto [index, shape] : accum_buffers) {
    HloInstruction* accumulation_buffer =
        transformed_while->mutable_operand(0)->mutable_operand(index);
    EXPECT_THAT(accumulation_buffer, op::Constant());
    EXPECT_THAT(accumulation_buffer, op::Shape(shape));
  }
  EXPECT_EQ(absl::c_count_if(module->entry_computation()->instructions(),
                             Matches(op::ReduceScatter())),
            2);
}

TEST_F(WhileLoopAllReduceCodeMotionTest, MixMovableAllReduceWithNotMovable) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %reduction.0 {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(f32[] %x, f32[] %y)
    }

    %reduction.1 {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add = bf16[] add(bf16[] %x, bf16[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024], bf16[1024, 1024], bf16[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024], bf16[1024, 1024], bf16[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %gte.4 = bf16[1024, 1024] get-tuple-element(%param), index=4
      %gte.5 = bf16[1024, 1024] get-tuple-element(%param), index=5
      %all-reduce.0 = f32[1024, 1024] all-reduce(f32[1024, 1024] %gte.2), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction.0
      %accumulation.0 = f32[1024, 1024] add(f32[1024, 1024] %all-reduce.0, f32[1024, 1024] %gte.3)
      %all-reduce.1 = bf16[1024, 1024] all-reduce(bf16[1024, 1024] %gte.4), channel_id=2, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%reduction.1
      %accumulation.1 = bf16[1024, 1024] add(bf16[1024, 1024] %all-reduce.1, bf16[1024, 1024] %gte.5)
      %add.0 = bf16[1024, 1024] add(bf16[1024, 1024] %accumulation.1, bf16[1024, 1024] %gte.4)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024], bf16[1024, 1024], bf16[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation.0, %gte.4, %add.0)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = s32[] parameter(0)
      %param.1 = f32[1024, 1024] parameter(1)
      %param.2 = bf16[1024, 1024] parameter(2)
      %constant.0 = s32[] constant(1)
      %accumulation_buffer.0 = f32[1024, 1024] constant({...})
      %accumulation_buffer.1 = bf16[1024, 1024] constant({...})
      %while_init = (s32[], s32[], f32[1024, 1024], f32[1024, 1024], bf16[1024, 1024], bf16[1024, 1024]) tuple(s32[] %constant.0, s32[] %param.0, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer.0, bf16[1024, 1024] %param.2, bf16[1024, 1024] %accumulation_buffer.1)
      ROOT %while = (s32[], s32[], f32[1024, 1024], f32[1024, 1024], bf16[1024, 1024], bf16[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopAllReduceCodeMotion{}.Run(module.get()));
  ASSERT_TRUE(simplified_loop);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  HloInstruction* transformed_while =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) -> bool {
                       return Value(instruction, op::While());
                     }));

  ASSERT_THAT(transformed_while, NotNull());
  // One all-reduce is movable and the other is not movable.
  EXPECT_EQ(absl::c_count_if(transformed_while->while_body()->instructions(),
                             Matches(op::AllReduce())),
            1);
  HloInstruction* accumulation_buffer =
      transformed_while->mutable_operand(0)->mutable_operand(3);
  EXPECT_THAT(accumulation_buffer, op::Constant());
  EXPECT_EQ(absl::c_count_if(module->entry_computation()->instructions(),
                             Matches(op::AllReduce())),
            1);
}

TEST_F(WhileLoopAllReduceCodeMotionTest,
       DynamicSliceAllReduceDynamicUpdateSliceAccumulate) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %reduction {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add = bf16[] add(bf16[] %x, bf16[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[2, 1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.1, %gte.0), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[2, 1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[2, 1024, 1024] get-tuple-element(%param), index=3
      %offset-table = s32[8] constant({0, 0, 0, 0, 1, 1, 1, 1})
      %partition-id = u32[] partition-id()
      %offset-array = s32[1] dynamic-slice(%offset-table, %partition-id), dynamic_slice_sizes={1}
      %offset = s32[] reshape(%offset-array)
      %convert.0 = bf16[1024, 1024] convert(f32[1024, 1024] %gte.2)
      %all-reduce = bf16[1024, 1024] all-reduce(bf16[1024, 1024] %convert.0), channel_id=1, replica_groups={{0,1,2,3},{4,5,6,7}}, use_global_device_ids=true, to_apply=%reduction
      %convert.1 = f32[1024, 1024] convert(bf16[1024, 1024] %all-reduce)
      %reshape = f32[1,1024, 1024] reshape(f32[1024, 1024] %convert.1)
      %constant.2 = s32[] constant(0)
      %dynamic-slice = f32[1,1024,1024] dynamic-slice(f32[2, 1024, 1024] %gte.3, s32[] %offset, s32[] %constant.2, s32[] %constant.2), dynamic_slice_sizes={1, 1024, 1024}
      %accumulation = f32[1,1024,1024] add(f32[1, 1024, 1024] %reshape, f32[1, 1024, 1024] %dynamic-slice)
      %dynamic-update-slice = f32[2,1024,1024] dynamic-update-slice(f32[2, 1024, 1024] %gte.3, f32[1, 1024, 1024]  %accumulation, s32[] %offset, s32[] %constant.2, s32[] %constant.2)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.1, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[2, 1024, 1024]) tuple(s32[] %gte.0, s32[] %increment_iteration, f32[1024, 1024] %gte.2, f32[2, 1024, 1024] %dynamic-update-slice)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = f32[1024, 1024] parameter(0)
      %constant.0 = s32[] constant(8)
      %constant.1 = s32[] constant(1)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[2, 1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 1024], f32[2, 1024, 1024]) tuple(s32[] %constant.0, s32[] %constant.1, f32[1024, 1024] %param.0, f32[2, 1024, 1024] %accumulation_buffer)
      ROOT %while = (s32[], s32[], f32[1024, 1024], f32[2, 1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloModule, /*replica_count=*/1,
                                   /*num_partitions=*/8));
  module->config().set_use_spmd_partitioning(true);
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopAllReduceCodeMotion{}.Run(module.get()));
  ASSERT_TRUE(simplified_loop);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  HloInstruction* transformed_while =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) -> bool {
                       return Value(instruction, op::While());
                     }));

  ASSERT_THAT(transformed_while, NotNull());
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::AllReduce())));
  HloInstruction* accumulation_buffer =
      transformed_while->mutable_operand(0)->mutable_operand(3);
  EXPECT_THAT(accumulation_buffer, op::Constant());
  HloAllReduceInstruction* moved_all_reduce = DynCast<HloAllReduceInstruction>(
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) {
                       return Value(instruction, op::AllReduce());
                     })));
  EXPECT_TRUE(ShapeUtil::Equal(moved_all_reduce->shape(),
                               ShapeUtil::MakeShape(BF16, {2, 1024, 1024})));

  HloInstruction* add_delta_to_old_buffer =
      *(std::find_if(module->entry_computation()->instructions().begin(),
                     module->entry_computation()->instructions().end(),
                     [](HloInstruction* instruction) -> bool {
                       return Value(instruction, op::Add());
                     }));
  ASSERT_THAT(add_delta_to_old_buffer, NotNull());
  EXPECT_TRUE(ShapeUtil::Equal(add_delta_to_old_buffer->shape(),
                               ShapeUtil::MakeShape(F32, {2, 1024, 1024})));
  EXPECT_TRUE(ShapeUtil::Equal(add_delta_to_old_buffer->operand(0)->shape(),
                               ShapeUtil::MakeShape(F32, {2, 1024, 1024})));
  EXPECT_TRUE(ShapeUtil::Equal(add_delta_to_old_buffer->operand(1)->shape(),
                               ShapeUtil::MakeShape(F32, {2, 1024, 1024})));
}

// This test is almost the same as the one above but we change the all-reduce
// replica groups to make the dynamic-slice indices not replicated within each
// replica group
TEST_F(WhileLoopAllReduceCodeMotionTest,
       DynamicSliceAllReduceDynamicUpdateSliceAccumulateNotMoved) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %reduction {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add = bf16[] add(bf16[] %x, bf16[] %y)
    }

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[2, 1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.1, %gte.0), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[2, 1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[2, 1024, 1024] get-tuple-element(%param), index=3
      %offset-table = s32[8] constant({0, 0, 0, 0, 1, 1, 1, 1})
      %partition-id = u32[] partition-id()
      %offset-array = s32[1] dynamic-slice(%offset-table, %partition-id), dynamic_slice_sizes={1}
      %offset = s32[] reshape(%offset-array)
      %convert.0 = bf16[1024, 1024] convert(f32[1024, 1024] %gte.2)
      %all-reduce = bf16[1024, 1024] all-reduce(bf16[1024, 1024] %convert.0), channel_id=1, replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, to_apply=%reduction
      %convert.1 = f32[1024, 1024] convert(bf16[1024, 1024] %all-reduce)
      %reshape = f32[1,1024, 1024] reshape(f32[1024, 1024] %convert.1)
      %constant.2 = s32[] constant(0)
      %dynamic-slice = f32[1,1024,1024] dynamic-slice(f32[2, 1024, 1024] %gte.3, s32[] %offset, s32[] %constant.2, s32[] %constant.2), dynamic_slice_sizes={1, 1024, 1024}
      %accumulation = f32[1,1024,1024] add(f32[1, 1024, 1024] %reshape, f32[1, 1024, 1024] %dynamic-slice)
      %dynamic-update-slice = f32[2,1024,1024] dynamic-update-slice(f32[2, 1024, 1024] %gte.3, f32[1, 1024, 1024]  %accumulation, s32[] %offset, s32[] %constant.2, s32[] %constant.2)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.1, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[2, 1024, 1024]) tuple(s32[] %gte.0, s32[] %increment_iteration, f32[1024, 1024] %gte.2, f32[2, 1024, 1024] %dynamic-update-slice)
    }

    ENTRY accumulated_all_reduce {
      %param.0 = f32[1024, 1024] parameter(0)
      %constant.0 = s32[] constant(8)
      %constant.1 = s32[] constant(1)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[2, 1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 1024], f32[2, 1024, 1024]) tuple(s32[] %constant.0, s32[] %constant.1, f32[1024, 1024] %param.0, f32[2, 1024, 1024] %accumulation_buffer)
      ROOT %while = (s32[], s32[], f32[1024, 1024], f32[2, 1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloModule, /*replica_count=*/1,
                                   /*num_partitions=*/8));
  module->config().set_use_spmd_partitioning(true);
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopAllReduceCodeMotion{}.Run(module.get()));
  EXPECT_FALSE(simplified_loop);
}

}  // namespace
}  // namespace xla
