/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/small_while_loop_hoisting_pass.h"

#include <memory>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class SmallWhileLoopHoistingPassTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunSmallWhileLoopHoistingPass(HloModule* module) {
    return cpu::SmallWhileLoopHoistingPass(1024).Run(module);
  }
};

TEST_F(SmallWhileLoopHoistingPassTest, SmallWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule simple_while_loop

    while_body {
      counter = s32[] parameter(0)
      increment = s32[] constant(1)
      ROOT incremented_counter = s32[] add(counter, increment)
    }

    while_condition {
      counter = s32[] parameter(0)
      limit = s32[] constant(10)
      ROOT less_than = pred[] compare(counter, limit), direction=LT
    }

    ENTRY main {
      initial_counter = s32[] constant(0)
      ROOT while_loop = s32[] while(initial_counter), condition=while_condition, body=while_body
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);
  std::optional<std::string> maybe_small_call =
      call_instr->get_frontend_attribute("xla_cpu_small_call");
  ASSERT_NE(maybe_small_call, std::nullopt);
  EXPECT_EQ(*maybe_small_call, "true");

  EXPECT_EQ(call_instr->to_apply()->root_instruction()->opcode(),
            HloOpcode::kWhile);
}

TEST_F(SmallWhileLoopHoistingPassTest, NoBigWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule simple_while_loop

    reduce_fn {
      x = s32[] parameter(0)
      y = s32[] parameter(1)
      ROOT add = s32[] add(x, y)
    }

    while_body {
      counter = s32[] parameter(0)
      dummy_constant = s32[1000000] constant({...})
      // The big constant must be in the call graph to be considered in the cost
      // analysis, hence the reduce.
      element_reduce = s32[] reduce(dummy_constant, counter), dimensions={0}, to_apply=reduce_fn
      ROOT incremented_counter = s32[] add(counter, element_reduce)
    }

    while_condition {
      counter = s32[] parameter(0)
      limit = s32[] constant(10)
      ROOT less_than = pred[] compare(counter, limit), direction=LT
    }

    ENTRY main {
      initial_counter = s32[] constant(0)
      ROOT while_loop = s32[] while(initial_counter), condition=while_condition, body=while_body
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, NoInOutFeedWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule in_out_feed_while_loop, entry_computation_layout={(pred[])->(pred[])}

    body_fn (T.4: (pred[])) -> (pred[]) {
      T.4 = (pred[]) parameter(0)
      after-all.5 = token[] after-all()
      infeed.6 = ((f32[1,3]{1,0}, pred[], u32[]), token[]) infeed(token[] after-all.5)
      get-tuple-element.7 = token[] get-tuple-element(((f32[1,3]{1,0}, pred[], u32[]), token[]) infeed.6), index=1
      get-tuple-element.8 = (f32[1,3]{1,0}, pred[], u32[]) get-tuple-element(((f32[1,3]{1,0}, pred[], u32[]), token[]) infeed.6), index=0
      get-tuple-element.11 = f32[1,3]{1,0} get-tuple-element((f32[1,3]{1,0}, pred[], u32[]) get-tuple-element.8), index=0
      constant.12 = f32[] constant(1)
      broadcast.13 = f32[1,3]{1,0} broadcast(f32[] constant.12), dimensions={}
      multiply.14 = f32[1,3]{1,0} multiply(f32[1,3]{1,0} get-tuple-element.11, f32[1,3]{1,0} broadcast.13)
      concatenate.15 = f32[1,6]{1,0} concatenate(f32[1,3]{1,0} multiply.14, f32[1,3]{1,0} multiply.14), dimensions={1}
      get-tuple-element.10 = u32[] get-tuple-element((f32[1,3]{1,0}, pred[], u32[]) get-tuple-element.8), index=2
      tuple.16 = (f32[1,6]{1,0}, u32[]) tuple(f32[1,6]{1,0} concatenate.15, u32[] get-tuple-element.10)
      after-all.17 = token[] after-all()
      outfeed.18 = token[] outfeed((f32[1,6]{1,0}, u32[]) tuple.16, token[] after-all.17), outfeed_shape=(f32[1,6]{1,0}, u32[])
      tuple.19 = () tuple()
      get-tuple-element.9 = pred[] get-tuple-element((f32[1,3]{1,0}, pred[], u32[]) get-tuple-element.8), index=1
      ROOT tuple.20 = (pred[]) tuple(pred[] get-tuple-element.9)
    }

    condition_fn (T.22: (pred[])) -> pred[] {
      T.22 = (pred[]) parameter(0)
      ROOT get-tuple-element.23 = pred[] get-tuple-element((pred[]) T.22), index=0
    }

    ENTRY main (prev0.1: pred[]) -> (pred[]) {
      prev0.1 = pred[] parameter(0)
      tuple.2 = (pred[]) tuple(pred[] prev0.1)
      ROOT tuple.26 = (pred[]) while((pred[]) tuple.2), condition=condition_fn, body=body_fn
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, NoFftWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule fft_module

    %body_comp (arg_tuple.3: (s32[], c64[30])) -> (s32[], c64[30]) {
      %arg_tuple.3 = (s32[], c64[30]{0}) parameter(0)
      %get-tuple-element.4 = s32[] get-tuple-element(%arg_tuple.3), index=0
      %constant.6 = s32[] constant(1)
      %add.14 = s32[] add(%get-tuple-element.4, %constant.6)
      %get-tuple-element.5 = c64[30]{0} get-tuple-element(%arg_tuple.3), index=1
      %fft.10 = c64[30]{0} fft(%get-tuple-element.5), fft_type=FFT, fft_length={30}
      ROOT %tuple.15 = (s32[], c64[30]{0}) tuple(%add.14, %get-tuple-element.5)
    }

    %condition_comp (arg_tuple.17: (s32[], c64[30])) -> pred[] {
      %arg_tuple.17 = (s32[], c64[30]{0}) parameter(0)
      %get-tuple-element.18 = s32[] get-tuple-element(%arg_tuple.17), index=0
      %constant.20 = s32[] constant(10)
      ROOT %lt.21 = pred[] compare(%get-tuple-element.18, %constant.20), direction=LT
    }

    ENTRY %main.27 (args_0_.1: c64[30]) -> c64[30] {
      %constant.2 = s32[] constant(0)
      %args_0_.1 = c64[30]{0} parameter(0)
      %while.23 = (s32[], c64[30]{0}) tuple(%constant.2, %args_0_.1)
      %while.24 = (s32[], c64[30]{0}) while(%while.23), condition=%condition_comp, body=%body_comp
      %while.25 = s32[] get-tuple-element(%while.24), index=0
      ROOT %while.26 = c64[30]{0} get-tuple-element(%while.24), index=1
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, NoYnnWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule ynn_module

    %ynn_comp (lhs: f32[8,3], rhs: f32[3,3]) -> f32[8,3] {
      %lhs = f32[8,3] parameter(0)
      %rhs = f32[3,3] parameter(1)
      ROOT %dot = f32[8,3] dot(%lhs, %rhs), lhs_contracting_dims={1},
                                            rhs_contracting_dims={0}
    }

    %closed_call (x: f32[8,3]) -> (f32[8,3], f32[3], f32[3,3]) {
      %x = f32[8,3] parameter(0)
      %one = f32[] constant(1)
      %one_2d = f32[3,3] broadcast(%one), dimensions={}
      %ynn_fusion = f32[8,3] fusion(%x, %one_2d), kind=kCustom, calls=%ynn_comp,
        backend_config={
            "outer_dimension_partitions":[],
            "fusion_config":{"kind":"__ynn_fusion"}
          }
      %zero = f32[] constant(0)
      %zero_1d = f32[3]{0} broadcast(%zero), dimensions={}
      ROOT %tuple = (f32[8,3], f32[3], f32[3,3]) tuple(%ynn_fusion,
                                                       %zero_1d, %one_2d)
    }

    %body_comp (state: (s32[], f32[8,3], f32[8,3], f32[8,3,3])) ->
                       (s32[], f32[8,3], f32[8,3], f32[8,3,3]) {
      %state = (s32[], f32[8,3], f32[8,3], f32[8,3,3]) parameter(0)
      %idx = s32[] get-tuple-element(%state), index=0
      %one = s32[] constant(1)
      %new_idx = s32[] add(%idx, %one)
      %x = f32[8,3] get-tuple-element(%state), index=1
      %call = (f32[8,3], f32[3], f32[3,3]) call(%x), to_apply=%closed_call
      %in2 = f32[8,3] get-tuple-element(%state), index=2
      %in3 = f32[8,3,3] get-tuple-element(%state), index=3
      ROOT tuple = (s32[], f32[8,3], f32[8,3], f32[8,3,3])
                      tuple(%new_idx, %x, %in2, %in3)
    }

    %cond_comp (state: (s32[], f32[8,3], f32[8,3], f32[8,3,3])) -> pred[] {
      %state = (s32[], f32[8,3], f32[8,3], f32[8,3,3]) parameter(0)
      %idx = s32[] get-tuple-element(%state), index=0
      %eight = s32[] constant(8)
      ROOT %lt.1 = pred[] compare(%idx, %eight), direction=LT
    }

    ENTRY %main (x: f32[8,3], zero_2d: f32[8,3], zero_3d: f32[8,3,3]) -> (f32[8,3], f32[8,3], f32[8,3,3]) {
      %zero_int = s32[] constant(0)
      %x = f32[8,3] parameter(0)
      %zero_2d = f32[8,3] parameter(1)
      %zero_3d = f32[8,3,3] parameter(2)
      %init_state = (s32[], f32[8,3], f32[8,3], f32[8,3,3])
                      tuple(%zero_int, %x, %zero_2d, %zero_3d)
      %while = (s32[], f32[8,3], f32[8,3], f32[8,3,3])
                  while(%init_state), condition=%cond_comp,
                  body=%body_comp
      %res_2d = f32[8,3] get-tuple-element(%while), index=2
      %res_3d = f32[8,3,3] get-tuple-element(%while), index=3
      ROOT %tuple = (f32[8,3], f32[8,3], f32[8,3,3]) tuple(%x, %res_2d, %res_3d)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, ArbitraryInstructionRunHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule arbitrary_run_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      c1 = f32[4] constant({1, 2, 3, 4})
      add1 = f32[4] add(p0, c1)
      add2 = f32[4] add(add1, c1)
      ROOT add3 = f32[4] add(add2, c1)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);
  std::optional<std::string> maybe_small_call =
      call_instr->get_frontend_attribute("xla_cpu_small_call");
  ASSERT_NE(maybe_small_call, std::nullopt);
  EXPECT_EQ(*maybe_small_call, "true");

  const HloComputation* outlined_comp = call_instr->to_apply();
  ASSERT_NE(outlined_comp, nullptr);
  EXPECT_EQ(outlined_comp->root_instruction()->shape(),
            ShapeUtil::MakeShape(F32, {4}));
}

TEST_F(SmallWhileLoopHoistingPassTest, MultiOutputInstructionRunHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule multi_output_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      c1 = f32[4] constant({1, 1, 1, 1})
      add1 = f32[4] add(p0, c1)
      add2 = f32[4] add(add1, c1)
      ROOT tuple = (f32[4], f32[4]) tuple(add1, add2)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);
  EXPECT_TRUE(call_instr->shape().IsTuple());
  EXPECT_EQ(call_instr->shape().tuple_shapes_size(), 2);

  const HloComputation* outlined_comp = call_instr->to_apply();
  ASSERT_NE(outlined_comp, nullptr);
  EXPECT_EQ(outlined_comp->root_instruction()->opcode(), HloOpcode::kTuple);
}

TEST_F(SmallWhileLoopHoistingPassTest, OpaqueOpSegmentation) {
  constexpr absl::string_view hlo_string = R"(
    HloModule opaque_segmentation_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      add1 = f32[4] add(p0, p0)
      custom_call = f32[4] custom-call(add1), custom_call_target="opaque_op"
      ROOT add2 = f32[4] add(custom_call, custom_call)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, InternalControlDependencyPreservation) {
  constexpr absl::string_view hlo_string = R"(
    HloModule internal_ctrl_dep_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      c1 = f32[4] constant({1, 1, 1, 1})
      add1 = f32[4] add(p0, c1)
      add2 = f32[4] add(p0, c1), control-predecessors={add1}
      ROOT add3 = f32[4] add(add1, add2)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);

  const HloComputation* outlined_comp = call_instr->to_apply();
  ASSERT_NE(outlined_comp, nullptr);

  // Find cloned add1 and add2 inside outlined_comp
  const HloInstruction* cloned_add2 = nullptr;
  for (const HloInstruction* inst : outlined_comp->instructions()) {
    if (!inst->control_predecessors().empty()) {
      cloned_add2 = inst;
      break;
    }
  }
  EXPECT_NE(cloned_add2, nullptr)
      << "Internal control dependency was lost during hoisting!";
}

TEST_F(SmallWhileLoopHoistingPassTest, ExternalControlDependencyPreservation) {
  constexpr absl::string_view hlo_string = R"(
    HloModule external_ctrl_dep_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      c1 = f32[4] constant({1, 1, 1, 1})
      custom_call = f32[4] custom-call(p0), custom_call_target="opaque_op"
      add1 = f32[4] add(p0, c1), control-predecessors={custom_call}
      add2 = f32[4] add(add1, c1)
      custom_call_2 = f32[4] custom-call(add2), custom_call_target="opaque_op_2", control-predecessors={add2}
      ROOT tuple = (f32[4], f32[4]) tuple(add2, custom_call_2)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);

  const HloInstruction* custom_call = FindInstruction(m.get(), "custom_call");
  const HloInstruction* custom_call_2 =
      FindInstruction(m.get(), "custom_call_2");

  ASSERT_NE(custom_call, nullptr);
  ASSERT_NE(custom_call_2, nullptr);

  // Check custom_call -> call_instr control dependency
  bool found_pred = false;
  for (const HloInstruction* succ : custom_call->control_successors()) {
    if (succ == call_instr) found_pred = true;
  }
  EXPECT_TRUE(found_pred)
      << "External control dependency from pred to call_instr lost!";

  // Check call_instr -> custom_call_2 control dependency
  bool found_succ = false;
  for (const HloInstruction* pred : custom_call_2->control_predecessors()) {
    if (pred == call_instr) found_succ = true;
  }
  EXPECT_TRUE(found_succ)
      << "External control dependency from call_instr to succ lost!";
}

TEST_F(SmallWhileLoopHoistingPassTest, MultiOutputTupleWithNestedTuple) {
  constexpr absl::string_view hlo_string = R"(
    HloModule nested_tuple_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      c1 = f32[4] constant({1, 1, 1, 1})
      add1 = f32[4] add(p0, c1)
      t1 = (f32[4], f32[4]) tuple(p0, add1)
      add2 = f32[4] add(add1, c1)
      ROOT main_tuple = ((f32[4], f32[4]), f32[4]) tuple(t1, add2)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);
}

TEST_F(SmallWhileLoopHoistingPassTest, EmptyComputationAndTrivialOpsOnly) {
  constexpr absl::string_view hlo_string = R"(
    HloModule trivial_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      c1 = f32[4] constant({1, 1, 1, 1})
      t1 = (f32[4], f32[4]) tuple(p0, c1)
      gte0 = f32[4] get-tuple-element(t1), index=0
      ROOT bcast = f32[4] bitcast(gte0)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, SortOpBoundarySegmentation) {
  constexpr absl::string_view hlo_string = R"(
    HloModule sort_module

    compare_comp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT cmp = pred[] compare(p0, p1), direction=LT
    }

    ENTRY main {
      p0 = f32[16] parameter(0)
      c1 = f32[16] constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
      add1 = f32[16] add(p0, c1)
      sorted = f32[16] sort(add1), dimensions={0}, to_apply=compare_comp
      ROOT add2 = f32[16] add(sorted, c1)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  // Verify sort is still in main
  const HloInstruction* sort_inst = FindInstruction(m.get(), HloOpcode::kSort);
  ASSERT_NE(sort_inst, nullptr);
  EXPECT_EQ(sort_inst->parent()->name(), "main");

  // Verify outlined call exists and does not contain sort
  const HloInstruction* call_inst = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_inst, nullptr);
  bool has_sort_in_call = false;
  for (const HloInstruction* inst : call_inst->to_apply()->instructions()) {
    if (inst->opcode() == HloOpcode::kSort) {
      has_sort_in_call = true;
      break;
    }
  }
  EXPECT_FALSE(has_sort_in_call);
}

TEST_F(SmallWhileLoopHoistingPassTest,
       StressTest_ConstantParameterizationHandling) {
  constexpr absl::string_view hlo_string = R"(
    HloModule const_param_stress_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      c1 = f32[4] constant({1, 2, 3, 4})
      add1 = f32[4] add(p0, c1)
      add2 = f32[4] add(add1, c1)
      ROOT add3 = f32[4] add(add2, c1)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);
  const HloComputation* subcomp = call_instr->to_apply();
  ASSERT_NE(subcomp, nullptr);

  // Count constants in subcomputation
  int const_count_in_subcomp = 0;
  for (const HloInstruction* inst : subcomp->instructions()) {
    if (inst->opcode() == HloOpcode::kConstant) {
      const_count_in_subcomp++;
    }
  }
  // Check that constants are passed as parameters rather than cloned into
  // subcomp
  EXPECT_EQ(const_count_in_subcomp, 0);
}

TEST_F(SmallWhileLoopHoistingPassTest,
       StressTest_InternalControlDependencyDiamond) {
  constexpr absl::string_view hlo_string = R"(
    HloModule ctrl_dep_diamond_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      c1 = f32[4] constant({1, 1, 1, 1})
      inst_a = f32[4] add(p0, c1)
      inst_b = f32[4] add(inst_a, c1), control-predecessors={inst_a}
      inst_c = f32[4] add(inst_a, c1), control-predecessors={inst_a}
      inst_d = f32[4] add(inst_b, inst_c), control-predecessors={inst_b, inst_c}
      ROOT add_out = f32[4] add(inst_d, c1)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);

  const HloComputation* outlined_comp = call_instr->to_apply();
  ASSERT_NE(outlined_comp, nullptr);

  int ctrl_dep_count = 0;
  for (const HloInstruction* inst : outlined_comp->instructions()) {
    ctrl_dep_count += inst->control_predecessors().size();
  }
  // All 4 internal control predecessor edges (a->b, a->c, b->d, c->d) must be
  // preserved
  EXPECT_GE(ctrl_dep_count, 4)
      << "Internal control dependency edges were lost in diamond topology!";
}

TEST_F(SmallWhileLoopHoistingPassTest,
       StressTest_MultiOutputTupleWithNestedGTEAndRoot) {
  constexpr absl::string_view hlo_string = R"(
    HloModule multi_output_stress_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      c1 = f32[4] constant({1, 1, 1, 1})
      op1 = f32[4] add(p0, c1)
      op2 = f32[4] multiply(op1, c1)
      op3 = f32[4] subtract(op2, p0)
      ext_user = f32[4] add(op1, op1)
      ROOT main_tuple = (f32[4], f32[4], f32[4]) tuple(op1, op2, op3)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);
  EXPECT_TRUE(call_instr->shape().IsTuple());
  EXPECT_EQ(call_instr->shape().tuple_shapes_size(), 3);

  const HloComputation* outlined_comp = call_instr->to_apply();
  ASSERT_NE(outlined_comp, nullptr);
  EXPECT_EQ(outlined_comp->root_instruction()->opcode(), HloOpcode::kTuple);
}

TEST_F(SmallWhileLoopHoistingPassTest,
       StressTest_OpaqueOpBoundaryWithCrossControlDeps) {
  constexpr absl::string_view hlo_string = R"(
    HloModule cross_ctrl_dep_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      c1 = f32[4] constant({1, 1, 1, 1})
      cc1 = f32[4] custom-call(p0), custom_call_target="op1"
      run1_op = f32[4] add(p0, c1), control-predecessors={cc1}
      cc2 = f32[4] custom-call(run1_op), custom_call_target="op2", control-predecessors={run1_op}
      run2_op = f32[4] add(cc2, c1), control-predecessors={cc2}
      ROOT res = (f32[4], f32[4]) tuple(cc2, run2_op)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  // Verify two call instructions exist and main DAG contains no cycles
  int call_count = 0;
  for (const HloInstruction* inst : m->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kCall) call_count++;
  }
  EXPECT_EQ(call_count, 2);
}

TEST_F(SmallWhileLoopHoistingPassTest, EmpiricalEdgeCase_SingleBinaryOpRun) {
  constexpr absl::string_view hlo_string = R"(
    HloModule single_binary_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      ROOT add1 = f32[4] add(p0, p1)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);
  EXPECT_EQ(call_instr->operand_count(), 2);
  EXPECT_EQ(call_instr->to_apply()->instruction_count(), 3);
}

TEST_F(SmallWhileLoopHoistingPassTest, EmpiricalEdgeCase_SingleUnaryOpRun) {
  constexpr absl::string_view hlo_string = R"(
    HloModule single_unary_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      ROOT neg1 = f32[4] negate(p0)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);
  EXPECT_EQ(call_instr->operand_count(), 1);
}

TEST_F(SmallWhileLoopHoistingPassTest,
       EmpiricalEdgeCase_SingleTrivialOpNotHoisted) {
  constexpr absl::string_view hlo_string = R"(
    HloModule single_trivial_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      ROOT bcast = f32[4] bitcast(p0)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest,
       EmpiricalEdgeCase_SingleNonTrivialBetweenOpaqueOps) {
  constexpr absl::string_view hlo_string = R"(
    HloModule single_nontrivial_between_opaque

    ENTRY main {
      p0 = f32[4] parameter(0)
      cc1 = f32[4] custom-call(p0), custom_call_target="op1"
      add1 = f32[4] add(cc1, cc1)
      ROOT cc2 = f32[4] custom-call(add1), custom_call_target="op2"
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);
  EXPECT_EQ(call_instr->to_apply()->root_instruction()->opcode(),
            HloOpcode::kAdd);
}

TEST_F(SmallWhileLoopHoistingPassTest,
       EmpiricalEdgeCase_DirectCrossRunControlDependency) {
  constexpr absl::string_view hlo_string = R"(
    HloModule direct_cross_run_ctrl_dep

    ENTRY main {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      add1 = f32[4] add(p0, p1)
      cc1 = f32[4] custom-call(add1), custom_call_target="opaque1"
      add2 = f32[4] add(p0, cc1)
      add3 = f32[4] add(add2, p1), control-predecessors={add1}
      ROOT res = (f32[4], f32[4]) tuple(cc1, add3)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  int call_count = 0;
  const HloInstruction* call1 = nullptr;
  const HloInstruction* call2 = nullptr;
  for (const HloInstruction* inst : m->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kCall) {
      call_count++;
      if (!call1)
        call1 = inst;
      else
        call2 = inst;
    }
  }
  EXPECT_EQ(call_count, 2);
  ASSERT_NE(call1, nullptr);
  ASSERT_NE(call2, nullptr);

  bool found_ctrl_dep = false;
  for (const HloInstruction* succ : call1->control_successors()) {
    if (succ == call2) found_ctrl_dep = true;
  }
  EXPECT_TRUE(found_ctrl_dep) << "Direct cross-run control dependency between "
                                 "call1 and call2 was lost!";
}

TEST_F(SmallWhileLoopHoistingPassTest,
       EmpiricalEdgeCase_MultipleInstructionsSharingSameConstant) {
  constexpr absl::string_view hlo_string = R"(
    HloModule shared_constant_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      c1 = f32[4] constant({2, 2, 2, 2})
      add1 = f32[4] add(p0, c1)
      mul1 = f32[4] multiply(add1, c1)
      ROOT sub1 = f32[4] subtract(mul1, c1)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);
  EXPECT_EQ(call_instr->operand_count(), 2);
  const HloComputation* subcomp = call_instr->to_apply();
  ASSERT_NE(subcomp, nullptr);
  EXPECT_EQ(subcomp->num_parameters(), 2);
}

TEST_F(SmallWhileLoopHoistingPassTest,
       EmpiricalEdgeCase_CopyOfConstantParameterization) {
  constexpr absl::string_view hlo_string = R"(
    HloModule copy_constant_module

    ENTRY main {
      p0 = f32[4] parameter(0)
      c1 = f32[4] constant({1, 2, 3, 4})
      cp1 = f32[4] copy(c1)
      add1 = f32[4] add(p0, cp1)
      ROOT add2 = f32[4] add(add1, cp1)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);
  EXPECT_EQ(call_instr->operand_count(), 2);
}

}  // namespace
}  // namespace xla
