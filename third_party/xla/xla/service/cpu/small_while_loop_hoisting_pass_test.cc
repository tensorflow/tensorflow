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
    return cpu::SmallWhileLoopHoistingPass(256).Run(module);
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
