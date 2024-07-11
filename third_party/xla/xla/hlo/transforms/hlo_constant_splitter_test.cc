/* Copyright 2022 The OpenXLA Authors.
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

#include "xla/hlo/transforms/hlo_constant_splitter.h"

#include <cstdint>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_parser.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using HloConstantSplitterTest = HloTestBase;

TEST_F(HloConstantSplitterTest, SplitConstants) {
  const char* module_str = R"(
    HloModule test_module

    ENTRY entry_computation {
      param = (f32[], f32[]) parameter(0),
        sharding={{maximal device=0}, {maximal device=0}}
      gte0 = f32[] get-tuple-element(param), index=0
      gte1 = f32[] get-tuple-element(param), index=1
      constant = f32[] constant(94.1934)
      add1 = f32[] add(constant, gte0)
      add2 = f32[] add(constant, gte1)
      ROOT root = (f32[], f32[], f32[]) tuple(constant, add1, add2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_str));
  TF_ASSERT_OK(HloConstantSplitter().Run(module.get()).status());

  // Check that every constant has at most one user.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kConstant) {
        EXPECT_LE(instruction->user_count(), 1);
      }
    }
  }
}

TEST_F(HloConstantSplitterTest, PreservingConstantsWithZeroUsers) {
  const char* module_str = R"(
    HloModule test_module

    ENTRY entry_computation {
      param = (f32[], f32[]) parameter(0),
        sharding={{maximal device=0}, {maximal device=0}}
      gte0 = f32[] get-tuple-element(param), index=0
      gte1 = f32[] get-tuple-element(param), index=1
      constant1 = f32[] constant(94.1934)
      constant2 = f32[] constant(9.1934)
      ROOT root = (f32[], f32[]) tuple(gte0, gte1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_str));
  HloConstantSplitter pass = HloConstantSplitter();
  const auto status_or = HloTestBase::RunHloPass(&pass, module.get());
  TF_ASSERT_OK(status_or.status());
  // Verify that the changed flag returned is correct.
  EXPECT_FALSE(status_or.value());
}

TEST_F(HloConstantSplitterTest, SplittingExpressionsWithBroadcast) {
  const char* module_str = R"(
    HloModule test_module

    ENTRY entry_computation {
      gte0 = f32[1024] parameter(0)
      gte1 = f32[1024] parameter(1)
      constant1 = f32[1024] iota(), iota_dimension=0
      constant2 = f32[] constant(9.1934)
      constant3 = f32[] constant(0.0)
      constant4 = f32[] constant(1.0)
      b = f32[1024] broadcast(constant2), dimensions={}
      b2 = f32[1024] broadcast(constant3), dimensions={}
      b3 = f32[1024] broadcast(constant4), dimensions={}
      cmp = pred[1024] compare(constant1, b), direction=LT
      s = f32[1024] select(cmp, b2, b3)
      a1 = f32[1024] add(s, gte0)
      a2 = f32[1024] add(s, gte1)
      ROOT root = (f32[1024], f32[1024]) tuple(a1, a2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_str));
  HloConstantSplitter pass = HloConstantSplitter(/*split_expressions=*/true);
  const auto status_or = HloTestBase::RunHloPass(&pass, module.get());
  TF_ASSERT_OK(status_or.status());
  // Verify that the changed flag returned is correct.
  EXPECT_TRUE(status_or.value());
  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()).status());
  XLA_VLOG_LINES(1, module->entry_computation()->ToString());
  EXPECT_EQ(module->entry_computation()->instruction_count(), 23);
}

TEST_F(HloConstantSplitterTest, SplittingExpressionsWithSlice) {
  const char* module_str = R"(
    HloModule test_module

    ENTRY entry_computation {
      iota.0 = u32[64] iota(), iota_dimension=0
      slice.0 = u32[32] slice(iota.0), slice={[0:32]}
      broadcast.0 = u32[16,32] broadcast(slice.0), dimensions={1}
      broadcast.1 = u32[32,32] broadcast(slice.0), dimensions={1}
      p.0 = u32[16,32] parameter(0)
      p.1 = u32[32,32] parameter(1)
      add.0 = u32[16,32] add(p.0, broadcast.0)
      add.1 = u32[32,32] add(p.1, broadcast.1)
      ROOT root = (u32[16,32], u32[32,32]) tuple(add.0, add.1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_str));
  HloConstantSplitter pass = HloConstantSplitter(/*split_expressions=*/true);
  const auto status_or = HloTestBase::RunHloPass(&pass, module.get());
  TF_ASSERT_OK(status_or.status());
  // Verify that the changed flag returned is correct.
  EXPECT_TRUE(status_or.value());
  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()).status());
  XLA_VLOG_LINES(1, module->entry_computation()->ToString());
  EXPECT_EQ(module->entry_computation()->instruction_count(), 11);
}

TEST_F(HloConstantSplitterTest, NoSplittingSideEffectExpressions) {
  const char* module_str = R"(
    HloModule test_module

    ENTRY entry_computation {
      gte0 = f32[1024] parameter(0)
      gte1 = f32[1024] parameter(1)
      constant1 = f32[1024] iota(), iota_dimension=0
      constant2 = f32[] constant(9.1934)
      constant3 = f32[] constant(0.0)
      constant4 = f32[] constant(0.0)
      constant5 = f32[] constant(1.0)
      b = f32[1024] broadcast(constant2), dimensions={}
      b2 = f32[1024] broadcast(constant3), dimensions={}
      rng = f32[] rng(constant4, constant5), distribution=rng_uniform
      b3 = f32[1024] broadcast(rng), dimensions={}
      cmp = pred[1024] compare(constant1, b), direction=LT
      s = f32[1024] select(cmp, b2, b3)
      a1 = f32[1024] add(s, gte0)
      a2 = f32[1024] add(s, gte1)
      ROOT root = (f32[1024], f32[1024]) tuple(a1, a2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_str));
  HloConstantSplitter pass = HloConstantSplitter(/*split_expressions=*/true);

  const int64_t count_before = module->entry_computation()->instruction_count();
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          HloTestBase::RunHloPass(&pass, module.get()));
  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()).status());
  const int64_t count_after_dce =
      module->entry_computation()->instruction_count();

  // The HloConstantSplitter pass duplicates several constant expressions. Then
  // the DCE pass removes the dead instructions. Although the flag changed is
  // true, we do not alter the module in essense.
  EXPECT_TRUE(changed);
  EXPECT_EQ(count_before, count_after_dce);
  int64_t rng_count = 0;
  for (HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kRng) {
      rng_count++;
    }
  }
  EXPECT_EQ(rng_count, 1);
}

TEST_F(HloConstantSplitterTest, InstructionsWithOneUser) {
  // This HloModule is from b/302613851#comment3.
  const char* module_str = R"(
    HloModule test_module, entry_computation_layout={(f32[1024]{0:T(512)})->f32[1024]{0:T(512)}}

    reduce.add {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add = f32[] add(a, b)
    }

    ENTRY entry_computation {
      constant1 = f32[] constant(1.1)
      b1 = f32[1024]{0} broadcast(constant1), dimensions={}
      iota.1 = f32[1024]{0} iota(), iota_dimension=0
      add.1 = f32[1024]{0} add(b1, iota.1)
      p0 = f32[1024]{0} parameter(0), sharding={devices=[4]0,1,2,3}
      custom-call.0 = f32[256]{0} custom-call(p0), custom_call_target="SPMDFullToShardShape", sharding={manual}
      constant0 = f32[] constant(0)
      reduce.1 = f32[] reduce(custom-call.0, constant0), dimensions={0}, to_apply=reduce.add
      b3 = f32[1024]{0} broadcast(reduce.1), dimensions={}
      add.2 = f32[1024]{0} add(add.1, b3)
      custom-call.1 = f32[4096]{0} custom-call(add.2), custom_call_target="SPMDShardToFullShape", sharding={devices=[4]0,1,2,3}
      reshape = f32[4,1024]{1,0} reshape(custom-call.1)
      reduce.2 = f32[1024]{0} reduce(reshape, constant0), dimensions={0}, to_apply=reduce.add
      iota.2 = f32[1024]{0} iota(), iota_dimension=0
      mul = f32[1024]{0} multiply(b1, iota.2)
      ROOT sub = f32[1024]{0} subtract(reduce.2, mul), sharding={devices=[4]0,1,2,3}
    } // entry_computation
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_str));
  HloConstantSplitter pass = HloConstantSplitter(/*split_expressions=*/true);
  // Verify that the module is not changed as splitting on rng is prevented.
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          HloTestBase::RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  int64_t broadcast_count_before_dce = 0, broadcast_count_after_dce = 0;
  for (HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kBroadcast) {
      broadcast_count_before_dce++;
    }
  }
  EXPECT_EQ(broadcast_count_before_dce, 4);

  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()).status());
  for (HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kBroadcast) {
      broadcast_count_after_dce++;
    }
  }
  EXPECT_EQ(broadcast_count_after_dce, 3);
}

}  // namespace
}  // namespace xla
