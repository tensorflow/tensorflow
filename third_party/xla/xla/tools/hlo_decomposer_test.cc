/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/tools/hlo_decomposer.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::xla::testing::opcode_matchers::AsyncDone;
using ::xla::testing::opcode_matchers::AsyncStart;
using ::xla::testing::opcode_matchers::AsyncUpdate;
using ::xla::testing::opcode_matchers::Parameter;

class HloDecomposerTest : public HloTestBase {
 protected:
  std::unique_ptr<HloModule> GetModule() {
    absl::string_view kHlo = R"(
HloModule test_module, entry_computation_layout={(bf16[1024,8192]{1,0}, f32[8192]{0}, f32[16384]{0})->(bf16[1024]{0}, bf16[1024]{0}, f32[16384]{0}, f32[16384]{0})}

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add.1 = f32[] add(p0, p1)
}

fused_computation.1 {
  param_1.3 = f32[8192]{0} parameter(1)
  broadcast.2 = f32[1024,8192]{1,0} broadcast(param_1.3), dimensions={1}
  param_0.3 = bf16[1024,8192]{1,0} parameter(0)
  convert.5 = f32[1024,8192]{1,0} convert(param_0.3)
  multiply.2 = f32[1024,8192]{1,0} multiply(broadcast.2, convert.5)
  c0_1 = f32[] constant(0)
  reduce.1 = f32[1024]{0} reduce(multiply.2, c0_1), dimensions={1}, to_apply=add
  ROOT convert.4 = bf16[1024]{0} convert(reduce.1)
}

fused_computation.2 {
  p0.0 = bf16[1024,8192]{1,0} parameter(0)
  c.0 = f32[1024,8192]{1,0} convert(p0.0)
  co0_1.1 = f32[] constant(0)
  p.0 = f32[8192]{0} parameter(1)
  b.0 = f32[1024,8192]{1,0} broadcast(p.0), dimensions={1}
  m.0 = f32[1024,8192]{1,0} multiply(b.0, c.0)
  r.0 = f32[1024]{0} reduce(m.0, co0_1.1), dimensions={1}, to_apply=add
  ROOT c.1 = bf16[1024]{0} convert(r.0)
}

exp {
  param_0.5 = f32[16384]{0} parameter(0)
  m.4 = f32[16384]{0} multiply(param_0.5, param_0.5)
  e = f32[16384]{0} exponential(m.4)
  l.clone.1 = f32[16384]{0} log(m.4)
  ROOT tuple = (f32[16384]{0}, f32[16384]{0}) tuple(e, l.clone.1)
}

ENTRY main {
  p0.1 = bf16[1024,8192]{1,0} parameter(0)
  p1.1 = f32[8192]{0} parameter(1)
  fusion.1 = bf16[1024]{0} fusion(p0.1, p1.1), kind=kInput, calls=fused_computation.1
  fusion.2 = bf16[1024]{0} fusion(p0.1, p1.1), kind=kInput, calls=fused_computation.2
  p2 = f32[16384]{0} parameter(2)
  e.1 = (f32[16384]{0}, f32[16384]{0}) fusion(p2), kind=kInput, calls=exp
  get-tuple-element.1 = f32[16384]{0} get-tuple-element(e.1), index=1
  get-tuple-element = f32[16384]{0} get-tuple-element(e.1), index=0
  ROOT result = (bf16[1024]{0}, bf16[1024]{0}, f32[16384]{0}, f32[16384]{0}) tuple(fusion.1, fusion.2, get-tuple-element.1, get-tuple-element)
})";
    return ParseAndReturnVerifiedModule(kHlo).value();
  }

  void FindAndCompare(const std::vector<std::unique_ptr<HloModule>>& modules,
                      absl::string_view module_name,
                      absl::string_view pattern) {
    auto iter =
        absl::c_find_if(modules, [&](const std::unique_ptr<HloModule>& module) {
          return module->name() == module_name;
        });
    EXPECT_NE(iter, modules.end()) << "No module named " << module_name;
    if (iter == modules.end()) {
      return;
    }
    EXPECT_TRUE(*RunFileCheck((*iter)->ToString(), pattern));
  }
};

TEST_F(HloDecomposerTest, DecomposeNoDedup) {
  auto module = GetModule();
  TF_ASSERT_OK_AND_ASSIGN(
      auto decomposed,
      DecomposeHloModule(*module, /*deduplicate_modules=*/false));
  EXPECT_EQ(decomposed.size(), 3);

  FindAndCompare(decomposed, "fusion.1", R"(
CHECK: %add{{.*}} {
CHECK: %fused_computation.1
CHECK: ENTRY
CHECK-THEN: %parameter.0 = bf16[1024,8192]{1,0} parameter(0)
CHECK-THEN: %parameter.1 = f32[8192]{0} parameter(1)
CHECK-THEN: ROOT %fusion.1
)");

  FindAndCompare(decomposed, "fusion.2", R"(
CHECK: %add{{.*}} {
CHECK: %fused_computation.2
CHECK: ENTRY
CHECK-THEN: %parameter.0 = bf16[1024,8192]{1,0} parameter(0)
CHECK-THEN: %parameter.1 = f32[8192]{0} parameter(1)
CHECK-THEN: ROOT %fusion.2
)");

  FindAndCompare(decomposed, "e.1", R"(
CHECK: %exp{{.*}} {
CHECK: ENTRY
CHECK-THEN: %parameter.0 = f32[16384]{0} parameter(0)
CHECK-THEN: ROOT %e.1
)");
}

TEST_F(HloDecomposerTest, DecomposeDedup) {
  auto module = GetModule();
  TF_ASSERT_OK_AND_ASSIGN(
      auto decomposed,
      DecomposeHloModule(*module, /*deduplicate_modules=*/true));
  EXPECT_EQ(decomposed.size(), 2);

  FindAndCompare(decomposed, "fusion.1", R"(
CHECK: %add{{.*}} {
CHECK: %fused_computation.1
CHECK: ENTRY
CHECK-THEN: %parameter.0 = bf16[1024,8192]{1,0} parameter(0)
CHECK-THEN: %parameter.1 = f32[8192]{0} parameter(1)
CHECK-THEN: ROOT %fusion.1
)");

  FindAndCompare(decomposed, "e.1", R"(
CHECK: %exp{{.*}} {
CHECK: ENTRY
CHECK-THEN: %parameter.0 = f32[16384]{0} parameter(0)
CHECK-THEN: ROOT %e.1
)");
}

TEST_F(HloDecomposerTest, ExtractComputationIntoNewModule) {
  std::unique_ptr<HloModule> module = ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY main {
  p0 = s8[10,10] parameter(0)
  p1 = s8[10,10] parameter(1)
  ROOT r = s8[10,10] add(p0, p1)
})")
                                          .value();
  auto new_module =
      ExtractComputationIntoNewModule(*module->entry_computation());
  EXPECT_EQ(new_module->name(), module->entry_computation()->name());
}

TEST_F(HloDecomposerTest, ExtractProducerConsumersIntoNewModule) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY main {
  p0 = f32[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  dot = f32[10,10] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  p2 = f32[10,10] parameter(2)
  add = f32[10,10] add(dot, p2)
  subtract = f32[10,10] subtract(dot, p2)
  ROOT r = f32[10,10] add(add, subtract)
})"));
  HloInstruction* dot =
      module->entry_computation()->GetInstructionWithName("dot");
  auto new_module = ExtractProducerConsumersIntoNewModule(*dot);
  TF_ASSERT_OK_AND_ASSIGN(bool check_result,
                          RunFileCheck(new_module->ToString(), R"(
CHECK: ENTRY
CHECK-THEN: %parameter.0 = f32[10,10]{1,0} parameter(0)
CHECK-THEN: %parameter.1 = f32[10,10]{1,0} parameter(1)
CHECK-THEN: %dot.1 = f32[10,10]{1,0} dot(%parameter.0, %parameter.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
CHECK-THEN: %parameter.2 = f32[10,10]{1,0} parameter(2)
CHECK-THEN: %add.1 = f32[10,10]{1,0} add(%dot.1, %parameter.2)
CHECK-THEN: %subtract.1 = f32[10,10]{1,0} subtract(%dot.1, %parameter.2)
CHECK-THEN: ROOT %tuple.1 = (f32[10,10]{1,0}, f32[10,10]{1,0}) tuple(%add.1, %subtract.1)
  )"));
  EXPECT_TRUE(check_result);
}

TEST_F(HloDecomposerTest,
       ExtractInstructionIntoNewModulePreservesExecutionThread) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module

fused_computation {
  p0 = f32[8] parameter(0)
  ROOT neg = f32[8] negate(p0)
}, execution_thread="parallel"

parallel_comp {
  p0 = f32[8] parameter(0)
  ROOT loop_fusion = f32[8] fusion(p0), kind=kLoop, calls=fused_computation
}, execution_thread="parallel"

ENTRY main {
  p0 = f32[8] parameter(0)
  call-start = ((f32[8]), f32[8]) call-start(p0), async_execution_thread="parallel", to_apply=parallel_comp
  ROOT call-done = f32[8] call-done(call-start)
})"));
  HloComputation* parallel_comp =
      module->GetComputationWithName("parallel_comp");
  ASSERT_NE(parallel_comp, nullptr);
  HloInstruction* loop_fusion =
      parallel_comp->GetInstructionWithName("loop_fusion");
  ASSERT_NE(loop_fusion, nullptr);

  std::unique_ptr<HloModule> new_module =
      ExtractInstructionIntoNewModule(*loop_fusion);

  EXPECT_EQ(new_module->entry_computation()->execution_thread(), "parallel");
}

TEST_F(HloDecomposerTest,
       ExtractProducerConsumersIntoNewModulePreservesExecutionThread) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module

fused_computation {
  p0 = f32[8] parameter(0)
  ROOT neg = f32[8] negate(p0)
}, execution_thread="parallel"

parallel_comp {
  p0 = f32[8] parameter(0)
  loop_fusion = f32[8] fusion(p0), kind=kLoop, calls=fused_computation
  ROOT result = f32[8] add(loop_fusion, p0)
}, execution_thread="parallel"

ENTRY main {
  p0 = f32[8] parameter(0)
  call-start = ((f32[8]), f32[8]) call-start(p0), async_execution_thread="parallel", to_apply=parallel_comp
  ROOT call-done = f32[8] call-done(call-start)
})"));
  HloComputation* parallel_comp =
      module->GetComputationWithName("parallel_comp");
  ASSERT_NE(parallel_comp, nullptr);
  HloInstruction* loop_fusion =
      parallel_comp->GetInstructionWithName("loop_fusion");
  ASSERT_NE(loop_fusion, nullptr);

  std::unique_ptr<HloModule> new_module =
      ExtractProducerConsumersIntoNewModule(*loop_fusion);

  EXPECT_EQ(new_module->entry_computation()->execution_thread(), "parallel");
}

TEST_F(HloDecomposerTest, ExtractProducerConsumersIntoNewModuleSingleConsumer) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY main {
  p0 = f32[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  dot = f32[10,10] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  p2 = f32[10,10] parameter(2)
  ROOT add = f32[10,10] add(dot, p2)
})"));
  HloInstruction* dot =
      module->entry_computation()->GetInstructionWithName("dot");
  auto new_module = ExtractProducerConsumersIntoNewModule(*dot);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                          RunFileCheck(new_module->ToString(), R"(
CHECK: ENTRY
CHECK-THEN: %parameter.0 = f32[10,10]{1,0} parameter(0)
CHECK-THEN: %parameter.1 = f32[10,10]{1,0} parameter(1)
CHECK-THEN: %dot.1 = f32[10,10]{1,0} dot(%parameter.0, %parameter.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
CHECK-THEN: %parameter.2 = f32[10,10]{1,0} parameter(2)
CHECK-THEN: ROOT %add.1 = f32[10,10]{1,0} add(%dot.1, %parameter.2)
  )"));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(HloDecomposerTest,
       ExtractCollectiveOperationsIntoNewModuleFlattensTupleResults) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  ar = (f32[10], f32[10]) all-reduce(p0, p1), replica_groups={{0,1}}, to_apply=add
  ag = f32[20] all-gather(p0), dimensions={0}, replica_groups={{0,1}}
  gte0 = f32[10] get-tuple-element(ar), index=0
  gte1 = f32[10] get-tuple-element(ar), index=1
  ROOT result = (f32[10], f32[10], f32[20]) tuple(gte0, gte1, ag)
})"));
  HloInstruction* ar =
      module->entry_computation()->GetInstructionWithName("ar");
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("ag");
  std::unique_ptr<HloModule> new_module =
      ExtractCollectiveOperationsIntoNewModule(
          {ar, ag}, /*done_ops=*/{},
          /*non_optimized_ops=*/{HloOpcode::kAllReduce, HloOpcode::kAllGather},
          /*return_tuple=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                          RunFileCheck(new_module->ToString(), R"(
CHECK: ENTRY
CHECK-THEN: %[[P0:.*]] = f32[10]{0} parameter(0)
CHECK-THEN: %[[P1:.*]] = f32[10]{0} parameter(1)
CHECK-THEN: %[[AR:.*]] = (f32[10]{0}, f32[10]{0}) all-reduce(%[[P0]], %[[P1]])
CHECK-THEN: %[[P2:.*]] = f32[10]{0} parameter(2)
CHECK-THEN: %[[AG:.*]] = f32[20]{0} all-gather(%[[P2]])
CHECK-THEN: %[[GTE0:.*]] = f32[10]{0} get-tuple-element(%[[AR]]), index=0
CHECK-THEN: %[[GTE1:.*]] = f32[10]{0} get-tuple-element(%[[AR]]), index=1
CHECK-THEN: ROOT %{{.*}} = (f32[10]{0}, f32[10]{0}, f32[20]{0}) tuple(%[[GTE0]], %[[GTE1]], %[[AG]])
  )"));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(HloDecomposerTest,
       ExtractProducerConsumersIntoNewModuleFlattensTupleConsumers) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module

fused_comp {
  p0 = f32[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  add = f32[10,10] add(p0, p1)
  mul = f32[10,10] multiply(p0, p1)
  ROOT tuple = (f32[10,10], f32[10,10]) tuple(add, mul)
}

ENTRY main {
  p0 = f32[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  dot = f32[10,10] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  p2 = f32[10,10] parameter(2)
  fusion = (f32[10,10], f32[10,10]) fusion(dot, p2), kind=kLoop, calls=fused_comp
  subtract = f32[10,10] subtract(dot, p2)
  gte0 = f32[10,10] get-tuple-element(fusion), index=0
  gte1 = f32[10,10] get-tuple-element(fusion), index=1
  ROOT r = (f32[10,10], f32[10,10], f32[10,10]) tuple(gte0, gte1, subtract)
})"));
  HloInstruction* dot =
      module->entry_computation()->GetInstructionWithName("dot");
  std::unique_ptr<HloModule> new_module =
      ExtractProducerConsumersIntoNewModule(*dot);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                          RunFileCheck(new_module->ToString(), R"(
CHECK: ENTRY
CHECK-THEN: %[[P0:.*]] = f32[10,10]{1,0} parameter(0)
CHECK-THEN: %[[P1:.*]] = f32[10,10]{1,0} parameter(1)
CHECK-THEN: %[[DOT:.*]] = f32[10,10]{1,0} dot(%[[P0]], %[[P1]])
CHECK-THEN: %[[P2:.*]] = f32[10,10]{1,0} parameter(2)
CHECK-THEN: %[[FUSION:.*]] = (f32[10,10]{1,0}, f32[10,10]{1,0}) fusion(%[[DOT]], %[[P2]])
CHECK-THEN: %[[P3:.*]] = f32[10,10]{1,0} parameter(3)
CHECK-THEN: %[[SUB:.*]] = f32[10,10]{1,0} subtract(%[[DOT]], %[[P3]])
CHECK-THEN: %[[GTE0:.*]] = f32[10,10]{1,0} get-tuple-element(%[[FUSION]]), index=0
CHECK-THEN: %[[GTE1:.*]] = f32[10,10]{1,0} get-tuple-element(%[[FUSION]]), index=1
CHECK-THEN: ROOT %{{.*}} = (f32[10,10]{1,0}, f32[10,10]{1,0}, f32[10,10]{1,0}) tuple(%[[GTE0]], %[[GTE1]], %[[SUB]])
  )"));
  EXPECT_TRUE(filecheck_result);
}

std::vector<std::string> ModuleNames(
    const std::vector<std::unique_ptr<HloModule>>& modules) {
  std::vector<std::string> names;
  names.reserve(modules.size());
  for (const std::unique_ptr<HloModule>& module : modules) {
    names.push_back(module->name());
  }
  return names;
}

absl::Status VerifyExtractedModule(HloModule* module) {
  return HloVerifier(/*layout_sensitive=*/false,
                     /*allow_mixed_precision=*/false)
      .Run(module)
      .status();
}

constexpr absl::string_view kSparseCoreOffloadHlo = R"hlo(
HloModule main

wrapped {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ROOT add = f32[8] add(p0, p1)
}, execution_thread="sparsecore"

wrapped.1 {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ROOT add.1 = f32[8] add(p0, p1)
}, execution_thread="sparsecore"

ENTRY main {
  x = f32[8] parameter(0)
  y = f32[8] parameter(1)
  exp = f32[8] exponential(x)
  offload_start = ((f32[8], f32[8]), f32[8]) async-start(exp, y), async_execution_thread="sparsecore", calls=wrapped, backend_config={"sparse_core_config":{"core_ids":["0"]}}
  offload_done = f32[8] async-done(offload_start)
  offload_start.1 = ((f32[8], f32[8]), f32[8]) async-start(exp, y), async_execution_thread="sparsecore", calls=wrapped.1, backend_config={"sparse_core_config":{"core_ids":["1"]}}
  offload_done.1 = f32[8] async-done(offload_start.1)
  ROOT tuple = (f32[8], f32[8]) tuple(offload_done, offload_done.1)
}
)hlo";

TEST_F(HloDecomposerTest, SparseCoreOffloadChainIsOneUnit) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kSparseCoreOffloadHlo));
  module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes(
      "test_disabled_pass");

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<HloModule>> units,
      DecomposeHloModule(*module, /*deduplicate_modules=*/true));
  EXPECT_THAT(ModuleNames(units),
              UnorderedElementsAre("exp", "offload_done", "offload_done.1"));
  for (const std::unique_ptr<HloModule>& unit : units) {
    EXPECT_THAT(VerifyExtractedModule(unit.get()), IsOk()) << unit->ToString();
  }

  auto offload_it =
      absl::c_find_if(units, [](const std::unique_ptr<HloModule>& unit) {
        return unit->name() == "offload_done";
      });
  ASSERT_NE(offload_it, units.end());
  EXPECT_THAT((*offload_it)->config().debug_options().xla_disable_hlo_passes(),
              Contains("test_disabled_pass"));
  const HloInstruction* done =
      (*offload_it)->entry_computation()->root_instruction();
  EXPECT_THAT(done, AsyncDone(AsyncStart(Parameter(0), Parameter(1))));
  EXPECT_THAT(done->operand(0)->raw_backend_config_string(),
              HasSubstr("\"core_ids\":[\"0\"]"));
}

TEST_F(HloDecomposerTest,
       CompiledSparseCoreOffloadRestoresCallAndSchedulesParametersFirst) {
  constexpr absl::string_view kCompiledOffloadHlo = R"hlo(
HloModule main

program {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ROOT sub = f32[8] subtract(p0, p1)
}, execution_thread="sparsecore"

wrapped {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ROOT caller = f32[8] custom-call(p0, p1), custom_call_target="TODO(embedded_program)", called_computations={program}
}, execution_thread="sparsecore"

ENTRY main {
  x = f32[8] parameter(0)
  start = ((), (), (f32[8])) async-start(), async_execution_thread="sparsecore", calls=wrapped, output_to_operand_aliasing={{2, 0}: (0, {})}, backend_config={"sparse_core_config":{"core_ids":["0"]}}
  update = ((f32[8], f32[8]), f32[8], (f32[8])) async-update(start, x, x)
  ROOT done = f32[8] async-done(update)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kCompiledOffloadHlo));
  ASSERT_OK(HloTrivialScheduler().Run(module.get()));
  ASSERT_TRUE(module->has_schedule());

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<HloModule>> units,
      DecomposeHloModule(*module, /*deduplicate_modules=*/true));
  ASSERT_THAT(units, SizeIs(1));
  HloModule* unit = units[0].get();
  EXPECT_THAT(VerifyExtractedModule(unit), IsOk()) << unit->ToString();

  // Duplicate external uses of `x` share a single entry parameter(0).
  EXPECT_EQ(unit->entry_computation()->num_parameters(), 1);
  const HloInstruction* done = unit->entry_computation()->root_instruction();
  EXPECT_THAT(done,
              AsyncDone(AsyncUpdate(AsyncStart(), Parameter(0), Parameter(0))));

  const auto* start = Cast<HloAsyncStartInstruction>(done->async_chain_start());
  EXPECT_EQ(start->async_wrapped_instruction()->opcode(), HloOpcode::kCall);
  EXPECT_EQ(start->async_wrapped_instruction()->to_apply()->name(), "program");
  EXPECT_THAT(unit->ToString(), Not(HasSubstr("TODO(embedded_program)")));

  ASSERT_TRUE(unit->has_schedule());
  EXPECT_THAT(
      unit->schedule().sequence(unit->entry_computation()).instructions(),
      ElementsAre(Parameter(0), AsyncStart(), AsyncUpdate(), AsyncDone()));
}

TEST_F(HloDecomposerTest, UnextractableSparseCoreOffloadsAreSkipped) {
  constexpr absl::string_view kLoopCrossingHlo = R"hlo(
HloModule main

wrapped {
  p0 = f32[8] parameter(0)
  ROOT neg = f32[8] negate(p0)
}, execution_thread="sparsecore"

body {
  p = (((f32[8]), f32[8])) parameter(0)
  carried = ((f32[8]), f32[8]) get-tuple-element(p), index=0
  ROOT t = (((f32[8]), f32[8])) tuple(carried)
}

cond {
  p = (((f32[8]), f32[8])) parameter(0)
  ROOT c = pred[] constant(false)
}

ENTRY main {
  x = f32[8] parameter(0)
  exp = f32[8] exponential(x)
  start = ((f32[8]), f32[8]) async-start(exp), async_execution_thread="sparsecore", calls=wrapped
  init = (((f32[8]), f32[8])) tuple(start)
  loop = (((f32[8]), f32[8])) while(init), condition=cond, body=body
  carried = ((f32[8]), f32[8]) get-tuple-element(loop), index=0
  ROOT done = f32[8] async-done(carried)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> loop_module,
                       ParseAndReturnVerifiedModule(kLoopCrossingHlo));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<HloModule>> loop_units,
      DecomposeHloModule(*loop_module, /*deduplicate_modules=*/true));
  EXPECT_THAT(ModuleNames(loop_units), ElementsAre("exp"));
  for (const std::unique_ptr<HloModule>& unit : loop_units) {
    EXPECT_THAT(VerifyExtractedModule(unit.get()), IsOk()) << unit->ToString();
  }

  constexpr absl::string_view kBranchingHlo = R"hlo(
HloModule main

wrapped {
  p0 = f32[8] parameter(0)
  ROOT neg = f32[8] negate(p0)
}, execution_thread="sparsecore"

ENTRY main {
  x = f32[8] parameter(0)
  exp = f32[8] exponential(x)
  start = ((f32[8]), f32[8]) async-start(exp), async_execution_thread="sparsecore", calls=wrapped, backend_config={"sparse_core_config":{"core_ids":["0"]}}
  done = f32[8] async-done(start)
  done.1 = f32[8] async-done(start)
  ROOT tuple = (f32[8], f32[8]) tuple(done, done.1)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> branch_module,
                       ParseAndReturnUnverifiedModule(kBranchingHlo));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<HloModule>> branch_units,
      DecomposeHloModule(*branch_module, /*deduplicate_modules=*/true));
  EXPECT_THAT(ModuleNames(branch_units), ElementsAre("exp"));
  for (const std::unique_ptr<HloModule>& unit : branch_units) {
    EXPECT_THAT(VerifyExtractedModule(unit.get()), IsOk()) << unit->ToString();
  }
}

TEST_F(HloDecomposerTest, SparseCoreOffloadInsideWhileBodyIsExtractedAlone) {
  constexpr absl::string_view kWhileOffloadHlo = R"hlo(
HloModule main

program {
  p0 = f32[8] parameter(0)
  ROOT doubled = f32[8] add(p0, p0)
}, execution_thread="sparsecore"

wrapped {
  p0 = f32[8] parameter(0)
  ROOT caller = f32[8] custom-call(p0), custom_call_target="TODO(embedded_program)", called_computations={program}
}, execution_thread="sparsecore"

body {
  state = (f32[8]) parameter(0)
  val = f32[8] get-tuple-element(state), index=0
  start = ((f32[8]), f32[8]) async-start(val), async_execution_thread="sparsecore", calls=wrapped, backend_config={"sparse_core_config":{"core_ids":["0"]}}
  done = f32[8] async-done(start)
  ROOT next = (f32[8]) tuple(done)
}

cond {
  state = (f32[8]) parameter(0)
  ROOT c = pred[] constant(false)
}

ENTRY main {
  x = f32[8] parameter(0)
  init = (f32[8]) tuple(x)
  ROOT loop = (f32[8]) while(init), condition=cond, body=body
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kWhileOffloadHlo));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<HloModule>> units,
      DecomposeHloModule(*module, /*deduplicate_modules=*/true));
  EXPECT_THAT(ModuleNames(units), ElementsAre("done"));
  EXPECT_THAT(VerifyExtractedModule(units[0].get()), IsOk())
      << units[0]->ToString();
  EXPECT_THAT(units[0]->ToString(), Not(HasSubstr("TODO(embedded_program)")));
}

}  // namespace
}  // namespace xla
