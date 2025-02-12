/* Copyright 2021 The OpenXLA Authors.

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
#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class GpuSpmdE2ECompileTest : public GpuCodegenTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }
};

TEST_F(GpuSpmdE2ECompileTest, SinglePartition) {
  // Module with "Sharding" custom call and use_spmd_partitioning enabled.
  const char *const hlo_string = R"(
HloModule module

ENTRY entry {
 %parameter.3379 = f32[1,1]{1,0} parameter(0)
 %custom-call.3380 = f32[1,1]{1,0} custom-call(f32[1,1]{1,0} %parameter.3379),
   custom_call_target="Sharding", sharding={replicated}
 ROOT %reshape.6032 = f32[] reshape(f32[1,1]{1,0} %custom-call.3380)
})";

  HloModuleConfig config;
  config.set_use_spmd_partitioning(true);
  auto hlo_module = ParseAndReturnVerifiedModule(hlo_string, config).value();

  // Verify that compilation succeeded.
  absl::StatusOr<std::unique_ptr<Executable>> executable =
      CompileToExecutable(std::move(hlo_module));
  TF_EXPECT_OK(executable.status());
}

TEST_F(GpuSpmdE2ECompileTest, DotSharding) {
  const char *const hlo_string = R"(
HloModule test

ENTRY main {
  %x = bf16[1,1024,12288] parameter(0), sharding={replicated}
  %W = bf16[3,12288,320,128] parameter(1), sharding={devices=[1,1,2,1]0,1}
  %T = bf16[3,320,128,1,1024] parameter(2), sharding={devices=[1,2,1,1,1]0,1}
  // SPMD partitioning should propagate sharding=[1,2,1,1,1] for this dot. And post-partitioning there should not
  // be any collectives needed to exchange data between the 2 partitions.
  %dot = bf16[3,320,128,1,1024] dot(bf16[3,12288,320,128] %W, bf16[1,1024,12288] %x), lhs_contracting_dims={1}, rhs_contracting_dims={2}
  ROOT %r = bf16[3,320,128,1,1024] add(bf16[3,320,128,1,1024] %dot, bf16[3,320,128,1,1024] %T), sharding={devices=[1,2,1,1,1]0,1}
})";

  HloModuleConfig config;
  config.set_use_spmd_partitioning(true);
  config.set_num_partitions(2);
  config.set_debug_options(GetDebugOptionsForTest());
  auto hlo_module = ParseAndReturnVerifiedModule(hlo_string, config).value();

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(hlo_module)));

  // Validate that no collective communication operations are generated in this
  // module.
  const bool has_collective_ops = absl::c_any_of(
      optimized_module->entry_computation()->instructions(),
      [](const HloInstruction *inst) {
        return hlo_query::IsCollectiveCommunicationOp(inst->opcode());
      });
  EXPECT_FALSE(has_collective_ops);
}

TEST_F(GpuSpmdE2ECompileTest, CollectivesScheduleLinearizerNoDeps) {
  // Setup the module such that we will need to generate > 1 collective for
  // sharding
  const char *const hlo_string = R"(
HloModule test

ENTRY main {
  %x = f32[1024,1024] parameter(0), sharding={devices=[2,2]0,1,2,3}
  %y = f32[1024,1024] parameter(1), sharding={devices=[2,2]0,1,2,3}
  %dot1 = f32[1024,1024] dot(%x, %y), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={replicated}
  %dot2 = f32[1024,1024] dot(%dot1, %y), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={devices=[2,2]0,1,2,3}
  ROOT r = copy(%dot2), sharding={replicated}
})";

  HloModuleConfig config;
  config.set_use_spmd_partitioning(true);
  config.set_num_partitions(4);
  config.set_debug_options(GetDebugOptionsForTest());
  auto hlo_module = ParseAndReturnVerifiedModule(hlo_string, config).value();

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(hlo_module)));
  // Verify that none of the collective operations generated have control
  // dependencies.
  const HloComputation *entry = optimized_module->entry_computation();
  for (const HloInstruction *instr : entry->instructions()) {
    if (!hlo_query::IsCollectiveCommunicationOp(instr->opcode())) {
      continue;
    }
    EXPECT_TRUE(instr->control_predecessors().empty());
    EXPECT_TRUE(instr->control_successors().empty());
  }
}

TEST_F(GpuSpmdE2ECompileTest, CollectivesScheduleLinearizerDepsWithConv) {
  // Setup the module such that we will need to generate > 1 collective for
  // sharding, and verify that linearizer inserts control deps as there are
  // convolutions that can be auto tuned.
  const char *const hlo_string = R"(
HloModule test

ENTRY main {
  %x = f32[1024,1024] parameter(0), sharding={devices=[2,2]0,1,2,3}
  %y = f32[1024,1024] parameter(1), sharding={devices=[2,2]0,1,2,3}
  %dot1 = f32[1024,1024] dot(%x, %y), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={replicated}
  %dot2 = f32[1024,1024] dot(%dot1, %y), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={devices=[2,2]0,1,2,3}
  %p0 = f32[8,5,5,1] parameter(2), sharding={replicated}
  %p1 = f32[3,3,1,32] parameter(3), sharding={replicated}
  %conv = f32[8,5,5,32] convolution(%p0, %p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  ROOT %t = (f32[1024,1024], f32[8,5,5,32]) tuple(%dot2, %conv), sharding={replicated}
})";

  HloModuleConfig config;
  config.set_use_spmd_partitioning(true);
  config.set_num_partitions(4);
  config.set_debug_options(GetDebugOptionsFromFlags());
  auto hlo_module = ParseAndReturnVerifiedModule(hlo_string, config).value();

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(hlo_module)));
  // Verify that control dependencies are inserted for collectives.
  bool has_control_deps = false;
  const HloComputation *entry = optimized_module->entry_computation();
  for (const HloInstruction *instr : entry->instructions()) {
    if (!hlo_query::IsCollectiveCommunicationOp(instr->opcode())) {
      continue;
    }
    has_control_deps |= !instr->control_predecessors().empty() ||
                        !instr->control_successors().empty();
  }
  EXPECT_TRUE(has_control_deps);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
