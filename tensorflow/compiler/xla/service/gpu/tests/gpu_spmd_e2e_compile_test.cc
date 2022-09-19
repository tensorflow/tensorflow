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

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class GpuSpmdE2ECompileTest : public GpuCodegenTest {};

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
  StatusOr<std::unique_ptr<Executable>> executable =
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
  config.set_debug_options(GetDebugOptionsFromFlags());
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

}  // namespace
}  // namespace gpu
}  // namespace xla
