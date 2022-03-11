/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/cpu/test_target_triple_helper.h"
#include "tensorflow/compiler/xla/service/cpu/tests/cpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace cpu {
namespace {

class CpuSpmdCompileTest : public CpuCodegenTest {};

TEST_F(CpuSpmdCompileTest, SinglePartition) {
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
  auto hlo_module =
      ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();

  // Verify that compilation succeeded.
  StatusOr<std::unique_ptr<Executable>> executable =
      CompileToExecutable(std::move(hlo_module));
  TF_EXPECT_OK(executable.status());
}

TEST_F(CpuSpmdCompileTest, DotSharding) {
  const char *const hlo_string = R"(
HloModule test

ENTRY main {
  %Arg_0.1 = s64[8,6,4]{2,1,0} parameter(0), sharding={devices=[1,2,2]0,1,2,3}
  %Arg_1.2 = s64[4,2]{1,0} parameter(1), sharding={devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}
  %dot.3 = s64[8,6,2]{2,1,0} dot(s64[8,6,4]{2,1,0} %Arg_0.1, s64[4,2]{1,0} %Arg_1.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  %tuple.4 = (s64[8,6,2]{2,1,0}) tuple(s64[8,6,2]{2,1,0} %dot.3)
  ROOT %get-tuple-element.5 = s64[8,6,2]{2,1,0} get-tuple-element((s64[8,6,2]{2,1,0}) %tuple.4), index=0, sharding={devices=[2,1,1,2]0,1,2,3 last_tile_dim_replicate}
})";

  HloModuleConfig config;
  config.set_use_spmd_partitioning(true);
  config.set_replica_count(4);
  config.set_num_partitions(4);
  config.set_debug_options(GetDebugOptionsFromFlags());
  auto module = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();

  CpuAotCompilationOptions options{
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"main",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  std::string filecheck_pattern = R"(
CHECK: call void @__xla_cpu_runtime_AllToAll
CHECK: call void @__xla_cpu_runtime_AllReduce
)";

  CompileAheadOfTimeAndVerifyIr(std::move(module), options, filecheck_pattern,
                                /*match_optimized_ir=*/true);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
