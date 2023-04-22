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
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class GpuSpmdSinglePartitionTest : public GpuCodegenTest {};

TEST_F(GpuSpmdSinglePartitionTest, SpmdSinglePartition) {
  // Module with "Sharding" custom call and use_spmd_partitioning enabled.
  const char* const hlo_string = R"(
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

}  // namespace
}  // namespace gpu
}  // namespace xla
