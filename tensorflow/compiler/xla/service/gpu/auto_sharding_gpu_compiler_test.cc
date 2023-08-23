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

#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;

class AutoShardingTest : public HloTestBase {
 protected:
  const char* const dot_hlo_string_ = R"(
HloModule module
ENTRY matmul {
  parameter.1 = f32[32,64]{1,0} parameter(0)
  parameter.2 = f32[64,128]{1,0} parameter(1)
  ROOT root = f32[32,128]{1,0} dot(parameter.1, parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  std::unique_ptr<HloModule> CompileMatMul(bool use_autosharding,
                                       int num_partitions) {
    HloModuleConfig config;
    config.set_use_spmd_partitioning(true);
    config.set_use_auto_spmd_partitioning(use_autosharding);
    config.set_auto_spmd_partitioning_mesh_shape({num_partitions});
    config.set_num_partitions(num_partitions);

    auto module = ParseAndReturnVerifiedModule(dot_hlo_string_, config).value();
    std::unique_ptr<HloModule> compiled_module =
        backend()
            .compiler()
            ->RunHloPasses(module->Clone(), backend().default_stream_executor(),
                           /*device_allocator=*/nullptr)
            .value();
    VLOG(2) << compiled_module->ToString();
    return compiled_module;
  }
};

TEST_F(AutoShardingTest, MatMulWithAutosharding) {
  auto compiled_module = CompileMatMul(true, 4);
  auto* instruction = FindInstruction(compiled_module.get(), "param");
  VLOG(2) << instruction->ToString();
  EXPECT_THAT(instruction, op::Sharding("{devices=[4,1]0,1,2,3}"));
}

TEST_F(AutoShardingTest, MatMulWithoutAutosharding) {
  auto compiled_module = CompileMatMul(false, 4);
  auto* instruction = FindInstruction(compiled_module.get(), "param");
  VLOG(2) << instruction->ToString();
  EXPECT_THAT(instruction, op::Sharding("{replicated}"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
