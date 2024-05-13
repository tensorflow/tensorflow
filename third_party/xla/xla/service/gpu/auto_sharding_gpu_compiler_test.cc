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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

using ::testing::Conditional;

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
  std::unique_ptr<HloModule> compiled_module = CompileMatMul(true, 4);
  const HloInstruction* parameter1 =
      compiled_module->entry_computation()->parameter_instruction(0);
  const HloInstruction* parameter2 =
      compiled_module->entry_computation()->parameter_instruction(1);
  bool is_parameter1_replicated = ShapeUtil::Equal(
      parameter1->shape(), ShapeUtil::MakeShape(PrimitiveType::F32, {32, 64}));
  bool is_parameter2_replicated = ShapeUtil::Equal(
      parameter2->shape(), ShapeUtil::MakeShape(PrimitiveType::F32, {64, 128}));

  // Check that at least one of the parameters is sharded, thereby telling us
  // that the dot is as well.
  VLOG(2) << parameter1->ToString();
  EXPECT_THAT(
      parameter1,
      Conditional(
          is_parameter2_replicated,
          AnyOf(GmockMatch(m::Op().WithShape(PrimitiveType::F32, {8, 64})),
                GmockMatch(m::Op().WithShape(PrimitiveType::F32, {32, 16}))),
          GmockMatch(m::Op().WithShape(PrimitiveType::F32, {32, 64}))));

  VLOG(2) << parameter2->ToString();
  EXPECT_THAT(
      parameter2,
      Conditional(
          is_parameter1_replicated,
          AnyOf(GmockMatch(m::Op().WithShape(PrimitiveType::F32, {16, 128})),
                GmockMatch(m::Op().WithShape(PrimitiveType::F32, {64, 32}))),
          GmockMatch(m::Op().WithShape(PrimitiveType::F32, {64, 128}))));
}

TEST_F(AutoShardingTest, MatMulWithoutAutosharding) {
  auto compiled_module = CompileMatMul(false, 4);
  auto* instruction =
      compiled_module->entry_computation()->parameter_instruction(0);
  VLOG(2) << instruction->ToString();
  EXPECT_THAT(instruction, GmockMatch(m::Op().WithSharding("{replicated}")));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
