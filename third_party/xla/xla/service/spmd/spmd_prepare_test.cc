/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/spmd/spmd_prepare.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace spmd {
namespace {

namespace op = xla::testing::opcode_matchers;

class SpmdPrepareTest : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module, int64_t distance_threshold = 100) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(
                                         hlo_module, GetModuleConfigForTest()));
    HloPassPipeline pipeline("spmd-prepare");
    pipeline.AddPass<SpmdPrepare>();
    TF_RETURN_IF_ERROR(pipeline.Run(module.get()).status());
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }
};

TEST_F(SpmdPrepareTest, ScatterParallelIndexSplit) {
  absl::string_view hlo_string = R"(
HloModule module

region_157.5067 {
  Arg_0.5068 = f32[] parameter(0)
  Arg_1.5069 = f32[] parameter(1)
  ROOT add.5070 = f32[] add(Arg_0.5068, Arg_1.5069)
}

ENTRY entry {
  p0 = f32[16,1000,2000]{2,1,0} parameter(0), sharding={devices=[4,2,1]<=[8]}
  p1 = f32[16,1000,2000]{2,1,0} parameter(1), sharding={devices=[4,2,1]<=[8]}
  p2 = s32[16,1000,64,1]{3,2,1,0} parameter(2), sharding={devices=[4,2,1,1]<=[8]}
  p3 = f32[16,1000,64]{2,1,0} parameter(3), sharding={devices=[4,2,1]<=[8]}
  p4 = f32[16,1000,64]{2,1,0} parameter(4), sharding={devices=[4,2,1]<=[8]}
  iota.0 = s32[16,1000,64,1]{3,2,1,0} iota(), iota_dimension=0, sharding={devices=[4,2,1,1]<=[8]}
  iota.1 = s32[16,1000,64,1]{3,2,1,0} iota(), iota_dimension=1, sharding={devices=[4,2,1,1]<=[8]}
  iota.2 = s32[16,1000,64,1]{3,2,1,0} iota(), iota_dimension=0, sharding={devices=[4,2,1,1]<=[8]}
  iota.3 = s32[16,1000,64,1]{3,2,1,0} iota(), iota_dimension=1, sharding={devices=[4,2,1,1]<=[8]}
  concatenate.0 = s32[16,1000,64,3]{3,2,1,0} concatenate(iota.0, iota.1, p2), dimensions={3}, sharding={devices=[4,2,1,1]<=[8]}
  concatenate.1 = s32[16,1000,64,3]{3,2,1,0} concatenate(iota.2, iota.3, p2), dimensions={3}, sharding={devices=[4,2,1,1]<=[8]}
  concatenate.130 = s32[32,1000,64,3]{3,2,1,0} concatenate(concatenate.0, concatenate.1), dimensions={0}, sharding={devices=[4,2,1,1]<=[8]}
  concatenate.131 = f32[32,1000,64]{2,1,0} concatenate(p3, p4), dimensions={0}, sharding={devices=[4,2,1]<=[8]}
  add.190 = f32[16,1000,2000]{2,1,0} add(p0, p1), sharding={devices=[4,2,1]<=[8]}
  ROOT scatter.2 = f32[16,1000,2000]{2,1,0} scatter(add.190, concatenate.130, concatenate.131), update_window_dims={}, inserted_window_dims={0,1,2}, scatter_dims_to_operand_dims={0,1,2}, index_vector_dim=3, to_apply=region_157.5067, sharding={devices=[4,2,1]<=[8]}
})";

  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = std::move(module_status).value();
  HloInstruction* root = module->entry_computation()->root_instruction();
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_THAT(
      root,
      op::Scatter(
          op::Scatter(op::Add(),
                      op::Concatenate(op::Iota(), op::Iota(), op::Parameter()),
                      op::Parameter()),
          op::Concatenate(op::Iota(), op::Iota(), op::Parameter()),
          op::Parameter()));
}

}  // namespace
}  // namespace spmd
}  // namespace xla
