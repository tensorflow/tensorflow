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

#include "xla/backends/cpu/transforms/xnn_graph_fusion.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/cpu/xnn_fusion.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace op = xla::testing::opcode_matchers;

namespace xla::cpu {
namespace {

using XnnGraphFusionTest = HloTestBase;

TEST_F(XnnGraphFusionTest, BasicFusion) {
  std::string hlo_string = R"(
HloModule FusionDemonstration

ENTRY entry {
   %param.0 = f32[1024,1024]{1,0} parameter(0)
   %param.1 = f32[1024,1024]{1,0} parameter(1)
   %add.0 = f32[1024,1024]{1,0} add(f32[1024,1024]{1,0} %param.0, f32[1024,1024]{1,0} %param.1)
   %sub.0 = f32[1024,1024]{1,0} subtract(f32[1024,1024]{1,0} %param.0, f32[1024,1024]{1,0} %param.1)
   ROOT %result = f32[1024,1024]{1,0} multiply(f32[1024,1024]{1,0} %add.0, f32[1024,1024]{1,0} %sub.0)
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, XnnGraphFusion().Run(module.get()));
  ASSERT_TRUE(changed);
  EXPECT_THAT(module.get()->entry_computation()->root_instruction(),
              op::Fusion());
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion);
  HloFusionInstruction* fusion = Cast<HloFusionInstruction>(root);
  TF_ASSERT_OK_AND_ASSIGN(auto backend_config,
                          fusion->backend_config<BackendConfig>());
  ASSERT_TRUE(backend_config.has_fusion_config());
  EXPECT_EQ(backend_config.fusion_config().kind(), kXnnFusionKind);
}

}  // namespace
}  // namespace xla::cpu
