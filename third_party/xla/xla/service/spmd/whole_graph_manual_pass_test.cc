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

#include "xla/service/spmd/whole_graph_manual_pass.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace spmd {
namespace {

using ::testing::_;
using ::testing::AllOf;
namespace op = xla::testing::opcode_matchers;

class WholeGraphManualPassTest : public HloTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module) {
    TF_ASSIGN_OR_RETURN(
        auto module,
        ParseAndReturnVerifiedModule(
            hlo_module,
            GetModuleConfigForTest(/*replica_count=*/1, /*num_partitions=*/4)));
    HloPassPipeline pipeline("whole-graph-manual-pass");
    pipeline.AddPass<WholeGraphManualPass>();
    TF_RETURN_IF_ERROR(pipeline.Run(module.get()).status());
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }
  absl::Status RunPassOnModule(HloModule* module,
                               int64_t distance_threshold = 100) {
    HloPassPipeline pipeline("all-gather-cse");
    pipeline.AddPass<WholeGraphManualPass>();
    TF_RETURN_IF_ERROR(pipeline.Run(module).status());
    return absl::OkStatus();
  }
};

TEST_F(WholeGraphManualPassTest, SimpleRewrite) {
  absl::string_view hlo_string = R"(
HloModule module

 body {
   p_body = (f32[2], f32[2], f32[2], s32[]) parameter(0)
   val.0 = f32[2] get-tuple-element(p_body), index=0
   val.1 = f32[2] get-tuple-element(p_body), index=1
   add = f32[2] add(val.0, val.1)
   const = s32[] constant(-1)
   ROOT root = (f32[2], f32[2], f32[2], s32[]) tuple(val.0, val.1, add, const)
 }

 condition {
   p_cond = (f32[2], f32[2], f32[2], s32[]) parameter(0)
   gte = s32[] get-tuple-element(p_cond), index=3
   const = s32[] constant(42)
   ROOT result = pred[] compare(gte, const), direction=EQ
 }

ENTRY entry {
  param0 = (s32[8]{0}, s32[8]{0}) parameter(0)
  g1 = s32[8]{0} get-tuple-element(param0), index=0
  g2 = s32[8]{0} get-tuple-element(param0), index=1
  resh1 = s32[1,8]{1,0} reshape(g1)
  resh2 = s32[1,8]{1,0} reshape(g2)
  param1 = f32[2] parameter(1)
  param2 = s32[] parameter(2)
  while_init = (f32[2], f32[2], f32[2], s32[]) tuple(param1, param1, param1, param2)
  while = (f32[2], f32[2], f32[2], s32[]) while(while_init), condition=condition, body=body
  g3 = f32[2] get-tuple-element(while), index=0
  ROOT t = (s32[1,8]{1,0}, s32[1,8]{1,0}, f32[2]) tuple(resh1, resh2, g3), sharding={{devices=[1,4]0,1,2,3}, {devices=[1,4]0,1,2,3}, {replicated}}
})";
  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = std::move(module_status).value();
  for (auto* i : module->entry_computation()->instructions()) {
    if (module->entry_computation()->root_instruction() == i) {
      EXPECT_THAT(i, op::Sharding("{{manual}, "
                                  "{manual}, {manual}}"));
    } else if (i->opcode() == HloOpcode::kParameter) {
      EXPECT_THAT(i, AnyOf(op::Sharding("{manual}"),
                           op::Sharding("{{manual},{manual}}")));
    }
  }
}

TEST_F(WholeGraphManualPassTest, SimplePartitionIdCollectives) {
  absl::string_view hlo_string = R"(
HloModule module

 body {
   p_body = (f32[2], f32[2], f32[2], s32[]) parameter(0)
   val.0 = f32[2] get-tuple-element(p_body), index=0
   val.1 = f32[2] get-tuple-element(p_body), index=1
   t = token[] after-all()
   p = u32[] partition-id()
   ag = f32[8] all-gather(val.1), dimensions={0}, replica_groups={{0,1,2,3}}, use_global_device_ids=true, channel_id=1
   s = (f32[8], s32[], token[]) send(ag, t), channel_id=2
   sd = token[] send-done(s), channel_id=2
   add = f32[2] add(val.0, val.1)
   const = s32[] constant(-1)
   ROOT root = (f32[2], f32[2], f32[2], s32[]) tuple(val.0, val.1, add, const)
 }

 condition {
   p_cond = (f32[2], f32[2], f32[2], s32[]) parameter(0)
   gte = s32[] get-tuple-element(p_cond), index=3
   const = s32[] constant(42)
   ROOT result = pred[] compare(gte, const), direction=EQ
 }

ENTRY entry {
  param0 = (s32[8]{0}, s32[8]{0}) parameter(0)
  g1 = s32[8]{0} get-tuple-element(param0), index=0
  g2 = s32[8]{0} get-tuple-element(param0), index=1
  resh1 = s32[1,8]{1,0} reshape(g1)
  resh2 = s32[1,8]{1,0} reshape(g2)
  param1 = f32[2] parameter(1)
  param2 = s32[] parameter(2)
  while_init = (f32[2], f32[2], f32[2], s32[]) tuple(param1, param1, param1, param2)
  while = (f32[2], f32[2], f32[2], s32[]) while(while_init), condition=condition, body=body
  g3 = f32[2] get-tuple-element(while), index=0
  ROOT t = (s32[1,8]{1,0}, s32[1,8]{1,0}, f32[2]) tuple(resh1, resh2, g3), sharding={{devices=[1,4]0,1,2,3}, {devices=[1,4]0,1,2,3}, {replicated}}
})";
  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = std::move(module_status).value();
  for (auto* c : module->computations()) {
    for (auto* i : c->instructions()) {
      if (c->root_instruction() == i) {
        EXPECT_THAT(
            i, AnyOf(op::Sharding("{manual}"),
                     op::Sharding("{{manual},{manual},{manual}}"),
                     op::Sharding("{{manual}, {manual}, {manual}, {manual}}")));
      } else if (i->opcode() == HloOpcode::kParameter) {
        EXPECT_THAT(
            i,
            AnyOf(op::Sharding("{manual}"), op::Sharding("{{manual},{manual}}"),
                  op::Sharding("{{manual},{manual},{manual},{manual}}")));
      } else if (i->opcode() == HloOpcode::kPartitionId ||
                 i->opcode() == HloOpcode::kAllGather ||
                 i->opcode() == HloOpcode::kSendDone) {
        EXPECT_THAT(i, op::Sharding("{manual}"));
      } else if (i->opcode() == HloOpcode::kSend) {
        EXPECT_THAT(i, op::Sharding("{{manual},{manual},{manual}}"));
      } else {
        EXPECT_FALSE(i->has_sharding());
      }
    }
  }
}

}  // namespace
}  // namespace spmd
}  // namespace xla
