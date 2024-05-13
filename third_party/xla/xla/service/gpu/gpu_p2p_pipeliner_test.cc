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

#include "xla/service/gpu/gpu_p2p_pipeliner.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/statusor.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

class GpuP2PPipelinerTest : public HloTestBase {
 public:
  GpuP2PPipelinerTest() {
    const int64_t kNumReplicas = 1;
    const int64_t kNumComputations = 4;
    config_ = GetModuleConfigForTest(/*replica_count=*/kNumReplicas,
                                     /*num_partitions=*/kNumComputations);
  }

  absl::StatusOr<bool> RunOptimizer(HloModule* module) {
    HloPassPipeline pipeline("optimizer");
    pipeline.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                                  /*allow_mixed_precision=*/false);
    AddP2PPipeliner(pipeline);
    pipeline.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                                  /*allow_mixed_precision=*/false);
    return pipeline.Run(module);
  }

 protected:
  HloModuleConfig config_;
};

TEST_F(GpuP2PPipelinerTest,
       TransformRecvSendBackwardsWithMetaDataPostProcessing) {
  const char* kHloStr = R"(
  HloModule module
  cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(10)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    after-all.0 = token[] after-all()
    recv.0 = (u32[2], u32[], token[]) recv(after-all.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{1,0}}",
        _xla_send_recv_pipeline="0",
        _xla_send_recv_validation="{{1,7}}"
      }
    after-all.0.s = token[] after-all()
    send.0 = (u32[2], u32[], token[]) send(send-data, after-all.0.s),
      channel_id=1, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{1,0}}",
        _xla_send_recv_pipeline="0",
        _xla_send_recv_validation="{{1,7}}"
      }
    recv-done.0 = (u32[2], token[]) recv-done(recv.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }, control-predecessors={send.0}
    recv-data = u32[2] get-tuple-element(recv-done.0), index=0

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

    send-done.0 = token[] send-done(send.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    ROOT result = (u32[], u32[2]) tuple(new_count, s)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    r = u32[] replica-id()
    a = u32[] add(c1, r)
    init = u32[2] broadcast(a), dimensions={}
    while_init = (u32[], u32[2]) tuple(c0, init)
    while_result = (u32[], u32[2]) while(while_init), body=body, condition=cond
    ROOT result = u32[2] get-tuple-element(while_result), index=1
  })";

  auto module = ParseAndReturnUnverifiedModule(kHloStr, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get()).value());
  XLA_VLOG_LINES(10, module->ToString());
  auto while_op = FindInstruction(module.get(), "while");
  EXPECT_EQ(while_op->opcode(), HloOpcode::kWhile);
  EXPECT_EQ(while_op->shape().tuple_shapes().size(), 5);
  auto recv1 =
      DynCast<HloRecvInstruction>(FindInstruction(module.get(), "recv.1"));
  EXPECT_NE(recv1, nullptr);
  auto recv2 =
      DynCast<HloRecvInstruction>(FindInstruction(module.get(), "recv.2"));
  EXPECT_NE(recv2, nullptr);
  EXPECT_EQ(recv1->channel_id(), recv2->channel_id());

  auto send1 =
      DynCast<HloSendInstruction>(FindInstruction(module.get(), "send.1"));
  EXPECT_NE(send1, nullptr);
  auto send2 =
      DynCast<HloSendInstruction>(FindInstruction(module.get(), "send.2"));
  EXPECT_NE(send2, nullptr);
  EXPECT_EQ(send1->channel_id(), send2->channel_id());

  const char* kPeeledAttr = "_xla_send_recv_validation=\"invalid\"";
  const char* kRotatedAttr = "_xla_send_recv_validation=\"{{0,6}}\"";
  EXPECT_THAT(send1->ToString(), ::testing::HasSubstr(kPeeledAttr));
  EXPECT_THAT(recv1->ToString(), ::testing::HasSubstr(kPeeledAttr));
  EXPECT_THAT(send2->ToString(), ::testing::HasSubstr(kRotatedAttr));
  EXPECT_THAT(recv2->ToString(), ::testing::HasSubstr(kRotatedAttr));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
