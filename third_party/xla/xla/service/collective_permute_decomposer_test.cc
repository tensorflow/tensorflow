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

#include "xla/service/collective_permute_decomposer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_parser.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

using ::testing::HasSubstr;
namespace op = xla::testing::opcode_matchers;
using CollectivePermuteDecomposerTest = HloTestBase;

TEST_F(CollectivePermuteDecomposerTest, WithCycleNotTransformed) {
  const absl::string_view kModuleStr = R"(
      HloModule test
      ENTRY test_computation {
        p = u32[] replica-id()
        ROOT cp = u32[] collective-permute(p), channel_id=1,
          source_target_pairs={{0,1}, {1,0}}
      }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteDecomposerTest, WithContextDataNotTransformed) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p = u32[] replica-id()
    ROOT cp = (u32[], u32[], u32[], u32[]) collective-permute(p), channel_id=1,
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteDecomposerTest, TransformedExplicitChannelId) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p = u32[] replica-id()
    ROOT cp = u32[] collective-permute(p), channel_id=1,
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}},
      metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py" source_line=35}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);

  auto check_metadata = [](const HloInstruction* inst) {
    EXPECT_EQ(inst->metadata().op_name(), "op1/op2/add");
    EXPECT_EQ(inst->metadata().source_file(), "foo/bar/mysource.py");
    EXPECT_EQ(inst->metadata().source_line(), 35);
  };

  auto check_not_pipelined = [](const HloInstruction* instr) {
    const FrontendAttributes& attributes = instr->frontend_attributes();
    EXPECT_EQ(attributes.map().end(),
              attributes.map().find(kSendRecvPipelineAttr));
  };

  HloInstruction* after_all = FindInstruction(module.get(), "after-all");
  HloInstruction* recv = FindInstruction(module.get(), "recv");
  EXPECT_EQ(recv->operand(0), after_all);
  EXPECT_EQ(recv->channel_id().value(), 1);
  EXPECT_THAT(
      recv->ToString(),
      HasSubstr(
          "_xla_send_recv_source_target_pairs=\"{{0,1},{1,2},{2,3},{3,4}}\""));
  check_metadata(recv);
  check_not_pipelined(recv);
  HloInstruction* recv_done = FindInstruction(module.get(), "recv-done");
  EXPECT_EQ(recv_done->operand(0), recv);

  HloInstruction* send = FindInstruction(module.get(), "send");
  EXPECT_EQ(send->operand(1), after_all);
  EXPECT_EQ(send->channel_id().value(), 1);
  EXPECT_THAT(
      send->ToString(),
      HasSubstr(
          "_xla_send_recv_source_target_pairs=\"{{0,1},{1,2},{2,3},{3,4}}\""));
  check_metadata(send);
  check_not_pipelined(send);
  HloInstruction* send_done = FindInstruction(module.get(), "send-done");
  EXPECT_EQ(send_done->operand(0), send);

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(recv_done, 0));
}

TEST_F(CollectivePermuteDecomposerTest, NotTransformedDefaultChannelId) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p = u32[] replica-id()
    ROOT cp = u32[] collective-permute(p),
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteDecomposerTest, ThresholdNotTransformed) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p = u32[] replica-id()
    ROOT cp = u32[] collective-permute(p), channel_id=1,
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}},
      metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py" source_line=35}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/8);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteDecomposerTest, Pipeline1) {
  const char* const kModuleStr = R"(
  HloModule module
  cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(2)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    recv-data = u32[2] collective-permute(send-data), channel_id=1,
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}},
      frontend_attributes={_xla_other_attribute="xyz"}

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* recv = FindInstruction(module.get(), "recv");
  EXPECT_EQ(recv->channel_id().value(), 1);
  EXPECT_THAT(
      recv->ToString(),
      HasSubstr(
          "_xla_send_recv_source_target_pairs=\"{{0,1},{1,2},{2,3},{3,4}}\""));
  EXPECT_THAT(recv->ToString(), HasSubstr("_xla_send_recv_pipeline=\"0\""));
  EXPECT_THAT(recv->ToString(), HasSubstr("_xla_other_attribute=\"xyz\""));
  HloInstruction* recv_done = FindInstruction(module.get(), "recv-done");
  EXPECT_THAT(recv_done->ToString(),
              HasSubstr("_xla_send_recv_pipeline=\"0\""));

  HloInstruction* send = FindInstruction(module.get(), "send");
  EXPECT_EQ(send->channel_id().value(), 1);
  EXPECT_THAT(
      send->ToString(),
      HasSubstr(
          "_xla_send_recv_source_target_pairs=\"{{0,1},{1,2},{2,3},{3,4}}\""));
  EXPECT_THAT(send->ToString(), HasSubstr("_xla_send_recv_pipeline=\"0\""));
  EXPECT_THAT(send->ToString(), HasSubstr("_xla_other_attribute=\"xyz\""));
  HloInstruction* send_done = FindInstruction(module.get(), "send-done");
  EXPECT_THAT(send_done->ToString(),
              HasSubstr("_xla_send_recv_pipeline=\"0\""));
}

TEST_F(CollectivePermuteDecomposerTest, ForwardPipeline2) {
  const char* const kModuleStr = R"(
  HloModule module
  cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(2)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    recv-data.0 = u32[2] collective-permute(send-data), channel_id=1,
      source_target_pairs={{3,0}}

    recv-data.1 = u32[2] collective-permute(send-data), channel_id=2,
      source_target_pairs={{0,1}, {1,2}, {2,3}}

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=EQ
    compare = pred[2] broadcast(compare0), dimensions={}
    recv-data = u32[2] select(compare, recv-data.0, recv-data.1)

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* recv = FindInstruction(module.get(), "recv");
  EXPECT_EQ(recv->channel_id().value(), 1);
  EXPECT_THAT(recv->ToString(),
              HasSubstr("_xla_send_recv_source_target_pairs=\"{{3,0}}\""));
  EXPECT_THAT(recv->ToString(), HasSubstr("_xla_send_recv_pipeline=\"0\""));
  HloInstruction* send = FindInstruction(module.get(), "send");
  EXPECT_THAT(send->ToString(),
              HasSubstr("_xla_send_recv_source_target_pairs=\"{{3,0}}\""));
  EXPECT_THAT(send->ToString(), HasSubstr("_xla_send_recv_pipeline=\"0\""));

  HloInstruction* recv1 = FindInstruction(module.get(), "recv.1");
  EXPECT_EQ(recv1->channel_id().value(), 2);
  EXPECT_THAT(
      recv1->ToString(),
      HasSubstr("_xla_send_recv_source_target_pairs=\"{{0,1},{1,2},{2,3}}\""));
  EXPECT_THAT(recv1->ToString(), HasSubstr("_xla_send_recv_pipeline=\"1\""));
  HloInstruction* recv_done1 = FindInstruction(module.get(), "recv-done.1");
  EXPECT_THAT(recv_done1->ToString(),
              HasSubstr("_xla_send_recv_pipeline=\"1\""));
  HloInstruction* send1 = FindInstruction(module.get(), "send.1");
  EXPECT_THAT(
      send1->ToString(),
      HasSubstr("_xla_send_recv_source_target_pairs=\"{{0,1},{1,2},{2,3}}\""));
  EXPECT_THAT(send1->ToString(), HasSubstr("_xla_send_recv_pipeline=\"1\""));
  HloInstruction* send_done1 = FindInstruction(module.get(), "send-done.1");
  EXPECT_THAT(send_done1->ToString(),
              HasSubstr("_xla_send_recv_pipeline=\"1\""));
}

TEST_F(CollectivePermuteDecomposerTest, BackwardPipeline2) {
  const char* const kModuleStr = R"(
  HloModule module
  cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(2)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    recv-data.0 = u32[2] collective-permute(send-data), channel_id=1,
      source_target_pairs={{1,0},{2,1},{3,2}}

    recv-data.1 = u32[2] collective-permute(send-data), channel_id=2,
      source_target_pairs={{0,3}}

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=NE
    compare = pred[2] broadcast(compare0), dimensions={}
    recv-data = u32[2] select(compare, recv-data.0, recv-data.1)

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* recv = FindInstruction(module.get(), "recv");
  EXPECT_EQ(recv->channel_id().value(), 1);
  EXPECT_THAT(
      recv->ToString(),
      HasSubstr("_xla_send_recv_source_target_pairs=\"{{1,0},{2,1},{3,2}}\""));
  EXPECT_THAT(recv->ToString(), HasSubstr("_xla_send_recv_pipeline=\"1\""));
  HloInstruction* send = FindInstruction(module.get(), "send");
  EXPECT_THAT(
      send->ToString(),
      HasSubstr("_xla_send_recv_source_target_pairs=\"{{1,0},{2,1},{3,2}}\""));
  EXPECT_THAT(send->ToString(), HasSubstr("_xla_send_recv_pipeline=\"1\""));

  HloInstruction* recv1 = FindInstruction(module.get(), "recv.1");
  EXPECT_EQ(recv1->channel_id().value(), 2);
  EXPECT_THAT(recv1->ToString(),
              HasSubstr("_xla_send_recv_source_target_pairs=\"{{0,3}}\""));
  EXPECT_THAT(recv1->ToString(), HasSubstr("_xla_send_recv_pipeline=\"0\""));
  HloInstruction* send1 = FindInstruction(module.get(), "send.1");
  EXPECT_THAT(send1->ToString(),
              HasSubstr("_xla_send_recv_source_target_pairs=\"{{0,3}}\""));
  EXPECT_THAT(send1->ToString(), HasSubstr("_xla_send_recv_pipeline=\"0\""));
}

}  // namespace
}  // namespace xla
