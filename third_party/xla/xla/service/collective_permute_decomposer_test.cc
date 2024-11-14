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
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
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
          "_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}"));
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
          "_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}"));
  check_metadata(send);
  check_not_pipelined(send);
  HloInstruction* send_done = FindInstruction(module.get(), "send-done");
  EXPECT_EQ(send_done->operand(0), send);

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(recv_done, 0));
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

    recv-data = u32[2] collective-permute(send-data), {{CHANNEL_ID}}
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

  const char* kFileCheckStr = R"(
  // CHECK-LABEL: %body
  // CHECK-DAG:     %[[P1:.+]] = parameter(0)
  // CHECK-DAG:     %[[COUNT:.+]] = get-tuple-element(%[[P1]]), index=0
  // CHECK-DAG:     %[[C1:.+]] = constant(1)
  // CHECK-DAG:     %[[NEW_COUNT:.+]] = add(%[[COUNT]], %[[C1]])
  // CHECK-DAG:     %[[R:.+]] = broadcast(%[[C1]]), dimensions={}
  // CHECK-DAG:     %[[SEND_DATA:.+]] = get-tuple-element(%[[P1]]), index=1
  // CHECK-DAG:     %[[AFTER_ALL:.+]] = after-all()
  // CHECK:         %[[SEND:.+]] = send(%[[SEND_DATA]], %[[AFTER_ALL]]), {{CHANNEL_ID}} 
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_other_attribute="xyz",_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}}
  // CHECK:         %[[RECV:.+]] = recv(%[[AFTER_ALL]]), {{CHANNEL_ID}} 
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_other_attribute="xyz",_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}}
  // CHECK-DAG:     %[[RECV_DONE:.+]] = recv-done(%[[RECV]]), {{CHANNEL_ID}} frontend_attributes={_xla_send_recv_pipeline="0"}, control-predecessors={%[[SEND]]}
  // CHECK-DAG:     %[[GTE:.+]] = get-tuple-element(%[[RECV_DONE]]), index=0
  // CHECK-DAG:     %[[S:.+]] = add(%[[R]], %[[GTE]])
  // CHECK-DAG:     ROOT %{{.+}} = tuple(%[[NEW_COUNT]], %[[S]])
  // CHECK-DAG:     %{{.+}} = send-done(%[[SEND]]), {{CHANNEL_ID}} frontend_attributes={_xla_send_recv_pipeline="0"}
  })";

  std::string channel_id_hlo =
      absl::StrReplaceAll(kModuleStr, {{"{{CHANNEL_ID}}", "channel_id=1, "}});
  std::string channel_id_filecheck = absl::StrReplaceAll(
      kFileCheckStr, {{"{{CHANNEL_ID}}", "channel_id=1, "}});
  std::string no_channel_id_hlo =
      absl::StrReplaceAll(kModuleStr, {{"{{CHANNEL_ID}}", ""}});
  std::string no_channel_id_filecheck = absl::StrReplaceAll(
      kFileCheckStr, {{"{{CHANNEL_ID}}", "channel_id=0, "}});

  auto check_module = [](std::string module_str, std::string filecheck_str) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnUnverifiedModule((module_str)));
    CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
    TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
    EXPECT_TRUE(changed);

    TF_ASSERT_OK_AND_ASSIGN(
        bool filecheck_status,
        RunFileCheck(/*input=*/module->ToString(
                         /*options=*/HloPrintOptions{}
                             .set_print_operand_shape(false)
                             .set_print_result_shape(false)),
                     /*pattern=*/filecheck_str));
    EXPECT_TRUE(filecheck_status);
  };
  check_module(channel_id_hlo, channel_id_filecheck);
  check_module(no_channel_id_hlo, no_channel_id_filecheck);
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

    recv-data.0 = u32[2] collective-permute(send-data), {{CHANNEL_ID_1}}
      source_target_pairs={{3,0}}

    recv-data.1 = u32[2] collective-permute(send-data), {{CHANNEL_ID_2}}
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

  const char* kFilecheckStr = R"(
  // CHECK-LABEL: %body
  // CHECK-DAG:     %[[p1:.+]] = parameter(0)
  // CHECK-DAG:     %[[count:.+]] = get-tuple-element(%[[p1]]), index=0
  // CHECK-DAG:     %[[c1:.+]] = constant(1)
  // CHECK-DAG:     %[[new_count:.+]] = add(%[[count]], %[[c1]])
  // CHECK-DAG:     %[[r:.+]] = broadcast(%[[c1]]), dimensions={}
  // CHECK-DAG:     %[[replica:.+]] = replica-id()
  // CHECK-DAG:     %[[c0:.+]] = constant(0)
  // CHECK-DAG:     %[[compare0:.+]] = compare(%[[replica]], %[[c0]]), direction=EQ
  // CHECK-DAG:     %[[compare:.+]] = broadcast(%[[compare0]]), dimensions={}
  // CHECK-DAG:     %[[send_data:.+]] = get-tuple-element(%[[p1]]), index=1
  // CHECK-DAG:     %[[after_all:.+]] = after-all()
  // CHECK:         %[[send:.+]] = send(%[[send_data]], %[[after_all]]), {{CHANNEL_ID_1}} 
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{3,0}}}
  // CHECK:         %[[recv:.+]] = recv(%[[after_all]]), {{CHANNEL_ID_1}} 
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{3,0}}}
  // CHECK-DAG:     %[[recv_done:.+]] = recv-done(%[[recv]]), {{CHANNEL_ID_1}} frontend_attributes={_xla_send_recv_pipeline="0"}, control-predecessors={%[[send]]}
  // CHECK-DAG:     %[[gte:.+]] = get-tuple-element(%[[recv_done]]), index=0
  // CHECK-DAG:     %[[after_all1:.+]] = after-all()
  // CHECK:         %[[send1:.+]] = send(%[[send_data]], %[[after_all1]]), {{CHANNEL_ID_2}} 
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
  // CHECK:         %[[recv1:.+]] = recv(%[[after_all1]]), {{CHANNEL_ID_2}} 
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
  // CHECK-DAG:     %[[recv_done1:.+]] = recv-done(%[[recv1]]), {{CHANNEL_ID_2}} frontend_attributes={_xla_send_recv_pipeline="1"}, control-predecessors={%[[send1]]}
  // CHECK-DAG:     %[[gte1:.+]] = get-tuple-element(%[[recv_done1]]), index=0
  // CHECK-DAG:     %[[recv_data2:.+]] = select(%[[compare]], %[[gte]], %[[gte1]])
  // CHECK-DAG:     %[[s:.+]] = add(%[[r]], %[[recv_data2]])
  // CHECK-DAG:     ROOT %{{.+}} = tuple(%[[new_count]], %[[s]])
  // CHECK-DAG:     %{{.+}} = send-done(%[[send]]), {{CHANNEL_ID_1}} frontend_attributes={_xla_send_recv_pipeline="0"}
  // CHECK-DAG:     %{{.+}} = send-done(%[[send1]]), {{CHANNEL_ID_2}} frontend_attributes={_xla_send_recv_pipeline="1"}
  })";

  std::string channel_id_hlo =
      absl::StrReplaceAll(kModuleStr, {{"{{CHANNEL_ID_1}}", "channel_id=1, "},
                                       {"{{CHANNEL_ID_2}}", "channel_id=2, "}});
  std::string channel_id_filecheck = absl::StrReplaceAll(
      kFilecheckStr, {{"{{CHANNEL_ID_1}}", "channel_id=1, "},
                      {"{{CHANNEL_ID_2}}", "channel_id=2, "}});
  std::string no_channel_id_hlo = absl::StrReplaceAll(
      kModuleStr, {{"{{CHANNEL_ID_1}}", ""}, {"{{CHANNEL_ID_2}}", ""}});
  std::string no_channel_id_filecheck = absl::StrReplaceAll(
      kFilecheckStr, {{"{{CHANNEL_ID_1}}", "channel_id=0, "},
                      {"{{CHANNEL_ID_2}}", "channel_id=0, "}});

  auto check_module = [](const std::string module_str,
                         std::string filecheck_str) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnUnverifiedModule((module_str)));
    CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
    TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
    EXPECT_TRUE(changed);
    TF_ASSERT_OK_AND_ASSIGN(
        bool filecheck_status,
        RunFileCheck(/*input=*/module->ToString(
                         /*options=*/HloPrintOptions{}
                             .set_print_operand_shape(false)
                             .set_print_result_shape(false)),
                     /*pattern=*/filecheck_str));
    EXPECT_TRUE(filecheck_status);
  };

  check_module(channel_id_hlo, channel_id_filecheck);
  check_module(no_channel_id_hlo, no_channel_id_filecheck);
}

TEST_F(CollectivePermuteDecomposerTest, ForwardPipelineWithMatmul) {
  // The HLO module below is generated by passing the HLO in
  // CollectiveOpsTest.CollectivePermute_CircularPipelinePreOptimization through
  // the collective_permute_cycle_decomposer.transformation.
  const char* const kModuleStr = R"(
  HloModule test

  while_body {
    inputs = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(inputs), index=0
    iter_increment = u32[] constant(1)
    next_iter = u32[] add(iter, iter_increment)
    partition-id = u32[] partition-id()
    zero = u32[] constant(0)
    compare = pred[] compare(partition-id, zero), direction=EQ
    broadcast = pred[2,2] broadcast(compare), dimensions={}

    weights = f32[2,2] get-tuple-element(inputs), index=2
    data = f32[2,2] get-tuple-element(inputs), index=1

    cp_back = f32[2,2] collective-permute(data), {{CHANNEL_ID_1}}
      source_target_pairs={{3,0}},
      frontend_attributes={_xla_send_recv_validation="{{3,10}}"}
    cp_forward = f32[2,2] collective-permute(data), {{CHANNEL_ID_2}}
      source_target_pairs={{0,1},{1,2},{2,3}},
      frontend_attributes={_xla_send_recv_validation="{{0,7},{1,8},{2,9}}"}

    select = f32[2,2] select(broadcast, cp_back, cp_forward)

    matmul = f32[2,2] dot(weights, select), lhs_contracting_dims={1},
        rhs_contracting_dims={0}

    ROOT result = (u32[], f32[2,2], f32[2,2]) tuple(next_iter, matmul, weights)
  }

  while_cond {
    inputs = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(inputs), index=0
    max_iter = u32[] constant(3)
    ROOT compare = pred[] compare(iter, max_iter), direction=LT
  }

  ENTRY test_computation {
    start_iter = u32[] constant(0)
    input_data = f32[2,2] parameter(0)
    input_weights = f32[2,2] parameter(1)
    input = (u32[], f32[2,2], f32[2,2]) tuple(start_iter, input_data,
        input_weights)
    while_result = (u32[], f32[2,2], f32[2,2]) while(input),
        condition=while_cond, body=while_body
    ROOT data_out = f32[2,2] get-tuple-element(while_result), index=1
  })";

  const char* kFilecheckStr = R"(
  // CHECK-LABEL: %while_body
  // CHECK-DAG:     %[[p:.+]] = parameter(0)
  // CHECK-DAG:     %[[iter:.+]] = get-tuple-element(%[[p]]), index=0
  // CHECK-DAG:     %[[c1:.+]] = constant(1)
  // CHECK-DAG:     %[[next_iter:.+]] = add(%[[iter]], %[[c1]])
  // CHECK-DAG:     %[[weights:.+]] = get-tuple-element(%[[p]]), index=2
  // CHECK-DAG:     %[[partition:.+]] = partition-id()
  // CHECK-DAG:     %[[c0:.+]] = constant(0)
  // CHECK-DAG:     %[[compare:.+]] = compare(%[[partition]], %[[c0]]), direction=EQ
  // CHECK-DAG:     %[[broadcast:.+]] = broadcast(%[[compare]]), dimensions={}
  // CHECK-DAG:     %[[data:.+]] = get-tuple-element(%[[p]]), index=1
  // CHECK-DAG:     %[[after_all:.+]] = after-all()
  // CHECK:         %[[send:.+]] = send(%[[data]], %[[after_all]]), {{CHANNEL_ID_1}}
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{3,0}},_xla_send_recv_validation={{3,10}}}
  // CHECK:         %[[recv:.+]] = recv(%[[after_all]]), {{CHANNEL_ID_1}}
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{3,0}},_xla_send_recv_validation={{3,10}}}
  // CHECK-DAG:     %[[recv_done:.+]] = recv-done(%[[recv]]), {{CHANNEL_ID_1}}frontend_attributes={_xla_send_recv_pipeline="0"}, control-predecessors={%[[send]]}
  // CHECK-DAG:     %[[gte:.+]] = get-tuple-element(%[[recv_done]]), index=0
  // CHECK-DAG:     %[[after_all1:.+]] = after-all()
  // CHECK:         %[[send1:.+]] = send(%[[data]], %[[after_all1]]), {{CHANNEL_ID_2}}
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}},_xla_send_recv_validation={{0,7},{1,8},{2,9}}}
  // CHECK:         %[[recv1:.+]] = recv(%[[after_all1]]), {{CHANNEL_ID_2}}
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}},_xla_send_recv_validation={{0,7},{1,8},{2,9}}}
  // CHECK-DAG:     %[[recv_done1:.+]] = recv-done(%[[recv1]]), {{CHANNEL_ID_2}} frontend_attributes={_xla_send_recv_pipeline="1"}, control-predecessors={%[[send1]]}
  // CHECK-DAG:     %[[gte1:.+]] = get-tuple-element(%[[recv_done1]]), index=0
  // CHECK-DAG:     %[[select:.+]] = select(%[[broadcast]], %[[gte]], %[[gte1]])
  // CHECK-DAG:     %[[matmul:.+]] = dot(%[[weights]], %[[select]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  // CHECK-DAG:     ROOT %{{.+}} = tuple(%[[next_iter]], %[[matmul]], %[[weights]])
  // CHECK-DAG:     %{{.+}} = send-done(%[[send]]), {{CHANNEL_ID_1}} frontend_attributes={_xla_send_recv_pipeline="0"}
  // CHECK-DAG:     %{{.+}} = send-done(%[[send1]]), {{CHANNEL_ID_2}} frontend_attributes={_xla_send_recv_pipeline="1"}
  })";
  std::string channel_id_hlo =
      absl::StrReplaceAll(kModuleStr, {{"{{CHANNEL_ID_1}}", "channel_id=1, "},
                                       {"{{CHANNEL_ID_2}}", "channel_id=2, "}});
  std::string channel_id_filecheck = absl::StrReplaceAll(
      kFilecheckStr, {{"{{CHANNEL_ID_1}}", "channel_id=1, "},
                      {"{{CHANNEL_ID_2}}", "channel_id=2, "}});
  std::string no_channel_id_hlo = absl::StrReplaceAll(
      kModuleStr, {{"{{CHANNEL_ID_1}}", ""}, {"{{CHANNEL_ID_2}}", ""}});
  std::string no_channel_id_filecheck = absl::StrReplaceAll(
      kFilecheckStr, {{"{{CHANNEL_ID_1}}", "channel_id=0, "},
                      {"{{CHANNEL_ID_2}}", "channel_id=0, "}});

  auto check_module = [](std::string module_str, std::string filecheck_str) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnUnverifiedModule((module_str)));
    CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
    TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
    EXPECT_TRUE(changed);
    // TODO: b/356201477 - Investigate potential NCCL deadlock in
    // collective_permute_decomposer because of control_predecessor checks
    // (recv->send->recv_done)
    // Check the annotations and ordering of the decomposed send-recv pairs.
    // We expect the recv to come before the send in the while body, both for
    // the forward edge ({0,1},{1,2},{2,3}}) and the backward edge ({3,0}). This
    // is an XLA invariant that shouldn't be broken (see
    // https://openxla.org/xla/operation_semantics#send for details of the
    // semantics).
    TF_ASSERT_OK_AND_ASSIGN(
        bool filecheck_status,
        RunFileCheck(/*input=*/module->ToString(
                         /*options=*/HloPrintOptions{}
                             .set_print_operand_shape(false)
                             .set_print_result_shape(false)),
                     /*pattern=*/filecheck_str));
    EXPECT_TRUE(filecheck_status);
  };

  check_module(channel_id_hlo, channel_id_filecheck);
  check_module(no_channel_id_hlo, no_channel_id_filecheck);
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

    recv-data.0 = u32[2] collective-permute(send-data), {{CHANNEL_ID_1}}
      source_target_pairs={{1,0},{2,1},{3,2}}

    recv-data.1 = u32[2] collective-permute(send-data), {{CHANNEL_ID_2}}
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

  const char* kFilecheckStr = R"(
  // CHECK-LABEL: %body
  // CHECK-DAG:  %[[p:.+]] = parameter(0)
  // CHECK-DAG:  %[[iter:.+]] = get-tuple-element(%[[p]]), index=0
  // CHECK-DAG:  %[[c1:.+]] = constant(1)
  // CHECK-DAG:  %[[new_count:.+]] = add(%[[iter]], %[[c1]])
  // CHECK-DAG:  %[[r:.+]] = broadcast(%[[c1]]), dimensions={}
  // CHECK-DAG:  %[[replica:.+]] = replica-id()
  // CHECK-DAG:  %[[c0:.+]] = constant(0)
  // CHECK-DAG:  %[[compare0:.+]] = compare(%[[replica]], %[[c0]]), direction=NE
  // CHECK-DAG:  %[[compare:.+]] = broadcast(%[[compare0]]), dimensions={}
  // CHECK-DAG:  %[[send_data:.+]] = get-tuple-element(%[[p]]), index=1
  // CHECK-DAG:  %[[after_all:.+]] = after-all()
  // CHECK:      %[[send:.+]] = send(%[[send_data]], %[[after_all]]), {{CHANNEL_ID_1}} 
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{1,0},{2,1},{3,2}}}
  // CHECK:      %[[recv:.+]] = recv(%[[after_all]]), {{CHANNEL_ID_1}}
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{1,0},{2,1},{3,2}}}
  // CHECK-DAG:  %[[recv_done:.+]] = recv-done(%[[recv]]), {{CHANNEL_ID_1}} frontend_attributes={_xla_send_recv_pipeline="1"}, control-predecessors={%[[send]]}
  // CHECK-DAG:  %[[gte:.+]] = get-tuple-element(%[[recv_done]]), index=0
  // CHECK-DAG:  %[[after_all1:.+]] = after-all()
  // CHECK:      %[[send1:.+]] = send(%[[send_data]], %[[after_all1]]), {{CHANNEL_ID_2}}
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{0,3}}}
  // CHECK:      %[[recv1:.+]] = recv(%[[after_all1]]), {{CHANNEL_ID_2}}
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{0,3}}}
  // CHECK-DAG:  %[[recv_done_1:.+]] = recv-done(%[[recv1]]), {{CHANNEL_ID_2}} frontend_attributes={_xla_send_recv_pipeline="0"}, control-predecessors={%[[send1]]}
  // CHECK-DAG:  %[[gte1:.+]] = get-tuple-element(%[[recv_done_1]]), index=0
  // CHECK-DAG:  %[[recv_data2:.+]] = select(%[[compare]], %[[gte]], %[[gte1]])
  // CHECK-DAG:  %[[s:.+]] = add(%[[r]], %[[recv_data2]])
  // CHECK-DAG:  ROOT %{{.+}} = tuple(%[[new_count]], %[[s]])
  // CHECK-DAG:  %{{.+}} = send-done(%[[send]]), {{CHANNEL_ID_1}} frontend_attributes={_xla_send_recv_pipeline="1"}
  // CHECK-DAG:  %{{.+}} = send-done(%[[send1]]), {{CHANNEL_ID_2}} frontend_attributes={_xla_send_recv_pipeline="0"}
  })";

  std::string channel_id_hlo =
      absl::StrReplaceAll(kModuleStr, {{"{{CHANNEL_ID_1}}", "channel_id=1, "},
                                       {"{{CHANNEL_ID_2}}", "channel_id=2, "}});
  std::string channel_id_filecheck = absl::StrReplaceAll(
      kFilecheckStr, {{"{{CHANNEL_ID_1}}", "channel_id=1, "},
                      {"{{CHANNEL_ID_2}}", "channel_id=2, "}});
  std::string no_channel_id_hlo = absl::StrReplaceAll(
      kModuleStr, {{"{{CHANNEL_ID_1}}", ""}, {"{{CHANNEL_ID_2}}", ""}});
  std::string no_channel_id_filecheck = absl::StrReplaceAll(
      kFilecheckStr, {{"{{CHANNEL_ID_1}}", "channel_id=0, "},
                      {"{{CHANNEL_ID_2}}", "channel_id=0, "}});
  auto check_module = [](std::string module_str, std::string filecheck_str) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnUnverifiedModule((module_str)));
    CollectivePermuteDecomposer decomposer(/*threshold_in_bytes=*/0);
    TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
    EXPECT_TRUE(changed);
    TF_ASSERT_OK_AND_ASSIGN(
        bool filecheck_status,
        RunFileCheck(/*input=*/module->ToString(
                         /*options=*/HloPrintOptions{}
                             .set_print_operand_shape(false)
                             .set_print_result_shape(false)),
                     /*pattern=*/filecheck_str));
    EXPECT_TRUE(filecheck_status);
  };

  check_module(channel_id_hlo, channel_id_filecheck);
  check_module(no_channel_id_hlo, no_channel_id_filecheck);
}

TEST_F(CollectivePermuteDecomposerTest, DefaultChannelIdNoCycle) {
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
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_status,
                          RunFileCheck(/*input=*/module->ToString(
                                           /*options=*/HloPrintOptions{}
                                               .set_print_operand_shape(false)
                                               .set_print_result_shape(false)),
                                       /*pattern=*/R"(
  // CHECK: ENTRY %test_computation
  // CHECK:   %[[REPLICA_ID:.+]] = replica-id()
  // CHECK:   %[[TOKEN:.+]] = after-all()
  // CHECK:   %[[SEND:.+]] = send(%[[REPLICA_ID]], %[[TOKEN]]), channel_id=0, 
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}}
  // CHECK:   %[[SEND_DONE:.+]] = send-done(%[[SEND]]), channel_id=0
  // CHECK:   %[[RECV:.+]] = recv(%[[TOKEN]]), channel_id=0,
  // CHECK-SAME{LITERAL}: frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}}
  // CHECK:   %[[RECV_DONE:.+]] = recv-done(%[[RECV]]), channel_id=0, control-predecessors={%[[SEND]]}
  // CHECK:   ROOT {{.+}} = get-tuple-element(%[[RECV_DONE]]), index=0
  )"));
  EXPECT_TRUE(filecheck_status);
}

}  // namespace
}  // namespace xla
