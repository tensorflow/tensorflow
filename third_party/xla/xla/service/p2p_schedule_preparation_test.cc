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

#include "xla/service/p2p_schedule_preparation.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_parser.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class P2PSchedulePreparationTest : public HloTestBase {
 public:
  // Verifies that no control dependence is added to the P2P group.
  void VerifyP2PNotTransformed(HloModule* module,
                               const std::string& suffix = "") {
    HloInstruction* recv = FindInstruction(module, "recv" + suffix);
    HloInstruction* recv_done = FindInstruction(module, "recv-done" + suffix);
    HloInstruction* send_done = FindInstruction(module, "send-done" + suffix);
    EXPECT_EQ(recv->control_predecessors().size(), 0);
    EXPECT_EQ(recv_done->control_predecessors().size(), 0);
    EXPECT_EQ(send_done->control_predecessors().size(), 0);
  }

  // Verifies that the control dependence enforces this ordering:
  //   recv => send => recv-done => send-done
  void VerifyP2P1GroupChain(HloModule* module, const std::string& suffix) {
    HloInstruction* send = FindInstruction(module, "send" + suffix);
    HloInstruction* recv = FindInstruction(module, "recv" + suffix);
    HloInstruction* recv_done = FindInstruction(module, "recv-done" + suffix);
    HloInstruction* send_done = FindInstruction(module, "send-done" + suffix);
    EXPECT_EQ(send->control_predecessors()[0], recv);
    EXPECT_EQ(recv_done->control_predecessors()[0], send);
    EXPECT_EQ(send_done->control_predecessors()[0], recv_done);
  }

  // Verifies that the control dependence enforces this ordering for an
  // unpipelined Send-Recv chain:
  //   recv => send => recv-done => send-done
  void VerifyUnpipelinedP2P(HloModule* module, const std::string& suffix = "") {
    VerifyP2P1GroupChain(module, suffix);
  }

  // Verifies that the control dependence enforces this ordering for a pipelined
  // Send-Recv chain in the while-body:
  //   recv => send => recv-done => send-done
  void VerifyPipelinedP2PChild(HloModule* module,
                               const std::string& suffix = "") {
    VerifyP2P1GroupChain(module, suffix);
  }

  // Verifies that the control dependence enforces this ordering for a pipelined
  // Send-Recv chain in the while-loop calling computation:
  //   recv => send => while-loop => recv-done => send-done.
  void VerifyPipelinedP2PParent(HloModule* module,
                                const std::string& suffix = "") {
    VerifyP2P1GroupChain(module, suffix);
  }

  // Verifies that the control dependence enforces this ordering:
  //  recv.0 => send.0 => recv.1 => send.1 =>
  //  recv-done.0 => recv-done.1 => send-done.0 => send-done.1
  void VerifyP2P2GroupChain(HloModule* module, const std::string& suffix0,
                            const std::string& suffix1) {
    HloInstruction* send0 = FindInstruction(module, "send" + suffix0);
    HloInstruction* recv0 = FindInstruction(module, "recv" + suffix0);
    HloInstruction* recv_done0 = FindInstruction(module, "recv-done" + suffix0);
    HloInstruction* send_done0 = FindInstruction(module, "send-done" + suffix0);
    HloInstruction* send1 = FindInstruction(module, "send" + suffix1);
    HloInstruction* recv1 = FindInstruction(module, "recv" + suffix1);
    HloInstruction* recv_done1 = FindInstruction(module, "recv-done" + suffix1);
    HloInstruction* send_done1 = FindInstruction(module, "send-done" + suffix1);

    EXPECT_EQ(recv_done1->control_predecessors()[0], recv_done0);
    EXPECT_EQ(send_done0->control_predecessors()[0], recv_done1);
    EXPECT_EQ(send_done1->control_predecessors()[0], send_done0);

    EXPECT_EQ(send0->control_predecessors()[0], recv0);
    EXPECT_EQ(recv1->control_predecessors()[0], send0);
    EXPECT_EQ(send1->control_predecessors()[0], recv1);

    EXPECT_EQ(recv_done0->control_predecessors()[0], send1);
  }

  // Verifies that the control dependence enforces this ordering for a pipelined
  // chain with two Send-Recv groups in a while-body:
  //  recv.0 => send.0 => recv.1 => send.1 =>
  //  recv-done.0 => send-done.0 => recv-done.1 => send-done.1
  void VerifyPipelined2P2PChild(HloModule* module, const std::string& suffix0,
                                const std::string& suffix1) {
    VerifyP2P2GroupChain(module, suffix0, suffix1);
  }

  // Verifies that the control dependence enforces this ordering for a pipelined
  // chain with two Send-Recv groups in the while-loop calling computation:
  //   recv.0 => send.0 => recv.1 => send.1 =>
  //   => recv-done.0 => send-done.0 => recv-done.1 => send-done.1
  void VerifyPipelined2P2PParent(HloModule* module, const std::string& suffix0,
                                 const std::string& suffix1) {
    VerifyP2P2GroupChain(module, suffix0, suffix1);
  }
};

constexpr char kEmpty[] = "";
constexpr char kHostTransfer[] = ", is_host_transfer=true";

// Returns an HLO module string for testing unnested P2P chain. The string is
// generated from a templated string with placeholders for specifying the
// following values:
//  Whether the Send/Recv operations are host transfer
//  Whether the Send/Recv operations form a complete P2P chain
//
std::string GetUnnestedP2PModuleString(bool is_host = false,
                                       bool incomplete = false) {
  constexpr char kSend[] = R"(
    send = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all),
      channel_id=2, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0,1}, {1,2}}"
    } %s
    send-done = token[] send-done(send), channel_id=2 %s
)";
  constexpr char kSimpleModule[] = R"(
  HloModule test
  ENTRY main {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=2,
      frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0,1}, {1,2}}"
    } %s
    recv-done = (f32[1, 1024, 1024], token[]) recv-done(recv), channel_id=2 %s
    %s
    ROOT recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done), index=0
  }
)";

  const char* is_host_str = is_host ? kHostTransfer : kEmpty;
  if (incomplete) {
    return absl::StrFormat(kSimpleModule, is_host_str, is_host_str, kEmpty);
  }
  std::string send_str = absl::StrFormat(kSend, is_host_str, is_host_str);
  return absl::StrFormat(kSimpleModule, is_host_str, is_host_str, send_str);
}

TEST_F(P2PSchedulePreparationTest, UnnestedP2PChainHostNotTransformed) {
  std::string kModuleStr = GetUnnestedP2PModuleString(/*is_host=*/true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  P2PSchedulePreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(P2PSchedulePreparationTest, UnnestedP2PChainIncompleteNotTransformed) {
  std::string kModuleStr =
      GetUnnestedP2PModuleString(/*is_host=*/false, /*incomplete*/ true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  P2PSchedulePreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(P2PSchedulePreparationTest, UnnestedP2PChainTransformed) {
  std::string kModuleStr = GetUnnestedP2PModuleString();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  P2PSchedulePreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_TRUE(changed);
  VerifyUnpipelinedP2P(module.get());
}

// Returns an HLO module string for testing nested unpipelined P2P chains. The
// string is generated from a templated string with placeholders for specifying
// the following values:
//  Whether the Send/Recv operations in the while-body are host transfer
//  Whether the Send/Recv operations in the main computation are host transfer
//
std::string GetNestedP2PModuleString(bool while_p2p_is_host = false,
                                     bool main_p2p_is_host = false) {
  constexpr char kModuleTemplate[] = R"(
  HloModule test
  while-cond {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(25)
    ROOT cond-result = pred[] compare(count, ub), direction=LT
  }

  while-body {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=1,
      frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}"
    } %s
    send = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all),
      channel_id=1, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}}"
    } %s
    recv-done = (f32[1, 1024, 1024], token[]) recv-done(recv), channel_id=1 %s
    recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done), index=0
    send-done = token[] send-done(send), channel_id=1 %s
    c1 = u32[] constant(1)
    new-count = u32[] add(count, c1)
    ROOT body-result = (u32[], f32[1, 1024, 1024]) tuple(new-count, recv-data)
  }

  ENTRY main {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all.1 = token[] after-all()
    recv.1 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.1), channel_id=2,
      frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}}"
    } %s
    send.1 = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all.1),
      channel_id=2, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}}"
    } %s
    recv-done.1 = (f32[1, 1024, 1024], token[]) recv-done(recv.1), channel_id=2 %s
    send-done.1 = token[] send-done(send.1), channel_id=2 %s
    recv-data.1 = f32[1, 1024, 1024] get-tuple-element(recv-done.1), index=0

    while-init = (u32[], f32[1, 1024, 1024]) tuple(c0, recv-data.1)
    while-result = (u32[], f32[1, 1024, 1024]) while(while-init),
      body=while-body, condition=while-cond

    while-result-data = f32[1, 1024, 1024] get-tuple-element(while-result), index=1
    ROOT entry-result = f32[1, 1024, 1024] add(while-result-data, recv-data.1)
  }
  )";
  const char* while_p2p = while_p2p_is_host ? kHostTransfer : kEmpty;
  const char* main_p2p = main_p2p_is_host ? kHostTransfer : kEmpty;
  return absl::StrFormat(kModuleTemplate, while_p2p, while_p2p, while_p2p,
                         while_p2p, main_p2p, main_p2p, main_p2p, main_p2p);
}

TEST_F(P2PSchedulePreparationTest, WhileP2PIsHostNotMainTransformed) {
  std::string kModuleStr = GetNestedP2PModuleString(/*while_p2p_is_host=*/true);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  P2PSchedulePreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_TRUE(changed);

  VLOG(10) << module->ToString();
  VerifyP2PNotTransformed(module.get());
  VerifyUnpipelinedP2P(module.get(), ".1");
  // Verify that while-loop is scheduled after Send-done even though the
  // while-loop only contains host P2P operations.
  HloInstruction* send_done = FindInstruction(module.get(), "send-done.1");
  HloInstruction* while_loop = FindInstruction(module.get(), "while-result");
  EXPECT_EQ(while_loop->control_predecessors()[0], send_done);
}

TEST_F(P2PSchedulePreparationTest, MainP2PIsHostNotWhileTransformed) {
  std::string kModuleStr = GetNestedP2PModuleString(/*while_p2p_is_host=*/false,
                                                    /*main_p2p_is_host=*/true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  P2PSchedulePreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_TRUE(changed);

  VLOG(10) << module->ToString();
  VerifyUnpipelinedP2P(module.get());
  VerifyP2PNotTransformed(module.get(), ".1");
}

TEST_F(P2PSchedulePreparationTest, NestedP2PChainTransformed) {
  std::string kModuleStr = GetNestedP2PModuleString();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  P2PSchedulePreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_TRUE(changed);

  VLOG(10) << module->ToString();
  VerifyUnpipelinedP2P(module.get());
  VerifyUnpipelinedP2P(module.get(), ".1");

  HloInstruction* send_done = FindInstruction(module.get(), "send-done.1");
  HloInstruction* recv_user = FindInstruction(module.get(), "while-result");
  EXPECT_EQ(recv_user->control_predecessors()[0], send_done);
}

// Returns an HLO module string for testing pipelined P2P chains. The string
// is generated from a templated string with placeholders for specifying the
// following values:
//  Whether the main computation contains another nested P2P chain besides the
//    pipelined P2P chain.
//  Whether the pipelined while-body contains another P2P chain besides the
//    pipelined P2P chain.
//  Whether the main computation uses a custom-call or a collective-permute for
//    the purpose of testing its ordering with respect to P2P chain.
std::string GetPipelinedP2PModuleString(bool nested_p2p_in_main = false,
                                        bool other_p2p_in_while = false,
                                        bool test_custom_call = false) {
  // This is to support the while-loop with nested P2P chains called from the
  // main computation.
  constexpr char kWhileForMain[] = R"(
  while-cond-2 {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(25)
    ROOT cond-result-2 = pred[] compare(count, ub), direction=LT
  }

  while-body-2 {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    after-all.3 = token[] after-all()
    recv.3 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.3), channel_id=3,
      frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}"
    }
    send.3 = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all.3),
      channel_id=3, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}}"
    }
    recv-done.3 = (f32[1, 1024, 1024], token[]) recv-done(recv.3), channel_id=3
    recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done.3), index=0
    send-done.3 = token[] send-done(send.3), channel_id=3
    c1 = u32[] constant(1)
    new-count = u32[] add(count, c1)
    ROOT body-result-2 = (u32[], f32[1, 1024, 1024]) tuple(new-count, recv-data)
  }
)";

  // This is the result for the main computation, if it doesn't have another
  // while-loop with nested P2P chains.
  constexpr char kUnnestedResult[] = R"(
  while-result-1 = f32[1, 1024, 1024] get-tuple-element(while-result), index=1
  collective-permute.2 = f32[1, 1024, 1024] collective-permute(init),
    source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}}
  ROOT entry-result = f32[1, 1024, 1024] add(while-result-1, collective-permute.2)
)";

  // Similar to the above, but for test_custom_call = true.
  constexpr char kUnnestedResultWithCustomCall[] = R"(
  while-result-1 = f32[1, 1024, 1024] get-tuple-element(while-result), index=1
  custom-call = f32[1, 1024, 1024] custom-call(init),
    custom_call_target="my_custom_call"
  ROOT entry-result = f32[1, 1024, 1024] add(while-result-1, custom-call)
)";

  // This is the result for the main computation, if it has another while-loop
  // with nested P2P chains.
  constexpr char kNestedResult[] = R"(
  while-result-1 = f32[1, 1024, 1024] get-tuple-element(while-result), index=1
  while-init-2 =  (u32[], f32[1, 1024, 1024]) tuple(c0, init)
  while-2 = (u32[], f32[1, 1024, 1024]) while(while-init-2),
      body=while-body-2, condition=while-cond-2,
      backend_config={"known_trip_count":{"n":"25"}}
  while-result-2 = f32[1, 1024, 1024] get-tuple-element(while-2), index=1
  ROOT entry-result = f32[1, 1024, 1024] add(while-result-1, while-result-2)
)";

  constexpr char kPipelinedWhileBodyWithoutOtherP2P[] = R"(
  while-body {
    param = (u32[], (f32[1, 1024, 1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0

    recv-done.1.q = (f32[1, 1024, 1024], token[]) get-tuple-element(param), index=1
    recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done.1.q), index=0

    c1 = u32[] constant(1)
    new-count = u32[] add(count, c1)
    replica = u32[] replica-id()
    c10 = u32[] constant(10)
    sum = u32[] add(replica, c10)
    sum2 = u32[] add(sum, count)
    conv = f32[] convert(sum2)
    p = f32[1, 1024, 1024] broadcast(conv), dimensions={}
    b = f32[1, 1024, 1024] add(p, recv-data)
    c = f32[1, 1024, 1024] multiply(b, b)
    d = f32[1, 1024, 1024] tan(c)
    s = f32[1, 1024, 1024] dot(c, d), lhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    collective-permute.1 = f32[1, 1024, 1024] collective-permute(s),
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}}
    new-data = f32[1, 1024, 1024] add(c, collective-permute.1)

    after-all.1 = token[] after-all()
    send.1 = (f32[1, 1024, 1024], token[]) send(new-data, after-all.1),
      channel_id=1, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
      _xla_send_recv_pipeline="0"
    }
    send-done.1 = token[] send-done(send.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    recv.1 = (f32[1, 1024, 1024], token[]) recv(after-all.1), channel_id=1,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
       _xla_send_recv_pipeline="0"
    }
    recv-done.1 = (f32[1, 1024, 1024], token[]) recv-done(recv.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }

    ROOT body-result = (u32[], (f32[1, 1024, 1024], token[]), token[])
      tuple(new-count, recv-done.1, send-done.1)
  }
)";

  constexpr char kPipelinedWhileBodyWithOtherP2P[] = R"(
  while-body {
    param = (u32[], (f32[1, 1024, 1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0
    recv-done.1.q = (f32[1, 1024, 1024], token[])get-tuple-element(param), index=1
    recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done.1.q), index=0

    c1 = u32[] constant(1)
    new-count = u32[] add(count, c1)
    replica = u32[] replica-id()
    c10 = u32[] constant(10)
    sum = u32[] add(replica, c10)
    sum2 = u32[] add(sum, count)
    conv = f32[] convert(sum2)
    p = f32[1, 1024, 1024] broadcast(conv), dimensions={}
    b = f32[1, 1024, 1024] add(p, recv-data)
    c = f32[1, 1024, 1024] multiply(b, b)
    d = f32[1, 1024, 1024] tan(c)
    s = f32[1, 1024, 1024] dot(c, d), lhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    collective-permute.1 = f32[1, 1024, 1024] collective-permute(s),
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}}
    send-data = f32[1, 1024, 1024] add(c, collective-permute.1)

    after-all.4 = token[] after-all()
    send.4 = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all.4),
      channel_id=4, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}"
      }
    send-done.4 = token[] send-done(send.4), channel_id=4
    recv.4 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.4), channel_id=4,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}"
      }
    recv-done.4 = (f32[1, 1024, 1024], token[]) recv-done(recv.4), channel_id=4
    new-data = f32[1, 1024, 1024] get-tuple-element(recv-done.4), index=0

    after-all.1 = token[] after-all()
    send.1 = (f32[1, 1024, 1024], token[]) send(new-data, after-all.1),
      channel_id=1, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    send-done.1 = token[] send-done(send.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    recv.1 = (f32[1, 1024, 1024], token[]) recv(after-all.1), channel_id=1,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
       _xla_send_recv_pipeline="0"
    }
    recv-done.1 = (f32[1, 1024, 1024], token[]) recv-done(recv.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    ROOT body-result = (u32[], (f32[1, 1024, 1024], token[]), token[])
      tuple(new-count, recv-done.1, send-done.1)
  }
)";

  constexpr char kModuleTemplate[] = R"(
  HloModule test

  while-cond {
    param = (u32[], (f32[1, 1024, 1024], u32[], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(25)
    ROOT cond-result = pred[] compare(count, ub), direction=LT
  }

  // The code that support the while-loop with nested P2P chains goes here.
  %s

  // The pipelined while-body goes here.
  %s

  ENTRY test-computation {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all.2 = token[] after-all()
    recv.2 = (f32[1, 1024, 1024], token[]) recv(after-all.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.2 = (f32[1, 1024, 1024], token[]) recv-done(recv.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send.2 = (f32[1, 1024, 1024], token[]) send(init, after-all.2),
      channel_id=1, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    send-done.2 = token[] send-done(send.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }

    while-init =  (u32[], (f32[1, 1024, 1024], token[]), token[])
      tuple(c0, recv-done.2, send-done.2)
    while-result =  (u32[], (f32[1, 1024, 1024], token[]), token[])
      while(while-init),
      body=while-body, condition=while-cond,
      backend_config={"known_trip_count":{"n":"25"}}

    recv-done.2.q = (f32[1, 1024, 1024], token[]) get-tuple-element(while-result), index=1
    recv-data.2.q = f32[1, 1024, 1024] get-tuple-element(recv-done.2.q), index=0

    // The code for the computation result goes here.
    %s
  }
)";

  const char* while_str = nested_p2p_in_main ? kWhileForMain : kEmpty;
  const char* pipelined_while_body_str =
      other_p2p_in_while ? kPipelinedWhileBodyWithOtherP2P
                         : kPipelinedWhileBodyWithoutOtherP2P;
  const char* result_str =
      nested_p2p_in_main ? kNestedResult
                         : (test_custom_call ? kUnnestedResultWithCustomCall
                                             : kUnnestedResult);
  return absl::StrFormat(kModuleTemplate, while_str, pipelined_while_body_str,
                         result_str);
}

TEST_F(P2PSchedulePreparationTest, UnnestedPipelinedP2PChainTransformed) {
  std::string kModuleStr = GetPipelinedP2PModuleString();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  P2PSchedulePreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_TRUE(changed);

  VLOG(10) << module->ToString();
  // Verify the pipelined P2P chain in the while-body.
  VerifyPipelinedP2PChild(module.get(), ".1");
  // Verify the pipelined P2P chain in the main computation.
  VerifyPipelinedP2PParent(module.get(), ".2");

  // Verify in the while-body collective-permute is scheduled before recv.
  HloInstruction* recv_1 = FindInstruction(module.get(), "recv.1");
  HloInstruction* collective_1 =
      FindInstruction(module.get(), "collective-permute.1");
  EXPECT_EQ(recv_1->control_predecessors()[0], collective_1);

  // Verify in the main computation collective-permute is either scheduled
  // after send-done or before recv of the pipelined P2P chain.
  HloInstruction* send_done_2 = FindInstruction(module.get(), "send-done.2");
  HloInstruction* recv_2 = FindInstruction(module.get(), "recv.2");
  HloInstruction* collective_2 =
      FindInstruction(module.get(), "collective-permute.2");
  EXPECT_TRUE((!collective_2->control_predecessors().empty() &&
               collective_2->control_predecessors()[0] == send_done_2) ||
              (!recv_2->control_predecessors().empty() &&
               recv_2->control_predecessors()[0] == collective_2));
}

TEST_F(P2PSchedulePreparationTest, NestedPipelinedP2PChainTransformed) {
  std::string kModuleStr =
      GetPipelinedP2PModuleString(/*nested_p2p_in_main=*/true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  P2PSchedulePreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_TRUE(changed);

  VLOG(10) << module->ToString();
  // Verify the pipelined P2P chain in the while-body.
  VerifyPipelinedP2PChild(module.get(), ".1");
  // Verify the pipelined P2P chain in the main computation.
  VerifyPipelinedP2PParent(module.get(), ".2");
  // Verify the unpipelined P2P chain in the other while-body.
  VerifyUnpipelinedP2P(module.get(), ".3");

  // Verify in the while-loop with nested P2P is either scheduled after
  // end-done or before recv of the pipelined P2P chain.
  HloInstruction* send_done_2 = FindInstruction(module.get(), "send-done.2");
  HloInstruction* recv_2 = FindInstruction(module.get(), "recv.2");
  HloInstruction* while_2 = FindInstruction(module.get(), "while-2");
  EXPECT_TRUE((!while_2->control_predecessors().empty() &&
               while_2->control_predecessors()[0] == send_done_2) ||
              (!recv_2->control_predecessors().empty() &&
               recv_2->control_predecessors()[0] == while_2));
}

TEST_F(P2PSchedulePreparationTest,
       UnnestedPipelinedP2PChainWithOtherP2PTransformed) {
  std::string kModuleStr = GetPipelinedP2PModuleString(
      /*nested_p2p_in_main=*/false, /*other_p2p_in_while=*/true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  P2PSchedulePreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_TRUE(changed);

  VLOG(10) << module->ToString();
  // Verify the pipelined P2P chain in the while-body.
  VerifyPipelinedP2PChild(module.get(), ".1");
  // Verify the pipelined P2P chain in the main computation.
  VerifyPipelinedP2PParent(module.get(), ".2");
  // Verify the other unpipelined P2P chain in the while-body.
  VerifyUnpipelinedP2P(module.get(), ".4");

  // Verify that in the pipelined while-body, the pipelined recv is ordered
  // after other P2P.
  HloInstruction* pipelined_recv = FindInstruction(module.get(), "recv.1");
  HloInstruction* other_send_done =
      FindInstruction(module.get(), "send-done.4");
  EXPECT_EQ(1, absl::c_count(pipelined_recv->control_predecessors(),
                             other_send_done));
}

TEST_F(P2PSchedulePreparationTest,
       UnnestedPipelinedP2PChainWithCustomCallTransformed) {
  std::string kModuleStr = GetPipelinedP2PModuleString(
      /*nested_p2p_in_main=*/false, /*other_p2p_in_while=*/false,
      /*test_custom_call=*/true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  P2PSchedulePreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_TRUE(changed);

  // Verify in the main computation, custom-call is either scheduled after
  // end-done or before recv of the pipelined P2P chain.
  HloInstruction* send_done_2 = FindInstruction(module.get(), "send-done.2");
  HloInstruction* recv_2 = FindInstruction(module.get(), "recv.2");
  HloInstruction* custom_call = FindInstruction(module.get(), "custom-call");
  EXPECT_TRUE((!custom_call->control_predecessors().empty() &&
               custom_call->control_predecessors()[0] == send_done_2) ||
              (!recv_2->control_predecessors().empty() &&
               recv_2->control_predecessors()[0] == custom_call));
}

TEST_F(P2PSchedulePreparationTest, PipelinedP2PChain2Transformed) {
  const char* const kModuleStr = R"(
  HloModule test

cond {
    param = (u32[], (u32[2], token[]), (u32[2], token[]),
      token[], token[]) parameter(0)
    count = get-tuple-element(%param), index=0
    ub = u32[] constant(10)
    ROOT result = pred[] compare(count, ub), direction=LT
 }

body {
    param = (u32[], (u32[2], token[]), (u32[2], token[]),
      token[], token[]) parameter(0)
    count = get-tuple-element(param), index=0

    recv-done.0.f = (u32[2], token[]) get-tuple-element(param), index=1
    recv-data.0 = u32[2] get-tuple-element(recv-done.0.f), index=0
    recv-done.1.f = (u32[2], token[]) get-tuple-element(param), index=2
    recv-data.1 = u32[2] get-tuple-element(recv-done.1.f), index=0

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=EQ
    compare = pred[2] broadcast(compare0), dimensions={}
    recv-data = u32[2] select(compare, recv-data.0, recv-data.1)

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

    // The Recv "rotated" from the beginning of the loop to the end of the loop.
    after-all.0.n = token[] after-all()
    recv.0 = (u32[2], u32[], token[]) recv(after-all.0.n), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_pipeline="0"
      }
    send.0 = (u32[2], u32[], token[]) send(s, after-all.0.n),
      channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.0 = (u32[2], token[]) recv-done(recv.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.0 = token[] send-done(send.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }

    after-all.1.n = token[] after-all()
    recv.1 = (u32[2], u32[], token[]) recv(after-all.1.n), channel_id=2,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3}}",
        _xla_send_recv_pipeline="1"
      }
    send.1 = (u32[2], u32[], token[]) send(s, after-all.1.n),
      channel_id=2,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3}}",
        _xla_send_recv_pipeline="1"
      }
    recv-done.1 = (u32[2], token[]) recv-done(recv.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }
    send-done.1 = token[] send-done(send.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }
    ROOT result = (u32[], (u32[2], token[]), (u32[2], token[]), token[], token[])
      tuple(new_count, recv-done.0, recv-done.1, send-done.0, send-done.1)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    r = u32[] replica-id()
    a = u32[] add(c1, r)
    init = u32[2] broadcast(a), dimensions={}

    // Peel off both Recv.
    after-all.0.p = token[] after-all()
    recv.2 = (u32[2], u32[], token[]) recv(after-all.0.p), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_pipeline="0"
      }
    send.2 = (u32[2], u32[], token[]) send(init, after-all.0.p),
      channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.2 = (u32[2], token[]) recv-done(recv.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.2 = token[] send-done(send.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    after-all.1.p = token[] after-all()
    recv.3 = (u32[2], u32[], token[]) recv(after-all.1.p), channel_id=2,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3}}",
        _xla_send_recv_pipeline="1"
      }
    send.3 = (u32[2], u32[], token[]) send(init, after-all.1.p),
      channel_id=2,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3}}",
        _xla_send_recv_pipeline="1"
      }
    recv-done.3 = (u32[2], token[]) recv-done(recv.3), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }
    send-done.3 = token[] send-done(send.3), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }
    // This is the pipelined loop.
    while_init = (u32[], (u32[2], token[]), (u32[2], token[]),
      token[], token[]) tuple(c0, recv-done.2, recv-done.3, send-done.2, send-done.3)
    while_result = (u32[], (u32[2], u32[], token[]), (u32[2], u32[], token[]),
      token[], token[]) while(while_init), body=body, condition=cond,
    backend_config={"known_trip_count":{"n":"10"}}

    // This is the remaining Send/Send-done/Recv-done for the pipeline.
    // Use .q as suffix for HLO name.
     recv-done.0.q = (u32[2], u32[], token[]) get-tuple-element(while_result), index=1
     recv-data.0.q = u32[2] get-tuple-element(recv-done.0.q), index=0

     recv-done.1.q = (u32[2], u32[], token[]) get-tuple-element(while_result), index=2
     recv-data.1.q = u32[2] get-tuple-element(recv-done.1.q), index=0

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=EQ
    compare = pred[2] broadcast(compare0), dimensions={}
    recv-data = u32[2] select(compare, recv-data.0.q, recv-data.1.q)

    s = u32[2] add(c1, recv-data)

    ROOT result = u32[2] add(s, recv-data)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  P2PSchedulePreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);

  // Verify the pipelined P2P chain in the while-body.
  VerifyPipelined2P2PChild(module.get(), ".0", ".1");
  // Verify the pipelined P2P chain in the main computation.
  VerifyPipelined2P2PParent(module.get(), ".2", ".3");
}

TEST_F(P2PSchedulePreparationTest, UnpipelinedP2PChain2Transformed) {
  const char* const kModuleStr = R"(
  HloModule test

cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(%param), index=0
    ub = u32[] constant(11)
    ROOT result = pred[] compare(count, ub), direction=LT
 }

body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = u32[2] get-tuple-element(param), index=1

    after-all.0.n = token[] after-all()
    recv.0 = (u32[2], u32[], token[]) recv(after-all.0.n), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_pipeline="0"
      }
    send.0 = (u32[2], u32[], token[]) send(send-data, after-all.0.n),
      channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.0 = (u32[2], token[]) recv-done(recv.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.0 = token[] send-done(send.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }

    after-all.1 = token[] after-all()
    recv.1 = (u32[2], u32[], token[]) recv(after-all.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3}}",
        _xla_send_recv_pipeline="1"
      }
    send.1 = (u32[2], u32[], token[]) send(send-data, after-all.1),
      channel_id=2,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3}}",
        _xla_send_recv_pipeline="1"
      }
    recv-done.1 = (u32[2], token[]) recv-done(recv.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }
    send-done.1 = token[] send-done(send.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }

    recv-data.0 = u32[2] get-tuple-element(recv-done.0), index=0
    recv-data.1 = u32[2] get-tuple-element(recv-done.1), index=0

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
    while_result = (u32[], u32[2]) while(while_init), body=body, condition=cond,
      backend_config={"known_trip_count":{"n":"11"}}
    ROOT recv-data = u32[2] get-tuple-element(while_result), index=1
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  P2PSchedulePreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_TRUE(changed);

  // Verify the unpipelined P2P chain with two channels in the while-body.
  VerifyP2P2GroupChain(module.get(), ".0", ".1");
}

}  // namespace
}  // namespace xla
