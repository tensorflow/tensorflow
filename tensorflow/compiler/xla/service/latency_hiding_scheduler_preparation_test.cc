/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/latency_hiding_scheduler_preparation.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

using LatencyHidingSchedulerPreparationTest = HloTestBase;

constexpr char kEmpty[] = "";
constexpr char kHostTransfer[] = ", is_host_transfer=true";
constexpr char kChainRecvDoneToSendDone[] =
    ", control-predecessors={recv-done.1}";

// Returns an HLO module string for testing, which is generated from a
// templated string with placeholders for specifying the following values:
//  Whether the Send/Recv operations in the while-body are host transfer
//  Whether the Send/Recv operations in the main computation are host transfer
//  Wether the SendDone/RecvDone operations in the main computation are chain
//
std::string GetHloModuleString(bool whileP2PIsHost = false,
                               bool mainP2PIsHost = false,
                               bool chainRecvDoneToSendDone = true) {
  // A template string for the input HLO module.
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
      _xla_send_recv_source_target_pairs="{{0, 1}}"
    } %s
    send = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all),
      channel_id=1, control-predecessors={recv}, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}}"
    } %s
    recv-done = (f32[1, 1024, 1024], token[]) recv-done(recv), channel_id=1, control-predecessors={send} %s
    recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done), index=0
    send-done = token[] send-done(send), control-predecessors={recv-done}, channel_id=1 %s %s
    c1 = u32[] constant(1)
    new-count = u32[] add(count, c1)
    ROOT result = (u32[], f32[1, 1024, 1024]) tuple(new-count, recv-data)
  }

  ENTRY main {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all.1 = token[] after-all()
    recv.1 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.1), channel_id=2,
      frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}}"
    } %s
    send.1 = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all.1),
      channel_id=2, control-predecessors={recv.1}, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}}"
    } %s
    recv-done.1 = (f32[1, 1024, 1024], token[]) recv-done(recv.1), channel_id=2, control-predecessors={send.1} %s
    send-done.1 = token[] send-done(send.1), channel_id=2 %s
    recv-data.1 = f32[1, 1024, 1024] get-tuple-element(recv-done.1), index=0

    while-init = (u32[], f32[1, 1024, 1024]) tuple(c0, recv-data.1)
    while-result = (u32[], f32[1, 1024, 1024]) while(while-init),
      body=while-body, condition=while-cond

    while-result-data = f32[1, 1024, 1024] get-tuple-element(while-result), index=1
    ROOT entry-result = f32[1, 1024, 1024] add(while-result-data, recv-data.1)
  }
  )";
  const char* while_p2p = whileP2PIsHost ? kHostTransfer : kEmpty;
  const char* main_p2p = mainP2PIsHost ? kHostTransfer : kEmpty;
  const char* chain =
      chainRecvDoneToSendDone ? kChainRecvDoneToSendDone : kEmpty;
  return absl::StrFormat(kModuleTemplate, while_p2p, while_p2p, while_p2p,
                         while_p2p, main_p2p, main_p2p, main_p2p, main_p2p,
                         chain);
}

TEST_F(LatencyHidingSchedulerPreparationTest, WhileP2PIsHostNotTransformed) {
  std::string kModuleStr = GetHloModuleString(/*whileP2PIsHost=*/true);
  VLOG(0) << kModuleStr;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  LatencyHidingSchedulerPreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(LatencyHidingSchedulerPreparationTest, MainP2PIsHostNotTransformed) {
  std::string kModuleStr = GetHloModuleString(/*whileP2PIsHost=*/false,
                                              /*mainP2PIsHost=*/true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  LatencyHidingSchedulerPreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(LatencyHidingSchedulerPreparationTest, MainP2PNotChainedNotTransformed) {
  std::string kModuleStr =
      GetHloModuleString(/*whileP2PIsHost=*/false,
                         /*mainP2PIsHost=*/false,
                         /*chainRecvDoneToSendDone=*/false);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  LatencyHidingSchedulerPreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(LatencyHidingSchedulerPreparationTest, ChainedWithNestedP2PTransformed) {
  std::string kModuleStr = GetHloModuleString();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  LatencyHidingSchedulerPreparation preparation;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, preparation.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* send_done = FindInstruction(module.get(), "send-done.1");
  HloInstruction* recv_data = FindInstruction(module.get(), "recv-data.1");
  EXPECT_EQ(recv_data->control_predecessors()[0], send_done);
}

}  // namespace
}  // namespace xla
