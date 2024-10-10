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

#include "xla/service/gpu/transforms/collective_send_recv_combiner.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using CollectiveSendRecvCombinerTest = HloTestBase;

// TODO: b/372132451 - add unit test in collective send/recv combiner to check
// control dependencies

TEST_F(CollectiveSendRecvCombinerTest, TransformedNoFrontEndAttr) {
  const char* kHloStr = R"(
  ENTRY main {
    data = f32[] constant(5)
    recv-start = token[] after-all()
    recv = (f32[], u32[], token[]) recv(recv-start), channel_id=1
    send = (f32[], u32[], token[]) send(data, recv-start), channel_id=1
    recv-done = (f32[], token[]) recv-done(recv), channel_id=1
    send-done = token[] send-done(send), channel_id=1
    ROOT out = f32[] get-tuple-element(recv-done), index=0
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kHloStr)));
  CollectiveSendRecvCombiner combiner;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combiner.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
     CHECK: ENTRY %[[MAIN:.*]] () -> f32[] {
     CHECK: %[[RECV_START:.*]] = token[] after-all()
     CHECK: %[[RECV_ASYNC:.*]] = ((token[]), (f32[], u32[], token[]), s32[]) 
      recv-start(token[] %[[RECV_START:.*]]), channel_id=1
     CHECK: %[[RECV_DONE:.*]] = (f32[], u32[], token[])
      recv-done(((token[]), (f32[], u32[], token[]), s32[]) %[[RECV_ASYNC:.*]])
     CHECK: ROOT %[[OUT:.*]] = f32[] get-tuple-element((f32[], u32[], token[])
      %[[RECV_DONE:.*]]), index=0
     CHECK: %[[DATA:.*]] = f32[] constant(5)
     CHECK: %[[SEND_ASYNC:.*]] = ((f32[], token[]), (f32[], u32[], token[]), s32[])
      send-start(f32[] %[[DATA]], token[] %[[RECV_ASYNC:.*]])
     CHECK: %[[SEND_DONE:.*]] = (f32[], u32[], token[])
      send-done(((f32[], token[]), (f32[], u32[], token[]), s32[]) %[[SEND_ASYNC:.*]])
  )")
                  .value());
}

TEST_F(CollectiveSendRecvCombinerTest, TrivialNoTransform) {
  const char* kHloStr = R"(
  ENTRY main {
    zero = f32[] constant(0)
    five = f32[] constant(5)
    ROOT out = f32[] add(zero, five)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kHloStr)));
  CollectiveSendRecvCombiner combiner;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combiner.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectiveSendRecvCombinerTest, PartiallyPipelinedSendRecvNoTransform) {
  const char* const kModuleStr = R"(
    HloModule test

    while_body {
      param = ((f32[16], u32[], token[]), f32[16]) parameter(0)
      prev_send = (f32[16], u32[], token[]) get-tuple-element(param), index=0
      data = f32[16] get-tuple-element(param), index=1
      send_done = (f32[16], token[]) send-done(prev_send), channel_id=1
      after_all = token[] after-all()
      send = (f32[16], u32[], token[]) send(data, after_all), channel_id=1,
          frontend_attributes={
            _xla_send_send_source_target_pairs={{0,1},{1,2},{2,3}}}
      ROOT tuple = ((f32[16], u32[], token[]), f32[16]) tuple(send, data)
    }

    // Infinite loop to keep IR small.
    while_condition {
      param = ((f32[16], u32[], token[]), f32[16]) parameter(0)
      ROOT infinite_loop = pred[] constant(true)
    }

    ENTRY main_spmd {
      data = f32[16] parameter(0)
      after_all = token[] after-all()
      send = (f32[16], u32[], token[]) send(data, after_all), channel_id=1,
          frontend_attributes={
            _xla_send_send_source_target_pairs={{0,1},{1,2},{2,3}}}
      init = ((f32[16], u32[], token[]), f32[16]) tuple(send, data)
      while = ((f32[16], u32[], token[]), f32[16]) while(init),
          condition=while_condition, body=while_body
      send_ctx = (f32[16], u32[], token[]) get-tuple-element(while), index=0
      ROOT send_done = (f32[16], token[]) send-done(send_ctx), channel_id=1
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kModuleStr)));
  CollectiveSendRecvCombiner combiner;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combiner.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectiveSendRecvCombinerTest, TransformedWithControlDependency) {
  const char* kHloStr = R"(
  ENTRY main {
    data = f32[] constant(5)
    recv-start = token[] after-all()
    recv = (f32[], u32[], token[]) recv(recv-start), channel_id=1
    send = (f32[], u32[], token[]) send(data, recv-start), channel_id=1
    recv-done = (f32[], token[]) recv-done(recv), channel_id=1, control-predecessors={send}
    send-done = token[] send-done(send), channel_id=1
    ROOT out = f32[] get-tuple-element(recv-done), index=0
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kHloStr)));
  CollectiveSendRecvCombiner combiner;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combiner.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
     CHECK: ENTRY %[[MAIN:.*]] () -> f32[] {
     CHECK: %[[DATA:.*]] = f32[] constant(5)
     CHECK: %[[RECV_START:.*]] = token[] after-all()
     CHECK: %[[SEND_ASYNC:.*]] = ((f32[], token[]), (f32[], u32[], token[]), s32[])
      send-start(f32[] %[[DATA]], token[] %[[RECV_START:.*]])
     CHECK: %[[RECV_ASYNC:.*]] = ((token[]), (f32[], u32[], token[]), s32[]) 
      recv-start(token[] %[[RECV_START:.*]]), channel_id=1
     CHECK: %[[RECV_DONE:.*]] = (f32[], u32[], token[])
      recv-done(((token[]), (f32[], u32[], token[]), s32[]) %[[RECV_ASYNC:.*]])
     CHECK: ROOT %[[OUT:.*]] = f32[] get-tuple-element((f32[], u32[], token[])
      %[[RECV_DONE:.*]]), index=0
     CHECK: %[[SEND_DONE:.*]] = (f32[], u32[], token[])
      send-done(((f32[], token[]), (f32[], u32[], token[]), s32[]) %[[SEND_ASYNC:.*]])
  )")
                  .value());
}
}  // namespace
}  // namespace xla
