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

#include "xla/service/gpu/transforms/collectives/collective_send_recv_combiner.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using CollectiveSendRecvCombinerTest = HloHardwareIndependentTestBase;

TEST_F(CollectiveSendRecvCombinerTest, TransformedWithSourceTargetPairs) {
  const char* kHloStr = R"(
  ENTRY main {
    data = f32[] constant(5)
    recv-start = token[] after-all()
    recv = (f32[], u32[], token[]) recv(recv-start), channel_id=1,
          frontend_attributes={
            _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
    send = (f32[], u32[], token[]) send(data, recv-start), channel_id=1,
          frontend_attributes={
            _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
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
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK: %[[WRAPPED_SEND_RECV:.*]] (param0: f32[], param1: token[], param2: token[]) ->
    CHECK-SAME: ((f32[], u32[], token[]), (f32[], u32[], token[]))
    CHECK-NEXT: %[[PARAM0:.*]] = f32[] parameter(0)
    CHECK: %[[PARAM1:.*]] = token[] parameter(1)
    CHECK: %[[SEND1:.*]] = ({{.*}}) send(%[[PARAM0]], %[[PARAM1]]), channel_id=1,
    CHECK-SAME: frontend_attributes{{.*}}_xla_send_recv_source_target_pairs{{.*}}0,1{{.*}}1,2{{.*}}2,3{{.*}}
    CHECK-NEXT: %[[PARAM2:.*]] = {{.*}} parameter(2)
    CHECK: %[[RECV1:.*]] = ({{.*}}) recv(%[[PARAM2]]), channel_id=1,
    CHECK-SAME: frontend_attributes{{.*}}_xla_send_recv_source_target_pairs{{.*}}0,1{{.*}}1,2{{.*}}2,3{{.*}}
    CHECK-NEXT: ROOT %[[OUT:.*]] = {{.*}} tuple(%[[SEND1]], %[[RECV1]])
    CHECK: ENTRY %[[MAIN:.*]] () -> f32[]
    CHECK: %[[DATA:.*]] = {{.*}} constant(5)
    CHECK: %[[RECV_START:.*]] = {{.*}} after-all()
    CHECK: %[[TUPLE_START:.*]] = {{.*}} async-start(%[[DATA]], %[[RECV_START]], %[[RECV_START]]), calls=%[[WRAPPED_SEND_RECV]]
    CHECK-NEXT: %[[TUPLE_DONE:.*]] = {{.*}} async-done(%[[TUPLE_START]])
    CHECK %[[GTE2:.*]] = {{.*}} get-tuple-element(%[[TUPLE_DONE]]), index=1
    CHECK %[[GTE3:.*]] = {{.*}} get-tuple-element(%[[GTE2]]), index=0
    CHECK %[[GTE4:.*]] = {{.*}} get-tuple-element(%[[GTE2]]), index=2
    CHECK %[[TUPLE1:.*]] = {{.*}} tuple(%[[GTE3]], %[[GTE4]])
    CHECK ROOT %[[OUT:.*]] = {{.*}} get-tuple-element(%[[TUPLE1]]), index=0
    CHECK %[[GTE:.*]] = {{.*}} get-tuple-element(%[[TUPLE_DONE]]), index=0
    CHECK %[[GTE1:.*]] = {{.*}} get-tuple-element(%[[GTE]]), index=2
  )"));
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
    recv-done = (f32[], token[]) recv-done(recv), channel_id=1,
      control-predecessors={send}
    send-done = token[] send-done(send), channel_id=1
    ROOT out = f32[] get-tuple-element(recv-done), index=0
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kHloStr)));
  CollectiveSendRecvCombiner combiner;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combiner.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK: %[[WRAPPED_SEND_RECV:.*]] (param0: f32[], param1: token[], param2: token[]) -> ((f32[], u32[], token[]), (f32[], u32[], token[])) {

    CHECK: %[[PARAM0:.*]] = f32[] parameter(0)
    CHECK: %[[PARAM1:.*]] = token[] parameter(1)
    CHECK: %[[SEND1:.*]] = (f32[], u32[], token[]) send(%[[PARAM0]], %[[PARAM1]]), channel_id=1
    CHECK: %[[PARAM2:.*]] = token[] parameter(2)
    CHECK: %[[RECV1:.*]] = (f32[], u32[], token[]) recv(%[[PARAM2]]), channel_id=1
    CHECK: ROOT %[[OUT:.*]] = ((f32[], u32[], token[]), (f32[], u32[], token[])) tuple(%[[SEND1]], %[[RECV1]])

    CHECK: ENTRY %[[MAIN:.*]] () -> f32[] {
    CHECK: %[[DATA:.*]] = f32[] constant(5)
    CHECK: %[[RECV_START:.*]] = token[] after-all()
    CHECK: %[[TUPLE_START:.*]] = ((f32[], token[], token[]), ((f32[], u32[], token[]), (f32[], u32[], token[])), s32[]) async-start(%[[DATA]], %[[RECV_START]], %[[RECV_START]]), calls=%[[WRAPPED_SEND_RECV]]
    CHECK: %[[TUPLE_DONE:.*]] = ((f32[], u32[], token[]), (f32[], u32[], token[])) async-done(%[[TUPLE_START]])
    CHECK %[[GTE2:.*]] = (f32[], u32[], token[]) get-tuple-element(%[[TUPLE_DONE]], index=1)
    CHECK %[[GTE3:.*]] = f32[] get-tuple-element(%[[GTE2]], index=0)
    CHECK %[[GTE4:.*]] = token[] get-tuple-element(%[[GTE2]], index=2)
    CHECK %[[TUPLE1:.*]] = (f32[], token[]) tuple(%[[GTE3]], %[[GTE4]]), control-predecessors={%[[TUPLE_START]]}
    CHECK ROOT %[[OUT:.*]] = f32[] get-tuple-element(%[[TUPLE1]], index=0)
    CHECK %[[GTE:.*]] = (f32[], u32[], token[]) get-tuple-element(%[[TUPLE_DONE]], index=0)
    CHECK %[[GTE1:.*]] = token[] get-tuple-element(%[[GTE]], index=2)
  )"));
}

TEST_F(CollectiveSendRecvCombinerTest, TransformedWithMultipleSendRecv) {
  const char* kHloStr = R"(
  ENTRY main {
    data-1 = f32[] constant(1)
    data-2 = f32[] constant(2)
    after-all-1 = token[] after-all()
    send-1 = (f32[], u32[], token[]) send(data-1, after-all-1), channel_id=1
    recv-1 = (f32[], u32[], token[]) recv(after-all-1), channel_id=1
    after-all-2 = token[] after-all()
    send-2 = (f32[], u32[], token[]) send(data-2, after-all-2), channel_id=2
    recv-2 = (f32[], u32[], token[]) recv(after-all-2), channel_id=2
    send-done-1 = token[] send-done(send-1), channel_id=1
    recv-done-1 = (f32[], token[]) recv-done(recv-1), channel_id=1,
      control-predecessors={send-1}
    send-done-2 = token[] send-done(send-2), channel_id=2
    recv-done-2 = (f32[], token[]) recv-done(recv-2), channel_id=2,
      control-predecessors={send-2}
    data-out-1 = f32[] get-tuple-element(recv-done-1), index=0
    data-out-2 = f32[] get-tuple-element(recv-done-2), index=0
    ROOT out = (f32[], f32[]) tuple(data-out-1, data-out-2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kHloStr)));
  CollectiveSendRecvCombiner combiner;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combiner.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK: %[[WRAPPED_SEND_RECV:.*]] (param0: f32[], param1: token[],
    CHECK-SAME: param2: f32[], param3: token[], param4: token[], param5: token[]) ->
    CHECK-SAME: ((f32[], u32[], token[]), (f32[], u32[], token[]), (f32[], u32[], token[]),
    CHECK-SAME: (f32[], u32[], token[]))
    CHECK-NEXT: %[[PARAM0:.*]] = {{.*}} parameter(0)
    CHECK: %[[PARAM1:.*]] = {{.*}} parameter(1)
    CHECK: %[[SEND1:.*]] = {{.*}} send(%[[PARAM0]], %[[PARAM1]]), channel_id=1
    CHECK: %[[PARAM2:.*]] = f32[] parameter(2)
    CHECK: %[[PARAM3:.*]] = {{.*}} parameter(3)
    CHECK: %[[SEND2:.*]] = {{.*}} send(%[[PARAM2]], %[[PARAM3]]), channel_id=2
    CHECK: %[[PARAM4:.*]] = {{.*}} parameter(4)
    CHECK: %[[RECV1:.*]] = {{.*}} recv(%[[PARAM4]]), channel_id=1
    CHECK: %[[PARAM5:.*]] = {{.*}} parameter(5)
    CHECK: %[[RECV2:.*]] = {{.*}} recv(%[[PARAM5]]), channel_id=2
    CHECK: ROOT %[[OUT:.*]] = {{.*}} tuple(%[[SEND1]], %[[SEND2]], %[[RECV1]], %[[RECV2]])

    CHECK: ENTRY %[[MAIN:.*]] () -> (f32[], f32[])
    CHECK: %[[DATA1:.*]] = {{.*}} constant(1)
    CHECK: %[[AFTER_ALL1:.*]] = {{.*}} after-all()
    CHECK: %[[DATA2:.*]] = {{.*}} constant(2)
    CHECK: %[[AFTER_ALL2:.*]] = {{.*}} after-all()
    CHECK: %[[TUPLE_START:.*]] = {{.*}} async-start{{.*}}calls=%[[WRAPPED_SEND_RECV]]
    CHECK: %[[TUPLE_DONE:.*]] = {{.*}} async-done(%[[TUPLE_START]])
    CHECK %[[GTE4:.*]] = {{.*}} get-tuple-element(%[[TUPLE_DONE]]), index=2
    CHECK %[[GTE5:.*]] = {{.*}} get-tuple-element(%[[GTE4]]), index=0
    CHECK %[[GTE6:.*]] = {{.*}} get-tuple-element(%[[GTE4]]), index=2
    CHECK %[[[TUPLE1:.*]]]] = {{.*}} tuple(%[[GTE5]], %[[GTE6]]), control-predecessors={%[[TUPLE_START]]]}
    CHECK %[[DATA_OUT1:.*]]] = {{.*}} get-tuple-element(%[[TUPLE1]]), index=0

    CHECK %[[GTE7:.*]]] = {{.*}} get-tuple-element(%[[TUPLE_DONE]]), index=3
    CHECK %[[GTE8:.*]] = {{.*}} get-tuple-element(%[[GTE7]]), index=0
    CHECK %[[GTE9:.*]]] = {{.*}} get-tuple-element(%[[GTE7]]), index=2
    CHECK %[[TUPLE2:.*]] = {{.*}} tuple(%[[GTE8]], %[[GTE9]]), control-predecessors={%[[TUPLE_START]]}
    CHECK %[[DATA_OUT2:.*]] = {{.*}} get-tuple-element(%[[TUPLE2]]), index=0
    CHECK ROOT %[[OUT:.*]] = {{.*}} tuple(%[[DATA_OUT1]], %[[DATA_OUT2]])
    CHECK %[[GTE:.*]] = {{.*}} get-tuple-element(%[[TUPLE_DONE]]), index=0
    CHECK %[[GTE1:.*]] = {{.*}} get-tuple-element(%[[GTE]]), index=2
    CHECK %[[GTE2:.*]] = {{.*}} get-tuple-element(%[[TUPLE_DONE]]), index=1
    CHECK %[[GTE3:.*]] = {{.*}} get-tuple-element(%[[GTE2]]), index=2
  )"));
}
}  // namespace
}  // namespace xla
