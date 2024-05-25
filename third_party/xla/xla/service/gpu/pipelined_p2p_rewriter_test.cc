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

#include "xla/service/gpu/pipelined_p2p_rewriter.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class PipelinedP2pRewriterTest : public HloTestBase {
 protected:
  void DoFileCheck(const HloModule* module, absl::string_view expected) {
    HloPrintOptions options;
    options.set_print_operand_shape(false);
    options.set_print_result_shape(false);
    TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                            RunFileCheck(module->ToString(options), expected));
    EXPECT_TRUE(filecheck_matched);
  }
};

TEST_F(PipelinedP2pRewriterTest, SendRecUnpipelinedNotTransform) {
  const char* kModuleStr = R"(
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

    recv-data = u32[2] get-tuple-element(recv-done.0), index=0

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

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  PipelinedP2PRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
}

// Tests the rewrite for a pipelined Send/Recv chain with only one channel
// group.
TEST_F(PipelinedP2pRewriterTest, SendRecvPipelined1) {
  const char* kModuleStr = R"(
  HloModule test, is_scheduled=true

  while-cond {
    param = (u32[], (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(25)
    ROOT cond-result = pred[] compare(count, ub), direction=LT
  }

  while-body {
    param = (u32[], (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0

    recv-done.q = (f32[1,1024,1024], token[]) get-tuple-element(param), index=1
    recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done.q), index=0

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
    send-data = f32[1, 1024, 1024] add(c, s)

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    send = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all),
      channel_id=1, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.p = (f32[1,1024,1024], token[]) recv-done(recv), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.p = token[] send-done(send), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    gte.0 = f32[1,1024,1024] get-tuple-element(recv-done.p), index=0
    gte.1 = token[] get-tuple-element(recv-done.p), index=1
    recv-done-tuple = (f32[1,1024,1024], token[]) tuple(gte.0, gte.1)
    ROOT body-result = (u32[], (f32[1,1024,1024], token[]), token[])
      tuple(new-count, recv-done-tuple, send-done.p)
  }

  ENTRY main {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all.1 = token[] after-all()
    recv.1 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    send.1 = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.1.p = (f32[1,1024,1024], token[]) recv-done(recv.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.1.p = token[] send-done(send.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    while-init.p =  (u32[], (f32[1,1024,1024], token[]), token[])
      tuple(c0, recv-done.1.p, send-done.1.p)
    while-result.p = (u32[], (f32[1,1024,1024], token[]), token[])
      while(while-init.p),
      body=while-body, condition=while-cond,
      backend_config={"known_trip_count":{"n":"25"}}

    recv-done.1.q = (f32[1,1024,1024], token[]) get-tuple-element(while-result.p), index=1

    ROOT entry-result = f32[1, 1024, 1024] get-tuple-element(recv-done.1.q), index=0
  }
  )";

  const char* kExpected = R"(
  CHECK: %while-body (param.1: (u32[], (f32[1,1024,1024], u32[], token[]), (f32[1,1024,1024], u32[], token[]))) -> (u32[], (f32[1,1024,1024], u32[], token[]), (f32[1,1024,1024], u32[], token[])) {
  CHECK: %param.1 = parameter(0)
  CHECK: %get-tuple-element = get-tuple-element(%param.1), index=1
  CHECK: %get-tuple-element.1 = get-tuple-element(%param.1), index=2
  CHECK: %count.1 = get-tuple-element(%param.1), index=0
  CHECK: %recv-done = recv-done(%get-tuple-element), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}
  CHECK: %recv-data = get-tuple-element(%recv-done), index=0
  CHECK: %c1 = constant(1)
  CHECK: %new-count = add(%count.1, %c1)
  CHECK: %replica = replica-id()
  CHECK: %c10 = constant(10)
  CHECK: %sum = add(%replica, %c10)
  CHECK: %sum2 = add(%sum, %count.1)
  CHECK: %conv = convert(%sum2)
  CHECK: %p = broadcast(%conv), dimensions={}
  CHECK: %b = add(%p, %recv-data)
  CHECK: %c = multiply(%b, %b)
  CHECK: %d = tan(%c)
  CHECK: %s = dot(%c, %d), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
  CHECK: %send-data = add(%c, %s)
  CHECK: %after-all = after-all()
  CHECK: %send-done = send-done(%get-tuple-element.1), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}
  CHECK{LITERAL}: %recv = recv(%after-all), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}"}
  CHECK{LITERAL}: %send = send(%send-data, %after-all), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}"}
  CHECK: ROOT %tuple = tuple(%new-count, %recv, %send)
  CHECK: }

  CHECK: %while-cond (param: (u32[], (f32[1,1024,1024], u32[], token[]), (f32[1,1024,1024], u32[], token[]))) -> pred[] {
  CHECK: %param = parameter(0)
  CHECK: %count = get-tuple-element(%param), index=0
  CHECK: %ub = constant(25)
  CHECK: ROOT %cond-result = compare(%count, %ub), direction=LT
  CHECK: }

  CHECK: ENTRY %main () -> f32[1,1024,1024] {
  CHECK: %c0 = constant(0)
  CHECK: %f0 = constant(0)
  CHECK: %init = broadcast(%f0), dimensions={}
  CHECK: %after-all.1 = after-all()
  CHECK{LITERAL}: %recv.1 = recv(%after-all.1), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}"}
  CHECK{LITERAL}: %send.1 = send(%init, %after-all.1), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}"}
  CHECK: %while-init = tuple(%c0, %recv.1, %send.1)
  CHECK: %while-result = while(%while-init), condition=%while-cond, body=%while-body,
  CHECK-SAME{LITERAL}: backend_config={"known_trip_count":{"n":"25"}}
  CHECK: %get-tuple-element.2 = get-tuple-element(%while-result), index=1
  CHECK: %get-tuple-element.3 = get-tuple-element(%while-result), index=2
  CHECK: %recv-done.1 = recv-done(%get-tuple-element.2), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}
  CHECK: %send-done.1 = send-done(%get-tuple-element.3), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}
  CHECK: ROOT %entry-result = get-tuple-element(%recv-done.1), index=0
  CHECK: })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  PipelinedP2PRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  DoFileCheck(module.get(), kExpected);
}

// Repeats the Send/Recv pattern in the previous test, to test that we can
// rewrite a routine with multiple pipelined loops without crashing.
TEST_F(PipelinedP2pRewriterTest, SendRecvTwoPipelinedWhileLoops) {
  const char* kModuleStr = R"(
  HloModule test, is_scheduled=true

  while-cond {
    param = (u32[], (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(25)
    ROOT cond-result = pred[] compare(count, ub), direction=LT
  }

  while-body {
    param = (u32[], (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0

    recv-done.q = (f32[1,1024,1024], token[]) get-tuple-element(param), index=1
    send-data = f32[1, 1024, 1024] get-tuple-element(recv-done.q), index=0

    c1 = u32[] constant(1)
    new-count = u32[] add(count, c1)

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    send = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all),
      channel_id=1, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.p = (f32[1,1024,1024], token[]) recv-done(recv), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.p = token[] send-done(send), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    gte.0 = f32[1,1024,1024] get-tuple-element(recv-done.p), index=0
    gte.1 = token[] get-tuple-element(recv-done.p), index=1
    recv-done-tuple = (f32[1,1024,1024], token[]) tuple(gte.0, gte.1)
    ROOT body-result = (u32[], (f32[1,1024,1024], token[]), token[])
      tuple(new-count, recv-done-tuple, send-done.p)
  }

  while-cond-2 {
    param = (u32[], (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(25)
    ROOT cond-result = pred[] compare(count, ub), direction=LT
  }

  while-body-2 {
    param = (u32[], (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0

    recv-done.q = (f32[1,1024,1024], token[]) get-tuple-element(param), index=1
    send-data = f32[1, 1024, 1024] get-tuple-element(recv-done.q), index=0

    c1 = u32[] constant(1)
    new-count = u32[] add(count, c1)

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    send = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all),
      channel_id=1, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.p = (f32[1,1024,1024], token[]) recv-done(recv), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.p = token[] send-done(send), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    gte.0 = f32[1,1024,1024] get-tuple-element(recv-done.p), index=0
    gte.1 = token[] get-tuple-element(recv-done.p), index=1
    recv-done-tuple = (f32[1,1024,1024], token[]) tuple(gte.0, gte.1)
    ROOT body-result = (u32[], (f32[1,1024,1024], token[]), token[])
      tuple(new-count, recv-done-tuple, send-done.p)
  }

  ENTRY main {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all.1 = token[] after-all()
    recv.1 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    send.1 = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.1.p = (f32[1,1024,1024], token[]) recv-done(recv.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.1.p = token[] send-done(send.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    while-init.p =  (u32[], (f32[1,1024,1024], token[]), token[])
      tuple(c0, recv-done.1.p, send-done.1.p)
    while-result.p = (u32[], (f32[1,1024,1024], token[]), token[])
      while(while-init.p),
      body=while-body, condition=while-cond,
      backend_config={"known_trip_count":{"n":"25"}}

    recv-done.1.q = (f32[1,1024,1024], token[]) get-tuple-element(while-result.p), index=1

    after-all-2.1 = token[] after-all()
    recv-2.1 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all-2.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    send-2.1 = (f32[1, 1024, 1024], u32[], token[]) send(recv-done.1.q, after-all-2.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done-2.1.p = (f32[1,1024,1024], token[]) recv-done(recv-2.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done-2.1.p = token[] send-done(send-2.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    while-init-2.p =  (u32[], (f32[1,1024,1024], token[]), token[])
      tuple(c0, recv-done-2.1.p, send-done-2.1.p)
    while-result-2.p = (u32[], (f32[1,1024,1024], token[]), token[])
      while(while-init-2.p),
      body=while-body-2, condition=while-cond-2,
      backend_config={"known_trip_count":{"n":"25"}}

    recv-done-2.1.q = (f32[1,1024,1024], token[]) get-tuple-element(while-result-2.p), index=1

    ROOT entry-result = f32[1, 1024, 1024] get-tuple-element(recv-done-2.1.q), index=0
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  PipelinedP2PRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  // Check that we transform the module without crashing.
  EXPECT_TRUE(changed);
}

// Tests the rewrite for a pipelined Send/Recv chain with two channel groups.
TEST_F(PipelinedP2pRewriterTest, SendRecvPipelined2) {
  const char* kModuleStr = R"(
  HloModule test, is_scheduled=true

  while-cond {
    param = (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(25)
    ROOT cond-result = pred[] compare(count, ub), direction=LT
  }

  while-body {
    param = (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0

    recv-done.0.q = (f32[1,1024,1024], token[]) get-tuple-element(param), index=1
    recv-data.0 = f32[1, 1024, 1024] get-tuple-element(recv-done.0.q), index=0
    recv-done.1.q = (f32[1,1024,1024], token[]) get-tuple-element(param), index=3
    recv-data.1 = f32[1, 1024, 1024] get-tuple-element(recv-done.1.q), index=0

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=EQ
    compare = pred[1, 1024, 1024] broadcast(compare0), dimensions={}
    recv-data = f32[1, 1024, 1024] select(compare, recv-data.0, recv-data.1)

    c1 = u32[] constant(1)
    new-count = u32[] add(count, c1)
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
    send-data = f32[1, 1024, 1024] add(c, s)

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_pipeline="0"
      }
    send = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all),
      channel_id=1, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.p = (f32[1,1024,1024], token[]) recv-done(recv), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.p = token[] send-done(send), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }

    after-all.1 = token[] after-all()
    recv.1 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}",
        _xla_send_recv_pipeline="1"
      }
    send.1 = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all.1),
      channel_id=2, frontend_attributes={
       _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}",
       _xla_send_recv_pipeline="1"
      }
    recv-done.1.p = (f32[1,1024,1024], token[]) recv-done(recv.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }
    send-done.1.p = token[] send-done(send.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }

    ROOT body-result = (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[])
      tuple(new-count, recv-done.p, send-done.p, recv-done.1.p, send-done.1.p)
  }

  ENTRY main {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all.2 = token[] after-all()
    recv.2 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.2), channel_id=1,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{3,0}}",
       _xla_send_recv_pipeline="0"
    }
    send.2 = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all.2), channel_id=1,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{3,0}}",
       _xla_send_recv_pipeline="0"
    }
    recv-done.2.p = (f32[1,1024,1024], token[]) recv-done(recv.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.2.p = token[] send-done(send.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }

    after-all.3 = token[] after-all()
    recv.3 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.3), channel_id=2,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}",
       _xla_send_recv_pipeline="1"
    }
    send.3 = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all.3), channel_id=2,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}",
       _xla_send_recv_pipeline="1"
    }
    recv-done.3.p = (f32[1,1024,1024], token[]) recv-done(recv.3), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }
    send-done.3.p = token[] send-done(send.3), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }

    while-init.p =  (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[]) tuple(c0, recv-done.2.p, send-done.2.p, recv-done.3.p, send-done.3.p)
    while-result.p = (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[]) while(while-init.p),
      body=while-body, condition=while-cond,
      backend_config={"known_trip_count":{"n":"25"}}

    recv-done.2.q = (f32[1,1024,1024], token[]) get-tuple-element(while-result.p), index=1
    recv-data.2 = f32[1, 1024, 1024] get-tuple-element(recv-done.2.q), index=0
    recv-done.3.q = (f32[1,1024,1024], token[]) get-tuple-element(while-result.p), index=3
    recv-data.3 = f32[1, 1024, 1024] get-tuple-element(recv-done.3.q), index=0

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=EQ
    compare = pred[1, 1024, 1024] broadcast(compare0), dimensions={}
    ROOT entry-result = f32[1, 1024, 1024] select(compare, recv-data.2, recv-data.3)
  }
  )";

  const char* kExpected = R"(
  CHECK: %while-body (param.1: (u32[], (f32[1,1024,1024], u32[], token[]), (f32[1,1024,1024], u32[], token[]), (f32[1,1024,1024], u32[], token[]), (f32[1,1024,1024], u32[], token[]))) -> (u32[], (f32[1,1024,1024], u32[], token[]), (f32[1,1024,1024], u32[], token[]), (f32[1,1024,1024], u32[], token[]), (f32[1,1024,1024], u32[], token[])) {
  CHECK: %param.1 = parameter(0)
  CHECK: %get-tuple-element = get-tuple-element(%param.1), index=1
  CHECK: %get-tuple-element.1 = get-tuple-element(%param.1), index=2
  CHECK: %get-tuple-element.2 = get-tuple-element(%param.1), index=3
  CHECK: %get-tuple-element.3 = get-tuple-element(%param.1), index=4
  CHECK: %count.1 = get-tuple-element(%param.1), index=0
  CHECK: %recv-done = recv-done(%get-tuple-element), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}
  CHECK: %recv-data.0 = get-tuple-element(%recv-done), index=0
  CHECK: %recv-done.1 = recv-done(%get-tuple-element.2), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1"}
  CHECK: %recv-data.1 = get-tuple-element(%recv-done.1), index=0
  CHECK: %replica = replica-id()
  CHECK: %constant0 = constant(0)
  CHECK: %compare0 = compare(%replica, %constant0), direction=EQ
  CHECK: %compare = broadcast(%compare0), dimensions={}
  CHECK: %recv-data.2 = select(%compare, %recv-data.0, %recv-data.1)
  CHECK: %c1 = constant(1)
  CHECK: %new-count = add(%count.1, %c1)
  CHECK: %c10 = constant(10)
  CHECK: %sum = add(%replica, %c10)
  CHECK: %sum2 = add(%sum, %count.1)
  CHECK: %conv = convert(%sum2)
  CHECK: %p = broadcast(%conv), dimensions={}
  CHECK: %b = add(%p, %recv-data.2)
  CHECK: %c = multiply(%b, %b)
  CHECK: %d = tan(%c)
  CHECK: %s = dot(%c, %d), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
  CHECK: %send-data = add(%c, %s)
  CHECK: %after-all = after-all()
  CHECK: %send-done = send-done(%get-tuple-element.1), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}
  CHECK: %send-done.1 = send-done(%get-tuple-element.3), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1"}
  CHECK{LITERAL}: %recv = recv(%after-all), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs="{{3,0}}"}
  CHECK{LITERAL}: %send = send(%send-data, %after-all), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs="{{3,0}}"}
  CHECK: %after-all.1 = after-all()
  CHECK{LITERAL}: %recv.1 = recv(%after-all.1), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}"}
  CHECK{LITERAL}: %send.1 = send(%send-data, %after-all.1), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}"}
  CHECK: ROOT %tuple = tuple(%new-count, %recv, %send, %recv.1, %send.1)
  CHECK: }

  CHECK: %while-cond (param: (u32[], (f32[1,1024,1024], u32[], token[]), (f32[1,1024,1024], u32[], token[]), (f32[1,1024,1024], u32[], token[]), (f32[1,1024,1024], u32[], token[]))) -> pred[] {
  CHECK: %param = parameter(0)
  CHECK: %count = get-tuple-element(%param), index=0
  CHECK: %ub = constant(25)
  CHECK: ROOT %cond-result = compare(%count, %ub), direction=LT
  CHECK: }

  CHECK: ENTRY %main () -> f32[1,1024,1024] {
  CHECK: %c0 = constant(0)
  CHECK: %f0 = constant(0)
  CHECK: %init = broadcast(%f0), dimensions={}
  CHECK: %after-all.2 = after-all()
  CHECK{LITERAL}: %recv.2 = recv(%after-all.2), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs="{{3,0}}"}
  CHECK{LITERAL}: %send.2 = send(%init, %after-all.2), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs="{{3,0}}"}
  CHECK: %after-all.3 = after-all()
  CHECK{LITERAL}: %recv.3 = recv(%after-all.3), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}"}
  CHECK{LITERAL}: %send.3 = send(%init, %after-all.3), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}"}
  CHECK: %while-init = tuple(%c0, %recv.2, %send.2, %recv.3, %send.3)
  CHECK{LITERAL}: %while-result = while(%while-init), condition=%while-cond, body=%while-body, backend_config={"known_trip_count":{"n":"25"}}
  CHECK: %get-tuple-element.4 = get-tuple-element(%while-result), index=1
  CHECK: %get-tuple-element.5 = get-tuple-element(%while-result), index=2
  CHECK: %get-tuple-element.6 = get-tuple-element(%while-result), index=3
  CHECK: %get-tuple-element.7 = get-tuple-element(%while-result), index=4
  CHECK: %recv-done.2 = recv-done(%get-tuple-element.4), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}
  CHECK: %recv-data.3 = get-tuple-element(%recv-done.2), index=0
  CHECK: %recv-done.3 = recv-done(%get-tuple-element.6), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1"}
  CHECK: %recv-data.4 = get-tuple-element(%recv-done.3), index=0
  CHECK: %replica.1 = replica-id()
  CHECK: %constant0.1 = constant(0)
  CHECK: %compare0.1 = compare(%replica.1, %constant0.1), direction=EQ
  CHECK: %compare.1 = broadcast(%compare0.1), dimensions={}
  CHECK: %send-done.2 = send-done(%get-tuple-element.5), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}
  CHECK: %send-done.3 = send-done(%get-tuple-element.7), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1"}
  CHECK: ROOT %entry-result = select(%compare.1, %recv-data.3, %recv-data.4)
  CHECK: })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  PipelinedP2PRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  DoFileCheck(module.get(), kExpected);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
