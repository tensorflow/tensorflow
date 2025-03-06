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
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

namespace m = xla::match;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class GpuP2PPipelinerTest : public HloTestBase {
 public:
  GpuP2PPipelinerTest() {
    const int64_t kNumReplicas = 1;
    const int64_t kNumPartitions = 4;
    config_ = GetModuleConfigForTest(/*replica_count=*/kNumReplicas,
                                     /*num_partitions=*/kNumPartitions);
  }

  absl::StatusOr<bool> RunOptimizer(
      HloModule* module, bool enable_partial_send_recv_pipelining = false) {
    HloPassPipeline pipeline("optimizer");
    pipeline.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                                  /*allow_mixed_precision=*/false);
    pipeline.AddPass<GpuP2PPipeliner>(enable_partial_send_recv_pipelining);
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
    recv_data = u32[2] get-tuple-element(recv-done.0), index=0

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv_data)

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
  const char* kRotatedAttr = "_xla_send_recv_validation={{0,6}}";
  EXPECT_THAT(send1->ToString(), ::testing::HasSubstr(kPeeledAttr));
  EXPECT_THAT(recv1->ToString(), ::testing::HasSubstr(kPeeledAttr));
  EXPECT_THAT(send2->ToString(), ::testing::HasSubstr(kRotatedAttr));
  EXPECT_THAT(recv2->ToString(), ::testing::HasSubstr(kRotatedAttr));
}

TEST_F(GpuP2PPipelinerTest, SendRecvForwardCycle) {
  const char* kHloStr = R"(
  HloModule test

  while_body {
    inputs = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(inputs), index=0
    iter_increment = u32[] constant(1)
    next_iter = u32[] add(iter, iter_increment)
    weights = f32[2,2] get-tuple-element(inputs), index=2
    partition-id = u32[] partition-id()
    zero = u32[] constant(0)
    compare = pred[] compare(partition-id, zero), direction=EQ
    broadcast = pred[2,2] broadcast(compare), dimensions={}
    data = f32[2,2] get-tuple-element(inputs), index=1
    after-all = token[] after-all()

    send = (f32[2,2], u32[], token[]) send(data, after-all), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0",
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_validation="{{3,10}}"
      }
    recv = (f32[2,2], u32[], token[]) recv(after-all), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0",
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_validation="{{3,10}}"
      }
    recv-done = (f32[2,2], token[]) recv-done(recv), channel_id=1,
      frontend_attributes={_xla_send_recv_pipeline="0"}, control-predecessors={send}
    recv-done-data = f32[2,2] get-tuple-element(recv-done), index=0
    after-all.1 = token[] after-all()
    send.1 = (f32[2,2], u32[], token[]) send(data, after-all.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1",
        _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3}}",
        _xla_send_recv_validation="{{0,7},{1,8},{2,9}}"
      }
    recv.1 = (f32[2,2], u32[], token[]) recv(after-all.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1",
        _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3}}",
        _xla_send_recv_validation="{{0,7},{1,8},{2,9}}"
      }
    recv-done.1 = (f32[2,2], token[]) recv-done(recv.1), channel_id=2,
      frontend_attributes={_xla_send_recv_pipeline="1"}, control-predecessors={send.1}
    recv-done-1-data = f32[2,2] get-tuple-element(recv-done.1), index=0
    select = f32[2,2] select(broadcast, recv-done-data, recv-done-1-data)
    matmul = f32[2,2] dot(weights, select),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}

    ROOT result = (u32[], f32[2,2], f32[2,2]) tuple(next_iter, matmul, weights)

    send-done = token[] send-done(send), channel_id=1,
      frontend_attributes={_xla_send_recv_pipeline="0"}
    send-done.1 = token[] send-done(send.1), channel_id=2,
      frontend_attributes={_xla_send_recv_pipeline="1"}
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
    input = (u32[], f32[2,2], f32[2,2]) tuple(start_iter, input_data, input_weights)
    while_result = (u32[], f32[2,2], f32[2,2]) while(input), condition=while_cond, body=while_body
    ROOT data_out = f32[2,2] get-tuple-element(while_result), index=1
  }
  )";
  auto module = ParseAndReturnUnverifiedModule(kHloStr, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get()).value());
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
    // Check there are two sets of send/recv in main while loop, one set for the
    // back edge and one set for the forward edge. Also check that the send/recv
    // target pairs and validation attributes are correct.
    CHECK: %[[RECV_BWD_START:.*]] = {{.*}} after-all()
    CHECK: %[[RECV_BWD:.*]] = {{.*}} recv(token[] %[[RECV_BWD_START:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{[{][{]}}3,0{{[}][}]}},_xla_send_recv_validation={{[{][{]}}2,9{{[}][}]}}}
    CHECK: %[[RECV_DONE_BWD:.*]] = {{.*}} recv-done({{.*}} %[[RECV_BWD:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}
    CHECK: %[[RECV_FWD_START:.*]] = {{.*}} after-all()
    CHECK: %[[RECV_FWD:.*]] = {{.*}} recv(token[] %[[RECV_FWD_START:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{[{][{]}}0,1},{1,2},{2,3{{[}][}]}},_xla_send_recv_validation={{[{][{]}}0,6},{0,7},{1,8{{[}][}]}}}
    CHECK: %[[RECV_DONE_FWD:.*]] = {{.*}} recv-done((f32[2,2]{1,0}, u32[], token[]) %[[RECV_FWD:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1"}
    CHECK: %[[SEND_BWD:.*]] = {{.*}} send({{.*}} %[[RECV_BWD_START:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{[{][{]}}3,0{{[}][}]}},_xla_send_recv_validation={{[{][{]}}2,9{{[}][}]}}}
    CHECK: %[[SEND_DONE_BWD:.*]] = {{.*}} send-done({{.*}} %[[SEND_BWD:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}
    CHECK: %[[SEND_FWD:.*]] = {{.*}} send({{.*}} %[[RECV_FWD_START:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{[{][{]}}0,1},{1,2},{2,3{{[}][}]}},_xla_send_recv_validation={{[{][{]}}0,6},{0,7},{1,8{{[}][}]}}}
    CHECK: %[[SEND_DONE_FWD:.*]] = {{.*}} send-done({{.*}} %[[SEND_FWD:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1"}
    // Check that the total iterations of the while loop in the output is 1
    // fewer than the max iteration of the input HLO.
    CHECK: %[[WHILE_COND:.*]] (cond_param: {{.*}}
    CHECK-NEXT: %[[COND_PARAM:.*]] = {{.*}} parameter(0)
    CHECK: %[[CURRENT_ITER:.*]] = {{.*}} get-tuple-element({{.*}} %[[COND_PARAM:.*]]), index=0
    CHECK: %[[TWO:.*]] = {{.*}} constant(2)
    CHECK: ROOT %[[COMPARE:.*]] = pred[] compare({{.*}} %[[CURRENT_ITER:.*]], {{.*}} %[[TWO:.*]]), direction=LT

    // Check that after transformation, main function in ENTRY contains the
    // first iteration of the while loop.
    CHECK: ENTRY %[[TEST_COMPUTATION:.*]] (input_data: {{.*}}

    // Set up dummy send and recv.
    CHECK: %[[RECV_BWD_DUMMY_START:.*]] = {{.*}} after-all()
    CHECK: %[[RECV_BWD_DUMMY:.*]] = {{.*}} recv(token[] %[[RECV_BWD_DUMMY_START:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{[{][{]}}3,0{{[}][}]}},_xla_send_recv_validation="invalid"}
    CHECK: %[[RECV_DONE_BWD_DUMMY:.*]] = {{.*}} recv-done({{.*}} %[[RECV_BWD_DUMMY:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}

    // Execute what was previously iter 0 of the while loop.
    CHECK: %[[RECV_FWD_FIRST_ITER_START:.*]] = {{.*}} after-all()
    CHECK: %[[RECV_FWD_FIRST_ITER:.*]] = {{.*}} recv(token[] %[[RECV_FWD_FIRST_ITER_START:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{[{][{]}}0,1},{1,2},{2,3{{[}][}]}},_xla_send_recv_validation={{[{][{]}}0,0},{1,0},{1,0{{[}][}]}}}
    CHECK: %[[RECV_DONE_FWD_FIRST_ITER:.*]] = {{.*}} recv-done({{.*}} %[[RECV_FWD_FIRST_ITER:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1"}
    CHECK: %[[SEND_BWD_DUMMY:.*]] = {{.*}} send({{.*}} %[[RECV_DUMMY_START:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{[{][{]}}3,0{{[}][}]}},_xla_send_recv_validation="invalid"}
    CHECK: %[[SEND_DONE_BWD_DUMMY:.*]] = {{.*}} send-done({{.*}} %[[SEND_BWD_DUMMY:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}
    CHECK: %[[SEND_FWD_FIRST_ITER:.*]] = {{.*}} send({{.*}} %[[RECV_FWD_FIRST_ITER_START:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{[{][{]}}0,1},{1,2},{2,3{{[}][}]}},_xla_send_recv_validation={{[{][{]}}0,0},{1,0},{1,0{{[}][}]}}}
    CHECK: %[[SEND_DONE_FWD_FIRST_ITER:.*]] = {{.*}} send-done({{.*}} %[[SEND_FWD_FIRST_ITER:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1"}

    // Set up main loop, starting from iter 1.
    CHECK: %[[START_LOOP_FROM_ITER_ONE:.*]] = u32[] constant(1)
    CHECK: %[[LOOP_INPUT:.*]] = {{.*}} tuple({{.*}} %[[START_LOOP_FROM_ITER_ONE:.*]])
    CHECK: %[[WHILE:.*]] = {{.*}} while({{.*}} %[[LOOP_INPUT:.*]]), {{.*}}
  )")
                  .value());
}

// Expect leading recv and recv-done to be pipelined but none of the other
// send/recv ops.
TEST_F(GpuP2PPipelinerTest, PipelineParallelismExperimentalOpt) {
  const char* kHloStr = R"(
    HloModule test

    while_cond {
      inputs = (u32[], f32[2,2], f32[2,2]) parameter(0)
      iter = u32[] get-tuple-element(inputs), index=0
      max_iter = u32[] constant(3)
      ROOT compare = pred[] compare(iter, max_iter), direction=LT
    }

    while_body {
      inputs = (u32[], f32[2,2], f32[2,2]) parameter(0)
      iter = u32[] get-tuple-element(inputs), index=0
      iter_increment = u32[] constant(1)
      next_iter = u32[] add(iter, iter_increment)
      weights = f32[2,2] get-tuple-element(inputs), index=2
      partition-id = u32[] partition-id()
      zero = u32[] constant(0)
      compare = pred[] compare(partition-id, zero), direction=EQ
      broadcast = pred[2,2] broadcast(compare), dimensions={}
      data = f32[2,2] get-tuple-element(inputs), index=1
      after-all = token[] after-all()
      recv = (f32[2,2], u32[], token[]) recv(after-all), channel_id=1,
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
      recv-done = (f32[2,2], token[]) recv-done(recv), channel_id=1
      send = (f32[2,2], u32[], token[]) send(data, after-all), channel_id=1,
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}},
          control-predecessors={recv}
      send-done = token[] send-done(send), channel_id=1,
          control-predecessors={recv-done}
      after-all.1 = token[] after-all()
      recv.1 = (f32[2,2], u32[], token[]) recv(after-all.1), channel_id=2,
          frontend_attributes={_xla_send_recv_source_target_pairs={{3,0}}},
          control-predecessors={send}
      recv-done.1 = (f32[2,2], token[]) recv-done(recv.1), channel_id=2,
          control-predecessors={send-done}
      send.1 = (f32[2,2], u32[], token[]) send(data, after-all.1), channel_id=2,
          frontend_attributes={_xla_send_recv_source_target_pairs={{3,0}}},
          control-predecessors={recv.1}
      send-done.1 = token[] send-done(send.1), channel_id=2,
          control-predecessors={recv-done.1}
      recv-done-data = f32[2,2] get-tuple-element(recv-done), index=0
      recv-done-1-data = f32[2,2] get-tuple-element(recv-done.1), index=0
      select = f32[2,2] select(broadcast, recv-done-data, recv-done-1-data)
      matmul = f32[2,2] dot(weights, select),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT result = (u32[], f32[2,2], f32[2,2]) tuple(next_iter, matmul,
          weights)
    }

    ENTRY test_computation {
      start_iter = u32[] constant(0)
      input_data = f32[2,2] parameter(0)
      input_weights = f32[2,2] parameter(1)
      input = (u32[], f32[2,2], f32[2,2]) tuple(start_iter, input_data,
      input_weights) while_result = (u32[], f32[2,2], f32[2,2]) while(input),
      condition=while_cond, body=while_body ROOT data_out = f32[2,2]
      get-tuple-element(while_result), index=1
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kHloStr, config_));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, RunOptimizer(module.get(),
                                 /*enable_partial_send_recv_pipelining=*/true));
  EXPECT_TRUE(changed);

  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
    // Check that recv and recv-done ops are rotated in loop body.
    // CHECK:      %[[WHILE_BODY:while_body[\w\.]+]]
    // CHECK:        send(
    // CHECK:        send-done(
    // CHECK:        recv(
    // CHECK:        recv-done(
    // CHECK:        send(
    // CHECK:        send-done(
    // CHECK:        recv(
    // CHECK:        recv-done(

    // Check that recv and recv-done ops are peeled out of the while loop.
    // CHECK:      ENTRY %test_computation
    // CHECK:        recv(
    // CHECK:        recv-done(
    // CHECK:        while(
    // CHECK-SAME:   body=%[[WHILE_BODY]]
    // CHECK:        send(
    // CHECK:        send-done(
    // CHECK:        recv(
    // CHECK:        recv-done(
    // CHECK:        send(
    // CHECK:        send-done(
    )")
                  .value());
}

TEST_F(GpuP2PPipelinerTest, OneSendRecvWithOneConflictingAllReduce) {
  const char* kHloStr = R"(
    HloModule test

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    cond {
      param = (u32[], f32[64], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      n = u32[] constant(2)
      ROOT result = pred[] compare(i, n), direction=LT
    }

    body {
      param = (u32[], f32[64], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data_a = f32[64] get-tuple-element(param), index=1
      data_b = f32[64] get-tuple-element(param), index=2

      // Decomposed cp_fwd.
      after-all = token[] after-all()
      recv = (f32[64], u32[], token[]) recv(after-all), channel_id=1,
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
      send = (f32[64], u32[], token[]) send(data_a, after-all), channel_id=1,
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}},
          control-predecessors={recv}
      recv_done = (f32[64], token[]) recv-done(recv), channel_id=1
      send_done = token[] send-done(send), channel_id=1
      recv_data = f32[64] get-tuple-element(recv_done), index=0

      // Conflicting all-reduce.
      ar = f32[64] all-reduce(data_b), channel_id=2, replica_groups={{0,1,2,3}},
          to_apply=add, control-predecessors={send_done}

      c1 = u32[] constant(1)
      i_ = u32[] add(u32[] i, u32[] c1)

      ROOT result = (u32[], f32[64], f32[64]) tuple(i_, recv_data, ar)
    }

    ENTRY entry {
      c0 = u32[] constant(0)
      a = f32[] constant(42)
      data = f32[64] broadcast(a), dimensions={}
      while_init = (u32[], f32[64], f32[64]) tuple(c0, data, data)
      ROOT result = (u32[], f32[64], f32[64]) while(while_init), condition=cond,
         body=body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kHloStr, config_));

  // Run pass.
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunOptimizer(module.get(), /*enable_partial_send_recv_pipelining=*/true));
  EXPECT_TRUE(changed);

  // Find while loop.
  HloInstruction* while_op = FindInstruction(module.get(), "while");
  HloComputation* body = while_op->while_body();

  // Find ops in while loop body.
  HloInstruction* send_op = FindInstruction(module.get(), "send.1");
  HloInstruction* send_done_op = FindInstruction(module.get(), "send_done.1");
  HloInstruction* ar_op = FindInstruction(module.get(), "ar.1");
  HloInstruction* recv_op = FindInstruction(module.get(), "recv.2");
  HloInstruction* recv_done_op = FindInstruction(module.get(), "recv_done.2");
  EXPECT_EQ(send_op->parent(), body);
  EXPECT_EQ(send_done_op->parent(), body);
  EXPECT_EQ(ar_op->parent(), body);
  EXPECT_EQ(recv_op->parent(), body);
  EXPECT_EQ(recv_done_op->parent(), body);

  // Expect control dependencies from rotated recv and recv-done to conflicting
  // all-reduce.
  EXPECT_THAT(send_op->control_predecessors(), IsEmpty());
  EXPECT_THAT(send_done_op->control_predecessors(), IsEmpty());
  EXPECT_THAT(ar_op->control_predecessors(),
              UnorderedElementsAre(send_done_op));
  EXPECT_THAT(recv_op->control_predecessors(), IsEmpty());
  EXPECT_THAT(recv_done_op->control_predecessors(),
              UnorderedElementsAre(send_op, ar_op));
}

TEST_F(GpuP2PPipelinerTest, OneSendRecvWithConflictingAllReduceAfterLoop) {
  const char* kHloStr = R"(
    HloModule test

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    cond {
      param = (u32[], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      n = u32[] constant(2)
      ROOT result = pred[] compare(i, n), direction=LT
    }

    body {
      param = (u32[], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data = f32[64] get-tuple-element(param), index=1

      // Decomposed cp_fwd.
      after-all = token[] after-all()
      recv = (f32[64], u32[], token[]) recv(after-all), channel_id=1,
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
      recv_done = (f32[64], token[]) recv-done(recv), channel_id=1
      send = (f32[64], u32[], token[]) send(data, after-all), channel_id=1,
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}},
          control-predecessors={recv}
      send_done = token[] send-done(send), channel_id=1,
          control-predecessors={recv_done}
      recv_data = f32[64] get-tuple-element(recv_done), index=0

      c1 = u32[] constant(1)
      i_ = u32[] add(u32[] i, u32[] c1)

      ROOT result = (u32[], f32[64]) tuple(i_, recv_data)
    }

    ENTRY entry {
      c0 = u32[] constant(0)
      a = f32[] constant(42)
      data = f32[64] broadcast(a), dimensions={}
      while_init = (u32[], f32[64]) tuple(c0, data)
      while = (u32[], f32[64]) while(while_init), condition=cond,
         body=body

      // Conflicting all-reduce after loop.
      while_dep_data = f32[64] get-tuple-element(while), index=1
      ROOT ar = f32[64] all-reduce(while_dep_data), channel_id=3,
          replica_groups={{0,1,2,3}}, to_apply=add
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kHloStr, config_));

  // Run pass.
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, RunOptimizer(module.get(),
                                 /*enable_partial_send_recv_pipelining=*/true));
  EXPECT_TRUE(changed);

  // Find while loop.
  HloInstruction* while_op = FindInstruction(module.get(), "while.1");
  HloComputation* body = while_op->while_body();

  // Find ops in while loop body.
  HloInstruction* send_op = FindInstruction(module.get(), "send.1");
  HloInstruction* send_done_op = FindInstruction(module.get(), "send_done.1");
  HloInstruction* recv_op = FindInstruction(module.get(), "recv.2");
  HloInstruction* recv_done_op = FindInstruction(module.get(), "recv_done.2");
  EXPECT_EQ(send_op->parent(), body);
  EXPECT_EQ(send_done_op->parent(), body);
  EXPECT_THAT(send_done_op, GmockMatch(m::SendDone(m::Op().Is(send_op))));
  EXPECT_EQ(recv_op->parent(), body);
  EXPECT_EQ(recv_done_op->parent(), body);
  EXPECT_THAT(recv_done_op, GmockMatch(m::RecvDone(m::Op().Is(recv_op))));

  // Find peeled ops before while loop.
  HloInstruction* peeled_recv_op = FindInstruction(module.get(), "recv.1");
  HloInstruction* peeled_recv_done_op =
      FindInstruction(module.get(), "recv_done.1");
  EXPECT_THAT(peeled_recv_done_op,
              GmockMatch(m::RecvDone(m::Op().Is(peeled_recv_op))));

  // Find peeled ops after while loop.
  HloInstruction* peeled_send_op = FindInstruction(module.get(), "send.2");
  HloInstruction* peeled_send_done_op =
      FindInstruction(module.get(), "send_done.2");
  EXPECT_THAT(peeled_send_done_op,
              GmockMatch(m::SendDone(m::Op().Is(peeled_send_op))));

  // Find conflicting all-reduce after loop.
  HloInstruction* ar_op = FindInstruction(module.get(), "ar");

  // Expect all peeled ops to have control dependencies to teh conflicting
  // all-reduce after the loop.
  EXPECT_THAT(ar_op->control_predecessors(),
              UnorderedElementsAre(peeled_send_done_op, peeled_recv_done_op));
}

TEST_F(GpuP2PPipelinerTest,
       OneSendRecvWithConflictingAllReduceBeforeAndAfterLoop) {
  const char* kHloStr = R"(
    HloModule test

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    cond {
      param = (u32[], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      n = u32[] constant(2)
      ROOT result = pred[] compare(i, n), direction=LT
    }

    body {
      param = (u32[], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data = f32[64] get-tuple-element(param), index=1

      // Decomposed cp_fwd.
      after-all = token[] after-all()
      recv = (f32[64], u32[], token[]) recv(after-all), channel_id=1,
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
      recv_done = (f32[64], token[]) recv-done(recv), channel_id=1
      send = (f32[64], u32[], token[]) send(data, after-all), channel_id=1,
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}},
          control-predecessors={recv}
      send_done = token[] send-done(send), channel_id=1,
          control-predecessors={recv_done}
      recv_data = f32[64] get-tuple-element(recv_done), index=0

      c1 = u32[] constant(1)
      i_ = u32[] add(u32[] i, u32[] c1)

      ROOT result = (u32[], f32[64]) tuple(i_, recv_data)
    }

    ENTRY entry {
      c0 = u32[] constant(0)
      a = f32[] constant(42)
      data = f32[64] broadcast(a), dimensions={}

      // Conflicting all-reduce before loop.
      ar = f32[64] all-reduce(data), channel_id=2, replica_groups={{0,1,2,3}},
          to_apply=add

      while_init = (u32[], f32[64]) tuple(c0, ar)
      while = (u32[], f32[64]) while(while_init), condition=cond,
         body=body

      // Conflicting all-reduce after loop.
      while_dep_data = f32[64] get-tuple-element(while), index=1
      ROOT final_ar = f32[64] all-reduce(while_dep_data), channel_id=3,
          replica_groups={{0,1,2,3}}, to_apply=add
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kHloStr, config_));

  // Run pass.
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, RunOptimizer(module.get(),
                                 /*enable_partial_send_recv_pipelining=*/true));
  EXPECT_TRUE(changed);

  // Find ops around the new while loop.
  HloInstruction* ar_op = FindInstruction(module.get(), "ar");
  HloInstruction* recv_op = FindInstruction(module.get(), "recv.1");
  HloInstruction* recv_done_op = FindInstruction(module.get(), "recv_done.1");
  EXPECT_THAT(recv_done_op, GmockMatch(m::RecvDone(m::Op().Is(recv_op))));
  HloInstruction* while_op = FindInstruction(module.get(), "while.1");
  EXPECT_THAT(while_op, GmockMatch(m::While(m::Tuple(m::Op(), m::Op().Is(ar_op),
                                                     m::Op().Is(recv_done_op),
                                                     m::Op()))));
  HloInstruction* send_op = FindInstruction(module.get(), "send.2");
  HloInstruction* send_done_op = FindInstruction(module.get(), "send_done.2");
  EXPECT_THAT(send_done_op, GmockMatch(m::SendDone(m::Op().Is(send_op))));
  HloInstruction* final_ar_op = FindInstruction(module.get(), "final_ar");

  // Expect control dependency from conflicting all-reduce before the while loop
  // to peeled recv (before the new loop) and peeled send (after the new loop).
  // Also, expect control dependency from peeled recv-done and peeled send-done
  // (before and after the new loop) to conflicting all-reduce after the loop.
  EXPECT_THAT(recv_op->control_predecessors(), UnorderedElementsAre(ar_op));
  EXPECT_THAT(send_op->control_predecessors(),
              UnorderedElementsAre(ar_op, recv_op, recv_done_op));
  EXPECT_THAT(final_ar_op->control_predecessors(),
              UnorderedElementsAre(recv_done_op, send_done_op));
}

TEST_F(GpuP2PPipelinerTest, TwoLoopsWithConflictingAllReduces) {
  const char* kHloStr = R"(
    HloModule test

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    cond {
      param = (u32[], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      n = u32[] constant(2)
      ROOT result = pred[] compare(i, n), direction=LT
    }

    body {
      param = (u32[], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data = f32[64] get-tuple-element(param), index=1

      // Decomposed cp_fwd.
      after-all = token[] after-all()
      recv = (f32[64], u32[], token[]) recv(after-all), channel_id=1,
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
      recv_done = (f32[64], token[]) recv-done(recv), channel_id=1
      send = (f32[64], u32[], token[]) send(data, after-all), channel_id=2,
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}},
          control-predecessors={recv}
      send_done = token[] send-done(send), channel_id=2,
          control-predecessors={recv_done}
      recv_data = f32[64] get-tuple-element(recv_done), index=0

      c1 = u32[] constant(1)
      i_ = u32[] add(u32[] i, u32[] c1)

      ROOT result = (u32[], f32[64]) tuple(i_, recv_data)
    }

    ENTRY entry {
      c0 = u32[] constant(0)
      a = f32[] constant(42)
      data = f32[64] broadcast(a), dimensions={}

      // Conflicting all-reduce before loop.
      ar = f32[64] all-reduce(data), channel_id=3, replica_groups={{0,1,2,3}},
          to_apply=add

      while_a_init = (u32[], f32[64]) tuple(c0, ar)
      while_a = (u32[], f32[64]) while(while_a_init), condition=cond, body=body

      // Conflicting all-reduce after loop.
      while_a_dep_data = f32[64] get-tuple-element(while_a), index=1
      sandwitched_ar = f32[64] all-reduce(while_a_dep_data), channel_id=4,
          replica_groups={{0,1,2,3}}, to_apply=add

      while_b_init = (u32[], f32[64]) tuple(c0, sandwitched_ar)
      while_b = (u32[], f32[64]) while(while_b_init), condition=cond, body=body

      // Conflicting all-reduce after loop.
      while_b_dep_data = f32[64] get-tuple-element(while_b), index=1
      ROOT final_ar = f32[64] all-reduce(while_b_dep_data), channel_id=5,
          replica_groups={{0,1,2,3}}, to_apply=add
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kHloStr, config_));

  // Run pass.
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunOptimizer(module.get(), /*enable_partial_send_recv_pipelining=*/true));
  EXPECT_TRUE(changed);

  // Find ops around the while loop.
  HloInstruction* ar_op = FindInstruction(module.get(), "ar");
  HloInstruction* recv_a_op = FindInstruction(module.get(), "recv.1");
  HloInstruction* recv_done_a_op = FindInstruction(module.get(), "recv_done.1");
  HloInstruction* send_a_op = FindInstruction(module.get(), "send.2");
  HloInstruction* send_done_a_op = FindInstruction(module.get(), "send_done.2");
  HloInstruction* sandwitched_ar_op =
      FindInstruction(module.get(), "sandwitched_ar");
  HloInstruction* recv_b_op = FindInstruction(module.get(), "recv.3");
  HloInstruction* recv_done_b_op = FindInstruction(module.get(), "recv_done.3");
  HloInstruction* send_b_op = FindInstruction(module.get(), "send.4");
  HloInstruction* send_done_b_op = FindInstruction(module.get(), "send_done.4");
  HloInstruction* final_ar_op = FindInstruction(module.get(), "final_ar");

  // Find the two while loops.
  HloInstruction* while_a_op = FindInstruction(module.get(), "while");
  HloInstruction* while_b_op = FindInstruction(module.get(), "while.1");

  // Assert relation between send/recv ops and while loops.
  EXPECT_THAT(recv_done_a_op, GmockMatch(m::RecvDone(m::Op().Is(recv_a_op))));
  EXPECT_THAT(
      while_a_op,
      GmockMatch(m::While(m::Tuple(m::Op(), m::Op().Is(ar_op),
                                   m::Op().Is(recv_done_a_op), m::Op()))));
  EXPECT_THAT(send_done_a_op, GmockMatch(m::SendDone(m::Op().Is(send_a_op))));
  EXPECT_THAT(recv_done_b_op, GmockMatch(m::RecvDone(m::Op().Is(recv_b_op))));
  EXPECT_THAT(
      while_b_op,
      GmockMatch(m::While(m::Tuple(m::Op(), m::Op().Is(sandwitched_ar_op),
                                   m::Op().Is(recv_done_b_op), m::Op()))));
  EXPECT_THAT(send_done_b_op, GmockMatch(m::SendDone(m::Op().Is(send_b_op))));

  // Expect control dependencies between peeled ops and the conflicting
  // collectives.
  EXPECT_THAT(recv_a_op->control_predecessors(), UnorderedElementsAre(ar_op));
  EXPECT_THAT(send_a_op->control_predecessors(),
              UnorderedElementsAre(ar_op, recv_a_op, recv_done_a_op));
  EXPECT_THAT(sandwitched_ar_op->control_predecessors(),
              UnorderedElementsAre(recv_done_a_op, send_done_a_op));
  EXPECT_THAT(recv_b_op->control_predecessors(),
              UnorderedElementsAre(ar_op, sandwitched_ar_op, send_done_a_op,
                                   send_a_op));
  EXPECT_THAT(send_b_op->control_predecessors(),
              UnorderedElementsAre(ar_op, sandwitched_ar_op, recv_a_op,
                                   recv_done_a_op, recv_b_op, recv_done_b_op));
  EXPECT_THAT(final_ar_op->control_predecessors(),
              UnorderedElementsAre(send_done_b_op, send_done_a_op,
                                   recv_done_b_op, recv_done_a_op));
}

TEST_F(GpuP2PPipelinerTest, ConflictingControlDependencies) {
  absl::string_view kHloStr = R"(
    HloModule test

    cond {
      param = (u32[], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      n = u32[] constant(2)
      ROOT result = pred[] compare(i, n), direction=LT
    }

    body {
      param = (u32[], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data = f32[64] get-tuple-element(param), index=1

      // Avoids pipelining send.
      after-all = token[] after-all()
      data_ = f32[64] add(data, data)
      send = (f32[64], u32[], token[]) send(data_, after-all), channel_id=2,
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}

      // Could be pipelined if it didn't have a control dependency on send.
      recv = (f32[64], u32[], token[]) recv(after-all), channel_id=1,
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}},
          control-predecessors={send}
      recv_done = (f32[64], token[]) recv-done(recv), channel_id=1
      send_done = token[] send-done(send), channel_id=2
      recv_data = f32[64] get-tuple-element(recv_done), index=0

      c1 = u32[] constant(1)
      i_ = u32[] add(u32[] i, u32[] c1)

      ROOT result = (u32[], f32[64]) tuple(i_, recv_data)
    }

    ENTRY entry {
      c0 = u32[] constant(0)
      a = f32[] constant(42)
      data = f32[64] broadcast(a), dimensions={}
      while_init = (u32[], f32[64]) tuple(c0, data)
      ROOT while = (u32[], f32[64]) while(while_init), condition=cond, body=body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kHloStr, config_));

  // Run pass and expect no change. We cannot pipeline recv/recv-done because
  // of the control dependency on send.
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, RunOptimizer(module.get(),
                                 /*enable_partial_send_recv_pipelining=*/true));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
