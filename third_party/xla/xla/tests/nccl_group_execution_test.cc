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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

// Tests NCCL group execution.

class NcclGroupExecutionTest : public HloTestBase {
 public:
  NcclGroupExecutionTest() {
    VLOG(1) << "Running with " << num_devices() << " devices";
  }
};

XLA_TEST_F(NcclGroupExecutionTest, NcclGroupSendRecvNoWhileLoop) {
  // TODO (rosiezou): remove the channel_id=0 workaround once it is optional.
  const absl::string_view kModuleStr = R"(
  HloModule module_main, entry_computation_layout={()->(f32[], f32[])}

  wrapped_send_recv {
    param0 = f32[] parameter(0)
    param1 = token[] parameter(1)
    send1 = (f32[], u32[], token[]) send(param0, param1), channel_id=0,
      frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2}}}
    param2 = f32[] parameter(2)
    param3 = token[] parameter(3)
    send2 = (f32[], u32[], token[]) send(param2, param3), channel_id=0,
      frontend_attributes={_xla_send_recv_source_target_pairs={{2,3}}}
    param4 = token[] parameter(4)
    recv1 = (f32[], u32[], token[]) recv(param4), channel_id=0,
      frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2}}}
    param5 = token[] parameter(5)
    recv2 = (f32[], u32[], token[]) recv(param5), channel_id=0,
      frontend_attributes={_xla_send_recv_source_target_pairs={{2,3}}}
    ROOT out = ((f32[], u32[], token[]), (f32[], u32[], token[]),
      (f32[], u32[], token[]), (f32[], u32[], token[]))
      tuple(send1, send2, recv1, recv2)
  }

  ENTRY main {
    data1 = f32[] constant(10)
    after-all1 = token[] after-all()
    data2 = f32[] constant(20)
    after-all2 = token[] after-all()
    async-comp-start = ((f32[], token[], f32[], token[], token[], token[]),
      ((f32[], u32[], token[]), (f32[], u32[], token[]), (f32[], u32[], token[]),
      (f32[], u32[], token[])), s32[]) async-start(data1, after-all1,
      data2, after-all2, after-all1, after-all2), calls=wrapped_send_recv
    async-comp-done = ((f32[], u32[], token[]), (f32[], u32[], token[]),
      (f32[], u32[], token[]), (f32[], u32[], token[])) async-done(async-comp-start)
    unpack-recv-done1 = (f32[], u32[], token[]) get-tuple-element(async-comp-done), index=2
    recv-done-data1 = f32[] get-tuple-element(unpack-recv-done1), index=0
    recv-done-token1 = token[] get-tuple-element(unpack-recv-done1), index=2
    recv-done1 = (f32[], token[]) tuple(recv-done-data1, recv-done-token1),
      control-predecessors={async-comp-start}
    data-out1 = f32[] get-tuple-element(recv-done1), index=0
    unpack-recv-done2 = (f32[], u32[], token[]) get-tuple-element(async-comp-done), index=3
    recv-done-data2 = f32[] get-tuple-element(unpack-recv-done2), index=0
    recv-done-token2 = token[] get-tuple-element(unpack-recv-done2), index=2
    recv-done2 = (f32[], token[]) tuple(recv-done-data2, recv-done-token2),
      control-predecessors={async-comp-start}
    data-out2 = f32[] get-tuple-element(recv-done2), index=0
    c100 = f32[] constant(100)
    res1 = f32[] dot(data-out1, c100)
    res2 = f32[] dot(data-out2, c100)
    ROOT out = (f32[], f32[]) tuple(res1, res2)
    unpack-send-done1 = (f32[], u32[], token[]) get-tuple-element(async-comp-done), index=0
    send-done1 = token[] get-tuple-element(unpack-send-done1), index=2
    unpack-send-done2 = (f32[], u32[], token[]) get-tuple-element(async-comp-done), index=1
    send-done2 = token[] get-tuple-element(unpack-send-done2), index=2
  }

  )";
  const int64_t kNumReplicas = 4;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  std::unique_ptr<VerifiedHloModule> module;
  TF_ASSERT_OK_AND_ASSIGN(module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), absl::Span<Literal* const>{},
                        kNumReplicas,
                        /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  // TODO (rosiezou): remove the string comparison once a tuple comparison
  // function is available in LiteralTestUtil.
  EXPECT_EQ(results[0].ToStringWithoutShapeOneline(), "( 0, 0 )");
  EXPECT_EQ(results[1].ToStringWithoutShapeOneline(), "( 1000, 0 )");
  EXPECT_EQ(results[2].ToStringWithoutShapeOneline(), "( 1000, 0 )");
  EXPECT_EQ(results[3].ToStringWithoutShapeOneline(), "( 0, 2000 )");
}

XLA_TEST_F(NcclGroupExecutionTest, BidirectionalCommunication) {
  const absl::string_view kModuleStr = R"(
  HloModule module_main, entry_computation_layout={()->(u32[], u32[])}

  bidirectional_ring {
    a = u32[] parameter(0)
    start = (u32[], u32[]) collective-permute-start(a), channel_id=2, source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
    done = u32[] collective-permute-done(start)
    start.1 = (u32[], u32[]) collective-permute-start(a), channel_id=1, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
    done.1 = u32[] collective-permute-done(start.1)
    ROOT tuple = (u32[], u32[]) tuple(done, done.1)
  }

  ENTRY main {
    id = u32[] replica-id()
    async-comm-start = ((u32[]), (u32[], u32[])) async-start(id), calls=bidirectional_ring,
      frontend_attributes={_collectives_group=""}
   ROOT async-comm-done = (u32[], u32[]) async-done(async-comm-start)
  }

  )";
  const int64_t kNumReplicas = 4;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  std::unique_ptr<VerifiedHloModule> module;
  TF_ASSERT_OK_AND_ASSIGN(module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), absl::Span<Literal* const>{},
                        kNumReplicas,
                        /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_EQ(results[0].ToStringWithoutShapeOneline(), "( 3, 1 )");
  EXPECT_EQ(results[1].ToStringWithoutShapeOneline(), "( 0, 2 )");
  EXPECT_EQ(results[2].ToStringWithoutShapeOneline(), "( 1, 3 )");
  EXPECT_EQ(results[3].ToStringWithoutShapeOneline(), "( 2, 0 )");
}

}  // namespace

}  // namespace xla
