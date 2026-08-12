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

#include "xla/backends/gpu/transforms/collectives/convert_async_collectives_to_sync.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;

// Note: The pass only processes modules that are already scheduled. If the test
// does not work as expected, make sure to check if "is_scheduled=true" is added
// to the HLO module string.
class GpuConvertAsyncCollectivesToSyncTest
    : public HloHardwareIndependentTestBase {
 public:
  absl::Status RunPass(HloModule* module, bool expect_change) {
    ABSL_ASSIGN_OR_RETURN(bool changed,
                     GpuConvertAsyncCollectivesToSync().Run(module));
    EXPECT_EQ(changed, expect_change);
    return absl::OkStatus();
  }
};

TEST_F(GpuConvertAsyncCollectivesToSyncTest, SimpleAllReduce) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3
        ROOT done = u32[] all-reduce-done(start)
      }
    )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %test_computation
    CHECK-NOT: all-reduce-start
    CHECK: ROOT %{{.*}} = u32[] all-reduce(%id)
    CHECK-SAME: channel_id=3
    CHECK-SAME: backend_config={{.*}}is_sync{{.*}}true
    CHECK-NOT: all-reduce-done
  )"),
              IsOkAndHolds(true));
}

TEST_F(GpuConvertAsyncCollectivesToSyncTest, SimpleAllReduceWithNop) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start = u32[] all-reduce-start(id), to_apply=apply_op,
          channel_id=3, replica_groups={{0,1}, {2,3}}
        id2 = f32[] bitcast(id)
        ROOT done = u32[] all-reduce-done(start)
      }
    )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %test_computation
    CHECK-NOT: all-reduce-start
    CHECK: %id2 = f32[] bitcast(%id)
    CHECK: ROOT %{{.*}} = u32[] all-reduce(%id)
    CHECK-SAME: channel_id=3
    CHECK-SAME: backend_config={{.*}}is_sync{{.*}}true
    CHECK-NOT: all-reduce-done
  )"),
              IsOkAndHolds(true));
}

TEST_F(GpuConvertAsyncCollectivesToSyncTest, SimpleCollectiveBroadcast) {
  const absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true

  collective_broadcast {
    p0 = u32[8] parameter(0)
    ROOT result = u32[8] collective-broadcast(p0),
      replica_groups={{0,1}, {2,3}}
  }

  ENTRY main {
    data = u32[8] parameter(0)
    cb-start = ((u32[8]{0}), u32[8]{0}) async-start(
      u32[8]{0} %data), calls=collective_broadcast
    ROOT %ars = u32[8]{0} async-done(
      ((u32[8]{0}), u32[8]{0}) %cb-start), calls=collective_broadcast
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %main
    CHECK-NOT: async-start
    CHECK: ROOT %{{.*}} = u32[8]{0} collective-broadcast(%data)
    CHECK-SAME: backend_config={{.*}}is_sync{{.*}}true
    CHECK-NOT: async-done
  )"),
              IsOkAndHolds(true));
}

TEST_F(GpuConvertAsyncCollectivesToSyncTest, SimpleAllReduceWithNonNop) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3
        id2 = u32[] add(id, id)
        ROOT done = u32[] all-reduce-done(start)
      }
    )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/false));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %test_computation
    CHECK: %start = u32[] all-reduce-start(%id)
    CHECK: %id2 = u32[] add(%id, %id)
    CHECK: ROOT %done = u32[] all-reduce-done(%start)
  )"),
              IsOkAndHolds(true));
}

TEST_F(GpuConvertAsyncCollectivesToSyncTest, SimpleAllGather) {
  const absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true
  ENTRY test_computation {
    a1 = u32[1, 2] parameter(0)
    ags = (u32[1, 2], u32[2, 2]) all-gather-start(a1),
      dimensions={0}, channel_id=3
    ROOT allgather = u32[2,2] all-gather-done(ags)
  })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %test_computation
    CHECK-NOT: all-gather-start
    CHECK: ROOT %{{.*}} = u32[2,2]{1,0} all-gather(%a1)
    CHECK-SAME: dimensions={0}
    CHECK-SAME: backend_config={{.*}}is_sync{{.*}}true
    CHECK-NOT: all-gather-done
  )"),
              IsOkAndHolds(true));
}

TEST_F(GpuConvertAsyncCollectivesToSyncTest, SimpleCollectivePermute) {
  const absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true

  ENTRY test_computation {
    p = u32[2] parameter(0)
    start = (u32[2], u32[2], u32[], u32[]) collective-permute-start(p),
      source_target_pairs={{0,1}, {1,0}}
    ROOT done = u32[2] collective-permute-done(start)
  })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %test_computation
    CHECK-NOT: collective-permute-start
    CHECK: ROOT %{{.*}} = u32[2]{0} collective-permute(%p)
    CHECK-SAME: backend_config={{.*}}is_sync{{.*}}true
    CHECK-NOT: collective-permute-done
  )"),
              IsOkAndHolds(true));
}

TEST_F(GpuConvertAsyncCollectivesToSyncTest, SimpleReduceScatter) {
  const absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true

  add {
    lhs = u32[] parameter(0)
    rhs = u32[] parameter(1)
    ROOT add = u32[] add(lhs, rhs)
  }

  reduce_scatter {
    p0 = u32[8] parameter(0)
    ROOT result = u32[4] reduce-scatter(p0), replica_groups={{0,3}, {1,2}},
                      dimensions={0}, to_apply=add
  }

  ENTRY main {
    data = u32[8] parameter(0)
    rs-start = ((u32[8]{0}), u32[4]{0}) async-start(
      u32[8]{0} %data), calls=reduce_scatter
    ROOT %ars = u32[4]{0} async-done(
      ((u32[8]{0}), u32[4]{0}) %rs-start), calls=reduce_scatter
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %main
    CHECK-NOT: async-start
    CHECK: ROOT %{{.*}} = u32[4]{0} reduce-scatter(%data)
    CHECK-SAME: dimensions={0}
    CHECK-SAME: backend_config={{.*}}is_sync{{.*}}true
    CHECK-NOT: async-done
  )"),
              IsOkAndHolds(true));
}

TEST_F(GpuConvertAsyncCollectivesToSyncTest, SimpleAllToAll) {
  const absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true

  all_to_all {
    p0 = u32[2] parameter(0)
    ROOT result = u32[2] all-to-all(p0), dimensions={0},
      replica_groups={{0,1}, {2,3}}
  }

  ENTRY test_computation {
    a1 = u32[2] parameter(0)
    a2a-start = ((u32[2]), u32[2]) async-start(u32[2] a1), calls=all_to_all
    ROOT a2s = u32[2] async-done(a2a-start), calls=all_to_all
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %test_computation
    CHECK-NOT: async-start
    CHECK: ROOT %{{.*}} = u32[2]{0} all-to-all(%a1)
    CHECK-SAME: dimensions={0}
    CHECK-SAME: backend_config={{.*}}is_sync{{.*}}true
    CHECK-NOT: async-done
  )"),
              IsOkAndHolds(true));
}

TEST_F(GpuConvertAsyncCollectivesToSyncTest, ControlDeps) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start1 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3
        done1 = u32[] all-reduce-done(start1)
        start2 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=4,
          control-predecessors={done1}
        done2 = u32[] all-reduce-done(start2)
        ROOT x = u32[] add(done1, done2)
      }
    )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %test_computation
    CHECK-NOT: all-reduce-start
    CHECK: %[[AR1:.*]] = u32[] all-reduce(%id)
    CHECK-SAME: channel_id=3
    CHECK: %[[AR2:.*]] = u32[] all-reduce(%id)
    CHECK-SAME: channel_id=4
    CHECK-SAME: control-predecessors={%[[AR1]]}
    CHECK: ROOT %{{.*}} = u32[] add(%[[AR1]], %[[AR2]])
    CHECK-NOT: all-reduce-done
  )"),
              IsOkAndHolds(true));
}

// Test multiple in-flight collectives that are ordered in a streaming fashion:
// i.e., ends are in start order (FIFO).
TEST_F(GpuConvertAsyncCollectivesToSyncTest, MultipleInFlightStreaming) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start1 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3
        start2 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=4
        done1 = u32[] all-reduce-done(start1)
        done2 = u32[] all-reduce-done(start2)
        ROOT x = u32[] add(done1, done2)
      }
    )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %test_computation
    CHECK-NOT: all-reduce-start
    CHECK: %[[AR1:.*]] = u32[] all-reduce(%id)
    CHECK-SAME: channel_id=3
    CHECK: %[[AR2:.*]] = u32[] all-reduce(%id)
    CHECK-SAME: channel_id=4
    CHECK: ROOT %{{.*}} = u32[] add(%[[AR1]], %[[AR2]])
    CHECK-NOT: all-reduce-done
  )"),
              IsOkAndHolds(true));
}

// Test multiple in-flight collectives that are nested: {s0,{s1,e1},e0}
TEST_F(GpuConvertAsyncCollectivesToSyncTest, MultipleInFlightNested) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start1 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3
        start2 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=4
        done2 = u32[] all-reduce-done(start2)
        done1 = u32[] all-reduce-done(start1)
        ROOT x = u32[] add(done1, done2)
      }
    )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %test_computation
    CHECK-NOT: all-reduce-start
    CHECK: %[[AR2:.*]] = u32[] all-reduce(%id)
    CHECK-SAME: channel_id=4
    CHECK: %[[AR1:.*]] = u32[] all-reduce(%id)
    CHECK-SAME: channel_id=3
    CHECK: ROOT %{{.*}} = u32[] add(%[[AR1]], %[[AR2]])
    CHECK-NOT: all-reduce-done
  )"),
              IsOkAndHolds(true));
}

// Test multiple in-flight collectives that are nested: {s0,{s1,e1},e0} where
// inner pair can be converted but not outer.
TEST_F(GpuConvertAsyncCollectivesToSyncTest, MultipleInFlightNestedPartial) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start1 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3
        start2 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=4
        done2 = u32[] all-reduce-done(start2)
        id2 = u32[] add(done2, done2)
        done1 = u32[] all-reduce-done(start1)
        ROOT x = u32[] add(done1, done2)
      }
    )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %test_computation
    CHECK: %start1 = u32[] all-reduce-start(%id)
    CHECK-SAME: channel_id=3
    CHECK-NOT: %start2
    CHECK: %[[AR2:.*]] = u32[] all-reduce(%id)
    CHECK-SAME: channel_id=4
    CHECK: %id2 = u32[] add(%[[AR2]], %[[AR2]])
    CHECK: %done1 = u32[] all-reduce-done(%start1)
    CHECK: ROOT %{{.*}} = u32[] add(%done1, %[[AR2]])
  )"),
              IsOkAndHolds(true));
}

TEST_F(GpuConvertAsyncCollectivesToSyncTest,
       PreMarkedSyncCollectiveNestedInAsyncWindowIsRestored) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        outer_start = u32[] all-reduce-start(id), to_apply=apply_op,
          channel_id=3
        inner_start = u32[] all-reduce-start(id), to_apply=apply_op,
          channel_id=4,
          backend_config={"collective_backend_config":{"is_sync":true}}
        inner_done = u32[] all-reduce-done(inner_start)
        outer_done = u32[] all-reduce-done(outer_start)
        ROOT result = (u32[], u32[]) tuple(outer_done, inner_done)
      }
    )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %test_computation
    CHECK: %outer_start = u32[] all-reduce-start(%id)
    CHECK-SAME: channel_id=3
    CHECK-NOT: %inner_start
    CHECK: %[[INNER:.*]] = u32[] all-reduce(%id)
    CHECK-SAME: channel_id=4
    CHECK-SAME: backend_config={{.*}}is_sync{{.*}}true
    CHECK: %outer_done = u32[] all-reduce-done(%outer_start)
    CHECK: ROOT %result = (u32[], u32[]) tuple(%outer_done, %[[INNER]])
  )"),
              IsOkAndHolds(true));
}

TEST_F(GpuConvertAsyncCollectivesToSyncTest,
       SimpleAllReducePreserveBackendConfig) {
  const absl::string_view hlo_string = R"(
      HloModule test, is_scheduled=true

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3,
          replica_groups={{0,1}, {2,3}},
          backend_config={"collective_backend_config":{"is_pipelined":true}}
        id2 = f32[] bitcast(id)
        ROOT done = u32[] all-reduce-done(start)
      }
    )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
    CHECK-LABEL: ENTRY %test_computation
    CHECK-NOT: all-reduce-start
    CHECK: %id2 = f32[] bitcast(%id)
    CHECK: ROOT %{{.*}} = u32[] all-reduce(%id)
    CHECK-SAME: "is_sync":true,"is_pipelined":true
  )"),
              IsOkAndHolds(true));
}

}  // namespace
}  // namespace xla::gpu
