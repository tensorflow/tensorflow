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

#include "tensorflow/compiler/xla/service/gpu/gpu_convert_async_collectives_to_sync.h"

#include <string_view>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::IsFalse;
using ::testing::IsTrue;

// Note: The pass only processes modules that are already scheduled. If the test
// does not work as epxected, make sure to check if "is_scheduled=true" is added
// to the HLO module string.
class GpuConvertAsyncCollectivesToSyncTest : public HloTestBase {
 public:
  Status RunPass(HloModule *module, bool expect_change,
                 HloPredicate is_nop = {}) {
    TF_ASSIGN_OR_RETURN(bool changed,
                        GpuConvertAsyncCollectivesToSync{is_nop}.Run(module));
    EXPECT_EQ(changed, expect_change);
    return OkStatus();
  }

  // Returns true if the instruction with the given name is synchronous.
  bool IsSync(HloModule *module, std::string_view name) {
    const HloInstruction *inst = FindInstruction(module, name);
    if (inst == nullptr) {
      return false;
    }
    auto backend_config =
        inst->backend_config<CollectiveBackendConfig>().value();
    return backend_config.is_sync();
  }

  HloPredicate is_nop_simple_ =
      HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kGetTupleElement,
                       HloOpcode::kParameter>;
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  EXPECT_THAT(IsSync(module.get(), "start"), IsTrue());
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
        start = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=3, replica_groups={{0,1}, {2,3}}
        id2 = f32[] bitcast(id)
        ROOT done = u32[] all-reduce-done(start)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, is_nop_simple_));
  EXPECT_THAT(IsSync(module.get(), "start"), IsTrue());
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/false));
}

TEST_F(GpuConvertAsyncCollectivesToSyncTest, SimpleAllGather) {
  const absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true
  ENTRY test_computation {
    a1 = u32[1, 2] parameter(0)
    ags = (u32[1, 2], u32[2, 2]) all-gather-start(a1), dimensions={0}, channel_id=3
    ROOT allgather = u32[2,2] all-gather-done(ags)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  EXPECT_THAT(IsSync(module.get(), "ags"), IsTrue());
}

TEST_F(GpuConvertAsyncCollectivesToSyncTest, SimpleCollectivePermute) {
  const absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true

  ENTRY test_computation {
    p = u32[2] parameter(0)
    start = (u32[2], u32[2], u32[], u32[]) collective-permute-start(p), source_target_pairs={{0,1}, {1,0}}
    ROOT done = u32[2] collective-permute-done(start)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  EXPECT_THAT(IsSync(module.get(), "start"), IsTrue());
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
    rs-start = ((u32[8]{0}), u32[4]{0}) async-start(u32[8]{0} %data), calls=reduce_scatter
    ROOT %ars = u32[4]{0} async-done(((u32[8]{0}), u32[4]{0}) %rs-start), calls=reduce_scatter
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  EXPECT_THAT(IsSync(module.get(), "rs-start"), IsTrue());
}

TEST_F(GpuConvertAsyncCollectivesToSyncTest, SimpleAllToAll) {
  const absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true

  all_to_all {
    p0 = u32[2] parameter(0)
    ROOT result = u32[2] all-to-all(p0), dimensions={0}, replica_groups={{0,1},{2,3}}
  }

  ENTRY test_computation {
    a1 = u32[2] parameter(0)
    a2a-start = ((u32[2]), u32[2]) async-start(u32[2] a1), calls=all_to_all
    ROOT a2s = u32[2] async-done(a2a-start), calls=all_to_all
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  EXPECT_THAT(IsSync(module.get(), "a2a-start"), IsTrue());
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
        start2 = u32[] all-reduce-start(id), to_apply=apply_op, channel_id=4, control-predecessors={done1}
        done2 = u32[] all-reduce-done(start2)
        ROOT x = u32[] add(done1, done2)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  EXPECT_THAT(IsSync(module.get(), "start1"), IsTrue());
  EXPECT_THAT(IsSync(module.get(), "start2"), IsTrue());
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  EXPECT_THAT(IsSync(module.get(), "start1"), IsTrue());
  EXPECT_THAT(IsSync(module.get(), "start2"), IsTrue());
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  EXPECT_THAT(IsSync(module.get(), "start1"), IsTrue());
  EXPECT_THAT(IsSync(module.get(), "start2"), IsTrue());
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true));
  EXPECT_THAT(IsSync(module.get(), "start1"), IsFalse());
  EXPECT_THAT(IsSync(module.get(), "start2"), IsTrue());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
