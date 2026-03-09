/* Copyright 2026 The OpenXLA Authors.

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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

// Tests cross-GPU operations.
//
// Several tests requires multiple GPUs. For instructions on running this
// within Google, see go/multi-gpu-unit-test.
class CollectiveOpsCommandBufferTest : public HloTestBase {
 public:
  CollectiveOpsCommandBufferTest() {
    VLOG(1) << "Running with " << num_devices() << " devices";
  }

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    // Disable async->sync collective conversion pass to enable unit testing
    // of async collectives.
    debug_options.add_xla_disable_hlo_passes(
        "gpu-convert-async-collectives-to-sync");
    return debug_options;
  }
};

// Fixture for Tests which require peer access between GPUs.
class CollectiveOpsCommandBufferPeerAccessTest
    : public CollectiveOpsCommandBufferTest {
 protected:
  void SetUp() override {
    HloTestBase::SetUp();  // Don't forget to call the base class SetUp

    stream_executor::Platform* platform = GetTestPlatform();

    int num_devices = platform->VisibleDeviceCount();
    if (num_devices < 2) {
      GTEST_SKIP()
          << "Skipping test suite: Test requires at least 2 GPUs, found "
          << num_devices;
    }

    // Check P2P capability
    for (int i = 0; i < num_devices; ++i) {
      ASSERT_OK_AND_ASSIGN(stream_executor::StreamExecutor * executor_i,
                           platform->ExecutorForDevice(i));
      for (int j = 0; j < num_devices; ++j) {
        if (i == j) {
          continue;
        }
        ASSERT_OK_AND_ASSIGN(stream_executor::StreamExecutor * executor_j,
                             platform->ExecutorForDevice(j));

        // If P2P is not supported between any two available devices, skip the
        // test suite.
        if (!executor_i->CanEnablePeerAccessTo(executor_j)) {
          GTEST_SKIP() << "Skipping test suite: Direct peer memory access not "
                          "supported between GPU "
                       << i << " and GPU " << j;
        }
      }
    }
  }
};

TEST_F(CollectiveOpsCommandBufferPeerAccessTest, RaggedAllToAll_Simple) {
  constexpr absl::string_view hlo_text = R"(
  HloModule module, num_partitions=1, replica_count=2

  ENTRY entry {
    p0 = f32[8] parameter(0)
    id = u32[] replica-id()
    ten = u32[] constant(10)
    id2 = u32[] multiply(id, ten)
    id3 = f32[] convert(id2)
    id4 = f32[8] broadcast(id3)
    input = f32[8] add(p0, id4)
    output = f32[8] constant({-1, -1, -1, -1, -1, -1, -1, -1})
    send_sizes = s32[2] constant({4, 4})
    recv_sizes = s32[2] constant({4, 4})
    input_offsets = s32[2] constant({0, 4})
    four = u32[] constant(4)
    oof = u32[] multiply(id, four)
    oof2 = s32[] convert(oof)
    output_offsets = s32[2] broadcast(oof2)

    ROOT ra2a = f32[8] ragged-all-to-all(input, output, input_offsets, send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  }
  )";

  const int64_t kNumReplicas = 2;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  bool run_hlo_passes = true;
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options().add_xla_gpu_enable_command_buffer(
      DebugOptions::COLLECTIVES);
  config.mutable_debug_options().set_xla_gpu_graph_min_graph_size(1);
  config.mutable_debug_options()
      .set_xla_gpu_experimental_ragged_all_to_all_use_barrier(true);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  CHECK_OK(PreprocessModuleForTestRunner(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> executable,
                          CreateExecutable(std::move(module), run_hlo_passes));

  // Execute compiled module multiple times to exercise warm-up, create, and
  // update paths. Last run uses new arguments to encourage device buffer
  // address changes.
  auto arg0 = LiteralUtil::CreateR1<float>({0., 1., 2., 3., 4., 5., 6., 7.});

  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_devices = kNumReplicas;
  options.arguments = {&arg0};
  options.run_hlo_passes = run_hlo_passes;

  // Multiple executions to Warm-up (may run thunks) and
  // Create (record and execute command buffer)
  for (int i = 0; i < 3; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                            test_runner().ExecuteReplicatedWithExecutable(
                                executable.get(), options));

    ASSERT_EQ(results.size(), kNumReplicas);
    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR1<float>({0., 1., 2., 3., 10., 11., 12., 13.}),
        results[0]));
    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR1<float>({4., 5., 6., 7., 14., 15., 16., 17.}),
        results[1]));
  }

  // Update (execute with new arguments to attempt buffer changes)
  auto arg2 = LiteralUtil::CreateR1<float>({7., 6., 5., 4., 3., 2., 1., 0.});
  options.arguments = {&arg2};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      test_runner().ExecuteReplicatedWithExecutable(executable.get(), options));

  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({7., 6., 5., 4., 17., 16., 15., 14.}),
      results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({3., 2., 1., 0., 13., 12., 11., 10.}),
      results[1]));
}

TEST_F(CollectiveOpsCommandBufferTest, SendRecv_Simple) {
  constexpr absl::string_view hlo_text = R"(
  HloModule test
  ENTRY test_computation {
    %p0 = u32[2] parameter(0)
    %replica = u32[] replica-id()
    %replica2 = u32[2] broadcast(%replica), dimensions={}
    %p = u32[2] add(%p0, %replica2)

    %after-all = token[] after-all()
    %recv = (u32[2], u32[], token[]) recv(%after-all), channel_id=0, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{1,0}}"
    }
    %send = (u32[2], u32[], token[]) send(%p, %after-all), channel_id=0, control-predecessors={%recv}, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{1,0}}"
    }

    %recv-done = (u32[2], token[]) recv-done(%recv), channel_id=0
    %recv-data = u32[2] get-tuple-element(%recv-done), index=0
    %send-done = token[] send-done(%send), channel_id=0, control-predecessors={%recv}
    ROOT copy = u32[2] copy(%recv-data)
  }
  )";

  const int64_t kNumReplicas = 2;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  bool run_hlo_passes = false;
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options().add_xla_gpu_enable_command_buffer(
      DebugOptions::COLLECTIVES);
  config.mutable_debug_options().set_xla_gpu_graph_min_graph_size(1);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  CHECK_OK(PreprocessModuleForTestRunner(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> executable,
                          CreateExecutable(std::move(module), run_hlo_passes));

  // Execute compiled module multiple times to exercise warm-up, create, and
  // update paths. Last run uses new arguments to encourage device buffer
  // address changes.
  auto arg0 = LiteralUtil::CreateR1<uint32_t>({10, 12});

  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_devices = kNumReplicas;
  options.arguments = {&arg0};
  options.run_hlo_passes = run_hlo_passes;

  // Multiple executions to Warm-up (may run thunks) and
  // Create (record and execute command buffer)
  for (int i = 0; i < 3; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                            test_runner().ExecuteReplicatedWithExecutable(
                                executable.get(), options));

    ASSERT_EQ(results.size(), kNumReplicas);
    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR1<uint32_t>({11, 13}), results[0]));
    EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<uint32_t>({0, 0}),
                                       results[1]));
  }

  // Update (execute with new arguments to attempt buffer changes)
  auto arg2 = LiteralUtil::CreateR1<uint32_t>({14, 16});
  options.arguments = {&arg2};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      test_runner().ExecuteReplicatedWithExecutable(executable.get(), options));

  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<uint32_t>({15, 17}),
                                     results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<uint32_t>({0, 0}),
                                     results[1]));
}

}  // namespace
}  // namespace xla
