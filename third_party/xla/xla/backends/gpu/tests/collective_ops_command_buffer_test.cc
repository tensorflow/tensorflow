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

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
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
  options.use_threads = true;

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
