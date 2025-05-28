/* Copyright 2021 The OpenXLA Authors.

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
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

// Tests replicated infeed/outfeed operations.

namespace xla {
namespace {

class ReplicatedIOFeedTest : public HloPjRtTestBase {};

TEST_F(ReplicatedIOFeedTest, InfeedAndOutfeed) {
  static constexpr int kNumReplicas = 4;
  static constexpr absl::string_view kHloText = R"(
  HloModule infeed
  ENTRY main {
    // Read from infeed, add replica_id, and send to outfeed.
    token0 = token[] after-all()
    infeed = (u32[], token[]) infeed(token0)
    infeed.data = u32[] get-tuple-element(infeed), index=0
    infeed.token = token[] get-tuple-element(infeed), index=1
    replica_id = u32[] replica-id()
    result = u32[] add(infeed.data, replica_id)
    outfeed = token[] outfeed(result, infeed.token), outfeed_shape=u32[]
  })";
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  std::vector<Literal> outfeed_literals;

  HloRunnerInterface::ReplicatedExecuteOptions opts;
  opts.num_replicas = kNumReplicas;

  // Initialize infeed literal = replica_id * 10
  std::vector<Literal> infeed_literals(kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    infeed_literals[i] = LiteralUtil::CreateR0<uint32_t>(i * 10);
    opts.infeed_values.push_back(&infeed_literals[i]);
  }
  opts.infeed_steps = 1;
  opts.outfeed_shape =
      ShapeUtil::MakeValidatedScalarShape(PrimitiveType::U32).value();
  opts.outfeed_values = &outfeed_literals;
  opts.use_threads = true;

  DeviceAssignment device_assn(/*replica_count=*/kNumReplicas,
                               /*computation_count=*/1);
  device_assn.FillIota(0);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(
                              kHloText, GetModuleConfigForTest(kNumReplicas)));
  TF_ASSERT_OK(test_runner()
                   .ExecuteReplicated(std::move(module), opts, &device_assn)
                   .status());

  // Verify that each infeed and outfeed is routed correctly. Each replica
  // should produce 10*replica (indeed) + replica (from HLO)
  for (int i = 0; i < kNumReplicas; ++i) {
    LiteralTestUtil::ExpectR0Equal<uint32_t>(10 * i + i, outfeed_literals[i]);
  }
}

}  // namespace
}  // namespace xla
