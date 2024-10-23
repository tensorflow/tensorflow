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

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/lib/core/status_test_util.h"

// Tests replicated infeed/outfeed operations.

namespace xla {

class ReplicatedIOFeedTest : public HloTestBase {};

static DeviceAssignment MakeDeviceAssn(size_t num_devices) {
  DeviceAssignment assn(/*replica_count=*/num_devices,
                        /*computation_count=*/1);
  for (int64_t i = 0; i < num_devices; ++i) {
    assn(i, 0) = i;
  }
  return assn;
}

XLA_TEST_F(ReplicatedIOFeedTest, InfeedAndOutfeed) {
  std::string hlo_text = R"(
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

  const int kNumReplicas = 4;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);

  auto config = GetModuleConfigForTest();
  config.set_replica_count(kNumReplicas);
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_text, config).value();
  auto executable =
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true).value();

  auto device_assn = MakeDeviceAssn(kNumReplicas);

  std::vector<Literal> outfeed_literals;

  HloRunner::ReplicatedExecuteOptions opts;
  opts.num_replicas = kNumReplicas;

  // Initialize infeed literal = replica_id * 10
  std::vector<Literal> infeed_literals(kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    infeed_literals[i] = LiteralUtil::CreateR0<uint32_t>(i * 10);
    opts.infeed_values.push_back(&infeed_literals[i]);
  }
  opts.infeed_steps = 1;
  opts.outfeed_shape = ShapeUtil::MakeScalarShape(PrimitiveType::U32);
  opts.outfeed_values = &outfeed_literals;
  opts.use_threads = true;

  TF_ASSERT_OK(
      ExecuteReplicatedWithHloRunner(executable.get(), opts, &device_assn)
          .status());

  // Verify that each infeed and outfeed is routed correctly. Each replica
  // should produce 10*replica (indeed) + replica (from HLO)
  for (int i = 0; i < kNumReplicas; ++i) {
    LiteralTestUtil::ExpectR0Equal<uint32_t>(10 * i + i, outfeed_literals[i]);
  }
}
}  // namespace xla
