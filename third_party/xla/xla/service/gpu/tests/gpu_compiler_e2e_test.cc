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

#include "absl/strings/string_view.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {
namespace {

namespace m = match;

class GpuCompilerE2ETest : public HloTestBase {};

TEST_F(GpuCompilerE2ETest, DegeneratedAllReduceRemoval) {
  constexpr absl::string_view kHloText = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

main {
  p0 = f32[8,16] parameter(0), parameter_replication={false}
  ROOT all-reduce = f32[8,16] all-reduce(p0),
    channel_id=1,
    use_global_device_ids=true,
    replica_groups={{0},{1},{2},{3},{4},{5},{6},{7}},
    to_apply=sum
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kHloText, /*replica_count=*/1,
                                                /*num_partitions=*/8));
  module->mutable_config().set_use_spmd_partitioning(true);
  TF_ASSERT_OK_AND_ASSIGN(auto optimized_module,
                          GetOptimizedModule(std::move(module)));
  EXPECT_THAT(optimized_module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
