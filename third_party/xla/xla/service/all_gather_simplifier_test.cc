/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/all_gather_simplifier.h"

#include <memory>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = match;

using AllGatherSimplifierTest = HloHardwareIndependentTestBase;

TEST_F(AllGatherSimplifierTest, ReplicatedParameters) {
  const absl::string_view kModuleStr = R"(
HloModule m

test {
  p0 = f32[1, 512, 1, 512] parameter(0)
  p1 = f32[1, 512, 1, 512] parameter(1)
  table = s32[16] constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
  all-gather = f32[16, 512, 1, 512] all-gather(p0), replica_groups={{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}, dimensions={0}, use_global_device_ids=true, channel_id=1
  replica-id = u32[] replica-id()
  ds_index = s32[1] dynamic-slice(table, replica-id), dynamic_slice_sizes={1}
  reshape = s32[] reshape(ds_index)
  zero = s32[] constant(0)
  dynamic-slice = f32[1, 512, 1, 512] dynamic-slice(all-gather, reshape, zero, zero, zero), dynamic_slice_sizes={1, 512, 1, 512}
  ROOT add = f32[1, 512, 1, 512] add(dynamic-slice, p1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleStr, /*replica_count=*/16));
  module->mutable_config().set_use_spmd_partitioning(true);
  AllGatherSimplifier ag_simplifier;
  ASSERT_TRUE(ag_simplifier.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Add(m::Parameter(0), m::Parameter(1))));
}

}  // namespace
}  // namespace xla
