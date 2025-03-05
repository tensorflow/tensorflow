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

#include "xla/hlo/transforms/collectives/all_reduce_normalizer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using AllReduceNormalizerTest = HloHardwareIndependentTestBase;

TEST_F(AllReduceNormalizerTest, Simple) {
  const absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = bf16[1,256,8,256]{1,3,2,0:T(8,128)(2,1)} parameter(0)
  ROOT all-reduce = bf16[1,256,8,256]{1,3,2,0:T(8,128)(2,1)}
    all-reduce(bf16[1,256,8,256]{1,3,2,0:T(8,128)(2,1)} p0), channel_id=115,
    replica_groups={{0,2},{4,6},{8,10},{12,14},{16,18},{20,22},{24,26},{28,30},
      {1,3},{5,7},{9,11},{13,15},{17,19},{21,23},{25,27},{29,31}},
    use_global_device_ids=true, to_apply=%add,
    backend_config={"flag_configs":[],
      "barrier_config":{"barrier_type":"CUSTOM","id":"16"},
      "scoped_memory_configs":[{
        "memory_space":"0","offset":"0","size":"67108864"}],
      "used_scoped_memory_configs":[]}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllReduceNormalizer normalizer(
      /*is_supported_all_reduce=*/[](const HloInstruction*) { return false; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, normalizer.Run(module.get()));
  VLOG(1) << module->ToString();
  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::Slice(op::Reshape(op::AllGather(op::Reshape(op::Reduce(
          op::AllToAll(op::Reshape(op::Pad(op::Parameter(0), op::Constant()))),
          op::Constant()))))));
}

TEST_F(AllReduceNormalizerTest, NonDivisible) {
  const absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = bf16[1,256,7,256]{1,3,2,0:T(8,128)(2,1)} parameter(0)
  ROOT all-reduce = bf16[1,256,7,256]{1,3,2,0:T(8,128)(2,1)}
    all-reduce(bf16[1,256,7,256]{1,3,2,0:T(8,128)(2,1)} p0), channel_id=115,
    replica_groups={{0,2},{4,6},{8,10},{12,14},{16,18},{20,22},{24,26},{28,30},
      {1,3},{5,7},{9,11},{13,15},{17,19},{21,23},{25,27},{29,31}},
    use_global_device_ids=true, to_apply=%add,
    backend_config={"flag_configs":[],
      "barrier_config":{"barrier_type":"CUSTOM","id":"16"},
      "scoped_memory_configs":[{
        "memory_space":"0","offset":"0","size":"67108864"}],
      "used_scoped_memory_configs":[]}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllReduceNormalizer normalizer(
      /*is_supported_all_reduce=*/[](const HloInstruction*) { return false; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, normalizer.Run(module.get()));
  VLOG(1) << module->ToString();
  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::Slice(op::Reshape(op::AllGather(op::Reshape(op::Reduce(
          op::AllToAll(op::Reshape(op::Pad(op::Parameter(0), op::Constant()))),
          op::Constant()))))));
}

}  // namespace
}  // namespace xla
