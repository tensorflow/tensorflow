/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/dead_dynamic_update_slice_elimination.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = ::xla::match;

class DeadDynamicUpdateSliceEliminationTest
    : public HloHardwareIndependentTestBase {};

TEST_F(DeadDynamicUpdateSliceEliminationTest, NoDeadDUS) {
  const absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %constant.0 = bf16[] constant(0)
  %idx.1806 = s32[] constant(1806)
  %idx.0 = s32[] constant(0)
  %param.0 = bf16[2408,16] parameter(0)
  %update_block = bf16[301,16] broadcast(%constant.0), dimensions={}
  %dus = bf16[2408,16] dynamic-update-slice(%param.0, %update_block, %idx.1806, %idx.0)
  ROOT %slice = bf16[602,16] slice(%dus), slice={[1505:2107], [0:16]}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  DeadDynamicUpdateSliceElimination dds;
  EXPECT_FALSE(dds.Run(module.get()).value());
}

TEST_F(DeadDynamicUpdateSliceEliminationTest, MultiUsersNoDeadDUS) {
  const absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %constant.0 = bf16[] constant(0)
  %idx.1806 = s32[] constant(1806)
  %idx.0 = s32[] constant(0)
  %param.0 = bf16[2408,16] parameter(0)
  %update_block = bf16[301,16] broadcast(%constant.0), dimensions={}
  %dus = bf16[2408,16] dynamic-update-slice(%param.0, %update_block, %idx.1806, %idx.0)
  %slice.0 = bf16[301,16] slice(%dus), slice={[1505:1806], [0:16]}
  %slice.1 = bf16[301,16] slice(%dus), slice={[1806:2107], [0:16]}
  ROOT %tuple = (bf16[301,16], bf16[301,16]) tuple(%slice.0, %slice.1)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  DeadDynamicUpdateSliceElimination dds;
  EXPECT_FALSE(dds.Run(module.get()).value());
}

TEST_F(DeadDynamicUpdateSliceEliminationTest, RemoveDeadDUS) {
  const absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %constant.0 = bf16[] constant(0)
  %idx.1806 = s32[] constant(1806)
  %idx.0 = s32[] constant(0)
  %param.0 = bf16[2408,16] parameter(0)
  %update_block = bf16[301,16] broadcast(%constant.0), dimensions={}
  %dus = bf16[2408,16] dynamic-update-slice(%param.0, %update_block, %idx.1806, %idx.0)
  ROOT %slice = bf16[301,16] slice(%dus), slice={[1505:1806], [0:16]}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  DeadDynamicUpdateSliceElimination dds;
  EXPECT_TRUE(dds.Run(module.get()).value());
  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).value());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Slice(m::Parameter(0))));
}

TEST_F(DeadDynamicUpdateSliceEliminationTest, RemoveDeadDUSChain) {
  const absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %param.0 = bf16[256,2408,1,16,256] parameter(0)
  %constant.bf16.0 = bf16[] constant(0)
  %broadcast.12717 = bf16[256,301,1,16,256] broadcast(%constant.bf16.0), dimensions={}
  %constant.6347 = s32[] constant(0)
  %constant.6386 = s32[] constant(2107)
  %constant.6387 = s32[] constant(1806)
  %constant.6388 = s32[] constant(1505)
  %constant.6389 = s32[] constant(1204)
  %constant.6390 = s32[] constant(903)
  %constant.6391 = s32[] constant(602)
  %constant.6392 = s32[] constant(301)
  %dynamic-update-slice.643 = bf16[256,2408,1,16,256] dynamic-update-slice(%param.0, %broadcast.12717, %constant.6347, %constant.6386, %constant.6347, %constant.6347, %constant.6347)
  %gather.214 = bf16[256,301,1,16,256] slice(%dynamic-update-slice.643), slice={[0:256], [1806:2107], [0:1], [0:16], [0:256]}
  %dynamic-update-slice.644 = bf16[256,2408,1,16,256] dynamic-update-slice(%dynamic-update-slice.643, %broadcast.12717, %constant.6347, %constant.6387, %constant.6347, %constant.6347, %constant.6347)
  %gather.215 = bf16[256,301,1,16,256] slice(%dynamic-update-slice.644), slice={[0:256], [1505:1806], [0:1], [0:16], [0:256]}
  %dynamic-update-slice.645 = bf16[256,2408,1,16,256] dynamic-update-slice(%dynamic-update-slice.644, %broadcast.12717, %constant.6347, %constant.6388, %constant.6347, %constant.6347, %constant.6347)
  %gather.216 = bf16[256,301,1,16,256] slice(%dynamic-update-slice.645), slice={[0:256], [1204:1505], [0:1], [0:16], [0:256]}
  %dynamic-update-slice.646 = bf16[256,2408,1,16,256] dynamic-update-slice(%dynamic-update-slice.645, %broadcast.12717, %constant.6347, %constant.6389, %constant.6347, %constant.6347, %constant.6347)
  %gather.217 = bf16[256,301,1,16,256] slice(%dynamic-update-slice.646), slice={[0:256], [903:1204], [0:1], [0:16], [0:256]}
  %dynamic-update-slice.647 = bf16[256,2408,1,16,256] dynamic-update-slice(%dynamic-update-slice.646, %broadcast.12717, %constant.6347, %constant.6390, %constant.6347, %constant.6347, %constant.6347)
  %gather.218 = bf16[256,301,1,16,256] slice(%dynamic-update-slice.647), slice={[0:256], [602:903], [0:1], [0:16], [0:256]}
  %dynamic-update-slice.648 = bf16[256,2408,1,16,256] dynamic-update-slice(%dynamic-update-slice.647, %broadcast.12717, %constant.6347, %constant.6391, %constant.6347, %constant.6347, %constant.6347)
  %gather.219 = bf16[256,301,1,16,256] slice(%dynamic-update-slice.648), slice={[0:256], [301:602], [0:1], [0:16], [0:256]}
  %dynamic-update-slice.649 = bf16[256,2408,1,16,256] dynamic-update-slice(%dynamic-update-slice.648, %broadcast.12717, %constant.6347, %constant.6392, %constant.6347, %constant.6347, %constant.6347)
  %gather.220 = bf16[256,301,1,16,256] slice(%dynamic-update-slice.649), slice={[0:256], [0:301], [0:1], [0:16], [0:256]}
  ROOT %result = (bf16[256,301,1,16,256], bf16[256,301,1,16,256], bf16[256,301,1,16,256], bf16[256,301,1,16,256], bf16[256,301,1,16,256], bf16[256,301,1,16,256], bf16[256,301,1,16,256]) tuple(%gather.214, %gather.215, %gather.216, %gather.217, %gather.218, %gather.219, %gather.220)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  DeadDynamicUpdateSliceElimination dds;
  EXPECT_TRUE(dds.Run(module.get()).value());
  for (HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kDynamicUpdateSlice);
  }
}

}  // namespace
}  // namespace xla
