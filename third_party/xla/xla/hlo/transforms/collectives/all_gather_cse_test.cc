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

#include "xla/hlo/transforms/collectives/all_gather_cse.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class AllGatherCSETest : public HloHardwareIndependentTestBase {
 protected:
  AllGatherCSE pass_;
};

TEST_F(AllGatherCSETest, ReplacesRedundantAllGather) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY main {
      param0 = s32[4] parameter(0)
      all-gather.1 = s32[8] all-gather(param0), dimensions={0}, replica_groups={{0,1}}
      all-gather.2 = s32[8] all-gather(param0), dimensions={0}, replica_groups={{0,1}}
      ROOT tuple = (s32[8], s32[8]) tuple(all-gather.1, all-gather.2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_TRUE(changed);
  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::AllGather(op::Parameter(0)),
                        op::AllGather(op::Parameter(0))));
}

TEST_F(AllGatherCSETest, HandlesRawParameterGetTupleElement) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY main {
      param0 = (s32[8], s32[8]) parameter(0)
      gte0 = s32[8] get-tuple-element(param0), index=0
      gte1 = s32[8] get-tuple-element(param0), index=1
      all-gather.1 = s32[16] all-gather(gte0), dimensions={0}, replica_groups={{0,1}}
      all-gather.2 = s32[16] all-gather(gte1), dimensions={0}, replica_groups={{0,1}}
      ROOT tuple = (s32[16], s32[16]) tuple(all-gather.1, all-gather.2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AllGatherCSETest, HandlesRawParameterTuple) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY main {
      param0 = s32[8] parameter(0)
      param1 = s32[8] parameter(1)
      tuple0 = (s32[8], s32[8]) tuple(param0, param1)
      gte0 = s32[8] get-tuple-element(tuple0), index=0
      gte1 = s32[8] get-tuple-element(tuple0), index=1
      all-gather.1 = s32[16] all-gather(gte0), dimensions={0}, replica_groups={{0,1}}
      all-gather.2 = s32[16] all-gather(gte1), dimensions={0}, replica_groups={{0,1}}
      ROOT tuple = (s32[16], s32[16]) tuple(all-gather.1, all-gather.2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AllGatherCSETest, HandlesRawParameterOptimizationBarrierCSE) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY main {
      param0 = (s32[8], s32[8]) parameter(0)
      opt_barrier = (s32[8], s32[8]) opt-barrier(param0)
      gte0 = s32[8] get-tuple-element(opt_barrier), index=0
      all-gather.1 = s32[16] all-gather(gte0), dimensions={0}, replica_groups={{0,1}}
      all-gather.2 = s32[16] all-gather(gte0), dimensions={0}, replica_groups={{0,1}}
      ROOT tuple = (s32[16], s32[16]) tuple(all-gather.1, all-gather.2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::AllGather(op::GetTupleElement(op::OptimizationBarrier())),
                op::AllGather(op::GetTupleElement(op::OptimizationBarrier()))));
}

TEST_F(AllGatherCSETest, HandlesRawParameterConvert) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY main {
      param0 = f32[8] parameter(0)
      convert0 = s32[8] convert(param0)
      all-gather.1 = s32[16] all-gather(convert0), dimensions={0}, replica_groups={{0,1}}
      all-gather.2 = s32[16] all-gather(convert0), dimensions={0}, replica_groups={{0,1}}
      ROOT tuple = (s32[16], s32[16]) tuple(all-gather.1, all-gather.2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::AllGather(op::Convert()), op::AllGather(op::Convert())));
}

TEST_F(AllGatherCSETest, HandlesNoAllGather) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY main {
      param0 = s32[8] parameter(0)
      ROOT tuple = (s32[8], s32[8]) tuple(param0, param0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AllGatherCSETest, HandlesNonParameterOperand) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY main {
      param0 = s32[8] parameter(0)
      negate0 = s32[8] negate(param0)
      all-gather.1 = s32[16] all-gather(negate0), dimensions={0}, replica_groups={{0,1}}
      ROOT tuple = (s32[16]) tuple(all-gather.1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AllGatherCSETest, RunsHloDCEAfterChanges) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY main {
      param0 = s32[8] parameter(0)
      all-gather.1 = s32[16] all-gather(param0), dimensions={0}, replica_groups={{0,1}}
      all-gather.2 = s32[16] all-gather(param0), dimensions={0}, replica_groups={{0,1}}
      ROOT tuple = (s32[16]) tuple(all-gather.1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  HloDCE dce;
  TF_ASSERT_OK(dce.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::AllGather(op::Parameter(0))));
}

}  // namespace
}  // namespace xla
