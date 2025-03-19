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

#include "xla/service/gpu/transforms/collectives/collective_annotator.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::Optional;
using ::tsl::testing::IsOkAndHolds;

using CollectiveAnnotatorTest = HloHardwareIndependentTestBase;

TEST_F(CollectiveAnnotatorTest, CollectiveId) {
  absl::string_view kHloText = R"(
    HloModule m

    add {
        p0 = f16[] parameter(0)
        p1 = f16[] parameter(1)
        ROOT add = f16[] add(p0, p1)
    }

    ENTRY main {
        p0 = f16[10000000]{0} parameter(0)
        p1 = f16[10000000]{0} parameter(1)
        ar0 = f16[10000000]{0} all-reduce(p0), replica_groups={}, to_apply=add
        ar1 = f16[10000000]{0} all-reduce(p1), replica_groups={}, to_apply=add
        ROOT result = tuple(ar0, ar1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  EXPECT_THAT(RunHloPass(CollectiveAnnotator(), module.get()),
              IsOkAndHolds(true));
  hlo_query::ForEachInstructionWithPred(
      *module,
      [](const HloInstruction* instr) {
        return hlo_query::IsCollectiveCommunicationOp(instr->opcode());
      },
      [](const HloInstruction* instr) {
        EXPECT_THAT(CollectiveId(instr),
                    Optional(absl::StrCat(instr->unique_id())));
      });
}

}  // namespace
}  // namespace xla::gpu
