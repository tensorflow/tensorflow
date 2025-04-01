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

#include "xla/service/gpu/transforms/all_reduce_decomposer.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/platform_util.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class AllReduceDecomposerTest : public HloRunnerAgnosticTestBase {
 public:
  AllReduceDecomposerTest()
      : HloRunnerAgnosticTestBase(std::make_unique<HloRunner>(
            PlatformUtil::GetDefaultPlatform().value())) {}
};

TEST_F(AllReduceDecomposerTest, SmallAllReduceIsDecomposed) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  input = f32[16] parameter(0)
  ROOT all-reduce = f32[16] all-reduce(input), replica_groups={{0,1}}, to_apply=add
}
)"));

  AllReduceDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));
  EXPECT_TRUE(changed);
  TF_EXPECT_OK(VerifyHloModule(module.get(), true, true));
  TF_EXPECT_OK(HloCSE(true).Run(module.get()));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: f32[1,16]{1,0} reshape
    // CHECK: f32[2,16]{1,0} all-gather
    // CHECK: f32[16]{0} reduce
  )"));
}

TEST_F(AllReduceDecomposerTest, LargeAllReduceIsNotDecomposed) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  input = f32[16777216] parameter(0)
  ROOT all-reduce = f32[16777216] all-reduce(input), replica_groups={{0,1}}, to_apply=add
}
)"));

  AllReduceDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
