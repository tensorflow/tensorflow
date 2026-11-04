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

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class AllReduceHloTest : public HloTestBase {};

TEST_F(AllReduceHloTest, DefaultDeviceAssnWithHloRunner) {
  // xla::HloRunnerPjRt passes a single device assignment. Test this returns an
  // error gracefully.
  const char* const hlo_string = R"(
    HloModule module, replica_count=2

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY test {
      param = f32[1024] parameter(0)
      ROOT result = f32[1024] all-reduce(param), to_apply=add, replica_groups={{0,1}}
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  Literal input = LiteralUtil::CreateR1<float>(std::vector<float>(1, 2));

  EXPECT_THAT(test_runner().Execute(std::move(module), {std::move(input)}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Mismatched number of replicas for device "
                                 "assignment and computation (1 vs 2).")));
}

}  // namespace
}  // namespace xla::gpu
