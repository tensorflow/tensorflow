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

#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using RngTest = HloTestBase;

void DisableRngExpanderPass(HloModule& module) {
  auto debug_options = module.config().debug_options();
  debug_options.add_xla_disable_hlo_passes("rng-expander");
  module.mutable_config().set_debug_options(debug_options);
}

// NOTE: This test is only valid for the CPU backend. Currently this whole test
// file is executed only for CPU, so it doesn't cause any issues.
XLA_TEST_F(RngTest, ReturnsErrorWhenRngExpanderDisabled) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT result = f32[] rng(p0, p1), distribution=rng_uniform
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  DisableRngExpanderPass(*module);

  Literal arg0 = LiteralUtil::CreateR0<float>(7.f);
  Literal arg1 = LiteralUtil::CreateR0<float>(42.f);

  auto status_or_result = Execute(std::move(module), {&arg0, &arg1});
  EXPECT_EQ(status_or_result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(status_or_result.status().message(),
              ::testing::HasSubstr("Rng should be expanded for CPU"));
}

}  // namespace
}  // namespace xla
