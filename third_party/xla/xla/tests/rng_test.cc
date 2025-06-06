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

#include <cstdint>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/transforms/expanders/rng_bit_generator_expander.h"
#include "xla/hlo/transforms/expanders/rng_expander.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using RngTest = HloPjRtTestBase;

void DisableHloPass(HloModule& module, absl::string_view pass_name) {
  auto debug_options = module.config().debug_options();
  debug_options.add_xla_disable_hlo_passes(pass_name.data());
  module.mutable_config().set_debug_options(debug_options);
}

void DisableRngExpanderPass(HloModule& module) {
  RngExpander expander;
  DisableHloPass(module, expander.name());
}

void DisableRngBitGeneratorExpanderPass(HloModule& module) {
  RngBitGeneratorExpander expander(RandomAlgorithm::RNG_PHILOX);
  DisableHloPass(module, expander.name());
}

// NOTE: Tests with RNG expanders disabled are only valid for the CPU backend.
// Currently this whole test file is executed only for CPU, so it doesn't cause
// any issues.
TEST_F(RngTest, ReturnsErrorWhenExpanderPassDisabled) {
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

using RngBitGeneratorTest = HloPjRtTestBase;

TEST_F(RngBitGeneratorTest, ReturnsErrorWhenExpanderPassDisabled_Default) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = u64[3]{0} parameter(0)
      ROOT result = (u64[3]{0}, u32[11,17]{1,0}) rng-bit-generator(p0), algorithm=rng_default
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  DisableRngBitGeneratorExpanderPass(*module);

  Literal arg0 = LiteralUtil::CreateR1<uint64_t>({7, 42, 43});

  auto status_or_result = Execute(std::move(module), {&arg0});
  EXPECT_EQ(status_or_result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(
      status_or_result.status().message(),
      ::testing::HasSubstr("RngBitGenerator should be expanded for CPU"));
}

TEST_F(RngBitGeneratorTest, ReturnsErrorWhenExpanderPassDisabled_ThreeFry) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = u64[2]{0} parameter(0)
      ROOT result = (u64[2]{0}, u32[11,17]{1,0}) rng-bit-generator(p0), algorithm=rng_three_fry
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  DisableRngBitGeneratorExpanderPass(*module);

  Literal arg0 = LiteralUtil::CreateR1<uint64_t>({7, 42});

  auto status_or_result = Execute(std::move(module), {&arg0});
  EXPECT_EQ(status_or_result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(
      status_or_result.status().message(),
      ::testing::HasSubstr("RngBitGenerator should be expanded for CPU"));
}

TEST_F(RngBitGeneratorTest, ReturnsErrorWhenExpanderPassDisabled_Philox) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = u64[3]{0} parameter(0)
      ROOT result = (u64[3]{0}, u32[11,17]{1,0}) rng-bit-generator(p0), algorithm=rng_philox
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  DisableRngBitGeneratorExpanderPass(*module);

  Literal arg0 = LiteralUtil::CreateR1<uint64_t>({7, 42, 43});

  auto status_or_result = Execute(std::move(module), {&arg0});
  EXPECT_EQ(status_or_result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(
      status_or_result.status().message(),
      ::testing::HasSubstr("RngBitGenerator should be expanded for CPU"));
}

}  // namespace
}  // namespace xla
