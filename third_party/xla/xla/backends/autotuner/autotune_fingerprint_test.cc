/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/autotuner/autotune_fingerprint.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace {

using AutotuneFingerprintTest = HloHardwareIndependentTestBase;

const char* kHlo1 = R"(
HloModule module1
ENTRY entry {
  p0 = f32[10,10]{1,0} parameter(0)
  p1 = f32[10,10]{1,0} parameter(1)
  ROOT dot = f32[10,10]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

TEST_F(AutotuneFingerprintTest, GetHloFingerprintConsistency) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  tsl::Fprint128 fp1 = GetHloFingerprint(*dot);
  tsl::Fprint128 fp2 = GetHloFingerprint(*dot);
  EXPECT_EQ(fp1, fp2);
}

TEST_F(AutotuneFingerprintTest, GetCodegenOptionsFingerprintToggles) {
  DebugOptions options;
  options.set_xla_gpu_cublas_fallback(false);

  std::string fp1 = GetCodegenOptionsFingerprint(options);

  options.set_xla_gpu_cublas_fallback(true);
  std::string fp2 = GetCodegenOptionsFingerprint(options);

  EXPECT_NE(fp1, fp2);
}

}  // namespace
}  // namespace xla
