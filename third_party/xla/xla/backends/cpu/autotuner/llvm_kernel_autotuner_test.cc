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

#include "xla/backends/cpu/autotuner/llvm_kernel_autotuner.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

constexpr absl::string_view kLlvmKernelConcatenateHlo = R"(
    HloModule fusion.1

    ENTRY e {
        p0 = f32[3,2] parameter(0)
        p1 = f32[1,2] parameter(1)
        ROOT result = f32[4,2] concatenate(p0, p1), dimensions={0}
    }
)";

class LlvmKernelAutotunerTest : public HloHardwareIndependentTestBase {};

TEST_F(LlvmKernelAutotunerTest, GetBestConfig) {
  LlvmKernelAutotuner autotuner;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kLlvmKernelConcatenateHlo));

  TF_ASSERT_OK_AND_ASSIGN(auto changed, autotuner.Run(module.get(), {}));

  EXPECT_TRUE(changed);
}

}  // namespace
}  // namespace xla::cpu
