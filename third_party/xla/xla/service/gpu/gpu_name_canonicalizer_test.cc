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

#include "xla/service/gpu/gpu_name_canonicalizer.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using GpuNameCanonicalizerTest = HloTestBase;

TEST_F(GpuNameCanonicalizerTest, RenamesNonCanonicalInstructionNames) {
  constexpr absl::string_view kHloString = R"(
    HloModule module

    ENTRY entry {
      p12.1.2.4 = f32[1] parameter(0)
      p12.1 = f32[1] parameter(1)
      add3 = f32[1] add(p12.1.2.4,p12.1)
      ROOT exp.1.23 = f32[1] exponential(add3)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  GpuNameCanonicalizer pass;
  TF_ASSERT_OK_AND_ASSIGN(bool _, pass.Run(module.get()));

  // We rename instructions post order: exp -> add -> p12s.
  constexpr absl::string_view kExpected = R"(
// CHECK: %p12 = {{.*}} parameter(0)
// CHECK: %p12.2 = {{.*}} parameter(1)
// CHECK: %add3.1 = {{.*}} add(%p12, %p12.2)
// CHECK: %exp = {{.*}} exponential(%add3.1)
  )";
  TF_ASSERT_OK_AND_ASSIGN(
      bool filecheck_matches,
      RunFileCheck(
          module->ToString(HloPrintOptions().set_print_operand_shape(false)),
          kExpected));
  EXPECT_TRUE(filecheck_matches);
}

TEST_F(GpuNameCanonicalizerTest, RenamesNonCanonicalComputationNames) {
  constexpr absl::string_view kHloString = R"(
    HloModule module

    fusion.2.3 {
      p.2 = f32[1] parameter(0)
      p.3 = f32[1] parameter(1)
      ROOT _ = f32[1] add(p.2,p.3)
    }

    fusion.7.8 {
      p.4 = f32[1] parameter(0)
      ROOT _ = f32[1] exponential(p.4)
    }

    ENTRY entry {
      p = f32[1] parameter(0)
      p.1 = f32[1] parameter(1)
      add = f32[1] fusion(p,p.1), calls=fusion.2.3, kind=kCustom
      ROOT _ = f32[1] fusion(add), calls=fusion.7.8, kind=kCustom
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  GpuNameCanonicalizer pass;
  TF_ASSERT_OK_AND_ASSIGN(bool _, pass.Run(module.get()));

  constexpr absl::string_view kExpected = R"(
// CHECK: %fusion {{.*}} {
// CHECK: %fusion.1 {{.*}} {
  )";
  TF_ASSERT_OK_AND_ASSIGN(
      bool filecheck_matches,
      RunFileCheck(
          module->ToString(HloPrintOptions().set_print_operand_shape(false)),
          kExpected));
  EXPECT_TRUE(filecheck_matches);
}

}  // namespace
}  // namespace xla::gpu
