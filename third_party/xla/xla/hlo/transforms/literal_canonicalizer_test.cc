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

#include "xla/hlo/transforms/literal_canonicalizer.h"

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_pool.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class LiteralCanonicalizerTest : public HloHardwareIndependentTestBase {};

TEST_F(LiteralCanonicalizerTest, CanonicalizeConstants) {
  absl::string_view hlo_string = R"(
    HloModule m

    ENTRY %entry {
      ROOT %c0 = f32[4] constant({1.0, 2.0, 3.0, 4.0})
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module0,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(auto module1,
                          ParseAndReturnVerifiedModule(hlo_string));

  LiteralPool literal_pool;
  LiteralCanonicalizer literal_canonicalizer(&literal_pool, 0);

  EXPECT_FALSE(literal_canonicalizer.Run(module0.get()).value());
  EXPECT_TRUE(literal_canonicalizer.Run(module1.get()).value());

  auto* c0 = Cast<HloConstantInstruction>(
      module0->entry_computation()->root_instruction());
  auto* c1 = Cast<HloConstantInstruction>(
      module1->entry_computation()->root_instruction());

  EXPECT_EQ(c0->literal(), c1->literal());
}

}  // namespace
}  // namespace xla
