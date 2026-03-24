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

#include "xla/hlo/transforms/shape_canonicalizer.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_pool.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class ShapeCanonicalizerTest : public HloHardwareIndependentTestBase {};

TEST_F(ShapeCanonicalizerTest, Canonicalize) {
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

  ShapePool shape_pool;
  ShapeCanonicalizer shape_canonicalizer(&shape_pool);

  EXPECT_FALSE(shape_canonicalizer.Run(module0.get()).value());
  EXPECT_TRUE(shape_canonicalizer.Run(module1.get()).value());

  auto* c0 = Cast<HloConstantInstruction>(
      module0->entry_computation()->root_instruction());
  auto* c1 = Cast<HloConstantInstruction>(
      module1->entry_computation()->root_instruction());

  // We compare instruction shape pointers for equality, as we expect them to
  // point to the same object in the shape pool.
  EXPECT_EQ(&c0->shape(), &c1->shape());
}

}  // namespace
}  // namespace xla
