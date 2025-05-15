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

#include "xla/service/gpu/transforms/dot_sparsity_rewriter.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;

class DotSparsityRewriterTest : public HloHardwareIndependentTestBase {
 public:
  DotSparsityRewriterTest()
      : HloHardwareIndependentTestBase(/*verifier_layout_sensitive=*/true) {}
};

TEST_F(DotSparsityRewriterTest, SparseDotRhsToLhs) {
  const char* module_string = R"(
HloModule m

ENTRY e {
  lhs = f16[4,2,16,8,64] parameter(0)
  rhs = f16[2,4,8,32,128] parameter(1)
  meta = u16[2,4,8,4,128] parameter(2)
  ROOT dot = f16[4,2,16,128] dot(lhs, rhs, meta),
    lhs_contracting_dims={3,4}, rhs_contracting_dims={2,3},
    lhs_batch_dims={0,1}, rhs_batch_dims={1,0}, sparsity=R.3@2:4
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          DotSparsityRewriter().Run(module.get()));
  EXPECT_TRUE(modified);

  const HloTransposeInstruction* transpose = DynCast<HloTransposeInstruction>(
      module->entry_computation()->root_instruction());
  ASSERT_TRUE(transpose != nullptr);
  EXPECT_THAT(transpose->dimensions(), ElementsAre(0, 1, 3, 2));

  const HloDotInstruction* dot =
      DynCast<HloDotInstruction>(transpose->operand(0));
  ASSERT_TRUE(dot != nullptr);

  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  EXPECT_EQ(dnums.lhs_contracting_dimensions(0), 2);
  EXPECT_EQ(dnums.lhs_contracting_dimensions(1), 3);
  EXPECT_EQ(dnums.rhs_contracting_dimensions(0), 3);
  EXPECT_EQ(dnums.rhs_contracting_dimensions(1), 4);
  EXPECT_EQ(dnums.lhs_batch_dimensions(0), 1);
  EXPECT_EQ(dnums.lhs_batch_dimensions(1), 0);
  EXPECT_EQ(dnums.rhs_batch_dimensions(0), 0);
  EXPECT_EQ(dnums.rhs_batch_dimensions(1), 1);

  EXPECT_EQ(dot->sparse_operands(), 1);
  EXPECT_EQ(dot->sparsity().front().index(), 0);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
