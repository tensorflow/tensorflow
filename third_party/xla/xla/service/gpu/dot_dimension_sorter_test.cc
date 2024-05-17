/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/dot_dimension_sorter.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class WithoutDotDimensionSorterTest : public GpuCodegenTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    // The pass is disabled here to preserve suboptimal dimension order in
    // 1) UnsortedDimsCreateTransposes to reveal the transposes.
    // 2) DimOrderCanBeChanged for the comparison of ordered vs unordered.
    // The pass does not touch SortedDimsDoNotCreateTransposes anyway because
    // the dimensions are already ordered there.
    debug_options.add_xla_disable_hlo_passes("dot_dimension_sorter");
    return debug_options;
  }
};

TEST_F(WithoutDotDimensionSorterTest, UnsortedDimsCreateTransposes) {
  const char* hlo_text = R"(
HloModule m

ENTRY e {
  p0 = f16[1,14,9,32] parameter(0)
  p1 = f16[12,9,32] parameter(1)
  ROOT _ = f16[1,14,12] dot(p0, p1),
    lhs_contracting_dims={3,2}, rhs_contracting_dims={2,1}
}
)";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: transpose
)");
}

TEST_F(WithoutDotDimensionSorterTest, SortedDimsDoNotCreateTransposes) {
  const char* hlo_text = R"(
HloModule m

ENTRY e {
  p0 = f16[1,14,9,32] parameter(0)
  p1 = f16[12,9,32] parameter(1)
  ROOT _ = f16[1,14,12] dot(p0, p1),
    lhs_contracting_dims={2,3}, rhs_contracting_dims={1,2}
}
)";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK-NOT: transpose
)");
}

TEST_F(WithoutDotDimensionSorterTest, DimOrderCanBeChanged) {
  const char* hlo_text_ref = R"(
HloModule m

ENTRY e {
  p0 = f16[1,14,9,32] parameter(0)
  p1 = f16[12,9,32] parameter(1)
  ROOT _ = f16[1,14,12] dot(p0, p1),
    lhs_contracting_dims={3,2}, rhs_contracting_dims={2,1}
}
)";

  const char* hlo_text_modified = R"(
HloModule m

ENTRY e {
  p0 = f16[1,14,9,32] parameter(0)
  p1 = f16[12,9,32] parameter(1)
  ROOT _ = f16[1,14,12] dot(p0, p1),
    lhs_contracting_dims={2,3}, rhs_contracting_dims={1,2}
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_modified,
                                      ErrorSpec{1e-5, 1e-3},
                                      /*run_hlo_passes=*/true));
}

using DotDimensionSorterTest = GpuCodegenTest;

TEST_F(DotDimensionSorterTest, SortContractingDims) {
  const char* module_string = R"(
HloModule m

ENTRY e {
  p0 = f16[1,144,96,32] parameter(0)
  p1 = f16[122,96,32] parameter(1)
  ROOT _ = f16[1,144,122] dot(p0, p1),
    lhs_contracting_dims={3,2}, rhs_contracting_dims={2,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  const auto& dims =
      module->entry_computation()->root_instruction()->dot_dimension_numbers();

  EXPECT_EQ(dims.lhs_contracting_dimensions(0), 3);
  EXPECT_EQ(dims.lhs_contracting_dimensions(1), 2);

  EXPECT_EQ(dims.rhs_contracting_dimensions(0), 2);
  EXPECT_EQ(dims.rhs_contracting_dimensions(1), 1);

  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          DotDimensionSorter().Run(module.get()));
  EXPECT_TRUE(modified);
  const auto& dims2 =
      module->entry_computation()->root_instruction()->dot_dimension_numbers();

  EXPECT_EQ(dims2.lhs_contracting_dimensions(0), 2);
  EXPECT_EQ(dims2.lhs_contracting_dimensions(1), 3);

  EXPECT_EQ(dims2.rhs_contracting_dimensions(0), 1);
  EXPECT_EQ(dims2.rhs_contracting_dimensions(1), 2);
}

TEST_F(DotDimensionSorterTest, NothingToReorder) {
  const char* module_string = R"(
HloModule m

ENTRY e {
  p0 = f16[1,144,96,32] parameter(0)
  p1 = f16[122,96,32] parameter(1)
  ROOT _ = f16[1,144,122] dot(p0, p1),
    lhs_contracting_dims={2,3}, rhs_contracting_dims={1,2}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));

  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          DotDimensionSorter().Run(module.get()));
  EXPECT_FALSE(modified);
}

TEST_F(DotDimensionSorterTest, SparseDotSortContractingDims) {
  const char* module_string = R"(
HloModule m

ENTRY e {
  p0 = f16[1,144,96,16] parameter(0)
  p1 = f16[122,96,32] parameter(1)
  meta = u16[1,144,96,2] parameter(2)
  ROOT _ = f16[1,144,122] dot(p0, p1, meta), sparsity=L.3@2:4,
    lhs_contracting_dims={3,2}, rhs_contracting_dims={2,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          DotDimensionSorter().Run(module.get()));
  EXPECT_TRUE(modified);
  HloDotInstruction* dot = DynCast<HloDotInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_TRUE(dot != nullptr && dot->sparse_operands() == 1);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
