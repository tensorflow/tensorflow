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

#include "xla/service/gpu/transforms/dot_precision_eliminator.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

class DotPrecisionEliminatorTest : public HloHardwareIndependentTestBase {
 protected:
  DotPrecisionEliminator pass_;
};

// Default precision specified, or no precision specified.
// Expect: no change, pass returns false.
TEST_F(DotPrecisionEliminatorTest, DefaultPrecisionNoChange) {
  const char* hlo_string_default = R"(
HloModule module
ENTRY main {
  p0 = f32[2,2]{1,0} parameter(0)
  p1 = f32[2,2]{1,0} parameter(1)
  ROOT dot = f32[2,2]{1,0} dot(p0, p1),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0},
    operand_precision={default,default}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_default,
                          ParseAndReturnVerifiedModule(hlo_string_default));
  TF_ASSERT_OK_AND_ASSIGN(bool changed_default,
                          pass_.Run(module_default.get()));
  EXPECT_FALSE(changed_default);
  EXPECT_TRUE(RunFileCheck(module_default->ToString(), R"(
// CHECK: ROOT {{.*}} = f32[2,2]{1,0} dot({{.*}}, {{.*}})
// CHECK-NOT: operand_precision
// CHECK-NOT: algorithm
)")
                  .value_or(false));

  const char* hlo_string_no_precision = R"(
HloModule module
ENTRY main {
  p0 = f32[2,2]{1,0} parameter(0)
  p1 = f32[2,2]{1,0} parameter(1)
  ROOT dot = f32[2,2]{1,0} dot(p0, p1),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module_no_precision,
      ParseAndReturnVerifiedModule(hlo_string_no_precision));
  TF_ASSERT_OK_AND_ASSIGN(bool changed_no_precision,
                          pass_.Run(module_no_precision.get()));
  EXPECT_FALSE(changed_no_precision);
  EXPECT_TRUE(RunFileCheck(module_no_precision->ToString(), R"(
// CHECK: ROOT {{.*}} = f32[2,2]{1,0} dot({{.*}}, {{.*}})
// CHECK-NOT: operand_precision
// CHECK-NOT: algorithm
)")
                  .value_or(false));
}

// Complex types
// Expect: no change, pass returns false.
TEST_F(DotPrecisionEliminatorTest, ComplexTypeNoChange) {
  const char* hlo_string = R"(
HloModule module
ENTRY main {
  p0 = c64[2,2]{1,0} parameter(0)
  p1 = c64[2,2]{1,0} parameter(1)
  ROOT dot = c64[2,2]{1,0} dot(p0, p1),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0},
    operand_precision={highest,highest}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
// CHECK: ROOT {{.*}} = c64[2,2]{1,0} dot({{.*}}, {{.*}})
// CHECK-SAME: operand_precision={highest,highest}
)")
                  .value_or(false));
}

// Non-floating point types
// Expect: operand_precision cleared, pass returns true.
TEST_F(DotPrecisionEliminatorTest, NonFloatingPointClearsOperandPrecision) {
  const char* hlo_string = R"(
HloModule module
ENTRY main {
  p0 = s32[2,2]{1,0} parameter(0)
  p1 = s32[2,2]{1,0} parameter(1)
  ROOT dot = s32[2,2]{1,0} dot(p0, p1),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0},
    operand_precision={highest,highest}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
// CHECK: ROOT {{.*}} = s32[2,2]{1,0} dot({{.*}}, {{.*}})
// CHECK-NOT: operand_precision
// CHECK-NOT: algorithm
)")
                  .value_or(false));
}

// F32 operands with HIGHEST precision.
// Expect: algorithm set to ALG_DOT_F32_F32_F32, operand_precision cleared, pass
// returns true.
TEST_F(DotPrecisionEliminatorTest, F32HighestPrecisionOperandsSetsAlgorithm) {
  const char* hlo_string = R"(
HloModule module
ENTRY main {
  p0 = f32[2,2]{1,0} parameter(0)
  p1 = f32[2,2]{1,0} parameter(1)
  ROOT dot = f32[2,2]{1,0} dot(p0, p1),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0},
    operand_precision={highest,highest}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
// CHECK: ROOT {{.*}} = f32[2,2]{1,0} dot({{.*}}, {{.*}})
// CHECK-NOT: operand_precision
// CHECK-SAME: algorithm=dot_f32_f32_f32
)")
                  .value_or(false));
}

// F32 operands with HIGH precision.
// Expect: operand_precision cleared, no algorithm set, pass returns true.
TEST_F(DotPrecisionEliminatorTest, F32HighPrecisionDoesntSetAlgorithm) {
  const char* hlo_string = R"(
HloModule module
ENTRY main {
  p0 = f32[2,2]{1,0} parameter(0)
  p1 = f32[2,2]{1,0} parameter(1)
  ROOT dot = f32[2,2]{1,0} dot(p0, p1),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0},
    operand_precision={high,high}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
// CHECK: ROOT {{.*}} = f32[2,2]{1,0} dot({{.*}}, {{.*}})
// CHECK-NOT: operand_precision
// CHECK-NOT: algorithm
)")
                  .value_or(false));
}

// Operands less precise than F32 (BF16) with non-default precision.
// Expect: Operands upcast to F32, operand_precision cleared, pass returns true.
TEST_F(DotPrecisionEliminatorTest, BF16OperandsUpcastToF32) {
  const char* hlo_string = R"(
HloModule module
ENTRY main {
  p0 = bf16[2,2]{1,0} parameter(0)
  p1 = bf16[2,2]{1,0} parameter(1)
  ROOT dot = bf16[2,2]{1,0} dot(p0, p1),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0},
    operand_precision={highest,highest}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
// CHECK: %[[P0CONV:.+]] = f32[2,2]{1,0} convert(%p0)
// CHECK: %[[P1CONV:.+]] = f32[2,2]{1,0} convert(%p1)
// CHECK: %[[DOT:.+]] = f32[2,2]{1,0} dot(%[[P0CONV]], %[[P1CONV]])
// CHECK-NOT: operand_precision
// CHECK-NOT: algorithm
// CHECK: ROOT {{.*}} = bf16[2,2]{1,0} convert(%[[DOT]])
)")
                  .value_or(false));
}

}  // namespace

}  // namespace gpu
}  // namespace xla
