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

#include "xla/hlo/transforms/expanders/ragged_dot_rewriter.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

using RaggedDotRewriterTest = HloHardwareIndependentTestBase;

TEST_F(RaggedDotRewriterTest, DontDoAnythingIfNoRaggedDot) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[64,4]{1,0} parameter(0)
    p1 = f32[64,4]{1,0} parameter(1)
    ROOT dot = f32[64]{0} dot(p0, p1), lhs_batch_dims={0},
                                       lhs_contracting_dims={1},
                                       rhs_batch_dims={0},
                                       rhs_contracting_dims={1}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RaggedDotRewriter().Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RaggedDotRewriterTest, RaggedNonContracting) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = bf16[64,9]{1,0} parameter(0)
    p1 = bf16[2,9,8]{2,1,0} parameter(1)
    p2 = s64[2]{0} parameter(2)
    ROOT ragged-dot = bf16[64,8]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RaggedDotRewriter().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Select(), op::Parameter()));
  RunAndFilecheckHloRewrite(module_string, RaggedDotRewriter(), R"(
  // CHECK: ROOT [[dot:%[^ ]+]] = bf16[64,8]{1,0}
  // CHECK-SAME: lhs_contracting_dims={0,2}, rhs_contracting_dims={0,1}
  )");
}

TEST_F(RaggedDotRewriterTest, RaggedNonContractingWithBatchDimensions) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = bf16[3,64,9]{2,1,0} parameter(0)
    p1 = bf16[3,2,9,8]{3,2,1,0} parameter(1)
    p2 = s64[3,2]{1,0} parameter(2)
    ROOT ragged-dot = bf16[3,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={2},
                      lhs_batch_dims={0}, rhs_batch_dims={0},
                      lhs_ragged_dims={1}, rhs_group_dims={1}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RaggedDotRewriter().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Select(), op::Parameter()));
  RunAndFilecheckHloRewrite(module_string, RaggedDotRewriter(), R"(
  // CHECK: ROOT [[dot:%[^ ]+]] = bf16[3,64,8]{2,1,0}
  // CHECK-SAME: lhs_batch_dims={0}, lhs_contracting_dims={1,3},
  // CHECK-SAME: rhs_batch_dims={0}, rhs_contracting_dims={1,2}
  )");
}

TEST_F(RaggedDotRewriterTest, RaggedContracting) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[11,5]{1,0} parameter(0)
    p1 = f32[5,7]{1,0} parameter(1)
    p2 = s32[3]{0} parameter(2)
    ROOT ragged-dot = f32[3,11,7]{2,1,0} ragged-dot(p0, p1, p2),
                                      lhs_contracting_dims={1},
                                      rhs_contracting_dims={0},
                                      lhs_ragged_dims={1}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RaggedDotRewriter().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Select(), op::Select()));
  RunAndFilecheckHloRewrite(module_string, RaggedDotRewriter(), R"(
  // CHECK: ROOT [[dot:%[^ ]+]] = f32[3,11,7]{2,1,0}
  // CHECK-SAME: lhs_batch_dims={0}, lhs_contracting_dims={2},
  // CHECK-SAME: rhs_batch_dims={0}, rhs_contracting_dims={1}
  )");
}

TEST_F(RaggedDotRewriterTest, BatchContracting) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = bf16[3,64,9]{2,1,0} parameter(0)
    p1 = bf16[3,9,8]{2,1,0} parameter(1)
    p2 = s64[3]{0} parameter(2)
    ROOT ragged-dot = bf16[3,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={1},
                      lhs_batch_dims={0}, rhs_batch_dims={0},
                      lhs_ragged_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RaggedDotRewriter().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Parameter(0), op::Parameter(1),
                      /*lhs_contracting_dim=*/2, /*rhs_contracting_dim=*/1));
}

}  // namespace
}  // namespace xla
