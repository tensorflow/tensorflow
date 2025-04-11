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

#include "xla/service/gpu/transforms/triangular_solve_rewriter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace m = ::xla::match;

namespace xla {
namespace gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

using TriangularSolveRewriterTest = HloHardwareIndependentTestBase;

TEST_F(TriangularSolveRewriterTest, TriangularSolveWithTranspose) {
  const char* const hlo_string = R"(
HloModule TriangularSolve

ENTRY main {
  a = f32[4,4]{1,0} parameter(0)
  b = f32[3,4]{1,0} parameter(1)
  ROOT triangular-solve = f32[3,4]{1,0} triangular-solve(a, b), lower=true,
                                          transpose_a=TRANSPOSE
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TriangularSolveRewriter rewriter;
  EXPECT_THAT(rewriter.Run(module.get()), IsOkAndHolds(true));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({"__cublas$triangularSolve"}))));
}

TEST_F(TriangularSolveRewriterTest, RightLowerNoTranspose) {
  const char* const hlo_string = R"(
HloModule TriangularSolve

ENTRY %RightLowerNoTranspose (a: f32[4,4], b: f32[3,4]) -> f32[3,4] {
  a = f32[4,4]{1,0} parameter(0)
  b = f32[3,4]{1,0} parameter(1)
  ROOT %solve = f32[3,4]{1,0} triangular-solve(a, b), lower=true, transpose_a=NO_TRANSPOSE
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TriangularSolveRewriter rewriter;
  EXPECT_THAT(rewriter.Run(module.get()), IsOkAndHolds(true));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({"__cublas$triangularSolve"}))));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
