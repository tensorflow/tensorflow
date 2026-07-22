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

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/error_spec.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using HloEvaluatorSparsityTest = HloHardwareIndependentTestBase;

TEST_F(HloEvaluatorSparsityTest, EvaluatorSparseConvMaterialization) {
  const char* kHloString = R"(
HloModule sparse_conv_module

ENTRY %entry (p0: f32[1, 4], p1: f32[1, 1], p2: s32[1, 1]) -> f32[1, 1] {
  %p0 = f32[1, 4] parameter(0)
  %p1 = f32[1, 1] parameter(1)
  %p2 = s32[1, 1] parameter(2)
  %kernel = (f32[1, 1], s32[1, 1]) tuple(f32[1, 1] %p1, s32[1, 1] %p2)
  ROOT %convolution.1 = f32[1, 1] convolution(%p0, %kernel), dim_labels=bf_io->bf,
      sparsity_config={
        rhs={sparsity=1x4 dimension=0 stride=1}
      }
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  auto p0 = LiteralUtil::CreateR2<float>({{1.0f, 2.0f, 3.0f, 4.0f}});
  auto p1 = LiteralUtil::CreateR2<float>({{10.0f}});
  auto p2 = LiteralUtil::CreateR2<int32_t>(
      {{1}});  // Maps to the 2.0f index in the block.

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator.Evaluate(*module->entry_computation(), {&p0, &p1, &p2}));

  auto expected = LiteralUtil::CreateR2<float>({{20.f}});
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, ErrorSpec{0.001, 0.001}));
}

TEST_F(HloEvaluatorSparsityTest, EvaluatorSparseConvMaterializationMultiBlock) {
  const char* kHloString = R"(
HloModule sparse_conv_multi_block_module

ENTRY %entry (p0: f32[1, 8], p1: f32[2, 1], p2: s32[2, 1]) -> f32[1, 1] {
  %p0 = f32[1, 8] parameter(0)
  %p1 = f32[2, 1] parameter(1)
  %p2 = s32[2, 1] parameter(2)
  %kernel = (f32[2, 1], s32[2, 1]) tuple(f32[2, 1] %p1, s32[2, 1] %p2)
  ROOT %convolution.1 = f32[1, 1] convolution(%p0, %kernel), dim_labels=bf_io->bf,
      sparsity_config={
        rhs={sparsity=1x4 dimension=0 stride=1}
      }
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  auto p0 = LiteralUtil::CreateR2<float>(
      {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}});
  auto p1 = LiteralUtil::CreateR2<float>({{10.0f}, {20.0f}});
  auto p2 = LiteralUtil::CreateR2<int32_t>(
      {{1}, {3}});  // Maps to index 1 in block 0 (2.0f) and index 3 in block 1
                    // (8.0f).

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator.Evaluate(*module->entry_computation(), {&p0, &p1, &p2}));

  auto expected = LiteralUtil::CreateR2<float>({{180.f}});
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, ErrorSpec{0.001, 0.001}));
}

TEST_F(HloEvaluatorSparsityTest, EvaluatorSparseConvMaterializationBlock8) {
  const char* kHloString = R"(
HloModule sparse_conv_block8_module

ENTRY %entry (p0: f32[1, 8], p1: f32[1, 1], p2: s32[1, 1]) -> f32[1, 1] {
  %p0 = f32[1, 8] parameter(0)
  %p1 = f32[1, 1] parameter(1)
  %p2 = s32[1, 1] parameter(2)
  %kernel = (f32[1, 1], s32[1, 1]) tuple(f32[1, 1] %p1, s32[1, 1] %p2)
  ROOT %convolution.1 = f32[1, 1] convolution(%p0, %kernel), dim_labels=bf_io->bf,
      sparsity_config={
        rhs={sparsity=1x8 dimension=0 stride=1}
      }
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  auto p0 = LiteralUtil::CreateR2<float>(
      {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}});
  auto p1 = LiteralUtil::CreateR2<float>({{10.0f}});
  auto p2 = LiteralUtil::CreateR2<int32_t>(
      {{5}});  // Maps to index 5 in the block (6.0f).

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator.Evaluate(*module->entry_computation(), {&p0, &p1, &p2}));

  auto expected = LiteralUtil::CreateR2<float>({{60.0f}});  // 10.0 * 6.0 = 60.0
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, ErrorSpec{0.001, 0.001}));
}

TEST_F(HloEvaluatorSparsityTest, EvaluatorSparseOverflowBlockSize) {
  const char* kHloString = R"(
HloModule sparse_overflow_module

ENTRY %entry (p0: f32[1, 4], p1: f32[1, 1], p2: s32[1, 1]) -> f32[1, 1] {
  %p0 = f32[1, 4] parameter(0)
  %p1 = f32[1, 1] parameter(1)
  %p2 = s32[1, 1] parameter(2)
  %kernel = (f32[1, 1], s32[1, 1]) tuple(f32[1, 1] %p1, s32[1, 1] %p2)
  ROOT %convolution.1 = f32[1, 1] convolution(%p0, %kernel), dim_labels=bf_io->bf,
      sparsity_config={
        rhs={sparsity=1x1000000000 dimension=0 stride=1}
      }
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kHloString));
  auto p0 = LiteralUtil::CreateR2<float>({{1.0f, 2.0f, 3.0f, 4.0f}});
  auto p1 = LiteralUtil::CreateR2<float>({{10.0f}});
  auto p2 = LiteralUtil::CreateR2<int32_t>({{1}});

  HloEvaluator evaluator;
  absl::StatusOr<Literal> result =
      evaluator.Evaluate(*module->entry_computation(), {&p0, &p1, &p2});

  EXPECT_THAT(
      result.status(),
      ::testing::AnyOf(
          absl_testing::StatusIs(absl::StatusCode::kInvalidArgument),
          absl_testing::StatusIs(absl::StatusCode::kResourceExhausted)));
}

}  // namespace
}  // namespace xla
