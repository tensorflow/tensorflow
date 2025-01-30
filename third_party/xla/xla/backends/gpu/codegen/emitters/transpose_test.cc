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

#include "xla/backends/gpu/codegen/emitters/transpose.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/tests/hlo_test_base.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;

class TransposeTest : public HloTestBase {
 public:
  TransposeSpec GetTransposeSpecFromRoot(absl::string_view hlo_text) {
    auto module = ParseAndReturnVerifiedModule(hlo_text).value();
    auto* root = module->entry_computation()->root_instruction();
    return GetTransposeSpec(Cast<HloTransposeInstruction>(root));
  }
};

TEST_F(TransposeTest, Transpose_10) {
  auto spec = GetTransposeSpecFromRoot(R"(ENTRY entry {
    p0 = f32[8, 32] parameter(0)
    ROOT transpose_p0 = f32[32, 8] transpose(p0), dimensions={1, 0}
  })");
  EXPECT_THAT(spec.permutation, ElementsAre(1, 0));
  EXPECT_THAT(spec.inv_permutation, ElementsAre(1, 0));
  EXPECT_THAT(spec.canonical_input_shape, ElementsAre(8, 1, 32, 1));
  EXPECT_THAT(spec.canonical_output_shape, ElementsAre(32, 1, 8, 1));
  EXPECT_THAT(spec.canonical_permutation, ElementsAre(2, 1, 0, 3));
  EXPECT_THAT(spec.canonical_inv_permutation, ElementsAre(2, 1, 0, 3));
}

TEST_F(TransposeTest, Transpose_210) {
  auto spec = GetTransposeSpecFromRoot(R"(ENTRY entry {
    p0 = f32[8, 2, 32] parameter(0)
    ROOT transpose_p0 = f32[32, 2, 8] transpose(p0), dimensions={2, 1, 0}
  })");
  EXPECT_THAT(spec.canonical_input_shape, ElementsAre(8, 2, 32, 1));
  EXPECT_THAT(spec.canonical_output_shape, ElementsAre(32, 2, 8, 1));
  EXPECT_THAT(spec.canonical_permutation, ElementsAre(2, 1, 0, 3));
  EXPECT_THAT(spec.canonical_inv_permutation, ElementsAre(2, 1, 0, 3));
}

TEST_F(TransposeTest, Transpose_102) {
  auto spec = GetTransposeSpecFromRoot(R"(ENTRY entry {
    p0 = f32[8, 2, 32] parameter(0)
    ROOT transpose_p0 = f32[2, 8, 32] transpose(p0), dimensions={1, 0, 2}
  })");
  EXPECT_THAT(spec.canonical_input_shape, ElementsAre(8, 1, 2, 32));
  EXPECT_THAT(spec.canonical_output_shape, ElementsAre(2, 1, 8, 32));
  EXPECT_THAT(spec.canonical_permutation, ElementsAre(2, 1, 0, 3));
  EXPECT_THAT(spec.canonical_inv_permutation, ElementsAre(2, 1, 0, 3));
}

TEST_F(TransposeTest, Transpose_42130) {
  auto spec = GetTransposeSpecFromRoot(R"(ENTRY entry {
    p0 = f32[8, 2, 32, 7, 6] parameter(0)
    ROOT transpose_p0 = f32[6, 32, 2, 7, 8] transpose(p0),
      dimensions={4, 2, 1, 3, 0}
  })");
  EXPECT_THAT(spec.canonical_input_shape, ElementsAre(8, 2, 32, 7, 6, 1));
  EXPECT_THAT(spec.canonical_output_shape, ElementsAre(6, 32, 2, 7, 8, 1));
  EXPECT_THAT(spec.canonical_permutation, ElementsAre(4, 2, 1, 3, 0, 5));
  EXPECT_THAT(spec.canonical_inv_permutation, ElementsAre(4, 2, 1, 3, 0, 5));
}

}  // namespace
}  // namespace xla::gpu
