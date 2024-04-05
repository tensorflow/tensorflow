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

#include "xla/service/layout_canonicalizer.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using LayoutCanonicalizerTest = HloTestBase;

TEST_F(LayoutCanonicalizerTest, CanonicalizeBroadcast) {
  const std::string hlo_string = R"(
  HloModule broadcast_module
    ENTRY %main {
      %p0 = f32[2,6]{0,1} parameter(0)
      %broadcast = f32[3,2,1,6]{2,0,1,3} broadcast(%p0), dimensions={1,3}
      ROOT %output = f32[3,2,1,6]{3,2,1,0} broadcast(%broadcast), dimensions={0,1,2,3}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  LayoutCanonicalizer canonicalizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, canonicalizer.Run(m.get()));
  ASSERT_TRUE(changed);

  // Layout should be descending.
  HloInstruction* output = m->entry_computation()->root_instruction();
  HloInstruction* broadcast = output->mutable_operand(0);
  EXPECT_EQ(broadcast->shape().layout().minor_to_major(),
            std::vector<int64_t>({3, 2, 1, 0}));

  // Logical dimensions should be as follows.
  EXPECT_EQ(broadcast->shape().dimensions(),
            std::vector<int64_t>({6, 2, 3, 1}));

  // Dimensions should change according to the new descending layout.
  EXPECT_EQ(broadcast->dimensions(), std::vector<int64_t>({1, 0}));
  EXPECT_EQ(output->dimensions(), std::vector<int64_t>({3, 1, 0, 2}));
  VLOG(3) << "module after:\n" << m->ToString();
}

TEST_F(LayoutCanonicalizerTest, CanonicalizeBroadcast2) {
  const std::string hlo_string = R"(
  HloModule broadcast_module
    ENTRY %main {
      %p0 = f32[2,6]{0,1} parameter(0)
      %broadcast = f32[3,2,1,6]{2,3,1,0} broadcast(%p0), dimensions={1,3}
      ROOT %output = f32[3,5,2,1,6]{3,4,2,1,0} broadcast(%broadcast), dimensions={0,2,3,4}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  LayoutCanonicalizer canonicalizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, canonicalizer.Run(m.get()));
  ASSERT_TRUE(changed);

  // Layout should be descending.
  HloInstruction* output = m->entry_computation()->root_instruction();
  HloInstruction* broadcast = output->mutable_operand(0);
  EXPECT_EQ(broadcast->shape().layout().minor_to_major(),
            std::vector<int64_t>({3, 2, 1, 0}));

  // Logical dimensions should be as follows.
  EXPECT_EQ(broadcast->shape().dimensions(),
            std::vector<int64_t>({3, 2, 6, 1}));

  // Dimensions should change according to the new descending layout.
  EXPECT_EQ(broadcast->dimensions(), std::vector<int64_t>({1, 2}));
  EXPECT_EQ(output->dimensions(), std::vector<int64_t>({0, 2, 4, 3}));
  VLOG(3) << "module after:\n" << m->ToString();
}

TEST_F(LayoutCanonicalizerTest, CanonicalizeBroadcast3) {
  const std::string hlo_string = R"(
  HloModule broadcast_module
    ENTRY %main {
      %p0 = f32[2,6]{0,1} parameter(0)
      %broadcast = f32[3,2,1,6]{2,3,0,1} broadcast(%p0), dimensions={1,3}
      %broadcast2 = f32[3,5,2,1,6]{3,4,0,2,1} broadcast(f32[3,2,1,6]{2,3,0,1} %broadcast), dimensions={0,2,3,4}
      ROOT %output = f32[3,5,2,1,6]{3,4,0,1,2} broadcast(f32[3,5,2,1,6]{3,4,0,2,1} %broadcast2), dimensions={0,1,2,3,4}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  LayoutCanonicalizer canonicalizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, canonicalizer.Run(m.get()));
  ASSERT_TRUE(changed);

  // Layout should be descending.
  HloInstruction* root = m->entry_computation()->root_instruction();
  HloInstruction* broadcast2 = root->mutable_operand(0);
  HloInstruction* broadcast = broadcast2->mutable_operand(0);
  EXPECT_EQ(broadcast->shape().layout().minor_to_major(),
            std::vector<int64_t>({3, 2, 1, 0}));
  EXPECT_EQ(broadcast2->shape().layout().minor_to_major(),
            std::vector<int64_t>({4, 3, 2, 1, 0}));

  // Logical dimensions should be as follows.
  EXPECT_EQ(broadcast->shape().dimensions(),
            std::vector<int64_t>({2, 3, 6, 1}));
  EXPECT_EQ(broadcast2->shape().dimensions(),
            std::vector<int64_t>({5, 2, 3, 6, 1}));

  // Dimensions should change according to the new descending layout.
  EXPECT_EQ(broadcast->dimensions(), std::vector<int64_t>({0, 2}));
  EXPECT_EQ(broadcast2->dimensions(), std::vector<int64_t>({1, 2, 3, 4}));
  EXPECT_EQ(root->dimensions(), std::vector<int64_t>({1, 2, 0, 4, 3}));
  VLOG(3) << "module after:\n" << m->ToString();
}

TEST_F(LayoutCanonicalizerTest, CanonicalLayout) {
  const std::string hlo_string = R"(
  HloModule broadcast_module
    ENTRY %main {
      %p0 = f32[2,6]{0,1} parameter(0)
      %broadcast = f32[3,2,1,6]{3,2,1,0} broadcast(%p0), dimensions={1,3}
      ROOT %output = f32[3,2,1,6]{3,2,1,0} broadcast(%broadcast), dimensions={0,1,2,3}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  LayoutCanonicalizer canonicalizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, canonicalizer.Run(m.get()));
  ASSERT_FALSE(changed);
}

TEST_F(LayoutCanonicalizerTest, CanonicalizeTranspose) {
  const std::string hlo_string = R"(
  HloModule broadcast_module
    ENTRY %main {
      %p0 = bf16[32,1,58,50]{3,2,1,0} parameter(0)
      %transpose = bf16[32,1,58,50]{1,0,2,3} transpose(%p0), dimensions={0,1,2,3}
      ROOT %output = bf16[32,1,58,50]{3,2,1,0} transpose(%transpose), dimensions={0,1,2,3}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  LayoutCanonicalizer canonicalizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, canonicalizer.Run(m.get()));
  ASSERT_TRUE(changed);

  // Layout should be descending.
  HloInstruction* output = m->entry_computation()->root_instruction();
  HloInstruction* transpose = output->mutable_operand(0);
  EXPECT_EQ(transpose->shape().layout().minor_to_major(),
            std::vector<int64_t>({3, 2, 1, 0}));

  // Logical dimensions should be as follows.
  EXPECT_EQ(transpose->shape().dimensions(),
            std::vector<int64_t>({50, 58, 32, 1}));

  // Dimensions should change according to the new descending layout.
  EXPECT_EQ(transpose->dimensions(), std::vector<int64_t>({3, 2, 0, 1}));
  EXPECT_EQ(output->dimensions(), std::vector<int64_t>({2, 3, 1, 0}));
  VLOG(3) << "module after:\n" << m->ToString();
}

TEST_F(LayoutCanonicalizerTest, CanonicalizeTransposeAndBroadcast) {
  const std::string hlo_string = R"(
  HloModule broadcast_module
    ENTRY %main {
      %p0 = f32[2,6]{0,1} parameter(0)
      %broadcast = f32[3,2,1,6]{2,0,1,3} broadcast(%p0), dimensions={1,3}
      ROOT %output = f32[3,2,1,6]{3,2,1,0} transpose(%broadcast), dimensions={0,1,2,3}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  LayoutCanonicalizer canonicalizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, canonicalizer.Run(m.get()));
  ASSERT_TRUE(changed);

  // Layout should be descending.
  HloInstruction* output = m->entry_computation()->root_instruction();
  HloInstruction* broadcast = output->mutable_operand(0);
  EXPECT_EQ(broadcast->shape().layout().minor_to_major(),
            std::vector<int64_t>({3, 2, 1, 0}));

  // Logical dimensions should be as follows.
  EXPECT_EQ(broadcast->shape().dimensions(),
            std::vector<int64_t>({6, 2, 3, 1}));

  // Dimensions should change according to the new descending layout.
  EXPECT_EQ(broadcast->dimensions(), std::vector<int64_t>({1, 0}));
  EXPECT_EQ(output->dimensions(), std::vector<int64_t>({2, 1, 3, 0}));
  VLOG(3) << "module after:\n" << m->ToString();
}

}  // namespace
}  // namespace xla
