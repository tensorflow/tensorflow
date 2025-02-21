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

#include "xla/hlo/ir/source_target_pairs.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class SourceTargetPairsTest : public ::testing::Test {
 protected:
  SourceTargetPairs fwd2_ = SourceTargetPairs({{0, 1}, {1, 0}});
  SourceTargetPairs bwd2_ = SourceTargetPairs({{0, 1}, {1, 0}});
  SourceTargetPairs fwd4_ = SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}, {3, 0}});
  SourceTargetPairs bwd4_ = SourceTargetPairs({{0, 3}, {1, 0}, {2, 1}, {3, 2}});
};

TEST_F(SourceTargetPairsTest, Compare) {
  EXPECT_EQ(SourceTargetPairs(), SourceTargetPairs());
  EXPECT_EQ(SourceTargetPairs({{0, 1}, {1, 0}}),
            SourceTargetPairs({{0, 1}, {1, 0}}));
}

TEST_F(SourceTargetPairsTest, ProtoConversion) {
  SourceTargetPairs in_pairs = SourceTargetPairs({{2, 3}, {3, 4}});

  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});
  std::unique_ptr<HloInstruction> p0 =
      HloInstruction::CreateParameter(0, shape, "p0");
  p0->SetUniqueId(0);
  std::unique_ptr<HloInstruction> in_cp =
      HloInstruction::CreateCollectivePermute(shape, p0.get(), in_pairs, 1);
  in_cp->SetUniqueId(1);
  HloInstructionProto proto = in_cp->ToProto();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloInstruction> deserialized,
      HloInstruction::CreateFromProto(proto, {{0, in_cp->mutable_operand(0)}}));

  EXPECT_EQ(deserialized->source_target_pairs(), in_pairs);
}

TEST_F(SourceTargetPairsTest, AbslStringify) {
  std::string formatted =
      absl::StrFormat("Source Target Pairs: %v",
                      ParseSourceTargetPairsOnly("{{0,1},{1,0}}").value());
  EXPECT_EQ(formatted, "Source Target Pairs: {{0,1},{1,0}}");
}

TEST_F(SourceTargetPairsTest, SourceTargetPairsString) {
  EXPECT_EQ(fwd2_.ToString(), "{{0,1},{1,0}}");
  EXPECT_EQ(bwd2_.ToString(), "{{0,1},{1,0}}");
  EXPECT_EQ(fwd4_.ToString(), "{{0,1},{1,2},{2,3},{3,0}}");
  EXPECT_EQ(bwd4_.ToString(), "{{0,3},{1,0},{2,1},{3,2}}");
}

TEST_F(SourceTargetPairsTest, GetMaxDeviceNum) {
  EXPECT_EQ(fwd2_.GetMaxDeviceNum(), 1);
  EXPECT_EQ(bwd2_.GetMaxDeviceNum(), 1);
  EXPECT_EQ(fwd4_.GetMaxDeviceNum(), 3);
  EXPECT_EQ(bwd4_.GetMaxDeviceNum(), 3);
  EXPECT_EQ(
      SourceTargetPairs({{0, 1}, {1, 2}, {2, 300}, {3, 4}}).GetMaxDeviceNum(),
      300);
}

TEST_F(SourceTargetPairsTest, IsSelfIdentity) {
  EXPECT_TRUE(SourceTargetPairs().IsSelfIdentity());
  EXPECT_TRUE(SourceTargetPairs({{0, 0}, {1, 1}}).IsSelfIdentity());
  EXPECT_FALSE(SourceTargetPairs({{0, 0}, {1, 0}}).IsSelfIdentity());
}

}  // namespace
}  // namespace xla
