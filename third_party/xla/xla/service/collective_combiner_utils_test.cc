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

#include "xla/service/collective_combiner_utils.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

std::unique_ptr<HloModule> CreateModule(const std::string& hlo_string) {
  auto module_or_status = ParseAndReturnUnverifiedModule(hlo_string);
  if (!module_or_status.ok()) {
    return nullptr;
  }
  return std::move(module_or_status.value());
}

TEST(CollectiveCombinerUtilsTest, MergeFrontendAttributesEmpty) {
  auto module = CreateModule(R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[8,32] parameter(0)
      ROOT result = f32[8,32] copy(p0)
    }
  )");
  ASSERT_NE(module, nullptr);

  HloInstruction* copy =
      module->entry_computation()->GetInstructionWithName("result");
  std::vector<HloInstruction*> instructions = {copy};

  FrontendAttributes result = MergeFrontendAttributes(instructions);
  EXPECT_TRUE(result.map().empty());
}

TEST(CollectiveCombinerUtilsTest, MergeFrontendAttributesSingleInstruction) {
  auto module = CreateModule(R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[8,32] parameter(0)
      ROOT ag0 = f32[8,32] all-gather(p0), dimensions={0},
            replica_groups={}, frontend_attributes={foo="bar"}
    }
  )");
  ASSERT_NE(module, nullptr);

  HloInstruction* ag0 =
      module->entry_computation()->GetInstructionWithName("ag0");
  std::vector<HloInstruction*> instructions = {ag0};

  FrontendAttributes result = MergeFrontendAttributes(instructions);
  EXPECT_EQ(result.map().at("foo"), "bar");
}

TEST(CollectiveCombinerUtilsTest,
     MergeFrontendAttributesDeduplicatesSharedKeys) {
  auto module = CreateModule(R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[8,32] parameter(0)
      p1 = f32[4,64] parameter(1)
      ag0 = f32[8,32] all-gather(p0), dimensions={0},
            replica_groups={}, frontend_attributes={key1="value1", key2="a"}
      ag1 = f32[4,64] all-gather(p1), dimensions={1},
            replica_groups={}, frontend_attributes={key1="value1", key2="b"}
      ROOT result = (f32[8,32], f32[4,64]) tuple(ag0, ag1)
    }
  )");
  ASSERT_NE(module, nullptr);

  HloInstruction* ag0 =
      module->entry_computation()->GetInstructionWithName("ag0");
  HloInstruction* ag1 =
      module->entry_computation()->GetInstructionWithName("ag1");
  std::vector<HloInstruction*> instructions = {ag0, ag1};

  FrontendAttributes result = MergeFrontendAttributes(instructions);

  // key1 has the same value in both → appears once.
  EXPECT_EQ(result.map().at("key1"), "value1");

  // key2 has different values → sorted and comma-joined (btree_set order).
  EXPECT_EQ(result.map().at("key2"), "a,b");
}

TEST(CollectiveCombinerUtilsTest, MergeFrontendAttributesSortedValues) {
  auto module = CreateModule(R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[8,32] parameter(0)
      p1 = f32[4,64] parameter(1)
      p2 = f32[16,16] parameter(2)
      ag0 = f32[8,32] all-gather(p0), dimensions={0},
            replica_groups={}, frontend_attributes={key="zebra"}
      ag1 = f32[4,64] all-gather(p1), dimensions={1},
            replica_groups={}, frontend_attributes={key="apple"}
      ag2 = f32[16,16] all-gather(p2), dimensions={1},
            replica_groups={}, frontend_attributes={key="mango"}
      ROOT result = (f32[8,32], f32[4,64], f32[16,16]) tuple(ag0, ag1, ag2)
    }
  )");
  ASSERT_NE(module, nullptr);

  HloInstruction* ag0 =
      module->entry_computation()->GetInstructionWithName("ag0");
  HloInstruction* ag1 =
      module->entry_computation()->GetInstructionWithName("ag1");
  HloInstruction* ag2 =
      module->entry_computation()->GetInstructionWithName("ag2");
  std::vector<HloInstruction*> instructions = {ag0, ag1, ag2};

  FrontendAttributes result = MergeFrontendAttributes(instructions);

  // Values are sorted by btree_set.
  EXPECT_EQ(result.map().at("key"), "apple,mango,zebra");
}

TEST(CollectiveCombinerUtilsTest, MergeMetadataEmpty) {
  std::vector<HloInstruction*> instructions;
  OpMetadata result = MergeMetadata(instructions);
  EXPECT_TRUE(result.source_file().empty());
  EXPECT_TRUE(result.op_name().empty());
}

TEST(CollectiveCombinerUtilsTest, MergeMetadataSingleInstruction) {
  auto module = CreateModule(R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[8,32] parameter(0)
      ROOT ag0 = f32[8,32] all-gather(p0), dimensions={0}, replica_groups={},
            metadata={op_name="test_op" source_file="test.py" source_line=42}
    }
  )");
  ASSERT_NE(module, nullptr);

  HloInstruction* ag0 =
      module->entry_computation()->GetInstructionWithName("ag0");
  std::vector<HloInstruction*> instructions = {ag0};

  OpMetadata result = MergeMetadata(instructions);
  EXPECT_EQ(result.op_name(), "test_op");
  EXPECT_EQ(result.source_file(), "test.py");
  EXPECT_EQ(result.source_line(), 42);
}

TEST(CollectiveCombinerUtilsTest, MergeMetadataMultipleInstructions) {
  auto module = CreateModule(R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[8,32] parameter(0)
      p1 = f32[4,64] parameter(1)
      ag0 = f32[8,32] all-gather(p0), dimensions={0}, replica_groups={},
            metadata={op_name="op0" source_file="test.py" source_line=42}
      ag1 = f32[4,64] all-gather(p1), dimensions={1}, replica_groups={},
            metadata={op_name="op1" source_file="test.py" source_line=50}
      ROOT result = (f32[8,32], f32[4,64]) tuple(ag0, ag1)
    }
  )");
  ASSERT_NE(module, nullptr);

  HloInstruction* ag0 =
      module->entry_computation()->GetInstructionWithName("ag0");
  HloInstruction* ag1 =
      module->entry_computation()->GetInstructionWithName("ag1");
  std::vector<HloInstruction*> instructions = {ag0, ag1};

  OpMetadata result = MergeMetadata(instructions);

  // Source locations concatenated as file:line pairs.
  EXPECT_EQ(result.source_file(), "test.py:42,test.py:50");
  EXPECT_EQ(result.source_line(), 0);
  EXPECT_EQ(result.source_end_line(), 0);

  // No common '/' prefix → "(op0:op1)".
  EXPECT_EQ(result.op_name(), "(op0:op1)");
}

TEST(CollectiveCombinerUtilsTest, MergeMetadataIdenticalOpNames) {
  auto module = CreateModule(R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[8,32] parameter(0)
      p1 = f32[4,64] parameter(1)
      ag0 = f32[8,32] all-gather(p0), dimensions={0}, replica_groups={},
            metadata={op_name="identical_op" source_file="test.py" source_line=42}
      ag1 = f32[4,64] all-gather(p1), dimensions={1}, replica_groups={},
            metadata={op_name="identical_op" source_file="test.py" source_line=50}
      ROOT result = (f32[8,32], f32[4,64]) tuple(ag0, ag1)
    }
  )");
  ASSERT_NE(module, nullptr);

  HloInstruction* ag0 =
      module->entry_computation()->GetInstructionWithName("ag0");
  HloInstruction* ag1 =
      module->entry_computation()->GetInstructionWithName("ag1");
  std::vector<HloInstruction*> instructions = {ag0, ag1};

  OpMetadata result = MergeMetadata(instructions);

  // All names identical → returned as-is.
  EXPECT_EQ(result.op_name(), "identical_op");
}

TEST(CollectiveCombinerUtilsTest, MergeMetadataCommonPrefixExtraction) {
  auto module = CreateModule(R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[8,32] parameter(0)
      p1 = f32[4,64] parameter(1)
      p2 = f32[16,16] parameter(2)
      ag0 = f32[8,32] all-gather(p0), dimensions={0}, replica_groups={},
            metadata={op_name="module/layer/op_0" source_file="test.py" source_line=42}
      ag1 = f32[4,64] all-gather(p1), dimensions={1}, replica_groups={},
            metadata={op_name="module/layer/op_1" source_file="test.py" source_line=50}
      ag2 = f32[16,16] all-gather(p2), dimensions={1}, replica_groups={},
            metadata={op_name="module/layer/op_2" source_file="test.py" source_line=60}
      ROOT result = (f32[8,32], f32[4,64], f32[16,16]) tuple(ag0, ag1, ag2)
    }
  )");
  ASSERT_NE(module, nullptr);

  HloInstruction* ag0 =
      module->entry_computation()->GetInstructionWithName("ag0");
  HloInstruction* ag1 =
      module->entry_computation()->GetInstructionWithName("ag1");
  HloInstruction* ag2 =
      module->entry_computation()->GetInstructionWithName("ag2");
  std::vector<HloInstruction*> instructions = {ag0, ag1, ag2};

  OpMetadata result = MergeMetadata(instructions);

  // Common prefix "module/layer/" extracted, suffixes joined with ':'.
  EXPECT_EQ(result.op_name(), "module/layer/(op_0:op_1:op_2)");
}

}  // namespace
}  // namespace xla
