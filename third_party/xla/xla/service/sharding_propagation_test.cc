/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/sharding_propagation.h"

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_op_metadata.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_splitter.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/protobuf_util.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using ShardingPropagationTest = HloTestBase;

void ClearMetadata(HloModule* module) {
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->metadata().ByteSizeLong() != 0) {
        instruction->set_metadata(OpMetadata());
      }
      if (!instruction->has_sharding()) {
        continue;
      }
      instruction->set_sharding(instruction->sharding().WithoutMetadata());
    }
  }
}

struct MetadataTestParameter {
  explicit MetadataTestParameter(bool propagate_metadata, bool clear_metadata)
      : propagate_metadata(propagate_metadata),
        clear_metadata(clear_metadata) {}

  bool propagate_metadata = false;
  bool clear_metadata = false;
};

struct MetadataTestParameterWithOutput {
  explicit MetadataTestParameterWithOutput(bool propagate_metadata,
                                           bool clear_metadata,
                                           bool allow_root_sharding_propagation)
      : propagate_metadata(propagate_metadata),
        clear_metadata(clear_metadata),
        allow_root_sharding_propagation(allow_root_sharding_propagation) {}

  bool propagate_metadata = false;
  bool clear_metadata = false;
  bool allow_root_sharding_propagation = false;
};

class ParameterizedMetadataTest
    : public HloTestBase,
      public ::testing::WithParamInterface<MetadataTestParameter> {};

class ParameterizedMetadataTestWithOutput
    : public HloTestBase,
      public ::testing::WithParamInterface<MetadataTestParameterWithOutput> {};

std::string OpMetadataListToString(absl::Span<const OpMetadata> metadata) {
  std::vector<std::string> metadata_strings;
  metadata_strings.reserve(metadata.size());
  for (const OpMetadata& element : metadata) {
    metadata_strings.push_back(
        absl::StrCat("{", OpMetadataToString(element), "}"));
  }
  return absl::StrCat("{", absl::StrJoin(metadata_strings, ", "), "}");
}

class HloShardingMetadataMatcher
    : public ::testing::MatcherInterface<const HloSharding&> {
 public:
  explicit HloShardingMetadataMatcher(absl::Span<const OpMetadata> metadata)
      : metadata_(metadata.begin(), metadata.end()) {}

  bool MatchAndExplain(
      const HloSharding& sharding,
      ::testing::MatchResultListener* listener) const override {
    if (sharding.metadata().size() != metadata_.size()) {
      *listener << sharding.ToString(/*include_metadata=*/true)
                << " has incorrect sharding metadata (expected: "
                << OpMetadataListToString(metadata_) << ")";
      return false;
    }

    for (int i = 0, e = metadata_.size(); i < e; ++i) {
      if (!protobuf_util::ProtobufEquals(sharding.metadata()[i],
                                         metadata_[i])) {
        *listener << sharding.ToString(/*include_metadata=*/true)
                  << " has incorrect sharding metadata (expected: "
                  << OpMetadataListToString(metadata_) << ")";
        return false;
      }
    }

    return true;
  }

  void DescribeTo(std::ostream* os) const override {
    *os << OpMetadataListToString(metadata_);
  }

 private:
  std::vector<OpMetadata> metadata_;
};

::testing::Matcher<const HloSharding&> ShardingMetadata(
    absl::Span<const OpMetadata> metadata) {
  return ::testing::MakeMatcher(new HloShardingMetadataMatcher(metadata));
}

OpMetadata CreateMetadata(const std::string& op_name) {
  OpMetadata metadata;
  metadata.set_op_name(op_name);
  return metadata;
}

INSTANTIATE_TEST_SUITE_P(
    ShardingPropagation, ParameterizedMetadataTest,
    ::testing::Values(MetadataTestParameter(/*propagate_metadata=*/false,
                                            /*clear_metadata=*/false),
                      MetadataTestParameter(/*propagate_metadata=*/false,
                                            /*clear_metadata=*/true),
                      MetadataTestParameter(/*propagate_metadata=*/true,
                                            /*clear_metadata=*/false),
                      MetadataTestParameter(/*propagate_metadata=*/true,
                                            /*clear_metadata=*/true)),
    [](const ::testing::TestParamInfo<MetadataTestParameter>& info) {
      return absl::StrCat(info.param.propagate_metadata
                              ? "MetadataPropagation"
                              : "NoMetadataPropagation",
                          "_",
                          info.param.clear_metadata ? "NoMetadataInModule"
                                                    : "MetadataInModule");
    });

INSTANTIATE_TEST_SUITE_P(
    ShardingPropagation, ParameterizedMetadataTestWithOutput,
    ::testing::Values(MetadataTestParameterWithOutput(
                          /*propagate_metadata=*/false,
                          /*clear_metadata=*/false,
                          /*allow_root_sharding_propagation=*/false),
                      MetadataTestParameterWithOutput(
                          /*propagate_metadata=*/false,
                          /*clear_metadata=*/true,
                          /*allow_root_sharding_propagation=*/false),
                      MetadataTestParameterWithOutput(
                          /*propagate_metadata=*/true,
                          /*clear_metadata=*/false,
                          /*allow_root_sharding_propagation=*/false),
                      MetadataTestParameterWithOutput(
                          /*propagate_metadata=*/true,
                          /*clear_metadata=*/true,
                          /*allow_root_sharding_propagation=*/false),
                      MetadataTestParameterWithOutput(
                          /*propagate_metadata=*/false,
                          /*clear_metadata=*/false,
                          /*allow_root_sharding_propagation=*/true),
                      MetadataTestParameterWithOutput(
                          /*propagate_metadata=*/false,
                          /*clear_metadata=*/true,
                          /*allow_root_sharding_propagation=*/true),
                      MetadataTestParameterWithOutput(
                          /*propagate_metadata=*/true,
                          /*clear_metadata=*/false,
                          /*allow_root_sharding_propagation=*/true),
                      MetadataTestParameterWithOutput(
                          /*propagate_metadata=*/true,
                          /*clear_metadata=*/true,
                          /*allow_root_sharding_propagation=*/true)),
    [](const ::testing::TestParamInfo<MetadataTestParameterWithOutput>& info) {
      return absl::StrCat(
          info.param.propagate_metadata ? "MetadataPropagation"
                                        : "NoMetadataPropagation",
          "_",
          info.param.clear_metadata ? "NoMetadataInModule" : "MetadataInModule",
          "_",
          info.param.allow_root_sharding_propagation ? "PropagateToRoot"
                                                     : "NoPropagateToRoot");
    });

TEST_P(ParameterizedMetadataTest, ShardingMetadataFromInstruction) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,2,1]0,1,2,3},
    metadata={op_name="test"}
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%param0)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  EXPECT_EQ(changed,
            GetParam().propagate_metadata && !GetParam().clear_metadata);
  auto* instruction = FindInstruction(module.get(), "param0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("test")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_F(ShardingPropagationTest, ShardingMetadataFromInstructionNoOverwrite) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,2,1]0,1,2,3 metadata={op_name="name"}},
    metadata={op_name="test"}
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%param0)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation(/*is_spmd=*/false,
                                              /*propagate_metadata=*/true)
                              .Run(module.get()));
  EXPECT_FALSE(changed);
  auto* instruction = FindInstruction(module.get(), "param0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  EXPECT_THAT(instruction->sharding(),
              ShardingMetadata({CreateMetadata("name")}));
}

TEST_F(ShardingPropagationTest, ShardingMetadataFromInstructionNoMetadata) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,2,1]0,1,2,3 metadata={op_name="name"}}
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%param0)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation(/*is_spmd=*/false,
                                              /*propagate_metadata=*/true)
                              .Run(module.get()));
  EXPECT_FALSE(changed);
  auto* instruction = FindInstruction(module.get(), "param0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  EXPECT_THAT(instruction->sharding(),
              ShardingMetadata({CreateMetadata("name")}));
}

TEST_F(ShardingPropagationTest, ShardingNoMetadataAndInstructionNoMetadata) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,2,1]0,1,2,3}
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%param0)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation(/*is_spmd=*/false,
                                              /*propagate_metadata=*/true)
                              .Run(module.get()));
  EXPECT_FALSE(changed);
  auto* instruction = FindInstruction(module.get(), "param0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
}

TEST_P(ParameterizedMetadataTest, ElementwiseOperationForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,2,1]0,1,2,3 metadata={op_name="a"}}
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1)
  %add = f32[5,7,11,13]{3,2,1,0} add(%param0, %param1)
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%add)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "add");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ElementwiseOperationBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0)
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1)
  %add = f32[5,7,11,13]{3,2,1,0} add(%param0, %param1)
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%add),
    sharding={devices=[1,2,2,1]0,1,2,3 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "add");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

// Regression Test for b/129569657.
TEST_P(ParameterizedMetadataTestWithOutput, BroadcastForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %broadcast {
  %param0 = f32[3,2048,2048]{2,1,0} parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 metadata={op_name="a"}}
  %broadcast = f32[3,2048,2048,3]{3,2,1,0} broadcast(%param0), dimensions={0,1,2}
  ROOT %copy = f32[3,2048,2048,3]{3,2,1,0} copy(%broadcast)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata,
                          {GetParam().allow_root_sharding_propagation})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "broadcast");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
  if (GetParam().allow_root_sharding_propagation) {
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  }
}

// Regression Test for b/129569657.
TEST_P(ParameterizedMetadataTestWithOutput, BroadcastForwardPassWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %broadcast {
  %param0 = f32[3,2048,2048]{2,1,0} parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 metadata={op_name="a"}}
  %shard-barrier-from = f32[3,2048,2048]{2,1,0} custom-call(%param0), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %broadcast = f32[3,2048,2048,3]{3,2,1,0} broadcast(%shard-barrier-from), dimensions={0,1,2}
  ROOT %copy = f32[3,2048,2048,3]{3,2,1,0} copy(%broadcast)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata,
                          {GetParam().allow_root_sharding_propagation})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "broadcast");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTest, BroadcastBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %broadcast {
  %param0 = f32[13]{0} parameter(0)
  %broadcast = f32[5,7,11,13]{3,2,1,0} broadcast(%param0), dimensions={3}
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%broadcast),
    sharding={devices=[1,2,2,1]0,1,2,3 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "broadcast");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, BroadcastBackwardPassWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %broadcast {
  %param0 = f32[13]{0} parameter(0)
  %param0_copy = f32[13]{0} copy(param0)
  %shard-barrier-to = f32[13]{0} custom-call(%param0_copy), custom_call_target="ShardBarrierTo", custom_call_has_side_effect=true
  %broadcast = f32[5,7,11,13]{3,2,1,0} broadcast(%shard-barrier-to), dimensions={3}
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%broadcast),
    sharding={devices=[1,1,2,2]0,1,2,3 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "param0_copy");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{replicated}"));
}

TEST_P(ParameterizedMetadataTest, Broadcast1DBackwardNoChange) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %broadcast {
  %param0 = s32[128]{0} parameter(0)
  %constant0 = s32[] constant(0), sharding={replicated}
  %broadcast = s32[128]{0} broadcast(%constant0), dimensions={}, sharding={replicated}
  ROOT %compare = pred[128]{0} compare(s32[128]{0} %param0, s32[128]{0} %broadcast),
    direction=NE, sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  EXPECT_FALSE(changed);
  auto* instruction = FindInstruction(module.get(), "broadcast");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{replicated}"));
}

TEST_P(ParameterizedMetadataTestWithOutput, BroadcastForwardPartial) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %broadcast {
  %param0 = f32[3,2048]parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="a"}}
  %broadcast = f32[3,2048,3] broadcast(%param0), dimensions={0,1}
  ROOT %copy = f32[3,2048,3] copy(%broadcast)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata,
                          {GetParam().allow_root_sharding_propagation})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "broadcast");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding("{devices=[1,2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
  if (GetParam().allow_root_sharding_propagation) {
    EXPECT_THAT(
        module->entry_computation()->root_instruction(),
        op::Sharding("{devices=[1,2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  }
}

TEST_P(ParameterizedMetadataTest, BroadcastMerge) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %broadcast {
  %param0 = f32[3,2048]parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="a"}}
  %broadcast = f32[3,2048,3] broadcast(%param0), dimensions={0,1}
  ROOT %copy = f32[3,2048,3] copy(%broadcast),
    sharding={devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate metadata={op_name="b"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "broadcast");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,2]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a"), CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, BroadcastUser) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %broadcast {
  %param0 = f32[24,8]{0,1} parameter(0)
  %copy = f32[24,8]{0,1} copy(%param0)
  ROOT %broadcast = f32[4,24,6,8]{3,2,1,0} broadcast(%copy), dimensions={1,3},
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "copy");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,4]0,1,2,3,4,5,6,7}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTestWithOutput, BroadcastUserPartial) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %broadcast {
  %param0 = f32[24,8]{0,1} parameter(0)
  %copy = f32[24,8]{0,1} copy(%param0)
  ROOT %broadcast = f32[4,24,6,8] broadcast(%copy), dimensions={1,3},
    sharding={devices=[4,2,1,1]0,1,2,3,4,5,6,7 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata,
                          {GetParam().allow_root_sharding_propagation})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "copy");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding("{devices=[2,1,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }

  if (GetParam().allow_root_sharding_propagation) {
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                op::Sharding("{devices=[4,2,1,1]0,1,2,3,4,5,6,7}"));
  }
}

TEST_P(ParameterizedMetadataTest, MaximalReduceForwardPass) {
  const char* const hlo_string = R"(
HloModule module
%add {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}
ENTRY %reduce {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,2,1]0,1,2,3 metadata={op_name="a"}}
  %init = f32[] parameter(1)
  %reduce = f32[5,7]{1,0} reduce(%param0, %init), dimensions={2,3}, to_apply=%add
  ROOT %copy = f32[5,7]{0,1} copy(%reduce)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "reduce");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{replicated}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_F(ShardingPropagationTest, ManualTupleReduceForwardPass) {
  const char* const hlo_string = R"(
HloModule module

%minmax_func {
  %lhs_value = f32[] parameter(0)
  %rhs_value = f32[] parameter(2)
  %compare.2 = pred[] compare(%lhs_value, %rhs_value), direction=GT
  %select.4 = f32[] select(%compare.2, %lhs_value, %rhs_value)
  %lhs_index = s32[] parameter(1)
  %rhs_index = s32[] parameter(3)
  %select.5 = s32[] select(%compare.2, %lhs_index, %rhs_index)
  ROOT %tuple.2 = (f32[], s32[]) tuple(%select.4, %select.5)
}
ENTRY %reduce {
  get-tuple-element.416 = f32[2,1,128]{2,1,0} parameter(0), sharding={manual}
  get-tuple-element.417 = s32[2,1,128]{2,1,0} parameter(1), sharding={manual}
  constant.3793 = f32[] constant(0)
  constant.3795 = s32[] constant(0)
  reduce.418 = (f32[2,1]{1,0}, s32[2,1]{1,0}) reduce(
    get-tuple-element.416, get-tuple-element.417, constant.3793, constant.3795),
    dimensions={2}, to_apply=minmax_func
  ROOT %copy = (f32[2,1]{1,0}, s32[2,1]{1,0}) copy(%reduce.418)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "reduce.418");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{{manual}, {manual}}"));
}

TEST_P(ParameterizedMetadataTest, ShardedReduceForwardPass) {
  const char* const hlo_string = R"(
HloModule module
%add {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}
ENTRY %reduce {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,2,1]0,1,2,3 metadata={op_name="a"}}
  %init = f32[] parameter(1)
  %reduce = f32[7,11]{1,0} reduce(%param0, %init), dimensions={0,3}, to_apply=%add
  ROOT %copy = f32[7,11]{0,1} copy(f32[7,11]{1,0} %reduce)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "reduce");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,2]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ReduceForwardPassWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
%add {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}
ENTRY %reduce {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,2,1]0,1,2,3 metadata={op_name="a"}}
  %init = f32[] parameter(1)
  %shard-barrier-from = f32[5,7,11,13]{3,2,1,0} custom-call(%param0), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %reduce = f32[7,11]{1,0} reduce(%shard-barrier-from, %init), dimensions={0,3}, to_apply=%add
  ROOT %copy = f32[7,11]{0,1} copy(f32[7,11]{1,0} %reduce)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "reduce");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTest, ReducePartiallyOnTiledDims) {
  const char* const hlo_string = R"(
HloModule module
%add {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}
ENTRY %reduce {
  %param0 = f32[8,8] parameter(0),
    sharding={devices=[2,2]0,1,2,3 metadata={op_name="a"}}
  %init = f32[] parameter(1)
  %reduce = f32[8] reduce(%param0, %init), dimensions={0}, to_apply=%add
  ROOT %copy = f32[8] copy(%reduce)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "reduce");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[2,2]0,2,1,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ReducePartiallyOnTiledDims2) {
  const char* const hlo_string = R"(
HloModule module
%add {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}
ENTRY %reduce {
  %param0 = f32[8,8] parameter(0),
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate metadata={op_name="a"}}
  %init = f32[] parameter(1)
  %reduce = f32[8] reduce(%param0, %init), dimensions={0}, to_apply=%add
  ROOT %copy = f32[8] copy(%reduce)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "reduce");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding("{devices=[2,4]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ReducePartiallyBackward) {
  const char* const hlo_string = R"(
HloModule module
%add {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}
ENTRY %reduce {
  %param0 = f32[8,8] parameter(0)
  %input = f32[8,8] copy(%param0)
  %init = f32[] parameter(1)
  %reduce = f32[8] reduce(%input, %init), dimensions={0}, to_apply=%add,
    sharding={devices=[2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="a"}}
  ROOT %copy = f32[8] copy(%reduce)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "input");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ReduceBackwardWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
%add {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}
ENTRY %reduce {
  %param0 = f32[8,8] parameter(0)
  %input = f32[8,8] copy(%param0)
  %init = f32[] parameter(1)
  %shard-barrier-to = f32[8,8] custom-call(%input), custom_call_target="ShardBarrierTo", custom_call_has_side_effect=true
  %reduce = f32[8] reduce(%shard-barrier-to, %init), dimensions={0}, to_apply=%add,
    sharding={devices=[2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="a"}}
  ROOT %copy = f32[8] copy(%reduce)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "input");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTestWithOutput,
       ShardedOnNonReduceDimTupleReduceForwardAndBackwardPass) {
  const char* const hlo_string = R"(
HloModule module

%minmax_func {
  %lhs_value = f32[] parameter(0)
  %rhs_value = f32[] parameter(2)
  %compare.2 = pred[] compare(%lhs_value, %rhs_value), direction=GT
  %select.4 = f32[] select(%compare.2, %lhs_value, %rhs_value)
  %lhs_index = s32[] parameter(1)
  %rhs_index = s32[] parameter(3)
  %select.5 = s32[] select(%compare.2, %lhs_index, %rhs_index)
  ROOT %tuple.2 = (f32[], s32[]) tuple(%select.4, %select.5)
}

ENTRY %main {
  %param0 = f32[28,10] parameter(0)
  %param1 = s32[28,10] parameter(1), sharding={devices=[2,1]0,1 metadata={op_name="a"}}
  %copy_param0 = f32[28,10] copy(%param0)
  %init0 = f32[] parameter(2)
  %init1 = s32[] parameter(3)
  %reduce = (f32[28], s32[28]) reduce(%copy_param0, %param1, %init0, %init1),
    dimensions={1}, to_apply=%minmax_func
  %gte0 = f32[28] get-tuple-element(%reduce), index=0
  %gte1 = s32[28] get-tuple-element(%reduce), index=1
  %copy0 = f32[28] copy(%gte0)
  %copy1 = s32[28] copy(%gte1)
  ROOT %tuple = (f32[28], s32[28]) tuple(%copy0, %copy1)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata,
                          {GetParam().allow_root_sharding_propagation})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* reduce = FindInstruction(module.get(), "reduce");
  ASSERT_NE(reduce, nullptr);
  EXPECT_THAT(reduce, op::Sharding("{{devices=[2]0,1},{devices=[2]0,1}}"));
  auto* copy_param0 = FindInstruction(module.get(), "copy_param0");
  ASSERT_NE(copy_param0, nullptr);
  EXPECT_THAT(copy_param0, op::Sharding("{devices=[2,1]0,1}"));
  for (const HloSharding& sharding :
       {copy_param0->sharding(), reduce->sharding().tuple_elements()[0],
        reduce->sharding().tuple_elements()[1]}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(sharding, ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(sharding, ShardingMetadata({}));
    }
  }
  if (GetParam().allow_root_sharding_propagation) {
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                op::Sharding("{{devices=[2]0,1},{devices=[2]0,1}}"));
  }
}

TEST_P(ParameterizedMetadataTestWithOutput,
       ShardedOnReduceDimTupleReduceForwardAndBackwardPass) {
  const char* const hlo_string = R"(
HloModule module

%minmax_func {
  %lhs_value = f32[] parameter(0)
  %rhs_value = f32[] parameter(2)
  %compare.2 = pred[] compare(%lhs_value, %rhs_value), direction=GT
  %select.4 = f32[] select(%compare.2, %lhs_value, %rhs_value)
  %lhs_index = s32[] parameter(1)
  %rhs_index = s32[] parameter(3)
  %select.5 = s32[] select(%compare.2, %lhs_index, %rhs_index)
  ROOT %tuple.2 = (f32[], s32[]) tuple(%select.4, %select.5)
}

ENTRY %main {
  %param0 = f32[28,10] parameter(0)
  %param1 = s32[28,10] parameter(1), sharding={devices=[2,2]0,1,2,3 metadata={op_name="a"}}
  %copy_param0 = f32[28,10] copy(%param0)
  %init0 = f32[] parameter(2)
  %init1 = s32[] parameter(3)
  %reduce = (f32[28], s32[28]) reduce(%copy_param0, %param1, %init0, %init1),
    dimensions={1}, to_apply=%minmax_func
  %gte0 = f32[28] get-tuple-element(%reduce), index=0
  %gte1 = s32[28] get-tuple-element(%reduce), index=1
  %copy0 = f32[28] copy(%gte0)
  %copy1 = s32[28] copy(%gte1)
  ROOT %tuple = (f32[28], s32[28]) tuple(%copy0, %copy1)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata,
                          {GetParam().allow_root_sharding_propagation})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* reduce = FindInstruction(module.get(), "reduce");
  ASSERT_NE(reduce, nullptr);
  EXPECT_THAT(reduce, op::Sharding("{{devices=[2,2]0,1,2,3 "
                                   "last_tile_dim_replicate},{devices=[2,2]0,1,"
                                   "2,3 last_tile_dim_replicate}}"));
  auto* copy_param0 = FindInstruction(module.get(), "copy_param0");
  ASSERT_NE(copy_param0, nullptr);
  EXPECT_THAT(copy_param0, op::Sharding("{devices=[2,2]0,1,2,3}"));
  for (const HloSharding& sharding :
       {copy_param0->sharding(), reduce->sharding().tuple_elements()[0],
        reduce->sharding().tuple_elements()[1]}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(sharding, ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(sharding, ShardingMetadata({}));
    }
  }
  if (GetParam().allow_root_sharding_propagation) {
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                op::Sharding("{{devices=[2,2]0,1,2,3 "
                             "last_tile_dim_replicate},{devices=[2,2]0,1,2,3 "
                             "last_tile_dim_replicate}}"));
  }
}

TEST_P(ParameterizedMetadataTestWithOutput, GetTupleElementForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %gte {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0)
  %tuple = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) tuple(
    %param0, %param0)
  %tuple.1 = (f32[5,7,11,13]{3,2,1,0},
              (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0})) tuple(
    %param0, %tuple),
    sharding={{devices=[1,2,2,1]0,1,2,3 metadata={op_name="a"}},
              {replicated metadata={op_name="b"}},
              {devices=[1,2,2,1]0,1,2,3 metadata={op_name="c"}}}
  %gte = f32[5,7,11,13]{3,2,1,0} get-tuple-element(%tuple.1), index=0
  %gte.1 = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) get-tuple-element(
    %tuple.1), index=1
  %gte.2 = f32[5,7,11,13]{3,2,1,0} get-tuple-element(%gte.1), index=0
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%gte.2)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata,
                          {GetParam().allow_root_sharding_propagation})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* gte = FindInstruction(module.get(), "gte");
  ASSERT_NE(gte, nullptr);
  EXPECT_THAT(gte, op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  auto* gte1 = FindInstruction(module.get(), "gte.1");
  ASSERT_NE(gte1, nullptr);
  EXPECT_THAT(gte1, op::Sharding("{{replicated}, {devices=[1,2,2,1]0,1,2,3}}"));
  auto* gte2 = FindInstruction(module.get(), "gte.2");
  ASSERT_NE(gte2, nullptr);
  EXPECT_THAT(gte2, op::Sharding("{replicated}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(gte->sharding(), ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(gte1->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("b")}));
    EXPECT_THAT(gte1->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("c")}));
    EXPECT_THAT(gte2->sharding(), ShardingMetadata({CreateMetadata("b")}));
  } else {
    for (const HloSharding& sharding :
         {gte->sharding(), gte1->sharding().tuple_elements()[0],
          gte1->sharding().tuple_elements()[1], gte2->sharding()}) {
      EXPECT_THAT(sharding, ShardingMetadata({}));
    }
  }
  if (GetParam().allow_root_sharding_propagation) {
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                op::Sharding("{replicated}"));
  }
}

TEST_P(ParameterizedMetadataTestWithOutput,
       GetTupleElementForwardPassWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %gte {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0)
  %tuple = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) tuple(
    %param0, %param0), sharding={{devices=[1,2,2,1]0,1,2,3 metadata={op_name="a"}},
    {replicated metadata={op_name="b"}}}
  %shard-barrier-from = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) custom-call(%tuple), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %gte = f32[5,7,11,13]{3,2,1,0} get-tuple-element(%shard-barrier-from), index=0
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%gte)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata,
                          {GetParam().allow_root_sharding_propagation})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* gte = FindInstruction(module.get(), "gte");
  ASSERT_NE(gte, nullptr);
  EXPECT_FALSE(gte->has_sharding());
}

TEST_P(ParameterizedMetadataTest, TupleForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %tuple {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1),
    sharding={devices=[1,2,2,1]0,1,2,3 metadata={op_name="b"}}
  %param2 = f32[5,7,11,13]{3,2,1,0} parameter(2)
  %tuple = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) tuple(
    %param1, %param2)
  %tuple.1 = (f32[5,7,11,13]{3,2,1,0},
              (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0})) tuple(
    %param0, %tuple)
  ROOT %copy = (f32[5,7,11,13]{3,2,1,0},
                (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0})) copy(
    %tuple.1)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* tuple = FindInstruction(module.get(), "tuple");
  ASSERT_NE(tuple, nullptr);
  EXPECT_THAT(tuple, op::Sharding("{{devices=[1,2,2,1]0,1,2,3},"
                                  " {replicated}}"));
  auto* tuple1 = FindInstruction(module.get(), "tuple.1");
  ASSERT_NE(tuple1, nullptr);
  EXPECT_THAT(tuple1, op::Sharding("{{replicated},"
                                   " {devices=[1,2,2,1]0,1,2,3},"
                                   " {replicated}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(tuple->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("b")}));
    EXPECT_THAT(tuple->sharding().tuple_elements()[1], ShardingMetadata({}));
    EXPECT_THAT(tuple1->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(tuple1->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("b")}));
    EXPECT_THAT(tuple1->sharding().tuple_elements()[2], ShardingMetadata({}));
  } else {
    for (const HloSharding& tuple_sharding :
         {tuple->sharding(), tuple1->sharding()}) {
      for (const HloSharding& sub_sharding : tuple_sharding.tuple_elements()) {
        EXPECT_THAT(sub_sharding, ShardingMetadata({}));
      }
    }
  }
}

TEST_P(ParameterizedMetadataTest, TupleForwardPass_SplatBug) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %tuple {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1),
    sharding={devices=[1,2,2,1,2]0,1,2,3,4,5,6,7  last_tile_dims={manual} metadata={op_name="b"}}
  %param2 = f32[5,7,11,13]{3,2,1,0} parameter(2)
  %tuple = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) tuple(
    %param1, %param2)
  ROOT %copy = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) copy(%tuple)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* tuple = FindInstruction(module.get(), "tuple");
  ASSERT_NE(tuple, nullptr);
  // Check that the sharding on param1 is not replicated on tuple element[1].
  EXPECT_THAT(tuple, op::Sharding("{{devices=[1,2,2,1,2]0,1,2,3,4,5,6,7 "
                                  "last_tile_dims={manual}}, {replicated}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(tuple->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("b")}));
    EXPECT_THAT(tuple->sharding().tuple_elements()[1], ShardingMetadata({}));
  } else {
    for (const HloSharding& sub_sharding : tuple->sharding().tuple_elements()) {
      EXPECT_THAT(sub_sharding, ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, TupleForwardPassAndBackWardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %tuple {
  %param0 =  f32[256,2]{1,0} parameter(0),
    sharding={manual metadata={op_name="a"}}
  %param1 =  f32[256,2]{1,0} parameter(1),
    sharding={devices=[1,2]0,1 metadata={op_name="b"}}
  %constant = s32[1,2]{1,0} constant({{0,1}})
  %gather = f32[1,32,2]{2,1,0} gather(param0, constant), offset_dims={1,2}, collapsed_slice_dims={}, start_index_map={0,1}, index_vector_dim=1, slice_sizes={32,2}
  %tuple = (f32[1,32,2]{2,1,0}, f32[256,2]{1,0}) tuple(
    %gather, %param1)
  ROOT %copy = (f32[1,32,2]{2,1,0}, f32[256,2]{1,0}) copy(%tuple)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* tuple = FindInstruction(module.get(), "tuple");
  ASSERT_NE(tuple, nullptr);
  // Check that the sharding on param1 is not replicated on tuple element[1].
  EXPECT_THAT(tuple, op::Sharding("{{manual}, {devices=[1,2]0,1}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(tuple->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(tuple->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    for (const HloSharding& sub_sharding : tuple->sharding().tuple_elements()) {
      EXPECT_THAT(sub_sharding, ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, TupleShapedBackWardPass) {
  const char* const hlo_string = R"(
HloModule module

%cond {
  %vars.cond = (u32[], f32[]) parameter(0)
  %count.cond = u32[] get-tuple-element(%vars.cond), index=0
  %limit = u32[] constant(10)
  ROOT %lt = pred[] compare(%count.cond, %limit), direction=LT
}

%body {
  %param = (u32[], f32[]) parameter(0)
  %count = u32[] get-tuple-element(%param), index=0
  %after-all = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(%after-all), channel_id=1
  %recv-done = (f32[], token[]) recv-done(%recv), channel_id=1
  %data = f32[] get-tuple-element(%recv-done), index=0
  ROOT %tuple = (u32[], f32[]) tuple(%count, %data)
}

ENTRY %entry {
  %zero = u32[] constant(0), sharding={replicated metadata={op_name="a"}}
  %p0 = f32[] parameter(0), sharding={manual metadata={op_name="b"}}
  %tuple = (u32[], f32[]) tuple(%zero, %p0)
  %while = (u32[], f32[]) while(%tuple), body=%body, condition=%cond,
    sharding={{manual metadata={op_name="c"}},
              {manual metadata={op_name="d"}}}
  ROOT %result = f32[] get-tuple-element(%while), index=1
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* tuple = FindInstruction(module.get(), "tuple");
  ASSERT_NE(tuple, nullptr);
  // Check that the sharding on param1 is not replicated on tuple element[1].
  EXPECT_THAT(tuple, op::Sharding("{{manual}, {manual}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(tuple->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("c")}));
    EXPECT_THAT(tuple->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("d")}));
  } else {
    for (const HloSharding& sub_sharding : tuple->sharding().tuple_elements()) {
      EXPECT_THAT(sub_sharding, ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       PartiallyManualTupleWithRepeatedOperandsBackWardPass) {
  const char* const hlo_string = R"(
HloModule module

%cond {
  %vars.cond = (s32[], s32[], s32[]) parameter(0)
  %count.cond = s32[] get-tuple-element(%vars.cond), index=0
  %limit = s32[] constant(10)
  ROOT %lt = pred[] compare(%count.cond, %limit), direction=LT
}

%body {
  %param = (s32[], s32[], s32[]) parameter(0)
  %count = s32[] get-tuple-element(%param), index=0
  %lhs = s32[] get-tuple-element(%param), index=1
  %rhs = s32[] get-tuple-element(%param), index=2
  %add = s32[] add(%lhs, %rhs)
  ROOT %tuple = (s32[], s32[], s32[]) tuple(%count, %lhs, %add)
}

ENTRY %entry {
  %zero = s32[] constant(0)
  %p0 = s32[] parameter(0), sharding={manual metadata={op_name="a"}}
  %tuple = (s32[], s32[], s32[]) tuple(%zero, %zero, %p0)
  %while = (s32[], s32[], s32[]) while(%tuple), body=%body, condition=%cond
  ROOT %copy = (s32[], s32[], s32[]) copy(%while)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* tuple = module->entry_computation()->root_instruction()->operand(0);
  ASSERT_NE(tuple, nullptr);
  // Check that the sharding on param1 is not replicated on tuple element[1].
  EXPECT_THAT(tuple, op::Sharding("{{manual}, {manual}, {manual}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(tuple->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(tuple->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(tuple->sharding().tuple_elements()[2],
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    for (const HloSharding& sub_sharding : tuple->sharding().tuple_elements()) {
      EXPECT_THAT(sub_sharding, ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, ForwardConvolutionForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %lhs = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={devices=[2,2,2,1]0,1,2,3,4,5,6,7 metadata={op_name="a"}}
  %rhs = f32[3,3,13,17]{3,2,1,0} parameter(1)
  %convolution = f32[5,7,11,17]{3,2,1,0} convolution(%lhs, %rhs),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  ROOT %copy = f32[5,7,11,17]{3,2,1,0} copy(%convolution)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "convolution");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,2,2,1]0,1,2,3,4,5,6,7}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ForwardConvolutionLargeDilationForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %lhs = f32[8,64,2]{2,1,0} parameter(0),
    sharding={devices=[1,4,1]0,1,2,3 metadata={op_name="a"}}
  %rhs = f32[3,2,2]{2,1,0} parameter(1)
  %convolution = f32[8,32,2]{2,1,0} convolution(%lhs, %rhs),
    window={size=3 rhs_dilate=16}, dim_labels=b0f_0io->b0f
  ROOT %copy = f32[8,32,2]{2,1,0} copy(%convolution)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "convolution");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,4,1]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ForwardConvolution3DSmallKernel) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %lhs = bf16[32,32,8,7,128]{4,3,2,1,0} parameter(0),
    sharding={devices=[1,4,1,1,1]0,1,2,3 metadata={op_name="a"}}
  %rhs = bf16[3,3,3,128,256]{4,3,2,1,0} parameter(1)
  %convolution = bf16[16,16,8,3,256]{4,3,2,1,0}
    convolution(bf16[32,32,8,7,128]{4,3,2,1,0} %lhs,
    bf16[3,3,3,128,256]{4,3,2,1,0} %rhs),
    window={size=3x3x3 stride=2x2x2 pad=1_1x1_1x0_0},
    dim_labels=01b2f_012io->01b2f
  ROOT %copy = bf16[16,16,8,3,256]{4,3,2,1,0} copy(%convolution)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "convolution");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,4,1,1,1]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, TransposeForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %transpose {
  %param = f32[7,11,13]{2,1,0} parameter(0),
    sharding={devices=[2,1,2]0,1,2,3 metadata={op_name="a"}}
  %transpose = f32[11,13,7]{2,1,0} transpose(%param), dimensions={1,2,0}
  ROOT %copy = f32[11,13,7]{2,1,0} copy(%transpose)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "transpose");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,2]0,2,1,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, TransposeForwardPassWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %transpose {
  %param = f32[7,11,13]{2,1,0} parameter(0),
    sharding={devices=[2,1,2]0,1,2,3 metadata={op_name="a"}}
  %shard-barrier-from = f32[7,11,13]{2,1,0} custom-call(%param), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %transpose = f32[11,13,7]{2,1,0} transpose(%shard-barrier-from), dimensions={1,2,0}
  ROOT %copy = f32[11,13,7]{2,1,0} copy(%transpose)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "transpose");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTest, TransposeBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %transpose {
  %param = f32[7,11,13]{2,1,0} parameter(0)
  %copy = f32[7,11,13]{2,1,0} copy(%param)
  ROOT %transpose = f32[11,13,7]{2,1,0} transpose(%copy), dimensions={1,2,0},
    sharding={devices=[1,2,2]0,1,2,3 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "copy");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,1,2]0,2,1,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, TransposeBackwardPassWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %transpose {
  %param = f32[7,11,13]{2,1,0} parameter(0)
  %copy = f32[7,11,13]{2,1,0} copy(%param)
  %shard-barrier-to = f32[7,11,13]{2,1,0} custom-call(%copy), custom_call_target="ShardBarrierTo", custom_call_has_side_effect=true
  ROOT %transpose = f32[11,13,7]{2,1,0} transpose(%shard-barrier-to), dimensions={1,2,0},
    sharding={devices=[1,2,2]0,1,2,3 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "copy");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTest, ReshapeForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[1430,1]{1,0} parameter(0),
    sharding={devices=[2,1]0,1 metadata={op_name="a"}}
  %reshape = f32[10,11,13]{2,1,0} reshape(%param0)
  ROOT %copy = f32[10,11,13]{2,1,0} copy(%reshape)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "reshape");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,1,1]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ReshapeForwardPassWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[1430,1]{1,0} parameter(0),
    sharding={devices=[2,1]0,1 metadata={op_name="a"}}
  %shard-barrier-from = f32[1430,1]{1,0} custom-call(%param0), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %reshape = f32[10,11,13]{2,1,0} reshape(%shard-barrier-from)
  ROOT %copy = f32[10,11,13]{2,1,0} copy(%reshape)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "reshape");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTest, ReshapeForwardPassPartialMatch) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[14,32] parameter(0),
    sharding={devices=[4,4]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 metadata={op_name="a"}}
  %reshape = f32[7,2,2,16] reshape(%param0)
  ROOT %copy = f32[7,2,2,16] copy(%reshape)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "reshape");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,1,2,2,4]0,4,8,12,1,5,9,13,2,6,10,14,3,"
                           "7,11,15 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ReshapeForwardPassPartialMatch2) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[12,8] parameter(0),
    sharding={devices=[2,4]0,1,2,3,4,5,6,7 metadata={op_name="a"}}
  %reshape = f32[8,12] reshape(%param0)
  ROOT %copy = f32[8,12] copy(%reshape)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "reshape");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding("{devices=[2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ReshapeForwardPassTranspose) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[6,4,5] parameter(0), sharding={devices=[6,2,1]<=[12] metadata={op_name="a"}}
  %reshape.1 = f32[2,3,20] reshape(%param0)
  %reshape.2 = f32[2,4,3,5] reshape(%param0)
  %reshape.3 = f32[20,6] reshape(%param0)
  %reshape.4 = f32[3,5,8] reshape(%param0)
  %reshape.5 = f32[10,4,3] reshape(%param0)
  %reshape.6 = f32[5,8,3] reshape(%param0)
  ROOT %tuple = tuple(%reshape.1, %reshape.2, %reshape.3, %reshape.4, %reshape.5, %reshape.6)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);

  std::vector<std::pair<std::string, std::string>> instruction_and_sharding = {
      {"reshape.1", "{devices=[2,3,2]<=[12]}"},
      {"reshape.2", "{devices=[2,1,1,1,6]<=[12] last_tile_dim_replicate}"},
      {"reshape.3", "{devices=[2,1,6]<=[12] last_tile_dim_replicate}"},
      {"reshape.4", "{devices=[3,1,1,4]<=[12] last_tile_dim_replicate}"},
      {"reshape.5", "{devices=[2,1,1,6]<=[12] last_tile_dim_replicate}"},
      {"reshape.6", "{replicated}"}};
  for (const auto& [name, sharding] : instruction_and_sharding) {
    auto* instruction = FindInstruction(module.get(), name);
    ASSERT_NE(instruction, nullptr);
    EXPECT_THAT(instruction, op::Sharding(sharding));
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, ReshapeBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[2002,1]{1,0} parameter(0)
  %copy = f32[2002,1]{1,0} copy(f32[2002,1]{1,0} %param0)
  ROOT %reshape = f32[14,11,13]{2,1,0} reshape(%copy),
    sharding={devices=[2,1,1]0,1 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "copy");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,1]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ReshapeBackwardPassWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[2002,1]{1,0} parameter(0)
  %copy = f32[2002,1]{1,0} copy(f32[2002,1]{1,0} %param0)
  %shard-barrier-to = f32[2002,1]{1,0} custom-call(%copy), custom_call_target="ShardBarrierTo", custom_call_has_side_effect=true
  ROOT %reshape = f32[14,11,13]{2,1,0} reshape(%shard-barrier-to),
    sharding={devices=[2,1,1]0,1 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "copy");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTest, PadForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %pad {
  %input = f32[11,17]{1,0} parameter(0),
    sharding={devices=[2,2]0,1,2,3 metadata={op_name="a"}}
  %pad_value = f32[] parameter(1)
  %pad = f32[27,51]{1,0} pad(%input, %pad_value), padding=2_4_1x1_1_2
  ROOT %copy = f32[27,51]{1,0} copy(%pad)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "pad");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,2]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, PadBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %pad {
  %input = f32[11,17]{1,0} parameter(0)
  %copy = f32[11,17]{1,0} copy(%input)
  %pad_value = f32[] parameter(1)
  %pad = f32[27,51]{1,0} pad(%copy, %pad_value), padding=2_4_1x1_1_2,
    sharding={devices=[2,2]0,1,2,3 metadata={op_name="a"}}
  ROOT %result = f32[27,51]{1,0} copy(%pad)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "copy");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,2]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, PartialReplicatedPadForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %pad {
  %input = f32[11,17]{1,0} parameter(0),
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate metadata={op_name="a"}}
  %pad_value = f32[] parameter(1)
  %pad = f32[27,51]{1,0} pad(%input, %pad_value), padding=2_4_1x1_1_2
  ROOT %copy = f32[27,51]{1,0} copy(%pad)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "pad");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding("{devices=[2,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ShardedPreferredOverReplicated) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %replicated {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %copy = f32[5,7,11,13]{3,2,1,0} copy(%param0)
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1),
    sharding={devices=[1,2,2,1]0,1,2,3 metadata={op_name="b"}}
  %copy.1 = f32[5,7,11,13]{3,2,1,0} copy(%param1)
  %add = f32[5,7,11,13]{3,2,1,0} add(%copy, %copy.1)
  ROOT %copy.2 = f32[5,7,11,13]{3,2,1,0} copy(%add)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* copy = FindInstruction(module.get(), "copy");
  ASSERT_NE(copy, nullptr);
  EXPECT_THAT(copy, op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  auto* copy1 = FindInstruction(module.get(), "copy.1");
  ASSERT_NE(copy1, nullptr);
  EXPECT_THAT(copy1, op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  auto* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);
  EXPECT_THAT(add, op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  for (const HloSharding& sharding :
       {copy->sharding(), copy1->sharding(), add->sharding()}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(sharding, ShardingMetadata({CreateMetadata("b")}));
    } else {
      EXPECT_THAT(sharding, ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, PartialReplicateReshapeForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[1430,1]{1,0} parameter(0),
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="a"}}
  %reshape = f32[10,11,13]{2,1,0} reshape(%param0)
  ROOT %copy = f32[10,11,13]{2,1,0} copy(%reshape)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "reshape");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding("{devices=[2,1,1,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, PartialReplicateReshapeBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[2002,1]{1,0} parameter(0)
  %copy = f32[2002,1]{1,0} copy(f32[2002,1]{1,0} %param0)
  ROOT %reshape = f32[14,11,13]{2,1,0} reshape(%copy),
    sharding={devices=[2,1,1,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "copy");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, DontShardTuplesIfAllInputIsMaximal) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %tuple {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={maximal device=0 metadata={op_name="a"}}
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1),
    sharding={maximal device=1 metadata={op_name="b"}}
  %tuple = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) tuple(
    %param0, %param1)
  ROOT %copy = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) copy(%tuple)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  EXPECT_EQ(changed,
            !GetParam().propagate_metadata && !GetParam().clear_metadata);
  auto* instruction = FindInstruction(module.get(), "tuple");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::NoSharding());
}

TEST_P(ParameterizedMetadataTest, ValidConvolution) {
  const char* const hlo_string = R"(
HloModule module

ENTRY conv {
  %lhs = f32[13,17,19]{2,1,0} parameter(0),
    sharding={devices=[1,2,1]0,1 metadata={op_name="a"}}
  %rhs = f32[19,5,19]{2,1,0} parameter(1)
  %conv = f32[13,13,19]{2,1,0} convolution(%lhs, %rhs),
    window={size=5}, dim_labels=b0f_i0o->b0f
  ROOT %tuple = (f32[13,13,19]{2,1,0}) tuple(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "conv");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,1]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, StridedSlice) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %slice {
  %param = f32[17,13]{1,0} parameter(0),
    sharding={devices=[2,1]0,1 metadata={op_name="a"}}
  %slice = f32[7,5]{1,0} slice(%param), slice={[1:15:2], [5:10:1]}
  ROOT %tuple = (f32[7,5]{1,0}) tuple(%slice)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "slice");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,1]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, PartialReplicatedStridedSlice) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %slice {
  %param = f32[17,13]{1,0} parameter(0),
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="a"}}
  %slice = f32[7,5]{1,0} slice(%param), slice={[1:15:2], [5:10:1]}
  ROOT %tuple = (f32[7,5]{1,0}) tuple(%slice)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "slice");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ReduceWindowBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
%add (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}
ENTRY %reduce_window {
  %param = f32[13,17]{1,0} parameter(0)
  %param.copy = f32[13,17]{1,0} copy(%param)
  %init = f32[] parameter(1)
  ROOT %reduce-window = f32[7,17]{1,0} reduce-window(%param.copy, %init),
    window={size=3x2 stride=2x1 pad=1_1x0_1}, to_apply=%add,
    sharding={devices=[2,1]0,1 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* param_copy = FindInstruction(module.get(), "param.copy");
  ASSERT_NE(param_copy, nullptr);
  EXPECT_THAT(param_copy, op::Sharding("{devices=[2,1]0,1}"));
  auto* reduce_window = FindInstruction(module.get(), "reduce-window");
  ASSERT_NE(reduce_window, nullptr);
  EXPECT_THAT(reduce_window, op::Sharding("{devices=[2,1]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(param_copy->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(reduce_window->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(param_copy->sharding(), ShardingMetadata({}));
    EXPECT_THAT(reduce_window->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ReduceWindowBackwardPassWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
%add (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}
ENTRY %reduce_window {
  %param = f32[13,17]{1,0} parameter(0)
  %param.copy = f32[13,17]{1,0} copy(%param)
  %init = f32[] parameter(1)
  %shard-barrier-to = f32[13,17]{1,0} custom-call(%param.copy), custom_call_target="ShardBarrierTo", custom_call_has_side_effect=true
  ROOT %reduce-window = f32[7,17]{1,0} reduce-window(%shard-barrier-to, %init),
    window={size=3x2 stride=2x1 pad=1_1x0_1}, to_apply=%add,
    sharding={devices=[2,1]0,1 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* param_copy = FindInstruction(module.get(), "param.copy");
  ASSERT_NE(param_copy, nullptr);
  EXPECT_FALSE(param_copy->has_sharding());
}

TEST_P(ParameterizedMetadataTest, VariadicReduceWindowBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
%add (a: f32[], b: s32[], c: f32[], d: s32[]) -> (f32[], s32[]) {
  %a = f32[] parameter(0)
  %b = s32[] parameter(1)
  %c = f32[] parameter(2)
  %d = s32[] parameter(3)
  %add.0 = f32[] add(%a, %c)
  %add.1 = s32[] add(%b, %d)
  ROOT %t = tuple(%add.0, %add.1)
}
ENTRY %reduce_window {
  %param.0 = f32[13,17]{1,0} parameter(0)
  %param.0.copy = f32[13,17]{1,0} copy(%param.0)
  %param.1 = s32[13,17]{1,0} parameter(1)
  %param.1.copy = s32[13,17]{1,0} copy(%param.1)
  %init.0 = f32[] parameter(2)
  %init.1 = s32[] parameter(3)
  ROOT %reduce-window = (f32[7,17]{1,0}, s32[7,17]{1,0}) reduce-window(%param.0.copy, %param.1.copy, %init.0, %init.1),
    window={size=3x2 stride=2x1 pad=1_1x0_1}, to_apply=%add,
    sharding={{devices=[2,1]0,1 metadata={op_name="a"}}, {devices=[2,1]0,1 metadata={op_name="b"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* param_0_copy = FindInstruction(module.get(), "param.0.copy");
  ASSERT_NE(param_0_copy, nullptr);
  EXPECT_THAT(param_0_copy, op::Sharding("{devices=[2,1]0,1}"));
  auto* param_1_copy = FindInstruction(module.get(), "param.1.copy");
  ASSERT_NE(param_1_copy, nullptr);
  EXPECT_THAT(param_1_copy, op::Sharding("{devices=[2,1]0,1}"));
  auto* reduce_window = FindInstruction(module.get(), "reduce-window");
  ASSERT_NE(reduce_window, nullptr);
  EXPECT_THAT(reduce_window,
              op::Sharding("{{devices=[2,1]0,1}, {devices=[2,1]0,1}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(param_0_copy->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(param_1_copy->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
    EXPECT_THAT(reduce_window->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(reduce_window->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(param_0_copy->sharding(), ShardingMetadata({}));
    EXPECT_THAT(param_1_copy->sharding(), ShardingMetadata({}));
    EXPECT_THAT(reduce_window->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ReplicatedConvolutionLhs) {
  const char* const hlo_string = R"(
HloModule module

ENTRY conv {
  %lhs = f32[3,2,3]{2,1,0} parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %rhs = f32[2,2,1]{2,1,0} parameter(1)
  %conv = f32[3,2,3]{2,1,0} convolution(%lhs, %rhs),
    window={size=1}, dim_labels=bf0_oi0->bf0
  ROOT %tuple = (f32[3,2,3]{2,1,0}) tuple(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* lhs = FindInstruction(module.get(), "lhs");
  ASSERT_NE(lhs, nullptr);
  EXPECT_THAT(lhs, op::Sharding("{replicated}"));
  auto* conv = FindInstruction(module.get(), "conv");
  ASSERT_NE(conv, nullptr);
  EXPECT_THAT(conv, op::Sharding("{replicated}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(lhs->sharding(), ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(conv->sharding(), ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(lhs->sharding(), ShardingMetadata({}));
    EXPECT_THAT(conv->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ConvolutionShardedFeature) {
  const char* const hlo_string = R"(
HloModule module

ENTRY conv {
  %lhs = f32[3,2,3]{2,1,0} parameter(0),
    sharding={devices=[1,2,1]0,1 metadata={op_name="a"}}
  %rhs = f32[2,2,1]{2,1,0} parameter(1)
  %conv = f32[3,2,3]{2,1,0} convolution(%lhs, %rhs),
    window={size=1}, dim_labels=bf0_oi0->bf0
  ROOT %tuple = (f32[3,2,3]{2,1,0}) tuple(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "conv");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{replicated}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ConvolutionDifferentDimensionNumbers) {
  const char* const hlo_string = R"(
HloModule module

ENTRY conv {
  %lhs = f32[8,16,512] parameter(0),
    sharding={devices=[1,2,1]0,1 metadata={op_name="a"}}
  %rhs = f32[8,2,512] parameter(1)
  %conv = f32[3,512,512] convolution(%lhs, %rhs),
    window={size=2 stride=5},
    dim_labels=f0b_i0o->0bf
  ROOT %tuple = (f32[3,512,512]) tuple(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "conv");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,1,1]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, Concatenate) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %concat {
  %param.0 = f32[5,7] parameter(0),
    sharding={devices=[2,1]0,1 metadata={op_name="a"}}
  %param.1 = f32[5,9] parameter(1),
    sharding={devices=[2,1]0,1 metadata={op_name="b"}}
  %concat = f32[5,16] concatenate(%param.0, %param.1),
    dimensions={1}
  ROOT %tuple = (f32[5,16]) tuple(%concat)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "concat");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,1]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ConcatenateForwardWithBarrier) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %concat {
  %param.0 = f32[5,7] parameter(0),
    sharding={devices=[2,1]0,1 metadata={op_name="a"}}
  %param.1 = f32[5,9] parameter(1),
    sharding={devices=[2,1]0,1 metadata={op_name="b"}}
  %shard-barrier-from.0 = f32[5,7] custom-call(%param.0), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %shard-barrier-from.1 = f32[5,9] custom-call(%param.1), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %concat = f32[5,16] concatenate(%shard-barrier-from.0, %shard-barrier-from.1),
    dimensions={1}
  ROOT %tuple = (f32[5,16]) tuple(%concat)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "concat");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTest, ConcatenateBackwardWithBarrier) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %concat {
  %param.0 = f32[5,7] parameter(0)
  %copy.0 = f32[5,7] copy(%param.0)
  %param.1 = f32[5,9] parameter(1)
  %copy.1 = f32[5,9] copy(%param.1)
  %shard-barrier-to = f32[5,9] custom-call(%copy.1), custom_call_target="ShardBarrierTo", custom_call_has_side_effect=true
  %concat = f32[5,16] concatenate(%copy.0, %shard-barrier-to),
    dimensions={1}, sharding={devices=[2,1]0,1 metadata={op_name="a"}}
  ROOT %tuple = (f32[5,16]) tuple(%concat)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "copy.1");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTest, TupleBackwardPass) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %tuple {
  %param.0 = f32[1] parameter(0)
  %param.1 = f32[3] parameter(1)
  %copy.0 = f32[1] copy(%param.0)
  %copy.1 = f32[3] copy(%param.1)
  ROOT %tuple = (f32[1], f32[3]) tuple(%copy.0, %copy.1),
    sharding={{replicated metadata={op_name="a"}},
              {devices=[2]0,1 metadata={op_name="b"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* copy0 = FindInstruction(module.get(), "copy.0");
  ASSERT_NE(copy0, nullptr);
  EXPECT_THAT(copy0, op::Sharding("{replicated}"));
  auto* copy1 = FindInstruction(module.get(), "copy.1");
  ASSERT_NE(copy1, nullptr);
  EXPECT_THAT(copy1, op::Sharding("{devices=[2]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(copy0->sharding(), ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(copy1->sharding(), ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(copy0->sharding(), ShardingMetadata({}));
    EXPECT_THAT(copy1->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, AllReduce) {
  const char* const hlo_string = R"(
HloModule module

%add (lhs: f32[], rhs: f32[]) -> f32[] {
  %add_lhs = f32[] parameter(0)
  %add_rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %add_lhs, f32[] %add_rhs)
}

ENTRY %entry {
  %param.0 = f32[3] parameter(0)
  %param.1 = f32[3] parameter(1)

  %copy_f_t = f32[3] copy(%param.1),
    sharding={devices=[2]0,1 metadata={op_name="a"}}
  %crs_f.tiled = f32[3] all-reduce(%copy_f_t), to_apply=%add
  %crs_f.none = f32[3] all-reduce(%copy_f_t), to_apply=%add,
    channel_id=1

  %crs_b.replicated = f32[3] all-reduce(%param.0), to_apply=%add
  %copy_b_r = f32[3] copy(%crs_b.replicated),
    sharding={replicated metadata={op_name="b"}}

  ROOT %tuple = (f32[3], f32[3], f32[3]) tuple(
    %crs_f.tiled, crs_f.none, %copy_b_r)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* crs_f_tiled = FindInstruction(module.get(), "crs_f.tiled");
  ASSERT_NE(crs_f_tiled, nullptr);
  EXPECT_THAT(crs_f_tiled, op::Sharding("{devices=[2]0,1}"));
  auto* crs_f_none = FindInstruction(module.get(), "crs_f.none");
  ASSERT_NE(crs_f_none, nullptr);
  EXPECT_THAT(crs_f_none, op::Sharding("{devices=[2]0,1}"));
  auto* crs_b_replicated = FindInstruction(module.get(), "crs_b.replicated");
  ASSERT_NE(crs_b_replicated, nullptr);
  EXPECT_THAT(crs_b_replicated, op::Sharding("{replicated}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(crs_f_tiled->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(crs_b_replicated->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(crs_f_tiled->sharding(), ShardingMetadata({}));
    EXPECT_THAT(crs_b_replicated->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, While) {
  const char* const hlo_string = R"(
HloModule module

%cond {
  %vars.cond = (u32[], f32[10,10]) parameter(0)
  %count.cond = u32[] get-tuple-element((u32[], f32[10,10]) %vars.cond), index=0
  %limit = u32[] constant(10)
  ROOT %lt = pred[] compare(u32[] %count.cond, u32[] %limit), direction=LT
}

%body {
  %vars = (u32[], f32[10,10]) parameter(0)
  %count = u32[] get-tuple-element(%vars), index=0
  %acc = f32[10,10] get-tuple-element((u32[], f32[10,10]) %vars), index=1

  %one = u32[] constant(1)
  %count.1 = u32[] add(u32[] %count, u32[] %one), sharding={replicated}
  %acc.1 = f32[10,10] add(f32[10,10] %acc, f32[10,10] %acc)
  ROOT %tuple = (u32[], f32[10,10]) tuple(u32[] %count.1, f32[10,10] %acc.1)
}

ENTRY %entry {
  %p0 = f32[10,10] parameter(0)
  %p0.copy = f32[10,10] copy(f32[10,10] %p0)
  %p1 = f32[10,10] parameter(1)
  %zero = u32[] constant(0)
  %init = (u32[], f32[10,10]) tuple(u32[] %zero, f32[10,10] %p0.copy)
  %while = (u32[], f32[10,10]) while((u32[], f32[10,10]) %init),
    body=%body, condition=%cond
  %res = f32[10,10] get-tuple-element((u32[], f32[10,10]) %while), index=1
  %prev = f32[10,10] get-tuple-element((u32[], f32[10,10]) %init), index=1
  %res.1 = f32[10,10] multiply(f32[10,10] %res, %prev)
  ROOT %res_tuple = (f32[10,10]) tuple(f32[10,10] %res.1)
})";

  auto while_is_sharded =
      [this](HloModule* module, const HloSharding& sharding,
             absl::Span<const absl::Span<const OpMetadata>> sharding_metadata) {
        if (GetParam().clear_metadata) {
          ClearMetadata(module);
        }
        TF_ASSERT_OK_AND_ASSIGN(
            bool changed,
            ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
                .Run(module));
        EXPECT_TRUE(changed);
        auto while_instr = FindInstruction(module, "while");
        EXPECT_NE(nullptr, while_instr);
        std::vector<const HloInstruction*> instructions{
            while_instr, while_instr->while_body()->root_instruction(),
            while_instr->while_body()->parameter_instruction(0),
            while_instr->while_condition()->parameter_instruction(0)};

        for (auto instr : instructions) {
          ASSERT_TRUE(instr->has_sharding());
          EXPECT_EQ(sharding, instr->sharding());
          ASSERT_EQ(instr->sharding().tuple_elements().size(),
                    sharding_metadata.size());
          for (int i = 0, e = sharding_metadata.size(); i < e; ++i) {
            if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
              EXPECT_THAT(instr->sharding().tuple_elements()[i],
                          ShardingMetadata(sharding_metadata[i]));
            } else {
              EXPECT_THAT(instr->sharding().tuple_elements()[i],
                          ShardingMetadata({}));
            }
          }
        }
      };
  {
    // Propagation of user-defined partial sharding of while-related instruction
    // (body root in this test).
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string));
    auto body_root = FindInstruction(module.get(), "tuple");
    EXPECT_NE(nullptr, body_root);
    auto sharding = ParseSharding(
                        "{{replicated metadata={op_name=\"b\"}}, "
                        "{devices=[2,1]0,1 metadata={op_name=\"c\"}}}")
                        .value();
    body_root->set_sharding(sharding);
    while_is_sharded(module.get(), sharding.WithoutMetadata(),
                     {{CreateMetadata("b")}, {CreateMetadata("c")}});
  }
  {
    // Propagation from acc.1 to the rest of the loop.
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string));
    auto acc_1 = FindInstruction(module.get(), "acc.1");
    EXPECT_NE(nullptr, acc_1);
    acc_1->set_sharding(
        ParseSharding("{devices=[2,1]0,1 metadata={op_name=\"b\"}}").value());

    while_is_sharded(
        module.get(),
        ParseSharding("{{replicated}, {devices=[2,1]0,1}}").value(),
        {{}, {CreateMetadata("b")}});
  }
  {
    // Merge partial sharding from operand and body.
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string));
    auto acc_1 = FindInstruction(module.get(), "acc.1");
    EXPECT_NE(nullptr, acc_1);
    acc_1->set_sharding(
        ParseSharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate "
                      "metadata={op_name=\"b\"}}")
            .value());
    auto p0 = FindInstruction(module.get(), "p0");
    p0->set_sharding(
        ParseSharding("{devices=[1,2,2]0,2,1,3 last_tile_dim_replicate "
                      "metadata={op_name=\"c\"}}")
            .value());

    while_is_sharded(module.get(),
                     ParseSharding("{{replicated}, "
                                   "{devices=[2,2]0,1,2,3}}")
                         .value(),
                     {{}, {CreateMetadata("c"), CreateMetadata("b")}});
  }
}

TEST_F(ShardingPropagationTest, PropagateShardingInWhileCondition) {
  const char* const hlo_string = R"(
HloModule module

%cond {
  %vars.cond = (u32[], f32[]) parameter(0)
  %count.cond = u32[] get-tuple-element(%vars.cond), index=0
  %limit = u32[] constant(10)
  ROOT %lt = pred[] compare(%count.cond, %limit), direction=LT
}

%body {
  %vars = (u32[], f32[]) parameter(0)
  %count = u32[] get-tuple-element(%vars), index=0
  %acc = f32[] get-tuple-element(%vars), index=1

  %one = u32[] constant(1)
  %count.1 = u32[] add(u32[] %count, u32[] %one)
  %acc.1 = f32[] add(f32[] %acc, f32[] %acc)
  ROOT %tuple = (u32[], f32[]) tuple(%count.1, %acc.1)
}

ENTRY %entry {
  %p0 = f32[] parameter(0), sharding={devices=[2,2]<=[4] last_tile_dims={manual, replicated}}
  %zero = u32[] constant(0), sharding={devices=[2,2]<=[4] last_tile_dims={manual, replicated}}
  %init = (u32[], f32[]) tuple(%zero, %p0)
  ROOT %while = (u32[], f32[]) while(%init), body=%body, condition=%cond
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, /*propagate_metadata=*/false,
                          /*allow_spmd_sharding_propagation_to_output=*/{true})
          .Run(module.get()));
  EXPECT_TRUE(changed);
  HloSharding single_sharding =
      ParseSharding("{devices=[2,2]<=[4] last_tile_dims={manual, replicated}}")
          .value();
  HloSharding tuple_sharding = HloSharding::SingleTuple(
      module->entry_computation()->root_instruction()->shape(),
      single_sharding);

  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      EXPECT_TRUE(instruction->has_sharding());
      EXPECT_EQ(instruction->sharding(), instruction->shape().IsTuple()
                                             ? tuple_sharding
                                             : single_sharding);
    }
  }
}

TEST_P(ParameterizedMetadataTest, WhileGetShardingFromRecvInBody) {
  const char* const hlo_string = R"(
HloModule module

%cond {
  %vars.cond = (u32[], f32[]) parameter(0)
  %count.cond = u32[] get-tuple-element(%vars.cond), index=0
  %limit = u32[] constant(10)
  ROOT %lt = pred[] compare(%count.cond, %limit), direction=LT
}

%body {
  %param = (u32[], f32[]) parameter(0)
  %count = u32[] get-tuple-element(%param), index=0
  %after-all = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(%after-all), channel_id=1,
    sharding={{maximal device=1 metadata={op_name="a"}},
              {maximal device=1}, {maximal device=1}}
  %recv-done = (f32[], token[]) recv-done(%recv), channel_id=1
  %data = f32[] get-tuple-element(%recv-done), index=0
  ROOT %tuple = (u32[], f32[]) tuple(%count, %data)
}

ENTRY %entry {
  %p0 = f32[] parameter(0)
  %zero = u32[] constant(0)
  %init = (u32[], f32[]) tuple(%zero, %p0)
  %while = (u32[], f32[]) while(%init), body=%body, condition=%cond
  ROOT %result = f32[] get-tuple-element(%while), index=1
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  // The change happens before the fixpt loop
  EXPECT_EQ(changed,
            !GetParam().propagate_metadata && !GetParam().clear_metadata);
  auto sharding =
      ParseSharding("{{maximal device=1}, {maximal device=1}}").value();
  auto while_instr = FindInstruction(module.get(), "while");
  ASSERT_NE(nullptr, while_instr);
  std::vector<const HloInstruction*> instructions{
      while_instr, while_instr->while_body()->root_instruction(),
      while_instr->while_body()->parameter_instruction(0),
      while_instr->while_condition()->parameter_instruction(0)};
  for (auto instr : instructions) {
    ASSERT_TRUE(instr->has_sharding());
    EXPECT_EQ(sharding, instr->sharding());
    for (const HloSharding& sub_sharding : instr->sharding().tuple_elements()) {
      EXPECT_THAT(sub_sharding, ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, WhileConflictingShardingInBodyBeforeRecv) {
  const char* const hlo_string = R"(
HloModule module

%cond {
  %vars.cond = (u32[], f32[]) parameter(0)
  %count.cond = u32[] get-tuple-element(%vars.cond), index=0
  %limit = u32[] constant(10)
  ROOT %lt = pred[] compare(%count.cond, %limit), direction=LT
}

%body {
  %param = (u32[], f32[]) parameter(0)
  %count = u32[] get-tuple-element(%param), index=0,
    sharding={maximal device=0 metadata={op_name="a"}}
  %after-all = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(%after-all), channel_id=1,
    sharding={{maximal device=1 metadata={op_name="b"}},
              {maximal device=1}, {maximal device=1}}
  %recv-done = (f32[], token[]) recv-done(%recv), channel_id=1
  %data = f32[] get-tuple-element(%recv-done), index=0
  ROOT %tuple = (u32[], f32[]) tuple(%count, %data)
}

ENTRY %entry {
  %p0 = f32[] parameter(0)
  %zero = u32[] constant(0)
  %init = (u32[], f32[]) tuple(%zero, %p0)
  %while = (u32[], f32[]) while(%init), body=%body, condition=%cond
  ROOT %result = f32[] get-tuple-element(%while), index=1
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  auto result =
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get());
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr(
                  "Instruction: count is on device: 0, which conflicts with "
                  "device: 1 of channel instruction: recv"));
}

TEST_P(ParameterizedMetadataTest, WhileConflictingShardingInBodyAfterRecv) {
  const char* const hlo_string = R"(
HloModule module

%cond {
  %vars.cond = (u32[], f32[]) parameter(0)
  %count.cond = u32[] get-tuple-element(%vars.cond), index=0
  %limit = u32[] constant(10)
  ROOT %lt = pred[] compare(%count.cond, %limit), direction=LT
}

%body {
  %param = (u32[], f32[]) parameter(0)
  %count = u32[] get-tuple-element(%param), index=0
  %after-all = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(%after-all), channel_id=1,
    sharding={{maximal device=1 metadata={op_name="a"}},
              {maximal device=1}, {maximal device=1}}
  %recv-done = (f32[], token[]) recv-done(%recv), channel_id=1
  %data = f32[] get-tuple-element(%recv-done), index=0,
    sharding={maximal device=0 metadata={op_name="b"}}
  ROOT %tuple = (u32[], f32[]) tuple(%count, %data)
}

ENTRY %entry {
  %p0 = f32[] parameter(0)
  %zero = u32[] constant(0)
  %init = (u32[], f32[]) tuple(%zero, %p0)
  %while = (u32[], f32[]) while(%init), body=%body, condition=%cond
  ROOT %result = f32[] get-tuple-element(%while), index=1
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  auto result =
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get());
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr(
                  "Instruction: data is on device: 0, which conflicts with "
                  "device: 1 of channel instruction: recv"));
}

TEST_P(ParameterizedMetadataTest, WhileConflictingShardingOnWhileInstruction) {
  const char* const hlo_string = R"(
HloModule module

%cond {
  %vars.cond = (u32[], f32[]) parameter(0)
  %count.cond = u32[] get-tuple-element(%vars.cond), index=0
  %limit = u32[] constant(10)
  ROOT %lt = pred[] compare(%count.cond, %limit), direction=LT
}

%body {
  %param = (u32[], f32[]) parameter(0)
  %count = u32[] get-tuple-element(%param), index=0
  %after-all = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(%after-all), channel_id=1,
    sharding={{maximal device=1 metadata={op_name="a"}},
              {maximal device=1}, {maximal device=1}}
  %recv-done = (f32[], token[]) recv-done(%recv), channel_id=1
  %data = f32[] get-tuple-element(%recv-done), index=0
  ROOT %tuple = (u32[], f32[]) tuple(%count, %data)
}

ENTRY %entry {
  %p0 = f32[] parameter(0)
  %zero = u32[] constant(0)
  %init = (u32[], f32[]) tuple(%zero, %p0)
  %while = (u32[], f32[]) while(%init), body=%body, condition=%cond,
    sharding={{maximal device=0 metadata={op_name="b"}},{maximal device=0}}
  ROOT %result = f32[] get-tuple-element(%while), index=1
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  auto result =
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get());
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr(
                  "Instruction: while is on device: 0, which conflicts with "
                  "device: 1 of channel instruction: recv"));
}

TEST_P(ParameterizedMetadataTest, WhileConv) {
  const char* const hlo_string = R"(
HloModule module

%cond {
  %vars.cond = (u32[], bf16[2, 2048, 768], bf16[128,512,2048], bf16[128,512,768], s32[]) parameter(0)
  %count.cond = u32[] get-tuple-element(%vars.cond), index=0
  %limit = u32[] constant(2)
  ROOT %lt = pred[] compare(%count.cond, %limit), direction=LT
}

%body {
  %param = (u32[], bf16[2, 2048, 768], bf16[128,512,2048], bf16[128,512,768], s32[]) parameter(0)
  %i0 = s32[] constant(0)
  %count = u32[] get-tuple-element(%param), index=0
  %gte0 = bf16[2,2048,768]{2,1,0}
   get-tuple-element(%param), index=1
  %index = s32[] get-tuple-element(%param), index=4
  %dys = bf16[1,2048,768]{2,1,0} dynamic-slice(%gte0, s32[] %index, s32[] %i0, s32[] %i0),
   dynamic_slice_sizes={1,2048,768}
  %kernel = bf16[2048, 768]{1,0}
   reshape(%dys)
  %lhs = bf16[128,512,2048]{2,1,0}
   get-tuple-element(%param), index=2,
   sharding={devices=[8,1,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  %reshape = bf16[2048,768,1]{2,1,0} reshape(bf16[2048,768]{1,0} %kernel)
  %convolution = bf16[128,512,768]{2,1,0}
    convolution(bf16[128,512,2048]{2,1,0} %lhs,
    bf16[2048,768,1]{2,1,0} %reshape), window={size=1},
    dim_labels=0bf_io0->0bf, sharding={devices=[8,1,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  ROOT %tuple = (u32[], bf16[2,2048,768], bf16[128,512,2048], bf16[128,512,768], s32[]) tuple(%count, %gte0, %lhs, %convolution, index)
}

ENTRY %entry {
  %p0 = bf16[2048,768] parameter(0),
    sharding={devices=[2,1,8]0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15 last_tile_dim_replicate}
  %p1 = bf16[128,512,2048] parameter(1),
   sharding={devices=[8,1,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  %p2 = bf16[128,512,768] parameter(2)
  %reshape0 = bf16[1,2048,768] reshape(%p0)
  %concat0 = bf16[2,2048,768] concatenate(%reshape0, %reshape0), dimensions={0}
  %zero = u32[] constant(0)
  %p3 = s32[] parameter(3)
  %init = (u32[], bf16[2, 2048, 768], bf16[128,512,2048], bf16[128,512,768], s32[]) tuple(%zero, %concat0, %p1, %p2, %p3)
  %while = (u32[], bf16[2, 2048, 768], bf16[128,512,2048], bf16[128,512,768], s32[]) while(%init), body=%body, condition=%cond
  ROOT %result = bf16[128,512,768] get-tuple-element(%while), index=3, sharding={replicated}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* kernel = FindInstruction(module.get(), "kernel");
  ASSERT_NE(kernel, nullptr);
  EXPECT_THAT(kernel, op::Sharding("{devices=[2,1,8]0,2,4,6,8,10,12,14,1,3,5,"
                                   "7,9,11,13,15 last_tile_dim_replicate}"));
}

TEST_P(ParameterizedMetadataTest, DoNotPassThroughConcatAtFirstIteration) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %p0 = bf16[16,2048,768] parameter(0),
    sharding={devices=[2,1,1,8]0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15 last_tile_dim_replicate}
  %concat = bf16[32,2048,768] concatenate(%p0, %p0), dimensions={0}
  %add = bf16[32,2048,768] add(%concat, %concat),
   sharding={devices=[8,1,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  ROOT %result = bf16[32,2048,768] copy(%add)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* kernel = FindInstruction(module.get(), "concat");
  ASSERT_NE(kernel, nullptr);
  EXPECT_THAT(kernel, op::Sharding("{devices=[8,1,2]0,1,2,3,4,5,6,7,8,"
                                   "9,10,11,12,13,14,15}"));
}

TEST_P(ParameterizedMetadataTest, DoNotPassThroughConcatAtFirstIteration2) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %p0 = bf16[16,2048,768] parameter(0),
    sharding={devices=[1,2,1,8]0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15 last_tile_dim_replicate}
  %concat = bf16[32,2048,768] concatenate(%p0, %p0), dimensions={0}
  %add = bf16[32,2048,768] add(%concat, %concat),
   sharding={devices=[8,1,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  ROOT %result = bf16[32,2048,768] copy(%add)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* kernel = FindInstruction(module.get(), "concat");
  ASSERT_NE(kernel, nullptr);
  EXPECT_THAT(kernel, op::Sharding("{devices=[8,1,2]0,1,2,3,4,5,6,7,8,"
                                   "9,10,11,12,13,14,15}"));
}

TEST_P(ParameterizedMetadataTest,
       DoNotPassThroughDynamicSliceAtFirstIteration) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %p0 = bf16[64,2048,768] parameter(0),
    sharding={devices=[2,1,1,8]0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15 last_tile_dim_replicate}
  %p1 = s32[] parameter(1)
  %i0 = s32[] constant(0)
  %dys = bf16[32,2048,768] dynamic-slice(%p0, s32[] %p1, s32[] %i0, s32[] %i0),
   dynamic_slice_sizes={32,2048,768}
  %add = bf16[32,2048,768] add(%dys, %dys),
   sharding={devices=[8,1,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  ROOT %result = bf16[32,2048,768] copy(%add)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* kernel = FindInstruction(module.get(), "dys");
  ASSERT_NE(kernel, nullptr);
  EXPECT_THAT(kernel, op::Sharding("{devices=[8,1,2]0,1,2,3,4,5,6,7,8,"
                                   "9,10,11,12,13,14,15}"));
}

TEST_P(ParameterizedMetadataTest, Dot) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %param.0 = f32[8,256,128] parameter(0)
  %param.1 = f32[8,128,512] parameter(1)
  %param.2 = f32[8,128] parameter(2)

  %p0_copy_0 = f32[8,256,128] copy(%param.0),
    sharding={devices=[1,4,1]0,1,2,3 metadata={op_name="a"}}
  %p1_copy_0 = f32[8,128,512] copy(%param.1),
    sharding={devices=[1,1,4]0,1,2,3 metadata={op_name="b"}}
  %p2_copy = f32[8,128] copy(%param.2)
  %dot_prop_rhs = f32[8,256,512] dot(%p0_copy_0, %p1_copy_0),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
  %dot_prop_lhs = f32[8,512,256] dot(%p1_copy_0, %p0_copy_0),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_contracting_dims={2}
  %dot_mat_vec = f32[8,256] dot(%p0_copy_0, %p2_copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}

  %p0_copy_1 = f32[8,256,128] copy(%param.0)
  %p1_copy_1 = f32[8,128,512] copy(%param.1)
  %dot_back_prop_rhs = f32[8,256,512] dot(%p0_copy_1, %p1_copy_1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
  %copy_back_prop_rhs = f32[8,256,512] copy(%dot_back_prop_rhs),
    sharding={devices=[1,2,2]0,1,2,3 metadata={op_name="c"}}

  ROOT %tuple = (f32[8,512,256], f32[8,256,512], f32[8,256], f32[8,256,512])
    tuple(%dot_prop_lhs, %dot_prop_rhs, %dot_mat_vec, %copy_back_prop_rhs)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* dot_prop_rhs = FindInstruction(module.get(), "dot_prop_rhs");
  ASSERT_NE(dot_prop_rhs, nullptr);
  EXPECT_THAT(dot_prop_rhs, op::Sharding("{devices=[1,1,4]0,1,2,3}"));
  auto* dot_prop_lhs = FindInstruction(module.get(), "dot_prop_lhs");
  ASSERT_NE(dot_prop_lhs, nullptr);
  EXPECT_THAT(dot_prop_lhs, op::Sharding("{devices=[1,4,1]0,1,2,3}"));
  auto* dot_mat_vec = FindInstruction(module.get(), "dot_mat_vec");
  ASSERT_NE(dot_mat_vec, nullptr);
  EXPECT_THAT(dot_mat_vec, op::Sharding("{devices=[1,4]0,1,2,3}"));

  auto* p0_copy_1 = FindInstruction(module.get(), "p0_copy_1");
  ASSERT_NE(p0_copy_1, nullptr);
  EXPECT_THAT(
      p0_copy_1,
      op::Sharding("{devices=[1,2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  auto* p1_copy_1 = FindInstruction(module.get(), "p1_copy_1");
  ASSERT_NE(p1_copy_1, nullptr);
  EXPECT_THAT(
      p1_copy_1,
      op::Sharding("{devices=[1,1,2,2]0,2,1,3  last_tile_dim_replicate}"));
  auto* dot_back_prop_rhs = FindInstruction(module.get(), "dot_back_prop_rhs");
  ASSERT_NE(dot_back_prop_rhs, nullptr);
  EXPECT_THAT(dot_back_prop_rhs, op::Sharding("{devices=[1,2,2]0,1,2,3}"));

  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(dot_prop_rhs->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
    EXPECT_THAT(dot_prop_lhs->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
    EXPECT_THAT(dot_mat_vec->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(p0_copy_1->sharding(), ShardingMetadata({CreateMetadata("c")}));
    EXPECT_THAT(p1_copy_1->sharding(), ShardingMetadata({CreateMetadata("c")}));
    EXPECT_THAT(dot_back_prop_rhs->sharding(),
                ShardingMetadata({CreateMetadata("c")}));
  } else {
    for (HloInstruction* instruction :
         {dot_prop_rhs, dot_prop_lhs, dot_mat_vec, p0_copy_1, p1_copy_1,
          dot_back_prop_rhs}) {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, DotTiledBatchDim) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[8,256,512] parameter(0)
  %p1 = f32[8,512,128] parameter(1)

  %add = f32[8,256,512] add(%p0, %p0)
  %dot = f32[8,256,128] dot(%add, %p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
  %res = f32[8,32768] reshape(%dot),
    sharding={devices=[2,2]0,1,2,3 metadata={op_name="a"}}

  ROOT %tuple = (f32[8,32768]) tuple(%res)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "add");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,2,1]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, DotMergeOperands) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[8,256,512] parameter(0),
    sharding={devices=[2,2,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate metadata={op_name="a"}}
  %p1 = f32[8,128,512] parameter(1),
    sharding={devices=[2,2,1,2]0,2,1,3,4,6,5,7 last_tile_dim_replicate metadata={op_name="b"}}
  %dot = f32[8,256,128] dot(%p0, %p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
  ROOT %copy = f32[8,256,128] copy(%dot)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "dot");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,2,2]0,1,2,3,4,5,6,7}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b"), CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, DotMergeOperands2) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[8,256,512] parameter(0),
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7 metadata={op_name="a"}}
  %p1 = f32[8,128,512] parameter(1),
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7 metadata={op_name="b"}}
  %dot = f32[8,256,128] dot(%p0, %p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
  ROOT %copy = f32[8,256,128] copy(%dot)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "dot");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding(
          "{devices=[2,2,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, DotMergeOperands3) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[256,512] parameter(0),
    sharding={devices=[2,4]0,1,2,3,4,5,6,7 metadata={op_name="a"}}
  %p1 = f32[128,512] parameter(1),
    sharding={devices=[4,2]0,4,2,6,3,7,1,5 metadata={op_name="b"}}
  %dot = f32[256,128] dot(%p0, %p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  ROOT %copy = f32[256,128] copy(%dot)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "dot");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,4]0,2,3,1,4,6,7,5}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b"), CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ForwardDotWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[8,256,512] parameter(0),
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7 metadata={op_name="a"}}
  %p1 = f32[8,128,512] parameter(1)
  %shard-barrier-from = f32[8,256,512] custom-call(%p0), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %dot = f32[8,256,128] dot(%shard-barrier-from, %p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
  ROOT %copy = f32[8,256,128] copy(%dot)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "dot");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTest, BackwardDotWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[8,256,512] parameter(0),
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7 metadata={op_name="a"}}
  %p1 = f32[8,128,512] parameter(1)
  %copy1 = f32[8,128,512] copy(%p1)
  %shard-barrier-to = f32[8,128,512] custom-call(%copy1), custom_call_target="ShardBarrierTo", custom_call_has_side_effect=true
  %dot = f32[8,256,128] dot(%p0, %shard-barrier-to),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[2,1,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate metadata={op_name="b"}}
  ROOT %copy = f32[8,256,128] copy(%dot)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "copy1");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{replicated}"));
}

TEST_P(ParameterizedMetadataTest, BackwardDotFromContracting) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[8,256,512] parameter(0),
    sharding={devices=[2,2,2]0,1,2,3,4,5,6,7 metadata={op_name="a"}}
  %p1 = f32[8,128,512] parameter(1)
  %copy1 = f32[8,128,512] copy(%p1)
  %dot = f32[8,256,128] dot(%p0, %copy1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2},
    sharding={devices=[2,1,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate metadata={op_name="b"}}
  ROOT %copy = f32[8,256,128] copy(%dot)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "copy1");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,2,2]0,1,2,3,4,5,6,7}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a"), CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, BackwardDotFromContractingWithManual) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %dot {
  %p0 = f32[8,512] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dims={manual} metadata={op_name="a"}}
  %p1 = f32[512,128] parameter(1)
  %copy1 = f32[512,128] copy(%p1)
  %dot = f32[8,128] dot(%p0, %copy1),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dims={replicated, manual} metadata={op_name="b"}}
  ROOT %copy = f32[8,128] copy(%dot)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "copy1");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dims={manual}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ConvAsDotOnTrivialDimsForward) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %lhs = f32[128,1,1,1001] parameter(0),
    sharding={devices=[1,2,1,1]0,1 metadata={op_name="a"}}
  %rhs = f32[1,1,1024,1001] parameter(1),
    sharding={devices=[1,2,1,1]0,1 metadata={op_name="b"}}
  %convolution = f32[128,1,1,1024] convolution(%lhs, %rhs),
    window={size=1x1 rhs_reversal=1x1}, dim_labels=b01f_01oi->b01f
  ROOT %copy = f32[128,1,1,1024] copy(%convolution)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "convolution");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,1,2,1]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ConvAsDotForwardWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %lhs = f32[128,1,1,1001] parameter(0),
    sharding={devices=[1,2,1,1]0,1 metadata={op_name="a"}}
  %rhs = f32[1,1,1024,1001] parameter(1),
    sharding={devices=[1,2,1,1]0,1 metadata={op_name="b"}}
  %shard-barrier-from = f32[1,1,1024,1001] custom-call(%rhs), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %convolution = f32[128,1,1,1024] convolution(%lhs, %shard-barrier-from),
    window={size=1x1 rhs_reversal=1x1}, dim_labels=b01f_01oi->b01f
  ROOT %copy = f32[128,1,1,1024] copy(%convolution)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "convolution");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{replicated}"));
}

TEST_P(ParameterizedMetadataTest, ConvAsDotOnTrivialDimsBackward) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[128,5,5,128] parameter(0)
  %lhs = f32[128,5,5,128] copy(%p0)
  %p1 = f32[5,5,128,768] parameter(1)
  %rhs = f32[5,5,128,768] copy(%p1)
  %convolution = f32[128,1,1,768] convolution(%lhs, %rhs), window={size=5x5},
    dim_labels=b01f_01io->b01f,
    sharding={devices=[1,2,1,1]0,1 metadata={op_name="a"}}
  ROOT %copy = f32[128,1,1,768] copy(%convolution)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* lhs = FindInstruction(module.get(), "lhs");
  ASSERT_NE(lhs, nullptr);
  auto* rhs = FindInstruction(module.get(), "rhs");
  ASSERT_NE(rhs, nullptr);
  for (HloInstruction* instruction : {lhs, rhs}) {
    EXPECT_THAT(instruction, op::Sharding("{replicated}"));
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, ConvAsDotBackwardWithBarrier) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[128,5,5,128] parameter(0)
  %lhs = f32[128,5,5,128] copy(%p0)
  %p1 = f32[5,5,128,768] parameter(1)
  %rhs = f32[5,5,128,768] copy(%p1)
  %shard-barrier-from = f32[128,5,5,128] custom-call(%lhs), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %convolution = f32[128,1,1,768] convolution(%shard-barrier-from, %rhs), window={size=5x5},
    dim_labels=b01f_01io->b01f,
    sharding={devices=[1,2,1,1]0,1 metadata={op_name="a"}}
  ROOT %copy = f32[128,1,1,768] copy(%convolution)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* lhs = FindInstruction(module.get(), "lhs");
  ASSERT_NE(lhs, nullptr);
  EXPECT_THAT(lhs, op::Sharding("{replicated}"));
}

TEST_P(ParameterizedMetadataTest,
       ConvolutionFilterIFOFPartitionedInputPartialReplicate) {
  const char* const hlo_string = R"(
  HloModule module

ENTRY entry {
  %lhs = f32[128,112,112,12] parameter(0)
  %lhs.copy = f32[128,112,112,12] copy(f32[128,112,112,12] %lhs),
    sharding={devices=[1,1,1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="a"}}
  %rhs = f32[7,7,12,64] parameter(1)
  %rhs.copy = f32[7,7,12,64] copy(f32[7,7,12,64] %rhs),
    sharding={devices=[1,1,2,2]0,1,2,3 metadata={op_name="b"}}
  %conv = f32[128,56,56,64] convolution(
    f32[128,112,112,12] %lhs.copy,
    f32[7,7,12,64] %rhs.copy),
    window={size=7x7 stride=2x2 pad=3_3x3_3},
    dim_labels=b01f_01io->b01f
  ROOT %copy = f32[128,56,56,64] copy(conv)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "conv");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding("{devices=[1,1,1,2,2]0,2,1,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ConvolutionDataParallelism) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[256,512,16,32] parameter(0), sharding={devices=[2,2,2,2]<=[16] metadata={op_name="lhs_sharding"}}
  p1 = f32[512,1,12,28] parameter(1), sharding={replicated metadata={op_name="rhs_sharding"}}
  conv = f32[256,512,5,5] convolution(p0, p1), window={size=12x28}, dim_labels=bf01_oi01->bf01, feature_group_count=512
  ROOT copy = f32[256,512,5,5] copy(conv)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "conv");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding("{devices=[2,1,1,1,8]<=[16] last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("lhs_sharding")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ConcatFromUserUnshardedDim) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[8,128] parameter(0)
  %p1 = f32[8,128] parameter(1)
  %c0 = f32[8,128] copy(%p0)
  %c1 = f32[8,128] copy(%p1)

  %concat = f32[16,128] concatenate(%c0, %c1),
    dimensions={0},
    sharding={devices=[1,2]0,1 metadata={op_name="a"}}
  ROOT %tuple = (f32[16,128]) tuple(%concat)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* c0 = FindInstruction(module.get(), "c0");
  ASSERT_NE(c0, nullptr);
  auto* c1 = FindInstruction(module.get(), "c1");
  ASSERT_NE(c1, nullptr);
  for (HloInstruction* instruction : {c0, c1}) {
    EXPECT_THAT(instruction, op::Sharding("{devices=[1,2]0,1}"));
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, ConcatFromUserShardedDim) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[8,128] parameter(0)
  %p1 = f32[8,128] parameter(1)
  %c0 = f32[8,128] copy(%p0)
  %c1 = f32[8,128] copy(%p1)

  %concat = f32[16,128] concatenate(%c0, %c1),
    dimensions={0},
    sharding={devices=[3,1]0,1,2 metadata={op_name="a"}}
  ROOT %tuple = (f32[16,128]) tuple(%concat)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* c0 = FindInstruction(module.get(), "c0");
  EXPECT_THAT(c0, op::Sharding("{devices=[2,1]0,1}"));
  ASSERT_NE(c0, nullptr);
  auto* c1 = FindInstruction(module.get(), "c1");
  ASSERT_NE(c1, nullptr);
  EXPECT_THAT(c1, op::Sharding("{devices=[2,1]1,2}"));
  for (HloInstruction* instruction : {c0, c1}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, ConcatFromUserShardedDimMaximalOperand) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[8,128] parameter(0)
  %p1 = f32[24,128] parameter(1)
  %c0 = f32[8,128] copy(%p0)
  %c1 = f32[24,128] copy(%p1)

  %concat = f32[32,128] concatenate(%c0, %c1),
    dimensions={0},
    sharding={devices=[4,1]0,1,2,3 metadata={op_name="a"}}
  ROOT %tuple = (f32[32,128]) tuple(%concat)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* c0 = FindInstruction(module.get(), "c0");
  ASSERT_NE(c0, nullptr);
  EXPECT_THAT(c0, op::NoSharding());
  auto* c1 = FindInstruction(module.get(), "c1");
  ASSERT_NE(c1, nullptr);
  EXPECT_THAT(c1, op::Sharding("{devices=[3,1]1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(c1->sharding(), ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(c1->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ReplicatedToSideEffecting) {
  const char* const hlo_string = R"(
HloModule module
ENTRY entry_computation {
  %const.0 = s32[] constant(0),
    sharding={replicated metadata={op_name="a"}}
  %const.1 = s32[] constant(2147483647),
    sharding={replicated metadata={op_name="b"}}
  %rng = s32[4]{0} rng(%const.0, %const.1),
    distribution=rng_uniform
  ROOT %root = (s32[4]{0}) tuple(%rng)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  EXPECT_EQ(changed,
            !GetParam().propagate_metadata && !GetParam().clear_metadata);
  auto* instruction = FindInstruction(module.get(), "rng");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::NoSharding());
}

TEST_P(ParameterizedMetadataTest, PartReplicatedTupleUser) {
  const char* const hlo_string = R"(
HloModule module
ENTRY entry_computation {
  %param.0 = f32[5] parameter(0)
  %param.1 = f32[7] parameter(1)
  %param.2 = f32[9] parameter(2)
  %tuple.0 = (f32[5], f32[7]) tuple(%param.0, %param.1)
  ROOT %tuple.1 = ((f32[5], f32[7]), f32[9]) tuple(%tuple.0, %param.2),
    sharding={{maximal device=0 metadata={op_name="a"}},
              {replicated metadata={op_name="b"}},
              {maximal device=1 metadata={op_name="c"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "tuple.0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{{maximal device=0}, {replicated}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(instruction->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    for (const HloSharding& sub_sharding :
         instruction->sharding().tuple_elements()) {
      EXPECT_THAT(sub_sharding, ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, Conditional) {
  const char* const hlo_string = R"(
HloModule module

%add-call {
  %x = f32[4,4] parameter(0)
  ROOT %add = f32[4,4] add(%x, %x)
}

%true_comp {
  %tp = (f32[3,5], f32[4,4]) parameter(0)
  %tgte.0 = f32[3,5] get-tuple-element(%tp), index=0
  %ttr = f32[5,3] transpose(%tgte.0), dimensions={1,0}
  %tgte.1 = f32[4,4] get-tuple-element(%tp), index=1
  %tadd = f32[4,4] call(%tgte.1), to_apply=%add-call
  ROOT %tr = (f32[5,3], f32[4,4]) tuple(%ttr, %tadd)
}

%mul-call {
  %y = f32[4,4] parameter(0)
  ROOT %mul = f32[4,4] multiply(%y, %y)
}

%false_comp {
  %fp = (f32[5,3], f32[4,4]) parameter(0)
  %fgte.0 = f32[5,3] get-tuple-element(%fp), index=0
  %fgte.1 = f32[4,4] get-tuple-element(%fp), index=1
  %fmul = f32[4,4] call(%fgte.1), to_apply=%mul-call
  ROOT %fr = (f32[5,3], f32[4,4]) tuple(%fgte.0, %fmul)
}

ENTRY entry {
  %cond = pred[] parameter(0)
  %tp.0 = f32[3,5] parameter(1), sharding={devices=[1,2]0,1 metadata={op_name="a"}}
  %fp.0 = f32[5,3] parameter(2), sharding={devices=[1,3]0,1,2 metadata={op_name="b"}}
  %constant = f32[4] constant({1,2,3,4}), sharding={devices=[4]0,1,2,3 metadata={op_name="c"}}
  %broadcast = f32[4,4] broadcast(%constant), dimensions={1}
  %add = f32[4,4] add(%broadcast, %broadcast)
  %true_param = (f32[3,5], f32[4,4]) tuple(%tp.0, %add)
  %false_param = (f32[5,3], f32[4,4]) tuple(%fp.0, %add)
  %conditional = (f32[5,3], f32[4,4]) conditional(
      %cond, %true_param, %false_param),
    true_computation=%true_comp,
    false_computation=%false_comp
  ROOT %root = f32[5,3] get-tuple-element(%conditional), index=0
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);

  auto* tp = FindInstruction(module.get(), "tp");
  auto* tgte_0 = FindInstruction(module.get(), "tgte.0");
  auto* ttr = FindInstruction(module.get(), "ttr");
  auto* tgte_1 = FindInstruction(module.get(), "tgte.1");
  auto* tadd = FindInstruction(module.get(), "tadd");
  auto* tr = FindInstruction(module.get(), "tr");

  auto* fp = FindInstruction(module.get(), "fp");
  auto* fgte_0 = FindInstruction(module.get(), "fgte.0");
  auto* fgte_1 = FindInstruction(module.get(), "fgte.1");
  auto* fmul = FindInstruction(module.get(), "fmul");
  auto* fr = FindInstruction(module.get(), "fr");

  auto* x = FindInstruction(module.get(), "x");
  auto* add = FindInstruction(module.get(), "add");
  auto* y = FindInstruction(module.get(), "y");
  auto* mul = FindInstruction(module.get(), "mul");

  auto* conditional = FindInstruction(module.get(), "conditional");

  const std::vector<HloInstruction*> instructions(
      {tp, tgte_0, ttr, tgte_1, tadd, tr, fp, fgte_0, fgte_1, fmul, fr, x, add,
       y, mul, conditional});

  for (HloInstruction* instruction : instructions) {
    EXPECT_NE(instruction, nullptr);
    EXPECT_TRUE(instruction->has_sharding());
  }

  for (HloInstruction* instruction :
       {tgte_1, tadd, fgte_1, fmul, x, add, y, mul}) {
    EXPECT_THAT(instruction, op::Sharding("{devices=[1,4]0,1,2,3}"));
  }
  for (HloInstruction* instruction : {tr, fr, conditional, fp}) {
    EXPECT_THAT(instruction,
                op::Sharding("{{devices=[1,3]0,1,2}, {devices=[1,4]0,1,2,3}}"));
  }
  EXPECT_THAT(tp, op::Sharding("{{devices=[1,2]0,1}, {devices=[1,4]0,1,2,3}}"));
  EXPECT_THAT(tgte_0, op::Sharding("{devices=[1,2]0,1}"));
  EXPECT_THAT(ttr, op::Sharding("{devices=[2,1]0,1}"));
  EXPECT_THAT(fgte_0, op::Sharding("{devices=[1,3]0,1,2}"));

  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    for (HloInstruction* instruction :
         {tgte_1, tadd, fgte_1, fmul, x, add, y, mul}) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("c")}));
    }
    for (HloInstruction* instruction : {tr, fr, conditional, fp}) {
      const std::vector<HloSharding>& shardings =
          instruction->sharding().tuple_elements();
      EXPECT_THAT(shardings[0], ShardingMetadata({CreateMetadata("b")}));
      EXPECT_THAT(shardings[1], ShardingMetadata({CreateMetadata("c")}));
    }
    for (HloInstruction* instruction : {tgte_0, ttr}) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    }
    EXPECT_THAT(fgte_0->sharding(), ShardingMetadata({CreateMetadata("b")}));
  } else {
    for (HloInstruction* instruction : instructions) {
      if (instruction->sharding().IsTuple()) {
        for (const HloSharding& tuple_element :
             instruction->sharding().tuple_elements()) {
          EXPECT_THAT(tuple_element, ShardingMetadata({}));
        }
      } else {
        EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
      }
    }
  }
}

TEST_P(ParameterizedMetadataTest, TupleFromUser) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[13] parameter(0)
  %p1 = f32[15] parameter(1)
  %p2 = f32[17] parameter(2)
  %t0 = (f32[13], f32[15]) tuple(%p0, %p1)
  %t1 = ((f32[13], f32[15]), f32[17]) tuple(%t0, %p2)
  %gte.0 = (f32[13], f32[15]) get-tuple-element(%t1), index=0
  %gte.1 = f32[13] get-tuple-element(%gte.0), index=0
  %gte.2 = f32[15] get-tuple-element(%gte.0), index=1
  %gte.3 = f32[17] get-tuple-element(%t1), index=1
  ROOT %t2 = (f32[13], f32[15], f32[17]) tuple(%gte.1, %gte.2, %gte.3),
    sharding={{replicated metadata={op_name="a"}},
              {devices=[2]0,1 metadata={op_name="b"}},
              {devices=[3]1,2,3 metadata={op_name="c"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* t0 = FindInstruction(module.get(), "t0");
  ASSERT_NE(t0, nullptr);
  EXPECT_THAT(t0, op::Sharding("{{replicated}, {devices=[2]0,1}}"));
  auto* t1 = FindInstruction(module.get(), "t1");
  ASSERT_NE(t1, nullptr);
  EXPECT_THAT(
      t1, op::Sharding("{{replicated}, {devices=[2]0,1}, {devices=[3]1,2,3}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(t0->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(t0->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("b")}));
    EXPECT_THAT(t1->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(t1->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("b")}));
    EXPECT_THAT(t1->sharding().tuple_elements()[2],
                ShardingMetadata({CreateMetadata("c")}));
  } else {
    for (HloInstruction* instruction : {t0, t1}) {
      for (const HloSharding& sub_sharding :
           instruction->sharding().tuple_elements()) {
        EXPECT_THAT(sub_sharding, ShardingMetadata({}));
      }
    }
  }
}

TEST_P(ParameterizedMetadataTest, DynamicSliceForwardPassWithBarrier) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0),
    sharding={devices=[1,1,2]0,1 metadata={op_name="a"}}
  %p1 = s32[] parameter(1)
  %i0 = s32[] constant(0)
  %shard-barrier-from = f32[11,13,15] custom-call(%c0), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %ds = f32[11,1,15] dynamic-slice(%shard-barrier-from, %i0, %p1, %i0),
    dynamic_slice_sizes={11,1,15}
  ROOT %root = (f32[11,1,15]) tuple(%ds)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "ds");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTest, DynamicSliceForwardPass) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0),
    sharding={devices=[2,2,2]<=[8] metadata={op_name="a"}}
  %p1 = s32[] parameter(1)
  %i0 = s32[] constant(0)
  %ds = f32[11,1,15] dynamic-slice(%c0, %i0, %p1, %i0),
    dynamic_slice_sizes={11,1,15}
  ROOT %root = (f32[11,1,15]) tuple(%ds)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "ds");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding(
          "{devices=[2,1,2,2]<=[2,2,2]T(0,2,1) last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, DynamicSliceBackwardPass) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0)
  %p1 = s32[] parameter(1)
  %i0 = s32[] constant(0)
  %ds = f32[11,1,15] dynamic-slice(%c0, %i0, %p1, %i0),
    dynamic_slice_sizes={11,1,15},
    sharding={devices=[2,2,2]<=[8] metadata={op_name="a"}}
  ROOT %root = (f32[11,1,15]) tuple(%ds)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "c0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding(
          "{devices=[2,1,2,2]<=[2,2,2]T(0,2,1) last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, DynamicSliceBackwardPassWithBarrier) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0)
  %p1 = s32[] parameter(1)
  %i0 = s32[] constant(0)
  %shard-barrier-to = f32[11,13,15] custom-call(%c0), custom_call_target="ShardBarrierTo", custom_call_has_side_effect=true
  %ds = f32[11,1,15] dynamic-slice(%shard-barrier-to, %i0, %p1, %i0),
    dynamic_slice_sizes={11,1,15},
    sharding={devices=[1,1,2]0,1 metadata={op_name="a"}}
  ROOT %root = (f32[11,1,15]) tuple(%ds)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "c0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTest, DynamicUpdateSliceForwardPassBase) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0),
    sharding={devices=[2,2,2]<=[8] metadata={op_name="a"}}
  %p1 = f32[11,1,15] parameter(1)
  %c1 = f32[11,1,15] copy(%p1)
  %p2 = s32[] parameter(2)
  %i0 = s32[] constant(0)
  %dus = f32[11,13,15] dynamic-update-slice(%c0, %c1, %i0, %p2, %i0)
  ROOT %root = (f32[11,13,15]) tuple(%dus)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* dus = FindInstruction(module.get(), "dus");
  ASSERT_NE(dus, nullptr);
  EXPECT_THAT(dus, op::Sharding("{devices=[2,2,2]<=[8]}"));
  auto* c1 = FindInstruction(module.get(), "c1");
  ASSERT_NE(c1, nullptr);
  EXPECT_THAT(
      c1, op::Sharding(
              "{devices=[2,1,2,2]<=[2,2,2]T(0,2,1) last_tile_dim_replicate}"));
  for (HloInstruction* instruction : {dus, c1}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, DynamicUpdateSliceForwardPassWithBarrier) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0),
    sharding={devices=[1,1,2]0,1 metadata={op_name="a"}}
  %p1 = f32[11,1,15] parameter(1)
  %c1 = f32[11,1,15] copy(%p1)
  %p2 = s32[] parameter(2)
  %i0 = s32[] constant(0)
  %shard-barrier-from = f32[11,13,15] custom-call(%c0), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %dus = f32[11,13,15] dynamic-update-slice(%shard-barrier-from, %c1, %i0, %p2, %i0)
  ROOT %root = (f32[11,13,15]) tuple(%dus)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* dus = FindInstruction(module.get(), "dus");
  ASSERT_NE(dus, nullptr);
  EXPECT_FALSE(dus->has_sharding());
}

TEST_P(ParameterizedMetadataTest, DynamicUpdateSliceForwardPassUpdate) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0)
  %p1 = f32[11,1,15] parameter(1)
  %c1 = f32[11,1,15] copy(%p1),
    sharding={devices=[2,2,2]<=[8] metadata={op_name="a"}}
  %p2 = s32[] parameter(2)
  %i0 = s32[] constant(0)
  %dus = f32[11,13,15] dynamic-update-slice(%c0, %c1, %i0, %p2, %i0)
  ROOT %root = (f32[11,13,15]) tuple(%dus)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* dus = FindInstruction(module.get(), "dus");
  ASSERT_NE(dus, nullptr);
  EXPECT_THAT(
      dus, op::Sharding(
               "{devices=[2,1,2,2]<=[2,2,2]T(0,2,1) last_tile_dim_replicate}"));
  auto* c0 = FindInstruction(module.get(), "c0");
  ASSERT_NE(c0, nullptr);
  EXPECT_THAT(
      c0, op::Sharding(
              "{devices=[2,1,2,2]<=[2,2,2]T(0,2,1) last_tile_dim_replicate}"));
  for (HloInstruction* instruction : {dus, c0}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, DynamicUpdateSliceBackwardPass) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0)
  %p1 = f32[11,1,15] parameter(1)
  %c1 = f32[11,1,15] copy(%p1)
  %p2 = s32[] parameter(2)
  %i0 = s32[] constant(0)
  %dus = f32[11,13,15] dynamic-update-slice(%c0, %c1, %i0, %p2, %i0),
    sharding={devices=[2,2,2]<=[8] metadata={op_name="a"}}
  ROOT %root = (f32[11,13,15]) tuple(%dus)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* c0 = FindInstruction(module.get(), "c0");
  ASSERT_NE(c0, nullptr);
  EXPECT_THAT(c0, op::Sharding("{devices=[2,2,2]<=[8]}"));
  auto* c1 = FindInstruction(module.get(), "c1");
  ASSERT_NE(c1, nullptr);
  EXPECT_THAT(
      c1, op::Sharding(
              "{devices=[2,1,2,2]<=[2,2,2]T(0,2,1) last_tile_dim_replicate}"));
  for (HloInstruction* instruction : {c0, c1}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, DynamicUpdateSliceBackwardPassWithBarrier) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0)
  %p1 = f32[11,1,15] parameter(1)
  %c1 = f32[11,1,15] copy(%p1)
  %p2 = s32[] parameter(2)
  %i0 = s32[] constant(0)
  %shard-barrier-to = f32[11,13,15] custom-call(%c0), custom_call_target="ShardBarrierTo", custom_call_has_side_effect=true
  %dus = f32[11,13,15] dynamic-update-slice(%shard-barrier-to, %c1, %i0, %p2, %i0),
    sharding={devices=[1,1,2]0,1 metadata={op_name="a"}}
  ROOT %root = (f32[11,13,15]) tuple(%dus)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* c0 = FindInstruction(module.get(), "c0");
  ASSERT_NE(c0, nullptr);
  EXPECT_FALSE(c0->has_sharding());
}

TEST_P(ParameterizedMetadataTestWithOutput, EinsumLHSBatchPartitioned) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64] parameter(0)
  %lhs.copy = f32[32,24,64] copy(%lhs),
    sharding={devices=[2,1,1]0,1 metadata={op_name="a"}}
  %rhs = f32[32,39296,64] parameter(1)
  %rhs.copy = f32[32,39296,64] copy(%rhs)
  %conv = f32[32,24,39296] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf_0oi->0bf, window={size=32 stride=31 lhs_dilate=32}
  ROOT %copy = f32[32,24,39296] copy(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata,
                          {GetParam().allow_root_sharding_propagation})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* rhs_copy = FindInstruction(module.get(), "rhs.copy");
  ASSERT_NE(rhs_copy, nullptr);
  EXPECT_THAT(rhs_copy, op::Sharding("{devices=[2,1,1]0,1}"));
  auto* conv = FindInstruction(module.get(), "conv");
  ASSERT_NE(conv, nullptr);
  EXPECT_THAT(conv, op::Sharding("{devices=[2,1,1]0,1}"));
  for (HloInstruction* instruction : {rhs_copy, conv}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
  if (GetParam().allow_root_sharding_propagation) {
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                op::Sharding("{devices=[2,1,1]0,1}"));
  }
}

TEST_P(ParameterizedMetadataTest, EinsumOutputBatchPartitioned) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64] parameter(0)
  %lhs.copy = f32[32,24,64] copy(%lhs)
  %rhs = f32[32,39296,64] parameter(1)
  %rhs.copy = f32[32,39296,64] copy(%rhs)
  %conv = f32[32,24,39296] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf_0oi->0bf, window={size=32 stride=31 lhs_dilate=32},
    sharding={devices=[2,1,1]0,1 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* lhs_copy = FindInstruction(module.get(), "lhs.copy");
  ASSERT_NE(lhs_copy, nullptr);
  EXPECT_THAT(lhs_copy, op::Sharding("{devices=[2,1,1]0,1}"));
  auto* rhs_copy = FindInstruction(module.get(), "rhs.copy");
  ASSERT_NE(rhs_copy, nullptr);
  EXPECT_THAT(rhs_copy, op::Sharding("{devices=[2,1,1]0,1}"));
  for (HloInstruction* instruction : {lhs_copy, rhs_copy}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, EinsumLHSNonContractingPartitioned) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs),
    sharding={devices=[1,2,1,2]0,1,2,3 metadata={op_name="a"}}
  %rhs = f32[32,39296,64,1] parameter(1)
  %rhs.copy = f32[32,39296,64,1] copy(%rhs)
  %conv = f32[32,24,39296,128] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf1_0oi1->0bf1, window={size=32x1 stride=31x1 lhs_dilate=32x1}
  ROOT %copy = f32[32,24,39296,128] copy(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "conv");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,1,2]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, EinsumOutputLHSNonContractingPartitioned) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs)
  %rhs = f32[32,39296,64,1] parameter(1)
  %rhs.copy = f32[32,39296,64,1] copy(%rhs)
  ROOT %conv = f32[32,24,39296,128] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf1_0oi1->0bf1, window={size=32x1 stride=31x1 lhs_dilate=32x1},
    sharding={devices=[1,2,1,2]0,1,2,3 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "lhs.copy");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,1,2]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, EinsumRHSNonContractingPartitioned) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,1] parameter(0)
  %lhs.copy = f32[32,24,64,1] copy(%lhs)
  %rhs = f32[32,39296,64,128] parameter(1)
  %rhs.copy = f32[32,39296,64,128] copy(%rhs),
    sharding={devices=[1,2,1,2]0,1,2,3 metadata={op_name="a"}}
  %conv = f32[32,24,39296,128] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf1_0oi1->0bf1,
    window={size=32x128 stride=31x1 pad=0_0x127_127 lhs_dilate=32x1 rhs_reversal=0x1}
  ROOT %copy = f32[32,24,39296,128] copy(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "conv");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,1,2,2]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, EinsumOutputRHSNonContractingPartitioned) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,1] parameter(0)
  %lhs.copy = f32[32,24,64,1] copy(%lhs)
  %rhs = f32[32,39296,64,128] parameter(1)
  %rhs.copy = f32[32,39296,64,128] copy(%rhs)
  ROOT %conv = f32[32,24,39296,128] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf1_0oi1->0bf1,
    window={size=32x128 stride=31x1 pad=0_0x127_127 lhs_dilate=32x1 rhs_reversal=0x1},
    sharding={devices=[1,1,2,2]0,1,2,3 metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "rhs.copy");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,1,2]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, EinsumChooseLargerOperand) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,1] parameter(0)
  %lhs.copy = f32[32,24,64,1] copy(%lhs),
    sharding={devices=[1,4,1,1]0,1,2,3 metadata={op_name="a"}}
  %rhs = f32[32,39296,64,128] parameter(1)
  %rhs.copy = f32[32,39296,64,128] copy(%rhs),
    sharding={devices=[1,2,1,2]0,1,2,3 metadata={op_name="b"}}
  %conv = f32[32,24,39296,128] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf1_0oi1->0bf1,
    window={size=32x128 stride=31x1 pad=0_0x127_127 lhs_dilate=32x1 rhs_reversal=0x1}
  ROOT %copy = f32[32,24,39296,128] copy(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "conv");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,1,2,2]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, EinsumChooseBatchFirst) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,1] parameter(0)
  %lhs.copy = f32[32,24,64,1] copy(%lhs),
    sharding={devices=[1,2,1,1]0,1 metadata={op_name="a"}}
  %rhs = f32[32,39296,64,128] parameter(1)
  %rhs.copy = f32[32,39296,64,128] copy(%rhs),
    sharding={devices=[2,1,1,1]0,1 metadata={op_name="b"}}
  %conv = f32[32,24,39296,128] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf1_0oi1->0bf1,
    window={size=32x128 stride=31x1 pad=0_0x127_127 lhs_dilate=32x1 rhs_reversal=0x1}
  ROOT %copy = f32[32,24,39296,128] copy(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "conv");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,1,1,1]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, GatherFromIndex) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,2,9] parameter(0),
    sharding={replicated  metadata={op_name="a"}}
  %indices = s32[2,3,4] parameter(1),
    sharding={devices=[1,2,1]0,1 metadata={op_name="b"}}
  %gather = f32[3,4,9] gather(%input, %indices), offset_dims={2},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,9}
  ROOT %copy = f32[3,4,9] copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "gather");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,1,1]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, GatherFromIndex_PartialReplicate) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %indices = s32[3] parameter(1),
   sharding={devices=[2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="b"}}
  %gather = f32[3,9] gather(%input, %indices), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1,
    slice_sizes={1,9}
  ROOT %copy = f32[3,9] copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "gather");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, GatherFromDataOperand) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={devices=[1,2]0,1 metadata={op_name="a"}}
  %indices = s32[3] parameter(1),
    sharding={replicated metadata={op_name="b"}}
  %gather = f32[3,9] gather(%input, %indices), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1,
    slice_sizes={1,9}
  ROOT %copy = f32[3,9] copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "gather");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, GatherFromDataOperand_PartialReplicate) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="a"}}
  %indices = s32[3] parameter(1),
    sharding={replicated metadata={op_name="b"}}
  %gather = f32[3,9] gather(%input, %indices), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1,
    slice_sizes={1,9}
  ROOT %copy = f32[3,9] copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "gather");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, GatherToIndex) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %p1 = s32[3] parameter(1)
  %indices = s32[3] copy(%p1)
  ROOT %gather = f32[3,9] gather(%input, %indices), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1,
    slice_sizes={1,9},
    sharding={devices=[2,1]0,1 metadata={op_name="b"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "indices");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, GatherToIndex_PartialReplicate) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %p1 = s32[3] parameter(1)
  %indices = s32[3] copy(%p1)
  ROOT %gather = f32[3,9] gather(%input, %indices), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1,
    slice_sizes={1,9},
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="b"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "indices");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[2,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, GatherToIndex2) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %input = bf16[2,4819,4] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %p1 = s32[2,1000,2] parameter(1)
  %indices = s32[2,1000,2] copy(%p1)
  ROOT %gather = bf16[2,1000,4]
    gather(bf16[2,4819,4] %input, s32[2,1000,2] %indices),
    offset_dims={2}, collapsed_slice_dims={0,1},
    start_index_map={0,1}, index_vector_dim=2, slice_sizes={1,1,4},
    sharding={devices=[1,2,1]0,1 metadata={op_name="b"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "indices");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,1]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, GatherToIndex2_PartialReplicate) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %input = bf16[2,4819,4] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %p1 = s32[2,1000,2] parameter(1)
  %indices = s32[2,1000,2] copy(%p1)
  ROOT %gather = bf16[2,1000,4]
    gather(bf16[2,4819,4] %input, s32[2,1000,2] %indices),
    offset_dims={2}, collapsed_slice_dims={0,1},
    start_index_map={0,1}, index_vector_dim=2, slice_sizes={1,1,4},
    sharding={devices=[1,2,1,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="b"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "indices");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding("{devices=[1,2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, GatherToIndex3) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %input = bf16[2,4819,4] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %p1 = s32[2,2,1000] parameter(1)
  %indices = s32[2,2,1000] copy(%p1)
  ROOT %gather = bf16[2,1000,4]
    gather(bf16[2,4819,4] %input, s32[2,2,1000] %indices),
    offset_dims={2}, collapsed_slice_dims={0,1},
    start_index_map={0,1}, index_vector_dim=1, slice_sizes={1,1,4},
    sharding={devices=[1,2,1]0,1 metadata={op_name="b"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "indices");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,1,2]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, GatherToDataOperand) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %p0 = f32[2,9] parameter(0)
  %input = f32[2,9] copy(%p0)
  %indices = s32[3] parameter(1),
    sharding={replicated metadata={op_name="a"}}
  ROOT %gather = f32[3,9] gather(%input, %indices), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1,
    slice_sizes={1,9},
    sharding={devices=[1,2]0,1 metadata={op_name="b"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "input");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, GatherToDataOperand_PartialReplicate) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %p0 = f32[2,9] parameter(0)
  %input = f32[2,9] copy(%p0)
  %indices = s32[3] parameter(1),
    sharding={replicated metadata={op_name="a"}}
  ROOT %gather = f32[3,9] gather(%input, %indices), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1,
    slice_sizes={1,9},
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="b"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "input");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, DataOperandToScatter) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={devices=[1,2]0,1 metadata={op_name="a"}}
  %indices = s32[3] parameter(1),
    sharding={replicated metadata={op_name="b"}}
  %updates = f32[3,9] parameter(2),
    sharding={replicated metadata={op_name="c"}}
  %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
  ROOT %copy = f32[2,9] copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, DataOperandToScatter_PartialReplicate) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="a"}}
  %indices = s32[3] parameter(1),
    sharding={replicated metadata={op_name="b"}}
  %updates = f32[3,9] parameter(2),
    sharding={replicated metadata={op_name="c"}}
  %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
  ROOT %copy = f32[2,9] copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, DataOperandToScatter_Variadic) {
  const char* const hlo_string = R"(
HloModule module

add (lhs.0: f32[], lhs.1: f32[], rhs.0: f32[], rhs.1: f32[]) -> (f32[], f32[]) {
  lhs.0 = f32[] parameter(0)
  lhs.1 = f32[] parameter(1)
  rhs.0 = f32[] parameter(2)
  rhs.1 = f32[] parameter(3)
  sum.0 = f32[] add(lhs.0, rhs.0)
  sum.1 = f32[] add(lhs.1, rhs.1)
  ROOT tuple = tuple(sum.0, sum.1)
}

ENTRY entry {
  %input.0 = f32[2,9] parameter(0),
    sharding={devices=[1,4]0,1,2,3 metadata={op_name="a"}}
  %input.1 = f32[2,9] parameter(1),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="b"}}
  %indices = s32[3] parameter(2),
    sharding={replicated metadata={op_name="c"}}
  %updates.0 = f32[3,9] parameter(3),
    sharding={replicated metadata={op_name="d"}}
  %updates.1 = f32[3,9] parameter(4),
    sharding={replicated metadata={op_name="e"}}
  %scatter = (f32[2,9],f32[2,9]) scatter(%input.0, %input.1, %indices, %updates.0, %updates.1),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
  ROOT %copy = (f32[2,9],f32[2,9]) copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{{devices=[1,4]0,1,2,3}, {devices=[1,2,2]0,1,2,3 "
                           "last_tile_dim_replicate}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(instruction->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, UpdateOperandToScatter) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %indices = s32[3] parameter(1),
    sharding={replicated metadata={op_name="b"}}
  %updates = f32[3,9] parameter(2),
    sharding={devices=[1,2]0,1 metadata={op_name="c"}}
  %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
  ROOT %copy = f32[2,9] copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("c")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, UpdateOperandToScatter_PartialReplicate) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %indices = s32[3] parameter(1),
    sharding={replicated metadata={op_name="b"}}
  %updates = f32[3,9] parameter(2),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="c"}}
  %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
  ROOT %copy = f32[2,9] copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("c")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, UpdateOperandToScatter_Variadic) {
  const char* const hlo_string = R"(
HloModule module

add (lhs.0: f32[], lhs.1: f32[], rhs.0: f32[], rhs.1: f32[]) -> (f32[], f32[]) {
  lhs.0 = f32[] parameter(0)
  lhs.1 = f32[] parameter(1)
  rhs.0 = f32[] parameter(2)
  rhs.1 = f32[] parameter(3)
  sum.0 = f32[] add(lhs.0, rhs.0)
  sum.1 = f32[] add(lhs.1, rhs.1)
  ROOT tuple = tuple(sum.0, sum.1)
}

ENTRY entry {
  %input.0 = f32[2,9] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %input.1 = f32[2,9] parameter(1),
    sharding={replicated metadata={op_name="b"}}
  %indices = s32[3] parameter(2),
    sharding={replicated metadata={op_name="c"}}
  %updates.0 = f32[3,9] parameter(3),
    sharding={devices=[1,4]0,1,2,3 metadata={op_name="d"}}
  %updates.1 = f32[3,9] parameter(4),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="e"}}
  %scatter = (f32[2,9],f32[2,9]) scatter(%input.0, %input.1, %indices, %updates.0, %updates.1),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
  ROOT %copy = (f32[2,9],f32[2,9]) copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{{devices=[1,4] 0,1,2,3}, {devices=[1,2,2]0,1,2,3 "
                           "last_tile_dim_replicate}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("d")}));
    EXPECT_THAT(instruction->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("e")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterToDataOperand_PartialReplicate) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %p0 = f32[2,9] parameter(0)
  %input = f32[2,9] copy(%p0)
  %indices = s32[3] parameter(1),
    sharding={replicated metadata={op_name="a"}}
  %updates = f32[3,9] parameter(2),
    sharding={replicated metadata={op_name="b"}}
  ROOT %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="c"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "input");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("c")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterToDataOperand) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %p0 = f32[2,9] parameter(0)
  %input = f32[2,9] copy(%p0)
  %indices = s32[3] parameter(1),
    sharding={replicated metadata={op_name="a"}}
  %updates = f32[3,9] parameter(2),
    sharding={replicated metadata={op_name="b"}}
  ROOT %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      sharding={devices=[1,2]0,1 metadata={op_name="c"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "input");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("c")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterToDataOperand_Variadic) {
  const char* const hlo_string = R"(
HloModule module

add (lhs.0: f32[], lhs.1: f32[], rhs.0: f32[], rhs.1: f32[]) -> (f32[], f32[]) {
  lhs.0 = f32[] parameter(0)
  lhs.1 = f32[] parameter(1)
  rhs.0 = f32[] parameter(2)
  rhs.1 = f32[] parameter(3)
  sum.0 = f32[] add(lhs.0, rhs.0)
  sum.1 = f32[] add(lhs.1, rhs.1)
  ROOT tuple = tuple(sum.0, sum.1)
}

ENTRY entry {
  %p0 = f32[2,9] parameter(0)
  %input.0 = f32[2,9] copy(%p0)
  %p1 = f32[2,9] parameter(1)
  %input.1 = f32[2,9] copy(%p1)
  %indices = s32[3] parameter(2),
    sharding={replicated metadata={op_name="a"}}
  %updates.0 = f32[3,9] parameter(3),
    sharding={replicated metadata={op_name="b"}}
  %updates.1 = f32[3,9] parameter(4),
    sharding={replicated metadata={op_name="c"}}
  ROOT %scatter = (f32[2,9],f32[2,9]) scatter(%input.0, %input.1, %indices, %updates.0, %updates.1),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      sharding={{devices=[1,4]0,1,2,3 metadata={op_name="d"}}, {devices=[1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="e"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "input.0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,4]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("d")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }

  instruction = FindInstruction(module.get(), "input.1");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("e")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterToUpdateOperand_PartialReplicate) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0)
  %indices = s32[3] parameter(1),
    sharding={replicated metadata={op_name="a"}}
  %p2 = f32[3,9] parameter(2)
  %updates = f32[3,9] copy(%p2)
  ROOT %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="b"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "updates");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterToUpdateOperand) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0)
  %indices = s32[3] parameter(1),
    sharding={replicated metadata={op_name="a"}}
  %p2 = f32[3,9] parameter(2)
  %updates = f32[3,9] copy(%p2)
  ROOT %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      sharding={devices=[1,2]0,1 metadata={op_name="b"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "updates");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterToUpdateOperand_Variadic) {
  const char* const hlo_string = R"(
HloModule module

add (lhs.0: f32[], lhs.1: f32[], rhs.0: f32[], rhs.1: f32[]) -> (f32[], f32[]) {
  lhs.0 = f32[] parameter(0)
  lhs.1 = f32[] parameter(1)
  rhs.0 = f32[] parameter(2)
  rhs.1 = f32[] parameter(3)
  sum.0 = f32[] add(lhs.0, rhs.0)
  sum.1 = f32[] add(lhs.1, rhs.1)
  ROOT tuple = tuple(sum.0, sum.1)
}

ENTRY entry {
  %input.0 = f32[2,9] parameter(0)
  %input.1 = f32[2,9] parameter(1)
  %indices = s32[3] parameter(2),
    sharding={replicated metadata={op_name="a"}}
  %p3 = f32[3,9] parameter(3)
  %updates.0 = f32[3,9] copy(%p3)
  %p4 = f32[3,9] parameter(4)
  %updates.1 = f32[3,9] copy(%p4)
  ROOT %scatter = (f32[2,9],f32[2,9]) scatter(%input.0, %input.1, %indices, %updates.0, %updates.1),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      sharding={{devices=[1,4]0,1,2,3 metadata={op_name="b"}}, {devices=[1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="c"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "updates.0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,4]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }

  instruction = FindInstruction(module.get(), "updates.1");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("c")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterUpdateToIndex) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %p1 = s32[3] parameter(1),
    sharding={replicated metadata={op_name="b"}}
  %indices = s32[3] copy(%p1)
  %updates = f32[3,9] parameter(2),
    sharding={devices=[2,1]0,1 metadata={op_name="c"}}
  ROOT %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      sharding={replicated metadata={op_name="d"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "indices");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("c")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterUpdateToIndex2) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %p1 = s32[1,3] parameter(1),
    sharding={replicated metadata={op_name="b"}}
  %indices = s32[1,3] copy(%p1)
  %updates = f32[3,9] parameter(2),
    sharding={devices=[2,1]0,1 metadata={op_name="c"}}
  ROOT %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=0,
      sharding={replicated metadata={op_name="d"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "indices");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("c")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterUpdateToIndex_PartialReplicate) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %p1 = s32[3] parameter(1),
    sharding={replicated metadata={op_name="b"}}
  %indices = s32[3] copy(%p1)
  %updates = f32[3,9] parameter(2),
    sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="c"}}
  ROOT %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      sharding={replicated metadata={op_name="d"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "indices");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[2,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("c")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterUpdateToIndex_RankMismatch) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[1,24,24,24,3,3] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %p1 = s32[1,24,24,24,5] parameter(1),
    sharding={replicated metadata={op_name="b"}}
  %indices = s32[1,24,24,24,5] copy(%p1)
  %updates = f32[1,24,24,24,3] parameter(2),
    sharding={devices=[1,2,2,2,1]0,1,2,3,4,5,6,7 metadata={op_name="c"}}
  %scatter = f32[1,24,24,24,3,3] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={4},
      inserted_window_dims={0,1,2,3,4},
      scatter_dims_to_operand_dims={0,1,2,3,4},
      index_vector_dim=4,
      sharding={replicated metadata={op_name="d"}}
  ROOT %copy = f32[1,24,24,24,3,3] copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "indices");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2,2,1]0,1,2,3,4,5,6,7}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("c")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterUpdateToIndex_Variadic) {
  const char* const hlo_string = R"(
HloModule module

add (lhs.0: f32[], lhs.1: f32[], rhs.0: f32[], rhs.1: f32[]) -> (f32[], f32[]) {
  lhs.0 = f32[] parameter(0)
  lhs.1 = f32[] parameter(1)
  rhs.0 = f32[] parameter(2)
  rhs.1 = f32[] parameter(3)
  sum.0 = f32[] add(lhs.0, rhs.0)
  sum.1 = f32[] add(lhs.1, rhs.1)
  ROOT tuple = tuple(sum.0, sum.1)
}

ENTRY entry {
  %input.0 = f32[2,9] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %input.1 = f32[2,9] parameter(1),
    sharding={replicated metadata={op_name="b"}}
  %p2 = s32[3,3] parameter(2),
    sharding={replicated metadata={op_name="c"}}
  %indices = s32[3,3] copy(%p2)
  %updates.0 = f32[3,3,9] parameter(3),
    sharding={devices=[2,1,1,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="d"}}
  %updates.1 = f32[3,3,9] parameter(4),
    sharding={devices=[1,2,1,2]0,2,1,3 last_tile_dim_replicate metadata={op_name="e"}}
  ROOT %scatter = (f32[2,9],f32[2,9]) scatter(%input.0, %input.1, %indices, %updates.0, %updates.1),
      to_apply=add,
      update_window_dims={2},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=2,
      sharding={{replicated metadata={op_name="d"}}, {replicated metadata={op_name="e"}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "indices");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,2]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("d"), CreateMetadata("e")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterIndexToUpdate) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %indices = s32[3] parameter(1),
    sharding={devices=[2]0,1 metadata={op_name="b"}}
  %p2 = f32[3,9] parameter(2),
    sharding={replicated metadata={op_name="c"}}
  %updates = f32[3,9] copy(%p2)
  ROOT %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      sharding={replicated metadata={op_name="d"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "updates");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,1]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterIndexToUpdate_PartialReplicate) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = f32[2,9] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %indices = s32[3] parameter(1),
    sharding={devices=[2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="b"}}
  %p2 = f32[3,9] parameter(2),
    sharding={replicated metadata={op_name="c"}}
  %updates = f32[3,9] copy(%p2)
  ROOT %scatter = f32[2,9] scatter(%input, %indices, %updates),
      to_apply=add,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      sharding={replicated metadata={op_name="d"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "updates");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterIndexToUpdate2_PartialReplicate) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY entry {
  %input = bf16[15,8] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %indices = s32[8,1,1] parameter(1),
    sharding={devices=[2,1,1,4]0,1,2,3,4,5,6,7
      last_tile_dim_replicate metadata={op_name="b"}}
  %p2 = bf16[8,1,8] parameter(2),
    sharding={replicated metadata={op_name="c"}}
  %updates = bf16[8,1,8] copy(%p2)
  ROOT %scatter = bf16[15,8]{1,0} scatter(bf16[15,8] %input,
    s32[8,1,1] %indices, bf16[8,1,8] %updates),
    update_window_dims={2},
    inserted_window_dims={0},
    scatter_dims_to_operand_dims={0}, index_vector_dim=2, to_apply=%add,
      sharding={replicated metadata={op_name="d"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "updates");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding(
          "{devices=[2,1,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ScatterIndexToUpdate_Variadic) {
  const char* const hlo_string = R"(
HloModule module

add (lhs.0: f32[], lhs.1: f32[], rhs.0: f32[], rhs.1: f32[]) -> (f32[], f32[]) {
  lhs.0 = f32[] parameter(0)
  lhs.1 = f32[] parameter(1)
  rhs.0 = f32[] parameter(2)
  rhs.1 = f32[] parameter(3)
  sum.0 = f32[] add(lhs.0, rhs.0)
  sum.1 = f32[] add(lhs.1, rhs.1)
  ROOT tuple = tuple(sum.0, sum.1)
}

ENTRY entry {
  %input.0 = f32[2,9] parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %input.1 = f32[2,9] parameter(1),
    sharding={replicated metadata={op_name="b"}}
  %indices = s32[3,3] parameter(2),
    sharding={devices=[2,2]0,1,2,3 metadata={op_name="c"}}
  %p3 = f32[3,3,9] parameter(3),
    sharding={replicated metadata={op_name="d"}}
  %updates.0 = f32[3,3,9] copy(%p3)
  %p4 = f32[3,3,9] parameter(4),
    sharding={replicated metadata={op_name="e"}}
  %updates.1 = f32[3,3,9] copy(%p4)
  ROOT %scatter = (f32[2,9],f32[2,9])scatter(%input.0, %input.1, %indices, %updates.0, %updates.1),
      to_apply=add,
      update_window_dims={2},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=2,
      sharding={replicated metadata={op_name="d"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "updates.0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,2,1]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("c")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }

  instruction = FindInstruction(module.get(), "updates.1");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,2,1]0,1,2,3}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("c")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, PartialShardingOnElementwise) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %p0 = f32[2,9] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate metadata={op_name="a"}}
  %p1 = f32[2,9] parameter(1),
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dim_replicate metadata={op_name="b"}}
  %lhs = f32[2,9] copy(%p0)
  %rhs = f32[2,9] copy(%p1)
  %add = f32[2,9] add(%lhs, %rhs)
  ROOT %copy = f32[2,9] copy(%add)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* lhs = FindInstruction(module.get(), "lhs");
  ASSERT_NE(lhs, nullptr);
  EXPECT_THAT(lhs, op::Sharding("{devices=[2,2]0,2,1,3}"));
  auto* rhs = FindInstruction(module.get(), "rhs");
  ASSERT_NE(rhs, nullptr);
  EXPECT_THAT(rhs, op::Sharding("{devices=[2,2]0,2,1,3}"));
  auto* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);
  EXPECT_THAT(add, op::Sharding("{devices=[2,2]0,2,1,3}"));
  for (HloInstruction* instruction : {lhs, rhs, add}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("b"), CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, PartialShardingOnElementwise2) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %p0 = f32[2,9] parameter(0),
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate metadata={op_name="a"}}
  %p1 = f32[2,9] parameter(1),
    sharding={devices=[2,1,4]0,1,4,5,2,3,6,7 last_tile_dim_replicate metadata={op_name="b"}}
  %lhs = f32[2,9] copy(%p0)
  %rhs = f32[2,9] copy(%p1)
  %add = f32[2,9] add(%lhs, %rhs)
  ROOT %copy = f32[2,9] copy(%add)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* lhs = FindInstruction(module.get(), "lhs");
  ASSERT_NE(lhs, nullptr);
  EXPECT_THAT(
      lhs,
      op::Sharding("{devices=[2,2,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  auto* rhs = FindInstruction(module.get(), "rhs");
  ASSERT_NE(rhs, nullptr);
  EXPECT_THAT(
      rhs,
      op::Sharding("{devices=[2,2,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  auto* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);
  EXPECT_THAT(
      add,
      op::Sharding("{devices=[2,2,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(lhs->sharding(),
                ShardingMetadata({CreateMetadata("b"), CreateMetadata("a")}));
    EXPECT_THAT(rhs->sharding(),
                ShardingMetadata({CreateMetadata("b"), CreateMetadata("a")}));
    EXPECT_THAT(add->sharding(),
                ShardingMetadata({CreateMetadata("b"), CreateMetadata("a")}));

  } else {
    for (HloInstruction* instruction : {lhs, rhs}) {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, PartialShardingTransposeForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %transpose {
  %param = f32[7,11,13]{2,1,0} parameter(0),
    sharding={devices=[2,1,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate metadata={op_name="a"}}
  %transpose = f32[11,13,7]{2,1,0} transpose(%param), dimensions={1,2,0}
  ROOT %copy = f32[11,13,7]{2,1,0} copy(%transpose)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "transpose");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding(
          "{devices=[1,2,2,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, PartialShardingTransposeBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %transpose {
  %param = f32[7,11,13]{2,1,0} parameter(0)
  %copy = f32[7,11,13]{2,1,0} copy(%param)
  ROOT %transpose = f32[11,13,7]{2,1,0} transpose(%copy), dimensions={1,2,0},
    sharding={devices=[1,2,2,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate metadata={op_name="a"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "copy");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding(
          "{devices=[2,1,2,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, GatherForwardPassWithBarrier) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[8,1,1,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %shard-barrier-from.0 = s32[8,4,2,2]{3,2,1,0} custom-call(%parameter.0), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %shard-barrier-from.1 = s32[2,8,4]{2,1,0} custom-call(%concatenate.19), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %shard-barrier-from.0,
    s32[2,8,4]{2,1,0} %shard-barrier-from.1), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "gather");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTest, GatherBackwardPassWithBarrier) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %copy.p = s32[8,4,2,2]{3,2,1,0} copy(%parameter.0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %shard-barrier-to = s32[8,4,2,2]{3,2,1,0} custom-call(%copy.p), custom_call_target="ShardBarrierTo", custom_call_has_side_effect=true
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %shard-barrier-to,
    s32[2,8,4]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2},
    sharding={devices=[8,1,1,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* concatenate = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(concatenate, op::Sharding("{devices=[1,8,1]0,1,4,5,2,3,6,7}"));
  auto* copy_p = FindInstruction(module.get(), "copy.p");
  ASSERT_NE(copy_p, nullptr);
  EXPECT_THAT(copy_p, op::Sharding("{replicated}"));
}

TEST_F(ShardingPropagationTest, GatherExplicitBatchDimsFromOperandToResult) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[10,3,14,4] parameter(0), sharding={devices=[2,2,2,2]<=[16]}
  %indices = s32[14,10,6,2] parameter(1)
  ROOT %gather = f32[14,10,6,4] gather(%input, %indices), offset_dims={3},
    collapsed_slice_dims={1}, operand_batching_dims={0,2},
    start_indices_batching_dims={1,0}, start_index_map={1,3},
    index_vector_dim=3, slice_sizes={1,1,1,4}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true,
                          /*allow_spmd_sharding_propagation_to_output=*/{true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{devices=[2,2,1,2,2]<=[2,2,2,2]T(2,0,"
                           "3,1) last_tile_dim_replicate}"));
}

TEST_F(ShardingPropagationTest, GatherExplicitBatchDimsFromIndicesToResult) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[10,3,14,4] parameter(0)
  %indices = s32[14,10,6,2] parameter(1), sharding={devices=[2,2,2,2]<=[16]}
  ROOT %gather = f32[14,10,6,4] gather(%input, %indices), offset_dims={3},
    collapsed_slice_dims={1}, operand_batching_dims={0,2},
    start_indices_batching_dims={1,0}, start_index_map={1,3},
    index_vector_dim=3, slice_sizes={1,1,1,4}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true,
                          /*allow_spmd_sharding_propagation_to_output=*/{true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Sharding("{devices=[2,2,2,1,2]<=[16] last_tile_dim_replicate}"));
}

TEST_F(ShardingPropagationTest, GatherBackwardWithExplicitBatchDims1) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %input = f32[10,3,14,4] parameter(0)
  %indices = s32[14,10,6,2] parameter(1)
  ROOT %gather = f32[14,10,6,4] gather(%input, %indices), offset_dims={3},
    collapsed_slice_dims={1}, operand_batching_dims={0,2},
    start_indices_batching_dims={1,0}, start_index_map={1,3},
    index_vector_dim=3, slice_sizes={1,1,1,4},
    sharding={devices=[2,2,2,2]<=[16]}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true, true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);

  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{devices=[2,1,2,2,2]<=[2,2,2,2]T(1,0,3,2) "
                           "last_tile_dim_replicate}"));
  EXPECT_THAT(
      module->entry_computation()->parameter_instruction(1),
      op::Sharding("{devices=[2,2,2,1,2]<=[16] last_tile_dim_replicate}"));
}

TEST_F(ShardingPropagationTest, GatherBackwardWithExplicitBatchDims2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %operand = bf16[32,32] parameter(0)
  %iota = s32[32,1,1] iota(), iota_dimension=0
  ROOT %gather = bf16[32,1] gather(%operand, %iota), offset_dims={},
    collapsed_slice_dims={1}, start_index_map={1}, operand_batching_dims={0},
    start_indices_batching_dims={0}, index_vector_dim=2, slice_sizes={1,1},
    sharding={devices=[2,2]<=[4]}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);

  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{devices=[2,1,2]<=[4] last_tile_dim_replicate}"));
  EXPECT_THAT(module->entry_computation()->root_instruction()->operand(1),
              op::Sharding("{devices=[2,2,1]<=[4]}"));
}

TEST_F(ShardingPropagationTest, ScatterExplicitBatchDimsFromOperandToResult) {
  const char* const hlo_string = R"(
HloModule module

min (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT min = f32[] minimum(lhs, rhs)
}

ENTRY entry {
  %input = f32[10,6,14,4] parameter(0), sharding={devices=[2,2,2,2]<=[16]}
  %indices = s32[14,10,6,2] parameter(1)
  %updates = f32[14,10,6,2] parameter(2)
  ROOT %scatter = f32[10,6,14,4] scatter(%input, %indices, %updates),
    to_apply=min, update_window_dims={3}, inserted_window_dims={1},
    scatter_dims_to_operand_dims={1,3}, input_batching_dims={0,2},
    scatter_indices_batching_dims={1,0}, index_vector_dim=3
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true,
                          /*allow_spmd_sharding_propagation_to_output=*/{true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{devices=[2,2,2,2]<=[16]}"));
}

TEST_F(ShardingPropagationTest, ScatterExplicitBatchDimsFromIndicesToResult) {
  const char* const hlo_string = R"(
HloModule module

min (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT min = f32[] minimum(lhs, rhs)
}

ENTRY entry {
  %input = f32[10,6,14,4] parameter(0)
  %indices = s32[14,10,6,2] parameter(1), sharding={devices=[2,2,2,2]<=[16]}
  %updates = f32[14,10,6,2] parameter(2)
  ROOT %scatter = f32[10,6,14,4] scatter(%input, %indices, %updates),
    to_apply=min, update_window_dims={3}, inserted_window_dims={1},
    scatter_dims_to_operand_dims={1,3}, input_batching_dims={0,2},
    scatter_indices_batching_dims={1,0}, index_vector_dim=3
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true,
                          /*allow_spmd_sharding_propagation_to_output=*/{true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Sharding(
          "{devices=[2,1,2,1,4]<=[2,2,4]T(1,0,2) last_tile_dim_replicate}"));
}

TEST_F(ShardingPropagationTest, ScatterExplicitBatchDimsFromUpdatesToResult) {
  const char* const hlo_string = R"(
HloModule module

min (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT min = f32[] minimum(lhs, rhs)
}

ENTRY entry {
  %input = f32[10,6,14,4] parameter(0)
  %indices = s32[14,10,6,2] parameter(1)
  %updates = f32[14,10,6,4] parameter(2), sharding={devices=[2,2,2,2]<=[16]}
  ROOT %scatter = f32[10,6,14,4] scatter(%input, %indices, %updates),
    to_apply=min, update_window_dims={3}, inserted_window_dims={1},
    scatter_dims_to_operand_dims={1,3}, input_batching_dims={0,2},
    scatter_indices_batching_dims={1,0}, index_vector_dim=3
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true,
                          /*allow_spmd_sharding_propagation_to_output=*/{true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{devices=[2,1,2,2,2]<=[2,2,2,2]T(1,0,3,2) "
                           "last_tile_dim_replicate}"));
}

TEST_F(ShardingPropagationTest, ScatterBackwardWithExplicitBatchDims) {
  const char* const hlo_string = R"(
HloModule module

min (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT min = f32[] minimum(lhs, rhs)
}

ENTRY entry {
  %input = f32[10,6,14,4] parameter(0)
  %indices = s32[14,10,6,2] parameter(1)
  %updates = f32[14,10,6,4] parameter(2)
  ROOT %scatter = f32[10,6,14,4] scatter(%input, %indices, %updates),
    to_apply=min, update_window_dims={3}, inserted_window_dims={1},
    scatter_dims_to_operand_dims={1,3}, input_batching_dims={0,2},
    scatter_indices_batching_dims={1,0}, index_vector_dim=3, sharding={devices=[2,2,2,2]<=[16]}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true, true, true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);

  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{devices=[2,2,2,2]<=[16]}"));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{devices=[2,2,1,1,4]<=[2,2,2,2]T(2,0,1,3) "
                           "last_tile_dim_replicate}"));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(2),
              op::Sharding("{devices=[2,2,1,2,2]<=[2,2,2,2]T(2,0,3,1) "
                           "last_tile_dim_replicate}"));
}

TEST_P(ParameterizedMetadataTest, ParallelGatherFromOperandForwardPass) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[8,1,1,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "gather");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[8,1,1,1]0,1,4,5,2,3,6,7}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ParallelGatherFromIndexForwardPass) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "gather");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[8,1,1,1]0,1,4,5,2,3,6,7}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ParallelGatherBackwardPass) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %copy.p = s32[8,4,2,2]{3,2,1,0} copy(%parameter.0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %copy.p,
    s32[2,8,4]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2},
    sharding={devices=[8,1,1,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* concatenate = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(concatenate, op::Sharding("{devices=[1,8,1]0,1,4,5,2,3,6,7}"));
  auto* copy_p = FindInstruction(module.get(), "copy.p");
  ASSERT_NE(copy_p, nullptr);
  EXPECT_THAT(copy_p, op::Sharding("{devices=[8,1,1,1]0,1,4,5,2,3,6,7}"));
  for (HloInstruction* instruction : {concatenate, copy_p}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, ParallelGatherBackwardPass2) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[4,8,2,2]{3,2,1,0} parameter(0)
  %copy.p = s32[4,8,2,2]{3,2,1,0} copy(%parameter.0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[4,8,2,2]{3,2,1,0} %copy.p,
    s32[2,8,4]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={1,0}, index_vector_dim=0,
    slice_sizes={1,1,2,2},
    sharding={devices=[1,4,1,1]0,1,4,5 metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* concatenate = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(concatenate, op::Sharding("{devices=[1,1,4]0,1,4,5}"));
  auto* copy_p = FindInstruction(module.get(), "copy.p");
  ASSERT_NE(copy_p, nullptr);
  EXPECT_THAT(copy_p, op::Sharding("{devices=[4,1,1,1]0,1,4,5}"));
  for (HloInstruction* instruction : {concatenate, copy_p}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       PartialShardingParallelGatherFromOperandForwardPass) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate metadata={op_name="a"}}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "gather");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding(
          "{devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest,
       PartialShardingParallelGatherFromIndexForwardPass) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,4,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate metadata={op_name="a"}}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate.19 = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate.19), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "gather");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding(
          "{devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, PartialShardingParallelGatherBackwardPass) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %copy.p = s32[8,4,2,2]{3,2,1,0} copy(%parameter.0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %copy.p,
    s32[2,8,4]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2},
    sharding={devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* concatenate = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(
      concatenate,
      op::Sharding(
          "{devices=[1,4,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  auto* copy_p = FindInstruction(module.get(), "copy.p");
  ASSERT_NE(copy_p, nullptr);
  EXPECT_THAT(
      copy_p,
      op::Sharding(
          "{devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  for (HloInstruction* instruction : {concatenate, copy_p}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, PartialShardingParallelGatherBackwardPass2) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[4,8,2,2]{3,2,1,0} parameter(0)
  %copy.p = s32[4,8,2,2]{3,2,1,0} copy(%parameter.0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[4,8,2,2]{3,2,1,0} %copy.p,
    s32[2,8,4]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={1,0}, index_vector_dim=0,
    slice_sizes={1,1,2,2},
    sharding={devices=[1,2,1,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* concatenate = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(
      concatenate,
      op::Sharding("{devices=[1,1,2,2]0,1,4,5 last_tile_dim_replicate}"));
  auto* copy_p = FindInstruction(module.get(), "copy.p");
  ASSERT_NE(copy_p, nullptr);
  EXPECT_THAT(
      copy_p,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  for (HloInstruction* instruction : {concatenate, copy_p}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, ScatterForwardPassWithBarrier) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[8,1,1,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1)
  %shard-barrier-from.0 = s32[8,4,2,2]{3,2,1,0} custom-call(%parameter.0), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %shard-barrier-from.1 = s32[2,8,4]{2,1,0} custom-call(%concatenate), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %shard-barrier-from.2 = s32[8,4,2,2]{3,2,1,0} custom-call(%parameter.1), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %shard-barrier-from.0,
    s32[2,8,4]{2,1,0} %shard-barrier-from.1,
    s32[8,4,2,2]{3,2,1,0} %shard-barrier-from.2),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(instruction->has_sharding());
}

TEST_P(ParameterizedMetadataTest, ScatterBackwardPassWithBarrier) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %copy.p0 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1)
  %copy.p1 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.1)
  %shard-barrier-to.0 = s32[8,4,2,2]{3,2,1,0} custom-call(%copy.p0), custom_call_target="ShardBarrierTo", custom_call_has_side_effect=true
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %shard-barrier-to.0,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %copy.p1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0,
    sharding={devices=[8,1,1,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* concatenate = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(concatenate, op::Sharding("{devices=[1,8,1]0,1,4,5,2,3,6,7}"));
  auto* copy_p0 = FindInstruction(module.get(), "copy.p0");
  ASSERT_NE(copy_p0, nullptr);
  EXPECT_THAT(copy_p0, op::Sharding("{replicated}"));
  auto* copy_p1 = FindInstruction(module.get(), "copy.p1");
  ASSERT_NE(copy_p1, nullptr);
  EXPECT_THAT(copy_p1, op::Sharding("{devices=[8,1,1,1]0,1,4,5,2,3,6,7}"));
}

TEST_P(ParameterizedMetadataTest, ParallelScatterFromOperandForwardPass) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[8,1,1,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[8,1,1,1]0,1,4,5,2,3,6,7}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ParallelScatterFromIndexForwardPass) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,8,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[8,1,1,1]0,1,4,5,2,3,6,7}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ParallelScatterFromUpdateForwardPass) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1),
    sharding={devices=[8,1,1,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}}
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[8,1,1,1]0,1,4,5,2,3,6,7}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ParallelScatterBackwardPass) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %copy.p0 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1)
  %copy.p1 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.1)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %copy.p0,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %copy.p1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0,
    sharding={devices=[8,1,1,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* concatenate = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(concatenate, op::Sharding("{devices=[1,8,1]0,1,4,5,2,3,6,7}"));
  auto* copy_p0 = FindInstruction(module.get(), "copy.p0");
  ASSERT_NE(copy_p0, nullptr);
  EXPECT_THAT(copy_p0, op::Sharding("{devices=[8,1,1,1]0,1,4,5,2,3,6,7}"));
  auto* copy_p1 = FindInstruction(module.get(), "copy.p1");
  ASSERT_NE(copy_p1, nullptr);
  EXPECT_THAT(copy_p1, op::Sharding("{devices=[8,1,1,1]0,1,4,5,2,3,6,7}"));
  for (HloInstruction* instruction : {concatenate, copy_p0, copy_p1}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, ParallelScatterBackwardPass2) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[4,8,2,2]{3,2,1,0} parameter(0)
  %copy.p0 = s32[4,8,2,2]{3,2,1,0} copy(%parameter.0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1)
  %copy.p1 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.1)
  %scatter = s32[4,8,2,2]{3,2,1,0} scatter(
    s32[4,8,2,2]{3,2,1,0} %copy.p0,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %copy.p1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={1,0},
    index_vector_dim=0,
    sharding={devices=[4,1,1,1]0,1,4,5 metadata={op_name="a"}}
  ROOT %copy = s32[4,8,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* concatenate = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(concatenate, op::Sharding("{devices=[1,1,4]0,1,4,5}"));
  auto* copy_p0 = FindInstruction(module.get(), "copy.p0");
  ASSERT_NE(copy_p0, nullptr);
  EXPECT_THAT(copy_p0, op::Sharding("{devices=[4,1,1,1]0,1,4,5}"));
  auto* copy_p1 = FindInstruction(module.get(), "copy.p1");
  ASSERT_NE(copy_p1, nullptr);
  EXPECT_THAT(copy_p1, op::Sharding("{devices=[1,4,1,1]0,1,4,5}"));
  for (HloInstruction* instruction : {concatenate, copy_p0, copy_p1}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       PartialShardingParallelScatterFromOperandForwardPass) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate metadata={op_name="a"}}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding(
          "{devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest,
       PartialShardingParallelScatterFromIndexForwardPass) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,4,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate metadata={op_name="a"}}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding(
          "{devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest,
       PartialShardingParallelScatterFromUpdateForwardPass) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1),
    sharding={devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate metadata={op_name="a"}}
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %parameter.1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding(
          "{devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, PartialShardingParallelScatterBackwardPass) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %copy.p0 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1)
  %copy.p1 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.1)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %copy.p0,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %copy.p1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0,
    sharding={devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* concatenate = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(
      concatenate,
      op::Sharding(
          "{devices=[1,4,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  auto* copy_p0 = FindInstruction(module.get(), "copy.p0");
  ASSERT_NE(copy_p0, nullptr);
  EXPECT_THAT(
      copy_p0,
      op::Sharding(
          "{devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  auto* copy_p1 = FindInstruction(module.get(), "copy.p1");
  ASSERT_NE(copy_p1, nullptr);
  EXPECT_THAT(
      copy_p1,
      op::Sharding(
          "{devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  for (HloInstruction* instruction : {concatenate, copy_p0, copy_p1}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, PartialShardingParallelScatterBackwardPass2) {
  const char* const hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %parameter.0 = s32[4,8,2,2]{3,2,1,0} parameter(0)
  %copy.p0 = s32[4,8,2,2]{3,2,1,0} copy(%parameter.0)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1)
  %copy.p1 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.1)
  %scatter = s32[4,8,2,2]{3,2,1,0} scatter(
    s32[4,8,2,2]{3,2,1,0} %copy.p0,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %copy.p1),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={1,0},
    index_vector_dim=0,
    sharding={devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="a"}}
  ROOT %copy = s32[4,8,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* concatenate = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(
      concatenate,
      op::Sharding("{devices=[1,1,2,2]0,1,4,5 last_tile_dim_replicate}"));
  auto* copy_p0 = FindInstruction(module.get(), "copy.p0");
  ASSERT_NE(copy_p0, nullptr);
  EXPECT_THAT(
      copy_p0,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  auto* copy_p1 = FindInstruction(module.get(), "copy.p1");
  ASSERT_NE(copy_p1, nullptr);
  EXPECT_THAT(
      copy_p1,
      op::Sharding("{devices=[1,2,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  for (HloInstruction* instruction : {concatenate, copy_p0, copy_p1}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       ParallelScatterFromOperandForwardPass_Variadic) {
  const char* const hlo_string = R"(
HloModule module

add (lhs.0: s32[], lhs.1: s32[], rhs.0: s32[], rhs.1: s32[]) -> (s32[], s32[]) {
  lhs.0 = s32[] parameter(0)
  lhs.1 = s32[] parameter(1)
  rhs.0 = s32[] parameter(2)
  rhs.1 = s32[] parameter(3)
  sum.0 = s32[] add(lhs.0, rhs.0)
  sum.1 = s32[] add(lhs.1, rhs.1)
  ROOT tuple = tuple(sum.0, sum.1)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[8,1,1,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}}
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1),
    sharding={devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate metadata={op_name="b"}}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %parameter.3 = s32[8,4,2,2]{3,2,1,0} parameter(3)
  %scatter = (s32[8,4,2,2]{3,2,1,0},s32[8,4,2,2]{3,2,1,0}) scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[8,4,2,2]{3,2,1,0} %parameter.1,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %parameter.2,
    s32[8,4,2,2]{3,2,1,0} %parameter.3),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = (s32[8,4,2,2]{3,2,1,0},s32[8,4,2,2]{3,2,1,0}) copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{{devices=[8,1,1,1]0,1,4,5,2,3,6,7},{devices=[4,1,"
                           "1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(instruction->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest,
       ParallelScatterFromIndexForwardPass_Variadic) {
  const char* const hlo_string = R"(
HloModule module

add (lhs.0: s32[], lhs.1: s32[], rhs.0: s32[], rhs.1: s32[]) -> (s32[], s32[]) {
  lhs.0 = s32[] parameter(0)
  lhs.1 = s32[] parameter(1)
  rhs.0 = s32[] parameter(2)
  rhs.1 = s32[] parameter(3)
  sum.0 = s32[] add(lhs.0, rhs.0)
  sum.1 = s32[] add(lhs.1, rhs.1)
  ROOT tuple = tuple(sum.0, sum.1)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1,
    sharding={devices=[1,4,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate metadata={op_name="a"}}
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %parameter.3 = s32[8,4,2,2]{3,2,1,0} parameter(3)
  %scatter = (s32[8,4,2,2]{3,2,1,0},s32[8,4,2,2]{3,2,1,0}) scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[8,4,2,2]{3,2,1,0} %parameter.1,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %parameter.2,
    s32[8,4,2,2]{3,2,1,0} %parameter.3),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = (s32[8,4,2,2]{3,2,1,0},s32[8,4,2,2]{3,2,1,0}) copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{{devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 "
                           "last_tile_dim_replicate},{devices=[4,1,1,1,2]0,1,4,"
                           "5,2,3,6,7 last_tile_dim_replicate}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(instruction->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest,
       ParallelScatterFromUpdateForwardPass_Variadic) {
  const char* const hlo_string = R"(
HloModule module

add (lhs.0: s32[], lhs.1: s32[], rhs.0: s32[], rhs.1: s32[]) -> (s32[], s32[]) {
  lhs.0 = s32[] parameter(0)
  lhs.1 = s32[] parameter(1)
  rhs.0 = s32[] parameter(2)
  rhs.1 = s32[] parameter(3)
  sum.0 = s32[] add(lhs.0, rhs.0)
  sum.1 = s32[] add(lhs.1, rhs.1)
  ROOT tuple = tuple(sum.0, sum.1)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.2 = s32[8,4,2,2]{3,2,1,0} parameter(2),
    sharding={devices=[1,8,1,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}}
  %parameter.3 = s32[8,4,2,2]{3,2,1,0} parameter(3),
    sharding={devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate metadata={op_name="b"}}
  %scatter = (s32[8,4,2,2]{3,2,1,0},s32[8,4,2,2]{3,2,1,0}) scatter(
    s32[8,4,2,2]{3,2,1,0} %parameter.0,
    s32[8,4,2,2]{3,2,1,0} %parameter.1,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %parameter.2,
    s32[8,4,2,2]{3,2,1,0} %parameter.3),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = (s32[8,4,2,2]{3,2,1,0},s32[8,4,2,2]{3,2,1,0}) copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "scatter");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{{devices=[1,8,1,1]0,1,4,5,2,3,6,7},{devices=[4,1,"
                           "1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding().tuple_elements()[0],
                ShardingMetadata({CreateMetadata("a")}));
    EXPECT_THAT(instruction->sharding().tuple_elements()[1],
                ShardingMetadata({CreateMetadata("b")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ParallelScatterBackwardPass_Variadic) {
  const char* const hlo_string = R"(
HloModule module

add (lhs.0: s32[], lhs.1: s32[], rhs.0: s32[], rhs.1: s32[]) -> (s32[], s32[]) {
  lhs.0 = s32[] parameter(0)
  lhs.1 = s32[] parameter(1)
  rhs.0 = s32[] parameter(2)
  rhs.1 = s32[] parameter(3)
  sum.0 = s32[] add(lhs.0, rhs.0)
  sum.1 = s32[] add(lhs.1, rhs.1)
  ROOT tuple = tuple(sum.0, sum.1)
}

ENTRY %module {
  %parameter.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %copy.p0 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.0)
  %parameter.1 = s32[8,4,2,2]{3,2,1,0} parameter(1)
  %copy.p1 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %copy.p2 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.2)
  %parameter.3 = s32[8,4,2,2]{3,2,1,0} parameter(3)
  %copy.p3 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.3)
  %scatter = (s32[8,4,2,2]{3,2,1,0},s32[8,4,2,2]{3,2,1,0}) scatter(
    s32[8,4,2,2]{3,2,1,0} %copy.p0,
    s32[8,4,2,2]{3,2,1,0} %copy.p1,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %copy.p2,
    s32[8,4,2,2]{3,2,1,0} %copy.p3),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0,
    sharding={{devices=[8,1,1,1]0,1,4,5,2,3,6,7 metadata={op_name="a"}},
              {devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate metadata={op_name="b"}}}
  ROOT %copy = (s32[8,4,2,2]{3,2,1,0},s32[8,4,2,2]{3,2,1,0}) copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* concatenate = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(concatenate, op::Sharding("{devices=[1,8,1]0,1,4,5,2,3,6,7}"));
  auto* copy_p0 = FindInstruction(module.get(), "copy.p0");
  ASSERT_NE(copy_p0, nullptr);
  EXPECT_THAT(copy_p0, op::Sharding("{devices=[8,1,1,1]0,1,4,5,2,3,6,7}"));
  auto* copy_p1 = FindInstruction(module.get(), "copy.p1");
  ASSERT_NE(copy_p1, nullptr);
  EXPECT_THAT(
      copy_p1,
      op::Sharding(
          "{devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  auto* copy_p2 = FindInstruction(module.get(), "copy.p2");
  ASSERT_NE(copy_p2, nullptr);
  EXPECT_THAT(copy_p2, op::Sharding("{devices=[8,1,1,1]0,1,4,5,2,3,6,7}"));
  auto* copy_p3 = FindInstruction(module.get(), "copy.p3");
  ASSERT_NE(copy_p3, nullptr);
  EXPECT_THAT(
      copy_p3,
      op::Sharding(
          "{devices=[4,1,1,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
  for (HloInstruction* instruction : {concatenate, copy_p0, copy_p2}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
  for (HloInstruction* instruction : {copy_p1, copy_p3}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("b")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, ParallelScatterBackwardPass2_Variadic) {
  const char* const hlo_string = R"(
HloModule module

add (lhs.0: s32[], lhs.1: s32[], rhs.0: s32[], rhs.1: s32[]) -> (s32[], s32[]) {
  lhs.0 = s32[] parameter(0)
  lhs.1 = s32[] parameter(1)
  rhs.0 = s32[] parameter(2)
  rhs.1 = s32[] parameter(3)
  sum.0 = s32[] add(lhs.0, rhs.0)
  sum.1 = s32[] add(lhs.1, rhs.1)
  ROOT tuple = tuple(sum.0, sum.1)
}

ENTRY %module {
  %parameter.0 = s32[4,8,2,2]{3,2,1,0} parameter(0)
  %copy.p0 = s32[4,8,2,2]{3,2,1,0} copy(%parameter.0)
  %parameter.1 = s32[4,8,2,2]{3,2,1,0} parameter(1)
  %copy.p1 = s32[4,8,2,2]{3,2,1,0} copy(%parameter.1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %iota2 = s32[1,8,4]{2,1,0} iota(), iota_dimension=2
  %concatenate = s32[2,8,4]{2,1,0} concatenate(s32[1,8,4]{2,1,0} %iota,
    s32[1,8,4]{2,1,0} %iota2), dimensions={0}
  %parameter.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %copy.p2 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.2)
  %parameter.3 = s32[8,4,2,2]{3,2,1,0} parameter(3)
  %copy.p3 = s32[8,4,2,2]{3,2,1,0} copy(%parameter.3)
  %scatter = (s32[4,8,2,2]{3,2,1,0},s32[4,8,2,2]{3,2,1,0}) scatter(
    s32[4,8,2,2]{3,2,1,0} %copy.p0,
    s32[4,8,2,2]{3,2,1,0} %copy.p1,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %copy.p2,
    s32[8,4,2,2]{3,2,1,0} %copy.p3),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={1,0},
    index_vector_dim=0,
    sharding={{devices=[4,1,1,1]0,1,4,5 metadata={op_name="a"}},
              {devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="b"}}}
  ROOT %copy = (s32[4,8,2,2]{3,2,1,0},s32[4,8,2,2]{3,2,1,0}) copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* concatenate = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(concatenate, op::Sharding("{devices=[1,1,4]0,1,4,5}"));
  auto* copy_p0 = FindInstruction(module.get(), "copy.p0");
  ASSERT_NE(copy_p0, nullptr);
  EXPECT_THAT(copy_p0, op::Sharding("{devices=[4,1,1,1]0,1,4,5}"));
  auto* copy_p1 = FindInstruction(module.get(), "copy.p1");
  ASSERT_NE(copy_p1, nullptr);
  EXPECT_THAT(
      copy_p1,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  auto* copy_p2 = FindInstruction(module.get(), "copy.p2");
  ASSERT_NE(copy_p2, nullptr);
  EXPECT_THAT(copy_p2, op::Sharding("{devices=[1,4,1,1]0,1,4,5}"));
  auto* copy_p3 = FindInstruction(module.get(), "copy.p3");
  ASSERT_NE(copy_p3, nullptr);
  EXPECT_THAT(
      copy_p3,
      op::Sharding("{devices=[1,2,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  for (HloInstruction* instruction : {concatenate, copy_p0, copy_p2}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
  for (HloInstruction* instruction : {copy_p1, copy_p3}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("b")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       GatherMergedIndexParallelAndOperandPassthroughFromOperandForwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0),
    sharding={devices=[2,1,2,1]0,1,4,5 metadata={op_name="a"}}
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[2,1,2,1]0,1,4,5 }"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(gather, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));

  for (const HloInstruction* instruction : {operand, indices, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       GatherMergedIndexParallelAndOperandPassthroughBackwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2},
    sharding={devices=[2,1,2,1]0,1,4,5 metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[2,1,2,1]0,1,4,5 }"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(gather, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));

  for (const HloInstruction* instruction : {operand, indices, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       GatherMergedIndexParallelAndIndexPassthroughFromIndicesForwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1),
    sharding={devices=[1,2,2]0,1,4,5 metadata={op_name="a"}}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(indices, op::Sharding("{devices=[1,2,2]0,1,4,5}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(gather, op::Sharding("{devices=[2,2,1,1]0,1,4,5}"));

  for (const HloInstruction* instruction : {operand, indices, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       GatherMergedIndexParallelAndIndexPassthroughBackwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2},
    sharding={devices=[2,2,1,1]0,1,4,5 metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(indices, op::Sharding("{devices=[1,2,2]0,1,4,5}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(gather, op::Sharding("{devices=[2,2,1,1]0,1,4,5}"));

  for (const HloInstruction* instruction : {operand, indices, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       GatherMergedIndexParallelAndTrivialSlicedOperandFromOperandForwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0),
    sharding={devices=[2,2,1,1]0,1,4,5 metadata={op_name="a"}}
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[2,2,1,1]0,1,4,5 }"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(
      gather,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction : {operand, indices, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       GatherMergedIndexParallelAndTrivialSlicedOperandBackwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate metadata={op_name="a"}}
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2},
    sharding={devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[2,2,1,1]0,1,4,5}"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(
      gather,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction : {operand, indices, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(
    ParameterizedMetadataTest,
    GatherMergedOperandPassthroughAndTrivialSlicedOperandFromOperandForwardPass) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0),
    sharding={devices=[1,2,2,1]0,4,1,5 metadata={op_name="a"}}
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1)
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[1,2,2,1]0,4,1,5}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(indices, op::Sharding("{replicated}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(
      gather,
      op::Sharding("{devices=[1,1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction : {operand, indices, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       GatherMergedOperandPassthroughAndTrivialSlicedOperandBackwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate metadata={op_name="a"}}
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1)
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2},
    sharding={devices=[1,1,2,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[1,2,2,1]0,4,1,5}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(indices, op::Sharding("{replicated}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(
      gather,
      op::Sharding("{devices=[1,1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction : {operand, indices, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       GatherMergedOperandAndIndexPassthroughFromOperandAndIndexForwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0),
    sharding={devices=[1,1,2,1,2]0,4,1,5 last_tile_dim_replicate metadata={op_name="a"}}
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1),
    sharding={devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="a"}}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[1,1,2,1,2]0,4,1,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(gather, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));

  for (const HloInstruction* instruction : {operand, indices, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       GatherMergedOperandPassthroughAndIndexPassthroughBackwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1)
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2},
    sharding={devices=[2,1,2,1]0,1,4,5 metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[1,1,2,1,2]0,4,1,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(gather, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));

  for (const HloInstruction* instruction : {operand, indices, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(
    ParameterizedMetadataTest,
    GatherMergedTrivialSlicedOperandAndIndexPassthroughFromOperandAndIndexForwardPass) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0),
    sharding={devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate metadata={op_name="a"}}
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1),
    sharding={devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="a"}}
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(
      gather,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction : {operand, indices, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       GatherMergedTrivialSlicedOperandAndIndexPassthroughBackwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate metadata={op_name="a"}}
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1)
  %gather = s32[8,4,2,2]{3,2,1,0} gather(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2},
    sharding={devices=[1,1,2,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[1,2,2,1]0,4,1,5}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(indices, op::Sharding("{replicated}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(
      gather,
      op::Sharding("{devices=[1,1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction : {operand, indices, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       ScatterMergedIndexParallelAndOperandPassthroughFromOperandForwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0),
    sharding={devices=[2,1,2,1]0,1,4,5 metadata={op_name="a"}}
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[2,1,2,1]0,1,4,5 }"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(update, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(scatter, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       ScatterMergedIndexParallelAndOperandPassthroughFromUpdateForwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2),
    sharding={devices=[2,1,2,1]0,1,4,5 metadata={op_name="a"}}
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[2,1,2,1]0,1,4,5 }"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(update, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(scatter, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       ScatterMergedIndexParallelAndOperandPassthroughBackwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0,
    sharding={devices=[2,1,2,1]0,1,4,5 metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[2,1,2,1]0,1,4,5 }"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(update, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(scatter, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(
    ParameterizedMetadataTest,
    ScatterMergedIndexParallelAndTrivialSlicedOperandFromOperandForwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0),
    sharding={devices=[2,2,1,1]0,1,4,5 metadata={op_name="a"}}
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[2,2,1,1]0,1,4,5 }"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(
      update,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(scatter, op::Sharding("{devices=[2,2,1,1]0,1,4,5}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       ScatterMergedIndexParallelAndTrivialSlicedOperandBackwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0,
    sharding={devices=[2,2,1,1]0,1,4,5 metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[2,2,1,1]0,1,4,5 }"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(
      update,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(scatter, op::Sharding("{devices=[2,2,1,1]0,1,4,5}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       ScatterMergedIndexParallelAndIndexPassthroughFromIndexForwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1),
    sharding={devices=[1,2,2]0,1,4,5 metadata={op_name="a"}}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(indices, op::Sharding("{devices=[1,2,2]0,1,4,5}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(update, op::Sharding("{devices=[2,2,1,1]0,1,4,5 }"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(
      scatter,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       ScatterMergedIndexParallelAndIndexPassthroughFromUpdateForwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1)
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2),
    sharding={devices=[2,2,1,1]0,1,4,5 metadata={op_name="a"}}
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(indices, op::Sharding("{devices=[1,2,2]0,1,4,5}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(update, op::Sharding("{devices=[2,2,1,1]0,1,4,5 }"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(
      scatter,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       ScatterMergedIndexParallelAndIndexPassthroughBackwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[1,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[1,8,4]{2,1,0} copy(s32[1,8,4]{2,1,0} %arg.1),
    sharding={devices=[1,1,2,2]0,4,1,5 last_tile_dim_replicate metadata={op_name="a"}}
  %iota = s32[1,8,4]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,8,4]{2,1,0} concatenate(
    s32[1,8,4]{2,1,0} %iota, s32[1,8,4]{2,1,0} %indices), dimensions={0}
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %concatenate,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0,
    sharding={devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "concatenate");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(indices, op::Sharding("{devices=[1,2,2]0,1,4,5}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(update, op::Sharding("{devices=[2,2,1,1]0,1,4,5 }"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(
      scatter,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(
    ParameterizedMetadataTest,
    ScatterMergedOperandPassthroughAndTrivialSlicedOperandFromOperandForwardPass) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0),
    sharding={devices=[1,2,2,1]0,1,4,5 metadata={op_name="a"}}
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1)
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[1,2,2,1]0,1,4,5}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(indices, op::Sharding("{replicated}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(
      update,
      op::Sharding("{devices=[1,1,2,1,2]0,4,1,5 last_tile_dim_replicate}"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(scatter, op::Sharding("{devices=[1,2,2,1]0,1,4,5}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       ScatterMergedOperandPassthroughAndTrivialSlicedOperandBackwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1)
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0,
    sharding={devices=[1,2,2,1]0,1,4,5 metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand, op::Sharding("{devices=[1,2,2,1]0,1,4,5}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(indices, op::Sharding("{replicated}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(
      update,
      op::Sharding("{devices=[1,1,2,1,2]0,4,1,5 last_tile_dim_replicate}"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(scatter, op::Sharding("{devices=[1,2,2,1]0,1,4,5}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       ScatterMergedOperandAndIndexPassthroughFromOperandAndIndexForwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0),
    sharding={devices=[1,1,2,1,2]0,4,1,5 last_tile_dim_replicate metadata={op_name="a"}}
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1),
    sharding={devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="a"}}
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[1,1,2,1,2]0,4,1,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(update, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(
      scatter,
      op::Sharding("{devices=[1,1,2,1,2]0,4,1,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(
    ParameterizedMetadataTest,
    ScatterMergedOperandPassthroughAndIndexPassthroughFromUpdateForwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1)
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2),
    sharding={devices=[2,1,2,1]0,1,4,5 metadata={op_name="a"}}
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[1,1,2,1,2]0,4,1,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(update, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(
      scatter,
      op::Sharding("{devices=[1,1,2,1,2]0,4,1,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       ScatterMergedOperandPassthroughAndIndexPassthroughBackwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1),
    sharding={devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="a"}}
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0,
    sharding={devices=[1,1,2,1,2]0,4,1,5 last_tile_dim_replicate metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[1,1,2,1,2]0,4,1,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(update, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(
      scatter,
      op::Sharding("{devices=[1,1,2,1,2]0,4,1,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(
    ParameterizedMetadataTest,
    ScatterMergedTrivialSlicedOperandAndIndexPassthroughFromOperandAndIndexForwardPass) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0),
    sharding={devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate metadata={op_name="a"}}
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1),
    sharding={devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="a"}}
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(
      update,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(
      scatter,
      op::Sharding("{devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(
    ParameterizedMetadataTest,
    ScatterMergedTrivialSlicedOperandAndIndexPassthroughFromOperandAndUpdateForwardPass) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0),
    sharding={devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate metadata={op_name="a"}}
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1)
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2),
    sharding={devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="a"}}
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(
      update,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(
      scatter,
      op::Sharding("{devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       ScatterMergedTrivialSlicedOperandAndIndexPassthroughBackwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

add (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY %module {
  %arg.0 = s32[8,4,2,2]{3,2,1,0} parameter(0)
  %arg.1 =  s32[2,8,4]{2,1,0} parameter(1)
  %arg.2 = s32[8,4,2,2]{3,2,1,0} parameter(2)
  %operand = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.0)
  %indices = s32[2,8,4]{2,1,0} copy(s32[2,8,4]{2,1,0} %arg.1),
    sharding={devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate metadata={op_name="a"}}
  %update = s32[8,4,2,2]{3,2,1,0} copy(s32[8,4,2,2]{3,2,1,0} %arg.2)
  %scatter = s32[8,4,2,2]{3,2,1,0} scatter(
    s32[8,4,2,2]{3,2,1,0} %operand,
    s32[2,8,4]{2,1,0} %indices,
    s32[8,4,2,2]{3,2,1,0} %update),
    to_apply=add,
    update_window_dims={2,3},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=0,
    sharding={devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate metadata={op_name="a"}}
  ROOT %copy = s32[8,4,2,2]{3,2,1,0} copy(%scatter)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(
      operand,
      op::Sharding("{devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate}"));
  const HloInstruction* indices = FindInstruction(module.get(), "indices");
  ASSERT_NE(indices, nullptr);
  EXPECT_THAT(
      indices,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(
      update,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(
      scatter,
      op::Sharding("{devices=[1,2,1,1,2]0,4,1,5 last_tile_dim_replicate}"));

  for (const HloInstruction* instruction :
       {operand, indices, update, scatter}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, CorrectlyReplicateGatherIndex) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = bf16[1,2,2,2,8]{4,3,2,1,0} parameter(0)
  %parameter.1 = s32[1,2,2]{2,1,0} parameter(1)
  %index = s32[1,2,2]{2,1,0} copy(%parameter.1)
  %gather = bf16[1,2,2,2,8]{4,3,2,1,0} gather(
    bf16[1,2,2,2,8]{4,3,2,1,0} %parameter.0, s32[1,2,2]{2,1,0} %index),
    offset_dims={2,3,4}, collapsed_slice_dims={0,1}, start_index_map={0,1},
    index_vector_dim=2, slice_sizes={1,1,2,2,8},
    sharding={devices=[1,1,2,1,1]0,1 metadata={op_name="a"}}
  ROOT %copy = bf16[1,2,2,2,8]{4,3,2,1,0} copy(%gather)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);

  const HloInstruction* index = FindInstruction(module.get(), "index");

  ASSERT_NE(index, nullptr);

  EXPECT_THAT(index, op::Sharding("{replicated}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(index->sharding(), ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(index->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, GatherToOperand_ParallelDimIsNotPartitioned) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %parameter.0 = s32[2,1000,1]{2,1,0} parameter(0)
  %parameter.1 = bf16[2,4819,4]{2,1,0} parameter(1)
  %iota = s32[2,1000,1]{1,0,2} iota(), iota_dimension=0
  %operand = bf16[2,4819,4]{2,1,0} copy(%parameter.1)
  %index = s32[2,1000,2]{2,1,0} concatenate(s32[2,1000,1]{1,0,2} %iota,
    s32[2,1000,1]{2,1,0} %parameter.0), dimensions={2},
    sharding={devices=[1,4,1]0,1,2,3}
  ROOT %gather = bf16[2,1000,4]{2,1,0} gather(bf16[2,4819,4]{2,1,0} %operand,
    s32[2,1000,2]{2,1,0} %index), offset_dims={2},
    collapsed_slice_dims={0,1}, start_index_map={0,1},
    index_vector_dim=2, slice_sizes={1,1,4},
    sharding={devices=[1,4,1]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);

  const HloInstruction* operand = FindInstruction(module.get(), "operand");
  EXPECT_THAT(operand, op::Sharding("{replicated}"));
}

TEST_P(ParameterizedMetadataTest, ManualSubgroupForward) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[6,3]{1,0} parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dims={manual} metadata={op_name="a"}}
  %copy = f32[6,3]{1,0} copy(%param0)
  %param1 = f32[6,3]{1,0} parameter(1),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dims={manual} metadata={op_name="a"}}
  %copy.1 = f32[6,3]{1,0} copy(%param1)
  %add = f32[6,3]{1,0} add(%copy, %copy.1)
  ROOT %copy.2 = f32[6,3]{1,0} copy(%add)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "add");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ManualSubgroup_SingleOperandHasSharding) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[6,3]{1,0} parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dims={manual} metadata={op_name="a"}}
  %copy = f32[6,3]{1,0} copy(%param0)
  %param1 = f32[6,3]{1,0} parameter(1)
  %copy.1 = f32[6,3]{1,0} copy(%param1)
  %add = f32[6,3]{1,0} add(%copy, %copy.1)
  ROOT %copy.2 = f32[6,3]{1,0} copy(%add)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "add");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }

  // Check other operand's sharding
  auto* operand = FindInstruction(module.get(), "copy");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(operand->sharding(), ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(operand->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ManualSubgroup_OneOperandReplicate) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[6,3]{1,0} parameter(0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dims={manual} metadata={op_name="a"}}
  %copy = f32[6,3]{1,0} copy(%param0)
  %param1 = f32[6,3]{1,0} parameter(1),
    sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dims={replicated, manual} metadata={op_name="a"}}
  %copy.1 = f32[6,3]{1,0} copy(%param1)
  %add = f32[6,3]{1,0} add(%copy, %copy.1)
  ROOT %copy.2 = f32[6,3]{1,0} copy(%add)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "add");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }

  // Check other operand's sharding
  auto* operand = FindInstruction(module.get(), "copy");
  ASSERT_NE(operand, nullptr);
  EXPECT_THAT(operand,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(operand->sharding(), ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(operand->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, ManualSubgroupBackward) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[6,3]{1,0} parameter(0)
  %copy = f32[6,3]{1,0} copy(%param0)
  %param1 = f32[6,3]{1,0} parameter(1)
  %copy.1 = f32[6,3]{1,0} copy(%param1)
  %add = f32[6,3]{1,0} add(%copy, %copy.1),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dims={manual} metadata={op_name="a"}}
  ROOT %copy.2 = f32[6,3]{1,0} copy(%add)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "copy");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_F(ShardingPropagationTest, SimpleManual) {
  const char* const hlo_string = R"(
HloModule module

%add {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}

ENTRY %entry {
  %param0 = f32[6,3] parameter(0)
  %copy = f32[6,3] copy(%param0), sharding={devices=[2,1]0,1}
  %annotate = f32[6,3] custom-call(%copy), custom_call_target="Sharding",
    sharding={devices=[2,1]0,1}
  %to_manual = f32[3,3] custom-call(%annotate),
    custom_call_target="SPMDFullToShardShape", sharding={manual}
  %zero = f32[] constant(0)
  %reduce = f32[3] reduce(%to_manual, %zero), dimensions={1}, to_apply=%add
  %annotate2 = f32[3] custom-call(%reduce), custom_call_target="Sharding",
    sharding={manual}
  %to_auto = f32[6] custom-call(%annotate2),
    custom_call_target="SPMDShardToFullShape", sharding={devices=[2]0,1}
  ROOT %copy.2 = f32[6] copy(%to_auto)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "reduce");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{manual}"));
}

TEST_F(ShardingPropagationTest, SimpleManualTuple) {
  const char* const hlo_string = R"(
HloModule module

%add {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}

ENTRY %entry {
  %param0 = f32[6,3] parameter(0)
  %copy = f32[6,3] copy(%param0), sharding={devices=[2,1]0,1}
  %annotate = f32[6,3] custom-call(%copy), custom_call_target="Sharding",
    sharding={devices=[2,1]0,1}
  %to_manual = f32[3,3] custom-call(%annotate),
    custom_call_target="SPMDFullToShardShape", sharding={manual}
  %t = (f32[3,3]) tuple(%to_manual)
  %gte = f32[3,3] get-tuple-element(%t), index=0
  %to_auto = f32[3,3] custom-call(%gte),
    custom_call_target="SPMDShardToFullShape", sharding={devices=[2,1]0,1}
  ROOT %copy.2 = f32[3,3] copy(%to_auto)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "t");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{{manual}}"));
  instruction = FindInstruction(module.get(), "gte");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{manual}"));
}

TEST_F(ShardingPropagationTest, DefaultManualCustomCallForward) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[6,3]{1,0} parameter(0),
    sharding={manual metadata={op_name="a"}}
  %copy = f32[6,3]{1,0} copy(%param0)
  %param1 = f32[6,3]{1,0} parameter(1)
  %copy.1 = f32[6,3]{1,0} copy(%param1)
  %param2 = f32[6,3]{1,0} parameter(2)
  %copy.2 = f32[6,3]{1,0} copy(%param2)
  %custom-call = (f32[], f32[6,3]{1,0}) custom-call(%copy, %copy.1, %copy.2), custom_call_target="some_custom_call"
  ROOT %copy.3 = (f32[], f32[6,3]{1,0}) copy(%custom-call)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "custom-call");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{{manual},{manual}}"));
}

TEST_F(ShardingPropagationTest, RefineUnspecifiedDims) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[6,3] parameter(0)
  %copy = f32[6,3] copy(%param0),
    sharding={devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}
  %annotate = f32[6,3] custom-call(%copy), custom_call_target="Sharding",
    backend_config="unspecified_dims=[1]",
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}
  %copy.2 = f32[6,3] copy(%annotate)
  ROOT %copy.3 = f32[6,3] copy(%copy.2)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "copy.2");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,2]0,2,1,3}"));
}

TEST_F(ShardingPropagationTest, RefineUnspecifiedDimsWithManualConversion) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[6,3,8] parameter(0)
  %copy = f32[6,3,8] copy(%param0),
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %annotate = f32[6,3,8] custom-call(%copy), custom_call_target="Sharding",
    backend_config="unspecified_dims=[1,2]",
    sharding={devices=[2,1,1,4]0,1,4,5,2,3,6,7 last_tile_dim_replicate}
  %to_manual = f32[3,3,8] custom-call(%annotate),
    custom_call_target="SPMDFullToShardShape",
    backend_config="unspecified_dims=[1,2]",
    sharding={devices=[1,1,1,4,2]0,2,1,3,4,6,5,7 last_tile_dims={replicated,manual}}
  %annotate2 = f32[3,3,8] custom-call(%to_manual), custom_call_target="Sharding",
    backend_config="unspecified_dims=[1,2]",
    sharding={devices=[1,1,1,4,2]0,2,1,3,4,6,5,7 last_tile_dims={replicated,manual}}
  %to_auto = f32[6,3,8] custom-call(%annotate2),
    custom_call_target="SPMDShardToFullShape",
    backend_config="unspecified_dims=[1,2]",
    sharding={devices=[2,1,1,4]0,1,4,5,2,3,6,7 last_tile_dim_replicate}
  %copy.2 = f32[6,3,8] copy(%to_auto)
  ROOT %copy.3 = f32[6,3,8] copy(%copy.2),
    sharding={devices=[1,1,2,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* copy2 = FindInstruction(module.get(), "copy.2");
  ASSERT_NE(copy2, nullptr);
  EXPECT_THAT(copy2, op::Sharding("{devices=[2,2,2]0,1,4,5,2,3,6,7}"));
  auto* to_manual = FindInstruction(module.get(), "to_manual");
  ASSERT_NE(to_manual, nullptr);
  EXPECT_THAT(
      to_manual,
      op::Sharding(
          "{devices=[1,2,2,2]0,2,1,3,4,6,5,7 last_tile_dims={manual}}"));
  auto* to_auto = FindInstruction(module.get(), "to_auto");
  ASSERT_NE(to_auto, nullptr);
  EXPECT_THAT(to_auto, op::Sharding("{devices=[2,2,2]0,1,4,5,2,3,6,7}"));
}

TEST_F(ShardingPropagationTest, RefineUnspecifiedDimsWithManualConversion2) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[6,3,8] parameter(0)
  %copy = f32[6,3,8] copy(%param0)
  %annotate1 = f32[6,3,8] custom-call(%copy), custom_call_target="Sharding",
    backend_config="unspecified_dims=[1,2]",
    sharding={devices=[2,1,1,4]0,1,4,5,2,3,6,7 last_tile_dim_replicate}
  %to_manual = f32[3,3,8] custom-call(%annotate1),
    custom_call_target="SPMDFullToShardShape",
    backend_config="unspecified_dims=[1,2]",
    sharding={devices=[1,1,1,4,2]0,2,1,3,4,6,5,7 last_tile_dims={replicated,manual}}
  %annotate2 = f32[3,3,8] custom-call(%to_manual), custom_call_target="Sharding",
    backend_config="unspecified_dims=[1,2]",
    sharding={devices=[1,1,1,4,2]0,2,1,3,4,6,5,7 last_tile_dims={replicated,manual}}
  %annotate3 = f32[3,3,8] custom-call(%annotate2), custom_call_target="Sharding",
    backend_config="unspecified_dims=[1,2]",
    sharding={devices=[1,1,1,4,2]0,2,1,3,4,6,5,7 last_tile_dims={replicated,manual}}
  %to_auto = f32[6,3,8] custom-call(%annotate3),
    custom_call_target="SPMDShardToFullShape",
    backend_config="unspecified_dims=[1,2]",
    sharding={devices=[2,1,1,4]0,1,4,5,2,3,6,7 last_tile_dim_replicate}
  %copy.2 = f32[6,3,8] copy(%to_auto),
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  ROOT %copy.3 = f32[6,3,8] copy(%copy.2)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* copy = FindInstruction(module.get(), "copy");
  ASSERT_NE(copy, nullptr);
  EXPECT_THAT(
      copy, op::Sharding(
                "{devices=[2,2,1,2]0,1,4,5,2,3,6,7 last_tile_dim_replicate}"));
}

TEST_F(ShardingPropagationTest, DoNotRefineUnspecifiedDimsOnManual) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[6,3] parameter(0), sharding={manual}
  %annotate = f32[6,3] custom-call(%param0), custom_call_target="Sharding",
    backend_config="unspecified_dims=[1]", sharding={manual}
  ROOT %copy.2 = f32[6,3] copy(%annotate), sharding={manual}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  // Sharding op is changed to a copy.
  EXPECT_TRUE(changed);
  for (auto* hlo : module->entry_computation()->instructions()) {
    EXPECT_TRUE(hlo->sharding().IsManual());
  }
}

TEST_F(ShardingPropagationTest, DoNotPassManualShardingToSPMDShardToFullShape) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  p.0 = f32[2,3]{1,0} parameter(0), sharding={replicated}
  custom-call.2 = f32[2,3]{1,0} custom-call(p.0), custom_call_target="Sharding", sharding={replicated}
  custom-call.3 = f32[2,3]{1,0} custom-call(custom-call.2), custom_call_target="SPMDFullToShardShape", sharding={manual}
  custom-call.4 = f32[2,3]{1,0} custom-call(custom-call.3), custom_call_target="Sharding", sharding={manual}
  ROOT custom-call.5 = f32[16,3]{1,0} custom-call(custom-call.4), custom_call_target="SPMDShardToFullShape", sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true,
                          /*allow_spmd_sharding_propagation_to_output=*/{true})
          .Run(module.get()));
  // Sharding op is changed to a copy.
  EXPECT_TRUE(changed);
  auto spmd_shard_to_full = module->entry_computation()->root_instruction();
  CHECK(spmd_shard_to_full->IsCustomCall("SPMDShardToFullShape"));
  EXPECT_FALSE(spmd_shard_to_full->sharding().IsManual());
}

TEST_F(ShardingPropagationTest, ManualShardingPassThroughSplitConstant) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  p.0 = f32[2,3]{1,0} parameter(0), sharding={replicated}
  p.1 = f32[2,3]{1,0} parameter(1), sharding={replicated}
  constant = f32[2,3]{1,0} constant({{0,1,2},{3,4,5}})
  custom-call.0 = f32[2,3]{1,0} custom-call(p.0), custom_call_target="Sharding", sharding={replicated}
  custom-call.1 = f32[2,3]{1,0} custom-call(custom-call.0), custom_call_target="SPMDFullToShardShape", sharding={manual}
  add.0 = f32[2,3]{1,0} add(constant, custom-call.1)
  custom-call.2 = f32[2,3]{1,0} custom-call(add.0), custom_call_target="SPMDShardToFullShape", sharding={replicated}
  add.1 = f32[2,3]{1,0} add(constant, p.1)
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3]{1,0}) tuple(custom-call.2, add.1)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool is_split,
      HloConstantSplitter(/*split_expressions=*/true).Run(module.get()));
  EXPECT_TRUE(is_split);
  TF_ASSERT_OK_AND_ASSIGN(auto _, HloDCE().Run(module.get()));
  (void)_;  // Suppress unused variable warning in OSS
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  // Sharding op is changed to a copy.
  EXPECT_TRUE(changed);
  const HloInstruction* add0 = FindInstruction(module.get(), "add.0");
  const HloInstruction* manual_constant = add0->operand(0);
  EXPECT_TRUE(manual_constant->IsConstant() &&
              manual_constant->sharding().IsManual());
  const HloInstruction* add1 = FindInstruction(module.get(), "add.1");
  const HloInstruction* replicate_constant = add1->operand(0);
  EXPECT_TRUE(replicate_constant->IsConstant() &&
              replicate_constant->sharding().IsReplicated());
}

TEST_F(ShardingPropagationTest, ReshapeNoMatchSubgroupManual) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[1,3,3] parameter(0),
    sharding={devices=[2,1,1,2]0,1,2,3 last_tile_dims={manual}}
  %reshape = f32[3,1,3,1] reshape(%param0)
  ROOT %copy = f32[3,1,3,1] copy(%reshape)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "reshape");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(
      instruction,
      op::Sharding(
          "{devices=[1,1,1,1,2,2]0,2,1,3 last_tile_dims={manual,replicated}}"));
}

TEST_F(ShardingPropagationTest, X64Combine) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[102,192,192] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3}
  %param1 = f32[102,192,192] parameter(1),
    sharding={devices=[1,2,2]0,1,2,3}
  %custom-call = f64[102,192,192] custom-call(f32[102,192,192] %param0, f32[102,192,192] %param1), custom_call_target="X64Combine"
  ROOT %copy = f64[102,192,192] copy(%custom-call),
    sharding={devices=[1,2,1,2]0,1,2,3 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "custom-call");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,2]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, LayoutConstraint) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[102,192,192] parameter(0),
    sharding={devices=[1,2,2]0,1,2,3}
  %custom-call = f32[102,192,192]{0,1,2} custom-call(f32[102,192,192] %param0), custom_call_target="LayoutConstraint"
  ROOT %copy = f32[102,192,192] copy(%custom-call),
    sharding={devices=[1,2,1,2]0,1,2,3 last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "custom-call");
  EXPECT_THAT(instruction->shape(), ShapeUtil::MakeShapeWithDenseLayout(
                                        F32, {102, 192, 192}, {0, 1, 2}));
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,2,2]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, OffloadingPropagation) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %offloading {
  %param0 = f32[1,256,128] parameter(0), sharding={devices=[1,1,4]0,1,2,3}
  %zero = f32[] constant(0.0)
  %broadcast = f32[256,256,128] broadcast(%zero), dimensions={}
  %izero = s32[] constant(0)
  %custom-call.0 = f32[1,256,128] custom-call(f32[1,256,128] %param0), custom_call_target="MoveToHost"
  %dynamic-update-slice = f32[256,256,128] dynamic-update-slice(%broadcast, %custom-call.0, %izero, %izero, %izero)
  %dynamic-slice = f32[1,256,128] dynamic-slice(%dynamic-update-slice, %izero, %izero, %izero), dynamic_slice_sizes={1,256,128}
  %custom-call.1 = f32[1,256,128] custom-call(f32[1,256,128] %dynamic-slice), custom_call_target="MoveToDevice"
  ROOT %copy = f32[1,256,128] copy(%custom-call.1), sharding={devices=[1,4,1]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);

  auto* to_host = FindInstruction(module.get(), "custom-call.0");
  EXPECT_THAT(to_host, op::Sharding("{devices=[1,1,4]0,1,2,3}"));

  auto* from_host_input =
      FindInstruction(module.get(), "custom-call.1")->operand(0);
  EXPECT_THAT(from_host_input, op::Sharding("{devices=[1,1,4]0,1,2,3}"));
}

TEST_P(ParameterizedMetadataTest, PropagateThroughSingleUsers) {
  const char* const hlo_string = R"(
HloModule module

%cond {
  %vars.cond = (u32[], f32[10,10], f32[10,10]) parameter(0)
  %count.cond = u32[] get-tuple-element((u32[], f32[10,10], f32[10,10]) %vars.cond), index=0
  %limit = u32[] constant(10)
  ROOT %lt = pred[] compare(u32[] %count.cond, u32[] %limit), direction=LT
}

%body {
  %vars = (u32[], f32[10,10], f32[10,10]) parameter(0)
  %count = u32[] get-tuple-element(%vars), index=0
  %acc = f32[10,10] get-tuple-element((u32[], f32[10,10],f32[10,10]) %vars), index=1
  %cvt = s32[10,10] convert(acc)

  %one = u32[] constant(1)
  %count.1 = u32[] add(u32[] %count, u32[] %one)
  %acc.i = s32[10,10] add(s32[10,10] %cvt, s32[10,10] %cvt), sharding={devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}
  %acc.1 = f32[10,10] convert(acc.i)
  ROOT %tuple = (u32[], f32[10,10], f32[10,10]) tuple(u32[] %count.1, f32[10,10] %acc, f32[10,10] %acc.1)
}

ENTRY %entry {
  %p0 = f32[10,10] parameter(0)
  %p0.copy = f32[10,10] copy(f32[10,10] %p0), sharding={devices=[4,1]0,1,2,3}
  %p1 = f32[10,10] parameter(1)
  %p2 = f32[10,10] parameter(2)
  %p2.copy = f32[10,10] copy(f32[10,10] %p2)
  %zero = u32[] constant(0)
  %init = (u32[], f32[10,10], f32[10,10]) tuple(u32[] %zero, f32[10,10] %p0.copy, f32[10,10] %p2.copy)
  %while = (u32[], f32[10,10], f32[10,10]) while((u32[], f32[10,10], f32[10,10]) %init),
    body=%body, condition=%cond
  %g1 = u32[] get-tuple-element((u32[], f32[10,10], f32[10,10]) %while), index=0
  %g2 = f32[10,10] get-tuple-element((u32[], f32[10,10], f32[10,10]) %while), index=1
  %g3 = f32[10,10] get-tuple-element((u32[], f32[10,10], f32[10,10]) %while), index=2
  ROOT %t = (u32[], f32[10,10], f32[10,10]) tuple(%g1, %g2, %g3)
})";

  // Propagation of user-defined partial sharding of while-related instruction
  // (body root in this test).
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto body_root = FindInstruction(module.get(), "tuple");
  EXPECT_NE(nullptr, body_root);
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  VLOG(1) << "Mod:";
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* convert_instr = FindInstruction(module.get(), "cvt");

  EXPECT_THAT(convert_instr, op::Sharding("{devices=[4,1]0,1,2,3}"));
}

TEST_P(ParameterizedMetadataTest, NestedTupleFromUserSharding) {
  const char* const hlo_string = R"(
HloModule module

%cond {
  %vars.cond = (u32[], ((f32[10,10], f32[10,10]), f32[]), f32[10,10]) parameter(0)
  %count.cond = u32[] get-tuple-element(%vars.cond), index=0
  %limit = u32[] constant(10)
  ROOT %lt = pred[] compare(u32[] %count.cond, u32[] %limit), direction=LT
}

%body {
  %vars = (u32[], ((f32[10,10], f32[10,10]), f32[]), f32[10,10]) parameter(0)
  %count = u32[] get-tuple-element(%vars), index=0
  %fwd = ((f32[10,10], f32[10,10]), f32[]) get-tuple-element(%vars), index=1
  %acc = f32[10,10] get-tuple-element(%vars), index=2
  %cvt = s32[10,10] convert(acc)

  %one = u32[] constant(1)
  %count.1 = u32[] add(u32[] %count, u32[] %one)
  %acc.i = s32[10,10] add(s32[10,10] %cvt, s32[10,10] %cvt)
  %acc.1 = f32[10,10] convert(acc.i)
  ROOT %tuple = (u32[], ((f32[10,10], f32[10,10]), f32[]), f32[10,10]) tuple(%count.1, %fwd, %acc.1)
}

ENTRY %entry {
  %p0 = f32[10,10] parameter(0)
  %p0.copy = f32[10,10] copy(f32[10,10] %p0)
  %p1 = f32[10,10] parameter(1)
  %p1.copy = f32[10,10] copy(f32[10,10] %p1)
  %p2 = f32[10,10] parameter(2)
  %p2.copy = f32[10,10] copy(f32[10,10] %p2)
  %zero = u32[] constant(0)
  %zerof = f32[] constant(0)
  %init0 = (f32[10,10], f32[10,10]) tuple(%p0.copy, %p1.copy)
  %init1 = ((f32[10,10], f32[10,10]), f32[]) tuple(%init0, %zerof)
  %init = (u32[], ((f32[10,10], f32[10,10]), f32[]), f32[10,10]) tuple(%zero, %init1, %p2.copy)
  %while = (u32[], ((f32[10,10], f32[10,10]), f32[]), f32[10,10]) while(%init),
    body=%body, condition=%cond
  %g1 = u32[] get-tuple-element(%while), index=0
  %g2 = ((f32[10,10], f32[10,10]), f32[]) get-tuple-element(%while), index=1
  %g2.0 = (f32[10,10], f32[10,10]) get-tuple-element(%g2), index=0
  %g2.0.0 = f32[10,10] get-tuple-element(%g2.0), index=0
  %g3 = f32[10,10] get-tuple-element(%while), index=2
  %copy.g3 = f32[10,10] copy(%g3), sharding={devices=[4,1]0,1,2,3}
  ROOT %t = (u32[], f32[10,10], f32[10,10]) tuple(%g1, %g2.0.0, %g3)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto body_root = FindInstruction(module.get(), "tuple");
  EXPECT_NE(nullptr, body_root);
  if (GetParam().clear_metadata) {
    ClearMetadata(module.get());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* convert_instr =
      FindInstruction(module.get(), "p2.copy");

  EXPECT_THAT(convert_instr, op::Sharding("{devices=[4,1]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, CSEPreventionOnly) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[] parameter(0), sharding={replicated}
  %br = f32[4] broadcast(%param0), dimensions={}
  %add = f32[4] add(%br, %br)
  %annotate = f32[4] custom-call(%add), custom_call_target="Sharding",
    backend_config="unspecified_dims=[0]", sharding={replicated}
  ROOT %copy = f32[4] copy(%annotate), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false},
          /*cse_prevention_only=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* br = FindInstruction(module.get(), "br");
  EXPECT_THAT(br, op::Sharding("{devices=[4]0,1,2,3}"));
  EXPECT_THAT(br->sharding(), ShardingMetadata({CreateMetadata(
                                  "_sharding_propagation_cse_prevention")}));
  EXPECT_THAT(FindInstruction(module.get(), "annotate"),
              AllOf(op::Sharding("{replicated}"), op::CustomCall()));
  EXPECT_FALSE(FindInstruction(module.get(), "add")->has_sharding());
}

TEST_F(ShardingPropagationTest, RemoveCSEPrevention) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[] parameter(0), sharding={replicated}
  %br = f32[4] broadcast(%param0), dimensions={},
    sharding={devices=[4]0,1,2,3 metadata={op_name="_sharding_propagation_cse_prevention"}}
  %add = f32[4] add(%br, %br)
  %annotate = f32[4] custom-call(%add), custom_call_target="Sharding",
    backend_config="unspecified_dims=[0]", sharding={replicated}
  ROOT %copy = f32[4] copy(%annotate), sharding={devices=[4]3,2,1,0}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  // Test that the CSE prevention sharding is replaced with the new sharding.
  EXPECT_THAT(FindInstruction(module.get(), "br"),
              op::Sharding("{devices=[4]3,2,1,0}"));
  EXPECT_THAT(FindInstruction(module.get(), "add"),
              op::Sharding("{devices=[4]3,2,1,0}"));
}

TEST_F(ShardingPropagationTest, ReshapeTrivialDimPartialReplicate) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[8,128] parameter(0), sharding={replicated}
  %c = f32[8,128] copy(%param0)
  %rsp = f32[8,1,128] reshape(%c),
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7}
  ROOT %copy = f32[8,1,128] copy(%rsp),
    sharding={devices=[1,2,4]0,1,2,3,4,5,6,7}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      FindInstruction(module.get(), "c"),
      op::Sharding("{devices=[1,4,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}"));
}

TEST_F(ShardingPropagationTest, EmptyTupleWithinTuple) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[2] parameter(0), sharding={replicated}
  %et = () tuple()
  %tuple = (f32[2], (), (), f32[2]) tuple(%param0, %et, %et, %param0)
  ROOT %copy = (f32[2], (), (), f32[2]) copy(%tuple)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
}

TEST_F(ShardingPropagationTest, ContractingAsNonContractingCrash) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %p0 = f32[20,64,56,56]{3,2,1,0} parameter(0), sharding={replicated}
  %p1 = f32[1,1,256,64]{2,3,1,0} parameter(1), sharding={devices=[4,2,1,1]0,1,2,3,4,5,6,7}
  %convolution.4512 = f32[20,256,56,56]{3,2,1,0} convolution(%p0, %p1), window={size=1x1}, dim_labels=bf01_01oi->bf01
  ROOT %copy = f32[20,256,56,56]{3,2,1,0} copy(%convolution.4512)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
}

TEST_F(ShardingPropagationTest, PropagateReduceManualTuple) {
  const char* const hlo_string = R"(
HloModule pjit

orclone {
  lhs.1 = u32[] parameter(0)
  rhs.1 = u32[] parameter(2)
  or.2 = u32[] or(lhs.1, rhs.1)
  lhs.0 = u32[] parameter(1)
  rhs.0 = u32[] parameter(3)
  or.3 = u32[] or(lhs.0, rhs.0)
  ROOT tuple.4 = (u32[], u32[]) tuple(or.2, or.3)
}

ENTRY %main.21 {
  select.104 = u32[2,2]{1,0} parameter(0), sharding={manual}
  shift-left.5 = u32[2,2]{1,0} parameter(1), sharding={manual}
  constant.4183 = u32[] constant(0), sharding={manual}
  reduce.1 = (u32[2]{0}, u32[2]{0}) reduce(shift-left.5, select.104, constant.4183, constant.4183), dimensions={1}, to_apply=orclone
  ROOT get-tuple-element.13 = u32[2]{0} get-tuple-element(reduce.1), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));

  EXPECT_THAT(FindInstruction(module.get(), "reduce.1"),
              op::Sharding("{{manual}, {manual}}"));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
}

TEST_F(ShardingPropagationTest, MergeCompatibleTiles) {
  const char* const hlo_string = R"(
HloModule pjit

ENTRY %main.21 {
  p = bf16[8,4,256,1024,12288]{4,3,2,1,0} parameter(0), sharding={devices=[8,1,1,1,1]0,1,2,3,4,5,6,7}
  p2 = bf16[8,4,256,1024,12288]{4,3,2,1,0} parameter(1), sharding={devices=[4,1,1,1,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  c0 =  bf16[8,4,256,1024,12288]{4,3,2,1,0} copy(p)
  c1 =  bf16[8,4,256,1024,12288]{4,3,2,1,0} copy(p2)
  a = bf16[8,4,256,1024,12288]{4,3,2,1,0} add(c0, c1)
  ROOT c2 = bf16[8,4,256,1024,12288]{4,3,2,1,0} copy(a), sharding={devices=[8,1,1,1,1]0,1,2,3,4,5,6,7}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "c1"),
              op::Sharding("{devices=[8,1,1,1,1]0,1,2,3,4,5,6,7}"));
}

TEST_F(ShardingPropagationTest, OutfeedUser) {
  const char* const hlo_string = R"(
HloModule pjit

ENTRY %main.21 {
  p = f32[10,128]{1,0} parameter(0)
  c = f32[10,128]{1,0} copy(p)
  t = (f32[10,128]{1,0}) tuple(c)
  a = token[] after-all()
  ROOT of = token[] outfeed((f32[10,128]{1,0}) %t, token[] %a), outfeed_shape=(f32[10,128]{1,0}), sharding={{devices=[2,1]0,1}, {maximal device=0}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "c"),
              op::Sharding("{devices=[2,1]0,1}"));
}

TEST_F(ShardingPropagationTest, SortForwardWithBarrier) {
  const char* const hlo_string = R"(
HloModule module

compare {
  p.0.lhs = f32[] parameter(0), sharding={replicated}
  p.0.rhs = f32[] parameter(1), sharding={replicated}
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, sharding={replicated}
}

ENTRY entry {
  param.0 = f32[1024,1024]{1,0} parameter(0)
  negate.0 = f32[1024,1024]{1,0} negate(param.0), sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  %shard-barrier-from = f32[1024,1024]{1,0} custom-call(%negate.0), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  sort.0 = f32[1024,1024]{1,0} sort(shard-barrier-from), dimensions={1}, is_stable=true, to_apply=compare
  ROOT copy.0 = f32[1024,1024]{1,0} copy(sort.0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_FALSE(FindInstruction(module.get(), "sort.0")->has_sharding());
}

TEST_F(ShardingPropagationTest, SortBackwardWithBarrier) {
  const char* const hlo_string = R"(
HloModule module

compare {
  p.0.lhs = f32[] parameter(0), sharding={replicated}
  p.0.rhs = f32[] parameter(1), sharding={replicated}
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, sharding={replicated}
}

ENTRY entry {
  param.0 = f32[1024,1024]{1,0} parameter(0)
  negate.0 = f32[1024,1024]{1,0} negate(param.0)
  %shard-barrier-to = f32[1024,1024]{1,0} custom-call(%negate.0), custom_call_target="ShardBarrierTo", custom_call_has_side_effect=true
  sort.0 = f32[1024,1024]{1,0} sort(shard-barrier-to), dimensions={1}, is_stable=true, to_apply=compare,
    sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  ROOT copy.0 = f32[1024,1024]{1,0} copy(sort.0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      std::ignore,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_THAT(FindInstruction(module.get(), "negate.0"),
              op::Sharding("{replicated}"));
}

TEST_F(ShardingPropagationTest, SortOperandShardedOnSortDim_RankOne) {
  const char* const hlo_string = R"(
HloModule module, entry_computation_layout={(f32[1024]{0})->(f32[1024]{0}, s32[1024]{0})}

compare {
  p.0.lhs = f32[] parameter(0), sharding={replicated}
  p.0.rhs = f32[] parameter(1), sharding={replicated}
  p.1.lhs = s32[] parameter(2), sharding={replicated}
  p.1.rhs = s32[] parameter(3), sharding={replicated}
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, sharding={replicated}
}

ENTRY entry {
  param.0 = f32[1024]{0} parameter(0)
  negate.0 = f32[1024]{0} negate(param.0), sharding={devices=[8]0,1,2,3,4,5,6,7}
  iota.0 = s32[1024]{0} iota(), iota_dimension=0
  sort.0 = (f32[1024]{0}, s32[1024]{0}) sort(negate.0, iota.0), dimensions={0}, is_stable=true, to_apply=compare
  ROOT copy.0 = (f32[1024]{0}, s32[1024]{0}) copy(sort.0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_FALSE(changed);  // Does not propagate the sharding for 1D operands
}

TEST_F(ShardingPropagationTest, SortOperandShardedOnSortDim_RankTwo) {
  const char* const hlo_string = R"(
HloModule module, entry_computation_layout={(f32[1024,1024]{1,0})->(f32[1024,1024]{1,0}, s32[1024,1024]{1,0})}

compare {
  p.0.lhs = f32[] parameter(0), sharding={replicated}
  p.0.rhs = f32[] parameter(1), sharding={replicated}
  p.1.lhs = s32[] parameter(2), sharding={replicated}
  p.1.rhs = s32[] parameter(3), sharding={replicated}
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, sharding={replicated}
}

ENTRY entry {
  param.0 = f32[1024,1024]{1,0} parameter(0)
  negate.0 = f32[1024,1024]{1,0} negate(param.0), sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  iota.0 = s32[1024,1024]{1,0} iota(), iota_dimension=1
  sort.0 = (f32[1024,1024]{1,0}, s32[1024,1024]{1,0}) sort(negate.0, iota.0), dimensions={1}, is_stable=true, to_apply=compare
  ROOT copy.0 = (f32[1024,1024]{1,0}, s32[1024,1024]{1,0}) copy(sort.0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "iota.0"),
              op::Sharding("{devices=[1,8]0,1,2,3,4,5,6,7}"));
  EXPECT_THAT(
      FindInstruction(module.get(), "sort.0"),
      op::Sharding(
          "{{devices=[1,8]0,1,2,3,4,5,6,7}, {devices=[1,8]0,1,2,3,4,5,6,7}}"));
}

TEST_F(ShardingPropagationTest, ConditionalManual) {
  const char* const hlo_string = R"(
HloModule module

%true_comp {
  %tp = (f32[3,5], f32[]) parameter(0)
  %tgte.0 = f32[3,5] get-tuple-element(%tp), index=0
  %tgte.1 = f32[] get-tuple-element(%tp), index=1
  %ttr = f32[5,3] transpose(%tgte.0), dimensions={1,0}

  %broadcast.1 = f32[5,3] broadcast(%tgte.1), dimensions={}
  %add.1 = f32[5,3] add(%broadcast.1, %ttr)

  ROOT %tr = (f32[5,3], f32[]) tuple(%add.1, %tgte.1)
}

%false_comp {
  %fp = (f32[5,3], f32[5,3], f32[]) parameter(0)
  %fgte.0 = f32[5,3] get-tuple-element(%fp), index=0
  %fgte.1 = f32[] get-tuple-element(%fp), index=2
  ROOT %fr = (f32[5,3], f32[]) tuple(%fgte.0, %fgte.1)
}

ENTRY entry {
  %cond = pred[] parameter(0), sharding={devices=[2,2]<=[4] last_tile_dims={manual, replicated}}
  %tp.0 = f32[3,5] parameter(1), sharding={devices=[1,1,2,2]<=[4] last_tile_dims={manual, replicated}}
  %fp.0 = f32[5,3] parameter(2), sharding={devices=[1,1,2,2]<=[4] last_tile_dims={manual, replicated}}
  %const0 = f32[] constant(0)
  %const1 = f32[] constant(1)
  %true_param = (f32[3,5], f32[]) tuple(%tp.0, %const0)
  %false_param = (f32[5,3], f32[5,3], f32[]) tuple(%fp.0, fp.0, %const1)
  ROOT %conditional = (f32[5,3], f32[]) conditional(
      %cond, %true_param, %false_param),
    true_computation=%true_comp,
    false_computation=%false_comp
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* tp = FindInstruction(module.get(), "tp");
  auto* true_param = FindInstruction(module.get(), "true_param");
  EXPECT_EQ(tp->sharding(), true_param->sharding());
  auto* fp = FindInstruction(module.get(), "fp");
  auto* false_param = FindInstruction(module.get(), "false_param");
  EXPECT_EQ(fp->sharding(), false_param->sharding());
}

TEST_F(ShardingPropagationTest, WhileDSManual) {
  const char* const hlo_string = R"(
HloModule module

while.condition {
  arg_tuple = (s32[], pred[2,8,4]) parameter(0)
  tripcount = s32[] get-tuple-element(arg_tuple), index=0
  triplimit = s32[] constant(2)
  ROOT compare.0 = pred[] compare(tripcount, triplimit), direction=LT
}

while.body {
  arg_tuple = (s32[], pred[2,8,4]) parameter(0)
  tripcount = s32[] get-tuple-element(arg_tuple), index=0
  one = s32[] constant(0)
  tripcount_next = s32[] add(tripcount, one)
  preds.1 = pred[2,8,4] get-tuple-element(arg_tuple), index=1
  zero.1 = s32[] constant(0)
  dynamic-slice.1 = pred[1,8,4] dynamic-slice(preds.1, tripcount, zero.1, zero.1), dynamic_slice_sizes={1,8,4}, sharding={devices=[1,1,1,2,4]<=[8] last_tile_dims={manual, replicated}}
  ROOT result = (s32[], pred[2,8,4]) tuple(tripcount_next, preds.1)
}

ENTRY entry {
  preds = pred[2,8,4] parameter(0), sharding={devices=[1,1,1,2,4]<=[8] last_tile_dims={manual, replicated}}
  zero = s32[] constant(0)
  tuple.13 = (s32[], pred[2,8,4]) tuple(zero, preds)
  ROOT result = while(tuple.13), condition=while.condition, body=while.body
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true)
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* tuple = FindInstruction(module.get(), "tuple.13");
  EXPECT_THAT(tuple, op::Sharding("{{replicated}, {devices=[1,1,1,2,4]<=[8] "
                                  "last_tile_dims={manual, replicated}}}"));
}

TEST_F(ShardingPropagationTest, PropagateToOutput) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[] parameter(0), sharding={replicated}
  %br = f32[4] broadcast(%param0), dimensions={}
  %annotate = f32[4] custom-call(%br), custom_call_target="Sharding",
    backend_config="unspecified_dims=[0]", sharding={devices=[4]0,1,2,3}
  ROOT %add = f32[4] add(%annotate, %annotate), sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true,
                          /*allow_spmd_sharding_propagation_to_output=*/{true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{devices=[4]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, PropagateToOutputTuplePartial) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[] parameter(0), sharding={replicated}
  %br = f32[4] broadcast(%param0), dimensions={}
  %annotate = f32[4] custom-call(%br), custom_call_target="Sharding",
    backend_config="unspecified_dims=[0]", sharding={devices=[4]0,1,2,3}
  %add = f32[4] add(%annotate, %annotate)
  %param1 = f32[] parameter(1), sharding={replicated}
  %br1 = f32[4] broadcast(%param1), dimensions={}
  %annotate1 = f32[4] custom-call(%br1), custom_call_target="Sharding",
    backend_config="unspecified_dims=[0]", sharding={devices=[4]0,1,2,3}
  %add1 = f32[4] add(%annotate1, %annotate1)
  ROOT t = (f32[4], f32[4]) tuple(add, add1), sharding={{replicated},{replicated}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true, false})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{{devices=[4]0,1,2,3},{replicated}}"));
}

TEST_F(ShardingPropagationTest, PropagateToOutputTupleFull) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[] parameter(0), sharding={replicated}
  %br = f32[4] broadcast(%param0), dimensions={}
  %annotate = f32[4] custom-call(%br), custom_call_target="Sharding",
    backend_config="unspecified_dims=[0]", sharding={devices=[4]0,1,2,3}
  %add = f32[4] add(%annotate, %annotate)
  %param1 = f32[] parameter(1), sharding={replicated}
  %br1 = f32[4] broadcast(%param1), dimensions={}
  %annotate1 = f32[4] custom-call(%br1), custom_call_target="Sharding",
    backend_config="unspecified_dims=[0]", sharding={devices=[4]0,1,2,3}
  %add1 = f32[4] add(%annotate1, %annotate1)
  ROOT t = (f32[4], f32[4]) tuple(add, add1), sharding={{replicated},{replicated}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(/*is_spmd=*/true, /*propagate_metadata=*/true,
                          /*allow_spmd_sharding_propagation_to_output=*/{true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{{devices=[4]0,1,2,3},{devices=[4]0,1,2,3}}"));
}

TEST_F(ShardingPropagationTest, PropagateToParametersNotEnabled1) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0)
  ROOT %add = f32[4] add(%param0, %param0), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, false})
          .Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_FALSE(
      module->entry_computation()->parameter_instruction(0)->has_sharding());
}

TEST_F(ShardingPropagationTest, PropagateToParametersNotEnabled2) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0), sharding={replicated}
  ROOT %add = f32[4] add(%param0, %param0), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, ShardingPropagation(/*is_spmd=*/true).Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{replicated}"));
}

TEST_F(ShardingPropagationTest, PropagateToParametersNotEnabled3) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0)
  %param1 = f32[4] parameter(1), sharding={replicated}
  ROOT %add = f32[4] add(%param0, %param1), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false})
          .Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_FALSE(
      module->entry_computation()->parameter_instruction(0)->has_sharding());
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{replicated}"));
}

TEST_F(ShardingPropagationTest, PropagateToParametersNotEnabled4) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0), sharding={replicated}
  %param1 = f32[4] parameter(1), sharding={replicated}
  ROOT %add = f32[4] add(%param0, %param1), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, false})
          .Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{replicated}"));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{replicated}"));
}

TEST_F(ShardingPropagationTest, PropagateToParametersPartial1) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0), sharding={replicated}
  %param1 = f32[4] parameter(1), sharding={replicated}
  ROOT %add = f32[4] add(%param0, %param1), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{replicated}"));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{devices=[4]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, PropagateToParametersPartial2) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0)
  %param1 = f32[4] parameter(1), sharding={replicated}
  ROOT %add = f32[4] add(%param0, %param1), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_FALSE(
      module->entry_computation()->parameter_instruction(0)->has_sharding());
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{devices=[4]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, PropagateToParametersPartial3) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0), sharding={replicated}
  %param1 = f32[4] parameter(1)
  ROOT %add = f32[4] add(%param0, %param1), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{replicated}"));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{devices=[4]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, PropagateToParametersPartial4) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0)
  %param1 = f32[4] parameter(1)
  ROOT %add = f32[4] add(%param0, %param1), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_FALSE(
      module->entry_computation()->parameter_instruction(0)->has_sharding());
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{devices=[4]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, PropagateToParametersFull1) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0)
  %param1 = f32[4] parameter(1)
  ROOT %add = f32[4] add(%param0, %param1), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{devices=[4]0,1,2,3}"));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{devices=[4]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, PropagateToParametersFull2) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0), sharding={replicated}
  %param1 = f32[4] parameter(1)
  ROOT %add = f32[4] add(%param0, %param1), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true, true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{devices=[4]0,1,2,3}"));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{devices=[4]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, PropagateToTupleParameter_WithoutSharding) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param = (f32[4], f32[4]) parameter(0)
  %gte0 = f32[4] get-tuple-element(%param), index=0
  %gte1 = f32[4] get-tuple-element(%param), index=1
  ROOT %add = f32[4] add(%gte0, %gte1), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true, true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{{devices=[4]0,1,2,3}, {devices=[4]0,1,2,3}}"));
}

TEST_F(ShardingPropagationTest, PropagateToTupleParameter_WithSharding1) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param = (f32[4], f32[4]) parameter(0), sharding={{replicated}, {replicated}}
  %gte0 = f32[4] get-tuple-element(%param), index=0
  %gte1 = f32[4] get-tuple-element(%param), index=1
  ROOT %add = f32[4] add(%gte0, %gte1), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{{replicated}, {devices=[4]0,1,2,3}}"));
}

TEST_F(ShardingPropagationTest, PropagateToTupleParameter_WithSharding2) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param = (f32[4], f32[4]) parameter(0), sharding={{replicated}, {replicated}}
  %gte0 = f32[4] get-tuple-element(%param), index=0
  %gte1 = f32[4] get-tuple-element(%param), index=1
  ROOT %add = f32[4] add(%gte0, %gte1), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true, false})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{{devices=[4]0,1,2,3}, {replicated}}"));
}

TEST_F(ShardingPropagationTest, PropagateManualOutfeed) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  p0 = f32[8]{0} parameter(0)
  p1 = f32[1]{0} parameter(1)
  tuple.1 = (f32[8]{0}) tuple(p0)
  constant.8 = u32[2]{0} constant({3, 12})
  tuple.10 = (u32[2]{0}) tuple(constant.8)
  aa.1 = token[] after-all()
  outfeed.1 = token[] outfeed(tuple.10, aa.1), outfeed_shape=(u32[2]{0}), sharding={{manual}, {manual}}
  outfeed.2 = token[] outfeed(tuple.1, outfeed.1), outfeed_shape=(f32[8]{0}), sharding={{manual}, {manual}}
  ROOT tuple.15 = (f32[1]{0}, token[]) tuple(p1, outfeed.2)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true, true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true, true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{{replicated}, {manual}}"));
}

TEST_F(ShardingPropagationTest, PropagateFromDanglingShardingCustomCall) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  p.0 = s32[40000]{0} parameter(0)
  add = s32[40000]{0} add(p.0, p.0)
  cc = s32[40000]{0} custom-call(add), custom_call_target="Sharding", sharding={devices=[4]0,1,2,3}
  ROOT mul = s32[40000]{0} multiply(add, add)

})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true})
          .Run(module.get()));
  EXPECT_TRUE(changed);

  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(bool dce_ed, RunHloPass(&dce, module.get()));
  EXPECT_TRUE(dce_ed);

  XLA_VLOG_LINES(1, module->ToString());
  // Check dangling sharding custom-call can be removed by DCE after
  // propagation.
  auto* instruction = FindInstruction(module.get(), "param0");
  EXPECT_EQ(instruction, nullptr);
  // Check sharding is correctly propagated.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{devices=[4]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest,
       DoNotPropagateToParameterIfNotDivisible_WithSharding) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0), sharding={replicated}
  %param1 = f32[3] parameter(1), sharding={replicated}
  %pad_value = f32[] constant(0)
  %pad = f32[4] pad(%param1, %pad_value), padding=0_1
  ROOT %add = f32[4] add(%param0, %pad), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{replicated}"));
  // Replicate the input since the propagated sharding does not evenly partition
  // it.
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{replicated}"));
}

TEST_F(ShardingPropagationTest,
       DoNotPropagateToParameterIfNotDivisible_WithoutSharding) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0), sharding={replicated}
  %param1 = f32[3] parameter(1)
  %pad_value = f32[] constant(0)
  %pad = f32[4] pad(%param1, %pad_value), padding=0_1
  ROOT %add = f32[4] add(%param0, %pad), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{replicated}"));
  // Replicate the input since the propagated sharding does not evenly partition
  // it.
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{replicated}"));
}

TEST_F(ShardingPropagationTest, DoNotPropagateToTupleParameterIfNotDivisible) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = (f32[4], f32[3]) parameter(0), sharding={{replicated}, {replicated}}
  %gte0 = f32[4] get-tuple-element(%param0), index=0
  %gte1 = f32[3] get-tuple-element(%param0), index=1
  %pad_value = f32[] constant(0)
  %pad = f32[4] pad(%gte1, %pad_value), padding=0_1
  ROOT %add = f32[4] add(%gte0, %pad), sharding={devices=[4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, true})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  // Replicate the second element of parameter since the propagated sharding
  // does not evenly partition it.
  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{{replicated}, {replicated}}"));
}

TEST_F(ShardingPropagationTest,
       DoNotPropagateToOutputIfNotDivisible_WithSharding) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0), sharding={replicated}
  %param1 = f32[4] parameter(1), sharding={replicated}
  %add = f32[4] add(%param0, %param1), sharding={devices=[4]0,1,2,3}
  ROOT %slice = f32[3] slice(%add), slice={[0:3:1]}, sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, false})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  // Replicate the output since the propagated sharding does not evenly
  // partition it.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{replicated}"));
}

TEST_F(ShardingPropagationTest,
       DoNotPropagateToOutputIfNotDivisible_WithoutSharding) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0), sharding={replicated}
  %param1 = f32[4] parameter(1), sharding={replicated}
  %add = f32[4] add(%param0, %param1), sharding={devices=[4]0,1,2,3}
  ROOT %slice = f32[3] slice(%add), slice={[0:3:1]}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, false})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  // Replicate the output since the propagated sharding does not evenly
  // partition it.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{replicated}"));
}

TEST_F(ShardingPropagationTest,
       DoNotPropagateToOutputTupleIfNotDivisible_WithSharding) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0), sharding={replicated}
  %param1 = f32[4] parameter(1), sharding={replicated}
  %add = f32[4] add(%param0, %param1), sharding={devices=[4]0,1,2,3}
  %slice = f32[3] slice(%add), slice={[0:3:1]}
  ROOT %tuple = (f32[4], f32[3]) tuple(%add, %slice), sharding={{replicated}, {replicated}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false, true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, false})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  // Replicate the output tuple element since the propagated sharding does not
  // evenly partition it.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{{replicated}, {replicated}}"));
}

TEST_F(ShardingPropagationTest,
       DoNotPropagateToOutputTupleIfNotDivisible_WithoutSharding) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[4] parameter(0), sharding={replicated}
  %param1 = f32[4] parameter(1), sharding={replicated}
  %add = f32[4] add(%param0, %param1), sharding={devices=[4]0,1,2,3}
  %slice = f32[3] slice(%add), slice={[0:3:1]}
  ROOT %tuple = (f32[4], f32[3]) tuple(%add, %slice)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true, true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, false})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  // Replicate the output tuple element since the propagated sharding does not
  // evenly partition it.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{{devices=[4]0,1,2,3}, {replicated}}"));
}

TEST_F(ShardingPropagationTest, PropagateShardLikeDifferentSharding) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  p.0 = s32[16,16] parameter(0), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  p.1 = s32[16,16] parameter(1), sharding={devices=[2,4]0,1,2,3,4,5,6,7}
  add.1 = s32[16,16] add(p.0, p.0)
  sharding.1 = s32[16,16] custom-call(add.1), custom_call_target="Sharding", sharding={unknown shard_like 0}
  add.2 = s32[16,16] add(p.1, p.1)
  sharding.2 = s32[16,16] custom-call(add.2), custom_call_target="Sharding", sharding={unknown shard_like 0}
  ROOT mul = s32[16,16] multiply(add.1, add.2)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, false})
          .Run(module.get()));
  EXPECT_TRUE(changed);

  XLA_VLOG_LINES(1, module->ToString());
  // Check dangling sharding custom-call can be removed by DCE after
  // propagation.
  auto* add_0 = FindInstruction(module.get(), "add.1");
  ASSERT_NE(add_0, nullptr);
  auto* add_1 = FindInstruction(module.get(), "add.2");
  ASSERT_NE(add_1, nullptr);
  // Check sharding is correctly propagated, and shard like wasn't able to force
  // the same sharding
  EXPECT_NE(add_0->sharding(), add_1->sharding());
}

TEST_F(ShardingPropagationTest, PropagateShardLikeSameSharding) {
  const char* const hlo_string = R"(
HloModule module

%add {
  %lhs = s32[] parameter(0)
  %rhs = s32[] parameter(1)
  ROOT %add = s32[] add(%lhs, %rhs)
}

ENTRY %entry {
  p.0 = s32[16,16] parameter(0), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  p.1 = s32[16,16] parameter(1)
  add.1 = s32[16,16] add(p.0, p.0)
  sharding.1 = s32[16,16] custom-call(add.1), custom_call_target="Sharding", sharding={unknown shard_like 0}
  init = s32[] constant(0)
  reduce.1 = s32[] reduce(add.1, init), dimensions={0,1}, to_apply=%add
  add.2 = s32[16,16] add(p.1, p.1)
  sharding.2 = s32[16,16] custom-call(add.2), custom_call_target="Sharding", sharding={unknown shard_like 0}
  reduce.2 = s32[] reduce(add.2, init), dimensions={0,1}, to_apply=%add
  ROOT mul = s32[] multiply(reduce.1, reduce.2)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, false})
          .Run(module.get()));
  EXPECT_TRUE(changed);

  XLA_VLOG_LINES(1, module->ToString());
  // Check dangling sharding custom-call can be removed by DCE after
  // propagation.
  auto* add_1 = FindInstruction(module.get(), "add.1");
  ASSERT_NE(add_1, nullptr);
  auto* add_2 = FindInstruction(module.get(), "add.2");
  ASSERT_NE(add_2, nullptr);
  // Check sharding is correctly propagated.
  EXPECT_EQ(add_1->sharding(), add_2->sharding());
}

TEST_F(ShardingPropagationTest, PropagateShardAs) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  p.0 = s32[16,16] parameter(0), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  p.1 = s32[16,16] parameter(1), sharding={devices=[2,4]0,1,2,3,4,5,6,7}
  add.1 = s32[16,16] add(p.0, p.0)
  sharding.1 = s32[16,16] custom-call(add.1), custom_call_target="Sharding", sharding={unknown shard_as 0}
  add.2 = s32[16,16] add(p.1, p.1)
  sharding.2 = s32[16,16] custom-call(add.2), custom_call_target="Sharding", sharding={unknown shard_as 0}
  ROOT mul = s32[16,16] multiply(add.1, add.2)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, false})
          .Run(module.get()));
  EXPECT_TRUE(changed);

  XLA_VLOG_LINES(1, module->ToString());
  // Check dangling sharding custom-call can be removed by DCE after
  // propagation.
  auto* add_1 = FindInstruction(module.get(), "add.1");
  ASSERT_NE(add_1, nullptr);
  auto* add_2 = FindInstruction(module.get(), "add.2");
  ASSERT_NE(add_2, nullptr);
  // Check sharding is correctly propagated.
  EXPECT_EQ(add_1->sharding(), add_2->sharding());
}

TEST_F(ShardingPropagationTest, PropagateShardAsToParameters) {
  const char* const hlo_string = R"(
HloModule module

%add {
  %lhs = s32[] parameter(0)
  %rhs = s32[] parameter(1)
  ROOT %add = s32[] add(%lhs, %rhs)
}

ENTRY %entry {
  p.0 = s32[16,16] parameter(0), sharding={unknown shard_as 0}
  p.1 = s32[16,16] parameter(1), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  add.1 = s32[16,16] add(p.0, p.0)
  init = s32[] constant(0)
  reduce.1 = s32[] reduce(add.1, init), dimensions={0,1}, to_apply=%add
  add.2 = s32[16,16] add(p.1, p.1)
  sharding.2 = s32[16,16] custom-call(add.2), custom_call_target="Sharding", sharding={unknown shard_as 0}
  reduce.2 = s32[] reduce(add.2, init), dimensions={0,1}, to_apply=%add
  ROOT mul = s32[] multiply(reduce.1, reduce.2)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true, true})
          .Run(module.get()));
  EXPECT_TRUE(changed);

  XLA_VLOG_LINES(1, module->ToString());
  // Check dangling sharding custom-call can be removed by DCE after
  // propagation.
  auto* p_0 = FindInstruction(module.get(), "p.0");
  ASSERT_NE(p_0, nullptr);
  auto* add_2 = FindInstruction(module.get(), "add.2");
  ASSERT_NE(add_2, nullptr);
  // Check sharding is correctly propagated.
  EXPECT_THAT(add_2, op::Sharding("{devices=[4,2]0,1,2,3,4,5,6,7}"));
  EXPECT_EQ(p_0->sharding(), add_2->sharding());
}
TEST_F(ShardingPropagationTest, PropagateShardAsToOutputs) {
  const char* const hlo_string = R"(
HloModule module

%add {
  %lhs = s32[] parameter(0)
  %rhs = s32[] parameter(1)
  ROOT %add = s32[] add(%lhs, %rhs)
}

ENTRY %entry {
  p.0 = s32[16,16] parameter(0), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  add.1 = s32[16,16] add(p.0, p.0)
  sharding.1 = s32[16,16] custom-call(add.1), custom_call_target="Sharding", sharding={unknown shard_as 0}
  init = s32[] constant(0)
  reduce.1 = s32[] reduce(add.1, init), dimensions={0,1}, to_apply=%add
  broadcast.1 = s32[16,16] broadcast(reduce.1), dimensions={}
  ROOT mul = s32[16,16] multiply(broadcast.1, broadcast.1), sharding={unknown shard_as 0}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false})
          .Run(module.get()));
  EXPECT_TRUE(changed);

  XLA_VLOG_LINES(1, module->ToString());
  // Check dangling sharding custom-call can be removed by DCE after
  // propagation.
  auto* add_1 = FindInstruction(module.get(), "add.1");
  ASSERT_NE(add_1, nullptr);
  auto* output = FindInstruction(module.get(), "mul");
  ASSERT_NE(output, nullptr);
  // Check sharding is correctly propagated.
  EXPECT_THAT(add_1, op::Sharding("{devices=[4,2]0,1,2,3,4,5,6,7}"));
  EXPECT_EQ(add_1->sharding(), output->sharding());
}

TEST_F(ShardingPropagationTest, PropagateShardAsBetweenInputOutput) {
  const char* const hlo_string = R"(
HloModule jit_zeros_like

ENTRY main.6 {
  Arg_0.1 = s64[8,2]{1,0} parameter(0), sharding={devices=[4,2]<=[8]}
  custom-call.4 = s64[8,2]{1,0} custom-call(Arg_0.1), custom_call_target="Sharding", sharding={unknown shard_as 0}
  constant.2 = s64[] constant(0)
  broadcast.3 = s64[8,2]{1,0} broadcast(constant.2), dimensions={}
  ROOT custom-call.5 = s64[8,2]{1,0} custom-call(broadcast.3), custom_call_target="Sharding", sharding={unknown shard_as 0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true})
          .Run(module.get()));
  EXPECT_TRUE(changed);
  VLOG(1) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{devices=[4,2]0,1,2,3,4,5,6,7}"));
}

TEST_F(ShardingPropagationTest, PropagateShardAsBetweenInputOutput2) {
  const char* const hlo_string = R"(
HloModule jit_f, entry_computation_layout={(f32[8]{0:T(256)})->(f32[8]{0:T(256)}, f32[8]{0:T(256)})}, allow_spmd_sharding_propagation_to_output={true,true}, num_partitions=4

ENTRY main.9 {
  Arg_0.1 = f32[8]{0} parameter(0)
  custom-call.6 = f32[8]{0} custom-call(Arg_0.1), custom_call_target="Sharding", custom_call_has_side_effect=true, sharding={unknown shard_as 0}, metadata={op_name="jit(f)/jit(main)/shard_alike" source_file="third_party/py/jax/tests/shard_alike_test.py" source_line=206}
  custom-call.4 = f32[8]{0} custom-call(Arg_0.1), custom_call_target="Sharding", sharding={devices=[4]<=[4]}, metadata={op_name="jit(f)/jit(main)/sharding_constraint[sharding=GSPMDSharding({devices=[4]<=[4]}) resource_env=ResourceEnv(mesh=Mesh(), ()) unconstrained_dims=set()]" source_file="third_party/py/jax/tests/shard_alike_test.py" source_line=204}
  constant.0 = f32[] constant(2)
  broadcast.0 = f32[8]{0} broadcast(constant.0), dimensions={}
  multiply.5 = f32[8]{0} multiply(custom-call.4, broadcast.0), metadata={op_name="jit(f)/jit(main)/mul" source_file="third_party/py/jax/tests/shard_alike_test.py" source_line=205}
  custom-call.7 = f32[8]{0} custom-call(multiply.5), custom_call_target="Sharding", custom_call_has_side_effect=true, sharding={unknown shard_as 0}, metadata={op_name="jit(f)/jit(main)/shard_alike" source_file="third_party/py/jax/tests/shard_alike_test.py" source_line=206}
  ROOT tuple.8 = (f32[8]{0}, f32[8]{0}) tuple(custom-call.6, custom-call.7)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true, true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true})
          .Run(module.get()));
  EXPECT_TRUE(changed);
  VLOG(1) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{{devices=[4]<=[4]}, {devices=[4]<=[4]}}"));
}

TEST_F(ShardingPropagationTest, LookaheadUsersOfDot) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  p0 = bf16[512,512,1024]{2,1,0} parameter(0), sharding={devices=[16,1,4]<=[64]}
  p1 = bf16[512,512,16,128]{3,2,1,0} parameter(1), sharding={devices=[16,1,4,1]<=[64]}
  p2 = bf16[16,1024,16,128]{3,2,1,0} parameter(2), sharding={devices=[1,4,4,1,4]<=[4,16]T(1,0) last_tile_dim_replicate}
  p3 = s32[] parameter(3)
  dot.1 = bf16[1024,16,128]{2,1,0} dot(p0, p1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
  reshape.1 = bf16[1,1024,16,128]{3,2,1,0} reshape(dot.1)
  constant.1 = s32[] constant(0)
  ROOT dynamic-update-slice.113 = bf16[16,1024,16,128]{3,2,1,0} dynamic-update-slice(p2, reshape.1, p3, constant.1, constant.1, /*index=5*/constant.1), sharding={devices=[1,4,4,1,4]<=[4,16]T(1,0) last_tile_dim_replicate}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true})
          .Run(module.get()));
  EXPECT_TRUE(changed);

  XLA_VLOG_LINES(1, module->ToString());
  // Check dangling sharding custom-call can be removed by DCE after
  // propagation.
  auto* instruction = FindInstruction(module.get(), "dot.1");
  // Check sharding is correctly propagated.
  EXPECT_THAT(instruction,
              op::Sharding(
                  "{devices=[4,4,1,4]<=[4,16]T(1,0) last_tile_dim_replicate}"));
}

TEST_F(ShardingPropagationTest, AsyncInstructionManualShardingArray) {
  const char* const hlo_string = R"(
HloModule module

called_computation {
  p0 = s32[8] parameter(0)
  p1 = s32[8] parameter(1)
  ROOT add = s32[8] add(p0, p1)
}, execution_thread="thread_1" // called_computation

ENTRY entry_computation {
  p0 = s32[8] parameter(0), sharding={manual}
  p1 = s32[8] parameter(1), sharding={manual}
  async-start = ((s32[8], s32[8]), s32[8], u32[]) call-start(p0, p1), async_execution_thread="thread_1", to_apply=called_computation
  ROOT async-done = s32[8] call-done(async-start)
}, execution_thread="thread_0" // entry_computation

)";

  {
    // Test with execution_threads = {"thread_0"}
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string));
    TF_ASSERT_OK_AND_ASSIGN(
        bool changed,
        ShardingPropagation(
            /*is_spmd=*/true, /*propagate_metadata=*/true,
            /*allow_spmd_sharding_propagation_to_output=*/{true},
            /*allow_spmd_sharding_propagation_to_parameters=*/{true})
            .Run(module.get(), {"thread_0"}));
    EXPECT_TRUE(changed);

    XLA_VLOG_LINES(1, module->ToString());

    auto* instruction = FindInstruction(module.get(), "async-start");
    ASSERT_NE(instruction, nullptr);
    EXPECT_THAT(instruction,
                op::Sharding("{{manual}, {manual}, {manual}, {manual}}"));

    auto* async_done = FindInstruction(module.get(), "async-done");
    ASSERT_NE(async_done, nullptr);
    EXPECT_THAT(async_done, op::Sharding("{manual}"));
  }

  {
    // Test with execution_threads = {"thread_0", "thread_1"}
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string));
    TF_ASSERT_OK_AND_ASSIGN(
        bool changed,
        ShardingPropagation(
            /*is_spmd=*/true, /*propagate_metadata=*/true,
            /*allow_spmd_sharding_propagation_to_output=*/{true},
            /*allow_spmd_sharding_propagation_to_parameters=*/{true})
            .Run(module.get(), {"thread_0", "thread_1"}));
    EXPECT_FALSE(changed);
  }

  {
    // Test with execution_threads = {}. Empty execution_threads means all
    // execution_threads are included.
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string));
    TF_ASSERT_OK_AND_ASSIGN(
        bool changed,
        ShardingPropagation(
            /*is_spmd=*/true, /*propagate_metadata=*/true,
            /*allow_spmd_sharding_propagation_to_output=*/{true},
            /*allow_spmd_sharding_propagation_to_parameters=*/{true})
            .Run(module.get()));
    EXPECT_FALSE(changed);
  }
}

TEST_F(ShardingPropagationTest, AsyncInstructionManualShardingTuple) {
  const char* const hlo_string = R"(
HloModule module

called_computation {
  p0 = s32[8] parameter(0)
  p1 = s32[8] parameter(1)
  add = s32[8] add(p0, p1)
  mul = s32[8] multiply(p0, p1)
  ROOT result = (s32[8], s32[8]) tuple(add, mul)
}, execution_thread="thread_1" // called_computation

ENTRY entry_computation {
  p0 = s32[8] parameter(0), sharding={manual}
  p1 = s32[8] parameter(1), sharding={manual}
  async-start = ((s32[8], s32[8]), (s32[8], s32[8]), u32[]) call-start(p0, p1), async_execution_thread="thread_1", to_apply=called_computation
  ROOT async-done = (s32[8], s32[8]) call-done(async-start)
}, execution_thread="thread_0" // entry_computation

)";

  {
    // Test with execution_threads = {"thread_0"}
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string));
    TF_ASSERT_OK_AND_ASSIGN(
        bool changed,
        ShardingPropagation(
            /*is_spmd=*/true, /*propagate_metadata=*/true,
            /*allow_spmd_sharding_propagation_to_output=*/{true},
            /*allow_spmd_sharding_propagation_to_parameters=*/{true})
            .Run(module.get(), {"thread_0"}));
    EXPECT_TRUE(changed);

    XLA_VLOG_LINES(1, module->ToString());

    auto* async_start = FindInstruction(module.get(), "async-start");
    ASSERT_NE(async_start, nullptr);
    EXPECT_THAT(
        async_start,
        op::Sharding("{{manual}, {manual}, {manual}, {manual}, {manual}}"));

    auto* async_done = FindInstruction(module.get(), "async-done");
    ASSERT_NE(async_done, nullptr);
    EXPECT_THAT(async_done, op::Sharding("{{manual}, {manual}}"));
  }

  {
    // Test with execution_threads = {"thread_0", "thread_1"}
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string));
    TF_ASSERT_OK_AND_ASSIGN(
        bool changed,
        ShardingPropagation(
            /*is_spmd=*/true, /*propagate_metadata=*/true,
            /*allow_spmd_sharding_propagation_to_output=*/{true},
            /*allow_spmd_sharding_propagation_to_parameters=*/{true})
            .Run(module.get(), {"thread_0", "thread_1"}));
    EXPECT_FALSE(changed);
  }

  {
    // Test with execution_threads = {}. Empty execution_threads means all
    // execution_threads are included.
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string));
    TF_ASSERT_OK_AND_ASSIGN(
        bool changed,
        ShardingPropagation(
            /*is_spmd=*/true, /*propagate_metadata=*/true,
            /*allow_spmd_sharding_propagation_to_output=*/{true},
            /*allow_spmd_sharding_propagation_to_parameters=*/{true})
            .Run(module.get()));
    EXPECT_FALSE(changed);
  }
}

TEST_F(ShardingPropagationTest, ShardAsWithShardBarrier) {
  const char* const hlo_string = R"(
HloModule pjit_f

ENTRY main.11 {
  Arg_0.1 = bf16[384,1408]{1,0} parameter(0), sharding={devices=[1,16,512]<=[8,16,64]T(1,0,2) last_tile_dim_replicate}
  broadcast.4 = bf16[8,384,1408]{2,1,0} broadcast(Arg_0.1), dimensions={1,2}
  custom-call.5 = bf16[8,384,1408]{2,1,0} custom-call(broadcast.4), custom_call_target="Sharding", custom_call_has_side_effect=true, sharding={unknown shard_as 1}
  broadcast.2 = bf16[8,384,1408]{2,1,0} broadcast(Arg_0.1), dimensions={1,2}
  custom-call.3 = bf16[8,384,1408]{2,1,0} custom-call(broadcast.2), custom_call_target="Sharding", sharding={devices=[8,1,1,1024]<=[8192] last_tile_dim_replicate}, backend_config="unspecified_dims=[1,2]"
  custom-call.6 = bf16[8,384,1408]{2,1,0} custom-call(custom-call.3), custom_call_target="Sharding", custom_call_has_side_effect=true, sharding={unknown shard_as 1}
  %shard-barrier-to = bf16[8,384,1408]{2,1,0} custom-call(%custom-call.6), custom_call_target="ShardBarrierTo", custom_call_has_side_effect=true
  slice.7 = bf16[1,384,1408]{2,1,0} slice(shard-barrier-to), slice={[1:2], [0:384], [0:1408]}
  reshape.8 = bf16[384,1408]{1,0} reshape(slice.7)
  tuple.9 = (bf16[384,1408]{1,0}) tuple(reshape.8)
  get-tuple-element.10 = bf16[384,1408]{1,0} get-tuple-element(tuple.9), index=0, sharding={devices=[16,1,512]<=[8,16,64]T(1,0,2) last_tile_dim_replicate}
  ROOT tuple.13 = (bf16[384,1408]{1,0}, bf16[8,384,1408]{2,1,0}) tuple(get-tuple-element.10, custom-call.5)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, false})
          .Run(module.get()));
  EXPECT_TRUE(changed);

  XLA_VLOG_LINES(1, module->ToString());
  auto* broadcast_4 = FindInstruction(module.get(), "broadcast.4");
  ASSERT_NE(broadcast_4, nullptr);
  EXPECT_THAT(
      broadcast_4,
      op::Sharding("{devices=[8,1,16,64]<=[8192] last_tile_dim_replicate}"));
  auto* copy = FindInstruction(module.get(), "copy");
  ASSERT_NE(copy, nullptr);
  EXPECT_THAT(
      copy,
      op::Sharding("{devices=[8,1,16,64]<=[8192] last_tile_dim_replicate}"));
}

TEST_F(ShardingPropagationTest, ShardAsWithShardBarrier2) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0)
  %custom-call.0 = f32[5,7,11,13]{3,2,1,0} custom-call(param0), custom_call_target="Sharding", sharding={devices=[2,1,1,1,4]<=[8] last_tile_dim_replicate}, backend_config="unspecified_dims=[1,2,3]"
  %shard-barrier-from = f32[5,7,11,13]{3,2,1,0} custom-call(%custom-call.0), custom_call_target="ShardBarrierFrom", custom_call_has_side_effect=true
  %custom-call.2 = f32[5,7,11,13]{3,2,1,0} custom-call(shard-barrier-from), custom_call_target="Sharding", custom_call_has_side_effect=true, sharding={unknown shard_as 1}
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1)
  %custom-call.1 = f32[5,7,11,13]{3,2,1,0} custom-call(param1), custom_call_target="Sharding", sharding={devices=[1,2,2,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}, backend_config="unspecified_dims=[0]"
  %custom-call.3 = f32[5,7,11,13]{3,2,1,0} custom-call(custom-call.1), custom_call_target="Sharding", custom_call_has_side_effect=true, sharding={unknown shard_as 1}
  ROOT %tuple = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) tuple(%custom-call.0, %custom-call.3)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false, false})
          .Run(module.get()));
  EXPECT_TRUE(changed);

  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Sharding(
          "{{devices=[2,2,2,1]<=[8]}, {devices=[1,2,2,1,2]<=[2,4]T(1,0) "
          "last_tile_dim_replicate}}"));
}

TEST_F(ShardingPropagationTest, CallPropagation) {
  const absl::string_view hlo_string = R"(
HloModule module

called_computation {
  p0 = bf16[20,2,68096,8512] parameter(0)
  %add_called_comp = bf16[20,2,68096,8512] add(p0, p0)
  ROOT tuple = (bf16[20,2,68096,8512]) tuple(add_called_comp)
}

ENTRY main {
  %param0 = bf16[20,2,68096,8512] parameter(0)
  %add = bf16[20,2,68096,8512] add(param0, param0)
  ROOT %call = (bf16[20,2,68096,8512]) call(add), to_apply=%called_computation, sharding={{devices=[1,1,16,64]<=[64,16]T(1,0)}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);
  EXPECT_THAT(add, op::Sharding("{devices=[1,1,16,64]<=[64,16]T(1,0)}"));
}

// Modified from b/357703299. Check that we do not propagate sharding to
// SPMDShardToFullShape.
TEST_F(ShardingPropagationTest, CallPropagationWithSPMDShardToFullShape) {
  const absl::string_view hlo_string = R"(
HloModule module

called_computation {
  p0 = bf16[4096,4096] parameter(0)
  %add_called_comp = bf16[4096,4096] add(p0, p0)
  ROOT tuple = (bf16[4096,4096]) tuple(add_called_comp)
}

ENTRY main {
  %param0 = bf16[4096,4096] parameter(0)
  %add = bf16[4096,4096] add(param0, param0)
  %custom-call.1 = bf16[4096,4096]{1,0} custom-call(add), custom_call_target="Sharding", sharding={devices=[2,1,2]<=[4] last_tile_dim_replicate}
  %custom-call.2 = bf16[2048,4096]{1,0} custom-call(custom-call.1), custom_call_target="SPMDFullToShardShape", sharding={manual}
  %custom-call.3 = bf16[2048,4096]{1,0} custom-call(custom-call.2), custom_call_target="Sharding", sharding={manual}
  %custom-call.4 = bf16[4096,4096]{1,0} custom-call(bf16[2048,4096]{1,0} %custom-call.3), custom_call_target="SPMDShardToFullShape", sharding={devices=[2,1,2]<=[4] last_tile_dim_replicate}
  ROOT %call = (bf16[4096,4096]) call(custom-call.4), to_apply=%called_computation, sharding={devices=[2,2]<=[4]}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{false},
          /*allow_spmd_sharding_propagation_to_parameters=*/{false})
          .Run(module.get()));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(changed);
  auto* custom_call_4 = FindInstruction(module.get(), "custom-call.4");
  ASSERT_NE(custom_call_4, nullptr);
  auto* operand = custom_call_4->operand(0);
  EXPECT_THAT(operand, op::Shape("bf16[2048,4096]"));
  EXPECT_THAT(custom_call_4, op::Shape("bf16[4096,4096]"));
  EXPECT_THAT(custom_call_4,
              op::Sharding("{devices=[2,1,2]<=[4] last_tile_dim_replicate}"));
}

TEST_F(ShardingPropagationTest, ReplicateRngBitGeneratorSeed) {
  const char* const hlo_string = R"(
HloModule module
apply_or {
  x = u64[] parameter(0)
  y = u64[] parameter(1)
  ROOT x_or_y = or(x, y)
}
ENTRY main {
  p = s32[2,2]{1,0} parameter(0), sharding={devices=[2,2]<=[4]}
  up = u64[2,2] convert(p)
  i = u64[] constant(0)
  seed = u64[2] reduce(up, i), dimensions={1}, to_apply=apply_or
  rbg = u32[2048,4096] rng-bit-generator(seed), algorithm=rng_default
  ROOT s = u32[2048,4096]{1,0} custom-call(rbg), custom_call_target="Sharding", sharding={devices=[2,2]<=[4]}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ShardingPropagation(
          /*is_spmd=*/true, /*propagate_metadata=*/true,
          /*allow_spmd_sharding_propagation_to_output=*/{true},
          /*allow_spmd_sharding_propagation_to_parameters=*/{true})
          .Run(module.get()));
  EXPECT_TRUE(changed);

  XLA_VLOG_LINES(1, module->ToString());
  auto* instruction = FindInstruction(module.get(), "seed");
  // Check sharding is correctly propagated.
  EXPECT_TRUE(instruction->sharding().IsReplicated());
}

}  // namespace
}  // namespace xla
