/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/sharding_propagation.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_op_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

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

class ParameterizedMetadataTest
    : public HloTestBase,
      public ::testing::WithParamInterface<MetadataTestParameter> {};

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
TEST_P(ParameterizedMetadataTest, BroadcastForwardPass) {
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
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
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

TEST_P(ParameterizedMetadataTest, BroadcastForwardPartial) {
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
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
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

TEST_P(ParameterizedMetadataTest, BroadcastUserPartial) {
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
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
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

TEST_P(ParameterizedMetadataTest, ShardedTupleReduceForwardAndBackwardPass) {
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
      ShardingPropagation(/*is_spmd=*/true, GetParam().propagate_metadata)
          .Run(module.get()));
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
}

TEST_P(ParameterizedMetadataTest, GetTupleElementForwardPass) {
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
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
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
    EXPECT_THAT(tuple1->sharding().tuple_elements()[0], ShardingMetadata({}));
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

TEST_P(ParameterizedMetadataTest, ReplicatedConvolutionLhs) {
  const char* const hlo_string = R"(
HloModule module

ENTRY conv {
  %lhs = f32[3,2,3]{2,1,0} parameter(0),
    sharding={replicated metadata={op_name="a"}}
  %rhs = f32[2,2,1]{2,1,0} parameter(1)
  %conv = f32[3,2,3]{2,1,0} convolution(%lhs, %rhs),
    window={size=1}, dim_labels=bf0_oi0->bf0
  ROOT %tuple = f32[3,2,3]{2,1,0} tuple(%conv)
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
  ROOT %tuple = f32[3,512,512] tuple(%conv)
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

  ROOT %tuple = (f32[3], f32[3], f32[3], f32[3]) tuple(
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
  EXPECT_TRUE(changed);
  auto* crs_f_tiled = FindInstruction(module.get(), "crs_f.tiled");
  ASSERT_NE(crs_f_tiled, nullptr);
  EXPECT_THAT(crs_f_tiled, op::Sharding("{devices=[2]0,1}"));
  auto* crs_f_none = FindInstruction(module.get(), "crs_f.none");
  ASSERT_NE(crs_f_none, nullptr);
  EXPECT_THAT(crs_f_none, op::NoSharding());
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

  auto while_is_sharded = [this](HloModule* module, const HloSharding& sharding,
                                 absl::Span<const absl::Span<const OpMetadata>>
                                     sharding_metadata) {
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
                        .ConsumeValueOrDie();
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
        ParseSharding("{devices=[2,1]0,1 metadata={op_name=\"b\"}}")
            .ConsumeValueOrDie());

    while_is_sharded(
        module.get(),
        ParseSharding("{{replicated}, {devices=[2,1]0,1}}").ConsumeValueOrDie(),
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
            .ConsumeValueOrDie());
    auto p0 = FindInstruction(module.get(), "p0");
    p0->set_sharding(
        ParseSharding("{devices=[1,2,2]0,2,1,3 last_tile_dim_replicate "
                      "metadata={op_name=\"c\"}}")
            .ConsumeValueOrDie());

    while_is_sharded(module.get(),
                     ParseSharding("{{replicated}, "
                                   "{devices=[2,2]0,1,2,3}}")
                         .ConsumeValueOrDie(),
                     {{}, {CreateMetadata("c"), CreateMetadata("b")}});
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
    sharding={maximal device=1 metadata={op_name="a"}}
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
  auto sharding = ParseSharding("{{maximal device=1}, {maximal device=1}}")
                      .ConsumeValueOrDie();
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
    sharding={maximal device=1 metadata={op_name="b"}}
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
  EXPECT_THAT(result.status().error_message(),
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
    sharding={maximal device=1 metadata={op_name="a"}}
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
  EXPECT_THAT(result.status().error_message(),
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
    sharding={maximal device=1 metadata={op_name="a"}}
  %recv-done = (f32[], token[]) recv-done(%recv), channel_id=1
  %data = f32[] get-tuple-element(%recv-done), index=0
  ROOT %tuple = (u32[], f32[]) tuple(%count, %data)
}

ENTRY %entry {
  %p0 = f32[] parameter(0)
  %zero = u32[] constant(0)
  %init = (u32[], f32[]) tuple(%zero, %p0)
  %while = (u32[], f32[]) while(%init), body=%body, condition=%cond,
    sharding={maximal device=0 metadata={op_name="b"}}
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
  EXPECT_THAT(result.status().error_message(),
              ::testing::HasSubstr(
                  "Instruction: while is on device: 0, which conflicts with "
                  "device: 1 of channel instruction: recv"));
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

  ROOT %tuple = (f32[8,256,256], f32[8,256,256], f32[8,256])
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

TEST_P(ParameterizedMetadataTest, ConvAsDotOnTrivialDims) {
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

%true_comp {
  %tp = (f32[3,5]) parameter(0)
  %tgte = f32[3,5] get-tuple-element(%tp), index=0
  %ttr = f32[5,3] transpose(%tgte), dimensions={1,0}
  ROOT %tr = (f32[5,3]) tuple(%ttr)
}

%false_comp {
  %fp = (f32[5,3]) parameter(0)
  %fgte = f32[5,3] get-tuple-element(%fp), index=0
  ROOT %fr = (f32[5,3]) tuple(%fgte)
}

ENTRY entry {
  %cond = pred[] parameter(0)
  %true_param = (f32[3,5]) parameter(1),
    sharding={{devices=[1,2]0,1 metadata={op_name="a"}}}
  %false_param = (f32[5,3]) parameter(2),
    sharding={{devices=[1,3]0,1,2 metadata={op_name="b"}}}
  %conditional = (f32[5,3]) conditional(
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
  EXPECT_TRUE(changed);
  auto* tp = FindInstruction(module.get(), "tp");
  ASSERT_NE(tp, nullptr);
  EXPECT_THAT(tp, op::Sharding("{{devices=[1,2]0,1}}"));
  auto* tgte = FindInstruction(module.get(), "tgte");
  ASSERT_NE(tgte, nullptr);
  EXPECT_THAT(tgte, op::Sharding("{devices=[1,2]0,1}"));
  auto* ttr = FindInstruction(module.get(), "ttr");
  ASSERT_NE(ttr, nullptr);
  EXPECT_THAT(ttr, op::Sharding("{devices=[2,1]0,1}"));
  auto* tr = FindInstruction(module.get(), "tr");
  ASSERT_NE(tr, nullptr);
  EXPECT_THAT(tr, op::Sharding("{{devices=[1,3]0,1,2}}"));
  auto* fp = FindInstruction(module.get(), "fp");
  ASSERT_NE(fp, nullptr);
  EXPECT_THAT(fp, op::Sharding("{{devices=[1,3]0,1,2}}"));
  auto* fgte = FindInstruction(module.get(), "fgte");
  ASSERT_NE(fgte, nullptr);
  EXPECT_THAT(fgte, op::Sharding("{devices=[1,3]0,1,2}"));
  auto* fr = FindInstruction(module.get(), "fr");
  ASSERT_NE(fr, nullptr);
  EXPECT_THAT(fr, op::Sharding("{{devices=[1,3]0,1,2}}"));
  auto* conditional = FindInstruction(module.get(), "conditional");
  ASSERT_NE(conditional, nullptr);
  EXPECT_THAT(conditional, op::Sharding("{{devices=[1,3]0,1,2}}"));

  auto check_metadata = [](const HloSharding& sharding,
                           const OpMetadata& metadata) {
    if (sharding.IsTuple()) {
      EXPECT_THAT(sharding.tuple_elements()[0], ShardingMetadata({metadata}));
    } else {
      EXPECT_THAT(sharding, ShardingMetadata({metadata}));
    }
  };

  auto check_empty_metadata = [](const HloSharding& sharding) {
    if (sharding.IsTuple()) {
      EXPECT_THAT(sharding.tuple_elements()[0], ShardingMetadata({}));
    } else {
      EXPECT_THAT(sharding, ShardingMetadata({}));
    }
  };

  for (HloInstruction* instruction : {tp, tgte, ttr}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      check_metadata(instruction->sharding(), CreateMetadata("a"));
    } else {
      check_empty_metadata(instruction->sharding());
    }
  }
  for (HloInstruction* instruction : {tr, fp, fgte, fr, conditional}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      check_metadata(instruction->sharding(), CreateMetadata("b"));
    } else {
      check_empty_metadata(instruction->sharding());
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

TEST_P(ParameterizedMetadataTest, DynamicSliceForwardPass) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0),
    sharding={devices=[1,1,2]0,1 metadata={op_name="a"}}
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
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "ds");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,1,2]0,1}"));
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
    sharding={devices=[1,1,2]0,1 metadata={op_name="a"}}
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
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "c0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[1,1,2]0,1}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(instruction->sharding(),
                ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
  }
}

TEST_P(ParameterizedMetadataTest, DynamicUpdateSliceForwardPassBase) {
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
  EXPECT_TRUE(changed);
  auto* dus = FindInstruction(module.get(), "dus");
  ASSERT_NE(dus, nullptr);
  EXPECT_THAT(dus, op::Sharding("{devices=[1,1,2]0,1}"));
  auto* c1 = FindInstruction(module.get(), "c1");
  ASSERT_NE(c1, nullptr);
  EXPECT_THAT(c1, op::Sharding("{devices=[1,1,2]0,1}"));
  for (HloInstruction* instruction : {dus, c1}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, DynamicUpdateSliceForwardPassUpdate) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0)
  %p1 = f32[11,1,15] parameter(1)
  %c1 = f32[11,1,15] copy(%p1),
    sharding={devices=[1,1,2]0,1 metadata={op_name="a"}}
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
  EXPECT_TRUE(changed);
  auto* dus = FindInstruction(module.get(), "dus");
  ASSERT_NE(dus, nullptr);
  EXPECT_THAT(dus, op::Sharding("{devices=[1,1,2]0,1}"));
  auto* c0 = FindInstruction(module.get(), "c0");
  ASSERT_NE(c0, nullptr);
  EXPECT_THAT(c0, op::Sharding("{devices=[1,1,2]0,1}"));
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
    sharding={devices=[1,1,2]0,1 metadata={op_name="a"}}
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
  EXPECT_TRUE(changed);
  auto* c0 = FindInstruction(module.get(), "c0");
  ASSERT_NE(c0, nullptr);
  EXPECT_THAT(c0, op::Sharding("{devices=[1,1,2]0,1}"));
  auto* c1 = FindInstruction(module.get(), "c1");
  ASSERT_NE(c1, nullptr);
  EXPECT_THAT(c1, op::Sharding("{devices=[1,1,2]0,1}"));
  for (HloInstruction* instruction : {c0, c1}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, EinsumLHSBatchPartitioned) {
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
      ShardingPropagation(/*is_spmd=*/false, GetParam().propagate_metadata)
          .Run(module.get()));
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

TEST_P(ParameterizedMetadataTest, GatherParallelAndPassthroughMerged) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg0 = s32[4,8,2,2]{3,2,1,0} parameter(0)
  %arg1 =  s32[4]{0} parameter(1)
  %input = s32[4,8,2,2]{3,2,1,0} copy(%arg0),
    sharding={devices=[2,1,2,1]0,1,4,5 metadata={op_name="a"}}
  %seq_size = s32[4]{0} copy(s32[4]{0} %arg1)
  %seq_b = s32[1,4,8]{2,1,0} broadcast(s32[4]{0} %seq_size
  ), dimensions={1}
  %iota.11 = s32[1,4,8]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,4,8]{2,1,0} concatenate(s32[1,4,8]{2,1,0} %iota.11,
    s32[1,4,8]{2,1,0} %seq_b), dimensions={0}
  %gather = s32[4,8,2,2]{3,2,1,0} gather(s32[4,8,2,2]{3,2,1,0} %input,
    s32[2,4,8]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[4,8,2,2]{3,2,1,0} copy(%gather)
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
  EXPECT_TRUE(changed);
  const HloInstruction* input = FindInstruction(module.get(), "input");
  ASSERT_NE(input, nullptr);
  EXPECT_THAT(input, op::Sharding("{devices=[2,1,2,1]0,1,4,5 }"));
  const HloInstruction* concatenate =
      FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(
      concatenate,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(gather, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));

  for (const HloInstruction* instruction : {input, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest, GatherParallelAndTrivialMerged) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg0 = s32[4,8,2,2]{3,2,1,0} parameter(0)
  %arg1 =  s32[4]{0} parameter(1)
  %input = s32[4,8,2,2]{3,2,1,0} copy(%arg0),
    sharding={devices=[2,2,1,1]0,1,4,5 metadata={op_name="a"}}
  %seq_size = s32[4]{0} copy(s32[4]{0} %arg1)
  %seq_b = s32[1,4,1]{2,1,0} broadcast(s32[4]{0} %seq_size), dimensions={1}
  %iota.11 = s32[1,4,1]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,4,1]{2,1,0} concatenate(s32[1,4,1]{2,1,0} %iota.11,
    s32[1,4,1]{2,1,0} %seq_b), dimensions={0}
  %gather = s32[4,1,2,2]{3,2,1,0} gather(s32[4,8,2,2]{3,2,1,0} %input,
    s32[2,4,1]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2}
  ROOT %copy = s32[4,1,2,2]{3,2,1,0} copy(%gather)
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
  EXPECT_TRUE(changed);
  const HloInstruction* input = FindInstruction(module.get(), "input");
  ASSERT_NE(input, nullptr);
  EXPECT_THAT(input, op::Sharding("{devices=[2,2,1,1]0,1,4,5}"));
  const HloInstruction* concatenate =
      FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(
      concatenate,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(
      gather,
      op::Sharding("{devices=[2,1,1,1,2]0,1,4,5 last_tile_dim_replicate}"));
  for (const HloInstruction* instruction : {input, gather}) {
    if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
      EXPECT_THAT(instruction->sharding(),
                  ShardingMetadata({CreateMetadata("a")}));
    } else {
      EXPECT_THAT(instruction->sharding(), ShardingMetadata({}));
    }
  }
}

TEST_P(ParameterizedMetadataTest,
       GatherParallelAndPassthroughMergedBackwardPass) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY %module {
  %arg0 = s32[4,8,2,2]{3,2,1,0} parameter(0)
  %arg1 =  s32[4]{0} parameter(1)
  %input = s32[4,8,2,2]{3,2,1,0} copy(%arg0)
  %seq_size = s32[4]{0} copy(s32[4]{0} %arg1)
  %seq_b = s32[1,4,8]{2,1,0} broadcast(s32[4]{0} %seq_size
  ), dimensions={1}
  %iota.11 = s32[1,4,8]{2,1,0} iota(), iota_dimension=1
  %concatenate = s32[2,4,8]{2,1,0} concatenate(s32[1,4,8]{2,1,0} %iota.11,
    s32[1,4,8]{2,1,0} %seq_b), dimensions={0}
  %gather = s32[4,8,2,2]{3,2,1,0} gather(s32[4,8,2,2]{3,2,1,0} %input,
    s32[2,4,8]{2,1,0} %concatenate), offset_dims={2,3},
    collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=0,
    slice_sizes={1,1,2,2},
    sharding={devices=[2,1,2,1]0,1,4,5 metadata={op_name="a"}}
  ROOT %copy = s32[4,8,2,2]{3,2,1,0} copy(%gather)
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
  EXPECT_TRUE(changed);
  const HloInstruction* input = FindInstruction(module.get(), "input");
  ASSERT_NE(input, nullptr);
  EXPECT_THAT(input, op::Sharding("{devices=[2,1,2,1]0,1,4,5 }"));
  const HloInstruction* concatenate =
      FindInstruction(module.get(), "concatenate");
  ASSERT_NE(concatenate, nullptr);
  EXPECT_THAT(
      concatenate,
      op::Sharding("{devices=[1,2,1,2]0,1,4,5 last_tile_dim_replicate}"));
  const HloInstruction* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(gather, op::Sharding("{devices=[2,1,2,1]0,1,4,5}"));
  if (GetParam().propagate_metadata && !GetParam().clear_metadata) {
    EXPECT_THAT(gather->sharding(), ShardingMetadata({CreateMetadata("a")}));
  } else {
    EXPECT_THAT(gather->sharding(), ShardingMetadata({}));
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

}  // namespace
}  // namespace xla
