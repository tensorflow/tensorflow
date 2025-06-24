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

#include "xla/service/spmd/shardy/shardy_xla_pass.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace sdy {

namespace {

using ShardyXLATest = HloHardwareIndependentTestBase;

void runShardy(VerifiedHloModule* module, bool stablehloImport,
               bool runSdyShardingPropagation = true,
               bool expectChanged = true) {
  FrontendAttributes attrs;
  if (stablehloImport) {
    attrs.mutable_map()->try_emplace(xla::sdy::kImportMhloShardings, "t");
  }
  module->add_frontend_attributes(attrs);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardyXLA(runSdyShardingPropagation).Run(module));
  VLOG(1) << module->ToString();
  if (expectChanged) {
    EXPECT_TRUE(changed);
  } else {
    EXPECT_FALSE(changed);
  }
}

void runShardyWithStablehloImport(VerifiedHloModule* module,
                                  bool runSdyShardingPropagation = true,
                                  bool expectChanged = true) {
  runShardy(module, /*stablehloImport=*/true, runSdyShardingPropagation,
            expectChanged);
}

void runShardyWithSdyImport(VerifiedHloModule* module) {
  runShardy(module, /*stablehloImport=*/false);
}

}  // namespace

TEST_F(ShardyXLATest, AllowSpmdShardingPropagationParametersOutputRespected) {
  const char* const hloString = R"(
    HloModule module, allow_spmd_sharding_propagation_to_parameters={false,true}, allow_spmd_sharding_propagation_to_output={true}
    ENTRY %conv {
      %p0 = f32[8,256,512] parameter(0), sharding={replicated}
      %p1 = f32[8,128,512] parameter(1), sharding={devices=[2,1,1,4]<=[8] last_tile_dim_replicate}
      %dot = f32[8,256,128] dot(%p0, %p1),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2}, sharding={devices=[2,2,2]<=[8]}
      ROOT %tuple = (f32[8,256,128]) tuple(%dot)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{replicated}"));
  EXPECT_THAT(
      module->entry_computation()->parameter_instruction(1),
      op::Sharding(
          "{devices=[2,2,1,2]<=[2,2,2]T(0,2,1) last_tile_dim_replicate}"));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{devices=[2,2,2]<=[8]}"));
}

TEST_F(ShardyXLATest, ElementWise) {
  const char* const hloString = R"(
    HloModule module

    ENTRY %entry {
      p0 = f32[6,3] parameter(0)
      p1 = f32[6,3] parameter(1)
      copy.p0 = f32[6,3] copy(p0)
      copy.p1 = f32[6,3] copy(p1)
      add = f32[6,3] add(copy.p0, copy.p1), sharding={devices=[2,1]<=[2]}, metadata={op_name="simple_example/add" source_file="source.txt" source_line=42}
      ROOT result = f32[6,3] copy(add)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  HloInstruction* add = FindInstruction(module.get(), xla::HloOpcode::kAdd);
  EXPECT_NE(add, nullptr);
  EXPECT_THAT(add, op::Sharding("{devices=[2,1]<=[2]}"));
  EXPECT_EQ(add->metadata().op_name(), "simple_example/add");
  EXPECT_EQ(add->metadata().source_file(), "source.txt");
  EXPECT_EQ(add->metadata().source_line(), 42);

  for (HloInstruction* param :
       module->entry_computation()->parameter_instructions()) {
    EXPECT_THAT(param, op::Sharding("{devices=[2,1]<=[2]}"));
  }

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{devices=[2,1]<=[2]}"));
}

TEST_F(ShardyXLATest, CostantSplitter) {
  const char* const hloString = R"(
    HloModule module
    ENTRY %constant_splitter {
      %p0 = f32[8,8] parameter(0), sharding={devices=[2,2]<=[4]}
      %constant = f32[] constant(3.14)
      %broadcast = f32[8,16] broadcast(%constant), dimensions={}
      %dot = f32[8,8] dot(%broadcast, %broadcast),
        lhs_contracting_dims={1}, rhs_contracting_dims={1}
      ROOT %add = f32[8,8] add(%p0, %dot)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  HloInstruction* dot = FindInstruction(module.get(), xla::HloOpcode::kDot);

  // The two operands of the dot are different constant expressions (constant
  // and broadcast).
  EXPECT_EQ(dot->operand_count(), 2);
  EXPECT_EQ(dot->operand(0)->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(dot->operand(1)->opcode(), HloOpcode::kBroadcast);
  EXPECT_NE(dot->operand(0), dot->operand(1));
  EXPECT_THAT(dot->operand(0),
              op::Sharding("{devices=[2,1,2]<=[4] last_tile_dim_replicate}"));
  EXPECT_THAT(
      dot->operand(1),
      op::Sharding("{devices=[2,1,2]<=[2,2]T(1,0) last_tile_dim_replicate}"));

  EXPECT_EQ(dot->operand(0)->operand(0)->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(dot->operand(1)->operand(0)->opcode(), HloOpcode::kConstant);

  // Constants with identical shardings are expected to be merged.
  // TODO(tomnatan): Uncomment this test once sdy pun bumped (3/31/25).
  // EXPECT_EQ(dot->operand(0)->operand(0), dot->operand(1)->operand(0));
}

TEST_F(ShardyXLATest, Dot) {
  const char* const hloString = R"(
    HloModule module
    ENTRY %conv {
      %p0 = f32[8,256,128] parameter(0)
      %p1 = f32[8,128,512] parameter(1)
      %p2 = f32[8,128] parameter(2)

      %dot0 = f32[8,512,256] dot(%p1, %p0),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={1}, rhs_contracting_dims={2}
      %dot1 = f32[8,256,512] dot(%p0, %p1),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
      %dot2 = f32[8,256] dot(%p0, %p2),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
      %dot3 = f32[8,256,512] dot(%p0, %p1),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1},
        sharding={devices=[2,2,2]<=[8]}

      ROOT %tuple = (f32[8,512,256], f32[8,256,512], f32[8,256], f32[8,256,512])
        tuple(%dot0, %dot1, %dot2, %dot3)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{devices=[2,2,1,2]<=[8] last_tile_dim_replicate}"));
  EXPECT_THAT(
      module->entry_computation()->parameter_instruction(1),
      op::Sharding(
          "{devices=[2,1,2,2]<=[2,2,2]T(0,2,1) last_tile_dim_replicate}"));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(2),
              op::Sharding("{devices=[2,1,4]<=[8] last_tile_dim_replicate}"));

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Sharding("{{devices=[2,2,2]<=[2,2,2]T(0,2,1)}, "
                   "{devices=[2,2,2]<=[8]}, {devices=[2,2,2]<=[8] "
                   "last_tile_dim_replicate}, {devices=[2,2,2]<=[8]}}"));
}

TEST_F(ShardyXLATest, DotTiledBatchDim) {
  const char* const hloString = R"(
    HloModule module
    ENTRY %conv {
      %p0 = f32[8,256,512] parameter(0)
      %p1 = f32[8,512,128] parameter(1)

      %add = f32[8,256,512] add(%p0, %p0)
      %dot = f32[8,256,128] dot(%add, %p1),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
      %res = f32[8,32768] reshape(%dot), sharding={devices=[2,2]<=[4]}

      ROOT %tuple = (f32[8,32768]) tuple(%res)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{devices=[2,2,1]<=[4]}"));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{devices=[2,1,1,2]<=[4] last_tile_dim_replicate}"));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{devices=[2,2]<=[4]}"));
}

TEST_F(ShardyXLATest, DotMergeOperands1) {
  const char* const hloString = R"(
    HloModule module
    ENTRY %conv {
      %p0 = f32[8,256,512] parameter(0),
        sharding={devices=[2,2,1,2]<=[8] last_tile_dim_replicate}
      %p1 = f32[8,128,512] parameter(1),
        sharding={devices=[2,2,1,2]<=[2,2,2]T(0,2,1) last_tile_dim_replicate}
      %dot = f32[8,256,128] dot(%p0, %p1),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2}
      ROOT %copy = f32[8,256,128] copy(%dot)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{devices=[2,2,1,2]<=[8] last_tile_dim_replicate}"));
  EXPECT_THAT(
      module->entry_computation()->parameter_instruction(1),
      op::Sharding(
          "{devices=[2,2,1,2]<=[2,2,2]T(0,2,1) last_tile_dim_replicate}"));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{devices=[2,2,2]<=[8]}"));
}

TEST_F(ShardyXLATest, DotMergeOperands2) {
  const char* const hloString = R"(
    HloModule module
    ENTRY %conv {
      %p0 = f32[8,256,512] parameter(0), sharding={devices=[2,2,2]<=[8]}
      %p1 = f32[8,128,512] parameter(1), sharding={devices=[2,2,2]<=[8]}
      %dot = f32[8,256,128] dot(%p0, %p1),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2}
      ROOT %copy = f32[8,256,128] copy(%dot)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{devices=[2,2,2]<=[8]}"));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{devices=[2,2,2]<=[8]}"));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{devices=[2,2,1,2]<=[8] last_tile_dim_replicate}"));
}

TEST_F(ShardyXLATest, DotMergeOperands3) {
  const char* const hloString = R"(
    HloModule module
    ENTRY %conv {
      %p0 = f32[256,512] parameter(0), sharding={devices=[2,4]<=[8]}
      %p1 = f32[128,512] parameter(1), sharding={devices=[4,2]<=[2,2,2]T(2,1,0)}
      %dot = f32[256,128] dot(%p0, %p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={1}
      ROOT %copy = f32[256,128] copy(%dot)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{devices=[2,4]<=[8]}"));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{devices=[4,2]<=[2,2,2]T(2,1,0)}"));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{devices=[2,4]<=[2,2,2]T(0,2,1)}"));
}

TEST_F(ShardyXLATest, BackwardDotFromContracting) {
  const char* const hloString = R"(
    HloModule module
    ENTRY %conv {
      %p0 = f32[8,256,512] parameter(0), sharding={devices=[2,2,2]<=[8]}
      %p1 = f32[8,128,512] parameter(1)
      %copy1 = f32[8,128,512] copy(%p1)
      %dot = f32[8,256,128] dot(%p0, %copy1),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2},
        sharding={devices=[2,1,2,2]<=[8] last_tile_dim_replicate}
      ROOT %copy = f32[8,256,128] copy(%dot)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{devices=[2,2,2]<=[8]}"));
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{devices=[2,2,2]<=[8]}"));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{devices=[2,1,2,2]<=[8] last_tile_dim_replicate}"));
}

TEST_F(ShardyXLATest, EntryComputationLayoutSingleResult) {
  const char* const hloString = R"(
    HloModule module, entry_computation_layout={(f32[3,8,32,4]{2,1,3,0:T(8,128)},f32[3,8,32,4]{2,1,3,0:T(8,128)})->f32[3,8,32,4]{2,1,3,0:T(8,128)}}

    ENTRY %entry {
      %p0 = f32[3,8,32,4] parameter(0)
      %p1 = f32[3,8,32,4] parameter(1)
      %copy.p0 = f32[3,8,32,4] copy(%p0)
      %copy.p1 = f32[3,8,32,4] copy(%p1)
      %add = f32[3,8,32,4] add(%copy.p0, %copy.p1), sharding={devices=[2,1,1,1]<=[2]}
      ROOT %result = f32[3,8,32,4] copy(%add)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_EQ(
      module->entry_computation_layout().ToString(),
      "(f32[3,8,32,4]{2,1,3,0:T(8,128)}, "
      "f32[3,8,32,4]{2,1,3,0:T(8,128)})->f32[3,8,32,4]{2,1,3,0:T(8,128)}");
}

TEST_F(ShardyXLATest, EntryComputationLayoutNestedTuple) {
  const char* const hloString = R"(
    HloModule module, entry_computation_layout={((f32[4,2]{0,1:T(2,128)},(f32[4,2]{0,1:T(2,128)},f32[4,2]{0,1:T(2,128)})),f32[4,2]{0,1:T(2,128)})->((f32[4,2]{0,1:T(2,128)},(f32[4,2]{0,1:T(2,128)},f32[4,2]{0,1:T(2,128)})),f32[4,2]{0,1:T(2,128)})}

    ENTRY %main {
      %p0 = (f32[4,2], (f32[4,2], f32[4,2])) parameter(0)
      %p1 = f32[4,2] parameter(1)
      ROOT %result = ((f32[4,2], (f32[4,2], f32[4,2])), f32[4,2]) tuple(%p0, %p1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_EQ(module->entry_computation_layout().ToString(),
            "(f32[4,2]{0,1:T(2,128)}, f32[4,2]{0,1:T(2,128)}, "
            "f32[4,2]{0,1:T(2,128)}, "
            "f32[4,2]{0,1:T(2,128)})->(f32[4,2]{0,1:T(2,128)}, "
            "f32[4,2]{0,1:T(2,128)}, f32[4,2]{0,1:T(2,128)}, "
            "f32[4,2]{0,1:T(2,128)})");
}

TEST_F(ShardyXLATest, EntryComputationLayoutMissingLayout) {
  const char* const hloString = R"(
    HloModule module, entry_computation_layout={(f32[3,8,32,4]{2,1,3,0:T(8,128)},f32[3,8,32,4])->f32[3,8,32,4]}

    ENTRY %entry {
      %p0 = f32[3,8,32,4] parameter(0)
      %p1 = f32[3,8,32,4] parameter(1)
      %copy.p0 = f32[3,8,32,4] copy(%p0)
      %copy.p1 = f32[3,8,32,4] copy(%p1)
      %add = f32[3,8,32,4] add(%copy.p0, %copy.p1), sharding={devices=[2,1,1,1]<=[2]}
      ROOT %result = f32[3,8,32,4] copy(%add)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_EQ(module->entry_computation_layout().ToString(),
            "(f32[3,8,32,4]{2,1,3,0:T(8,128)}, "
            "f32[3,8,32,4]{3,2,1,0})->f32[3,8,32,4]{3,2,1,0}");
}

TEST_F(ShardyXLATest, InputOutputAliasConfigSingleResult) {
  const char* const hloString = R"(
    HloModule module, input_output_alias={ {}: (1, {}, may-alias) }

    ENTRY %entry {
      %p0 = f32[3,8,32,4] parameter(0)
      %p1 = f32[3,8,32,4] parameter(1)
      %add = f32[3,8,32,4] add(%p0, %p1)
      ROOT %result = f32[3,8,32,4] copy(%add)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_EQ(module->input_output_alias_config().ToShortString(),
            "{}: (1, {}, may-alias)");
}

TEST_F(ShardyXLATest, InputOutputAliasConfigSingleResultNestedParams) {
  const char* const hloString = R"(
    HloModule module, input_output_alias={ {}: (0, {1}, may-alias) }

    ENTRY %entry {
      %p0 = (f32[4,2], f32[4,2]) parameter(0)
      %get-tuple-element.0 = f32[4,2] get-tuple-element((f32[4,2], f32[4,2]) %p0), index=0
      %get-tuple-element.1 = f32[4,2] get-tuple-element((f32[4,2], f32[4,2]) %p0), index=1
      %add = f32[4,2] add(%get-tuple-element.0, %get-tuple-element.1)
      ROOT %result = f32[4,2] copy(%add)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_EQ(module->input_output_alias_config().ToShortString(),
            "{}: (1, {}, may-alias)");
}

TEST_F(ShardyXLATest, InputOutputAliasConfigNestedResultAndParams) {
  const char* const hloString = R"(
    HloModule module, input_output_alias={ {0, 1, 0}: (0, {1, 0}, may-alias), {1}: (1, {}, may-alias) }

    ENTRY %main {
      %p0 = (f32[4,2], (f32[4,2], f32[4,2])) parameter(0)
      %p1 = f32[4,2] parameter(1)
      ROOT %result = ((f32[4,2], (f32[4,2], f32[4,2])), f32[4,2]) tuple(%p0, %p1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_EQ(module->input_output_alias_config().ToShortString(),
            "{1}: (1, {}, may-alias), {3}: (3, {}, may-alias)");
}

TEST_F(ShardyXLATest, BufferDonorConfigSingleResult) {
  const char* const hloString = R"(
    HloModule module, buffer_donor={ (1, {}) }

    ENTRY %entry {
      %p0 = f32[3,8,32,4] parameter(0)
      %p1 = f32[3,8,32,4] parameter(1)
      %add = f32[3,8,32,4] add(%p0, %p1)
      ROOT %result = f32[3,8,32,4] copy(%add)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_EQ(module->buffer_donor_config().ToShortString(), "(1, {})");
}

TEST_F(ShardyXLATest, BufferDonorConfigNestedTuple) {
  const char* const hloString = R"(
    HloModule module, buffer_donor={ (0, {0}), (0, {1, 1}) }

    ENTRY %main {
      %p0 = (f32[4,2], (f32[4,2], f32[4,2])) parameter(0)
      %p1 = f32[4,2] parameter(1)
      ROOT %result = ((f32[4,2], (f32[4,2], f32[4,2])), f32[4,2]) tuple(%p0, %p1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_EQ(module->buffer_donor_config().ToShortString(), "(0, {}), (2, {})");
}

TEST_F(ShardyXLATest, ShardingCustomCall) {
  const char* const hloString = R"(
    HloModule module
    ENTRY %main {
      %p0 = f32[8,8] parameter(0), sharding={devices=[2,1]<=[2]}
      %annotate = f32[8,8] custom-call(%p0), custom_call_target="Sharding",
        sharding={devices=[1,2]<=[2]}
      ROOT %add = f32[8,8] add(%p0, %annotate)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_THAT(module->entry_computation()->parameter_instruction(0),
              op::Sharding("{devices=[2,1]<=[2]}"));
  // Sharding custom-call is converted into copy instruction.
  EXPECT_THAT(module->entry_computation()->root_instruction()->operand(1),
              op::Copy());
}

TEST_F(ShardyXLATest, RngBitGenerator) {
  const char* const hloString = R"(
    HloModule module

    ENTRY main {
      state.1 = u64[8]{0} parameter(0), sharding={devices=[8,4]<=[32] last_tile_dim_replicate}
      state.2 = u64[8]{0} add(state.1, state.1), sharding={devices=[2,16]<=[32] last_tile_dim_replicate}
      rng.1 = u32[512,256] rng-bit-generator(state.1), algorithm=rng_default, sharding={devices=[16,2]<=[32]}
      rng.2 = (u64[8]{0}, u32[512,256]) rng-bit-generator(state.2), algorithm=rng_default, sharding={{devices=[4,8]<=[32] last_tile_dim_replicate}, {devices=[8,4]<=[32]}}
      gte = u32[512,256] get-tuple-element(rng.2), index=1
      ROOT tuple = (u32[512,256], u32[512,256]) tuple(rng.1, gte)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sharding("{{devices=[16,2]<=[32]}, {devices=[8,4]<=[32]}}"));
}

TEST_F(ShardyXLATest, WhileWithFreeVariables) {
  const char* const hloString = R"(
    HloModule main, entry_computation_layout={(f32[32,96]{1,0}, f32[32,96]{1,0})->f32[32,96]{1,0}}

    %region_0.7 (arg_tuple.8: (f32[32,96], s32[], s32[], s32[], f32[32,96])) -> (f32[32,96], s32[], s32[], s32[], f32[32,96]) {
      %arg_tuple.8 = (f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) parameter(0)
      %get-tuple-element.9 = f32[32,96]{1,0} get-tuple-element((f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) %arg_tuple.8), index=0
      %get-tuple-element.13 = f32[32,96]{1,0} get-tuple-element((f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) %arg_tuple.8), index=4
      %add.15 = f32[32,96]{1,0} add(f32[32,96]{1,0} %get-tuple-element.9, f32[32,96]{1,0} %get-tuple-element.13)
      %get-tuple-element.10 = s32[] get-tuple-element((f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) %arg_tuple.8), index=1
      %get-tuple-element.12 = s32[] get-tuple-element((f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) %arg_tuple.8), index=3
      %add.14 = s32[] add(s32[] %get-tuple-element.10, s32[] %get-tuple-element.12)
      %get-tuple-element.11 = s32[] get-tuple-element((f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) %arg_tuple.8), index=2
      ROOT %tuple.16 = (f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) tuple(f32[32,96]{1,0} %add.15, s32[] %add.14, s32[] %get-tuple-element.11, s32[] %get-tuple-element.12, f32[32,96]{1,0} %get-tuple-element.13)
    }

    %region_1.17 (arg_tuple.18: (f32[32,96], s32[], s32[], s32[], f32[32,96])) -> pred[] {
      %arg_tuple.18 = (f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) parameter(0)
      %get-tuple-element.19 = f32[32,96]{1,0} get-tuple-element((f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) %arg_tuple.18), index=0
      %get-tuple-element.22 = s32[] get-tuple-element((f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) %arg_tuple.18), index=3
      %get-tuple-element.23 = f32[32,96]{1,0} get-tuple-element((f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) %arg_tuple.18), index=4
      %get-tuple-element.20 = s32[] get-tuple-element((f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) %arg_tuple.18), index=1
      %get-tuple-element.21 = s32[] get-tuple-element((f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) %arg_tuple.18), index=2
      ROOT %compare.24 = pred[] compare(s32[] %get-tuple-element.20, s32[] %get-tuple-element.21), direction=LT
    }

    ENTRY %main.30 (Arg_0.1: f32[32,96], Arg_1.2: f32[32,96]) -> f32[32,96] {
      %Arg_0.1 = f32[32,96]{1,0} parameter(0), sharding={devices=[2,2]<=[4]}
      %constant.3 = s32[] constant(0)
      %constant.5 = s32[] constant(32)
      %constant.4 = s32[] constant(1)
      %Arg_1.2 = f32[32,96]{1,0} parameter(1), sharding={devices=[2,1,2]<=[4] last_tile_dim_replicate}
      %tuple.6 = (f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) tuple(f32[32,96]{1,0} %Arg_0.1, s32[] %constant.3, s32[] %constant.5, s32[] %constant.4, f32[32,96]{1,0} %Arg_1.2)
      %while.25 = (f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) while((f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) %tuple.6), condition=%region_1.17, body=%region_0.7
      %get-tuple-element.27 = s32[] get-tuple-element((f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) %while.25), index=1
      %get-tuple-element.26 = f32[32,96]{1,0} get-tuple-element((f32[32,96]{1,0}, s32[], s32[], s32[], f32[32,96]{1,0}) %while.25), index=0
      %tuple.28 = (f32[32,96]{1,0}) tuple(f32[32,96]{1,0} %get-tuple-element.26)
      ROOT %get-tuple-element.29 = f32[32,96]{1,0} get-tuple-element((f32[32,96]{1,0}) %tuple.28), index=0
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  HloInstruction* whileInst =
      FindInstruction(module.get(), xla::HloOpcode::kWhile);
  EXPECT_NE(whileInst, nullptr);
  // Verify that the sharding of parameter(1) hasn't changed.
  EXPECT_THAT(module->entry_computation()->parameter_instruction(1),
              op::Sharding("{devices=[2,1,2]<=[4] last_tile_dim_replicate}"));
  // Verify the sharding of the while, and specifically that the sharding of the
  // result that corresponds to parameter(1) is further sharded.
  EXPECT_THAT(whileInst, op::Sharding("{{devices=[2,2]<=[4]}, {replicated}, "
                                      "{devices=[2,2]<=[4]}}"));
}

TEST_F(ShardyXLATest, ShardMap) {
  const char* const hloString = R"(
    HloModule shard_map

    region_add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    shmap_body.11 {
      Arg_0.12 = f32[2,8] parameter(0)
      add.14 = f32[2,8] add(Arg_0.12, Arg_0.12)
      Arg_1.13 = f32[8,32] parameter(1)
      dot.15 = f32[2,32] dot(add.14, Arg_1.13), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT all-reduce.16 = f32[2,32] all-reduce(dot.15), channel_id=1, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, to_apply=region_add
    }

    ENTRY main {
      p0 = f32[8,16] parameter(0)
      custom-call.3 = f32[8,16] custom-call(p0), custom_call_target="Sharding", sharding={devices=[4,2]<=[8]}
      custom-call.4 = f32[2,8] custom-call(custom-call.3), custom_call_target="SPMDFullToShardShape", sharding={manual}
      p1 = f32[16,32] parameter(1)
      custom-call.5 = f32[16,32] custom-call(p1), custom_call_target="Sharding", sharding={devices=[2,1,4]<=[4,2]T(1,0) last_tile_dim_replicate}
      custom-call.6 = f32[8,32] custom-call(custom-call.5), custom_call_target="SPMDFullToShardShape", sharding={manual}
      call.17 = f32[2,32] call(custom-call.4, custom-call.6), to_apply=shmap_body.11
      custom-call.18 = f32[2,32] custom-call(call.17), custom_call_target="Sharding", sharding={manual}
      ROOT custom-call.19 = f32[8,32] custom-call(custom-call.18), custom_call_target="SPMDShardToFullShape", sharding={devices=[4,1,2]<=[8] last_tile_dim_replicate}
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  // The entry computation and the region_add for the all-reduce. shmap_body is
  // inlined.
  EXPECT_EQ(module->computation_count(), 2);
  EXPECT_EQ(FindInstruction(module.get(), xla::HloOpcode::kCall), nullptr);

  auto* dot = FindInstruction(module.get(), xla::HloOpcode::kDot);
  EXPECT_NE(dot, nullptr);
  EXPECT_TRUE(dot->has_sharding());
  EXPECT_TRUE(dot->sharding().IsManual());

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), xla::HloOpcode::kCustomCall);
  EXPECT_EQ(root->custom_call_target(), "SPMDShardToFullShape");
  EXPECT_THAT(root,
              op::Sharding("{devices=[4,1,2]<=[8] last_tile_dim_replicate}"));
}

// Be able to handle an empty computation layout and input_output_alias_config.
TEST_F(ShardyXLATest, EmptyModule) {
  const char* const hloString = R"(
    HloModule pjit_f, entry_computation_layout={()->()}, num_partitions=2

    ENTRY %main.2 () -> () {
      ROOT %tuple.1 = () tuple()
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_EQ(module->entry_computation_layout().ToString(), "()->()");
  EXPECT_EQ(module->input_output_alias_config().ToShortString(), "");
}

// Due to the module having `xla.sdy.use_tuple_args`, this means the output HLO
// has to have a single tuple param that wraps the flattened parameters. Check
// we can handle this. Specifically for:
// - input_output_alias
// - entry_computation_layout
// - buffer_donor
TEST_F(ShardyXLATest, TestUseTuplesTrue) {
  const char* const hloString = R"(
    HloModule pjit_f, buffer_donor={ (1, {}) }, input_output_alias={ {}: (2, {}, must-alias) }, entry_computation_layout={(f32[8,16]{1,0:T(8,128)}, f32[16,32]{1,0:T(8,128)}, f32[8,32]{1,0:T(8,128)})->f32[8,32]{1,0:T(8,128)}}, allow_spmd_sharding_propagation_to_parameters={false,false,false}, num_partitions=8, frontend_attributes={xla.sdy.use_tuple_args="t"}

    ENTRY %main.7 (Arg_0.1: f32[8,16], Arg_1.2: f32[16,32], Arg_2.3: f32[8,32]) -> f32[8,32] {
      %Arg_0.1 = f32[8,16]{1,0} parameter(0)
      %Arg_1.2 = f32[16,32]{1,0} parameter(1)
      %dot.4 = f32[8,32]{1,0} dot(f32[8,16]{1,0} %Arg_0.1, f32[16,32]{1,0} %Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      %Arg_2.3 = f32[8,32]{1,0} parameter(2)
      ROOT %add.5 = f32[8,32]{1,0} add(f32[8,32]{1,0} %dot.4, f32[8,32]{1,0} %Arg_2.3)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get());

  EXPECT_EQ(module->entry_computation()->parameter_instructions().size(), 1);
  EXPECT_EQ(module->buffer_donor_config().ToShortString(), "(0, {1})");
  EXPECT_EQ(module->input_output_alias_config().ToShortString(),
            "{}: (0, {2}, must-alias)");
  EXPECT_EQ(module->entry_computation_layout().ToString(),
            "((f32[8,16]{1,0:T(8,128)}, f32[16,32]{1,0:T(8,128)}, "
            "f32[8,32]{1,0:T(8,128)}))->f32[8,32]{1,0:T(8,128)}");
}

TEST_F(ShardyXLATest, TestRunShardingPropagationFalseUseTuplesFalse) {
  const char* const hloString = R"(
    HloModule pjit_f, buffer_donor={ (1, {}) }, input_output_alias={ {}: (2, {}, must-alias) }, entry_computation_layout={(f32[8,16]{1,0:T(8,128)}, f32[16,32]{1,0:T(8,128)}, f32[8,32]{1,0:T(8,128)})->f32[8,32]{1,0:T(8,128)}}, allow_spmd_sharding_propagation_to_parameters={false,false,false}, num_partitions=8

    ENTRY %main.7 (Arg_0.1: f32[8,16], Arg_1.2: f32[16,32], Arg_2.3: f32[8,32]) -> f32[8,32] {
      %Arg_0.1 = f32[8,16]{1,0} parameter(0)
      %Arg_1.2 = f32[16,32]{1,0} parameter(1)
      %dot.4 = f32[8,32]{1,0} dot(f32[8,16]{1,0} %Arg_0.1, f32[16,32]{1,0} %Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      %Arg_2.3 = f32[8,32]{1,0} parameter(2)
      ROOT %add.5 = f32[8,32]{1,0} add(f32[8,32]{1,0} %dot.4, f32[8,32]{1,0} %Arg_2.3)
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get(),
                               /*runSdyShardingPropagation=*/false,
                               /*expectChanged=*/false);

  EXPECT_EQ(module->entry_computation()->parameter_instructions().size(), 3);
  EXPECT_EQ(module->buffer_donor_config().ToShortString(), "(1, {})");
  EXPECT_EQ(module->input_output_alias_config().ToShortString(),
            "{}: (2, {}, must-alias)");
  EXPECT_EQ(module->entry_computation_layout().ToString(),
            "(f32[8,16]{1,0:T(8,128)}, f32[16,32]{1,0:T(8,128)}, "
            "f32[8,32]{1,0:T(8,128)})->f32[8,32]{1,0:T(8,128)}");
}

TEST_F(ShardyXLATest, TestRunShardingPropagationFalseUseTuplesTrue) {
  const char* const hloString = R"(
    HloModule pjit_f, buffer_donor={ (1, {}) }, input_output_alias={ {}: (2, {}, must-alias) }, entry_computation_layout={(f32[8,16]{1,0:T(8,128)}, f32[16,32]{1,0:T(8,128)}, f32[8,32]{1,0:T(8,128)})->f32[8,32]{1,0:T(8,128)}}, allow_spmd_sharding_propagation_to_parameters={false,false,false}, num_partitions=8, frontend_attributes={xla.sdy.use_tuple_args="t"}

    ENTRY %main.7 (Arg_0.1: f32[8,16], Arg_1.2: f32[16,32], Arg_2.3: f32[8,32]) -> f32[8,32] {
      %Arg_0.1 = f32[8,16]{1,0} parameter(0)
      %Arg_1.2 = f32[16,32]{1,0} parameter(1)
      %dot.4 = f32[8,32]{1,0} dot(f32[8,16]{1,0} %Arg_0.1, f32[16,32]{1,0} %Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      %Arg_2.3 = f32[8,32]{1,0} parameter(2)
      ROOT %add.5 = f32[8,32]{1,0} add(f32[8,32]{1,0} %dot.4, f32[8,32]{1,0} %Arg_2.3)
    })";
  const char* const expected = R"(
  // CHECK:       HloModule pjit_f,
  // CHECK-SAME:    input_output_alias={ {}: (0, {2}, must-alias) },
  // CHECK-SAME:    buffer_donor={ (0, {1}) },
  // CHECK-SAME:    entry_computation_layout={((f32[8,16]{1,0:T(8,128)}, f32[16,32]{1,0:T(8,128)}, f32[8,32]{1,0:T(8,128)}))->f32[8,32]{1,0:T(8,128)}},
  // CHECK-SAME:    allow_spmd_sharding_propagation_to_parameters={false,false,false}, num_partitions=8
  //
  // CHECK:       ENTRY %main.0 (arg_tuple.1: (f32[8,16], f32[16,32], f32[8,32])) -> f32[8,32] {
  // CHECK-NEXT:    %arg_tuple.1 = (f32[8,16], f32[16,32], f32[8,32]) parameter(0)
  // CHECK-NEXT:    %get-tuple-element.2 = f32[8,16] get-tuple-element(%arg_tuple.1), index=0
  // CHECK-NEXT:    %get-tuple-element.3 = f32[16,32] get-tuple-element(%arg_tuple.1), index=1
  // CHECK-NEXT:    %dot.5 = f32[8,32] dot(%get-tuple-element.2, %get-tuple-element.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  // CHECK-NEXT:    %get-tuple-element.4 = f32[8,32] get-tuple-element(%arg_tuple.1), index=2
  // CHECK-NEXT:    ROOT %add.6 = f32[8,32] add(%dot.5, %get-tuple-element.4)
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithStablehloImport(module.get(),
                               /*runSdyShardingPropagation=*/false);
  LOG(INFO) << module->ToString(
      HloPrintOptions{}.set_include_layout_in_shapes(false));
  EXPECT_TRUE(*RunFileCheck(
      module->ToString(HloPrintOptions{}.set_include_layout_in_shapes(false)),
      expected));
}

TEST_F(ShardyXLATest, TestMaximalShardingNoResults) {
  const char* const hloString = R"(
HloModule maximal_sharding_module, entry_computation_layout={(s64[2]{0})->s64[2]{0}},
    frontend_attributes={xla.sdy.meshes={maximal_mesh_0 = #sdy.mesh<[], device_ids=[0]>}}

ENTRY %main.0 (Arg_0.0: s64[2]) -> s64[2] {
  ROOT %Arg_0.0 = s64[2] parameter(0)
  %custom-call.0 = () custom-call(s64[2] %Arg_0.0), custom_call_target="xla_ffi_python_cpu_callback",
      operand_layout_constraints={s64[2]{0}}, custom_call_has_side_effect=true, api_version=API_VERSION_TYPED_FFI,
      frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<@maximal_mesh_0, []>]>"},
      sharding={{maximal device=0}}
}
)";
  const char* const expected = R"(
  // CHECK:               ENTRY %main.3 (Arg_0.1: s64[2]) -> s64[2] {
  // CHECK-NEXT:            ROOT %Arg_0.1 = s64[2] parameter(0)
  // CHECK-NEXT{LITERAL}:   %custom-call.2 = () custom-call(%Arg_0.1), custom_call_target="xla_ffi_python_cpu_callback",
  // CHECK-SAME{LITERAL}:   operand_layout_constraints={s64[2]{0}}, custom_call_has_side_effect=true, api_version=API_VERSION_TYPED_FFI,
  // CHECK-SAME{LITERAL}:   sharding={{maximal device=0}}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithSdyImport(module.get());
  LOG(INFO) << module->ToString(
      HloPrintOptions{}.set_include_layout_in_shapes(false));
  EXPECT_TRUE(*RunFileCheck(
      module->ToString(HloPrintOptions{}.set_include_layout_in_shapes(false)),
      expected));
}

TEST_F(ShardyXLATest, WhileShardingOnlyOnFreeVariables) {
  const char* const hloString = R"(
    HloModule main, entry_computation_layout={(f32[32,96]{1,0}, f32[32,96]{1,0})->f32[32,96]{1,0}}, frontend_attributes={xla.sdy.meshes={mesh = #sdy.mesh<["x"=4]>}}

    %region_0.6 (arg_tuple.7: (f32[32,96], s32[], f32[32,96])) -> (f32[32,96], s32[], f32[32,96]) {
      %arg_tuple.7 = (f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) parameter(0)
      %get-tuple-element.8 = f32[32,96]{1,0} get-tuple-element((f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) %arg_tuple.7), index=0
      %sine.11 = f32[32,96]{1,0} sine(f32[32,96]{1,0} %get-tuple-element.8)
      %custom-call.12 = f32[32,96]{1,0} custom-call(f32[32,96]{1,0} %sine.11), custom_call_target="Sharding", sharding={replicated}, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"}
      %get-tuple-element.10 = f32[32,96]{1,0} get-tuple-element((f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) %arg_tuple.7), index=2
      %add.13 = f32[32,96]{1,0} add(f32[32,96]{1,0} %custom-call.12, f32[32,96]{1,0} %get-tuple-element.10)
      %custom-call.14 = f32[32,96]{1,0} custom-call(f32[32,96]{1,0} %add.13), custom_call_target="Sharding", sharding={replicated}, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"}
      %get-tuple-element.9 = s32[] get-tuple-element((f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) %arg_tuple.7), index=1
      %constant.15 = s32[] constant(1)
      %add.16 = s32[] add(s32[] %get-tuple-element.9, s32[] %constant.15)
      ROOT %tuple.17 = (f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) tuple(f32[32,96]{1,0} %custom-call.14, s32[] %add.16, f32[32,96]{1,0} %get-tuple-element.10)
    }

    %region_1.18 (arg_tuple.19: (f32[32,96], s32[], f32[32,96])) -> pred[] {
      %arg_tuple.19 = (f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) parameter(0)
      %get-tuple-element.20 = f32[32,96]{1,0} get-tuple-element((f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) %arg_tuple.19), index=0
      %get-tuple-element.22 = f32[32,96]{1,0} get-tuple-element((f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) %arg_tuple.19), index=2
      %get-tuple-element.21 = s32[] get-tuple-element((f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) %arg_tuple.19), index=1
      %constant.23 = s32[] constant(32)
      ROOT %compare.24 = pred[] compare(s32[] %get-tuple-element.21, s32[] %constant.23), direction=LT
    }

    ENTRY %main.28 (Arg_0.1: f32[32,96], Arg_1.2: f32[32,96]) -> f32[32,96] {
      %Arg_0.1 = f32[32,96]{1,0} parameter(0)
      %constant.3 = s32[] constant(0)
      %Arg_1.2 = f32[32,96]{1,0} parameter(1), sharding={devices=[4,1]<=[4]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<@mesh, [{\"x\", ?}, {?}]>"}
      %custom-call.4 = f32[32,96]{1,0} custom-call(f32[32,96]{1,0} %Arg_1.2), custom_call_target="Sharding", sharding={replicated}, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<@mesh, [{?}, {?}]>]>"}
      %tuple.5 = (f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) tuple(f32[32,96]{1,0} %Arg_0.1, s32[] %constant.3, f32[32,96]{1,0} %custom-call.4)
      %while.25 = (f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) while((f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) %tuple.5), condition=%region_1.18, body=%region_0.6
      ROOT %get-tuple-element.26 = f32[32,96]{1,0} get-tuple-element((f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) %while.25), index=0
      %get-tuple-element.27 = s32[] get-tuple-element((f32[32,96]{1,0}, s32[], f32[32,96]{1,0}) %while.25), index=1
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithSdyImport(module.get());

  HloInstruction* whileInst =
      FindInstruction(module.get(), xla::HloOpcode::kWhile);
  EXPECT_NE(whileInst, nullptr);
  // Verify the sharding of the while, and specifically that the sharding of the
  // result that corresponds to parameter(1) is further sharded.
  EXPECT_THAT(whileInst, op::Sharding("{{replicated}, {replicated}, "
                                      "{devices=[4,1]<=[4]}}"));
}

TEST_F(ShardyXLATest, EmptyResultLayout) {
  const char* const hloString = R"(
    HloModule pjit_f_, entry_computation_layout={(s64[2,2,2]{2,1,0})->()}, num_partitions=2, frontend_attributes={xla.sdy.meshes={maximal_mesh_0 = #sdy.mesh<[], device_ids=[0]>, mesh = #sdy.mesh<["x"=2]>}}

    ENTRY %main.5 (Arg_0.1: s64[2,2,2]) -> () {
      %Arg_0.0 = s64[2,2,2]{2,1,0} parameter(0), sharding={devices=[2,1,1]<=[2]}, metadata={op_name="x"}
      ROOT %tuple.0 = () tuple()
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithSdyImport(module.get());

  EXPECT_EQ(module.get()->entry_computation_layout().ToString(),
            "(s64[2,2,2]{2,1,0})->()");
}

TEST_F(ShardyXLATest, EmptyOperandLayout) {
  const char* const hloString = R"(
    HloModule pjit_f_, entry_computation_layout={()->s64[2,2]{1,0}}, num_partitions=2, frontend_attributes={xla.sdy.meshes={maximal_mesh_0 = #sdy.mesh<[], device_ids=[0]>, mesh = #sdy.mesh<["x"=2]>}}

    ENTRY %main.5 () -> s64[2,2] {
      ROOT %constant = s64[2,2]{1,0} constant({{1,1},{1,1}})
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithSdyImport(module.get());

  EXPECT_EQ(module.get()->entry_computation_layout().ToString(),
            "()->s64[2,2]{1,0}");
}

TEST_F(ShardyXLATest, RaggedDotMode1) {
  const char* const hloString = R"(
  HloModule ragged_dot, allow_spmd_sharding_propagation_to_parameters={true,true,true}, allow_spmd_sharding_propagation_to_output={true}, frontend_attributes={xla.sdy.meshes={mesh = #sdy.mesh<["a"=2, "b"=2, "c"=2]>}}
    ENTRY entry {
      p0 = f32[16,32,64] parameter(0), frontend_attributes={xla.sdy.sharding="#sdy.sharding<@mesh, [{\"a\", ?}, {\"b\", ?}, {\"c\", ?}]>"}
      p1 = f32[4,16,64,8] parameter(1)
      p2 = s32[16,4] parameter(2)
      ROOT ragged-dot = f32[16,32,8] ragged-dot(p0, p1, p2), lhs_batch_dims={0}, rhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_contracting_dims={2}, lhs_ragged_dims={1}, rhs_group_dims={0}
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hloString));
  runShardyWithSdyImport(module.get());

  HloComputation* entry = module->entry_computation();
  EXPECT_THAT(
      entry->parameter_instruction(1),
      op::Sharding(
          "{devices=[1,2,2,1,2]<=[2,2,2]T(0,2,1) last_tile_dim_replicate}"));
  EXPECT_THAT(entry->parameter_instruction(2),
              op::Sharding("{devices=[2,1,4]<=[8] last_tile_dim_replicate}"));
  EXPECT_THAT(entry->root_instruction(),
              op::Sharding("{devices=[2,2,1,2]<=[8] last_tile_dim_replicate}"));
}

}  // namespace sdy
}  // namespace xla
