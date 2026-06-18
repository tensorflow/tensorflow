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

#include "xla/backends/gpu/transforms/dynamic_slice_copy.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/transforms/dynamic_slice_annotator.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;

using DynamicSliceCopyTest = HloHardwareIndependentTestBase;
using Offset = DynamicSliceFusion::Offset;
using Offsets = std::vector<Offset>;
using Parameter = DynamicSliceFusion::Parameter;
using Result = DynamicSliceFusion::Result;

DynamicSliceConfig MakeConfig(int64_t loop_index, int64_t offset,
                              int64_t stride) {
  DynamicSliceConfig config;
  config.set_loop_index(loop_index);
  config.set_byte_offset(offset);
  config.set_byte_stride(stride);
  return config;
}

DynamicSliceConfig MakeStaticConfig(int64_t offset) {
  DynamicSliceConfig config;
  config.set_byte_offset(offset);
  return config;
}

TEST_F(DynamicSliceCopyTest, AnalyzesDynamicSliceRootCopy) {
  constexpr char kHlo[] = R"(
    dynamic_slice {
      p0 = s32[4] parameter(0)
      c1 = s32[] constant(1)

      ROOT slice = s32[1] dynamic-slice(p0, c1), dynamic_slice_sizes={1},
          backend_config={"dynamic_slice_config":
              {"byte_offset":"4","byte_stride":"0"}}
    }

    ENTRY main {
      p0 = s32[4] parameter(0)
      ROOT fusion = s32[1] fusion(p0), kind=kLoop, calls=dynamic_slice
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::optional<DynamicSliceCopyFusion> copy,
                       AnalyzeDynamicSliceCopyFusion(fusion));
  ASSERT_TRUE(copy.has_value());
  EXPECT_EQ(copy->copy_operand->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_THAT(
      copy->parameters,
      ElementsAre(Parameter{0, ShapeUtil::MakeShape(S32, {4}),
                            ShapeUtil::MakeShape(S32, {1}), MakeStaticConfig(4),
                            Offsets{{0, Offset::Constant(1)}}}));
  EXPECT_THAT(copy->results,
              ElementsAre(Result{
                  std::nullopt, 0, ShapeUtil::MakeShape(S32, {1}),
                  ShapeUtil::MakeShape(S32, {1}), std::nullopt, std::nullopt}));
}

TEST_F(DynamicSliceCopyTest, AnalyzesDynamicUpdateSliceRootCopy) {
  constexpr char kHlo[] = R"(
    dynamic_update_slice {
      p0 = s32[4,8,8] parameter(0)
      p1 = s32[1,1,8] parameter(1)
      p2 = s32[] parameter(2)
      c1 = s32[] constant(1)

      ROOT update-slice = s32[4,8,8] dynamic-update-slice(p0, p1, p2, c1, c1),
          backend_config={"dynamic_slice_config":
              {"byte_offset":"32","byte_stride":"256","loop_index":"0"}}
    }

    ENTRY main {
      input = s32[4,8,8] parameter(0)
      val = s32[1,1,8] parameter(1)
      ivar = s32[] parameter(2)
      ROOT updated = s32[4,8,8] fusion(input, val, ivar), kind=kLoop,
          calls=dynamic_update_slice
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::optional<DynamicSliceCopyFusion> copy,
                       AnalyzeDynamicSliceCopyFusion(fusion));
  ASSERT_TRUE(copy.has_value());
  EXPECT_EQ(copy->copy_operand->opcode(), HloOpcode::kParameter);
  EXPECT_THAT(copy->parameters,
              ElementsAre(Parameter{1, ShapeUtil::MakeShape(S32, {1, 1, 8}),
                                    ShapeUtil::MakeShape(S32, {1, 1, 8}),
                                    std::nullopt, std::nullopt}));
  EXPECT_THAT(copy->results,
              ElementsAre(Result{0, 0, ShapeUtil::MakeShape(S32, {4, 8, 8}),
                                 ShapeUtil::MakeShape(S32, {1, 1, 8}),
                                 MakeConfig(0, 32, 256),
                                 Offsets{{0, Offset::Parameter(2)},
                                         {1, Offset::Constant(1)},
                                         {2, Offset::Constant(1)}}}));
}

TEST_F(DynamicSliceCopyTest, AnalyzesComputedDusOffset) {
  constexpr char kHlo[] = R"(
    dynamic_update_slice {
      p0 = s32[4,8,8] parameter(0)
      p1 = s32[1,1,8] parameter(1)
      p2 = s32[] parameter(2)
      c1 = s32[] constant(1)
      offset = s32[] add(p2, c1)

      ROOT update-slice = s32[4,8,8] dynamic-update-slice(p0, p1, offset, c1, c1),
          backend_config={"dynamic_slice_config":
              {"byte_offset":"32","byte_stride":"256","loop_index":"0"}}
    }

    ENTRY main {
      input = s32[4,8,8] parameter(0)
      val = s32[1,1,8] parameter(1)
      ivar = s32[] parameter(2)
      ROOT updated = s32[4,8,8] fusion(input, val, ivar), kind=kLoop,
          calls=dynamic_update_slice
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  ASSERT_OK_AND_ASSIGN(std::optional<DynamicSliceCopyFusion> copy,
                       AnalyzeDynamicSliceCopyFusion(
                           module->entry_computation()->root_instruction()));
  ASSERT_TRUE(copy.has_value());
  EXPECT_THAT(
      copy->results,
      ElementsAre(Result{
          0, 0, ShapeUtil::MakeShape(S32, {4, 8, 8}),
          ShapeUtil::MakeShape(S32, {1, 1, 8}), MakeConfig(0, 32, 256),
          Offsets{{0, Offset::Add(Offset::Parameter(2), Offset::Constant(1))},
                  {1, Offset::Constant(1)},
                  {2, Offset::Constant(1)}}}));
}

TEST_F(DynamicSliceCopyTest, RejectsComputedDusUpdate) {
  constexpr char kHlo[] = R"(
    dynamic_update_slice {
      p0 = s32[4] parameter(0)
      p1 = s32[1] parameter(1)
      one = s32[1] constant({1})
      update = s32[1] add(p1, one)
      c0 = s32[] constant(0)

      ROOT update-slice = s32[4] dynamic-update-slice(p0, update, c0),
          backend_config={"dynamic_slice_config":
              {"byte_offset":"0","byte_stride":"4","loop_index":"0"}}
    }

    ENTRY main {
      input = s32[4] parameter(0)
      val = s32[1] parameter(1)
      ROOT updated = s32[4] fusion(input, val), kind=kLoop,
          calls=dynamic_update_slice
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::optional<DynamicSliceCopyFusion> copy,
                       AnalyzeDynamicSliceCopyFusion(
                           module->entry_computation()->root_instruction()));
  EXPECT_FALSE(copy.has_value());
}

TEST_F(DynamicSliceCopyTest, AnalyzesDusWithStaticSliceUpdate) {
  constexpr char kHlo[] = R"(
    dynamic_update_slice {
      p0 = s32[4,8] parameter(0)
      p1 = s32[4,8] parameter(1)
      p2 = s32[] parameter(2)
      c0 = s32[] constant(0)

      update = s32[1,8] slice(p1), slice={[1:2], [0:8]}

      ROOT update-slice = s32[4,8] dynamic-update-slice(p0, update, p2, c0),
          backend_config={"dynamic_slice_config":
              {"byte_offset":"0","byte_stride":"32","loop_index":"0"}}
    }

    ENTRY main {
      input = s32[4,8] parameter(0)
      val = s32[4,8] parameter(1)
      ivar = s32[] parameter(2)
      ROOT updated = s32[4,8] fusion(input, val, ivar), kind=kLoop,
          calls=dynamic_update_slice
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  ASSERT_OK_AND_ASSIGN(std::optional<DynamicSliceCopyFusion> copy,
                       AnalyzeDynamicSliceCopyFusion(
                           module->entry_computation()->root_instruction()));
  ASSERT_TRUE(copy.has_value());
  EXPECT_THAT(copy->parameters,
              ElementsAre(Parameter{1, ShapeUtil::MakeShape(S32, {4, 8}),
                                    ShapeUtil::MakeShape(S32, {1, 8}),
                                    MakeStaticConfig(32), std::nullopt}));
}

TEST_F(DynamicSliceCopyTest, RejectsStridedStaticSliceUpdate) {
  constexpr char kHlo[] = R"(
    dynamic_update_slice {
      p0 = s32[4,8] parameter(0)
      p1 = s32[4,8] parameter(1)
      p2 = s32[] parameter(2)
      c0 = s32[] constant(0)

      update = s32[2,8] slice(p1), slice={[0:4:2], [0:8]}

      ROOT update-slice = s32[4,8] dynamic-update-slice(p0, update, p2, c0),
          backend_config={"dynamic_slice_config":
              {"byte_offset":"0","byte_stride":"32","loop_index":"0"}}
    }

    ENTRY main {
      input = s32[4,8] parameter(0)
      val = s32[4,8] parameter(1)
      ivar = s32[] parameter(2)
      ROOT updated = s32[4,8] fusion(input, val, ivar), kind=kLoop,
          calls=dynamic_update_slice
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::optional<DynamicSliceCopyFusion> copy,
                       AnalyzeDynamicSliceCopyFusion(
                           module->entry_computation()->root_instruction()));
  EXPECT_FALSE(copy.has_value());
}

TEST_F(DynamicSliceCopyTest, AnalyzesStaticSliceRootCopy) {
  constexpr char kHlo[] = R"(
    wrapped_slice_computation {
      param_0 = f32[8,16]{1,0} parameter(0)
      ROOT slice = f32[3,16]{1,0} slice(param_0),
          slice={[2:5], [0:16]}
    }

    ENTRY main {
      param_0 = f32[8,16]{1,0} parameter(0)
      ROOT wrapped_slice = f32[3,16]{1,0} fusion(param_0),
          kind=kLoop, calls=wrapped_slice_computation
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());

  ASSERT_OK_AND_ASSIGN(std::optional<StaticSliceCopyFusion> copy,
                       AnalyzeStaticSliceCopyFusion(fusion));
  ASSERT_TRUE(copy.has_value());
  EXPECT_EQ(copy->parameter_number, 0);
  EXPECT_EQ(copy->source_byte_offset, 128);
  EXPECT_TRUE(
      ShapeUtil::Equal(copy->slice_shape, ShapeUtil::MakeShape(F32, {3, 16})));
}

TEST_F(DynamicSliceCopyTest, DynamicVariableUsesPerVariableInitStep) {
  constexpr char kHlo[] = R"(
    dynamic_slice_comp {
      p0 = s32[4,8,8] parameter(0)
      p1 = s32[] parameter(1)
      c0 = s32[] constant(0)
      ROOT slice = s32[1,8,8] dynamic-slice(p0, p1, c0, c0),
          dynamic_slice_sizes={1,8,8}
    }

    body {
      p0 = (s32[], s32[4,8,8], s32[]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[4,8,8] get-tuple-element(p0), index=1
      counter = s32[] get-tuple-element(p0), index=2

      sliced = s32[1,8,8] fusion(input, counter), kind=kLoop,
          calls=dynamic_slice_comp

      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      next_counter = s32[] add(counter, c1)

      ROOT result = (s32[], s32[4,8,8], s32[])
          tuple(next_ivar, input, next_counter)
    }

    condition {
      p0 = (s32[], s32[4,8,8], s32[]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c3 = s32[] constant(3)
      ROOT cmp = pred[] compare(ivar, c3), direction=LT
    }

    ENTRY main {
      input = s32[4,8,8] parameter(0)
      c2 = s32[] constant(2)
      c3 = s32[] constant(3)
      tuple = (s32[], s32[4,8,8], s32[]) tuple(c2, input, c3)
      ROOT while = (s32[], s32[4,8,8], s32[]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"3"},
                          "known_init_step":{"init":"2","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"},
                          "dynamic_variables":[{"tuple_index":"2","init":"3","step":"1"}]}
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(DynamicSliceAnnotator().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* fusion =
      module->GetComputationWithName("body")->GetInstructionWithName("sliced");
  ASSERT_OK_AND_ASSIGN(std::optional<DynamicSliceCopyFusion> copy,
                       AnalyzeDynamicSliceCopyFusion(fusion));
  ASSERT_TRUE(copy.has_value());
  EXPECT_THAT(copy->parameters,
              ElementsAre(Parameter{0, ShapeUtil::MakeShape(S32, {4, 8, 8}),
                                    ShapeUtil::MakeShape(S32, {1, 8, 8}),
                                    MakeConfig(0, 768, 256),
                                    Offsets{{0, Offset::Parameter(1)},
                                            {1, Offset::Constant(0)},
                                            {2, Offset::Constant(0)}}}));
}

}  // namespace
}  // namespace xla::gpu
