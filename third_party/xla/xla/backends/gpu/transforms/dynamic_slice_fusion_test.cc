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

#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using Parameter = DynamicSliceFusion::Parameter;
using Result = DynamicSliceFusion::Result;
using Offset = DynamicSliceFusion::Offset;

using Offsets = std::vector<Offset>;

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

class DynamicSliceFusionTest : public HloHardwareIndependentTestBase {};

//===----------------------------------------------------------------------===//
// DynamicSliceFusion::FindHero tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionTest, FindHeroCustomCall) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %fill = f32[4] custom-call(), custom_call_target="fill"
      %bitcast = f32[1,4] bitcast(%fill)
      %zero = s32[] constant(0)
      ROOT %dus = f32[4,4] dynamic-update-slice(%p0, %bitcast, %zero, %zero)
    }

    ENTRY main {
      %input = f32[4,4] parameter(0)
      ROOT %fusion = f32[4,4] fusion(%input), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  ASSERT_NE(body, nullptr);

  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);
  EXPECT_EQ(hero->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(hero->custom_call_target(), "fill");
}

TEST_F(DynamicSliceFusionTest, FindHeroSkipsInfrastructure) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[8,8] parameter(0)
      %p1 = f32[8,8] parameter(1)
      %slice = f32[4,8] slice(%p0), slice={[0:4], [0:8]}
      %bitcast = f32[4,8] bitcast(%slice)
      %dot = f32[4,8] dot(%bitcast, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      %zero = s32[] constant(0)
      ROOT %dus = f32[8,8] dynamic-update-slice(%p0, %dot, %zero, %zero)
    }

    ENTRY main {
      %a = f32[8,8] parameter(0)
      %b = f32[8,8] parameter(1)
      ROOT %fusion = f32[8,8] fusion(%a, %b), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");

  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);
  EXPECT_EQ(hero->opcode(), HloOpcode::kDot);
}

TEST_F(DynamicSliceFusionTest, FindHeroSkipsOffsetExpression) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = s32[4,2] parameter(0)
      %p1 = s32[] parameter(1)
      %zero = s32[] constant(0)
      %one = s32[] constant(1)
      %offset = s32[] add(%p1, %one)
      %ds = s32[1,2] dynamic-slice(%p0, %offset, %zero),
        dynamic_slice_sizes={1,2}
      ROOT %copy = s32[1,2] copy(%ds)
    }

    ENTRY main {
      %input = s32[4,2] parameter(0)
      %ivar = s32[] parameter(1)
      ROOT %fusion = s32[1,2] fusion(%input, %ivar), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");

  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);
  EXPECT_EQ(hero->opcode(), HloOpcode::kCopy);
}

TEST_F(DynamicSliceFusionTest, FindHeroReturnsNullWhenNoHero) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[8,4] parameter(0)
      %slice = f32[4,4] slice(%p0), slice={[0:4], [0:4]}
      ROOT %bitcast = f32[16] bitcast(%slice)
    }

    ENTRY main {
      %a = f32[8,4] parameter(0)
      ROOT %fusion = f32[16] fusion(%a), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");

  EXPECT_EQ(DynamicSliceFusion::FindHero(body), nullptr);
}

//===----------------------------------------------------------------------===//
// DynamicSliceFusion::ResolveParameters tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionTest, ResolveParamWithDynamicSliceConstantOffsets) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %zero = s32[] constant(0)
      %ds = f32[1,4] dynamic-slice(%p0, %zero, %zero), dynamic_slice_sizes={1,4},
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":16}}
      ROOT %custom = f32[1,4] custom-call(%ds), custom_call_target="hero"
    }

    ENTRY main {
      %input = f32[4,4] parameter(0)
      ROOT %fusion = f32[1,4] fusion(%input), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto params,
                       DynamicSliceFusion::ResolveParameters(hero));
  ASSERT_EQ(params.size(), 1);
  EXPECT_EQ(
      params[0],
      (Parameter{0, ShapeUtil::MakeShape(F32, {4, 4}),
                 ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
                 Offsets{{0, Offset::Constant(0)}, {1, Offset::Constant(0)}}}));
}

TEST_F(DynamicSliceFusionTest, ResolveParamWithDynamicSliceRuntimeOffsets) {
  // DS with one runtime offset (ivar from p1) and one constant offset.
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %zero = s32[] constant(0)
      %ds = f32[1,4] dynamic-slice(%p0, %p1, %zero), dynamic_slice_sizes={1,4},
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":16}}
      ROOT %custom = f32[1,4] custom-call(%ds), custom_call_target="hero"
    }

    ENTRY main {
      %input = f32[4,4] parameter(0)
      %ivar = s32[] parameter(1)
      ROOT %fusion = f32[1,4] fusion(%input, %ivar), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto params,
                       DynamicSliceFusion::ResolveParameters(hero));
  ASSERT_EQ(params.size(), 1);
  EXPECT_EQ(params[0],
            (Parameter{
                0, ShapeUtil::MakeShape(F32, {4, 4}),
                ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
                Offsets{{0, Offset::Parameter(1)}, {1, Offset::Constant(0)}}}));
}

TEST_F(DynamicSliceFusionTest, ResolveParamWithComputedDynamicSliceOffset) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %one = s32[] constant(1)
      %zero = s32[] constant(0)
      %offset = s32[] add(%p1, %one)
      %ds = f32[1,4] dynamic-slice(%p0, %offset, %zero), dynamic_slice_sizes={1,4},
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":16}}
      ROOT %custom = f32[1,4] custom-call(%ds), custom_call_target="hero"
    }

    ENTRY main {
      %input = f32[4,4] parameter(0)
      %ivar = s32[] parameter(1)
      ROOT %fusion = f32[1,4] fusion(%input, %ivar), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = body->GetInstructionWithName("custom");
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto params,
                       DynamicSliceFusion::ResolveParameters(hero));
  ASSERT_EQ(params.size(), 1);
  EXPECT_EQ(
      params[0],
      (Parameter{
          0, ShapeUtil::MakeShape(F32, {4, 4}),
          ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
          Offsets{{0, Offset::Add(Offset::Parameter(1), Offset::Constant(1))},
                  {1, Offset::Constant(0)}}}));
}

TEST_F(DynamicSliceFusionTest, ResolveParamWithSelectOffsetExpression) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %one = s32[] constant(1)
      %zero = s32[] constant(0)
      %is_negative = pred[] compare(%p1, %zero), direction=LT
      %incremented = s32[] add(%p1, %one)
      %offset = s32[] select(%is_negative, %zero, %incremented)
      %ds = f32[1,4] dynamic-slice(%p0, %offset, %zero), dynamic_slice_sizes={1,4},
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":16}}
      ROOT %custom = f32[1,4] custom-call(%ds), custom_call_target="hero"
    }

    ENTRY main {
      %input = f32[4,4] parameter(0)
      %ivar = s32[] parameter(1)
      ROOT %fusion = f32[1,4] fusion(%input, %ivar), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = body->GetInstructionWithName("custom");
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto params,
                       DynamicSliceFusion::ResolveParameters(hero));
  ASSERT_EQ(params.size(), 1);
  EXPECT_EQ(
      params[0],
      (Parameter{
          0, ShapeUtil::MakeShape(F32, {4, 4}),
          ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
          Offsets{{0, Offset::Select(Offset::Compare(ComparisonDirection::kLt,
                                                     Offset::Parameter(1),
                                                     Offset::Constant(0)),
                                     Offset::Constant(0),
                                     Offset::Add(Offset::Parameter(1),
                                                 Offset::Constant(1)))},
                  {1, Offset::Constant(0)}}}));
}

TEST_F(DynamicSliceFusionTest, ResolveParamWithStaticSlice) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[2,8,8]{2,1,0} parameter(0)
      %slice = f32[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast = f32[8,8]{1,0} bitcast(%slice)
      ROOT %custom = f32[8,8]{1,0} custom-call(%bitcast), custom_call_target="hero"
    }

    ENTRY main {
      %input = f32[2,8,8]{2,1,0} parameter(0)
      ROOT %fusion = f32[8,8]{1,0} fusion(%input), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto params,
                       DynamicSliceFusion::ResolveParameters(hero));
  ASSERT_EQ(params.size(), 1);
  // f32[2,8,8] has byte strides [256, 32, 4]. Slice starts at [1,0,0].
  // byte_offset = 1 * 256 = 256. Static slice has no DS offsets.
  // slice_shape is the static slice output shape (before bitcast to hero).
  EXPECT_EQ(
      params[0],
      (Parameter{0,
                 ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 8, 8}, {2, 1, 0}),
                 ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 8, 8}, {2, 1, 0}),
                 MakeStaticConfig(256)}));
}

TEST_F(DynamicSliceFusionTest, ResolveParamDirectParameter) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %p1 = f32[4,4] parameter(1)
      ROOT %add = f32[4,4] add(%p0, %p1)
    }

    ENTRY main {
      %a = f32[4,4] parameter(0)
      %b = f32[4,4] parameter(1)
      ROOT %fusion = f32[4,4] fusion(%a, %b), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto params,
                       DynamicSliceFusion::ResolveParameters(hero));
  ASSERT_EQ(params.size(), 2);
  Shape f32_4x4 = ShapeUtil::MakeShape(F32, {4, 4});
  EXPECT_EQ(params[0], (Parameter{0, f32_4x4, f32_4x4}));
  EXPECT_EQ(params[1], (Parameter{1, f32_4x4, f32_4x4}));
}

//===----------------------------------------------------------------------===//
// DynamicSliceFusion::ResolveResults tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionTest, ResolveResultWithDUSConstantOffsets) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %fill = f32[4] custom-call(), custom_call_target="fill"
      %bitcast = f32[1,4] bitcast(%fill)
      %zero = s32[] constant(0)
      ROOT %dus = f32[4,4] dynamic-update-slice(%p0, %bitcast, %zero, %zero),
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":16}}
    }

    ENTRY main {
      %input = f32[4,4] parameter(0)
      ROOT %fusion = f32[4,4] fusion(%input), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto results, DynamicSliceFusion::ResolveResults(hero));
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(
      results[0],
      (Result{0, 0, ShapeUtil::MakeShape(F32, {4, 4}),
              ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
              Offsets{{0, Offset::Constant(0)}, {1, Offset::Constant(0)}}}));
}

TEST_F(DynamicSliceFusionTest, ResolveResultWithDUSRuntimeOffsets) {
  // DUS with one runtime offset (ivar from p1) and one constant.
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %fill = f32[4] custom-call(), custom_call_target="fill"
      %bitcast = f32[1,4] bitcast(%fill)
      %zero = s32[] constant(0)
      ROOT %dus = f32[4,4] dynamic-update-slice(%p0, %bitcast, %p1, %zero),
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":16}}
    }

    ENTRY main {
      %input = f32[4,4] parameter(0)
      %ivar = s32[] parameter(1)
      ROOT %fusion = f32[4,4] fusion(%input, %ivar), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto results, DynamicSliceFusion::ResolveResults(hero));
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(
      results[0],
      (Result{0, 0, ShapeUtil::MakeShape(F32, {4, 4}),
              ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
              Offsets{{0, Offset::Parameter(1)}, {1, Offset::Constant(0)}}}));
}

TEST_F(DynamicSliceFusionTest, ResolveResultWithComputedDusOffset) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %fill = f32[4] custom-call(), custom_call_target="fill"
      %bitcast = f32[1,4] bitcast(%fill)
      %one = s32[] constant(1)
      %zero = s32[] constant(0)
      %offset = s32[] add(%p1, %one)
      ROOT %dus = f32[4,4] dynamic-update-slice(%p0, %bitcast, %offset, %zero),
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":16}}
    }

    ENTRY main {
      %input = f32[4,4] parameter(0)
      %ivar = s32[] parameter(1)
      ROOT %fusion = f32[4,4] fusion(%input, %ivar), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = body->GetInstructionWithName("fill");
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto results, DynamicSliceFusion::ResolveResults(hero));
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(
      results[0],
      (Result{
          0, 0, ShapeUtil::MakeShape(F32, {4, 4}),
          ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
          Offsets{{0, Offset::Add(Offset::Parameter(1), Offset::Constant(1))},
                  {1, Offset::Constant(0)}}}));
}

TEST_F(DynamicSliceFusionTest, ResolveResultWithBitcastThenDUS) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %fill = f32[1,4] custom-call(), custom_call_target="fill"
      %bitcast = f32[4] bitcast(%fill)
      %bitcast2 = f32[1,4] bitcast(%bitcast)
      %zero = s32[] constant(0)
      ROOT %dus = f32[4,4] dynamic-update-slice(%p0, %bitcast2, %zero, %zero),
        backend_config={"dynamic_slice_config":{"loop_index":1,"byte_offset":32,"byte_stride":64}}
    }

    ENTRY main {
      %input = f32[4,4] parameter(0)
      ROOT %fusion = f32[4,4] fusion(%input), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto results, DynamicSliceFusion::ResolveResults(hero));
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(
      results[0],
      (Result{0, 0, ShapeUtil::MakeShape(F32, {4, 4}),
              ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(1, 32, 64),
              Offsets{{0, Offset::Constant(0)}, {1, Offset::Constant(0)}}}));
}

TEST_F(DynamicSliceFusionTest, ResolveResultNoDUS) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %p1 = f32[4,4] parameter(1)
      ROOT %add = f32[4,4] add(%p0, %p1)
    }

    ENTRY main {
      %a = f32[4,4] parameter(0)
      %b = f32[4,4] parameter(1)
      ROOT %fusion = f32[4,4] fusion(%a, %b), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto results, DynamicSliceFusion::ResolveResults(hero));
  ASSERT_EQ(results.size(), 1);
  Shape f32_4x4 = ShapeUtil::MakeShape(F32, {4, 4});
  EXPECT_EQ(results[0], (Result{std::nullopt, 0, f32_4x4, f32_4x4}));
}

TEST_F(DynamicSliceFusionTest, ResolveResultDUSWithoutConfig) {
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %fill = f32[4] custom-call(), custom_call_target="fill"
      %bitcast = f32[1,4] bitcast(%fill)
      %zero = s32[] constant(0)
      ROOT %dus = f32[4,4] dynamic-update-slice(%p0, %bitcast, %zero, %zero)
    }

    ENTRY main {
      %input = f32[4,4] parameter(0)
      ROOT %fusion = f32[4,4] fusion(%input), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto results, DynamicSliceFusion::ResolveResults(hero));
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(
      results[0],
      (Result{0, 0, ShapeUtil::MakeShape(F32, {4, 4}),
              ShapeUtil::MakeShape(F32, {1, 4}), std::nullopt,
              Offsets{{0, Offset::Constant(0)}, {1, Offset::Constant(0)}}}));
}

//===----------------------------------------------------------------------===//
// Combined: both parameters and results with runtime offsets
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionTest, ParamsAndResultsWithRuntimeOffsets) {
  // DS and DUS share the same runtime offset (ivar from p2).
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %p1 = f32[4,4] parameter(1)
      %p2 = s32[] parameter(2)
      %zero = s32[] constant(0)
      %ds = f32[1,4] dynamic-slice(%p0, %p2, %zero), dynamic_slice_sizes={1,4},
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %custom = f32[1,4] custom-call(%ds), custom_call_target="hero"
      ROOT %dus = f32[4,4] dynamic-update-slice(%p1, %custom, %p2, %zero),
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":16}}
    }

    ENTRY main {
      %a = f32[4,4] parameter(0)
      %b = f32[4,4] parameter(1)
      %ivar = s32[] parameter(2)
      ROOT %fusion = f32[4,4] fusion(%a, %b, %ivar), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);
  EXPECT_EQ(hero->opcode(), HloOpcode::kCustomCall);

  ASSERT_OK_AND_ASSIGN(auto params,
                       DynamicSliceFusion::ResolveParameters(hero));
  ASSERT_EQ(params.size(), 1);
  EXPECT_EQ(params[0],
            (Parameter{
                0, ShapeUtil::MakeShape(F32, {4, 4}),
                ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
                Offsets{{0, Offset::Parameter(2)}, {1, Offset::Constant(0)}}}));

  ASSERT_OK_AND_ASSIGN(auto results, DynamicSliceFusion::ResolveResults(hero));
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(
      results[0],
      (Result{1, 0, ShapeUtil::MakeShape(F32, {4, 4}),
              ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
              Offsets{{0, Offset::Parameter(2)}, {1, Offset::Constant(0)}}}));
}

//===----------------------------------------------------------------------===//
// Nested tuple results
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionTest, ResolveResultsFlatTupleWithDUS) {
  // Hero returns (f32[1,4], f32[1,8]) — both elements feed into DUS.
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %p1 = f32[4,8] parameter(1)
      %p2 = s32[] parameter(2)
      %zero = s32[] constant(0)
      %custom = (f32[1,4], f32[1,8]) custom-call(), custom_call_target="hero"
      %gte0 = f32[1,4] get-tuple-element(%custom), index=0
      %gte1 = f32[1,8] get-tuple-element(%custom), index=1
      %dus0 = f32[4,4] dynamic-update-slice(%p0, %gte0, %p2, %zero),
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %dus1 = f32[4,8] dynamic-update-slice(%p1, %gte1, %zero, %zero),
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":32}}
      ROOT %tuple = (f32[4,4], f32[4,8]) tuple(%dus0, %dus1)
    }

    ENTRY main {
      %a = f32[4,4] parameter(0)
      %b = f32[4,8] parameter(1)
      %ivar = s32[] parameter(2)
      ROOT %fusion = (f32[4,4], f32[4,8]) fusion(%a, %b, %ivar), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto results, DynamicSliceFusion::ResolveResults(hero));
  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(
      results[0],
      (Result{0, 0, ShapeUtil::MakeShape(F32, {4, 4}),
              ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
              Offsets{{0, Offset::Parameter(2)}, {1, Offset::Constant(0)}}}));
  EXPECT_EQ(
      results[1],
      (Result{1, 1, ShapeUtil::MakeShape(F32, {4, 8}),
              ShapeUtil::MakeShape(F32, {1, 8}), MakeConfig(0, 0, 32),
              Offsets{{0, Offset::Constant(0)}, {1, Offset::Constant(0)}}}));
}

TEST_F(DynamicSliceFusionTest, ResolveResultsNestedTupleWithDUS) {
  // Hero returns ((f32[4], f32[4]), f32[8]) — 3 leaves, all feed into DUS.
  // Leaf 0 = {0,0}, Leaf 1 = {0,1}, Leaf 2 = {1}.
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %p1 = f32[4,4] parameter(1)
      %p2 = f32[4,8] parameter(2)
      %zero = s32[] constant(0)
      %custom = ((f32[4], f32[4]), f32[8]) custom-call(), custom_call_target="hero"
      %gte_outer0 = (f32[4], f32[4]) get-tuple-element(%custom), index=0
      %gte00 = f32[4] get-tuple-element(%gte_outer0), index=0
      %gte01 = f32[4] get-tuple-element(%gte_outer0), index=1
      %gte1 = f32[8] get-tuple-element(%custom), index=1
      %bitcast00 = f32[1,4] bitcast(%gte00)
      %bitcast01 = f32[1,4] bitcast(%gte01)
      %bitcast1 = f32[1,8] bitcast(%gte1)
      %dus0 = f32[4,4] dynamic-update-slice(%p0, %bitcast00, %zero, %zero),
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %dus1 = f32[4,4] dynamic-update-slice(%p1, %bitcast01, %zero, %zero),
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %dus2 = f32[4,8] dynamic-update-slice(%p2, %bitcast1, %zero, %zero),
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":32}}
      ROOT %tuple = (f32[4,4], f32[4,4], f32[4,8]) tuple(%dus0, %dus1, %dus2)
    }

    ENTRY main {
      %a = f32[4,4] parameter(0)
      %b = f32[4,4] parameter(1)
      %c = f32[4,8] parameter(2)
      ROOT %fusion = (f32[4,4], f32[4,4], f32[4,8]) fusion(%a, %b, %c), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto results, DynamicSliceFusion::ResolveResults(hero));
  ASSERT_EQ(results.size(), 3);
  Offsets const_2d{{0, Offset::Constant(0)}, {1, Offset::Constant(0)}};
  EXPECT_EQ(results[0], (Result{0, 0, ShapeUtil::MakeShape(F32, {4, 4}),
                                ShapeUtil::MakeShape(F32, {1, 4}),
                                MakeConfig(0, 0, 16), const_2d}));
  EXPECT_EQ(results[1], (Result{1, 1, ShapeUtil::MakeShape(F32, {4, 4}),
                                ShapeUtil::MakeShape(F32, {1, 4}),
                                MakeConfig(0, 0, 16), const_2d}));
  EXPECT_EQ(results[2], (Result{2, 2, ShapeUtil::MakeShape(F32, {4, 8}),
                                ShapeUtil::MakeShape(F32, {1, 8}),
                                MakeConfig(0, 0, 32), const_2d}));
}

TEST_F(DynamicSliceFusionTest, ResolveResultsNestedTuplePartialDUS) {
  // Hero returns ((f32[4], f32[4]), f32[8]) — only leaf 0 and 2 have DUS.
  const char* hlo = R"(
    HloModule test

    %fused {
      %p0 = f32[4,4] parameter(0)
      %p1 = f32[4,8] parameter(1)
      %zero = s32[] constant(0)
      %custom = ((f32[4], f32[4]), f32[8]) custom-call(), custom_call_target="hero"
      %gte_outer0 = (f32[4], f32[4]) get-tuple-element(%custom), index=0
      %gte00 = f32[4] get-tuple-element(%gte_outer0), index=0
      %gte1 = f32[8] get-tuple-element(%custom), index=1
      %bitcast00 = f32[1,4] bitcast(%gte00)
      %bitcast1 = f32[1,8] bitcast(%gte1)
      %dus0 = f32[4,4] dynamic-update-slice(%p0, %bitcast00, %zero, %zero),
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %dus2 = f32[4,8] dynamic-update-slice(%p1, %bitcast1, %zero, %zero),
        backend_config={"dynamic_slice_config":{"loop_index":0,"byte_offset":0,"byte_stride":32}}
      ROOT %tuple = (f32[4,4], f32[4,8]) tuple(%dus0, %dus2)
    }

    ENTRY main {
      %a = f32[4,4] parameter(0)
      %b = f32[4,8] parameter(1)
      ROOT %fusion = (f32[4,4], f32[4,8]) fusion(%a, %b), kind=kCustom, calls=%fused
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const HloComputation* body = module->GetComputationWithName("fused");
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);

  ASSERT_OK_AND_ASSIGN(auto results, DynamicSliceFusion::ResolveResults(hero));
  ASSERT_EQ(results.size(), 3);
  Offsets const_2d{{0, Offset::Constant(0)}, {1, Offset::Constant(0)}};
  EXPECT_EQ(results[0], (Result{0, 0, ShapeUtil::MakeShape(F32, {4, 4}),
                                ShapeUtil::MakeShape(F32, {1, 4}),
                                MakeConfig(0, 0, 16), const_2d}));
  EXPECT_EQ(results[1], (Result{std::nullopt, 1, ShapeUtil::MakeShape(F32, {4}),
                                ShapeUtil::MakeShape(F32, {4})}));
  EXPECT_EQ(results[2], (Result{1, 2, ShapeUtil::MakeShape(F32, {4, 8}),
                                ShapeUtil::MakeShape(F32, {1, 8}),
                                MakeConfig(0, 0, 32), const_2d}));
}

//===----------------------------------------------------------------------===//
// Offset evaluation tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionTest, EvaluateOffsetExprConstantAndParameter) {
  ASSERT_OK_AND_ASSIGN(int64_t constant,
                       DynamicSliceFusion::Evaluate(Offset::Constant(42), {}));
  EXPECT_EQ(constant, 42);

  ASSERT_OK_AND_ASSIGN(int64_t parameter, DynamicSliceFusion::Evaluate(
                                              Offset::Parameter(3), {{3, -7}}));
  EXPECT_EQ(parameter, -7);
}

TEST_F(DynamicSliceFusionTest, EvaluateOffsetExprArithmetic) {
  auto expr = Offset::Subtract(
      Offset::Multiply(Offset::Add(Offset::Parameter(0), Offset::Constant(2)),
                       Offset::Parameter(1)),
      Offset::Constant(5));

  ASSERT_OK_AND_ASSIGN(int64_t result,
                       DynamicSliceFusion::Evaluate(expr, {{0, 4}, {1, 3}}));
  EXPECT_EQ(result, 13);
}

TEST_F(DynamicSliceFusionTest, EvaluateOffsetExprCompareAndSelect) {
  auto expr = Offset::Select(
      Offset::Compare(ComparisonDirection::kLt, Offset::Parameter(0),
                      Offset::Constant(3)),
      Offset::Add(Offset::Parameter(0), Offset::Constant(1)),
      Offset::Multiply(Offset::Parameter(1), Offset::Constant(2)));

  ASSERT_OK_AND_ASSIGN(int64_t on_true,
                       DynamicSliceFusion::Evaluate(expr, {{0, 2}, {1, 9}}));
  EXPECT_EQ(on_true, 3);

  ASSERT_OK_AND_ASSIGN(int64_t on_false,
                       DynamicSliceFusion::Evaluate(expr, {{0, 3}, {1, 9}}));
  EXPECT_EQ(on_false, 18);
}

TEST_F(DynamicSliceFusionTest, EvaluateOffsetExprMissingParameterFails) {
  auto status =
      DynamicSliceFusion::Evaluate(Offset::Parameter(7), {{3, 12}}).status();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("Missing value for offset parameter 7"));
}

TEST_F(DynamicSliceFusionTest, OffsetIsExprChecksScalarIntegerOperations) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      p0 = s32[] parameter(0)
      p1 = s64[] parameter(1)
      pred_param = pred[] parameter(2)
      float_param = f32[] parameter(3)
      vector_param = s32[1] parameter(4)
      c0 = s32[] constant(0)
      c1 = s64[] constant(1)
      pred_const = pred[] constant(true)
      add = s32[] add(p0, c0)
      multiply = s64[] multiply(p1, c1)
      compare = pred[] compare(p0, c0), direction=LT
      select = s32[] select(compare, p0, c0)
      pred_select = pred[] select(pred_param, compare, pred_const)
      convert = s64[] convert(p0)
      bitcast = s32[] bitcast(p0)
      scalar_reshape = s32[] reshape(vector_param)
      vector_reshape = s32[1] reshape(p0)
      maximum = s32[] maximum(p0, c0)
      float_add = f32[] add(float_param, float_param)
      ROOT root = (s32[], s64[], pred[], s32[], pred[], s64[], s32[],
                   s32[], s32[1], s32[], f32[]) tuple(
          add, multiply, compare, select, pred_select, convert,
          bitcast, scalar_reshape, vector_reshape,
          maximum, float_add)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloComputation* c = module->entry_computation();

  EXPECT_TRUE(Offset::IsExpr(c->GetInstructionWithName("p0")));
  EXPECT_TRUE(Offset::IsExpr(c->GetInstructionWithName("p1")));
  EXPECT_TRUE(Offset::IsExpr(c->GetInstructionWithName("pred_param")));
  EXPECT_TRUE(Offset::IsExpr(c->GetInstructionWithName("c0")));
  EXPECT_TRUE(Offset::IsExpr(c->GetInstructionWithName("c1")));
  EXPECT_TRUE(Offset::IsExpr(c->GetInstructionWithName("pred_const")));
  EXPECT_TRUE(Offset::IsExpr(c->GetInstructionWithName("add")));
  EXPECT_TRUE(Offset::IsExpr(c->GetInstructionWithName("multiply")));
  EXPECT_TRUE(Offset::IsExpr(c->GetInstructionWithName("compare")));
  EXPECT_TRUE(Offset::IsExpr(c->GetInstructionWithName("select")));
  EXPECT_TRUE(Offset::IsExpr(c->GetInstructionWithName("pred_select")));

  EXPECT_FALSE(Offset::IsExpr(c->GetInstructionWithName("float_param")));
  EXPECT_FALSE(Offset::IsExpr(c->GetInstructionWithName("vector_param")));
  EXPECT_FALSE(Offset::IsExpr(c->GetInstructionWithName("convert")));
  EXPECT_FALSE(Offset::IsExpr(c->GetInstructionWithName("bitcast")));
  EXPECT_FALSE(Offset::IsExpr(c->GetInstructionWithName("scalar_reshape")));
  EXPECT_FALSE(Offset::IsExpr(c->GetInstructionWithName("vector_reshape")));
  EXPECT_FALSE(Offset::IsExpr(c->GetInstructionWithName("maximum")));
  EXPECT_FALSE(Offset::IsExpr(c->GetInstructionWithName("float_add")));
}

TEST_F(DynamicSliceFusionTest, CollectOffsetParameters) {
  auto expr =
      Offset::Select(Offset::Parameter(2),
                     Offset::Add(Offset::Parameter(4), Offset::Parameter(2)),
                     Offset::Constant(0));

  EXPECT_THAT(DynamicSliceFusion::CollectOffsetParameters(expr),
              ::testing::ElementsAre(2, 4));
}

//===----------------------------------------------------------------------===//
// Stringify tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionTest, StringifyParameterWithConfig) {
  Parameter p{0, ShapeUtil::MakeShape(F32, {4, 4}),
              ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
              Offsets{{0, Offset::Parameter(1)}, {1, Offset::Constant(0)}}};
  std::string s = absl::StrCat(p);
  EXPECT_EQ(s,
            "Parameter{param=0 f32[4,4]->f32[1,4], "
            "config{loop=0, offset=0, stride=16}, "
            "offsets=[o(d0,p1), o(d1,0)]}");
}

TEST_F(DynamicSliceFusionTest, StringifyOffsetExpr) {
  auto expr =
      Offset::Select(Offset::Compare(ComparisonDirection::kLt,
                                     Offset::Parameter(0), Offset::Constant(3)),
                     Offset::Add(Offset::Parameter(0), Offset::Constant(1)),
                     Offset::Constant(0));
  EXPECT_EQ(absl::StrCat(expr), "select(cmp(LT, p0, 3), add(p0, 1), 0)");
}

TEST_F(DynamicSliceFusionTest, StringifyParameterWithoutConfig) {
  Parameter p{1, ShapeUtil::MakeShape(F32, {8, 8}),
              ShapeUtil::MakeShape(F32, {8, 8})};
  std::string s = absl::StrCat(p);
  EXPECT_EQ(s,
            "Parameter{param=1 f32[8,8]->f32[8,8], "
            "config{}, offsets=none}");
}

TEST_F(DynamicSliceFusionTest, StringifyResultWithConfig) {
  Result r{0,
           0,
           ShapeUtil::MakeShape(F32, {4, 4}),
           ShapeUtil::MakeShape(F32, {1, 4}),
           MakeConfig(0, 0, 16),
           Offsets{{0, Offset::Constant(0)}, {1, Offset::Constant(0)}}};
  std::string s = absl::StrCat(r);
  EXPECT_EQ(s,
            "Result{param=0, result=0 f32[1,4]->f32[4,4], "
            "config{loop=0, offset=0, stride=16}, "
            "offsets=[o(d0,0), o(d1,0)]}");
}

TEST_F(DynamicSliceFusionTest, StringifyResultWithoutConfig) {
  Result r{std::nullopt, 0, ShapeUtil::MakeShape(F32, {4, 4}),
           ShapeUtil::MakeShape(F32, {4, 4})};
  std::string s = absl::StrCat(r);
  EXPECT_EQ(s,
            "Result{param=none, result=0 f32[4,4]->f32[4,4], "
            "config{}, offsets=none}");
}

}  // namespace
}  // namespace xla::gpu
