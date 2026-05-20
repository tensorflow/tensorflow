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
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
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

using CstOff = DynamicSliceFusion::ConstantOffset;
using RtOff = DynamicSliceFusion::RuntimeOffset;

using Offsets = std::vector<DynamicSliceFusion::Offset>;

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
  EXPECT_EQ(params[0],
            (Parameter{0, ShapeUtil::MakeShape(F32, {4, 4}),
                       ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
                       Offsets{CstOff{0, 0}, CstOff{0, 1}}}));
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
            (Parameter{0, ShapeUtil::MakeShape(F32, {4, 4}),
                       ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
                       Offsets{RtOff{1, 0}, CstOff{0, 1}}}));
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
  EXPECT_EQ(results[0],
            (Result{0, 0, ShapeUtil::MakeShape(F32, {4, 4}),
                    ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
                    Offsets{CstOff{0, 0}, CstOff{0, 1}}}));
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
  EXPECT_EQ(results[0],
            (Result{0, 0, ShapeUtil::MakeShape(F32, {4, 4}),
                    ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
                    Offsets{RtOff{1, 0}, CstOff{0, 1}}}));
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
  EXPECT_EQ(results[0],
            (Result{0, 0, ShapeUtil::MakeShape(F32, {4, 4}),
                    ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(1, 32, 64),
                    Offsets{CstOff{0, 0}, CstOff{0, 1}}}));
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
  EXPECT_EQ(results[0], (Result{0, 0, ShapeUtil::MakeShape(F32, {4, 4}),
                                ShapeUtil::MakeShape(F32, {1, 4}), std::nullopt,
                                Offsets{CstOff{0, 0}, CstOff{0, 1}}}));
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
            (Parameter{0, ShapeUtil::MakeShape(F32, {4, 4}),
                       ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
                       Offsets{RtOff{2, 0}, CstOff{0, 1}}}));

  ASSERT_OK_AND_ASSIGN(auto results, DynamicSliceFusion::ResolveResults(hero));
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0],
            (Result{1, 0, ShapeUtil::MakeShape(F32, {4, 4}),
                    ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
                    Offsets{RtOff{2, 0}, CstOff{0, 1}}}));
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
  EXPECT_EQ(results[0],
            (Result{0, 0, ShapeUtil::MakeShape(F32, {4, 4}),
                    ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
                    Offsets{RtOff{2, 0}, CstOff{0, 1}}}));
  EXPECT_EQ(results[1],
            (Result{1, 1, ShapeUtil::MakeShape(F32, {4, 8}),
                    ShapeUtil::MakeShape(F32, {1, 8}), MakeConfig(0, 0, 32),
                    Offsets{CstOff{0, 0}, CstOff{0, 1}}}));
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
  Offsets const_2d{CstOff{0, 0}, CstOff{0, 1}};
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
  Offsets const_2d{CstOff{0, 0}, CstOff{0, 1}};
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
// Stringify tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionTest, StringifyParameterWithConfig) {
  Parameter p{0, ShapeUtil::MakeShape(F32, {4, 4}),
              ShapeUtil::MakeShape(F32, {1, 4}), MakeConfig(0, 0, 16),
              Offsets{RtOff{1, 0}, CstOff{0, 1}}};
  std::string s = absl::StrCat(p);
  EXPECT_EQ(s,
            "Parameter{param=0 f32[4,4]->f32[1,4], "
            "config{loop=0, offset=0, stride=16}, "
            "offsets=[r(d0,p1), c(d1,0)]}");
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
           Offsets{CstOff{0, 0}, CstOff{0, 1}}};
  std::string s = absl::StrCat(r);
  EXPECT_EQ(s,
            "Result{param=0, result=0 f32[1,4]->f32[4,4], "
            "config{loop=0, offset=0, stride=16}, "
            "offsets=[c(d0,0), c(d1,0)]}");
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
