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

#include "xla/backends/gpu/transforms/dynamic_slice_analysis.h"

#include <algorithm>
#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu {
namespace {

using DynamicSliceAnalysisTest = HloHardwareIndependentTestBase;

// Helper: runs AnalyzeDynamicSlice on an instruction by name in the "body"
// computation.
absl::StatusOr<std::optional<DynamicSliceDescriptor>> AnalyzeByName(
    HloModule* module, absl::string_view name) {
  auto* instr =
      module->GetComputationWithName("body")->GetInstructionWithName(name);
  return AnalyzeDynamicSlice(instr);
}

// DS with init=0, step=1 on dim0 of s32[4,8,8].
// dim0 byte_stride = 8*8*4 = 256. offset = 0*256 = 0, stride = 1*256 = 256.
TEST_F(DynamicSliceAnalysisTest, DsForwardStep1) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[4,8,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      slice = s32[1,8,8] dynamic-slice(input, ivar, c0, c0),
          dynamic_slice_sizes={1,8,8}
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], s32[4,8,8]) tuple(next_ivar, input)
    }

    condition {
      p0 = (s32[], s32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = s32[4,8,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8,8]) tuple(c0, input)
      ROOT while = (s32[], s32[4,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(auto desc, AnalyzeByName(module.get(), "slice"));
  ASSERT_TRUE(desc.has_value());
  EXPECT_THAT(desc->loop_index, ::testing::Optional(0));
  EXPECT_EQ(desc->byte_offset, 0);
  EXPECT_EQ(desc->byte_stride, 256);
}

// DUS with init=0, step=1, offset at dim0 (ivar) and dim1=constant(1).
// s32[4,8,8]: dim0 stride=256, dim1 stride=32. offset=0+1*32=32, stride=256.
TEST_F(DynamicSliceAnalysisTest, DusForwardStep1) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[4,8,8] get-tuple-element(p0), index=1
      val = s32[1,1,8] constant({{{1,2,3,4,5,6,7,8}}})
      c1 = s32[] constant(1)
      c0 = s32[] constant(0)
      updated = s32[4,8,8] dynamic-update-slice(input, val, ivar, c1, c0)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], s32[4,8,8]) tuple(next_ivar, updated)
    }

    condition {
      p0 = (s32[], s32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = s32[4,8,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8,8]) tuple(c0, input)
      ROOT while = (s32[], s32[4,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(auto desc, AnalyzeByName(module.get(), "updated"));
  ASSERT_TRUE(desc.has_value());
  EXPECT_THAT(desc->loop_index, ::testing::Optional(0));
  EXPECT_EQ(desc->byte_offset, 32);
  EXPECT_EQ(desc->byte_stride, 256);
}

// DS with init=2, step=3 on s32[8,8,8]. dim0 stride=256.
// offset = 2*256 = 512, stride = 3*256 = 768.
TEST_F(DynamicSliceAnalysisTest, NonTrivialInitAndStep) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[8,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[8,8,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      slice = s32[1,8,8] dynamic-slice(input, ivar, c0, c0),
          dynamic_slice_sizes={1,8,8}
      c3 = s32[] constant(3)
      next_ivar = s32[] add(ivar, c3)
      ROOT result = (s32[], s32[8,8,8]) tuple(next_ivar, input)
    }

    condition {
      p0 = (s32[], s32[8,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c8 = s32[] constant(8)
      ROOT cmp = pred[] compare(ivar, c8), direction=LT
    }

    ENTRY main {
      input = s32[8,8,8] parameter(0)
      c2 = s32[] constant(2)
      tuple = (s32[], s32[8,8,8]) tuple(c2, input)
      ROOT while = (s32[], s32[8,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"2"},
                          "known_init_step":{"init":"2","step":"3"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(auto desc, AnalyzeByName(module.get(), "slice"));
  ASSERT_TRUE(desc.has_value());
  EXPECT_THAT(desc->loop_index, ::testing::Optional(0));
  EXPECT_EQ(desc->byte_offset, 2 * 256);
  EXPECT_EQ(desc->byte_stride, 3 * 256);
}

// Backward: init=3, step=-1 on s32[4,8,8]. dim0 stride=256.
// offset = 3*256 = 768, stride = -1*256 = -256.
TEST_F(DynamicSliceAnalysisTest, BackwardStep) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[4,8,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      slice = s32[1,8,8] dynamic-slice(input, ivar, c0, c0),
          dynamic_slice_sizes={1,8,8}
      cn1 = s32[] constant(-1)
      next_ivar = s32[] add(ivar, cn1)
      ROOT result = (s32[], s32[4,8,8]) tuple(next_ivar, input)
    }

    condition {
      p0 = (s32[], s32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c0 = s32[] constant(0)
      ROOT cmp = pred[] compare(ivar, c0), direction=GE
    }

    ENTRY main {
      input = s32[4,8,8] parameter(0)
      c3 = s32[] constant(3)
      tuple = (s32[], s32[4,8,8]) tuple(c3, input)
      ROOT while = (s32[], s32[4,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"3","step":"-1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(auto desc, AnalyzeByName(module.get(), "slice"));
  ASSERT_TRUE(desc.has_value());
  EXPECT_THAT(desc->loop_index, ::testing::Optional(0));
  EXPECT_EQ(desc->byte_offset, 3 * 256);
  EXPECT_EQ(desc->byte_stride, -256);
}

// Forward step=2: init=0, step=2 on s32[8,8,8]. stride = 2*256 = 512.
TEST_F(DynamicSliceAnalysisTest, ForwardStepTwo) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[8,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[8,8,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      slice = s32[1,8,8] dynamic-slice(input, ivar, c0, c0),
          dynamic_slice_sizes={1,8,8}
      c2 = s32[] constant(2)
      next_ivar = s32[] add(ivar, c2)
      ROOT result = (s32[], s32[8,8,8]) tuple(next_ivar, input)
    }

    condition {
      p0 = (s32[], s32[8,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c8 = s32[] constant(8)
      ROOT cmp = pred[] compare(ivar, c8), direction=LT
    }

    ENTRY main {
      input = s32[8,8,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[8,8,8]) tuple(c0, input)
      ROOT while = (s32[], s32[8,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"2"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(auto desc, AnalyzeByName(module.get(), "slice"));
  ASSERT_TRUE(desc.has_value());
  EXPECT_THAT(desc->loop_index, ::testing::Optional(0));
  EXPECT_EQ(desc->byte_offset, 0);
  EXPECT_EQ(desc->byte_stride, 2 * 256);
}

// Backward step=-2: init=6, step=-2 on s32[8,8,8].
// offset = 6*256 = 1536, stride = -2*256 = -512.
TEST_F(DynamicSliceAnalysisTest, BackwardStepTwo) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[8,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[8,8,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      slice = s32[1,8,8] dynamic-slice(input, ivar, c0, c0),
          dynamic_slice_sizes={1,8,8}
      cn2 = s32[] constant(-2)
      next_ivar = s32[] add(ivar, cn2)
      ROOT result = (s32[], s32[8,8,8]) tuple(next_ivar, input)
    }

    condition {
      p0 = (s32[], s32[8,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c0 = s32[] constant(0)
      ROOT cmp = pred[] compare(ivar, c0), direction=GE
    }

    ENTRY main {
      input = s32[8,8,8] parameter(0)
      c6 = s32[] constant(6)
      tuple = (s32[], s32[8,8,8]) tuple(c6, input)
      ROOT while = (s32[], s32[8,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"6","step":"-2"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(auto desc, AnalyzeByName(module.get(), "slice"));
  ASSERT_TRUE(desc.has_value());
  EXPECT_THAT(desc->loop_index, ::testing::Optional(0));
  EXPECT_EQ(desc->byte_offset, 6 * 256);
  EXPECT_EQ(desc->byte_stride, -2 * 256);
}

// Staggered induction variable after loop pipelining. DUS offset comes from
// tuple index 3 which carries the previous iteration's ivar value.
// f32[4,4]: dim0 stride=16. Staggered init=0, step=1. offset=0, stride=16.
TEST_F(DynamicSliceAnalysisTest, StaggeredInductionVariable) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], f32[4,8], f32[4,4], s32[]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = f32[4,8] get-tuple-element(p0), index=1
      output = f32[4,4] get-tuple-element(p0), index=2
      prev_ivar = s32[] get-tuple-element(p0), index=3
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      val = f32[1,4] constant({{1,2,3,4}})
      dus = f32[4,4] dynamic-update-slice(output, val, prev_ivar, c0)
      ROOT result = (s32[], f32[4,8], f32[4,4], s32[])
          tuple(next_ivar, input, dus, ivar)
    }

    condition {
      p0 = (s32[], f32[4,8], f32[4,4], s32[]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = f32[4,8] parameter(0)
      c0 = f32[] constant(0)
      output = f32[4,4] broadcast(c0), dimensions={}
      c0_s32 = s32[] constant(0)
      c1 = s32[] constant(1)
      tuple = (s32[], f32[4,8], f32[4,4], s32[])
          tuple(c1, input, output, c0_s32)
      ROOT while = (s32[], f32[4,8], f32[4,4], s32[]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"3"},
                          "known_init_step":{"init":"1","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(auto desc, AnalyzeByName(module.get(), "dus"));
  ASSERT_TRUE(desc.has_value());
  EXPECT_THAT(desc->loop_index, ::testing::Optional(0));
  EXPECT_EQ(desc->byte_offset, 0);
  EXPECT_EQ(desc->byte_stride, 16);
}

// DS with all constant offsets inside a while loop body returns a static
// descriptor (stride=0, no loop).
TEST_F(DynamicSliceAnalysisTest, AllConstantOffsetsInLoop) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[4,8,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      slice = s32[1,8,8] dynamic-slice(input, c1, c0, c0),
          dynamic_slice_sizes={1,8,8}
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], s32[4,8,8]) tuple(next_ivar, input)
    }

    condition {
      p0 = (s32[], s32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = s32[4,8,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8,8]) tuple(c0, input)
      ROOT while = (s32[], s32[4,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(auto desc, AnalyzeByName(module.get(), "slice"));
  ASSERT_TRUE(desc.has_value());
  EXPECT_FALSE(desc->while_loop.has_value());
  EXPECT_FALSE(desc->loop_index.has_value());
  // s32[4,8,8] dim0 byte_stride = 8*8*4 = 256. offset = 1*256 = 256.
  EXPECT_EQ(desc->byte_offset, 256);
  EXPECT_EQ(desc->byte_stride, 0);
}

// DUS with all constant offsets inside a while loop body returns a static
// descriptor.
TEST_F(DynamicSliceAnalysisTest, AllConstantDusOffsetsInLoop) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      buf = s32[4,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      val = s32[1,8] constant({{1,2,3,4,5,6,7,8}})
      updated = s32[4,8] dynamic-update-slice(buf, val, c1, c0)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], s32[4,8]) tuple(next_ivar, updated)
    }

    condition {
      p0 = (s32[], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = s32[4,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8]) tuple(c0, input)
      ROOT while = (s32[], s32[4,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(auto desc, AnalyzeByName(module.get(), "updated"));
  ASSERT_TRUE(desc.has_value());
  EXPECT_FALSE(desc->while_loop.has_value());
  EXPECT_FALSE(desc->loop_index.has_value());
  // s32[4,8] dim0 byte_stride = 8*4 = 32. offset = 1*32 = 32.
  EXPECT_EQ(desc->byte_offset, 32);
  EXPECT_EQ(desc->byte_stride, 0);
}

// DS with constant offsets (not in a while loop) returns a static descriptor.
TEST_F(DynamicSliceAnalysisTest, StaticDsOutsideLoop) {
  constexpr absl::string_view kHlo = R"(
    ENTRY main {
      input = s32[4,8] parameter(0)
      c1 = s32[] constant(1)
      c0 = s32[] constant(0)
      ROOT slice = s32[1,8] dynamic-slice(input, c1, c0),
          dynamic_slice_sizes={1,8}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  auto* slice = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(auto desc, AnalyzeDynamicSlice(slice));
  ASSERT_TRUE(desc.has_value());
  EXPECT_FALSE(desc->while_loop.has_value());
  EXPECT_FALSE(desc->loop_index.has_value());
  // s32[4,8] dim0 byte_stride = 8*4 = 32. offset = 1*32 = 32.
  EXPECT_EQ(desc->byte_offset, 32);
  EXPECT_EQ(desc->byte_stride, 0);
}

// DS with data-dependent offset (not an ivar) returns nullopt.
TEST_F(DynamicSliceAnalysisTest, DataDependentDsReturnsNullopt) {
  constexpr absl::string_view kHlo = R"(
    ENTRY main {
      input = s32[4,8] parameter(0)
      offset = s32[] parameter(1)
      c0 = s32[] constant(0)
      ROOT slice = s32[1,8] dynamic-slice(input, offset, c0),
          dynamic_slice_sizes={1,8}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  auto* slice = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(auto desc, AnalyzeDynamicSlice(slice));
  EXPECT_FALSE(desc.has_value());
}

//===-----------------------------------------------------------------------===/
// FindDynamicSliceChain
//===-----------------------------------------------------------------------===/

// DS and DUS on the same parameter are collected into one set.
TEST_F(DynamicSliceAnalysisTest, FindSetDsAndDusSameBuffer) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      buf = s32[4,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      slice = s32[1,8] dynamic-slice(buf, ivar, c0),
          dynamic_slice_sizes={1,8}
      val = s32[1,8] constant({{1,2,3,4,5,6,7,8}})
      updated = s32[4,8] dynamic-update-slice(buf, val, ivar, c0)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], s32[4,8]) tuple(next_ivar, updated)
    }

    condition {
      p0 = (s32[], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = s32[4,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8]) tuple(c0, input)
      ROOT while = (s32[], s32[4,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  auto* body = module->GetComputationWithName("body");
  auto* slice = body->GetInstructionWithName("slice");
  auto* updated = body->GetInstructionWithName("updated");

  ASSERT_OK_AND_ASSIGN(auto chain, FindDynamicSliceChain(slice));
  EXPECT_EQ(chain.slices.size(), 1);
  EXPECT_EQ(chain.updates.size(), 1);
  EXPECT_NE(std::find(chain.slices.begin(), chain.slices.end(),
                      Cast<HloDynamicSliceInstruction>(slice)),
            chain.slices.end());
  EXPECT_NE(std::find(chain.updates.begin(), chain.updates.end(),
                      Cast<HloDynamicUpdateSliceInstruction>(updated)),
            chain.updates.end());
  EXPECT_EQ(chain.result, updated);
}

// DS on one buffer does not include DUS on a different buffer.
TEST_F(DynamicSliceAnalysisTest, FindSetSeparateBuffers) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[4,8], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      buf0 = s32[4,8] get-tuple-element(p0), index=1
      buf1 = s32[4,8] get-tuple-element(p0), index=2
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      slice = s32[1,8] dynamic-slice(buf0, ivar, c0),
          dynamic_slice_sizes={1,8}
      val = s32[1,8] constant({{1,2,3,4,5,6,7,8}})
      updated = s32[4,8] dynamic-update-slice(buf1, val, ivar, c0)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], s32[4,8], s32[4,8]) tuple(next_ivar, buf0, updated)
    }

    condition {
      p0 = (s32[], s32[4,8], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input0 = s32[4,8] parameter(0)
      input1 = s32[4,8] parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8], s32[4,8]) tuple(c0, input0, input1)
      ROOT while = (s32[], s32[4,8], s32[4,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  auto* body = module->GetComputationWithName("body");
  auto* slice = body->GetInstructionWithName("slice");

  ASSERT_OK_AND_ASSIGN(auto chain, FindDynamicSliceChain(slice));
  EXPECT_EQ(chain.slices.size(), 1);
  EXPECT_EQ(chain.updates.size(), 0);
  EXPECT_EQ(chain.result, nullptr);
}

// Two DUS operations chained on the same buffer: DUS(DUS(buf, v1, off1), v2,
// off2). Both are collected and `result` points to the last one.
TEST_F(DynamicSliceAnalysisTest, FindSetDusChain) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      buf = s32[4,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      c4 = s32[] constant(4)
      c1 = s32[] constant(1)
      val = s32[1,4] constant({{1,2,3,4}})
      dus0 = s32[4,8] dynamic-update-slice(buf, val, ivar, c0)
      dus1 = s32[4,8] dynamic-update-slice(dus0, val, ivar, c4)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], s32[4,8]) tuple(next_ivar, dus1)
    }

    condition {
      p0 = (s32[], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = s32[4,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8]) tuple(c0, input)
      ROOT while = (s32[], s32[4,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  auto* body = module->GetComputationWithName("body");
  auto* dus0 = body->GetInstructionWithName("dus0");
  auto* dus1 = body->GetInstructionWithName("dus1");

  ASSERT_OK_AND_ASSIGN(auto chain, FindDynamicSliceChain(dus0));
  EXPECT_EQ(chain.updates.size(), 2);
  EXPECT_NE(std::find(chain.updates.begin(), chain.updates.end(),
                      Cast<HloDynamicUpdateSliceInstruction>(dus0)),
            chain.updates.end());
  EXPECT_NE(std::find(chain.updates.begin(), chain.updates.end(),
                      Cast<HloDynamicUpdateSliceInstruction>(dus1)),
            chain.updates.end());
  EXPECT_EQ(chain.result, dus1);
}

//===-----------------------------------------------------------------------===/
// IsNonOverlapping
//===-----------------------------------------------------------------------===/

// DS and DUS on the same range: non-overlapping because only DUS writes are
// checked (DS reads are allowed to alias with DUS writes).
TEST_F(DynamicSliceAnalysisTest, DsAndDusSameRangeIsNonOverlapping) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      buf = s32[4,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      slice = s32[1,8] dynamic-slice(buf, ivar, c0),
          dynamic_slice_sizes={1,8}
      val = s32[1,8] constant({{1,2,3,4,5,6,7,8}})
      updated = s32[4,8] dynamic-update-slice(buf, val, ivar, c0)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], s32[4,8]) tuple(next_ivar, updated)
    }

    condition {
      p0 = (s32[], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = s32[4,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8]) tuple(c0, input)
      ROOT while = (s32[], s32[4,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  auto* slice =
      module->GetComputationWithName("body")->GetInstructionWithName("slice");
  ASSERT_OK_AND_ASSIGN(auto chain, FindDynamicSliceChain(slice));
  auto result = IsNonOverlapping(chain);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

// Two DUS writing to different offsets (constant dim1 differs) are
// non-overlapping.
TEST_F(DynamicSliceAnalysisTest, NonOverlappingTwoDus) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      buf = s32[4,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      c4 = s32[] constant(4)
      c1 = s32[] constant(1)
      val = s32[1,4] constant({{1,2,3,4}})
      dus0 = s32[4,8] dynamic-update-slice(buf, val, ivar, c0)
      dus1 = s32[4,8] dynamic-update-slice(buf, val, ivar, c4)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], s32[4,8]) tuple(next_ivar, dus1)
    }

    condition {
      p0 = (s32[], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = s32[4,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8]) tuple(c0, input)
      ROOT while = (s32[], s32[4,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  auto* dus0 =
      module->GetComputationWithName("body")->GetInstructionWithName("dus0");
  ASSERT_OK_AND_ASSIGN(auto chain, FindDynamicSliceChain(dus0));
  EXPECT_EQ(chain.updates.size(), 2);
  auto result = IsNonOverlapping(chain);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

// Two DUS writing to the same offset — overlapping.
TEST_F(DynamicSliceAnalysisTest, OverlappingTwoDus) {
  constexpr absl::string_view kHlo = R"(
    body {
      p0 = (s32[], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      buf = s32[4,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      val = s32[1,8] constant({{1,2,3,4,5,6,7,8}})
      dus0 = s32[4,8] dynamic-update-slice(buf, val, ivar, c0)
      dus1 = s32[4,8] dynamic-update-slice(dus0, val, ivar, c0)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], s32[4,8]) tuple(next_ivar, dus1)
    }

    condition {
      p0 = (s32[], s32[4,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = s32[4,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8]) tuple(c0, input)
      ROOT while = (s32[], s32[4,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  auto* dus0 =
      module->GetComputationWithName("body")->GetInstructionWithName("dus0");
  ASSERT_OK_AND_ASSIGN(auto chain, FindDynamicSliceChain(dus0));
  EXPECT_EQ(chain.updates.size(), 2);
  auto result = IsNonOverlapping(chain);
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(*result);
}

}  // namespace
}  // namespace xla::gpu
