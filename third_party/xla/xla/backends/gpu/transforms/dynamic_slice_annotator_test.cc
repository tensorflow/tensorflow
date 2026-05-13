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

#include "xla/backends/gpu/transforms/dynamic_slice_annotator.h"

#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"

namespace xla::gpu {
namespace {

using DynamicSliceAnnotatorTest = HloHardwareIndependentTestBase;

std::optional<DynamicSliceConfig> GetDynamicSliceConfig(
    const HloInstruction* instr) {
  auto config = instr->backend_config<GpuBackendConfig>();
  if (!config.ok() || !config->has_dynamic_slice_config()) {
    return std::nullopt;
  }
  return config->dynamic_slice_config();
}

TEST_F(DynamicSliceAnnotatorTest, AnnotatesDsInWhileLoop) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(DynamicSliceAnnotator().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  auto* slice =
      module->GetComputationWithName("body")->GetInstructionWithName("slice");
  auto config = GetDynamicSliceConfig(slice);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->loop_index(), 0);
  EXPECT_EQ(config->byte_offset(), 0);
  EXPECT_EQ(config->byte_stride(), 256);
}

TEST_F(DynamicSliceAnnotatorTest, AnnotatesDusInWhileLoop) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(DynamicSliceAnnotator().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  auto* updated =
      module->GetComputationWithName("body")->GetInstructionWithName("updated");
  auto config = GetDynamicSliceConfig(updated);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->loop_index(), 0);
  EXPECT_EQ(config->byte_offset(), 32);
  EXPECT_EQ(config->byte_stride(), 256);
}

TEST_F(DynamicSliceAnnotatorTest, AnnotatesConstantOffsetOutsideWhileLoop) {
  constexpr absl::string_view kHlo = R"(
    ENTRY main {
      input = s32[4,8] parameter(0)
      c1 = s32[] constant(1)
      c0 = s32[] constant(0)
      ROOT slice = s32[1,8] dynamic-slice(input, c1, c0),
          dynamic_slice_sizes={1,8}
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(DynamicSliceAnnotator().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  auto* slice = module->entry_computation()->root_instruction();
  auto config = GetDynamicSliceConfig(slice);
  ASSERT_TRUE(config.has_value());
  EXPECT_FALSE(config->has_loop_index());
  EXPECT_EQ(config->byte_offset(), 32);
  EXPECT_EQ(config->byte_stride(), 0);
}

TEST_F(DynamicSliceAnnotatorTest, AnnotatesDusInAsyncComputation) {
  constexpr absl::string_view kHlo = R"(
    async_computation {
      p0 = f32[4]{0} parameter(0)
      p1 = f32[1]{0} parameter(1)
      p2 = s32[] parameter(2)
      ROOT dus = f32[4]{0} dynamic-update-slice(p0, p1, p2)
    }

    body {
      p0 = (s32[], f32[4]{0}) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      buf = f32[4]{0} get-tuple-element(p0), index=1
      val = f32[] constant(1)
      reshape = f32[1]{0} reshape(val)
      async-start = ((f32[4]{0}, f32[1]{0}, s32[]), f32[4]{0}, u32[])
          async-start(buf, reshape, ivar), calls=async_computation
      updated = f32[4]{0} async-done(async-start), calls=async_computation
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], f32[4]{0}) tuple(next_ivar, updated)
    }

    condition {
      p0 = (s32[], f32[4]{0}) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      buf = f32[4]{0} parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[4]{0}) tuple(c0, buf)
      ROOT while = (s32[], f32[4]{0}) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(DynamicSliceAnnotator().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  auto* dus = module->GetComputationWithName("async_computation")
                  ->GetInstructionWithName("dus");
  auto config = GetDynamicSliceConfig(dus);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->loop_index(), 0);
  EXPECT_EQ(config->byte_offset(), 0);
  EXPECT_EQ(config->byte_stride(), 4);
}

TEST_F(DynamicSliceAnnotatorTest, AnnotatesDusInNestedCalls) {
  constexpr absl::string_view kHlo = R"(
    inner_computation {
      p0 = f32[4]{0} parameter(0)
      p1 = f32[1]{0} parameter(1)
      p2 = s32[] parameter(2)
      ROOT dus = f32[4]{0} dynamic-update-slice(p0, p1, p2)
    }

    async_computation {
      p0 = f32[4]{0} parameter(0)
      p1 = f32[1]{0} parameter(1)
      p2 = s32[] parameter(2)
      ROOT call = f32[4]{0} call(p0, p1, p2), to_apply=inner_computation
    }

    body {
      p0 = (s32[], f32[4]{0}) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      buf = f32[4]{0} get-tuple-element(p0), index=1
      val = f32[] constant(1)
      reshape = f32[1]{0} reshape(val)
      async-start = ((f32[4]{0}, f32[1]{0}, s32[]), f32[4]{0}, u32[])
          async-start(buf, reshape, ivar), calls=async_computation
      updated = f32[4]{0} async-done(async-start), calls=async_computation
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], f32[4]{0}) tuple(next_ivar, updated)
    }

    condition {
      p0 = (s32[], f32[4]{0}) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      buf = f32[4]{0} parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[4]{0}) tuple(c0, buf)
      ROOT while = (s32[], f32[4]{0}) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(DynamicSliceAnnotator().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  auto* dus = module->GetComputationWithName("inner_computation")
                  ->GetInstructionWithName("dus");
  auto config = GetDynamicSliceConfig(dus);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->loop_index(), 0);
  EXPECT_EQ(config->byte_offset(), 0);
  EXPECT_EQ(config->byte_stride(), 4);
}

}  // namespace
}  // namespace xla::gpu
