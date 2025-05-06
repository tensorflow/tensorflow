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
#include "xla/backends/gpu/codegen/copy.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace gpu {
namespace {

using CopyFusionTest = HloHardwareIndependentTestBase;

using ::testing::IsEmpty;

const HloFusionInstruction& GetFusion(HloModule* module) {
  const HloInstruction* fusion =
      module->GetComputationWithName("dynamic_slice")->FusionInstruction();
  return *Cast<HloFusionInstruction>(fusion);
}

TEST_F(CopyFusionTest, ValidCandidate) {
  auto module = ParseAndReturnVerifiedModule(R"(
    dynamic_slice {
      p0 = f32[100] parameter(0)
      p1 = s32[] parameter(1)
      ROOT slice = f32[10] dynamic-slice(p0, p1), dynamic_slice_sizes={10}
    }

    ENTRY main {
      p0 = f32[100] parameter(0)
      p1 = s32[] parameter(1)
      ROOT fusion = f32[10] fusion(p0, p1), kind=kLoop, calls=dynamic_slice
    }
  )")
                    .value();

  EXPECT_TRUE(DynamicMemcpyFusion::IsCandidateFusion(GetFusion(module.get())));
}

TEST_F(CopyFusionTest, ValidCandidateClamped) {
  auto module = ParseAndReturnVerifiedModule(R"(
    dynamic_slice {
      p0 = f32[100] parameter(0)
      p1 = s32[] parameter(1)
      c1 = s32[] constant(1)
      sum = s32[] add(p1, c1)
      ROOT slice = f32[100] dynamic-slice(p0, sum), dynamic_slice_sizes={100}
    }

    ENTRY main {
      p0 = f32[100] parameter(0)
      p1 = s32[] parameter(1)
      ROOT fusion = f32[100] fusion(p0, p1), kind=kLoop, calls=dynamic_slice
    }
  )")
                    .value();

  EXPECT_TRUE(DynamicMemcpyFusion::IsCandidateFusion(GetFusion(module.get())));
}

TEST_F(CopyFusionTest, ClampedConstantPositive) {
  auto module = ParseAndReturnVerifiedModule(R"(
    dynamic_slice {
      p0 = f32[200] parameter(0)
      c195 = s32[] constant(195)
      ROOT slice = f32[100] dynamic-slice(p0, c195), dynamic_slice_sizes={100}
    }

    ENTRY main {
      p0 = f32[200] parameter(0)
      ROOT fusion = f32[100] fusion(p0), kind=kLoop, calls=dynamic_slice
    }
  )")
                    .value();

  auto descriptor = DynamicMemcpyFusion::GetMemcpyDescriptorForFusion(
      GetFusion(module.get()));

  ASSERT_TRUE(descriptor.has_value());
  EXPECT_THAT(descriptor->src_dynamic_offsets, IsEmpty());
  EXPECT_THAT(descriptor->dst_dynamic_offsets, IsEmpty());
  EXPECT_EQ(descriptor->src_byte_static_offset, sizeof(float) * 100);
  EXPECT_EQ(descriptor->dst_byte_static_offset, 0);
}

TEST_F(CopyFusionTest, ClampedConstantNegative) {
  auto module = ParseAndReturnVerifiedModule(R"(
    dynamic_slice {
      p0 = f32[200] parameter(0)
      cn1 = s32[] constant(-1)
      ROOT slice = f32[100] dynamic-slice(p0, cn1), dynamic_slice_sizes={100}
    }

    ENTRY main {
      p0 = f32[200] parameter(0)
      ROOT fusion = f32[100] fusion(p0), kind=kLoop, calls=dynamic_slice
    }
  )")
                    .value();

  auto descriptor = DynamicMemcpyFusion::GetMemcpyDescriptorForFusion(
      GetFusion(module.get()));

  ASSERT_TRUE(descriptor.has_value());
  EXPECT_THAT(descriptor->src_dynamic_offsets, IsEmpty());
  EXPECT_THAT(descriptor->dst_dynamic_offsets, IsEmpty());
  EXPECT_EQ(descriptor->src_byte_static_offset, 0);
  EXPECT_EQ(descriptor->dst_byte_static_offset, 0);
}

constexpr char kSliceMemcpyModule[] = R"(
    dynamic_slice {
      p0 = s32[4,8,8,8] parameter(0)
      p1 = s32[] parameter(1)
      c1 = s32[] constant(1)
      p2 = s32[] parameter(2)

      // Test all supported kinds of offsets: derived from the while loop's
      // induction variable (p1), constant (c1) and always clamped to 0, so
      // the value is irrelevant (p2).
      // Also test an offset that is computed inside the fusion (o3).
      o3 = s32[] add(p1, c1)
      ROOT slice = s32[1,1,1,8] dynamic-slice(p0, p1, c1, o3, p2),
          dynamic_slice_sizes={1,1,1,8}
    }

    body {
      p0 = (s32[], s32[4,8,8,8], s32[1,1,1,8], s32[]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[4,8,8,8] get-tuple-element(p0), index=1

      c1 = s32[] constant(1)
      c5 = s32[] constant(5)

      offset1 = s32[] remainder(ivar, c5)
      offset2 = s32[] get-tuple-element(p0), index=3

      slice = s32[1,1,1,8] fusion(input, offset1, offset2), kind=kLoop, calls=dynamic_slice,
          backend_config={"fusion_backend_config":{"kind":"__dynamic_memcpy"}}
      next_ivar = s32[] add(ivar, c1)

      ROOT result = (s32[], s32[4,8,8,8], s32[1,1,1,8], s32[])
          tuple(next_ivar, input, slice, offset2)
    }

    condition {
      p0 = (s32[], s32[4,8,8,8], s32[1,1,1,8], s32[]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c6 = s32[] constant(6)
      ROOT cmp = pred[] compare(ivar, c6), direction=LT
    }

    ENTRY main {
      input = s32[4,8,8,8] parameter(0)
      init_acc = s32[1,1,1,8] parameter(1)
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      tuple = (s32[], s32[4,8,8,8], s32[1,1,1,8], s32[]) tuple(c0, input, init_acc, c1)
      ROOT while = (s32[], s32[4,8,8,8], s32[1,1,1,8], s32[]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"6"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

TEST_F(CopyFusionTest, BuildSliceDescriptor) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kSliceMemcpyModule));

  auto descriptor = DynamicMemcpyFusion::GetMemcpyDescriptorForFusion(
      GetFusion(module.get()));

  ASSERT_TRUE(descriptor.has_value());
  ASSERT_THAT(descriptor->src_dynamic_offsets, ::testing::SizeIs(2));

  EXPECT_EQ(descriptor->src_byte_static_offset, 8 * 8 * sizeof(float));

  const auto& offset1 = descriptor->src_dynamic_offsets[0];
  EXPECT_EQ(offset1.while_loop->name(), "while");
  EXPECT_EQ(offset1.induction_variable->name(), "ivar");
  EXPECT_EQ(offset1.offset->name(), "p1");
  EXPECT_EQ(offset1.dimension_size, 4);
  EXPECT_EQ(offset1.byte_stride, 8 * 8 * 8 * sizeof(float));

  const auto& offset2 = descriptor->src_dynamic_offsets[1];
  EXPECT_EQ(offset2.while_loop->name(), "while");
  EXPECT_EQ(offset2.induction_variable->name(), "ivar");
  EXPECT_EQ(offset2.offset->name(), "o3");
  EXPECT_EQ(offset2.dimension_size, 8);
  EXPECT_EQ(offset2.byte_stride, 8 * sizeof(float));

  EXPECT_THAT(descriptor->dst_dynamic_offsets, ::testing::IsEmpty());
  EXPECT_EQ(descriptor->dst_byte_static_offset, 0);
}

constexpr char kUpdateSliceMemcpyModule[] = R"(
    dynamic_slice {
      p0 = s32[4,8,8] parameter(0)
      p1 = s32[1,1,8] parameter(1)
      p2 = s32[] parameter(2)
      c1 = s32[] constant(1)

      ROOT update-slice = s32[4,8,8] dynamic-update-slice(p0, p1, p2, c1, c1)
    }

    body {
      p0 = (s32[], s32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[4,8,8] get-tuple-element(p0), index=1
      val = s32[1,1,8] constant({{{1,2,3,4,5,6,7,8}}})

      updated = s32[4,8,8] fusion(input, val, ivar), kind=kLoop, calls=dynamic_slice,
          backend_config={"fusion_backend_config":{"kind":"__dynamic_memcpy"}}
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)

      ROOT result = (s32[], s32[4,8,8])
          tuple(next_ivar, updated)
    }

    condition {
      p0 = (s32[], s32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c6 = s32[] constant(6)
      ROOT cmp = pred[] compare(ivar, c6), direction=LT
    }

    ENTRY main {
      input = s32[4,8,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8,8]) tuple(c0, input)
      ROOT while = (s32[], s32[4,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"6"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

TEST_F(CopyFusionTest, BuildUpdateSliceDescriptor) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kUpdateSliceMemcpyModule));

  auto descriptor = DynamicMemcpyFusion::GetMemcpyDescriptorForFusion(
      GetFusion(module.get()));

  ASSERT_TRUE(descriptor.has_value());
  EXPECT_THAT(descriptor->src_dynamic_offsets, ::testing::IsEmpty());
  EXPECT_EQ(descriptor->src_byte_static_offset, 0);

  ASSERT_THAT(descriptor->dst_dynamic_offsets, ::testing::SizeIs(1));
  const auto& offset = descriptor->dst_dynamic_offsets[0];
  EXPECT_EQ(descriptor->dst_byte_static_offset, 32);
  EXPECT_EQ(offset.while_loop->name(), "while");
  EXPECT_EQ(offset.induction_variable->name(), "ivar");
  EXPECT_EQ(offset.offset->name(), "p2");
  EXPECT_EQ(offset.dimension_size, 4);
  EXPECT_EQ(offset.byte_stride, 8 * 8 * sizeof(float));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
