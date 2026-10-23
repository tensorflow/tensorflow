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
#include "xla/backends/gpu/codegen/emitters/reduction_base.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;

using MlirReductionBaseTest = HloHardwareIndependentTestBase;

TEST_F(MlirReductionBaseTest, TwoGroups) {
  auto module = ParseAndReturnVerifiedModule(R"(
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %p0 = f32[2] parameter(0)
      %p1 = f32[2] parameter(1)
      %c0 = f32[] constant(-inf)
      %r0 = f32[] reduce(%p0, %c0), dimensions={0}, to_apply=add
      %c1 = f32[] constant(inf)
      %r1 = f32[] reduce(%p1, %c1), dimensions={0}, to_apply=add
      ROOT %tuple = (f32[], f32[]) tuple(%r0, %r1)
    }
    ENTRY entry {
      %p0 = f32[2] parameter(0)
      %p1 = f32[2] parameter(1)
      ROOT %fusion = (f32[], f32[]) fusion(%p0, %p1), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  auto analysis = HloFusionAnalysis::Create(*root, device_info);
  auto reduction_groups = GroupDisjointReductions(analysis);
  EXPECT_THAT(reduction_groups.grouped_roots,
              ElementsAre(ElementsAre(&analysis.fusion_root(0).instruction()),
                          ElementsAre(&analysis.fusion_root(1).instruction())));
}

TEST_F(MlirReductionBaseTest, OneGroup) {
  auto module = ParseAndReturnVerifiedModule(R"(
    %add {
      %p0 = c128[] parameter(0)
      %p1 = c128[] parameter(1)
      ROOT %add.35 = c128[] add(c128[] %p0, c128[] %p1)
    }
    %fusion {
      %p0 = c128[1,2] parameter(0)
      %c0 = c128[] constant((0, 0))
      %reduce = c128[] reduce(%p0, %c0), dimensions={0,1}, to_apply=%add
      %real = f64[] real(c128[] %reduce)
      %imag = f64[] imag(c128[] %reduce)
      %negate = f64[] negate(f64[] %imag)
      ROOT %tuple.29 = (f64[], f64[]) tuple(f64[] %real, f64[] %negate)
    }
    ENTRY entry {
      %p0 = c128[1,2] parameter(0)
      ROOT %fusion = (f64[], f64[]) fusion(%p0), kind=kInput, calls=fusion
    })")
                    .value();

  auto device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info);
  auto reduction_groups = GroupDisjointReductions(analysis);
  EXPECT_THAT(reduction_groups.grouped_roots, SizeIs(1));
}

TEST_F(MlirReductionBaseTest, TwoGroupsSharedBroadcast) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %p0 = s16[4,8] parameter(0)
      %conv = f32[4,8] convert(%p0)
      %reshape = f32[32] reshape(%conv)
      %bitcast = f32[32] bitcast(%reshape)
      %b0 = f32[32,64] broadcast(%bitcast), dimensions={0}
      %iota = f32[64] iota(), iota_dimension=0
      %b1 = f32[32,64] broadcast(%iota), dimensions={1}
      %p1 = f32[32,64] parameter(1)
      %p2 = f32[32,64] parameter(2)
      %a0 = f32[32,64] add(%p1, %b0)
      %a0_iota = f32[32,64] add(%a0, %b1)
      %a1 = f32[32,64] add(%p2, %b0)
      %a1_iota = f32[32,64] add(%a1, %b1)
      %c0 = f32[] constant(0)
      %r0 = f32[32] reduce(%a0_iota, %c0), dimensions={1}, to_apply=add
      %r1 = f32[32] reduce(%a1_iota, %c0), dimensions={1}, to_apply=add
      ROOT %tuple = (f32[32], f32[32]) tuple(%r0, %r1)
    }
    ENTRY entry {
      %p0 = s16[4,8] parameter(0)
      %p1 = f32[32,64] parameter(1)
      %p2 = f32[32,64] parameter(2)
      ROOT %fusion = (f32[32], f32[32]) fusion(%p0, %p1, %p2), kind=kInput, calls=fusion
    })"));

  auto device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info);
  auto reduction_groups = GroupDisjointReductions(analysis);
  EXPECT_THAT(reduction_groups.grouped_roots,
              ElementsAre(ElementsAre(&analysis.fusion_root(0).instruction()),
                          ElementsAre(&analysis.fusion_root(1).instruction())));
}

TEST_F(MlirReductionBaseTest, OneGroupSharedExpensiveBroadcast) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %p0 = f32[32] parameter(0)
      %p1 = f32[32] parameter(1)
      %sum = f32[32] add(%p0, %p1)
      %b0 = f32[32,64] broadcast(%sum), dimensions={0}
      %p2 = f32[32,64] parameter(2)
      %p3 = f32[32,64] parameter(3)
      %a0 = f32[32,64] add(%p2, %b0)
      %a1 = f32[32,64] add(%p3, %b0)
      %c0 = f32[] constant(0)
      %r0 = f32[32] reduce(%a0, %c0), dimensions={1}, to_apply=add
      %r1 = f32[32] reduce(%a1, %c0), dimensions={1}, to_apply=add
      ROOT %tuple = (f32[32], f32[32]) tuple(%r0, %r1)
    }
    ENTRY entry {
      %p0 = f32[32] parameter(0)
      %p1 = f32[32] parameter(1)
      %p2 = f32[32,64] parameter(2)
      %p3 = f32[32,64] parameter(3)
      ROOT %fusion = (f32[32], f32[32]) fusion(%p0, %p1, %p2, %p3), kind=kInput, calls=fusion
    })"));

  auto device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info);
  auto reduction_groups = GroupDisjointReductions(analysis);
  EXPECT_THAT(reduction_groups.grouped_roots, SizeIs(1));
}

TEST_F(MlirReductionBaseTest, TwoGroupsBroadcastToFirstRootDirectToSecondRoot) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %p0 = f32[32] parameter(0)
      %b0 = f32[32,64] broadcast(%p0), dimensions={0}
      %p1 = f32[32,64] parameter(1)
      %a0 = f32[32,64] add(%p1, %b0)
      %c0 = f32[] constant(0)
      %r0 = f32[32] reduce(%a0, %c0), dimensions={1}, to_apply=add
      %r1 = f32[] reduce(%p0, %c0), dimensions={0}, to_apply=add
      ROOT %tuple = (f32[32], f32[]) tuple(%r0, %r1)
    }
    ENTRY entry {
      %p0 = f32[32] parameter(0)
      %p1 = f32[32,64] parameter(1)
      ROOT %fusion = (f32[32], f32[]) fusion(%p0, %p1), kind=kInput, calls=fusion
    })"));

  auto device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info);
  auto reduction_groups = GroupDisjointReductions(analysis);
  EXPECT_THAT(reduction_groups.grouped_roots,
              ElementsAre(ElementsAre(&analysis.fusion_root(0).instruction()),
                          ElementsAre(&analysis.fusion_root(1).instruction())));
}

TEST_F(MlirReductionBaseTest, OneGroupDiamondWithShortBroadcastPath) {
  // %conv (and %p0) reaches %r0 via BOTH a shorter cheap-broadcast path
  // (%r0 -> %r0_bcast -> %b0 -> %conv, 3 hops) and a longer unbroadcasted path
  // (%r0 -> %r0_epilogue -> %neg2 -> %neg1 -> %conv, 4 hops), while also
  // reaching %r1 directly (%r1 -> %conv). Even though BFS visits %b0 before
  // %neg1, skipping %b0 ensures the unbroadcasted path still reaches %conv and
  // %p0, grouping %r0 and %r1 into a single reduction group.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %p0 = s16[32] parameter(0)
      %conv = f32[32] convert(%p0)
      %b0 = f32[32,64] broadcast(%conv), dimensions={0}
      %p1 = f32[32,64] parameter(1)
      %p2 = f32[32,64] parameter(2)
      %c0 = f32[] constant(0)
      %r0_base = f32[32] reduce(%p1, %c0), dimensions={1}, to_apply=add
      %neg1 = f32[32] negate(%conv)
      %neg2 = f32[32] negate(%neg1)
      %r0_epilogue = f32[32] add(%r0_base, %neg2)
      %r0_bcast = f32[32] reduce(%b0, %c0), dimensions={1}, to_apply=add
      %r0 = f32[32] add(%r0_epilogue, %r0_bcast)
      %r1_base = f32[32] reduce(%p2, %c0), dimensions={1}, to_apply=add
      %r1 = f32[32] add(%r1_base, %conv)
      ROOT %tuple = (f32[32], f32[32]) tuple(%r0, %r1)
    }
    ENTRY entry {
      %p0 = s16[32] parameter(0)
      %p1 = f32[32,64] parameter(1)
      %p2 = f32[32,64] parameter(2)
      ROOT %fusion = (f32[32], f32[32]) fusion(%p0, %p1, %p2), kind=kInput, calls=fusion
    })"));

  auto device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info);
  auto reduction_groups = GroupDisjointReductions(analysis);
  EXPECT_THAT(reduction_groups.grouped_roots, SizeIs(1));
}

TEST_F(MlirReductionBaseTest, TwoGroupsWithBroadcastSideOutput) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %p0 = s16[32] parameter(0)
      %conv = f32[32] convert(%p0)
      %b0 = f32[32,64] broadcast(%conv), dimensions={0}
      %p1 = f32[32,64] parameter(1)
      %p2 = f32[32,64] parameter(2)
      %a0 = f32[32,64] add(%p1, %b0)
      %a1 = f32[32,64] add(%p2, %b0)
      %c0 = f32[] constant(0)
      %r0 = f32[32] reduce(%a0, %c0), dimensions={1}, to_apply=add
      %r1 = f32[32] reduce(%a1, %c0), dimensions={1}, to_apply=add
      ROOT %tuple = (f32[32], f32[32], f32[32,64]) tuple(%r0, %r1, %b0)
    }
    ENTRY entry {
      %p0 = s16[32] parameter(0)
      %p1 = f32[32,64] parameter(1)
      %p2 = f32[32,64] parameter(2)
      ROOT %fusion = (f32[32], f32[32], f32[32,64])
          fusion(%p0, %p1, %p2), kind=kInput, calls=fusion
    })"));

  auto device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info);
  auto reduction_groups = GroupDisjointReductions(analysis);
  EXPECT_THAT(reduction_groups.grouped_roots,
              ElementsAre(ElementsAre(&analysis.fusion_root(0).instruction(),
                                      &analysis.fusion_root(2).instruction()),
                          ElementsAre(&analysis.fusion_root(1).instruction())));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
