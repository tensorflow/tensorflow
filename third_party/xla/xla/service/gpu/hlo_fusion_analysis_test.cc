/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/service/gpu/hlo_fusion_analysis.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;

class HloFusionAnalysisTest : public HloHardwareIndependentTestBase {};

TEST_F(HloFusionAnalysisTest, DoesNotPeekOutsideBoundary) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    ENTRY main {
      %p0 = f32[1024] parameter(0)
      %p1 = f32[] parameter(1)
      %reduce = f32[] reduce(%p0, %p1), dimensions={0}, to_apply=add
      ROOT %bitcast = s32[] bitcast(%reduce)
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kLoop);

  auto analysis_fused =
      HloFusionAnalysis::Create(*root->operand(0), *root, device_info);
  EXPECT_EQ(analysis_fused.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionWithMultipleUsers) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    fused_computation {
      %p0 = f32[1024] parameter(0)
      %p1 = f32[] parameter(1)
      %reduce = f32[] reduce(%p0, %p1), dimensions={0}, to_apply=add
      %negate = f32[] negate(%reduce)
      %log = f32[] log(%reduce)
      ROOT %tuple = (f32[], f32[]) tuple(%negate, %log)
    }

    ENTRY main {
      %p0 = f32[1024] parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %fusion = (f32[], f32[]) fusion(%p0, %p1), kind=kLoop, calls=fused_computation
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto analysis = HloFusionAnalysis::Create(
      FusionBackendConfig::default_instance(),
      HloFusionAdaptor::ForInstruction(
          module->entry_computation()->root_instruction()),
      &device_info);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionEpilogueFusion) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    fused_computation {
      %p0 = f32[1024] parameter(0)
      %p1 = f32[] parameter(1)
      %reduce = f32[] reduce(%p0, %p1), dimensions={0}, to_apply=add
      ROOT %negate = f32[] negate(%reduce)
    }

    ENTRY main {
      %p0 = f32[1024] parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %fusion = f32[] fusion(%p0, %p1), kind=kInput, calls=fused_computation
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(
      FusionBackendConfig::default_instance(),
      HloFusionAdaptor::ForInstruction(root), &device_info);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionEpilogueFusionPartiallyFused) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    fusion {
      %p0 = f32[1024] parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %reduce = f32[] reduce(%p0, %p1), dimensions={0}, to_apply=add
    }

    ENTRY main {
      %p0 = f32[1024] parameter(0)
      %p1 = f32[] parameter(1)
      %fusion = f32[] fusion(%p0, %p1), kind=kInput, calls=fusion
      ROOT %negate = f32[] negate(%fusion)
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();

  auto analysis =
      HloFusionAnalysis::Create(*root->operand(0), *root, device_info);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionEpilogueFusionPartiallyFusedInConsumer) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    fusion {
      %p0 = f32[] parameter(0)
      ROOT %negate = f32[] negate(%p0)
    }

    ENTRY main {
      %p0 = f32[1024] parameter(0)
      %p1 = f32[] parameter(1)
      %reduce = f32[] reduce(%p0, %p1), dimensions={0}, to_apply=add
      ROOT %fusion = f32[] fusion(%reduce), kind=kInput, calls=fusion
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis =
      HloFusionAnalysis::Create(*root->operand(0), *root, device_info);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionEpilogueFusionPartiallyFusedInBoth) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    fusion.1 {
      %p0 = f32[1024] parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %reduce = f32[] reduce(%p0, %p1), dimensions={0}, to_apply=add
    }

    fusion.2 {
      %p0 = f32[] parameter(0)
      ROOT %negate = f32[] negate(%p0)
    }

    ENTRY main {
      %p0 = f32[1024] parameter(0)
      %p1 = f32[] parameter(1)
      %fusion.1 = f32[] fusion(%p0, %p1), kind=kInput, calls=fusion.1
      ROOT %fusion.2 = f32[] fusion(%fusion.1), kind=kInput, calls=fusion.2
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis =
      HloFusionAnalysis::Create(*root->operand(0), *root, device_info);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReduceMultiOutputFusionWithTransposeBitcast) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    fusion {
      %p0 = f32[1024, 512]{1,0} parameter(0)
      %p1 = f32[] parameter(1)
      %reduce = f32[1024]{0} reduce(%p0, %p1), dimensions={1}, to_apply=add
      %bitcast = f32[512, 1024]{0,1} bitcast(%p0)
      ROOT res = (f32[1024]{0}, f32[512, 1024]{0,1}) tuple(%reduce, %bitcast)
    }

    ENTRY main {
      %p0 = f32[1024, 512]{1,0} parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %fusion = (f32[1024]{0}, f32[512, 1024]{0,1}) fusion(%p0, %p1), kind=kInput, calls=fusion
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, InvalidReduceMultiOutputFusion) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    fusion {
      %p0 = f32[1024, 1024]{1,0} parameter(0)
      %p1 = f32[] parameter(1)
      %reduce = f32[1024]{0} reduce(%p0, %p1), dimensions={0}, to_apply=add
      %reduce2 = f32[1024]{0} reduce(%p0, %p1), dimensions={1}, to_apply=add
      ROOT res = (f32[1024]{0}, f32[1024]{0}) tuple(reduce, reduce2)
    }

    ENTRY main {
      %p0 = f32[1024, 1024]{1,0} parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %fusion = (f32[1024]{0}, f32[1024]{0}) fusion(%p0, %p1), kind=kInput, calls=fusion
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info);
  // We expect to fallback to the loop emitter, because the two reductions are
  // not compatible as they reduce over different dimensions.
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kLoop);
}

TEST_F(HloFusionAnalysisTest, InvalidDevice) {
  // Verifies that an analysis can be created even with an invalid/empty device
  // info, and that the emitter type is determined correctly.
  // Don't rely on this behavior.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    ENTRY main {
      %p0 = f32[1024,128] parameter(0)
      %p1 = f32[] parameter(1)
      %reduce = f32[128] reduce(%p0, %p1), dimensions={0}, to_apply=add
      ROOT %bitcast = s32[128] bitcast(%reduce)
    })"));

  stream_executor::GpuDeviceInfoProto device_info_proto;
  ASSERT_OK_AND_ASSIGN(
      auto device_info,
      stream_executor::DeviceDescription::FromProto(device_info_proto));
  device_info.set_threads_per_warp(32);

  auto* root = module->entry_computation()->root_instruction();
  auto analysis_fused =
      HloFusionAnalysis::Create(*root->operand(0), *root, device_info);
  EXPECT_EQ(analysis_fused.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ConcatFusion) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    fused_computation {
      %p0 = f32[128] parameter(0)
      %p1 = f32[128] parameter(1)
      %add = f32[128] add(p0, p0)
      %concat = f32[256] concatenate(%add, %p1), dimensions={0}
      ROOT %negate = f32[256] negate(%concat)
    }

    ENTRY main {
      %p0 = f32[128] parameter(0)
      %p1 = f32[128] parameter(1)
      ROOT %fusion = f32[256] fusion(%p0, %p1), kind=kInput, calls=fused_computation
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(
      FusionBackendConfig::default_instance(),
      HloFusionAdaptor::ForInstruction(root), &device_info);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kConcatenate);
}

TEST_F(HloFusionAnalysisTest, SortFusion) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    less_than {
      lhs.0 = f32[] parameter(0)
      rhs.0 = f32[] parameter(1)
      lhs.1 = s32[] parameter(2)
      rhs.1 = s32[] parameter(3)
      ROOT lt = pred[] compare(lhs.0, rhs.0), direction=LT
    }

    fused_computation {
      p0 = f32[256] parameter(0)
      iota = s32[256] iota(), iota_dimension=0
      ROOT sort = (f32[256], s32[256]) sort(p0, iota), dimensions={0}, to_apply=less_than, is_stable=false
    }

    ENTRY main {
      p = f32[256] parameter(0)
      ROOT fusion = (f32[256], s32[256]) fusion(p), kind=kInput, calls=fused_computation
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(
      FusionBackendConfig::default_instance(),
      HloFusionAdaptor::ForInstruction(root), &device_info);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kSort);
}

TEST_F(HloFusionAnalysisTest, ExtractValidGpuBackendConfig) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

    fused_computation.1 {
      %x = s32[64] parameter(0)
      %y = s32[64] parameter(1)
      ROOT %root = s32[64] add(%x, %y)
    }

    fused_computation.2 {
      %x = s32[64] parameter(0)
      %y = s32[64] parameter(1)
      ROOT %root = s32[64] add(%x, %y)
    }

    ENTRY entry {
      %x = s32[64] parameter(0)
      %y = s32[64] parameter(1)
      %fusion.1 = s32[64] fusion(%x, %y), kind=kLoop, calls=fused_computation.1, backend_config={"fusion_backend_config": {kind: "__triton"}}
      ROOT %fusion.2 = s32[64] fusion(%fusion.1, %y), kind=kLoop, calls=fused_computation.2
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  auto* consumer = module->entry_computation()->root_instruction();
  auto* producer = consumer->operand(0);

  auto producer_analysis = HloFusionAnalysis::Create(*producer, device_info);
  EXPECT_EQ(producer_analysis.fusion_backend_config().kind(),
            kTritonFusionKind);

  auto producer_consumer_analysis =
      HloFusionAnalysis::Create(*producer, *consumer, device_info);
  EXPECT_EQ(producer_consumer_analysis.fusion_backend_config().kind(),
            kTritonFusionKind);
}

TEST_F(HloFusionAnalysisTest,
       InvalidGpuBackendConfig_SingleInstruction_Ignored) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

    ENTRY entry {
      %x = s32[64,64,64] parameter(0)
      %y = s32[64,64,64] parameter(1)
      ROOT %root = s32[64,128,64] concatenate(x, y), dimensions={1}, backend_config={"outer_dimension_partitions": ["1"]}
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info);

  EXPECT_THAT(analysis.fusion_backend_config(),
              EqualsProto(FusionBackendConfig::default_instance()));
}

TEST_F(HloFusionAnalysisTest,
       InvalidGpuBackendConfig_ProducerConsumer_Ignored) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

    fused_computation {
      %x = s32[64] parameter(0)
      %y = s32[64] parameter(1)
      ROOT %root = s32[64] add(%x, %y)
    }

    ENTRY entry {
      %x = s32[64] parameter(0)
      %y = s32[64] parameter(1)
      %fusion = s32[64] fusion(%x, %y), kind=kLoop, calls=fused_computation, backend_config={"invalid_field": "some_value"}
      ROOT %root = s32[128] concatenate(fusion, y), dimensions={0}, backend_config={"invalid_field": "some_value"}
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  auto* consumer = module->entry_computation()->root_instruction();
  auto* producer = consumer->operand(0);
  auto analysis = HloFusionAnalysis::Create(*producer, *consumer, device_info);

  EXPECT_THAT(analysis.fusion_backend_config(),
              EqualsProto(FusionBackendConfig::default_instance()));
}

TEST_F(HloFusionAnalysisTest, ConcatenateFusion) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

    fusion {
      p0 = bf16[128] parameter(0)
      p1 = bf16[128] parameter(1)
      p2 = bf16[256] parameter(2)
      concatenate = bf16[256] concatenate(p0, p1), dimensions={0}
      ROOT multiply = bf16[256] multiply(concatenate, p2)
    }

    ENTRY entry_computation {
      p0 = bf16[128] parameter(0)
      p1 = bf16[128] parameter(1)
      p2 = bf16[256] parameter(2)
      ROOT fusion = bf16[256] fusion(p0, p1, p2), kind=kLoop, calls=fusion
  })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto* multiply = root->fused_instructions_computation()->root_instruction();
  auto* concatenate = multiply->operand(0);
  auto analysis = HloFusionAnalysis::Create(*root, device_info);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kConcatenate);
  EXPECT_EQ(&analysis.fusion_root(0).instruction(), multiply);
  EXPECT_EQ(&analysis.fusion_hero(0).instruction(), concatenate);
}

TEST_F(HloFusionAnalysisTest, ConcatenateFusionFallbackToLoop) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

    fusion {
      p0 = bf16[32] parameter(0)
      p1 = bf16[32] parameter(1)
      p2 = bf16[32] parameter(2)
      p3 = bf16[32] parameter(3)
      p4 = bf16[32] parameter(4)
      p5 = bf16[160] parameter(5)
      concatenate = bf16[160] concatenate(p0, p1, p2, p3, p4), dimensions={0}
      ROOT multiply = bf16[160] multiply(concatenate, p5)
    }

    ENTRY entry_computation {
      p0 = bf16[32] parameter(0)
      p1 = bf16[32] parameter(1)
      p2 = bf16[32] parameter(2)
      p3 = bf16[32] parameter(3)
      p4 = bf16[32] parameter(4)
      p5 = bf16[160] parameter(5)
      ROOT fusion = bf16[160] fusion(p0, p1, p2, p3, p4, p5), kind=kLoop, calls=fusion
  })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto* multiply = root->fused_instructions_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kLoop);
  EXPECT_EQ(&analysis.fusion_root(0).instruction(), multiply);
  EXPECT_EQ(&analysis.fusion_hero(0).instruction(), multiply);
}

}  // namespace
}  // namespace xla::gpu
