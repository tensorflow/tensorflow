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

#include <gtest/gtest.h>
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/protobuf_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

class HloFusionAnalysisTest : public HloTestBase {};

TEST_F(HloFusionAnalysisTest, DoesNotPeekOutsideBoundary) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
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
  EXPECT_EQ(analysis.GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kLoop);

  auto analysis_fused =
      HloFusionAnalysis::Create(*root->operand(0), *root, device_info);
  EXPECT_EQ(analysis_fused.GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionWithMultipleUsers) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
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
  EXPECT_EQ(analysis.GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionEpilogueFusion) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
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
  EXPECT_EQ(analysis.GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionEpilogueFusionPartiallyFused) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
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
  EXPECT_EQ(analysis.GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionEpilogueFusionPartiallyFusedInConsumer) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
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
  EXPECT_EQ(analysis.GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionEpilogueFusionPartiallyFusedInBoth) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
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
  EXPECT_EQ(analysis.GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReduceMultiOutputFusionWithTransposeBitcast) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
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
  EXPECT_EQ(analysis.GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, InvalidReduceMultiOutputFusion) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
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
  EXPECT_EQ(analysis.GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kLoop);
}

TEST_F(HloFusionAnalysisTest, InvalidDevice) {
  // Verifies that an analysis can be created even with an invalid/empty device
  // info, and that the emitter type is determined correctly.
  // Don't rely on this behavior.
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
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
  stream_executor::DeviceDescription device_info(device_info_proto);
  device_info.set_threads_per_warp(32);

  auto* root = module->entry_computation()->root_instruction();
  auto analysis_fused =
      HloFusionAnalysis::Create(*root->operand(0), *root, device_info);
  EXPECT_EQ(analysis_fused.GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ConcatFusion) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
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
  EXPECT_EQ(analysis.GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kConcatenate);
}

TEST_F(HloFusionAnalysisTest, ExtractValidGpuBackendConfig) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

    ENTRY entry {
      %x = s32[64,64,64] parameter(0)
      %y = s32[64,64,64] parameter(1)
      ROOT %root = s32[64,128,64] concatenate(x, y), dimensions={1}, backend_config={"outer_dimension_partitions": ["1"]}
    })"));

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info);

  EXPECT_TRUE(
      protobuf_util::ProtobufEquals(analysis.fusion_backend_config(),
                                    FusionBackendConfig::default_instance()));
}

TEST_F(HloFusionAnalysisTest,
       InvalidGpuBackendConfig_ProducerConsumer_Ignored) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
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

  EXPECT_TRUE(
      protobuf_util::ProtobufEquals(analysis.fusion_backend_config(),
                                    FusionBackendConfig::default_instance()));
}

}  // namespace
}  // namespace xla::gpu
