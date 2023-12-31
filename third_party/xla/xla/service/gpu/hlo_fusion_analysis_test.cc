/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

class HloFusionAnalysisTest : public HloTestBase {};

TEST_F(HloFusionAnalysisTest, DoesNotPeekOutsideBoundary) {
  auto module = ParseAndReturnVerifiedModule(R"(
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
    })")
                    .value();

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info);
  ASSERT_NE(analysis, std::nullopt);
  EXPECT_EQ(analysis->GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kLoop);

  auto analysis_fused =
      AnalyzeProducerConsumerFusion(*root->operand(0), *root, device_info);
  ASSERT_NE(analysis_fused, std::nullopt);
  EXPECT_EQ(analysis_fused->GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionWithMultipleUsers) {
  auto module = ParseAndReturnVerifiedModule(R"(
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
    })")
                    .value();

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  TF_ASSERT_OK_AND_ASSIGN(
      auto analysis, HloFusionAnalysis::Create(
                         FusionBackendConfig::default_instance(),
                         HloFusionAdaptor::ForInstruction(
                             module->entry_computation()->root_instruction()),
                         &device_info));
  EXPECT_EQ(analysis.GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionEpilogueFusion) {
  auto module = ParseAndReturnVerifiedModule(R"(
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
    })")
                    .value();

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(
      auto analysis, HloFusionAnalysis::Create(
                         FusionBackendConfig::default_instance(),
                         HloFusionAdaptor::ForInstruction(root), &device_info));
  EXPECT_EQ(analysis.GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionEpilogueFusionPartiallyFused) {
  auto module = ParseAndReturnVerifiedModule(R"(
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
    })")
                    .value();

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();

  auto analysis =
      AnalyzeProducerConsumerFusion(*root->operand(0), *root, device_info);
  ASSERT_NE(analysis, std::nullopt);
  EXPECT_EQ(analysis->GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionEpilogueFusionPartiallyFusedInConsumer) {
  auto module = ParseAndReturnVerifiedModule(R"(
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
    })")
                    .value();

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis =
      AnalyzeProducerConsumerFusion(*root->operand(0), *root, device_info);
  ASSERT_NE(analysis, std::nullopt);
  EXPECT_EQ(analysis->GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, ReductionEpilogueFusionPartiallyFusedInBoth) {
  auto module = ParseAndReturnVerifiedModule(R"(
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
    })")
                    .value();

  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis =
      AnalyzeProducerConsumerFusion(*root->operand(0), *root, device_info);
  ASSERT_NE(analysis, std::nullopt);
  EXPECT_EQ(analysis->GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, InvalidDevice) {
  // Verifies that an analysis can be created even with an invalid/empty device
  // info, and that the emitter type is determined correctly.
  // Don't rely on this behavior.
  auto module = ParseAndReturnVerifiedModule(R"(
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
    })")
                    .value();

  stream_executor::GpuDeviceInfoProto device_info_proto;
  stream_executor::DeviceDescription device_info(device_info_proto);

  auto* root = module->entry_computation()->root_instruction();
  auto analysis_fused =
      AnalyzeProducerConsumerFusion(*root->operand(0), *root, device_info);
  ASSERT_NE(analysis_fused, std::nullopt);
  EXPECT_EQ(analysis_fused->GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);
}

TEST_F(HloFusionAnalysisTest, TritonSoftmaxFusion) {
#ifndef GOOGLE_CUDA
  GTEST_SKIP() << "Triton fusion only enable for CUDA devices.";
#endif

  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule t

    add {
      Arg_0 = f32[] parameter(0)
      Arg_1 = f32[] parameter(1)
      ROOT add = f32[] add(Arg_0, Arg_1)
    }

    auxiliary_computation {
      parameter_0 = f32[125]{0} parameter(0)
      ROOT broadcast = f32[125,127]{1,0} broadcast(parameter_0), dimensions={0}
    }

    triton_softmax_computation {
      parameter_0 = f32[125,127]{1,0} parameter(0)
      multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
      constant_0 = f32[] constant(0)
      reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
      broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
      ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
    }

    ENTRY main {
      param_0 = f32[125]{0} parameter(0)
      auxiliary_fusion = f32[125,127]{1,0} fusion(param_0), kind=kLoop, calls=auxiliary_computation
      ROOT triton_softmax = f32[125,127]{1,0} fusion(auxiliary_fusion), kind=kCustom, calls=triton_softmax_computation, backend_config={"kind":"__triton_softmax"}
      })")
                    .value();

  stream_executor::GpuDeviceInfoProto device_info_proto;
  stream_executor::DeviceDescription device_info(device_info_proto);

  auto* root = module->entry_computation()->root_instruction();
  auto analysis_fused =
      AnalyzeProducerConsumerFusion(*root->operand(0), *root, device_info);
  ASSERT_NE(analysis_fused, std::nullopt);
  EXPECT_EQ(analysis_fused->GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kTriton);

  TF_ASSERT_OK_AND_ASSIGN(auto launch_dimensions,
                          analysis_fused->GetLaunchDimensions());
  EXPECT_EQ(launch_dimensions.num_blocks(), 125);
  EXPECT_EQ(launch_dimensions.num_threads_per_block(), 32);

  auto analysis_consumer = AnalyzeFusion(*root, device_info);
  ASSERT_NE(analysis_consumer, std::nullopt);
  EXPECT_EQ(analysis_consumer->GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kTriton);
}

}  // namespace
}  // namespace xla::gpu
