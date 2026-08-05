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

#include "xla/backends/gpu/transforms/cudnn_fusion_compiler.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/substitute.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

class CudnnFusionCompilerConstTest
    : public HloPjRtGpuTestBase,
      public ::testing::WithParamInterface<PrimitiveType> {
 protected:
  se::StreamExecutor* stream_executor() const {
    auto platform =
        se::PlatformManager::PlatformWithId(stream_executor_platform_id());
    CHECK_OK(platform);
    absl::StatusOr<se::StreamExecutor*> executor =
        (*platform)->ExecutorForDevice(0);
    CHECK_OK(executor);
    return *executor;
  }
};

TEST_P(CudnnFusionCompilerConstTest,
       GetAvailablePlanCountFromCudnnConvolutionFusionWithConst) {
  PrimitiveType type = GetParam();
  std::string type_name = primitive_util::LowercasePrimitiveTypeName(type);

  std::string hlo_text = absl::Substitute(R"(
  fusion1 {
    p0 = $0[4,48,48,64] parameter(0)
    p1 = $0[64,3,3,64] parameter(1)
    conv = $0[4,48,48,64] convolution(p0, p1),
      window={size=3x3 pad=1_1x1_1},
      dim_labels=b01f_o01i->b01f,
      convolution_kind=fprop
    one = $0[] constant(1)
    ones = $0[4,48,48,64] broadcast(one), dimensions={}
    result = $0[4,48,48,64] add(conv, ones)
    zero = $0[] constant(0)
    zeros = $0[4,48,48,64] broadcast(zero), dimensions={}
    ROOT relu = $0[4,48,48,64] maximum(result, zeros)
  }

  ENTRY e {
    p0 = $0[4,48,48,64] parameter(0)
    p1 = $0[64,3,3,64] parameter(1)
    ROOT _ = $0[4,48,48,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
      backend_config={
        "fusion_backend_config": {
          "kind": "__cudnn$$fusion",
        }
      }
  })",
                                          type_name);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_text));

  const HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  auto* fusion = Cast<HloFusionInstruction>(root);

  ASSERT_OK_AND_ASSIGN(int plan_count,
                       CuDnnFusionCompiler::GetAvailablePlanCount(
                           stream_executor(),
                           stream_executor()->GetDeviceDescription(), *fusion));
  EXPECT_GT(plan_count, 0);
}

INSTANTIATE_TEST_SUITE_P(
    CudnnFusionCompilerConstTestTypes, CudnnFusionCompilerConstTest,
    ::testing::Values(F16, BF16, F32, S8),
    [](const ::testing::TestParamInfo<PrimitiveType>& info) {
      return std::string(
          primitive_util::LowercasePrimitiveTypeName(info.param));
    });

class CudnnFusionCompilerWgradTest
    : public HloPjRtGpuTestBase,
      public ::testing::WithParamInterface<PrimitiveType> {
 protected:
  se::StreamExecutor* stream_executor() const {
    auto platform =
        se::PlatformManager::PlatformWithId(stream_executor_platform_id());
    CHECK_OK(platform);
    absl::StatusOr<se::StreamExecutor*> executor =
        (*platform)->ExecutorForDevice(0);
    CHECK_OK(executor);
    return *executor;
  }
};

TEST_P(CudnnFusionCompilerWgradTest,
       GetAvailablePlanCountFromCudnnConvolutionFusionWgrad) {
  PrimitiveType type = GetParam();
  std::string type_name = primitive_util::LowercasePrimitiveTypeName(type);

  std::string hlo_text = absl::Substitute(R"(
  fusion1 {
    p0 = $0[2,1,6,9] parameter(0)
    p1 = $0[2,2,5,8] parameter(1)
    ROOT conv_wgrad = $0[2,1,4,4] convolution(p0, p1),
      window={size=5x8 pad=1_1x1_1},
      dim_labels=fb01_io01->fb01,
      convolution_kind=wgrad
  }

  ENTRY e {
    p0 = $0[2,1,6,9] parameter(0)
    p1 = $0[2,2,5,8] parameter(1)
    ROOT _ = $0[2,1,4,4] fusion(p0, p1), kind=kCustom, calls=fusion1,
      backend_config={
        "fusion_backend_config": {
          "kind": "__cudnn$$fusion",
        }
      }
  })",
                                          type_name);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_text));

  const HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  auto* fusion = Cast<HloFusionInstruction>(root);

  ASSERT_OK_AND_ASSIGN(int plan_count,
                       CuDnnFusionCompiler::GetAvailablePlanCount(
                           stream_executor(),
                           stream_executor()->GetDeviceDescription(), *fusion));
  EXPECT_GT(plan_count, 0);
}

INSTANTIATE_TEST_SUITE_P(
    CudnnFusionCompilerWgradTestTypes, CudnnFusionCompilerWgradTest,
    ::testing::Values(F16, BF16, F32, F64),
    [](const ::testing::TestParamInfo<PrimitiveType>& info) {
      return std::string(
          primitive_util::LowercasePrimitiveTypeName(info.param));
    });

TEST_F(CudnnFusionCompilerConstTest,
       GetAvailablePlanCountFrom0DConvolutionFusion) {
  std::string hlo_text = R"(
  fusion0d {
    p0 = f32[10,5] parameter(0)
    p1 = f32[7,5] parameter(1)
    ROOT conv = f32[10,7] convolution(p0, p1),
      window={},
      dim_labels=bf_oi->bf,
      convolution_kind=fprop
  }

  ENTRY e {
    p0 = f32[10,5] parameter(0)
    p1 = f32[7,5] parameter(1)
    ROOT _ = f32[10,7] fusion(p0, p1), kind=kCustom, calls=fusion0d,
      backend_config={
        "fusion_backend_config": {
          "kind": "__cudnn$$fusion",
        }
      }
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_text));

  const HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  auto* fusion = Cast<HloFusionInstruction>(root);

  ASSERT_OK_AND_ASSIGN(int plan_count,
                       CuDnnFusionCompiler::GetAvailablePlanCount(
                           stream_executor(),
                           stream_executor()->GetDeviceDescription(), *fusion));
  EXPECT_GT(plan_count, 0);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
