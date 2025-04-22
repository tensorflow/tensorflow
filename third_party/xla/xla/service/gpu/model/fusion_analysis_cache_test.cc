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

#include "xla/service/gpu/model/fusion_analysis_cache.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

class FusionAnalysisCacheTest : public HloHardwareIndependentTestBase {
 public:
  stream_executor::DeviceDescription device_{
      TestGpuDeviceInfo::RTXA6000DeviceInfo()};
  HloFusionAnalysisCache cache_{device_};
};

TEST_F(FusionAnalysisCacheTest, CachesAndInvalidates) {
  absl::string_view hlo_string = R"(
    HloModule m

    f {
      c0 = f32[] constant(0)
      b0 = f32[1000] broadcast(c0)
      ROOT n0 = f32[1000] negate(b0)
    }

    ENTRY e {
      ROOT r.1 = f32[1000] fusion(), kind=kLoop, calls=f
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto* computation = module->GetComputationWithName("f");
  auto* broadcast = computation->GetInstructionWithName("b0");
  auto* negate = computation->GetInstructionWithName("n0");
  auto* fusion = module->entry_computation()->root_instruction();

  EXPECT_EQ(&cache_.Get(*fusion).fusion_root(0).instruction(), negate);

  computation->set_root_instruction(broadcast);

  EXPECT_EQ(&cache_.Get(*fusion).fusion_root(0).instruction(), negate)
      << "Analysis should be cached.";

  cache_.Invalidate(*fusion);
  EXPECT_EQ(&cache_.Get(*fusion).fusion_root(0).instruction(), broadcast)
      << "Analysis should have been recomputed";
}

TEST_F(FusionAnalysisCacheTest, CachesAndInvalidatesProducerConsumerFusions) {
  absl::string_view hlo_string = R"(
    HloModule m

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    f {
      c0 = f32[] constant(0)
      b0 = f32[1000] broadcast(c0)
      ROOT r0 = f32[] reduce(b0, c0), dimensions={0}, to_apply=add
    }

    ENTRY e {
      f0 = f32[] fusion(), kind=kInput, calls=f
      ROOT n0 = f32[] negate(f0)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto* fusion = module->entry_computation()->GetInstructionWithName("f0");
  auto* neg = module->entry_computation()->GetInstructionWithName("n0");

  auto* computation = module->GetComputationWithName("f");
  auto* constant = computation->GetInstructionWithName("c0");

  EXPECT_EQ(cache_.Get(*fusion, *neg).GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction);

  computation->set_root_instruction(constant);

  EXPECT_EQ(cache_.Get(*fusion, *neg).GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kReduction)
      << "Analysis should be cached.";

  cache_.Invalidate(*fusion);
  EXPECT_EQ(cache_.Get(*fusion, *neg).GetEmitterFusionKind(),
            HloFusionAnalysis::EmitterFusionKind::kLoop)
      << "Analysis should have been recomputed";
}

}  // namespace
}  // namespace xla::gpu
