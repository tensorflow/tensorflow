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

#include "xla/service/gpu/priority_fusion.h"

#include <stdint.h>

#include <memory>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"

namespace m = ::xla::match;

namespace xla {
namespace gpu {

class PriorityFusionTest : public HloTestBase {
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [&](const Shape& shape) {
      constexpr int64_t kPointerSize = 8;
      return ShapeUtil::ByteSizeOf(shape, kPointerSize);
    };
  }

 public:
  GpuPriorityFusion priority_fusion_{
      TestGpuDeviceInfo::RTXA6000DeviceInfo(),
      GpuHloCostAnalysis::Options{ShapeSizeBytesFunction(),
                                  /*per_second_rates=*/{},
                                  /*count_multiple_input_accesses=*/true}};
};

TEST_F(PriorityFusionTest, FuseWithSharedArgument) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY main {
      %p0 = f32[] parameter(0)
      %p1 = f32[] parameter(1)
      %subtract = f32[] subtract(%p0, %p1)
      %compare = pred[] compare(%subtract, %subtract), direction=NE
      %add = f32[] add(%p0, %p1)
      %abs = f32[] abs(%subtract)
      ROOT %select = f32[] select(%compare, %add, %abs)
    })")
                    .value();

  EXPECT_TRUE(priority_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Fusion()));
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kLoop);
}

TEST_F(PriorityFusionTest, FusionFusionWithDuplication) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    square {
      p = f32[16384]{0} parameter(0)
      ROOT m = f32[16384]{0} multiply(p, p)
    }

    exp {
      p = f32[16384]{0} parameter(0)
      ROOT e = f32[16384]{0} exponential(p)
    }

    log {
      p = f32[16384]{0} parameter(0)
      ROOT l = f32[16384]{0} log(p)
    }

    ENTRY main {
      p = f32[16384]{0} parameter(0)
      s = f32[16384]{0} fusion(p), kind=kLoop, calls=square
      e = f32[16384]{0} fusion(s), kind=kLoop, calls=exp
      l = f32[16384]{0} fusion(s), kind=kInput, calls=log
      ROOT t = (f32[16384], f32[16384]) tuple(l, e)
    })")
                    .value();

  EXPECT_TRUE(priority_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Tuple(m::Fusion(m::Parameter(0)),
                                        m::Fusion(m::Parameter(0)))));
}

TEST_F(PriorityFusionTest, FuseWideningConvertIntoConsumers) {
  // Because of the bitcast consumer, the convert is currently fused only with
  // the log, resulting in three fusions:
  //   1. log + convert
  //   2. multiply
  //   3. bitcast
  // This is a bug.
  absl::string_view kHlo = R"(
    HloModule test_module

    ENTRY main {
      p = f16[512]{0} parameter(0)
      l = f16[512]{0} log(p)
      c = f32[512]{0} convert(l)
      s = f32[512]{0} multiply(c, c)
      bc = s32[512]{0} bitcast(c)
      ROOT t = (f32[512], s32[512]) tuple(s, bc)
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK:      ENTRY
CHECK-NEXT: %[[PARAM:.*]] = f16[512]{0} parameter(0)
CHECK-NEXT: %[[FUSION:.*]] = f32[512]{0} fusion(%[[PARAM]])
CHECK-NEXT: %[[MUL:.*]] = f32[512]{0} multiply(%[[FUSION]], %[[FUSION]])
CHECK-NEXT: %[[BITCAST:.*]] = s32[512]{0} bitcast(%[[FUSION]])
CHECK-NEXT: ROOT %{{.*}} = (f32[512]{0}, s32[512]{0}) tuple(%[[MUL]], %[[BITCAST]])
  )");
}

}  // namespace gpu
}  // namespace xla
