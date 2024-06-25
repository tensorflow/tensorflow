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

#include "xla/service/gpu/priority_fusion.h"

#include <stdint.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace m = ::xla::match;

using ::testing::UnorderedElementsAre;
using ::tsl::testing::IsOk;
using ::tsl::testing::IsOkAndHolds;

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
  std::vector<HloFusionAnalysis::EmitterFusionKind> RunAndGetFusionKinds(
      absl::string_view hlo) {
    auto module = ParseAndReturnVerifiedModule(hlo).value();
    EXPECT_THAT(priority_fusion_.Run(module.get()), IsOkAndHolds(true));
    EXPECT_THAT(module->RemoveUnusedComputations(), IsOk());
    std::vector<HloFusionAnalysis::EmitterFusionKind> kinds;
    for (auto computation : module->computations()) {
      if (!computation->FusionInstruction()) continue;

      auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();
      auto analysis = HloFusionAnalysis::Create(
          Cast<HloFusionInstruction>(computation->FusionInstruction()),
          &device_info);
      kinds.push_back(analysis.GetEmitterFusionKind());
    }
    return kinds;
  }

  GpuPriorityFusion priority_fusion_{
      /*thread_pool=*/nullptr, TestGpuDeviceInfo::RTXA6000DeviceInfo(),
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

  EXPECT_THAT(priority_fusion_.Run(module.get()), IsOkAndHolds(true));

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Fusion()));
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kLoop);
}

TEST_F(PriorityFusionTest, FusionFusionWithDuplication) {
  absl::string_view kHlo = R"(
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
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK: ENTRY
CHECK-NEXT: %[[PARAM:.*]] = f32[16384]{0} parameter(0)
CHECK-NEXT: %[[FUSION_0:.*]] = f32[16384]{0} fusion(%[[PARAM]])
CHECK-NEXT: %[[FUSION_1:.*]] = f32[16384]{0} fusion(%[[PARAM]])
CHECK-NEXT: ROOT {{.*}} tuple(%[[FUSION_0]], %[[FUSION_1]])
  )");
}

TEST_F(PriorityFusionTest, FuseBroadcastIntoBitcastConsumers) {
  absl::string_view kHlo = R"(
    HloModule test_module

    ENTRY main {
      param_0 = f32[96]{0} parameter(0)
      broadcast = f32[8,96,128,7]{3,2,1,0} broadcast(param_0), dimensions={1}
      bitcast.6079.2 = f32[8,24,4,128,7]{4,3,2,1,0} bitcast(broadcast)
      ROOT transpose.1990.2 = f32[8,24,128,7,4]{4,3,2,1,0} transpose(bitcast.6079.2), dimensions={0,1,3,4,2}
    }
  )";
  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK:      ENTRY
CHECK-NEXT: %[[PARAM:.*]] = f32[96]{0} parameter(0)
CHECK-NEXT: ROOT %{{.*}} fusion(%[[PARAM]])
  )");
}

TEST_F(PriorityFusionTest, FuseWideningConvertIntoConsumers) {
  absl::string_view kHlo = R"(
    HloModule test_module

    ENTRY main {
      p = f16[512]{0} parameter(0)
      a = f16[512]{0} add(p, p)
      c = f32[512]{0} convert(a)
      s = f32[512]{0} multiply(c, c)
      bc = s32[512]{0} bitcast(c)
      ROOT t = (f32[512], s32[512]) tuple(s, bc)
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK:      ENTRY
CHECK-NEXT: %[[PARAM:.*]] = f16[512]{0} parameter(0)
CHECK-NEXT: %[[FUSION_F32:.*]] = f32[512]{0} fusion(%[[PARAM]])
CHECK-NEXT: %[[CONVERT_FUSION:.*]] = f32[512]{0} fusion(%[[PARAM]])
CHECK-NEXT: %[[BITCAST:.*]] = s32[512]{0} bitcast(%[[CONVERT_FUSION]])
CHECK-NEXT: ROOT %{{.*}} = (f32[512]{0}, s32[512]{0}) tuple(%[[FUSION_F32]], %[[BITCAST]])
  )");
}

TEST_F(PriorityFusionTest, FuseConvertIntoReduce) {
  absl::string_view kHlo = R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add.13235 = f32[] add(p0, p1)
    }

    ENTRY main {
      param_0_0.79 = bf16[1024,8192]{1,0} parameter(0)
      param_1_0.79 = bf16[1024,8192]{1,0} parameter(1)
      param_2.483 = f32[8192]{0} parameter(2)
      param_4.2892 = bf16[1024,8192]{1,0} parameter(3)
      convert.21854 = f32[1024,8192]{1,0} convert(param_0_0.79)
      convert.21855 = f32[1024,8192]{1,0} convert(param_1_0.79)
      constant_7773 = f32[] constant(0)
      broadcast.14555 = f32[1024,8192]{1,0} broadcast(param_2.483), dimensions={1}
      multiply.6906 = f32[1024,8192]{1,0} multiply(broadcast.14555, convert.21854)
      reduce.4813 = f32[1024]{0} reduce(multiply.6906, constant_7773), dimensions={1}, to_apply=add
      convert.13970 = bf16[1024]{0} convert(reduce.4813)
      convert.21534 = f32[1024,8192]{1,0} convert(param_4.2892)
      multiply.6910.clone.1 = f32[1024,8192]{1,0} multiply(broadcast.14555, convert.21534)
      reduce.4811.clone.1 = f32[1024]{0} reduce(multiply.6910.clone.1, constant_7773), dimensions={1}, to_apply=add
      convert.13967.clone.1 = bf16[1024]{0} convert(reduce.4811.clone.1)
      multiply.6908.clone.1 = f32[1024,8192]{1,0} multiply(broadcast.14555, convert.21855)
      reduce.4812.clone.1 = f32[1024]{0} reduce(multiply.6908.clone.1, constant_7773), dimensions={1}, to_apply=add
      convert.13969.clone.1 = bf16[1024]{0} convert(reduce.4812.clone.1)
      ROOT fusion.241 = (bf16[1024]{0}, bf16[1024]{0}, bf16[1024]{0}) tuple(convert.13970, convert.13967.clone.1, convert.13969.clone.1)
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK-COUNT-3: ROOT {{.*}} convert(
CHECK: ENTRY %main
CHECK-COUNT-3: fusion
  )");
}

TEST_F(PriorityFusionTest, ReductionEpilogueFusionRegressionTest) {
  // Regression test for epilogue fusion of convert into a reduction, even if
  // the convert has a bitcast as consumer.
  absl::string_view kHlo = R"(
    HloModule test_module

    add {
      rhs.407 = f32[] parameter(1)
      lhs.407 = f32[] parameter(0)
      ROOT add.24451 = f32[] add(lhs.407, rhs.407)
    }

    ENTRY main {
      param_1.15162 = f32[2752]{0} parameter(1)
      convert.44829 = bf16[2752]{0} convert(param_1.15162)
      bitcast.24686 = bf16[1,1,2752]{2,1,0} bitcast(convert.44829)
      convert.44468 = f32[1,1,2752]{2,1,0} convert(bitcast.24686)
      constant_13722 = bf16[] constant(1)
      convert.17451 = f32[] convert(constant_13722)
      broadcast.17565 = f32[1,1,2752]{2,1,0} broadcast(convert.17451), dimensions={}
      negate.167 = f32[1,1,2752]{2,1,0} negate(convert.44468)
      exponential.569 = f32[1,1,2752]{2,1,0} exponential(negate.167)
      add.1850 = f32[1,1,2752]{2,1,0} add(broadcast.17565, exponential.569)
      divide.1376 = f32[1,1,2752]{2,1,0} divide(broadcast.17565, add.1850)
      multiply.9709 = f32[1,1,2752]{2,1,0} multiply(convert.44468, divide.1376)
      param_0.15005 = f32[2752]{0} parameter(0)
      convert.44826 = bf16[2752]{0} convert(param_0.15005)
      bitcast.24683 = bf16[1,1,2752]{2,1,0} bitcast(convert.44826)
      convert.44467 = f32[1,1,2752]{2,1,0} convert(bitcast.24683)
      multiply.9708 = f32[1,1,2752]{2,1,0} multiply(multiply.9709, convert.44467)
      convert.16959 = bf16[1,1,2752]{2,1,0} convert(multiply.9708)
      fusion.3203 = bf16[2752]{0} bitcast(convert.16959)
      convert.15093 = f32[2752]{0} convert(fusion.3203)
      broadcast.13841 = f32[8192,2752]{1,0} broadcast(convert.15093), dimensions={1}
      param_0.15525 = bf16[8192,2752]{1,0} parameter(2)
      convert.13738 = f32[8192,2752]{1,0} convert(param_0.15525)
      multiply.6422 = f32[8192,2752]{1,0} multiply(broadcast.13841, convert.13738)
      constant_14382 = f32[] constant(0)
      fusion.339 = f32[8192]{0} reduce(multiply.6422, constant_14382), dimensions={1}, to_apply=add
      convert.44633 = bf16[8192]{0} convert(fusion.339)
      ROOT bitcast.24487 = bf16[1,1,8192]{2,1,0} bitcast(convert.44633)
    }
                                                     )";

  EXPECT_THAT(
      RunAndGetFusionKinds(kHlo),
      UnorderedElementsAre(HloFusionAnalysis::EmitterFusionKind::kLoop,
                           HloFusionAnalysis::EmitterFusionKind::kReduction));

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK: ENTRY
CHECK: ROOT {{.*}} bitcast({{.*}}fusion{{.*}})
  )");
}

TEST_F(PriorityFusionTest, DoNotChangeReductionFusionToLoopFusion) {
  // Regression test for epilogue fusion of slice into a reduction. The fusion
  // kind for the reduction fusion is intentionally chosen to be set to kLoop,
  // as we cannot rely on reductions always having fusion kind kInput.
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      rhs.407 = f32[] parameter(1)
      lhs.407 = f32[] parameter(0)
      ROOT add.24451 = f32[] add(lhs.407, rhs.407)
    }

    fused_computation {
      p0 = f32[16,64]{1,0} parameter(0)
      zero = f32[] constant(0.0)
      ROOT reduce = f32[16]{0} reduce(p0, zero), dimensions={1}, to_apply=add
    }

    ENTRY main {
      param0 = f32[16,64]{1,0} parameter(0)
      fusion = f32[16]{0} fusion(param0), kind=kLoop, calls=fused_computation
      ROOT slice = f32[8]{0} slice(fusion), slice={[0:8]}
    })");
  EXPECT_THAT(priority_fusion_.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(PriorityFusionTest, DoNotFuseTransposeIntoReduce) {
  absl::string_view kHlo = R"(
    HloModule test_module

    add {
      Arg_1.1046 = f32[] parameter(1)
      Arg_0.1045 = f32[] parameter(0)
      ROOT add.3303 = f32[] add(Arg_0.1045, Arg_1.1046)
    }

    ENTRY main {
      param_0.17323 = pred[2048,2048]{1,0} parameter(0)
      broadcast.22829 = pred[1,12,2048,2048]{3,2,1,0} broadcast(param_0.17323), dimensions={2,3}
      param_1.19761 = bf16[2048,24576]{1,0} parameter(1)
      convert.29880.clone.1 = f32[2048,24576]{1,0} convert(param_1.19761)
      constant_10033_clone_1 = bf16[] constant(0.02002)
      convert.30056.clone.1 = f32[] convert(constant_10033_clone_1)
      broadcast.18898.clone.1 = f32[2048,24576]{1,0} broadcast(convert.30056.clone.1), dimensions={}
      multiply.13451.clone.1 = f32[2048,24576]{1,0} multiply(convert.29880.clone.1, broadcast.18898.clone.1)
      tanh.798.clone.1 = f32[2048,24576]{1,0} tanh(multiply.13451.clone.1)
      constant_10244_clone_1 = bf16[] constant(50)
      convert.30039.clone.1 = f32[] convert(constant_10244_clone_1)
      broadcast.18310.clone.1 = f32[2048,24576]{1,0} broadcast(convert.30039.clone.1), dimensions={}
      multiply.12550.clone.1 = f32[2048,24576]{1,0} multiply(tanh.798.clone.1, broadcast.18310.clone.1)
      convert.29370.clone.1 = bf16[2048,24576]{1,0} convert(multiply.12550.clone.1)
      bitcast.22330 = bf16[1,2048,2048,12]{3,2,1,0} bitcast(convert.29370.clone.1)
      transpose.6582 = bf16[1,12,2048,2048]{3,2,1,0} transpose(bitcast.22330), dimensions={0,3,2,1}
      convert.33705 = f32[1,12,2048,2048]{3,2,1,0} convert(transpose.6582)
      constant_10212 = f32[] constant(-2.38197633e+38)
      broadcast.22828 = f32[1,12,2048,2048]{3,2,1,0} broadcast(constant_10212), dimensions={}
      select.589 = f32[1,12,2048,2048]{3,2,1,0} select(broadcast.22829, convert.33705, broadcast.22828)
      bitcast.22075 = f32[12,2048,2048]{2,1,0} bitcast(select.589)
      constant_10192 = f32[] constant(-inf)
      reduce.1614 = f32[12,2048]{1,0} reduce(bitcast.22075, constant_10192), dimensions={2}, to_apply=add

      predarg = pred[1,1,2048,2048]{3,2,1,0} parameter(2)
      bitcast.11069 = pred[2048,2048]{1,0} bitcast(predarg)

      broadcast.22825 = pred[1,12,2048,2048]{3,2,1,0} broadcast(bitcast.11069), dimensions={2,3}
      bitcast.22331 = bf16[1,2048,2048,12]{3,2,1,0} bitcast(convert.29370.clone.1)
      transpose.6580 = bf16[1,12,2048,2048]{3,2,1,0} transpose(bitcast.22331), dimensions={0,3,2,1}
      convert.33703 = f32[1,12,2048,2048]{3,2,1,0} convert(transpose.6580)
      constant_10213 = f32[] constant(-2.38197633e+38)
      broadcast.22824 = f32[1,12,2048,2048]{3,2,1,0} broadcast(constant_10213), dimensions={}
      select.587 = f32[1,12,2048,2048]{3,2,1,0} select(broadcast.22825, convert.33703, broadcast.22824)
      broadcast.22819 = f32[1,12,2048,2048]{3,2,1,0} broadcast(reduce.1614), dimensions={1,2}
      subtract.1129 = f32[1,12,2048,2048]{3,2,1,0} subtract(select.587, broadcast.22819)
      exponential.418 = f32[1,12,2048,2048]{3,2,1,0} exponential(subtract.1129)
      bitcast.22074 = f32[12,2048,2048]{2,1,0} bitcast(exponential.418)
      constant_10490 = f32[] constant(0)
      reduce.1613 = f32[12,2048]{1,0} reduce(bitcast.22074, constant_10490), dimensions={2}, to_apply=add

      constant_468 = f32[] constant(-2.38197633e+38)
      broadcast.22833 = pred[1,12,2048,2048]{3,2,1,0} broadcast(bitcast.11069), dimensions={2,3}
      bitcast.22332 = bf16[1,2048,2048,12]{3,2,1,0} bitcast(convert.29370.clone.1)
      transpose.6584 = bf16[1,12,2048,2048]{3,2,1,0} transpose(bitcast.22332), dimensions={0,3,2,1}
      convert.33707 = f32[1,12,2048,2048]{3,2,1,0} convert(transpose.6584)
      broadcast.22832 = f32[1,12,2048,2048]{3,2,1,0} broadcast(constant_468), dimensions={}
      select.591 = f32[1,12,2048,2048]{3,2,1,0} select(broadcast.22833, convert.33707, broadcast.22832)
      broadcast.22821 = f32[1,12,2048,2048]{3,2,1,0} broadcast(reduce.1614), dimensions={1,2}
      subtract.1131 = f32[1,12,2048,2048]{3,2,1,0} subtract(select.591, broadcast.22821)
      exponential.420 = f32[1,12,2048,2048]{3,2,1,0} exponential(subtract.1131)
      broadcast.18351 = f32[1,12,2048,2048]{3,2,1,0} broadcast(reduce.1613), dimensions={1,2}
      divide.340 = f32[1,12,2048,2048]{3,2,1,0} divide(exponential.420, broadcast.18351)
      ROOT convert.29418 = bf16[1,12,2048,2048]{3,2,1,0} convert(divide.340)
    })";

  using Kind = HloFusionAnalysis::EmitterFusionKind;
  EXPECT_THAT(
      RunAndGetFusionKinds(kHlo),
      UnorderedElementsAre(Kind::kLoop, Kind::kLoop, Kind::kLoop,
                           Kind::kReduction, Kind::kReduction, Kind::kTranspose,
                           Kind::kTranspose, Kind::kTranspose));
}

TEST_F(PriorityFusionTest, DoNotFuseReduceIntoReduce) {
  absl::string_view kHlo = R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add.13235 = f32[] add(p0, p1)
    }

    ENTRY main {
      p0 = f32[8,4,128,226]{3,2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r0 = f32[8,4,128]{2,1,0} reduce(p0, c0), dimensions={3}, to_apply=add
      ROOT r1 = f32[8,4]{1,0} reduce(r0, c0), dimensions={2}, to_apply=add
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK: ROOT {{.*}} reduce(
CHECK: ROOT {{.*}} reduce(
  )");
}

TEST_F(PriorityFusionTest, ConvertFusedIntoReduce) {
  absl::string_view kHlo = R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add.13235 = f32[] add(p0, p1)
    }

    ENTRY main {
      param_0_0.79 = bf16[1024,8192]{1,0} parameter(0)
      param_1_0.79 = bf16[1024,8192]{1,0} parameter(1)
      param_2.483 = f32[8192]{0} parameter(2)
      param_4.2892 = bf16[1024,8192]{1,0} parameter(3)
      convert.21854 = f32[1024,8192]{1,0} convert(param_0_0.79)
      convert.21855 = f32[1024,8192]{1,0} convert(param_1_0.79)
      constant_7773 = f32[] constant(0)
      broadcast.14555 = f32[1024,8192]{1,0} broadcast(param_2.483), dimensions={1}
      multiply.6906 = f32[1024,8192]{1,0} multiply(broadcast.14555, convert.21854)
      reduce.4813 = f32[1024]{0} reduce(multiply.6906, constant_7773), dimensions={1}, to_apply=add
      convert.13970 = bf16[1024]{0} convert(reduce.4813)
      convert.21534 = f32[1024,8192]{1,0} convert(param_4.2892)
      multiply.6910.clone.1 = f32[1024,8192]{1,0} multiply(broadcast.14555, convert.21534)
      reduce.4811.clone.1 = f32[1024]{0} reduce(multiply.6910.clone.1, constant_7773), dimensions={1}, to_apply=add
      convert.13967.clone.1 = bf16[1024]{0} convert(reduce.4811.clone.1)
      multiply.6908.clone.1 = f32[1024,8192]{1,0} multiply(broadcast.14555, convert.21855)
      reduce.4812.clone.1 = f32[1024]{0} reduce(multiply.6908.clone.1, constant_7773), dimensions={1}, to_apply=add
      convert.13969.clone.1 = bf16[1024]{0} convert(reduce.4812.clone.1)
      ROOT fusion.241 = (bf16[1024]{0}, bf16[1024]{0}, bf16[1024]{0}) tuple(convert.13970, convert.13967.clone.1, convert.13969.clone.1)
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK-COUNT-3: ROOT {{.*}} convert(
CHECK: ENTRY %main
CHECK-COUNT-3: fusion(
CHECK-NOT: fusion(
  )");
}

TEST_F(PriorityFusionTest, DoNotFuseDynamicUpdateSliceIntoReduce) {
  GTEST_SKIP() << "b/294198633";
  absl::string_view kHlo = R"(
    HloModule test_module

add {
  Arg_1.1046 = f32[] parameter(1)
  Arg_0.1045 = f32[] parameter(0)
  ROOT add.3303 = f32[] add(Arg_0.1045, Arg_1.1046)
}

ENTRY main {
  param_0.10549 = f32[4,2112]{1,0} parameter(0)
  param_5.2561 = pred[] parameter(5)
  broadcast.19725 = pred[4,1]{1,0} broadcast(param_5.2561), dimensions={}
  param_1.11587 = pred[4]{0} parameter(1)
  constant_5837 = f32[] constant(1)
  broadcast.19723 = f32[4]{0} broadcast(constant_5837), dimensions={}
  param_2.5952 = f32[4,8000]{1,0} parameter(2)
  param_3.4004 = f32[4]{0} parameter(3)
  broadcast.19718 = f32[4,8000]{1,0} broadcast(param_3.4004), dimensions={0}
  subtract.1112 = f32[4,8000]{1,0} subtract(param_2.5952, broadcast.19718)
  exponential.418 = f32[4,8000]{1,0} exponential(subtract.1112)
  constant_6254 = f32[] constant(0)
  reduce.1154 = f32[4]{0} reduce(exponential.418, constant_6254), dimensions={1}, to_apply=add
  log.38 = f32[4]{0} log(reduce.1154)
  broadcast.19717 = f32[4,8000]{1,0} broadcast(log.38), dimensions={0}
  subtract.1111 = f32[4,8000]{1,0} subtract(subtract.1112, broadcast.19717)
  iota.170 = s32[4,1]{1,0} iota(), iota_dimension=0
  constant_6281 = s32[] constant(0)
  broadcast.19735 = s32[4]{0} broadcast(constant_6281), dimensions={}
  param_4.3400 = s32[4,8000]{1,0} parameter(4)
  slice.3186 = s32[4,40]{1,0} slice(param_4.3400), slice={[0:4], [0:40]}
  iota.168 = s32[4,1]{1,0} iota(), iota_dimension=0
  param_7.1596 = s32[4]{0} parameter(7)
  compare.341 = pred[4]{0} compare(param_7.1596, broadcast.19735), direction=LT
  constant_5833 = s32[] constant(40)
  broadcast.19731 = s32[4]{0} broadcast(constant_5833), dimensions={}
  add.8348 = s32[4]{0} add(param_7.1596, broadcast.19731)
  select.418 = s32[4]{0} select(compare.341, add.8348, param_7.1596)
  bitcast.20942 = s32[4,1]{1,0} bitcast(select.418)
  concatenate.1337 = s32[4,2]{1,0} concatenate(iota.168, bitcast.20942), dimensions={1}
  gather.43 = s32[4,1,1]{2,1,0} gather(slice.3186, concatenate.1337), offset_dims={1,2}, collapsed_slice_dims={}, start_index_map={0,1}, index_vector_dim=1, slice_sizes={1,1}
  bitcast.20941 = s32[4]{0} bitcast(gather.43)
  select.398 = s32[4]{0} select(param_1.11587, broadcast.19735, bitcast.20941)
  compare.334 = pred[4]{0} compare(select.398, broadcast.19735), direction=LT
  constant_6260 = s32[] constant(8000)
  broadcast.19720 = s32[4]{0} broadcast(constant_6260), dimensions={}
  add.8336 = s32[4]{0} add(select.398, broadcast.19720)
  select.396 = s32[4]{0} select(compare.334, add.8336, select.398)
  bitcast.20830 = s32[4,1]{1,0} bitcast(select.396)
  concatenate.1308 = s32[4,2]{1,0} concatenate(iota.170, bitcast.20830), dimensions={1}
  gather.41 = f32[4,1,1]{2,1,0} gather(subtract.1111, concatenate.1308), offset_dims={1,2}, collapsed_slice_dims={}, start_index_map={0,1}, index_vector_dim=1, slice_sizes={1,1}
  bitcast.20824 = f32[4]{0} bitcast(gather.41)
  select.389 = f32[4]{0} select(param_1.11587, broadcast.19723, bitcast.20824)
  bitcast.20823 = f32[4,1]{1,0} bitcast(select.389)
  param_6.1719 = s32[] parameter(6)
  constant_6323 = s32[] constant(2048)
  add.8549 = s32[] add(param_6.1719, constant_6323)
  compare.388 = pred[] compare(add.8549, constant_6281), direction=LT
  constant_5436 = s32[] constant(4160)
  add.8339 = s32[] add(param_6.1719, constant_5436)
  select.409 = s32[] select(compare.388, add.8339, add.8549)
  dynamic-slice.36 = f32[4,1]{1,0} dynamic-slice(param_0.10549, constant_6281, select.409), dynamic_slice_sizes={4,1}
  select.388 = f32[4,1]{1,0} select(broadcast.19725, bitcast.20823, dynamic-slice.36)
  ROOT dynamic-update-slice.307 = f32[4,2112]{1,0} dynamic-update-slice(param_0.10549, select.388, constant_6281, select.409)
})";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK: ROOT {{.*}} dynamic-update-slice(
CHECK: %[[REDUCE:.*]] = {{.*}} reduce(
CHECK: ROOT {{.*}} log(%[[REDUCE]])
CHECK: ENTRY
CHECK-COUNT-2: fusion(
  )");
}

TEST_F(PriorityFusionTest, DontFuseIntoFirstOperandOfScatter) {
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY FuseIntoScatter {
      p0 = s32[3,3] parameter(0)
      operand = s32[3,3] add(p0, p0)
      p1 = s32[2] parameter(1)
      indices = s32[2] add(p1, p1)
      p2 = s32[2,3] parameter(2)
      updates = s32[2,3] add(p2, p2)
      scatter = s32[3,3] scatter(operand, indices, updates),
          to_apply=add,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
      ROOT add = s32[3,3] add(scatter, scatter)
    })");

  EXPECT_THAT(priority_fusion_.Run(module.get()), IsOkAndHolds(true));

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Add(m::Fusion(&fusion), m::Fusion())));
  EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kInput);
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Scatter(m::Parameter(), m::Add(), m::Add())));
}

// This test is similar to DontFuseIntoFirstOperandOfScatter, but PriorityFusion
// has a separate run to fuse constants. Fusing anything into a scatter fusion
// will fail in the emitter.
TEST_F(PriorityFusionTest, DontFuseConstantIntoFirstOperandOfScatter) {
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY FuseIntoScatter {
      operand = s32[1] constant({0})
      indices = s32[24,1] parameter(0)
      constant = s32[] constant(1)
      updates = s32[24,1] broadcast(constant)
      ROOT scatter = s32[1] scatter(operand, indices, updates),
          to_apply=add,
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    })");

  EXPECT_THAT(priority_fusion_.Run(module.get()), IsOkAndHolds(true));

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::Fusion(m::Constant(), m::Parameter())));
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kInput);
  EXPECT_THAT(root->fused_expression_root(),
              GmockMatch(m::Scatter(m::Parameter(), m::Parameter(),
                                    m::Broadcast(m::Constant()))));
}

TEST_F(PriorityFusionTest, DoNotFuseReduceIntoReduceEvenIfOccupancyIsHigh) {
  constexpr absl::string_view kHlo = R"(
    HloModule test_module

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      p0 = f32[4,3584,128,168]{3,2,1,0} parameter(0)
      c = f32[] constant(0)
      r1 = f32[4,3584,128]{2,1,0} reduce(p0, c), dimensions={3}, to_apply=add
      ROOT r2 = f32[4,3584]{1,0} reduce(r1, c), dimensions={2}, to_apply=add
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK: ROOT {{.*}} reduce(
CHECK: ROOT {{.*}} reduce(
  )");
}

TEST_F(PriorityFusionTest, FuseReductionEpilogueWithMultipleUsers) {
  // Regression test that verifies we correctly fuse the `log` into the reduce.
  constexpr absl::string_view kHlo = R"(
    HloModule test_module

    add {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT add = f32[] add(x, y)
    }

    fused_computation {
      p0 = f32[64,16384]{1,0} parameter(0)
      c0 = f32[] constant(0)
      ROOT reduce.858 = f32[64]{0} reduce(p0, c0), dimensions={1}, to_apply=add
    }

    ENTRY main {
      p0 = f32[64,16384]{1,0} parameter(0)
      fusion = f32[64]{0} fusion(p0), kind=kInput, calls=fused_computation
      log = f32[64]{0} log(fusion)
      negate = f32[64]{0} custom-call(log), custom_call_target="negate"
      ROOT add = f32[64]{0} add(negate, log)
    }
  )";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
    CHECK: ENTRY
    CHECK: %[[PARAM:.*]] = {{.*}} parameter(0)
    CHECK: %[[FUSION:.*]] = {{.*}} fusion(%[[PARAM]])
    CHECK: custom-call(%[[FUSION]])
  )");
}

TEST_F(PriorityFusionTest, EpilogueFusion) {
  absl::string_view kHlo = R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add.13235 = f32[] add(p0, p1)
    }

    fused_computation.1 {
      p0 = f32[8,4,128,226]{3,2,1,0} parameter(0)
      c0 = f32[] constant(0)
      ROOT r0 = f32[8,4,128]{2,1,0} reduce(p0, c0), dimensions={3}, to_apply=add
    }

    fused_computation.2 {
      p0 = f32[8,4,128]{2,1,0} parameter(0)
      r1 = f32[8,4,128]{2,1,0} log(p0)
      ROOT r2 = f32[8,4,128]{2,1,0} log(r1)
    }

    ENTRY main {
      p0 = f32[8,4,128,226]{3,2,1,0} parameter(0)
      f1 = f32[8,4,128]{2,1,0} fusion(p0), kind=kInput, calls=%fused_computation.1
      ROOT fusion = f32[8,4,128]{2,1,0} fusion(f1), kind=kLoop, calls=%fused_computation.2
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK: ROOT {{.*}} = f32[8,4,128]{2,1,0} fusion(%p{{.*}}), kind=kInput, calls=%fused_computation)");
}

TEST_F(PriorityFusionTest, EpilogueFusionFails) {
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add.13235 = f32[] add(p0, p1)
    }

    fused_computation.1 {
      p0 = f32[28672,4096]{1,0} parameter(0)
      c0 = f32[] constant(0)
      ROOT r = f32[28672]{0} reduce(p0, c0), dimensions={1}, to_apply=add
    }

    fused_computation.2 {
      p0 = f32[28672]{0} parameter(0)
      p1 = f32[28672]{0} parameter(1)
      ROOT a = f32[28672]{0} add(p0, p1)
    }

    ENTRY main {
      p0 = f32[28672,4096]{1,0} parameter(0)
      p1 = f32[28672]{0} parameter(1)
      f = f32[28672]{0} fusion(p0), kind=kInput, calls=%fused_computation.1
      ROOT fusion = f32[28672]{0} fusion(f,p1), kind=kLoop, calls=%fused_computation.2
    })");

  EXPECT_THAT(priority_fusion_.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(PriorityFusionTest, DoNotFuseIntoRoot) {
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY %main (p.0: u32[2], p.1: u32[]) -> u32[2] {
      %p.0 = u32[2]{0} parameter(0)
      %p.1 = u32[] parameter(1)
      ROOT %broadcast = u32[2]{0} broadcast(u32[] %p.1), dimensions={}, sharding={replicated}
      %add = u32[2]{0} add(u32[2]{0} %p.0, u32[2]{0} %broadcast)
      %tuple.1 = (u32[2]{0}) tuple(u32[2]{0} %add)
      %token.0 = token[] after-all()
      %outfeed.6 = token[] outfeed((u32[2]{0}) %tuple.1, token[] %token.0), outfeed_shape=(u32[2]{0}), sharding={maximal device=0}
    })");

  EXPECT_THAT(priority_fusion_.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(PriorityFusionTest, DontFuseConcat) {
  // Regression test that verifies we don't fuse concat into a column reduction.
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule module

    %maximum (param_0: f32[], param_1: f32[]) -> f32[] {
      %param_0 = f32[] parameter(0)
      %param_1 = f32[] parameter(1)
      ROOT %maximum = f32[] maximum(f32[] %param_0, f32[] %param_1)
    }

    %fused_concat (param_0: f32[1,4,401,8,8], param_1: f32[1,1,4,1023,8], param_2: bf16[1,4,1023,8,8]) -> f32[1,4,1424,8,8] {
      %param_2 = bf16[1,4,1023,8,8]{4,3,2,1,0} parameter(2)
      %convert = f32[1,4,1023,8,8]{4,3,2,1,0} convert(bf16[1,4,1023,8,8]{4,3,2,1,0} %param_2)
      %param_1 = f32[1,1,4,1023,8]{4,3,2,1,0} parameter(1)
      %bitcast = f32[4,1023,8]{2,1,0} bitcast(f32[1,1,4,1023,8]{4,3,2,1,0} %param_1)
      %broadcast = f32[1,4,1023,8,8]{4,3,2,1,0} broadcast(f32[4,1023,8]{2,1,0} %bitcast), dimensions={1,2,4}
      %add = f32[1,4,1023,8,8]{4,3,2,1,0} add(f32[1,4,1023,8,8]{4,3,2,1,0} %convert, f32[1,4,1023,8,8]{4,3,2,1,0} %broadcast)
      %param_0 = f32[1,4,401,8,8]{4,3,2,1,0} parameter(0)
      ROOT %concatenate = f32[1,4,1424,8,8]{4,3,2,1,0} concatenate(f32[1,4,1023,8,8]{4,3,2,1,0} %add, f32[1,4,401,8,8]{4,3,2,1,0} %param_0), dimensions={2}
    }

    %fused_reduce (param_0: f32[], param_1: f32[1,4,1424,8,8]) -> f32[4,8,8] {
      %param_1 = f32[1,4,1424,8,8]{4,3,2,1,0} parameter(1)
      %bitcast = f32[4,1424,8,8]{3,2,1,0} bitcast(f32[1,4,1424,8,8]{4,3,2,1,0} %param_1)
      %param_0 = f32[] parameter(0)
      ROOT %reduce = f32[4,8,8]{2,1,0} reduce(f32[4,1424,8,8]{3,2,1,0} %bitcast, f32[] %param_0), dimensions={1}, to_apply=%maximum
    }

    %fused_broadcast (param_0: f32[1,4,1424,8,8], param_1: f32[4,8,8]) -> f32[1,4,1424,8,8] {
      %param_0 = f32[1,4,1424,8,8]{4,3,2,1,0} parameter(0)
      %param_1 = f32[4,8,8]{2,1,0} parameter(1)
      %broadcast = f32[1,4,1424,8,8]{4,3,2,1,0} broadcast(f32[4,8,8]{2,1,0} %param_1), dimensions={1,3,4}
      ROOT %subtract = f32[1,4,1424,8,8]{4,3,2,1,0} subtract(f32[1,4,1424,8,8]{4,3,2,1,0} %param_0, f32[1,4,1424,8,8]{4,3,2,1,0} %broadcast)
    }

    ENTRY fusion {
      %param_0 = f32[1,4,401,8,8]{4,3,2,1,0} parameter(0)
      %param_1 = f32[1,1,4,1023,8]{4,3,2,1,0} parameter(1)
      %param_2 = bf16[1,4,1023,8,8]{4,3,2,1,0} parameter(2)
      %concat = f32[1,4,1424,8,8]{4,3,2,1,0} fusion(%param_0, %param_1, %param_2), kind=kLoop, calls=fused_concat
      %param_3 = f32[] parameter(3)
      %reduce = f32[4,8,8]{2,1,0} fusion(%param_3, %concat), kind=kLoop, calls=fused_reduce
      %param_4 = f32[4,8,8]{2,1,0} parameter(4)
      %broadcast = f32[1,4,1424,8,8]{4,3,2,1,0} fusion(%concat, %param_4), kind=kLoop, calls=fused_broadcast
      ROOT tuple = (f32[4,8,8]{2,1,0}, f32[1,4,1424,8,8]{4,3,2,1,0}) tuple(%reduce, %broadcast)
    }
  )");

  EXPECT_THAT(priority_fusion_.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(PriorityFusionTest, FuseOnlySmallConstant) {
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule module

    ENTRY main {
      param_0 = f32[32,32]{1,0} parameter(0)
      c_1 = f32[] constant(1)
      c_2 = f32[32,32] constant({...})
      broadcast = f32[32,32]{1,0} broadcast(c_1), dimensions={}
      add = f32[32,32]{1,0} add(param_0, broadcast)
      ROOT mul = f32[32,32]{1,0} multiply(c_2, add)
    }
  )");
  EXPECT_THAT(priority_fusion_.Run(module.get()), IsOkAndHolds(true));

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::Fusion(m::Constant(), m::Parameter())));
  EXPECT_THAT(root->fused_expression_root(),
              GmockMatch(m::Multiply(
                  m::Parameter(),
                  m::Add(m::Parameter(), m::Broadcast(m::Constant())))));
}

TEST_F(PriorityFusionTest, DoNotFuseProducerConsumerMergedTooLarge) {
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule module

    fused_computation.1 {
      iota.9.7 = s32[3,1,1]{2,1,0} iota(), iota_dimension=0
      param_3.29 = s32[] parameter(2)
      pad.2.7 = s32[3,1,2]{2,1,0} pad(iota.9.7, param_3.29), padding=0_0x0_0x0_1
      param_2.39 = s32[] parameter(1)
      broadcast.76.1 = s32[3,1,2]{2,1,0} broadcast(param_2.39), dimensions={}
      compare.9.1 = pred[3,1,2]{2,1,0} compare(pad.2.7, broadcast.76.1), direction=GE
      param_1.73 = s32[2]{0} parameter(0)
      broadcast.78.1 = s32[3,2]{1,0} broadcast(param_1.73), dimensions={1}
      bitcast.1 = s32[3,2]{1,0} bitcast(pad.2.7)
      compare.10.1 = pred[3,2]{1,0} compare(bitcast.1, broadcast.78.1), direction=LE
      bitcast.2 = pred[3,1,2]{2,1,0} bitcast(compare.10.1)
      ROOT and.3.1 = pred[3,1,2]{2,1,0} and(compare.9.1, bitcast.2)
    }

    and {
      x = pred[] parameter(0)
      y = pred[] parameter(1)
      ROOT and = pred[] and(x, y)
    }

    fused_computation.2 {
      param0 = pred[3,1,2]{2,1,0} parameter(0)
      slice = pred[1,1,2]{2,1,0} slice(param0), slice={[0:1], [0:1], [0:2]}
      bitcast = pred[2]{0} bitcast(slice)
      init = pred[] constant(true)
      reduce = pred[2]{0} reduce(param0, init), dimensions={0,1}, to_apply=and
      and = pred[2]{0} and(bitcast, reduce)
      pad = pred[3]{0} pad(and, init), padding=0_1
      broadcast = pred[3,2]{1,0} broadcast(pad), dimensions={0}
      bitcast2 = pred[6]{0} bitcast(broadcast)
      broadcast2 = pred[2,3]{1,0} broadcast(pad), dimensions={1}
      bitcast3 = pred[6]{0} bitcast(broadcast2)
      ROOT and2 = pred[6]{0} and(bitcast2, bitcast3)
    }

    ENTRY main {
      p0 = s32[2]{0} parameter(0)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      fusion1 = pred[3,1,2]{2,1,0} fusion(p0, p1, p2), kind=kLoop, calls=fused_computation.1
      ROOT fusion2 = pred[6]{0} fusion(fusion1), kind=kInput, calls=fused_computation.2
    }
  )");
  EXPECT_THAT(priority_fusion_.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(PriorityFusionTest, CanMergeTritonFusionWithBothProducerAndConsumer) {
#ifndef GOOGLE_CUDA
  GTEST_SKIP() << "Triton fusion only enable for CUDA devices.";
#endif

  const std::string kHloText = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

producer_computation {
  parameter_0 = f32[125]{0} parameter(0)
  ROOT broadcast = f32[125,127]{1,0} broadcast(parameter_0), dimensions={0}
}

consumer_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  parameter_1 = f32[125,127]{1,0} parameter(1)
  ROOT multiply = f32[125,127]{1,0} multiply(parameter_1, parameter_0)
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
  param_1 = f32[125,127]{1,0} parameter(1)
  producer_fusion = f32[125,127]{1,0} fusion(param_0), kind=kLoop, calls=producer_computation
  triton_softmax = f32[125,127]{1,0} fusion(producer_fusion), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config": {"kind":"__triton"}}
  ROOT consumer_fusion = f32[125,127]{1,0} fusion(param_1, triton_softmax), kind=kLoop, calls=consumer_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  auto debug_options = module->config().debug_options();
  debug_options.set_xla_gpu_enable_triton_softmax_priority_fusion(true);
  module->mutable_config().set_debug_options(debug_options);

  EXPECT_TRUE(priority_fusion_.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kCustom);
  EXPECT_TRUE(IsGenericTritonFusion(*root));
}

TEST_F(PriorityFusionTest, DoNotFuseInsideReducer) {
  auto module = *ParseAndReturnVerifiedModule(R"(
    %reducer {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      add = f32[] add(p0, p1)
      ROOT max = f32[] maximum(add, p0)
    }

    %fused_reduce {
      p0 = f32[256] parameter(0)
      p1 = f32[] parameter(1)
      ROOT reduce = f32[] reduce(p0, p1), dimensions={0}, to_apply=%reducer
    }

    ENTRY fusion {
      p0 = f32[256] parameter(0)
      p1 = f32[] parameter(1)
      ROOT %reduce = f32[] fusion(p0, p1), kind=kInput, calls=fused_reduce
    }
  )");
  EXPECT_THAT(priority_fusion_.Run(module.get()), IsOkAndHolds(false));
}

}  // namespace gpu
}  // namespace xla
