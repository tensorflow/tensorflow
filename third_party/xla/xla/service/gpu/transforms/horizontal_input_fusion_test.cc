/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/horizontal_input_fusion.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

class HorizontalInputFusionTest : public GpuCodegenTest {
 public:
  se::DeviceDescription device_description_{
      TestGpuDeviceInfo::RTXA6000DeviceInfo()};
  HorizontalInputFusion horizontal_input_fusion_{device_description_};
};

TEST_F(HorizontalInputFusionTest, BasicTest) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule BasicTest

  %add_f16 {
    %x = f16[] parameter(0)
    %y = f16[] parameter(1)
    ROOT %add = f16[] add(%x, %y)
  }

 fused_computation.1 {
   arg.1 = f16[1024]{0} parameter(0)
   constant0 = f16[] constant(0)
   ROOT reduce1 = f16[] reduce(arg.1, constant0), dimensions={0}, to_apply=%add_f16
 }

 fused_computation.2 {
   arg.1 = f16[1024]{0} parameter(0)
   constant0 = f16[] constant(0)
   ROOT reduce1 = f16[] reduce(arg.1, constant0), dimensions={0}, to_apply=%add_f16
 }

 ENTRY entry_computation {
   arg.1 = f16[1024]{0} parameter(0)
   arg.2 = f16[1024]{0} parameter(1)
   fusion.1 = f16[] fusion(arg.1), kind=kInput, calls=fused_computation.1
   fusion.2 = f16[] fusion(arg.2), kind=kInput, calls=fused_computation.2
   ROOT tuple.1 = (f16[], f16[]) tuple(fusion.1, fusion.2)
 }
)")
                    .value();

  EXPECT_TRUE(horizontal_input_fusion_.Run(module.get()).value());

  const HloInstruction* entry_root =
      module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(entry_root,
              GmockMatch(m::Tuple((m::GetTupleElement(m::Fusion(&fusion))),
                                  (m::GetTupleElement(m::Fusion())))));
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce())));
}

TEST_F(HorizontalInputFusionTest, MultiOutputFusionTest) {
  // This tests the below pattern. One known issue is that gtes (to fusions) can
  // be removed after their producer fusions are merged. In the below case, gte2
  // and gte6 will be gone if Fusion2 is fused into Fusion1.
  //
  // Fusion1   Fusion2
  //  |   |    |     |
  //  |  gte1 gte2   |
  //  |   |    |     |
  //  |   Fusion3    |
  //  |    |   |     |
  // gte3 gte4 gte5 gte6
  //  \  |     |    /
  //  =====ROOT=====
  //
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule MultiOutputFusionTest

  %add_f16 {
    %x = f16[] parameter(0)
    %y = f16[] parameter(1)
    ROOT %add = f16[] add(%x, %y)
  }

 fused_computation.1 {
   arg.1 = f16[1024]{0} parameter(0)
   constant0 = f16[] constant(0)
   reduce.1 = f16[] reduce(arg.1, constant0), dimensions={0}, to_apply=%add_f16
   add.0 = f16[1024] add(arg.1, arg.1)
   ROOT tuple.1 = (f16[], f16[1024]) tuple(reduce.1, add.0)
 }

 fused_computation.2 {
   arg.1 = f16[1024]{0} parameter(0)
   constant0 = f16[] constant(0)
   reduce.1 = f16[] reduce(arg.1, constant0), dimensions={0}, to_apply=%add_f16
   add.0 = f16[1024] add(arg.1, arg.1)
   ROOT tuple.1 = (f16[], f16[1024]) tuple(reduce.1, add.0)
 }

 fused_computation.3 {
   arg.0 = f16[1024]{0} parameter(0)
   arg.1 = f16[1024]{0} parameter(1)
   add.0 = f16[1024] add(arg.0, arg.1)
   mul.0 = f16[1024] multiply(arg.0, arg.1)
   ROOT tuple.1 = (f16[1024], f16[1024]) tuple(add.0, mul.0)
 }

 ENTRY entry_computation {
   arg.1 = f16[1024]{0} parameter(0)
   arg.2 = f16[1024]{0} parameter(1)
   fusion.1 = (f16[],f16[1024]) fusion(arg.1), kind=kInput, calls=fused_computation.1
   fusion.2 = (f16[],f16[1024]) fusion(arg.2), kind=kInput, calls=fused_computation.2
   gte.3 = f16[] get-tuple-element(fusion.1), index=0
   gte.1 = f16[1024]{0} get-tuple-element(fusion.1), index=1
   gte.2 = f16[1024]{0} get-tuple-element(fusion.2), index=1
   gte.6 = f16[] get-tuple-element(fusion.2), index=0
   fusion.3 = (f16[1024],f16[1024]) fusion(gte.1, gte.2),
       kind=kLoop, calls=fused_computation.3
   gte.4 = f16[1024] get-tuple-element(fusion.3), index=0
   gte.5 = f16[1024]{0} get-tuple-element(fusion.3), index=1
   ROOT tuple.1 = (f16[], f16[1024], f16[1024]{0}, f16[])
       tuple(gte.3, gte.4, gte.5, gte.6)
 }
)")
                    .value();

  EXPECT_TRUE(horizontal_input_fusion_.Run(module.get()).value());
}

TEST_F(HorizontalInputFusionTest, NonfusionInstrs) {
  auto module = ParseAndReturnVerifiedModule(R"(
 HloModule NonfusionInstrs

 %add_f16 {
   %x = f16[] parameter(0)
   %y = f16[] parameter(1)
   ROOT %add = f16[] add(%x, %y)
 }

 ENTRY entry_computation {
   arg.0 = f16[1024]{0} parameter(0)
   arg.1 = f16[1024]{0} parameter(1)
   constant0 = f16[] constant(0)
   reduce.0 = f16[] reduce(arg.0, constant0), dimensions={0}, to_apply=%add_f16
   reduce.1 = f16[] reduce(arg.1, constant0), dimensions={0}, to_apply=%add_f16
   ROOT tuple.0 = (f16[], f16[]) tuple(reduce.0, reduce.1)
 }
)")
                    .value();

  EXPECT_TRUE(horizontal_input_fusion_.Run(module.get()).value());

  const HloInstruction* entry_root =
      module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(entry_root,
              GmockMatch(m::Tuple((m::GetTupleElement(m::Fusion(&fusion))),
                                  (m::GetTupleElement(m::Fusion())))));
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Reduce(), m::Reduce())));
}

TEST_F(HorizontalInputFusionTest, DoesNotFuseCustomFusions) {
  auto module = ParseAndReturnVerifiedModule(R"(
max {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT max = f16[] maximum(p0, p1)
}

triton_a {
   p = f16[128,256] parameter(0)
   c = f16[] constant(0)
   ROOT n = f16[128] reduce(p, c), dimensions={1}, to_apply=max
}

triton_b {
   p = f16[128,256] parameter(0)
   c = f16[] constant(0)
   ROOT n = f16[128] reduce(p, c), dimensions={1}, to_apply=max
}

 ENTRY entry_computation {
   p = f16[128,256] parameter(0)
   fa = f16[128] fusion(p), kind=kCustom, calls=triton_a
   fb = f16[128] fusion(p), kind=kCustom, calls=triton_b
   ROOT tuple = (f16[128], f16[128]) tuple(fa, fb)
 }
)")
                    .value();

  EXPECT_FALSE(horizontal_input_fusion_.Run(module.get()).value());
}

TEST_F(HorizontalInputFusionTest, ChangedModuleIsReportedCorrectly) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
g {
  a = s8[] parameter(0)
  b = s8[] parameter(1)
  c = s8[] add(a, b)
}

f {
  p0 = s8[8] parameter(0)
  p1 = s8[8] parameter(1)
  c = s8[] constant(0)
  r1 = s8[] reduce(p0, c), dimensions={0}, to_apply=g
  r2 = s8[] reduce(p1, c), dimensions={0}, to_apply=g
  a = s8[] add(r1, r2)
}

e {
  p0 = s8[8] parameter(0)
  p1 = s8[8] parameter(1)
  c = s8[] call(p0, p1), to_apply=f
})"));
  EXPECT_TRUE(horizontal_input_fusion_.Run(module.get()).value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
