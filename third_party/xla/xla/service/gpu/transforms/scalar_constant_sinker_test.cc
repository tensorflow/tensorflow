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

#include "xla/service/gpu/transforms/scalar_constant_sinker.h"

#include <memory>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ScalarConstantSinkerTest = HloHardwareIndependentTestBase;

TEST_F(ScalarConstantSinkerTest, SinksScalars) {
  RunAndFilecheckHloRewrite(R"(
        // CHECK: fused_computation
        fused_computation {
          // CHECK-DAG: %[[P0:.*]] = s32[200,200,200,200]{3,2,1,0} parameter(0)
          // CHECK-DAG: %[[P1:.*]] = s32[] parameter(1)
          // CHECK-DAG: %[[P2:.*]] = s32[] parameter(2)
          // CHECK-DAG: %[[C1:.*]] = s32[] constant(1)
          // CHECK-DAG: %[[C2:.*]] = s32[] constant(2)
          // CHECK: dynamic-slice(%[[P0]], %[[C2]], %[[P1]], %[[C1]], %[[P2]])
          p0 = s32[200,200,200,200] parameter(0)
          p1 = s32[] parameter(1)
          p2 = s32[] parameter(2)
          p3 = s32[] parameter(3)
          p4 = s32[] parameter(4)
          ROOT slice = s32[100,100,100,100] dynamic-slice(p0, p1, p2, p3, p4),
              dynamic_slice_sizes={100,100,100,100}
        }

        // CHECK: ENTRY
        ENTRY main {
          // CHECK-DAG: %[[P0:.*]] = s32[200,200,200,200]{3,2,1,0} parameter(0)
          // CHECK-DAG: %[[C1:.*]] = s32[] constant(1)
          // CHECK-DAG: %[[P1:.*]] = s32[] parameter(1)
          // CHECK-DAG: %[[P1P1:.*]] = s32[] add
          p0 = s32[200,200,200,200] parameter(0)
          c1 = s32[] constant(1)
          c2 = s32[] constant(2)
          p1 = s32[] parameter(1)
          p1p1 = s32[] add(c1, p1)

          // There should be only three parameters left.
          // CHECK: fusion(%[[P0]], %[[P1P1]], %[[P1]])
          ROOT fusion = s32[100,100,100,100] fusion(p0, c2, p1p1, c1, p1),
              kind=kLoop, calls=fused_computation
        })",
                            ScalarConstantSinker());
}

TEST_F(ScalarConstantSinkerTest, DoesNotSinkTensors) {
  constexpr char kHlo[] = R"(
        fused_computation {
          p0 = s32[2] parameter(0)
          p1 = s32[] parameter(1)
          ROOT slice = s32[1] dynamic-slice(p0, p1), dynamic_slice_sizes={1}
        }

        ENTRY main {
          c0 = s32[2] constant({0,1})
          p0 = s32[] parameter(0)
          ROOT fusion = s32[1] fusion(c0, p0), kind=kLoop,
              calls=fused_computation
        })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(ScalarConstantSinker().Run(module.get()).value());
}

TEST_F(ScalarConstantSinkerTest, DoesNotSinkIntoCustomFusions) {
  constexpr char kHlo[] = R"(
        fused_computation {
          p0 = s32[] parameter(0)
          p1 = s32[] parameter(1)
          ROOT add = s32[] add(p0, p1)
        }

        ENTRY main {
          c0 = s32[] constant(0)
          c1 = s32[] constant(1)
          ROOT fusion = s32[] fusion(c0, c1), kind=kCustom,
              calls=fused_computation
        })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(ScalarConstantSinker().Run(module.get()).value());
}

TEST_F(ScalarConstantSinkerTest, DoesNotSinkIntoNonFusions) {
  constexpr char kHlo[] = R"(
        computation {
          p0 = s32[] parameter(0)
          p1 = s32[] parameter(1)
          ROOT add = s32[] add(p0, p1)
        }

        ENTRY main {
          c0 = s32[] constant(0)
          c1 = s32[] constant(1)
          ROOT fusion = s32[] call(c0, c1), to_apply=computation
        })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(ScalarConstantSinker().Run(module.get()).value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
