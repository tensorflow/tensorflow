/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/hlo_fusion_stats.h"

#include <string>

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace gpu {
namespace {

using HloFusionStatsTest = HloTestBase;

TEST_F(HloFusionStatsTest, LoopFusionAndReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    scalar_add_computation {
      scalar_lhs.0 = f32[] parameter(0)
      scalar_rhs.0 = f32[] parameter(1)
      ROOT add.0 = f32[] add(scalar_lhs.0, scalar_rhs.0)
    }

    fused_select {
      p1.1 = f32[32,32,32]{2,1,0} parameter(1)
      c0 = f32[] constant(0)
      broadcast = f32[32,32,32]{2,1,0} broadcast(f32[] c0), dimensions={}
      greater-than = pred[32,32,32]{2,1,0} compare(f32[32,32,32]{2,1,0} p1.1,
        f32[32,32,32]{2,1,0} broadcast), direction=GT
      p0.1 = f32[32,32,32]{2,1,0} parameter(0)
      ROOT select = f32[32,32,32]{2,1,0} select(pred[32,32,32]{2,1,0}
        greater-than, f32[32,32,32]{2,1,0} p0.1, f32[32,32,32]{2,1,0} broadcast)
    }

    another_fused_select {
      p1.1 = f32[32,32,32]{2,1,0} parameter(1)
      c0 = f32[] constant(0)
      broadcast = f32[32,32,32]{2,1,0} broadcast(f32[] c0), dimensions={}
      greater-than = pred[32,32,32]{2,1,0} compare(f32[32,32,32]{2,1,0} p1.1,
        f32[32,32,32]{2,1,0} broadcast), direction=GT
      p0.1 = f32[32,32,32]{2,1,0} parameter(0)
      ROOT select = f32[32,32,32]{2,1,0} select(pred[32,32,32]{2,1,0}
        greater-than, f32[32,32,32]{2,1,0} p0.1, f32[32,32,32]{2,1,0} broadcast)
    }

    fused_reduce {
      p0.2 = f32[32,32,32]{2,1,0} parameter(0)
      c1 = f32[] constant(0)
      r1 = f32[32,32]{1,0} reduce(p0.2, c1), dimensions={2},
        to_apply=scalar_add_computation
      mul = f32[32,32,32]{2,1,0} multiply(p0.2, p0.2)
      r2 = f32[32,32]{1,0} reduce(mul, c1), dimensions={2},
        to_apply=scalar_add_computation
      ROOT tuple = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(r1, r2)
    }

    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[32,32,32]{2,1,0} parameter(1)
      select = f32[32,32,32]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_select
      select_2 = f32[32,32,32]{2,1,0} fusion(p0, p1), kind=kLoop, calls=another_fused_select
      fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(select), kind=kInput,
        calls=fused_reduce
      gte0 = f32[32,32]{1,0} get-tuple-element(fusion), index=0
      gte1 = f32[32,32]{1,0} get-tuple-element(fusion), index=1
      ROOT root = (f32[32,32]{1,0}, f32[32,32]{1,0}, f32[32,32,32]{2,1,0}, f32[32,32,32]{2,1,0})
        tuple(gte1, gte1, select, select_2)
    })")
                    .ValueOrDie();
  HloFusionStatsVisitor fusion_stats_visitor;
  TF_ASSERT_OK(
      module.get()->entry_computation()->Accept(&fusion_stats_visitor));
  SCOPED_TRACE(module->ToString());

  std::string stats = fusion_stats_visitor.ToString();
  ASSERT_TRUE(absl::StrContains(stats, "Number of fusion ops: 3"));
  ASSERT_TRUE(absl::StrContains(stats, "Number of kLoop fusions: 2"));
  ASSERT_TRUE(absl::StrContains(stats, "{broadcast, compare, select}: 2"));
  ASSERT_TRUE(absl::StrContains(stats, "Number of kInput fusions: 1"));
  ASSERT_TRUE(absl::StrContains(stats, "{cwise, reduce, tuple}: 1"));
}

TEST_F(HloFusionStatsTest, AggregateCwiseOps) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    fused_computation {
      p0.1 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      mul = f32[8,1,5,16,1,2]{5,4,3,2,1,0} multiply(p0.1, p0.1)
      ROOT exp = f32[8,1,5,16,1,2]{5,4,3,2,1,0} exponential(mul)
    }

    ENTRY entry {
      p0 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      ROOT fusion = f32[8,1,5,16,1,2]{5,4,3,2,1,0} fusion(p0), kind=kLoop,
        calls=fused_computation
    })")
                    .ValueOrDie();
  HloFusionStatsVisitor fusion_stats_visitor;
  TF_ASSERT_OK(
      module.get()->entry_computation()->Accept(&fusion_stats_visitor));
  SCOPED_TRACE(module->ToString());

  std::string stats = fusion_stats_visitor.ToString();
  ASSERT_TRUE(absl::StrContains(stats, "{cwise}: 1")) << stats;
}

}  // namespace
}  // namespace gpu
}  // namespace xla
