/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_hlo_ordering.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

class ConcurrentRegionsHloOrderingTest : public HloHardwareIndependentTestBase {
};

TEST_F(ConcurrentRegionsHloOrderingTest, ExecutesBeforeInConcurrentRegion) {
  auto module = CreateNewVerifiedModule();
  const Shape small_shape = ShapeUtil::MakeShape(xla::F32, {1024, 1024});

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, small_shape, "param"));
  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateUnary(small_shape, HloOpcode::kNegate, param));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateUnary(small_shape, HloOpcode::kAbs, param));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateUnary(small_shape, HloOpcode::kAcos, param));
  HloInstruction* d = builder.AddInstruction(
      HloInstruction::CreateUnary(small_shape, HloOpcode::kCeil, param));
  HloInstruction* root =
      builder.AddInstruction(HloInstruction::CreateTuple({a, b, c, d}));
  HloComputation* entry =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/root));

  HloSchedule schedule(module.get());
  schedule.set_sequence(entry, {param, a, b, c, d, root});
  TF_ASSERT_OK(schedule.Verify());
  ConcurrentRegionsHloOrdering ordering(schedule);
  // There are no data dependencies between a, b, c, and d. All ops can be
  // executed concurrently.
  EXPECT_FALSE(ordering.ExecutesBefore(a, b));
  EXPECT_FALSE(ordering.ExecutesBefore(b, c));
  EXPECT_FALSE(ordering.ExecutesBefore(c, d));

  // All ops land in the same concurrent region.
  EXPECT_EQ(ordering.GetConcurrentRegionId(a), 0);
  EXPECT_EQ(ordering.GetConcurrentRegionId(b), 0);
  EXPECT_EQ(ordering.GetConcurrentRegionId(c), 0);
  EXPECT_EQ(ordering.GetConcurrentRegionId(d), 0);
}

TEST_F(ConcurrentRegionsHloOrderingTest, DataDependentOpsInSameRegion) {
  auto module = CreateNewVerifiedModule();
  const Shape small_shape = ShapeUtil::MakeShape(xla::F32, {1024, 1024});

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, small_shape, "param"));
  HloInstruction* a0 = builder.AddInstruction(
      HloInstruction::CreateUnary(small_shape, HloOpcode::kNegate, param));
  HloInstruction* a1 = builder.AddInstruction(
      HloInstruction::CreateUnary(small_shape, HloOpcode::kAbs, a0));
  HloInstruction* b0 = builder.AddInstruction(
      HloInstruction::CreateUnary(small_shape, HloOpcode::kAcos, param));
  HloInstruction* b1 = builder.AddInstruction(
      HloInstruction::CreateUnary(small_shape, HloOpcode::kCeil, b0));
  HloInstruction* root =
      builder.AddInstruction(HloInstruction::CreateTuple({a1, b1}));
  HloComputation* entry =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/root));

  HloSchedule schedule(module.get());
  schedule.set_sequence(entry, {param, a0, a1, b0, b1, root});
  TF_ASSERT_OK(schedule.Verify());
  ConcurrentRegionsHloOrdering ordering(schedule);
  // a1 has a data dependency on a0.
  EXPECT_TRUE(ordering.ExecutesBefore(a0, a1));
  // b1 has a data dependency on b0.
  EXPECT_TRUE(ordering.ExecutesBefore(b0, b1));
  // a0 and b0 have no data dependency and can be executed concurrently.
  EXPECT_FALSE(ordering.ExecutesBefore(a0, b0));
  // a1 and b1 have no data dependency and can be executed concurrently.
  EXPECT_FALSE(ordering.ExecutesBefore(a1, b1));

  // All ops land in the same concurrent region.
  EXPECT_EQ(ordering.GetConcurrentRegionId(a0), 0);
  EXPECT_EQ(ordering.GetConcurrentRegionId(b0), 0);
  EXPECT_EQ(ordering.GetConcurrentRegionId(a1), 0);
  EXPECT_EQ(ordering.GetConcurrentRegionId(b1), 0);
}

TEST_F(ConcurrentRegionsHloOrderingTest, GemmSeparatesConcurrentRegions) {
  auto module = CreateNewVerifiedModule();
  const Shape small_shape = ShapeUtil::MakeShape(xla::F32, {1024, 1024});

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, small_shape, "param"));
  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateUnary(small_shape, HloOpcode::kNegate, param));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateUnary(small_shape, HloOpcode::kAbs, param));

  // Create a fused computation for the GEMM.
  auto fused_builder = HloComputation::Builder("fused_computation");
  HloInstruction* fused_param_a = fused_builder.AddInstruction(
      HloInstruction::CreateParameter(0, small_shape, "fused_param_a"));
  HloInstruction* fused_param_b = fused_builder.AddInstruction(
      HloInstruction::CreateParameter(1, small_shape, "fused_param_b"));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* fused_dot = fused_builder.AddInstruction(
      HloInstruction::CreateDot(small_shape, fused_param_a, fused_param_b,
                                dot_dnums, DefaultPrecisionConfig(2)));
  HloComputation* fused_computation =
      module->AddEmbeddedComputation(fused_builder.Build(fused_dot));

  HloInstruction* gemm = builder.AddInstruction(HloInstruction::CreateFusion(
      small_shape, HloInstruction::FusionKind::kCustom, {a, b},
      fused_computation, "gemm_"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateUnary(small_shape, HloOpcode::kAcos, gemm));
  HloInstruction* d = builder.AddInstruction(
      HloInstruction::CreateUnary(small_shape, HloOpcode::kCeil, gemm));
  HloInstruction* root =
      builder.AddInstruction(HloInstruction::CreateTuple({c, d}));
  HloComputation* entry =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/root));

  HloSchedule schedule(module.get());
  schedule.set_sequence(entry, {param, a, b, gemm, c, d, root});
  TF_ASSERT_OK(schedule.Verify());
  ConcurrentRegionsHloOrdering ordering(schedule);
  // No data dependency between a and b and can be executed concurrently.
  EXPECT_FALSE(ordering.ExecutesBefore(a, b));
  // The GEMM is executed sequentially after a and b.
  EXPECT_TRUE(ordering.ExecutesBefore(a, gemm));
  EXPECT_TRUE(ordering.ExecutesBefore(b, gemm));
  // c and d have no data dependency on gemm.
  EXPECT_TRUE(ordering.ExecutesBefore(gemm, c));
  EXPECT_TRUE(ordering.ExecutesBefore(gemm, d));
  // c and d have no data dependency and can be executed concurrently.
  EXPECT_FALSE(ordering.ExecutesBefore(c, d));

  // All ops land in the same concurrent region.
  EXPECT_EQ(ordering.GetConcurrentRegionId(a), 0);
  EXPECT_EQ(ordering.GetConcurrentRegionId(b), 0);
  EXPECT_EQ(ordering.GetConcurrentRegionId(gemm), 1);
  EXPECT_EQ(ordering.GetConcurrentRegionId(c), 2);
  EXPECT_EQ(ordering.GetConcurrentRegionId(d), 2);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
