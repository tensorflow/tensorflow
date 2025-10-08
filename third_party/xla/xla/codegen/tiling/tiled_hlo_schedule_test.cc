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

#include "xla/codegen/tiling/tiled_hlo_schedule.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/substitute.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

class TiledHloScheduleTest : public HloHardwareIndependentTestBase {
 protected:
  mlir::MLIRContext ctx_;
};

TEST_F(TiledHloScheduleTest,
       MajorToMinorTiledHloScheduleSatisfiesRootScheduleProperties) {
  constexpr int64_t batch_size = 5;
  constexpr int64_t lhs_non_contracting_size = 2;
  constexpr int64_t rhs_non_contracting_size = 4;
  constexpr int64_t contracting_size = 97;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(absl::Substitute(
                              R"(
ENTRY main {
  p0 = f32[$0,$1,$3] parameter(0)
  p1 = f32[$0,$3,$2] parameter(1)
  ROOT dot = f32[$0,$1,$2] dot(p0, p1),
    lhs_contracting_dims={2}, rhs_contracting_dims={1},
    lhs_batch_dims={0}, rhs_batch_dims={0}
})",
                              batch_size, lhs_non_contracting_size,
                              rhs_non_contracting_size, contracting_size)));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  // The dot instruction has 4 tiling parameters (1 batch, 2 non-contracting
  // dimensions, and 1 contracting dimension).
  constexpr int64_t kNumDotTilingParameters = 4;
  TilingSpecification::ParameterMapping parameter_mapping{
      {root, kNumDotTilingParameters}};

  MajorToMinorTiledHloSchedule schedule;
  TF_ASSERT_OK_AND_ASSIGN(
      IndexingMap indexing_map,
      schedule.RootSchedule(root, parameter_mapping, &ctx_));

  // (1) the map must have exactly as many parameters as there are tiling
  //     parameters in `parameter_mapping`.
  EXPECT_EQ(indexing_map.GetDimVarsCount(), kNumDotTilingParameters);

  // (2) the parameters in the resulting map must appear in the same order as
  //     they appear in `parameter_mapping`.
  Interval contracting_interval{0, contracting_size - 1};
  Interval batch_interval{0, batch_size - 1};
  Interval lhs_non_contracting_interval{0, lhs_non_contracting_size - 1};
  Interval rhs_non_contracting_interval{0, rhs_non_contracting_size - 1};

  auto bounds = indexing_map.GetDimensionBounds();

  EXPECT_THAT(bounds, ElementsAre(contracting_interval, batch_interval,
                                  lhs_non_contracting_interval,
                                  rhs_non_contracting_interval));
  // (3) the map must have as many results as there are output dimensions in
  //     the instruction---although the results are allowed to be outside the
  //     range of the instruction's output space.
  EXPECT_EQ(indexing_map.GetNumResults(), root->shape().dimensions().size());

  // (4) iterating over the entire input space of the map must yield the
  //     entire output space of the instruction.
  // TODO(b/449934916): fix the layering violation.
  gpu::SymbolicExprContext symbolic_expr_context(&ctx_);
  EXPECT_EQ(indexing_map.GetAffineMap(),
            ParseAffineMap("(d0, d1, d2, d3) -> (d1, d2, d3)",
                           &symbolic_expr_context));
}

}  // namespace
}  // namespace xla
