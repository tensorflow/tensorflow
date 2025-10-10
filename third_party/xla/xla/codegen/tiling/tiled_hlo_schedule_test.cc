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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class TiledHloScheduleTest : public HloHardwareIndependentTestBase {
 protected:
  mlir::MLIRContext mlir_context_;
  gpu::SymbolicExprContext symbolic_expr_context_{&mlir_context_};
};

using MajorToMinorTiledHloScheduleTest = TiledHloScheduleTest;

TEST_F(MajorToMinorTiledHloScheduleTest,
       MajorToMinorTiledHloScheduleSatisfiesScheduleProperties) {
  IndexingMap offsets_indexing = *ParseIndexingMap(R"(
      (d0, d1, d2, d3) -> (d2, d3),
      domain: d0 in [0, 1], d1 in [0, 2], d2 in [0, 4], d3 in [0, 6])",
                                                   &symbolic_expr_context_);
  auto bound = [&offsets_indexing](int64_t dim) {
    return offsets_indexing.GetDimensionBound(dim).upper + 1;
  };
  std::vector<DimensionInfo> iteration_space = {
      {/*dimension_id=*/2, /*dimension_size=*/bound(2)},
      {/*dimension_id=*/0, /*dimension_size=*/bound(0)},
      {/*dimension_id=*/1, /*dimension_size=*/bound(1)},
      {/*dimension_id=*/3, /*dimension_size=*/bound(3)},
  };

  MajorToMinorTiledHloSchedule scheduler;
  TF_ASSERT_OK_AND_ASSIGN(IndexingMap scheduled_indexing,
                          scheduler.Schedule(offsets_indexing, iteration_space,
                                             &symbolic_expr_context_));

  // (1) the map must have a single input whose range of values is the size of
  //     the iteration space (i.e. the product of `iteration_space`'s
  //     `dimension_size`s);
  EXPECT_EQ(scheduled_indexing.GetDimVarsCount(), 1);
  int64_t iteration_space_size = bound(0) * bound(1) * bound(2) * bound(3);
  Interval expected_parameter_interval{0, iteration_space_size - 1};
  EXPECT_EQ(scheduled_indexing.GetDimensionBound(0),
            expected_parameter_interval);

  // (2) the set of results generatable with the map must be equal to the set
  //     of results of `tile_offsets_indexing` (i.e. the map may only reorder
  //     how the results are generated, but may not change the results
  //     themselves);
  EXPECT_EQ(scheduled_indexing, *ParseIndexingMap(R"(
    (pid_0) -> (pid_0 floordiv 42, pid_0 mod 7), domain: pid_0 in [0, 209]
  )",
                                                  &symbolic_expr_context_));

  // `pid_0 floordiv 42` has the same upper bound as `d2`.
  EXPECT_EQ(iteration_space_size / 42, bound(2));
  // `pid_0 mod 7` has the same upper bound as `d3`.
  EXPECT_EQ(7, bound(3));
}

TEST_F(MajorToMinorTiledHloScheduleTest,
       MajorToMinorTiledHloScheduleFailsForInvalidIterationSpace) {
  IndexingMap offsets_indexing =
      *ParseIndexingMap("(d0, d1) -> (d1), domain: d0 in [0, 1], d1 in [0, 2]",
                        &symbolic_expr_context_);
  MajorToMinorTiledHloSchedule scheduler;

  // The iteration space has the wrong number of dimensions.
  EXPECT_THAT(
      scheduler.Schedule(offsets_indexing, /*iteration_space=*/{},
                         &symbolic_expr_context_),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Expected iteration space to have exactly as many dimensions")));

  // The iteration space has an out-of-bounds dimension ID.
  EXPECT_THAT(scheduler.Schedule(offsets_indexing, /*iteration_space=*/
                                 {{/*dimension_id=*/0, /*dimension_size=*/1},
                                  {/*dimension_id=*/2, /*dimension_size=*/0}},
                                 &symbolic_expr_context_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Dimension id 2 is out of bounds")));
}

}  // namespace
}  // namespace xla
