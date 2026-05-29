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

#include "xla/codegen/tiling/experimental/scheduling.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_traversal.h"

namespace xla::gpu::experimental {
namespace {

using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;
using mlir::MLIRContext;

}  // namespace

MATCHER_P(MatchSchedule, schedule_string, "") {
  return ExplainMatchResult(
      true, ApproximateMatch(schedule_string, arg.ToString()), result_listener);
}

class SchedulingTest : public HloHardwareIndependentTestBase {
 public:
  SchedulingTest() : HloHardwareIndependentTestBase() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_string);
    CHECK_OK(module_or);
    module_ = std::move(module_or.value());
    return module_->entry_computation()->root_instruction();
  }

  absl::StatusOr<TiledHloComputation> ParseAndTile(
      absl::string_view hlo_string, absl::Span<const int64_t> tile_sizes = {}) {
    HloInstruction* root = ParseAndGetRoot(hlo_string);
    auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
    auto tiling_space = TilingSpace::Create(*fusion_adaptor, &mlir_context_);
    if (!tile_sizes.empty()) {
      RETURN_IF_ERROR(tiling_space->AssignTileSizes(tile_sizes));
    }
    return TiledHloComputation::Tile(*fusion_adaptor, std::move(tiling_space));
  }

  mlir::MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

TEST_F(SchedulingTest, OnlyParallelDimensions) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    fusion {
      p0 = f32[2,97]{1,0} parameter(0)
      p1 = f32[2,97]{1,0} parameter(1)
      ROOT subtract = f32[2,97]{1,0} subtract(p0, p1)
    }
    ENTRY main {
      p0 = f32[2,97]{1,0} parameter(0)
      p1 = f32[2,97]{1,0} parameter(1)
      ROOT fusion = f32[2,97]{1,0} fusion(p0, p1), kind=kLoop, calls=fusion
    })",
                                    {1, 32}));
  auto scheduling = GetSchedule(tiled_computation);
  EXPECT_THAT(scheduling,
              IsOkAndHolds(MatchSchedule(
                  "d0 -> pid floordiv 4, d1 -> pid mod 4, pid_bounds=[0, 7]")));
}

TEST_F(SchedulingTest, ReductionsAndContractionsAreNotSupported) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    max {
      p1 = f32[] parameter(1)
      p0 = f32[] parameter(0)
      ROOT m = f32[] maximum(p0, p1)
    }
    fusion {
      p0 = f32[2,97]{1,0} parameter(0)
      constant = f32[] constant(-inf)
      reduce = f32[2] reduce(p0, constant), dimensions={1}, to_apply=max
      broadcast = f32[2,97]{1,0} broadcast(reduce), dimensions={0}
      ROOT subtract = f32[2,97]{1,0} subtract(p0, broadcast)
    }
    ENTRY main {
      p0 = f32[2,97]{1,0} parameter(0)
      ROOT fusion = f32[2,97]{1,0} fusion(p0), kind=kLoop, calls=fusion
    })",
                                    {1, 32, /*reduction_tile_size=*/8}));
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(MatchSchedule(
                  "d0 -> pid floordiv 4, d1 -> pid mod 4, pid_bounds=[0, 7]")));
}

TEST_F(SchedulingTest, GetDotPermutationMultipleBatchDims) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    fusion {
      p0 = f32[2,3,16,128]{3,2,1,0} parameter(0)
      p1 = f32[2,3,128,4096]{3,2,1,0} parameter(1)
      ROOT dot = f32[2,3,16,4096]{3,2,1,0} dot(p0, p1),
        lhs_batch_dims={0,1}, lhs_contracting_dims={3},
        rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
    ENTRY main {
      p0 = f32[2,3,16,128]{3,2,1,0} parameter(0)
      p1 = f32[2,3,128,4096]{3,2,1,0} parameter(1)
      ROOT fusion = f32[2,3,16,4096]{3,2,1,0} fusion(p0, p1), kind=kLoop, calls=fusion
    })",
                                    {1, 1, 8, 64, 32}));
  // d0, d1 are batch. d2 is m, d3 is n.
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(MatchSchedule(
                  "d0 -> pid floordiv 384, d1 -> (pid mod 384) floordiv 128, "
                  "d2 -> pid mod 384 mod 128 mod 2, "
                  "d3 -> (pid mod 384 mod 128) floordiv 2, "
                  "pid_bounds=[0, 767]")));
}

}  // namespace xla::gpu::experimental
