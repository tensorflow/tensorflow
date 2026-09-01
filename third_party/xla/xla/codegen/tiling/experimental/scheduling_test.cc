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
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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
    ABSL_ASSIGN_OR_RETURN(auto tiling_space,
                     TilingSpace::Create(*fusion_adaptor, &mlir_context_));
    if (!tile_sizes.empty()) {
      ABSL_RETURN_IF_ERROR(tiling_space->AssignTileSizes(tile_sizes));
    }
    ABSL_ASSIGN_OR_RETURN(
        TiledHloComputation tiled_computation,
        TiledHloComputation::Tile(*fusion_adaptor, std::move(tiling_space)));
    tiled_computation.Simplify();
    tiled_computation.SortInstructionsPostOrder();
    return tiled_computation;
  }

  mlir::MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

TEST_F(SchedulingTest, OnlyParallelDimensionsTwoTilesPerProgramId) {
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
  auto scheduling = GetSchedule(tiled_computation, 2);
  EXPECT_THAT(scheduling,
              IsOkAndHolds(MatchSchedule("d0 -> (pid * 2 + tid) / 4, "
                                         "d1 -> (pid * 2 + tid) mod 4, "
                                         "num_pids=4, num_tiles=8")));
}

TEST_F(SchedulingTest, OnlyParallelDimensionsThreeTilesPerProgramId) {
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
  auto scheduling = GetSchedule(tiled_computation, 3);
  EXPECT_THAT(scheduling,
              IsOkAndHolds(MatchSchedule("d0 -> (pid * 3 + tid) / 4, "
                                         "d1 -> (pid * 3 + tid) mod 4, "
                                         "num_pids=3, num_tiles=8")));
}

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
              IsOkAndHolds(MatchSchedule("d0 -> pid / 4, d1 -> pid mod 4, "
                                         "num_pids=8, num_tiles=8")));
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
              IsOkAndHolds(MatchSchedule("d0 -> pid / 4, d1 -> pid mod 4, "
                                         "num_pids=8, num_tiles=8")));
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
  EXPECT_THAT(
      GetSchedule(tiled_computation),
      IsOkAndHolds(MatchSchedule("d0 -> pid / 384, d1 -> (pid mod 384) / 128, "
                                 "d2 -> pid mod 384 mod 128 mod 2, "
                                 "d3 -> (pid mod 384 mod 128) / 2, "
                                 "num_pids=768, num_tiles=768")));
}

// kRaggedNonContracting M/N swap heuristic:
// When M_avg = M_total/G < N, the scheduler traverses N more slowly
// so that LHS activation tiles (size M_avg) stay in L2 cache.
//
// Shapes: LHS f32[64,256] (M=64,K=256), RHS f32[8,256,128] (G=8,K=256,N=128)
//   group_sizes s32[8], output f32[64,128]
//   M_avg = 64/8 = 8 < N = 128  →  swap M and N traversal order.
//
// TilingSpace:
//   dim 0: M=64   kParallel
//   dim 1: N=128  kParallel
//   dim 2: G=8    kSequential  (G outer loop, tile_size=1)
//   dim 3: K=256  kSequential  (K inner loop, tile_size=32)
//
// Tile sizes: [BLOCK_M=8, BLOCK_N=16, tile_G=1, BLOCK_K=32]
//   M blocks = 64/8 = 8, N blocks = 128/16 = 8, total = 64 tiles.
//
// Without swap: d0(M) is outer (slower), d1(N) is inner (faster).
// With swap:    d1(N) is outer (slower), d0(M) is inner (faster).
//   → d0 → pid % 8, d1 → pid / 8.
TEST_F(SchedulingTest, GetRaggedDotNonContractingPermutationSwapsMN) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    fusion {
      lhs = f32[64,256] parameter(0)
      rhs = f32[8,256,128] parameter(1)
      gs  = s32[8] parameter(2)
      ROOT rd = f32[64,128] ragged-dot(lhs, rhs, gs),
          lhs_contracting_dims={1}, rhs_contracting_dims={1},
          lhs_ragged_dims={0}, rhs_group_dims={0}
    }
    ENTRY main {
      p0 = f32[64,256] parameter(0)
      p1 = f32[8,256,128] parameter(1)
      p2 = s32[8] parameter(2)
      ROOT fusion = f32[64,128] fusion(p0, p1, p2), kind=kLoop, calls=fusion
    })",
                                    {8, 16, 1, 32}));
  // M_avg = 8 < N = 128 → swap: N outer (slower), M inner (faster).
  EXPECT_THAT(
      GetSchedule(tiled_computation),
      IsOkAndHolds(MatchSchedule(
          "d0 -> pid mod 8, d1 -> pid / 8, num_pids=64, num_tiles=64")));
}

// When M_avg >= N, no swap is beneficial (LHS is the larger operand).
// Shapes: LHS f32[512,256] (M=512,K=256), RHS f32[8,256,64] (G=8,K=256,N=64)
//   M_avg = 512/8 = 64 >= N = 64  →  no swap (condition is M_avg < N).
// Tile sizes: [BLOCK_M=32, BLOCK_N=32, tile_G=1, BLOCK_K=32]
//   M blocks = 512/32 = 16, N blocks = 64/32 = 2, total = 32 tiles.
TEST_F(SchedulingTest, GetRaggedDotNonContractingPermutationNoSwapWhenMGeN) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    fusion {
      lhs = f32[512,256] parameter(0)
      rhs = f32[8,256,64] parameter(1)
      gs  = s32[8] parameter(2)
      ROOT rd = f32[512,64] ragged-dot(lhs, rhs, gs),
          lhs_contracting_dims={1}, rhs_contracting_dims={1},
          lhs_ragged_dims={0}, rhs_group_dims={0}
    }
    ENTRY main {
      p0 = f32[512,256] parameter(0)
      p1 = f32[8,256,64] parameter(1)
      p2 = s32[8] parameter(2)
      ROOT fusion = f32[512,64] fusion(p0, p1, p2), kind=kLoop, calls=fusion
    })",
                                    {32, 32, 1, 32}));
  // M_avg = 64 >= N = 64 → no swap: d0(M) outer (slower), d1(N) inner (faster).
  EXPECT_THAT(
      GetSchedule(tiled_computation),
      IsOkAndHolds(MatchSchedule(
          "d0 -> pid / 2, d1 -> pid mod 2, num_pids=32, num_tiles=32")));
}

}  // namespace xla::gpu::experimental
