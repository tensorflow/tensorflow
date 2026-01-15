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
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::mlir::AffineExpr;
using ::testing::HasSubstr;

class TiledHloScheduleTest : public HloHardwareIndependentTestBase {
 protected:
  TiledHloScheduleTest() { RegisterSymbolicExprStorage(&mlir_context_); }
  mlir::MLIRContext mlir_context_;
};

using MajorToMinorTiledHloScheduleTest = TiledHloScheduleTest;

TEST_F(MajorToMinorTiledHloScheduleTest,
       MajorToMinorTiledHloScheduleSatisfiesScheduleProperties) {
  IndexingMap offsets_indexing = *ParseIndexingMap(R"(
      (d0, d1, d2, d3) -> (d2, d3),
      domain: d0 in [0, 1], d1 in [0, 2], d2 in [0, 4], d3 in [0, 6])",
                                                   &mlir_context_);
  auto bound = [&offsets_indexing](int64_t dim) {
    return offsets_indexing.GetDimensionBound(dim).upper + 1;
  };
  std::vector<DimensionInfo> iteration_space = {
      {/*dimension_id=*/2, /*dimension_size=*/bound(2)},
      {/*dimension_id=*/1, /*dimension_size=*/bound(1)},
      {/*dimension_id=*/3, /*dimension_size=*/bound(3)},
  };

  MajorToMinorTiledHloSchedule scheduler;
  TF_ASSERT_OK_AND_ASSIGN(
      IndexingMap scheduled_indexing,
      scheduler.Schedule(offsets_indexing, iteration_space, &mlir_context_));

  // (1) the map must have a single input whose range of values is the size of
  //     the iteration space (i.e. the product of `iteration_space`'s
  //     `dimension_size`s);
  EXPECT_EQ(scheduled_indexing.GetDimVarsCount(), 1);
  int64_t iteration_space_size = bound(1) * bound(2) * bound(3);
  Interval expected_parameter_interval{0, iteration_space_size - 1};
  EXPECT_EQ(scheduled_indexing.GetDimensionBound(0),
            expected_parameter_interval);

  // (2) the set of results generatable with the map must be equal to the set
  //     of results of `tile_offsets_indexing` on the subspace defined by the
  //     parameter iteration space (i.e. the map may only reorder how the
  //     results are generated, but may not change the results themselves);
  EXPECT_EQ(scheduled_indexing, *ParseIndexingMap(R"(
    (pid_0) -> (pid_0 floordiv 21, pid_0 mod 7), domain: pid_0 in [0, 104]
  )",
                                                  &mlir_context_));

  // `pid_0 floordiv 21` has the same upper bound as `d2`.
  EXPECT_EQ(iteration_space_size / 21, bound(2));
  // `pid_0 mod 7` has the same upper bound as `d3`.
  EXPECT_EQ(7, bound(3));
}

TEST_F(MajorToMinorTiledHloScheduleTest,
       MajorToMinorTiledHloScheduleFailsForInvalidIterationSpace) {
  IndexingMap offsets_indexing = *ParseIndexingMap(
      "(d0, d1) -> (d1), domain: d0 in [0, 1], d1 in [0, 2]", &mlir_context_);
  MajorToMinorTiledHloSchedule scheduler;

  // The iteration space has too many dimensions.
  EXPECT_THAT(
      scheduler.Schedule(offsets_indexing, /*iteration_space=*/
                         {{/*dimension_id=*/0, /*dimension_size=*/1},
                          {/*dimension_id=*/1, /*dimension_size=*/3},
                          {/*dimension_id=*/2, /*dimension_size=*/0}},
                         &mlir_context_),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Expected iteration space to have at most as many dimensions")));

  // The iteration space has an out-of-bounds dimension ID.
  EXPECT_THAT(scheduler.Schedule(offsets_indexing, /*iteration_space=*/
                                 {{/*dimension_id=*/0, /*dimension_size=*/1},
                                  {/*dimension_id=*/2, /*dimension_size=*/0}},
                                 &mlir_context_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Dimension id 2 is out of bounds")));
}

class TransposedDotTiledHloScheduleTest : public TiledHloScheduleTest {
 public:
  absl::StatusOr<TilingSpecification> TilingSpecificationForModule(
      HloModule* module) {
    SymbolicTileAnalysisOrError analysis_or_error =
        SymbolicTileAnalysis::AnalyzeComputation(
            *module->entry_computation()
                 ->root_instruction()
                 ->fused_instructions_computation(),
            &mlir_context_,
            /*emitter_specific_constraints_builder=*/nullptr);

    if (!std::holds_alternative<SymbolicTileAnalysis>(analysis_or_error)) {
      return absl::InvalidArgumentError(
          "SymbolicTileAnalysis expected to be present");
    }
    return std::get<SymbolicTileAnalysis>(std::move(analysis_or_error))
        .GetTilingSpecification();
  }
};

TEST_F(TransposedDotTiledHloScheduleTest,
       CanBeCreatedForFusionRootedInSingleDot) {
  constexpr absl::string_view kSupportedFusionHlo = R"(
lhs {
  ROOT p0 = bf16[2,3,8192,256] parameter(0)
}

rhs {
  ROOT p0 = bf16[2,3,256,512] parameter(0)
}

dot {
  p0 = bf16[2,3,8192,256] parameter(0)
  p1 = bf16[2,3,256,512] parameter(1)

  lhs = bf16[2,3,8192,256] fusion(p0), kind=kCustom, calls=lhs
  rhs = bf16[2,3,256,512] fusion(p1), kind=kCustom, calls=rhs

  ROOT dot = bf16[2,3,8192,512] dot(lhs, rhs),
    lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
    lhs_contracting_dims={3}, rhs_contracting_dims={2}
}

ENTRY main {
  p0 = bf16[2,3,8192,256] parameter(0)
  p1 = bf16[2,3,256,512] parameter(1)
  ROOT fusion = bf16[2,3,8192,512] fusion(p0, p1), kind=kCustom, calls=dot
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kSupportedFusionHlo));

  TF_ASSERT_OK_AND_ASSIGN(TilingSpecification tiling_specification,
                          TilingSpecificationForModule(module.get()));

  TF_EXPECT_OK(TransposedDotTiledHloSchedule::Create(tiling_specification));
}

TEST_F(TransposedDotTiledHloScheduleTest,
       CanNotBeCreatedForFusionRootedInNonDot) {
  constexpr absl::string_view kUnsupportedNonDotHlo = R"(
dot {
  ROOT p0 = bf16[128] parameter(0)
}

ENTRY main {
  p0 = bf16[128] parameter(0)
  ROOT fusion = bf16[128] fusion(p0), kind=kCustom, calls=dot
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kUnsupportedNonDotHlo));

  TF_ASSERT_OK_AND_ASSIGN(TilingSpecification tiling_specification,
                          TilingSpecificationForModule(module.get()));
  EXPECT_THAT(TransposedDotTiledHloSchedule::Create(tiling_specification),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expects its root to be a dot")));
}

TEST_F(TransposedDotTiledHloScheduleTest, CanNotBeCreatedForMultiDotFusion) {
  constexpr absl::string_view kUnsupportedMultiDotHlo = R"(
lhs {
  ROOT p0 = bf16[64,64] parameter(0)
}

nested_lhs {
  ROOT p0 = bf16[64,64] parameter(0)
}

nested_rhs {
  ROOT p0 = bf16[64,64] parameter(0)
}

rhs {
  p0 = bf16[64,64] parameter(0)
  lhs = bf16[64,64] fusion(p0), kind=kCustom, calls=nested_lhs
  rhs = bf16[64,64] fusion(p0), kind=kCustom, calls=nested_rhs
  ROOT dot = bf16[64,64] dot(lhs, rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
}

dot {
  p0 = bf16[64,64] parameter(0)
  p1 = bf16[64,64] parameter(1)

  lhs = bf16[64,64] fusion(p0), kind=kCustom, calls=lhs
  rhs = bf16[64,64] fusion(p1), kind=kCustom, calls=rhs
  ROOT dot = bf16[64,64] dot(lhs, rhs),
    lhs_batch_dims={}, rhs_batch_dims={},
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
}

ENTRY main {
  p0 = bf16[64,64] parameter(0)
  p1 = bf16[64,64] parameter(1)
  ROOT fusion = bf16[64,64] fusion(p0, p1), kind=kCustom, calls=dot
})";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(kUnsupportedMultiDotHlo));

  TF_ASSERT_OK_AND_ASSIGN(TilingSpecification tiling_specification,
                          TilingSpecificationForModule(module.get()));

  EXPECT_THAT(
      TransposedDotTiledHloSchedule::Create(tiling_specification),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "only supported for "
                   "TilingSpecifications specifying tiling for a single dot")));
}

TEST_F(TransposedDotTiledHloScheduleTest,
       CanNotBeCreatedForDotWithMultipleNonContractingDimensions) {
  constexpr absl::string_view kSupportedFusionHlo = R"(
lhs {
  ROOT p0 = bf16[2,8192,256] parameter(0)
}

rhs {
  ROOT p0 = bf16[256,512] parameter(0)
}

dot {
  p0 = bf16[2,8192,256] parameter(0)
  p1 = bf16[256,512] parameter(1)

  lhs = bf16[2,8192,256] fusion(p0), kind=kCustom, calls=lhs
  rhs = bf16[256,512] fusion(p1), kind=kCustom, calls=rhs

  ROOT dot = bf16[2,8192,512] dot(lhs, rhs),
    lhs_contracting_dims={2}, rhs_contracting_dims={0}
}

ENTRY main {
  p0 = bf16[2,8192,256] parameter(0)
  p1 = bf16[256,512] parameter(1)
  ROOT fusion = bf16[2,8192,512] fusion(p0, p1), kind=kCustom, calls=dot
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kSupportedFusionHlo));

  TF_ASSERT_OK_AND_ASSIGN(TilingSpecification tiling_specification,
                          TilingSpecificationForModule(module.get()));

  EXPECT_THAT(TransposedDotTiledHloSchedule::Create(tiling_specification),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("only supported for dot instructions with a "
                                 "single non-contracting dimension")));
}

TEST_F(
    TransposedDotTiledHloScheduleTest,
    SchedulingProducesTransposedIterationSpaceOfDotNonContractingDimensions) {
  constexpr absl::string_view kSupportedFusionHlo = R"(
lhs {
  ROOT p0 = bf16[1,3,1024,256] parameter(0)
}

rhs {
  ROOT p0 = bf16[1,3,256,512] parameter(0)
}

dot {
  p0 = bf16[1,3,1024,256] parameter(0)
  p1 = bf16[1,3,256,512] parameter(1)

  lhs = bf16[1,3,1024,256] fusion(p0), kind=kCustom, calls=lhs
  rhs = bf16[1,3,256,512] fusion(p1), kind=kCustom, calls=rhs

  ROOT dot = bf16[1,3,1024,512] dot(lhs, rhs),
    lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
    lhs_contracting_dims={3}, rhs_contracting_dims={2}
}

ENTRY main {
  p0 = bf16[1,3,1024,256] parameter(0)
  p1 = bf16[1,3,256,512] parameter(1)
  ROOT fusion = bf16[1,3,1024,512] fusion(p0, p1), kind=kCustom, calls=dot
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kSupportedFusionHlo));

  TF_ASSERT_OK_AND_ASSIGN(TilingSpecification tiling_specification,
                          TilingSpecificationForModule(module.get()));

  MajorToMinorTiledHloSchedule major_to_minor_scheduler;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TransposedDotTiledHloSchedule> transposed_scheduler,
      TransposedDotTiledHloSchedule::Create(tiling_specification));

  IndexingMap offsets_indexing = *ParseIndexingMap(R"(
      (d0, d1, d2, d3, d4) -> (d1, d2, d3, d4),
      domain: d0 in [0, 0], d1 in [0, 1], d2 in [0, 1], d3 in [0, 3],
              d4 in [0, 7])",
                                                   &mlir_context_);

  std::vector<DimensionInfo> iteration_space;
  iteration_space.reserve(offsets_indexing.GetDimVarsCount());
  int64_t linear_iteration_space_size = 1;
  for (const auto& [i, dim_var] :
       llvm::enumerate(offsets_indexing.GetDimVars())) {
    int64_t bound = offsets_indexing.GetDimensionBound(i).upper + 1;
    iteration_space.push_back({/*dimension_id=*/static_cast<int64_t>(i),
                               /*dimension_size=*/bound});
    linear_iteration_space_size *= bound;
  }

  TF_ASSERT_OK_AND_ASSIGN(
      IndexingMap major_to_minor_scheduled_indexing,
      major_to_minor_scheduler.Schedule(offsets_indexing, iteration_space,
                                        &mlir_context_));
  TF_ASSERT_OK_AND_ASSIGN(
      IndexingMap transposed_scheduled_indexing,
      transposed_scheduler->Schedule(offsets_indexing, iteration_space,
                                     &mlir_context_));

  int64_t m_bound = iteration_space[3].dimension_size;
  int64_t n_bound = iteration_space[4].dimension_size;

  // Check that evaluating the scheduled indexing map yields a transposed
  // schedule across the non-contracting dimensions of the dot.
  for (int64_t i = 0; i < linear_iteration_space_size; ++i) {
    mlir::AffineExpr pid = mlir::getAffineConstantExpr(i, &mlir_context_);
    llvm::SmallVector<int64_t> major_to_minor_indices =
        major_to_minor_scheduled_indexing.Evaluate({pid}, {});
    llvm::SmallVector<int64_t> transposed_indices =
        transposed_scheduled_indexing.Evaluate({pid}, {});

    // The first two dimensions should be identical.
    EXPECT_EQ(major_to_minor_indices[0], transposed_indices[0]);
    EXPECT_EQ(major_to_minor_indices[1], transposed_indices[1]);
    // The last two dimensions should be transposed!
    int64_t expected_major_to_minor_m_index = (i / n_bound) % m_bound;
    int64_t expected_major_to_minor_n_index = i % n_bound;
    EXPECT_EQ(major_to_minor_indices[2], expected_major_to_minor_m_index);
    EXPECT_EQ(major_to_minor_indices[3], expected_major_to_minor_n_index);

    int64_t expected_transposed_m_index = i % m_bound;
    int64_t expected_transposed_n_index = (i / m_bound) % n_bound;
    EXPECT_EQ(transposed_indices[2], expected_transposed_m_index);
    EXPECT_EQ(transposed_indices[3], expected_transposed_n_index);
  }
}

}  // namespace
}  // namespace xla
