/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/codegen/tiling/tiled_hlo_computation.h"

#include <memory>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiled_hlo_schedule.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/instruction_fusion.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::SizeIs;

class TiledHloComputationTest : public HloHardwareIndependentTestBase {
 public:
  TiledHloComputationTest() { RegisterSymbolicExprStorage(&mlir_context_); }

 protected:
  mlir::MLIRContext mlir_context_;
  TiledHloScheduleBuilder default_schedule_builder_ =
      CreateMajorToMinorTiledHloSchedule;

  absl::StatusOr<TiledHloComputation> TileRootFusionComputation(
      VerifiedHloModule& module, FlatTiling flat_tiling) {
    SymbolicTileAnalysisOrError analysis_or_error =
        SymbolicTileAnalysis::AnalyzeComputation(
            *module.entry_computation()
                 ->root_instruction()
                 ->fused_instructions_computation(),
            &mlir_context_, nullptr);
    if (!std::holds_alternative<SymbolicTileAnalysis>(analysis_or_error)) {
      return absl::InternalError(
          std::get<FusionDecision>(analysis_or_error).Explain());
    }
    const HloInstruction* fusion_root =
        module.entry_computation()->root_instruction()->fused_expression_root();
    return std::get<SymbolicTileAnalysis>(analysis_or_error)
        .ComputeTiledComputation(
            Tiling({{fusion_root, flat_tiling}}), default_schedule_builder_,
            /*constraints_are_known_satisfied=*/false,
            /*compute_all_tile_offset_indexing_maps=*/true);
  }
};

TEST_F(TiledHloComputationTest, Dot) {
  absl::string_view hlo_text = R"(
  HloModule m

  dot_fusion {
    p0 = f32[8,128] parameter(0)
    p1 = f32[128,32] parameter(1)
    ROOT dot = f32[8,32] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    p0 = f32[8,128] parameter(0)
    p1 = f32[128,32] parameter(1)
    ROOT fusion = f32[8,32] fusion(p0, p1), kind=kCustom, calls=dot_fusion
  })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(
      TiledHloComputation tiled_hlo_computation,
      TileRootFusionComputation(*module, FlatTiling({4, 8, 4})));
  EXPECT_EQ(tiled_hlo_computation.ToString(),
            R"(dot.tile_0 = dot(p0.1.tile_0, p1.1.tile_0)
  hlo: %dot = f32[8,32]{1,0} dot(%p0, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  tile_sizes: (8, 4)
  tile_strides: (1, 1)
  tile_offsets_indexing: (pid_0) -> (0, pid_0 * 4), domain: pid_0 in [0, 7]
  operands:
    %p0.1 = parameter()
    %p1.1 = parameter()
  region sizes: (2)
  region.0 {
    p0.1.tile_0 = parameter()
      hlo: %p0.1 = f32[8,128]{1,0} parameter(0)
      tile_sizes: (8, 4)
      tile_strides: (1, 1)
      tile_offsets_indexing: (pid_0) -> (0, (pid_0 mod 32) * 4), domain: pid_0 in [0, 255]
    p1.1.tile_0 = parameter()
      hlo: %p1.1 = f32[128,32]{1,0} parameter(1)
      tile_sizes: (4, 4)
      tile_strides: (1, 1)
      tile_offsets_indexing: (pid_0) -> ((pid_0 mod 32) * 4, (pid_0 floordiv 32) * 4), domain: pid_0 in [0, 255]
  })");
  EXPECT_THAT(tiled_hlo_computation.num_output_tiles_per_dim(),
              ElementsAre(1, 1, 8));
  EXPECT_EQ(tiled_hlo_computation.num_output_tiles(), 8);
  EXPECT_THAT(tiled_hlo_computation.roots(), SizeIs(1));
}

TEST_F(TiledHloComputationTest, Elementwise) {
  absl::string_view hlo_text = R"(
  HloModule m

  add_fusion {
    p0 = f32[128,128] parameter(0)
    p1 = f32[128,128] parameter(1)
    ROOT add = f32[128,128] add(p0, p1)
  }

  ENTRY main {
    p0 = f32[128,128] parameter(0)
    p1 = f32[128,128] parameter(1)
    ROOT fusion = f32[128,128] fusion(p0, p1), kind=kLoop, calls=add_fusion
  })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(
      TiledHloComputation tiled_hlo_computation,
      TileRootFusionComputation(*module, FlatTiling({16, 32})));
  EXPECT_EQ(tiled_hlo_computation.ToString(),
            R"(p0.1.tile_0 = parameter()
  hlo: %p0.1 = f32[128,128]{1,0} parameter(0)
  tile_sizes: (16, 32)
  tile_strides: (1, 1)
  tile_offsets_indexing: (pid_0) -> ((pid_0 floordiv 4) * 16, (pid_0 mod 4) * 32), domain: pid_0 in [0, 31]
p1.1.tile_0 = parameter()
  hlo: %p1.1 = f32[128,128]{1,0} parameter(1)
  tile_sizes: (16, 32)
  tile_strides: (1, 1)
  tile_offsets_indexing: (pid_0) -> ((pid_0 floordiv 4) * 16, (pid_0 mod 4) * 32), domain: pid_0 in [0, 31]
add.tile_0 = add(p0.1.tile_0, p1.1.tile_0)
  hlo: %add = f32[128,128]{1,0} add(%p0, %p1)
  tile_sizes: (16, 32)
  tile_strides: (1, 1)
  tile_offsets_indexing: (pid_0) -> ((pid_0 floordiv 4) * 16, (pid_0 mod 4) * 32), domain: pid_0 in [0, 31]
  operands:
    %p0.1 = parameter()
    %p1.1 = parameter()
)");
  EXPECT_THAT(tiled_hlo_computation.num_output_tiles_per_dim(),
              ElementsAre(8, 4));
  EXPECT_EQ(tiled_hlo_computation.num_output_tiles(), 32);
  EXPECT_THAT(tiled_hlo_computation.roots(), SizeIs(1));
}

TEST_F(TiledHloComputationTest, SoftmaxDiamond) {
  absl::string_view hlo_text = R"(
HloModule m

max_computation {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0, param_1)
}

add_computation {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT add = f32[] add(param_0, param_1)
}

fused_computation {
  param_0 = f32[8192,50304] parameter(0)
  bitcast = f32[4,2048,50304] bitcast(param_0)
  constant = f32[] constant(-inf)
  reduce = f32[8192] reduce(param_0, constant), dimensions={1}, to_apply=max_computation
  bitcast.1 = f32[4,2048] bitcast(reduce)
  broadcast = f32[4,2048,50304] broadcast(bitcast.1), dimensions={0,1}
  subtract = f32[4,2048,50304] subtract(bitcast, broadcast)
  exponential = f32[4,2048,50304] exponential(subtract)
  constant.1 = f32[] constant(0)
  reduce.1 = f32[4,2048] reduce(exponential, constant.1), dimensions={2}, to_apply=add_computation
  log = f32[4,2048] log(reduce.1)
  broadcast.1 = f32[4,2048,50304] broadcast(log), dimensions={0,1}
  ROOT subtract.1 = f32[4,2048,50304] subtract(subtract, broadcast.1)
}

ENTRY entry_computation {
  param_0 = f32[8192,50304] parameter(0)
  ROOT fusion = f32[4,2048,50304] fusion(param_0), kind=kCustom, calls=fused_computation, backend_config={"fusion_backend_config":{"kind":"__triton"}}
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(
      TiledHloComputation tiled_hlo_computation,
      TileRootFusionComputation(*module, FlatTiling({1, 64, 128})));
  EXPECT_THAT(
      tiled_hlo_computation.ToString(),
      AllOf(HasSubstr("param_0.3.tile_0"), HasSubstr("param_0.3.tile_1")));
  EXPECT_THAT(tiled_hlo_computation.num_output_tiles_per_dim(),
              ElementsAre(4, 32, 393));
  EXPECT_EQ(tiled_hlo_computation.num_output_tiles(), 4 * 32 * 393);
  EXPECT_THAT(tiled_hlo_computation.roots(), SizeIs(1));
}

}  // namespace
}  // namespace xla::gpu
