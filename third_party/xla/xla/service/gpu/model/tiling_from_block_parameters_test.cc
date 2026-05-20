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

#include "xla/service/gpu/model/tiling_from_block_parameters.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

class TilingFromBlockParametersTest : public HloHardwareIndependentTestBase {
 public:
  TilingFromBlockParametersTest() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  std::optional<SymbolicTileAnalysis> TryAnalyzeModule(HloModule* module) {
    SymbolicTileAnalysisOrError analysis_or_error =
        SymbolicTileAnalysis::AnalyzeComputation(
            *module->entry_computation()
                 ->root_instruction()
                 ->fused_instructions_computation(),
            &mlir_context_);

    if (std::holds_alternative<SymbolicTileAnalysis>(analysis_or_error)) {
      return std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));
    }
    return std::nullopt;
  }

  mlir::MLIRContext mlir_context_;
};

TEST_F(TilingFromBlockParametersTest, GeneratesTilingForSimpleMap) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"hlo(
HloModule m

fused_computation {
  param_0 = f32[128,256] parameter(0)
  ROOT log = f32[128,256] log(param_0)
}

ENTRY entry_computation {
  param_0 = f32[128,256] parameter(0)
  ROOT fusion = f32[128,256] fusion(param_0), kind=kCustom,
   calls=fused_computation
}
)hlo"));

  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 32}};

  TF_ASSERT_OK_AND_ASSIGN(
      Tiling tiling,
      TilingFromAnnotatedFusion(*analysis, block_level_parameters));

  const HloInstruction* log = module->entry_computation()
                                  ->root_instruction()
                                  ->fused_instructions_computation()
                                  ->root_instruction();

  EXPECT_THAT(tiling.TileSizesForInstruction(log),
              IsOkAndHolds(ElementsAre(16, 32)));
}

TEST_F(TilingFromBlockParametersTest, GeneratesTilingForDot) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"hlo(
HloModule m

fused_computation {
  p0 = f32[128,128] parameter(0)
  p1 = f32[128,128] parameter(1)
  ROOT dot = f32[128,128] dot(p0, p1),
   lhs_contracting_dims={1}, rhs_contracting_dims={0},
   backend_config={"sizes":["32"]}
}

ENTRY entry_computation {
  param_0 = f32[128,128] parameter(0)
  param_1 = f32[128,128] parameter(1)
  ROOT fusion = f32[128,128] fusion(param_0, param_1), kind=kCustom, calls=fused_computation
}
)hlo"));

  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 16}};

  TF_ASSERT_OK_AND_ASSIGN(
      Tiling tiling,
      TilingFromAnnotatedFusion(*analysis, block_level_parameters));

  const HloInstruction* dot = module->entry_computation()
                                  ->root_instruction()
                                  ->fused_instructions_computation()
                                  ->root_instruction();

  EXPECT_THAT(tiling.TileSizesForInstruction(dot),
              IsOkAndHolds(ElementsAre(32, 16, 16)));
}

TEST_F(TilingFromBlockParametersTest, GeneratesTilingForDotWithTilingOverride) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"hlo(
HloModule m

fused_computation {
  p0 = f32[128,128] parameter(0)
  p1 = f32[128,128] parameter(1)
  ROOT dot = f32[128,128] dot(p0, p1),
   lhs_contracting_dims={1}, rhs_contracting_dims={0}}

ENTRY entry_computation {
  param_0 = f32[128,128] parameter(0)
  param_1 = f32[128,128] parameter(1)
  ROOT fusion = f32[128,128] fusion(param_0, param_1), kind=kCustom, calls=fused_computation
}
)hlo"));

  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{16, 16}};

  Tile tile_override;
  tile_override.add_sizes(64);

  TF_ASSERT_OK_AND_ASSIGN(
      Tiling tiling, TilingFromAnnotatedFusion(
                         *analysis, block_level_parameters, &tile_override));

  const HloInstruction* dot = module->entry_computation()
                                  ->root_instruction()
                                  ->fused_instructions_computation()
                                  ->root_instruction();

  EXPECT_THAT(tiling.TileSizesForInstruction(dot),
              IsOkAndHolds(ElementsAre(64, 16, 16)));
}

class GetTileTilingSpaceConcreteSizesTest
    : public HloHardwareIndependentTestBase {
 public:
  GetTileTilingSpaceConcreteSizesTest() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  absl::StatusOr<llvm::SmallVector<int64_t>> ComputeConcreteTileSizesOfFusion(
      const HloInstruction* fusion) {
    TF_RET_CHECK(fusion->opcode() == HloOpcode::kFusion) << fusion->ToString();
    TF_ASSIGN_OR_RETURN(auto backend_config,
                        fusion->backend_config<GpuBackendConfig>());
    BlockLevelParameters block_level_parameters =
        BlockLevelParameters::FromBlockLevelFusionConfig(
            backend_config.fusion_backend_config().block_level_fusion_config());
    auto fusion_adaptor = HloFusionAdaptor::ForInstruction(fusion);
    auto tiling_space =
        experimental::TilingSpace::Create(*fusion_adaptor, &mlir_context_);
    return GetTilingSpaceConcreteSizes(*tiling_space, block_level_parameters);
  }

  mlir::MLIRContext mlir_context_;
};

TEST_F(GetTileTilingSpaceConcreteSizesTest, DotWithBackendConfig) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"hlo(
f {
  p0 = f32[64,128] parameter(0)
  p1 = f32[128,256] parameter(1)
  ROOT dot = f32[64,256] dot(p0, p1),
   lhs_contracting_dims={1}, rhs_contracting_dims={0},
   backend_config={sizes:[11]}
}

ENTRY entry {
  param_0 = f32[64,128] parameter(0)
  param_1 = f32[128,256] parameter(1)
  ROOT fusion = f32[64,256] fusion(param_0, param_1),
    kind=kCustom, calls=f,
    backend_config={fusion_backend_config:{block_level_fusion_config:{output_tiles:[{
      sizes:[3, 8]
    }]}}}
}
)hlo"));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(llvm::SmallVector<int64_t> tile_sizes,
                       ComputeConcreteTileSizesOfFusion(root));
  EXPECT_THAT(tile_sizes, ElementsAre(3, 8, 11));
}

TEST_F(GetTileTilingSpaceConcreteSizesTest, DotWithoutBackendConfig) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"hlo(
f {
  p0 = f32[64,128] parameter(0)
  p1 = f32[128,256] parameter(1)
  ROOT dot = f32[64,256] dot(p0, p1),
   lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  param_0 = f32[64,128] parameter(0)
  param_1 = f32[128,256] parameter(1)
  ROOT fusion = f32[64,256] fusion(param_0, param_1),
    kind=kCustom, calls=f,
    backend_config={fusion_backend_config:{block_level_fusion_config:{output_tiles:[{
      sizes:[3, 8]
    }]}}}
}
)hlo"));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(llvm::SmallVector<int64_t> tile_sizes,
                       ComputeConcreteTileSizesOfFusion(root));
  EXPECT_THAT(tile_sizes, ElementsAre(3, 8, 128));
}

TEST_F(GetTileTilingSpaceConcreteSizesTest, ReductionWithTwoReductionDims) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"hlo(
add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

f {
  p0 = f32[64,128,256] parameter(0)
  c0 = f32[] constant(0)
  ROOT reduce = f32[64] reduce(p0, c0), dimensions={1,2}, to_apply=add
}

ENTRY entry {
  param_0 = f32[64,128,256] parameter(0)
  ROOT fusion = f32[64] fusion(param_0),
    kind=kInput, calls=f,
    backend_config={fusion_backend_config:{block_level_fusion_config:{output_tiles:[{
      sizes:[16]
    }]}}}
}
)hlo"));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(llvm::SmallVector<int64_t> tile_sizes,
                       ComputeConcreteTileSizesOfFusion(root));
  EXPECT_THAT(tile_sizes, ElementsAre(16, 128, 256));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
