/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/cost_model/triton_emitter_constraints.h"

#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class TritonEmitterConstraintsTest : public HloHardwareIndependentTestBase {
 public:
  TritonEmitterConstraintsTest() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }
  std::optional<SymbolicTileAnalysis> TryAnalyzeModule(
      HloModule* module, bool with_triton_emitter_specific_constraints = true) {
    EmitterSpecificConstraintsBuilder constraints_builder = nullptr;

    if (with_triton_emitter_specific_constraints) {
      constraints_builder =
          TritonEmitterConstraints::GetBuilder(device_description_);
    }

    SymbolicTileAnalysisOrError analysis_or_error =
        SymbolicTileAnalysis::AnalyzeComputation(
            *module->entry_computation()
                 ->root_instruction()
                 ->fused_instructions_computation(),
            &mlir_context_, constraints_builder);

    if (std::holds_alternative<SymbolicTileAnalysis>(analysis_or_error)) {
      return std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));
    }
    VLOG(1) << "Cannot analyze module: "
            << std::get<FusionDecision>(analysis_or_error).Explain();
    return std::nullopt;
  }

  mlir::MLIRContext mlir_context_;
  se::DeviceDescription device_description_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
};

TEST_F(TritonEmitterConstraintsTest, TooBigTileSizesConstraintIsEnforced) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

max_computation {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0, param_1)
}

fused_computation {
  param_0 = f32[8192,50304] parameter(0)
  constant = f32[] constant(-inf)
  reduce = f32[8192] reduce(param_0, constant), dimensions={1}, to_apply=max_computation
  broadcast = f32[8192,50304] broadcast(reduce), dimensions={0}
  ROOT subtract = f32[8192,50304] subtract(param_0, broadcast)
}

ENTRY entry_computation {
  param_0 = f32[8192,50304] parameter(0)
  ROOT fusion = f32[8192,50304] fusion(param_0), kind=kCustom, calls=fused_computation, backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)"));

  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const HloInstruction* fusion_root =
      module->entry_computation()->root_instruction()->fused_expression_root();

  // The biggest tile in the program has 8 * 65536 = 524288 elements.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({8, 128})}})),
              absl_testing::IsOkAndHolds(true));

  // The biggest tile in the program is 18 * 50304 = 905472 elements which is
  // smaller than the limit of 1048576, but since Triton requires all tile sizes
  // to be a power of 2, the actual tile will be 32 * 65536 = 2097152 elements.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({18, 50304})}})),
              absl_testing::IsOkAndHolds(false));

  // Because of reduce, we need to load full rows from param_0 and the load tile
  // will be 1024 * 65536 = 67108864 elements, that is larger than the limit of
  // 1048576.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({1024, 1})}})),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(TritonEmitterConstraintsTest, DotOperandSizeConstraintIsEnforced) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

fused_computation {
  p0 = f32[4,5376] parameter(0)
  p1 = f32[32768,5376] parameter(1)
  ROOT dot = f32[4,32768] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY entry_computation {
  param_0 = f32[4,5376] parameter(0)
  param_1 =  f32[32768,5376] parameter(1)
  ROOT fusion = f32[4,32768] fusion(param_0, param_1), kind=kCustom, calls=fused_computation, backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)"));

  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const HloInstruction* fusion_root =
      module->entry_computation()->root_instruction()->fused_expression_root();

  EXPECT_THAT(analysis->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({4, 4, 4})}})),
              absl_testing::IsOkAndHolds(true));

  // Having any tile larger than 256 is not allowed for dots.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({512, 4, 4})}})),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(TritonEmitterConstraintsTest, TooManyBlocksConstraintIsEnforced) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

max_computation {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0, param_1)
}

fused_computation {
  param_0 = f32[65536,65536] parameter(0)
  ROOT log = f32[65536,65536] log(param_0)
}

ENTRY entry_computation {
  param_0 = f32[65536,65536] parameter(0)
  ROOT fusion = f32[65536,65536] fusion(param_0), kind=kCustom, calls=fused_computation, backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)"));

  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());
  const HloInstruction* fusion_root =
      module->entry_computation()->root_instruction()->fused_expression_root();

  // This tiling will require (65536 * 65536) / (128 * 128) = 262144 blocks.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({128, 128})}})),
              absl_testing::IsOkAndHolds(true));

  // This would require to run 65538 * 65538 = 4294967296 blocks that is larger
  // than the hardware limit of 2^32 - 1.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({1, 1})}})),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(TritonEmitterConstraintsTest, FusionHasValidTileSizes) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

fused_computation {
  param_0 = f32[36] parameter(0)
  abs = f32[36] abs(param_0)
  ROOT reshape = f32[6,6] reshape(abs)
}

ENTRY entry_computation {
  param_0 = f32[36] parameter(0)
  ROOT fusion = f32[6,6] fusion(param_0), kind=kCustom,
    calls=fused_computation, backend_config={"fusion_backend_config":{"kind":"__triton"}}
})"));
  std::optional<SymbolicTileAnalysis> analysis_without_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/false);
  ASSERT_TRUE(analysis_without_triton_constraints.has_value());
  const HloInstruction* fusion_root =
      module->entry_computation()->root_instruction()->fused_expression_root();

  // (1,3) is a theoretically valid tiling for this fusion, so
  // SymbolicTileAnalysis should allow it.
  EXPECT_THAT(analysis_without_triton_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({1, 3})}})),
              absl_testing::IsOkAndHolds(true));

  std::optional<SymbolicTileAnalysis> analysis_with_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/true);

  ASSERT_TRUE(analysis_with_triton_constraints.has_value());

  // (1,3) is a theoretically valid tiling for this fusion, but it does not pass
  // the triton specific condition that all tile sizes are either powers of 2,
  // or equal to the dimension size.
  EXPECT_THAT(analysis_with_triton_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({1, 3})}})),
              absl_testing::IsOkAndHolds(false));

  // However if we capture the last dimension fully, it should be valid.
  EXPECT_THAT(analysis_with_triton_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({1, 6})}})),
              absl_testing::IsOkAndHolds(true));

  // Also powers of 2 are valid.
  EXPECT_THAT(analysis_with_triton_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({2, 1})}})),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(analysis_with_triton_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({1, 8})}})),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(analysis_with_triton_constraints->ParametersSatisfyConstraints(
                  Tiling({{fusion_root, FlatTiling({1, 4})}})),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(TritonEmitterConstraintsTest, MultiOutputFusionHasPowerOfTwoTileSizes) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

fused_computation {
  param_0 = f32[36] parameter(0)
  abs = f32[36] abs(param_0)
  reshape = f32[3,12] reshape(abs)
  zero = f32[] constant(0)
  reduce = f32[3] reduce(reshape, zero), to_apply=add, dimensions={1}
  ROOT tuple = (f32[3], f32[36]) tuple(reduce, abs)
}

ENTRY entry_computation {
  param_0 = f32[36] parameter(0)
  ROOT fusion = (f32[3], f32[36]) fusion(param_0), kind=kCustom,
    calls=fused_computation, backend_config={"fusion_backend_config":{"kind":"__triton"}}
})"));
  std::optional<SymbolicTileAnalysis> analysis_without_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/false);
  ASSERT_TRUE(analysis_without_triton_constraints.has_value());
  const HloInstruction* reduce_root = module->entry_computation()
                                          ->root_instruction()
                                          ->fused_expression_root()
                                          ->operand(0);

  Tiling tiling({{reduce_root, FlatTiling({1})}});

  // (1,) is a theoretically valid tiling for this multi-output fusion, so
  // SymbolicTileAnalysis should allow it.
  EXPECT_THAT(
      analysis_without_triton_constraints->ParametersSatisfyConstraints(tiling),
      absl_testing::IsOkAndHolds(true));

  std::optional<SymbolicTileAnalysis> analysis_with_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/true);

  ASSERT_TRUE(analysis_with_triton_constraints.has_value());

  // (1,) is a theoretically valid tiling for this multi-output fusion, but the
  // propagated tile size of (1,12) for the extra output does not pass the
  // condition that all tile sizes are powers of 2. This can result in different
  // paddings for the different roots being used, which can cause problems if
  // buffers are shared.
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints(tiling),
      absl_testing::IsOkAndHolds(false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
