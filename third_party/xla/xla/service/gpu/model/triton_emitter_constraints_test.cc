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

#include "xla/service/gpu/model/triton_emitter_constraints.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::absl::Status;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::HasSubstr;

class TritonEmitterConstraintsTest : public HloHardwareIndependentTestBase {
 public:
  TritonEmitterConstraintsTest() = default;
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
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

class VerifyTritonConstraintsTest : public HloHardwareIndependentTestBase {
 protected:
  VerifyTritonConstraintsTest() = default;

  Status CheckTiling(HloModule* module, absl::Span<const int64_t> tile_sizes) {
    HloInstruction* root = module->entry_computation()->root_instruction();
    std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
        HloFusionAdaptor::ForInstruction(root);
    ASSIGN_OR_RETURN(
        std::unique_ptr<experimental::TilingSpace> tiling_space,
        experimental::TilingSpace::Create(*fusion_adaptor, &mlir_context_));
    RETURN_IF_ERROR(tiling_space->AssignTileSizes(tile_sizes));
    ASSIGN_OR_RETURN(experimental::TiledHloComputation tiled_comp,
                     experimental::TiledHloComputation::Tile(
                         *fusion_adaptor, std::move(tiling_space)));
    Decision decision =
        experimental::VerifyTritonConstraints(tiled_comp, device_description_);
    if (!decision) {
      return absl::InvalidArgumentError(decision.Explain());
    }
    return absl::OkStatus();
  }

  mlir::MLIRContext mlir_context_;
  se::DeviceDescription device_description_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
};

TEST_F(VerifyTritonConstraintsTest, TooBigTileSizesConstraintIsEnforced) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"hlo(
HloModule m

max_computation {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0, param_1)
}

fused_computation {
  param_0 = f32[8192,50304] parameter(0)
  constant = f32[] constant(-inf)
  reduce = f32[8192] reduce(param_0, constant), dimensions={1},
      to_apply=max_computation
  broadcast = f32[8192,50304] broadcast(reduce), dimensions={0}
  ROOT subtract = f32[8192,50304] subtract(param_0, broadcast)
}

ENTRY entry_computation {
  param_0 = f32[8192,50304] parameter(0)
  ROOT fusion = f32[8192,50304] fusion(param_0), kind=kCustom,
    calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)hlo"));

  EXPECT_OK(CheckTiling(module.get(), {8, 128, 8192}));
  EXPECT_THAT(
      CheckTiling(module.get(), {18, 50304, 8192}),
      StatusIs(_,
               HasSubstr("padded tile size product that exceeds the maximum")));
  EXPECT_THAT(
      CheckTiling(module.get(), {1024, 1, 8192}),
      StatusIs(_,
               HasSubstr("padded tile size product that exceeds the maximum")));
}

TEST_F(VerifyTritonConstraintsTest, DotOperandSizeConstraintIsEnforced) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"hlo(
HloModule m

fused_computation {
  p0 = f32[4,5376] parameter(0)
  p1 = f32[32768,5376] parameter(1)
  ROOT dot = f32[4,32768] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY entry_computation {
  param_0 = f32[4,5376] parameter(0)
  param_1 =  f32[32768,5376] parameter(1)
  ROOT fusion = f32[4,32768] fusion(param_0, param_1), kind=kCustom,
    calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)hlo"));

  EXPECT_OK(CheckTiling(module.get(), {4, 4, 4}));
  EXPECT_THAT(CheckTiling(module.get(), {512, 4, 4}),
              StatusIs(_, HasSubstr("exceeds the maximum MMA dimension size")));
}

TEST_F(VerifyTritonConstraintsTest, TooManyBlocksConstraintIsEnforced) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"hlo(
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
  ROOT fusion = f32[65536,65536] fusion(param_0), kind=kCustom,
    calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)hlo"));

  EXPECT_OK(CheckTiling(module.get(), {128, 128}));
  EXPECT_THAT(CheckTiling(module.get(), {1, 1}),
              StatusIs(_, HasSubstr("Number of blocks exceeds the device")));
}

TEST_F(VerifyTritonConstraintsTest, FusionHasValidTileSizes) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"hlo(
HloModule m

fused_computation {
  param_0 = f32[36] parameter(0)
  abs = f32[36] abs(param_0)
  ROOT reshape = f32[6,6] reshape(abs)
}

ENTRY entry_computation {
  param_0 = f32[36] parameter(0)
  ROOT fusion = f32[6,6] fusion(param_0), kind=kCustom,
    calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
})hlo"));

  EXPECT_THAT(CheckTiling(module.get(), {1, 3}),
              StatusIs(_, HasSubstr("neither a power of 2 nor equal")));
  EXPECT_OK(CheckTiling(module.get(), {1, 6}));
  EXPECT_OK(CheckTiling(module.get(), {2, 1}));
  EXPECT_OK(CheckTiling(module.get(), {1, 8}));
  EXPECT_OK(CheckTiling(module.get(), {1, 4}));
}

TEST_F(VerifyTritonConstraintsTest, MultiOutputFusionHasPowerOfTwoTileSizes) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"hlo(
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
    calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
})hlo"));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_unsupported_enable_triton_multi_output_fusion(true);
  EXPECT_THAT(CheckTiling(module.get(), {1, 12, 12}),
              StatusIs(_, HasSubstr("neither a power of 2 nor equal")));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
