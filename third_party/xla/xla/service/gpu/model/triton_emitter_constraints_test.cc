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
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/tiled_hlo.h"
#include "xla/codegen/tiling/tiling_space.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::absl::Status;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::HasSubstr;

class VerifyTritonConstraintsTest : public HloHardwareIndependentTestBase {
 protected:
  VerifyTritonConstraintsTest() = default;

  Status CheckTiling(HloModule* module, absl::Span<const int64_t> tile_sizes) {
    HloInstruction* root = module->entry_computation()->root_instruction();
    std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
        HloFusionAdaptor::ForInstruction(root);
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<experimental::TilingSpace> tiling_space,
        experimental::TilingSpace::Create(*fusion_adaptor, &mlir_context_));
    ABSL_RETURN_IF_ERROR(tiling_space->AssignTileSizes(tile_sizes));
    ABSL_ASSIGN_OR_RETURN(experimental::TiledHloComputation tiled_comp,
                     experimental::TiledHloComputation::Tile(
                         *fusion_adaptor, std::move(tiling_space)));
    tiled_comp.Simplify();
    tiled_comp.SortInstructionsPostOrder();
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

  // The biggest tile in the program has 8 * 8192 = 65536 elements.
  EXPECT_OK(CheckTiling(module.get(), {8, 128, 8192}));
  // The biggest tile in the program is 18 * 50304 = 905472 elements which is
  // smaller than the limit of 1048576, but since Triton requires all tile sizes
  // to be a power of 2, the actual tile will be 32 * 65536 = 2097152 elements.
  EXPECT_THAT(
      CheckTiling(module.get(), {18, 50304, 8192}),
      StatusIs(_,
               HasSubstr("padded tile size product that exceeds the maximum")));
  // The reduce operand tile will be 1024 * 8192 = 8388608 elements, which is
  // larger than the limit of 1048576.
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
  // Having any tile larger than 256 is not allowed for dots.
  EXPECT_THAT(CheckTiling(module.get(), {512, 4, 4}),
              StatusIs(_, HasSubstr("exceeds the maximum MMA dimension size")));
}

TEST_F(VerifyTritonConstraintsTest, TransposeSharedMemoryConstraintIsEnforced) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"hlo(
HloModule m

fused_computation {
  param_0 = f32[1024,1024] parameter(0)
  ROOT transpose = f32[1024,1024] transpose(param_0), dimensions={1,0}
}

ENTRY entry_computation {
  param_0 = f32[1024,1024] parameter(0)
  ROOT fusion = f32[1024,1024] fusion(param_0), kind=kCustom,
    calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)hlo"));

  // The RTX A6000 test device has 99 * 1024 = 101376 bytes of opt-in shared
  // memory. A [128, 128] f32 transpose operand tile requires
  // 128 * 128 * 4 = 65536 bytes, which fits.
  EXPECT_OK(CheckTiling(module.get(), {128, 128}));

  // A [256, 128] f32 transpose operand tile requires
  // 256 * 128 * 4 = 131072 bytes, which exceeds the shared memory limit. The
  // tile is otherwise valid (32768 elements is below the tensor size limit,
  // tile sizes are powers of 2, and the number of blocks fits on the grid), so
  // it is the transpose shared memory constraint that rejects it.
  EXPECT_THAT(CheckTiling(module.get(), {256, 128}),
              StatusIs(_, HasSubstr("exceeds the device limit")));
}

TEST_F(VerifyTritonConstraintsTest, NonTransposeIsNotChargedToSharedMemory) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"hlo(
HloModule m

fused_computation {
  param_0 = f32[1024,1024] parameter(0)
  ROOT log = f32[1024,1024] log(param_0)
}

ENTRY entry_computation {
  param_0 = f32[1024,1024] parameter(0)
  ROOT fusion = f32[1024,1024] fusion(param_0), kind=kCustom,
    calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)hlo"));

  // A [256, 128] f32 tile requires 256 * 128 * 4 = 131072 bytes, which would
  // exceed the shared memory limit if it were charged to shared memory. Because
  // there is no transpose op, the shared memory constraint does not apply and
  // the tiling is accepted.
  EXPECT_OK(CheckTiling(module.get(), {256, 128}));
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

  // This tiling will require (65536 * 65536) / (128 * 128) = 262144 blocks.
  EXPECT_OK(CheckTiling(module.get(), {128, 128}));
  // This would require to run 65536 * 65536 = 4294967296 blocks that is larger
  // than the hardware limit of 2^32 - 1.
  EXPECT_THAT(CheckTiling(module.get(), {1, 1}),
              StatusIs(_, HasSubstr("exceeds the device grid")));
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

  // (1,3) is a theoretically valid tiling for this fusion, but it does not pass
  // the triton specific condition that all tile sizes are either powers of 2,
  // or equal to the dimension size.
  EXPECT_THAT(CheckTiling(module.get(), {1, 3}),
              StatusIs(_, HasSubstr("neither a power of 2 nor equal")));
  // However if we capture the last dimension fully, it should be valid.
  EXPECT_OK(CheckTiling(module.get(), {1, 6}));
  // Also powers of 2 are valid.
  EXPECT_OK(CheckTiling(module.get(), {2, 1}));
  EXPECT_OK(CheckTiling(module.get(), {1, 8}));
  EXPECT_OK(CheckTiling(module.get(), {1, 4}));
}

TEST_F(VerifyTritonConstraintsTest, MultiOutputFusionHasPowerOfTwoTileSizes) {
  GTEST_SKIP() << "Arbitrary multi-output fusions are currently not supported "
                  "in TilingSpace (b/502910372).";
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
  // (1,) is a theoretically valid tiling for this multi-output fusion, but the
  // propagated tile size of (1,12) for the extra output does not pass the
  // condition that all tile sizes are powers of 2. This can result in different
  // paddings for the different roots being used, which can cause problems if
  // buffers are shared.
  EXPECT_THAT(CheckTiling(module.get(), {1, 12, 12}),
              StatusIs(_, HasSubstr("neither a power of 2 nor equal")));
}

class VerifySubsetOfTritonConstraintsTest
    : public HloHardwareIndependentTestBase {
 protected:
  mlir::MLIRContext mlir_context_;
  se::DeviceDescription device_description_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
};

TEST_F(VerifySubsetOfTritonConstraintsTest, ValidTilingPasses) {
  auto space = std::make_unique<experimental::TilingSpace>();
  space->AppendDimension(
      nullptr, 0, 1024,
      experimental::TilingSpace::DimensionSemantics::kParallel);

  Decision decision = experimental::VerifySubsetOfTritonConstraints(
      {64}, *space, device_description_);
  EXPECT_TRUE(decision.IsAllowed());
}

TEST_F(VerifySubsetOfTritonConstraintsTest, RejectsExcessiveElements) {
  auto space = std::make_unique<experimental::TilingSpace>();
  space->AppendDimension(
      nullptr, 0, 1024,
      experimental::TilingSpace::DimensionSemantics::kParallel);
  space->AppendDimension(
      nullptr, 1, 2048,
      experimental::TilingSpace::DimensionSemantics::kParallel);

  // Cumulative product 1024 * 2048 = 2097152 exceeds 1048576 elements:
  Decision decision = experimental::VerifySubsetOfTritonConstraints(
      {1024, 2048}, *space, device_description_);
  EXPECT_TRUE(decision.IsForbidden());
  EXPECT_THAT(decision.Explain(),
              HasSubstr("Padded tile size product exceeds the maximum number "
                        "of elements of a Triton tensor"));
}

TEST_F(VerifySubsetOfTritonConstraintsTest, RejectsDotTileExceedingMMALimit) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule m
    ENTRY e {
      p0 = f32[16,512] parameter(0)
      p1 = f32[512,16] parameter(1)
      ROOT dot = f32[16,16] dot(p0, p1),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )"));
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->root_instruction());
  ASSERT_OK_AND_ASSIGN(auto space, experimental::TilingSpace::Create(
                                       *fusion_adaptor, &mlir_context_));

  // Tile sizes [16, 16, 512] has contracting tile 512 > 256 (MMA limit).
  Decision invalid_decision = experimental::VerifySubsetOfTritonConstraints(
      {16, 16, 512}, *space, device_description_);
  EXPECT_TRUE(invalid_decision.IsForbidden());
  EXPECT_THAT(invalid_decision.Explain(),
              HasSubstr("exceeds the maximum MMA dimension size"));

  // Contracting tile 256 (<= 256) passes.
  Decision valid_decision = experimental::VerifySubsetOfTritonConstraints(
      {16, 16, 256}, *space, device_description_);
  EXPECT_TRUE(valid_decision.IsAllowed()) << valid_decision.Explain();
}

TEST_F(VerifySubsetOfTritonConstraintsTest, MultiOutputFusionEvaluatedPerRoot) {
  auto space = std::make_unique<experimental::TilingSpace>();
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1000000] parameter(0)
      root0 = f32[1000000] exponential(p0)
      root1 = f32[1000000] negate(p0)
      ROOT tuple = (f32[1000000], f32[1000000]) tuple(root0, root1)
    }
  )"));
  const HloInstruction* root0 =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* root1 =
      module->entry_computation()->root_instruction()->operand(1);

  space->AppendDimension(
      root0, 0, 1000000,
      experimental::TilingSpace::DimensionSemantics::kParallel);
  space->AppendDimension(
      root1, 0, 1000000,
      experimental::TilingSpace::DimensionSemantics::kParallel);

  Decision decision = experimental::VerifySubsetOfTritonConstraints(
      {1000000, 1000000}, *space, device_description_);
  EXPECT_TRUE(decision.IsAllowed()) << decision.Explain();
}

}  // namespace
}  // namespace gpu
}  // namespace xla
