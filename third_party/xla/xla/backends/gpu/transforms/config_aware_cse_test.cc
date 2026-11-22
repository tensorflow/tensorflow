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

#include "xla/backends/gpu/transforms/config_aware_cse.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/xtile/xtile_config.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

using ConfigAwareCSETest = HloHardwareIndependentTestBase;

TEST_F(ConfigAwareCSETest, DropsTileSizesOfMergedReducesInScanFusion) {
  // The sequential dimensions are the scan dimension and one per reduce.
  constexpr absl::string_view kHloText = R"(
scan_combiner {
  in = f32[8]{0} parameter(0)
  carry = f32[8]{0} parameter(1)
  add = f32[8]{0} add(carry, in)
  ROOT t = (f32[8]{0}, f32[8]{0}) tuple(add, add)
}

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

fused_scan {
  input = f32[8,16]{1,0} parameter(0)
  init = f32[8]{0} parameter(1)
  zero = f32[] constant(0)
  reduce.0 = f32[8]{0} reduce(input, zero), dimensions={1}, to_apply=add
  reduce.1 = f32[8]{0} reduce(input, zero), dimensions={1}, to_apply=add
  sum = f32[8]{0} add(reduce.0, reduce.1)
  bcast = f32[8,16]{1,0} broadcast(sum), dimensions={0}
  scan = (f32[8,16]{1,0}, f32[8]{0}) scan(bcast, init), dimensions={1},
    num_carries=1, to_apply=scan_combiner, is_associative=true
  ROOT gte = f32[8,16]{1,0} get-tuple-element(scan), index=0
}

ENTRY entry {
  input = f32[8,16]{1,0} parameter(0)
  init = f32[8]{0} parameter(1)
  ROOT fusion = f32[8,16]{1,0} fusion(input, init), kind=kCustom,
    calls=fused_scan,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
        "sequential_tile_sizes":["4","8","8"],
        "num_warps":4,"num_ctas":1,"num_stages":1}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  mlir::MLIRContext mlir_context;

  EXPECT_THAT(ConfigAwareCSE(&mlir_context).Run(module.get()),
              IsOkAndHolds(true));

  const HloInstruction* fusion =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(fusion->fused_instructions_computation()->instruction_count(), 8);
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_THAT(config.fusion_backend_config()
                  .block_level_fusion_config()
                  .sequential_tile_sizes(),
              ElementsAre(4, 8));
}

TEST_F(ConfigAwareCSETest, KeepsTileSizesIfCSEChangesNothing) {
  constexpr absl::string_view kHloText = R"(
scan_combiner {
  in = f32[8]{0} parameter(0)
  carry = f32[8]{0} parameter(1)
  add = f32[8]{0} add(carry, in)
  ROOT t = (f32[8]{0}, f32[8]{0}) tuple(add, add)
}

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

fused_scan {
  input = f32[8,16]{1,0} parameter(0)
  init = f32[8]{0} parameter(1)
  zero = f32[] constant(0)
  reduce = f32[8]{0} reduce(input, zero), dimensions={1}, to_apply=add
  bcast = f32[8,16]{1,0} broadcast(reduce), dimensions={0}
  scan = (f32[8,16]{1,0}, f32[8]{0}) scan(bcast, init), dimensions={1},
    num_carries=1, to_apply=scan_combiner, is_associative=true
  ROOT gte = f32[8,16]{1,0} get-tuple-element(scan), index=0
}

ENTRY entry {
  input = f32[8,16]{1,0} parameter(0)
  init = f32[8]{0} parameter(1)
  ROOT fusion = f32[8,16]{1,0} fusion(input, init), kind=kCustom,
    calls=fused_scan,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
        "sequential_tile_sizes":["4","8"],
        "num_warps":4,"num_ctas":1,"num_stages":1}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  mlir::MLIRContext mlir_context;

  EXPECT_THAT(ConfigAwareCSE(&mlir_context).Run(module.get()),
              IsOkAndHolds(false));

  ASSERT_OK_AND_ASSIGN(GpuBackendConfig config,
                       module->entry_computation()
                           ->root_instruction()
                           ->backend_config<GpuBackendConfig>());
  EXPECT_THAT(config.fusion_backend_config()
                  .block_level_fusion_config()
                  .sequential_tile_sizes(),
              ElementsAre(4, 8));
}

// Different fusion computations can have instructions with the same local IDs.
TEST_F(ConfigAwareCSETest, KeepsTileSizesOfEachFusion) {
  constexpr absl::string_view kHloText = R"(
add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

fused_a {
  p = f32[8,16]{1,0} parameter(0)
  zero = f32[] constant(0)
  reduce.0 = f32[8]{0} reduce(p, zero), dimensions={1}, to_apply=add
  reduce.1 = f32[8]{0} reduce(p, zero), dimensions={1}, to_apply=add
  ROOT sum = f32[8]{0} add(reduce.0, reduce.1)
}

fused_b {
  p = f32[8,16]{1,0} parameter(0)
  zero = f32[] constant(0)
  reduce.0 = f32[8]{0} reduce(p, zero), dimensions={1}, to_apply=add
  reduce.1 = f32[8]{0} reduce(p, zero), dimensions={1}, to_apply=add
  ROOT sum = f32[8]{0} add(reduce.0, reduce.1)
}

ENTRY entry {
  p = f32[8,16]{1,0} parameter(0)
  a = f32[8]{0} fusion(p), kind=kCustom, calls=fused_a,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
        "sequential_tile_sizes":["4","4"],
        "num_warps":4,"num_ctas":1,"num_stages":1}}}
  b = f32[8]{0} fusion(p), kind=kCustom, calls=fused_b,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
        "sequential_tile_sizes":["8","8"],
        "num_warps":4,"num_ctas":1,"num_stages":1}}}
  ROOT t = (f32[8]{0}, f32[8]{0}) tuple(a, b)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  mlir::MLIRContext mlir_context;

  EXPECT_THAT(ConfigAwareCSE(&mlir_context).Run(module.get()),
              IsOkAndHolds(true));

  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig config_a,
                       root->operand(0)->backend_config<GpuBackendConfig>());
  EXPECT_THAT(config_a.fusion_backend_config()
                  .block_level_fusion_config()
                  .sequential_tile_sizes(),
              ElementsAre(4));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig config_b,
                       root->operand(1)->backend_config<GpuBackendConfig>());
  EXPECT_THAT(config_b.fusion_backend_config()
                  .block_level_fusion_config()
                  .sequential_tile_sizes(),
              ElementsAre(8));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
