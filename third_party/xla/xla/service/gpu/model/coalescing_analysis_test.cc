/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/model/coalescing_analysis.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;

class CoalescingTest : public HloHardwareIndependentTestBase {
 public:
  std::vector<bool> IsReadCoalescedPerOperand(absl::string_view hlo_string) {
    auto module = ParseAndReturnVerifiedModule(hlo_string).value();
    HloInstruction* root = module->entry_computation()->root_instruction();
    return IsReadCoalescedPerOperand(root);
  }

  std::vector<bool> IsReadCoalescedPerOperand(const HloInstruction* root) {
    auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
    auto analysis = HloFusionAnalysis::Create(*root, device_info_);
    auto emitter = GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis});
    auto fusion = dynamic_cast<KernelFusionInterface*>(emitter.get());
    EXPECT_NE(fusion, nullptr);

    CoalescingAnalysis coalescing_analysis(root, root->operands(), analysis,
                                           fusion, &mlir_context_,
                                           /*use_heuristic=*/false);

    std::vector<bool> results;
    for (const HloInstruction* operand : root->operands()) {
      results.push_back(coalescing_analysis.IsReadCoalesced(operand));
    }
    return results;
  }

  bool IsReadCoalescedHeuristic(absl::string_view hlo_string) {
    auto module = ParseAndReturnVerifiedModule(hlo_string).value();
    HloInstruction* root = module->entry_computation()->root_instruction();
    auto analysis = HloFusionAnalysis::Create(*root, device_info_);
    return xla::gpu::IsReadCoalescedHeuristic(
        analysis.GetEmitterFusionKind(), device_info_, root->operand(0), root);
  }

 protected:
  stream_executor::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  mlir::MLIRContext mlir_context_;
};

TEST_F(CoalescingTest, IdentityLayout) {
  absl::string_view ir = R"(
    HloModule m
    fusion {
      p0 = f32[100, 200] parameter(0)
      p1 = f32[100, 200] parameter(1)
      ROOT adthread_x = f32[100, 200] add(p0, p1)
    }
    ENTRY e {
      p0 = f32[100, 200] parameter(0)
      p1 = f32[100, 200] parameter(1)
      ROOT fusion = f32[100, 200] fusion(p0, p1), kind=kInput, calls=fusion
    }
  )";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x)
  // Operand 2: (thread_x) -> (thread_x)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true, true));
}

TEST_F(CoalescingTest, RhsTransposedLayout) {
  absl::string_view ir = R"(
    HloModule m
    fusion {
      p0 = f32[100, 200]{1, 0} parameter(0)
      p1 = f32[100, 200]{0, 1} parameter(1)
      ROOT exp = f32[100, 200]{1, 0} add(p0, p1)
    }
    ENTRY e {
      p0 = f32[100, 200]{1, 0} parameter(0)
      p1 = f32[100, 200]{0, 1} parameter(1)
      ROOT fusion = f32[100, 200]{1, 0} fusion(p0, p1), kind=kInput, calls=fusion
    }
  )";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x)
  // Operand 2: (thread_x) -> (thread_x * 100)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true, false));
}

TEST_F(CoalescingTest, OutputTransposedLayout) {
  absl::string_view ir = R"(
    HloModule m
    fusion {
      p0 = f32[100, 200]{1, 0} parameter(0)
      p1 = f32[100, 200]{1, 0} parameter(1)
      ROOT exp = f32[100, 200]{0, 1} add(p0, p1)
    }
    ENTRY e {
      p0 = f32[100, 200]{1, 0} parameter(0)
      p1 = f32[100, 200]{1, 0} parameter(1)
      ROOT fusion = f32[100, 200]{0, 1} fusion(p0, p1), kind=kInput, calls=fusion
    }
  )";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x * 200)
  // Operand 2: (thread_x) -> (thread_x * 200)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(false, false));
}

TEST_F(CoalescingTest, OutputAndLhsTransposedLayout) {
  absl::string_view ir = R"(
    HloModule m
    fusion {
      p0 = f32[100, 200]{1, 0} parameter(0)
      p1 = f32[100, 200]{0, 1} parameter(1)
      ROOT add = f32[100, 200]{1, 0} add(p0, p1)
    }
    ENTRY e {
      p0 = f32[100, 200]{1, 0} parameter(0)
      p1 = f32[100, 200]{0, 1} parameter(1)
      ROOT fusion = f32[100, 200]{1, 0} fusion(p0, p1), kind=kInput, calls=fusion
    }
  )";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x)
  // Operand 2: (thread_x) -> (thread_x * 100)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true, false));
}

TEST_F(CoalescingTest, Transpose) {
  absl::string_view ir = R"(
    HloModule module

    fusion {
      %input = f32[1, 6400, 32] parameter(0)
      ROOT transpose = f32[1, 32, 6400] transpose(%input), dimensions={0, 2, 1}
    }

    ENTRY entry {
      %input = f32[1, 6400, 32] parameter(0)
      ROOT %fusion = f32[1, 32, 6400] fusion(%input), kind=kLoop, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1:  (thread_x)[s0] -> (thread_x + s0 * 128) for s0 in [0, 7]
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true));
}

TEST_F(CoalescingTest, TransposeOfBroadcastHeuristic) {
  absl::string_view ir = R"(
    HloModule module

    fusion {
      input = f32[1, 32, 6400] parameter(0)
      ROOT slice = f32[1, 32, 100] slice(input), slice={[0:1:1], [0:32:1], [0:6400:64]}
    }

    ENTRY entry {
      p0 = f32[32] parameter(0)
      broadcast = f32[1, 6400, 32] broadcast(p0), dimensions={2}
      transpose = f32[1, 32, 6400] transpose(broadcast), dimensions={0, 2, 1}
      ROOT %fusion = f32[1, 32, 100] fusion(transpose), kind=kLoop, calls=fusion
  })";
  EXPECT_TRUE(IsReadCoalescedHeuristic(ir));
}

TEST_F(CoalescingTest, TransposeOfIotaHeuristic) {
  absl::string_view ir = R"(
    HloModule module

    fusion {
      p0 = f32[32, 100, 64] parameter(0)
      ROOT slice = f32[32, 100, 1] slice(p0), slice={[0:32:1], [0:100:1], [0:1:1]}
    }

    ENTRY entry {
      iota = f32[100, 64, 32] iota(), iota_dimension=1
      transpose = f32[32, 100, 64] transpose(iota), dimensions={2, 0, 1}
      ROOT %fusion = f32[32, 100, 1] fusion(transpose), kind=kLoop, calls=fusion
  })";
  EXPECT_TRUE(IsReadCoalescedHeuristic(ir));
}

TEST_F(CoalescingTest, TransposeOfAddHeuristic) {
  absl::string_view ir = R"(
    HloModule module

    fusion {
      p0 = f32[32, 100, 64] parameter(0)
      ROOT slice = f32[32, 100, 1] slice(p0), slice={[0:32:1], [0:100:1], [0:1:1]}
    }

    ENTRY entry {
      input = f32[100, 64, 32] parameter(0)
      add = f32[100, 64, 32] add(input, input)
      transpose = f32[32, 100, 64] transpose(add), dimensions={2, 0, 1}
      ROOT %fusion = f32[32, 100, 1] fusion(transpose), kind=kLoop, calls=fusion
  })";
  EXPECT_FALSE(IsReadCoalescedHeuristic(ir));
}

TEST_F(CoalescingTest, TransposeOnlyOuterDims) {
  absl::string_view ir = R"(
    HloModule module

    fusion {
      %input = f32[100, 32, 64] parameter(0)
      ROOT transpose = f32[32, 100, 64] transpose(%input), dimensions={1, 0, 2}
    }

    ENTRY entry {
      %input = f32[100, 32, 64] parameter(0)
      ROOT %fusion = f32[32, 100, 64] fusion(%input), kind=kLoop, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1:
  //   (thread_x) -> (thread_x * 4 + s0 + (thread_x floordiv 16) * 1984)
  //   for s0 in [0, 3]
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true));
}

TEST_F(CoalescingTest, PadOp) {
  absl::string_view ir = R"(
    HloModule module
    fusion {
      p0 = f32[997, 436] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[1024, 512] pad(p0, p1), padding=10_17x24_52
    }
    ENTRY entry {
      p0 = f32[997, 436] parameter(0)
      p1 = f32[] parameter(1)
      ROOT %fusion = f32[1024, 512] fusion(p0, p1), kind=kLoop, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x)[s0] -> (thread_x * 4 + s0 - 4384)
  //   for s0 in [0, 3] and thread_x * 4 + s0 in [24, 459]
  // Operand 2: (thread_x) -> ()
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true, true));
}

TEST_F(CoalescingTest, RowReduction) {
  absl::string_view ir = R"(
    HloModule module
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %input = f32[100,64,512] parameter(0)
      %c0 = f32[] constant(0)
      ROOT reduce = f32[100,64] reduce(%input, %c0), dimensions={2}, to_apply=add
    }
    ENTRY entry {
      %input = f32[100,64,512] parameter(0)
      ROOT %fusion = f32[100,64] fusion(%input), kind=kInput, calls=fusion
    })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x)[s0] -> (thread_x + s0 * 32) for s0 in [0, 15]
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true));
}

TEST_F(CoalescingTest, MultiRowReduction) {
  absl::string_view ir = R"(
    HloModule module
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %input = f32[100,64,4] parameter(0)
      %c0 = f32[] constant(0)
      ROOT reduce = f32[100,64] reduce(%input, %c0), dimensions={2}, to_apply=add
    }
    ENTRY entry {
      %input = f32[100,64,4] parameter(0)
      ROOT %fusion = f32[100,64] fusion(%input), kind=kInput, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true));
}

TEST_F(CoalescingTest, ColumnReduction) {
  absl::string_view ir = R"(
    HloModule module
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %input = f32[100,64,32] parameter(0)
      %c0 = f32[] constant(0)
      ROOT reduce = f32[100,32] reduce(%input, %c0),
        dimensions={1}, to_apply=add
    }
    ENTRY entry {
      %input = f32[100,64,32] parameter(0)
      ROOT %fusion = f32[100,32] fusion(%input), kind=kInput, calls=fusion
    })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x)[s0] -> (thread_x + s0 * 1024) for s0 in [0, 1]
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true));
}

TEST_F(CoalescingTest, VariadicReduceViaLoopEmitter) {
  absl::string_view ir = R"(
    HloModule module
    max {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      max01 = s32[] maximum(p0, p1)
      max23 = s32[] maximum(p2, p3)
      ROOT max = (s32[], s32[]) tuple(max01, max23)
    }
    fusion {
      p0 = s32 [5696,10,4] parameter(0)
      p1 = s32 [5696,10,4] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT reduce = (s32[5696,4], s32[5696,4]) reduce(s32[5696,10,4] p0,
        s32[5696,10,4] p1, s32[] p2, s32[] p3), dimensions={1}, to_apply=max
    }
    ENTRY entry {
      p0 = s32 [5696,10,4] parameter(0)
      p1 = s32 [5696,10,4] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT f = (s32[5696,4], s32[5696,4]) fusion(p0, p1, p2, p3),
          kind=kInput, calls=fusion
    })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operands 1, 2: (d0)[s0] -> ((d0 floordiv 4) * 40 + d0 mod 4 + s0 * 4)
  //  for s0 in [0, 9].
  EXPECT_THAT(IsReadCoalescedPerOperand(ir),
              ElementsAre(false, false, true, true));
}

TEST_F(CoalescingTest, VariadicReduceViaReductionEmitter) {
  absl::string_view ir = R"(
    HloModule module
    max {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      max01 = s32[] maximum(p0, p1)
      max23 = s32[] maximum(p2, p3)
      ROOT max = (s32[], s32[]) tuple(max01, max23)
    }
    fusion {
      p0 = s32[32,40] parameter(0)
      p1 = s32[32,40] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT reduce = (s32[32], s32[32])
        reduce(s32[32,40] p0, s32[32,40] p1, s32[] p2, s32[] p3),
        dimensions={1}, to_apply=max
    }
    ENTRY entry {
      p0 = s32[32,40] parameter(0)
      p1 = s32[32,40] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT f = (s32[32], s32[32]) fusion(p0, p1, p2, p3),
          kind=kInput, calls=fusion
    })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operands 1, 2: (d0)[s0] -> (d0 + s0 * 32)
  //  for s0 in [0, 1] and d0 + s0 * 32 in [0, 39].
  EXPECT_THAT(IsReadCoalescedPerOperand(ir),
              ElementsAre(true, true, true, true));
}

TEST_F(CoalescingTest, Gather) {
  absl::string_view ir = R"(
    HloModule module
    fusion {
      operand = f32[33, 76, 70] parameter(0)
      indices = s32[1806, 2] parameter(1)
      ROOT gather = f32[1806, 7, 8, 4] gather(operand, indices),
        offset_dims={1,2,3}, collapsed_slice_dims={}, start_index_map={0,1},
        index_vector_dim=1, slice_sizes={7,8,4}
    }
    ENTRY entry {
      p0 = f32[33, 76, 70] parameter(0)
      p1 = s32[1806, 2] parameter(1)
      ROOT %fusion = f32[1806, 7, 8, 4] fusion(p0, p1), kind=kLoop, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (d0)[s0] -> (
  //  (d0 floordiv 8) * 5320 + (d0 mod 8) * 70 + s0 * 70 + 34) for s0 in [0, 3]
  // Operand 2: (d0)[s0] -> (s0)
  //  for s0 in [0, 1].
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(false, true));
}

TEST_F(CoalescingTest, DynamicSlice) {
  absl::string_view ir = R"(
    HloModule module
    fusion {
      %src = s32[2,2,258] parameter(0)
      %of1 = s32[] parameter(1)
      %of2 = s32[] parameter(2)
      %of3 = s32[] parameter(3)
      ROOT %ds = s32[1,2,32] dynamic-slice(s32[2,2,258] %src,
        s32[] %of1, s32[] %of2, s32[] %of3),
        dynamic_slice_sizes={1, 2, 32}
    }
    ENTRY entry {
      %p0 = s32[2,2,258] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = s32[] parameter(2)
      %p3 = s32[] parameter(3)
      ROOT %fusion = s32[1,2,32] fusion(p0, p1, p2, p3), kind=kLoop, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (d0) -> (d0).
  EXPECT_THAT(IsReadCoalescedPerOperand(ir),
              ElementsAre(true, true, true, true));
}

TEST_F(CoalescingTest, UnusedParameter) {
  Shape shape = ShapeUtil::MakeShape(F32, {100000});

  auto module = std::make_unique<HloModule>("m", HloModuleConfig{});
  HloComputation::Builder b("b");
  auto p0 = b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  auto p1 = b.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));

  HloComputation::Builder sub_builder("subcomp");
  HloInstruction* p0f = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0f"));
  // p1f is not used.
  HloInstruction* p1f = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "p1f"));
  ASSERT_NE(p1f, nullptr);
  sub_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0f));

  HloComputation* subcomp = module->AddEmbeddedComputation(sub_builder.Build());
  auto fusion = HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kLoop, {p0, p1}, subcomp);
  b.AddInstruction(std::move(fusion));
  module->AddEntryComputation(b.Build());

  EXPECT_THAT(IsReadCoalescedPerOperand(
                  module->entry_computation()->root_instruction()),
              ElementsAre(true, true));
}

TEST_F(CoalescingTest, Param) {
  absl::string_view ir = R"(
    HloModule module
    fusion {
      %p0 = u32[48,2,1280] parameter(0)
      %p1 = u32[48,1,1280] parameter(1)
      %p2 = u32[48,1,1280] parameter(2)
      %concat = u32[48,2,1280] concatenate(u32[48,1,1280] %p1,
                                           u32[48,1,1280] %p2), dimensions={1}
      ROOT %shift = u32[48,2,1280] shift-right-logical(
        u32[48,2,1280] %concat, u32[48,2,1280] %p0)
    }
    ENTRY entry {
      %p0 = u32[48,2,1280] parameter(0)
      %p1 = u32[48,1,1280] parameter(1)
      %p2 = u32[48,1,1280] parameter(2)
      ROOT %fusion = u32[48,2,1280] fusion(p0, p1, p2), kind=kLoop, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true, true, true));
}

class CoalescingForTiledHloTest : public CoalescingTest {
 public:
  std::vector<bool> IsTiledReadCoalescedPerOperand(
      const HloInstruction* root, absl::Span<int64_t const> tile_sizes) {
    auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);

    SymbolicTileAnalysis symbolic_tile_analysis =
        std::get<SymbolicTileAnalysis>(SymbolicTileAnalysis::AnalyzeFusion(
            *fusion_adaptor, &mlir_context_));

    TiledHloComputation tiled_hlo_computation =
        *symbolic_tile_analysis.ComputeTiledHloInstructions(
            Tiling({{root, FlatTiling(tile_sizes.begin(), tile_sizes.end())}}),
            /*constraints_are_known_satisfied=*/true,
            /*compute_all_tile_offset_indexing_maps=*/true);

    const TiledHloInstruction* tiled_hlo_root =
        tiled_hlo_computation.GetRoots()[0];
    std::vector<bool> result;
    for (const TiledHloInstruction* operand : tiled_hlo_root->operands()) {
      result.push_back(IsTiledReadCoalescedHeuristic(*operand, device_info_));
    }
    return result;
  }

  std::vector<double> EffectiveBandwidthUtilizationRatePerOperand(
      const HloInstruction* root, absl::Span<int64_t const> tile_sizes) {
    auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);

    SymbolicTileAnalysis symbolic_tile_analysis =
        std::get<SymbolicTileAnalysis>(SymbolicTileAnalysis::AnalyzeFusion(
            *fusion_adaptor, &mlir_context_));

    TiledHloComputation tiled_hlo_computation =
        *symbolic_tile_analysis.ComputeTiledHloInstructions(
            Tiling({{root, FlatTiling(tile_sizes.begin(), tile_sizes.end())}}),
            /*constraints_are_known_satisfied=*/true,
            /*compute_all_tile_offset_indexing_maps=*/true);

    const TiledHloInstruction* tiled_hlo_root =
        tiled_hlo_computation.GetRoots()[0];
    std::vector<double> result;
    for (const TiledHloInstruction* operand : tiled_hlo_root->operands()) {
      result.push_back(BandwidthUtilizationRateHeuristicForTiledMemoryAccess(
          *operand, device_info_));
    }
    return result;
  }
};

TEST_F(CoalescingForTiledHloTest, TiledReadCoalescedHeuristic_Transpose) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY main {
  p0 = f32[2048, 48] parameter(0)
  ROOT transpose = f32[48, 2048] transpose(p0), dimensions={1, 0}
})"));

  const HloInstruction* root = module->entry_computation()->root_instruction();

  // The operand is not coalesced because the tile has stride 48.
  EXPECT_THAT(IsTiledReadCoalescedPerOperand(root, {1, 2048}),
              ElementsAre(false));

  // The operand is coalesced because we read 48 contiguous elements.
  EXPECT_THAT(IsTiledReadCoalescedPerOperand(root, {48, 32}),
              ElementsAre(true));
}

TEST_F(CoalescingForTiledHloTest,
       TiledReadCoalescedHeuristic_MaskingIsHandledCorrectly) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY main {
  p0 = f32[2048, 12] parameter(0)
  ROOT transpose = f32[12, 2048] transpose(p0), dimensions={1, 0}
})"));

  const HloInstruction* root = module->entry_computation()->root_instruction();

  constexpr int kNumBytesPerParamRow = 12 * 4;

  // The transaction size can be configured in different ways, and the minimum
  // possible value on A100 is 32 bytes---which would make this test fail.
  // Ensure that the transaction size is configured to be large enough.
  ASSERT_GT(device_info_.dram_to_l2_transaction_size_bytes(),
            kNumBytesPerParamRow);

  // The operand is coalesced because we read 4 * 12 = 48 contiguous elements
  // (though the tile contains 64 elements due to the mask).
  EXPECT_THAT(IsTiledReadCoalescedPerOperand(root, {16, 4}), ElementsAre(true));

  // The mask should be ignored when checking whether reads are coalesced.
  EXPECT_THAT(IsTiledReadCoalescedPerOperand(root, {1024, 1}),
              ElementsAre(false));
}

TEST_F(CoalescingForTiledHloTest, RhsTransposedLayout) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY main {
  p0 = f32[256, 512]{1,0} parameter(0)
  p1 = f32[256, 512]{0,1} parameter(1)
  ROOT add = f32[256, 512]{1,0} add(p0, p1)
})"));

  const HloInstruction* root = module->entry_computation()->root_instruction();

  constexpr int kExpectedDramToL2TransactionSize = 64;
  ASSERT_EQ(device_info_.dram_to_l2_transaction_size_bytes(),
            kExpectedDramToL2TransactionSize);

  EXPECT_THAT(IsTiledReadCoalescedPerOperand(root, {1, 16}),
              ElementsAre(true, false));
  EXPECT_THAT(IsTiledReadCoalescedPerOperand(root, {16, 1}),
              ElementsAre(false, true));
  EXPECT_THAT(IsTiledReadCoalescedPerOperand(root, {16, 16}),
              ElementsAre(true, true));
  EXPECT_THAT(IsTiledReadCoalescedPerOperand(root, {8, 8}),
              ElementsAre(false, false));
}

TEST_F(CoalescingForTiledHloTest, SmallDataTypes) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY main {
  p0 = s8[256, 512] parameter(0)
  p1 = s8[256, 512] parameter(1)
  ROOT add = s8[256, 512] add(p0, p1)
})"));

  const HloInstruction* root = module->entry_computation()->root_instruction();

  constexpr int kExpectedDramToL2TransactionSize = 64;
  ASSERT_EQ(device_info_.dram_to_l2_transaction_size_bytes(),
            kExpectedDramToL2TransactionSize);

  // To be coalesced, a contiguous chunk of memory load should be at least
  // kExpectedDramToL2TransactionSize bytes long.
  EXPECT_THAT(IsTiledReadCoalescedPerOperand(root, {16, 16}),
              ElementsAre(false, false));
  EXPECT_THAT(IsTiledReadCoalescedPerOperand(root, {16, 32}),
              ElementsAre(false, false));
  EXPECT_THAT(IsTiledReadCoalescedPerOperand(root, {16, 64}),
              ElementsAre(true, true));
  EXPECT_THAT(IsTiledReadCoalescedPerOperand(root, {16, 128}),
              ElementsAre(true, true));
}

TEST_F(
    CoalescingForTiledHloTest,
    EffectiveBandwidthUtilizationRateIsComputedCorrectlyForTiledMemoryAccess) {  // NOLINT(whitespace/line_length)
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY main {
  p0 = s8[256, 16] parameter(0)
  ROOT convert = s8[256, 16] convert(p0)
})"));

  const HloInstruction* root = module->entry_computation()->root_instruction();

  // Note: the tests below rely strongly on this value for the transaction size.
  // If the transaction size is changed, the tests will need to be updated.
  constexpr int kExpectedDramToL2TransactionSize = 64;
  ASSERT_EQ(device_info_.dram_to_l2_transaction_size_bytes(),
            kExpectedDramToL2TransactionSize);

  // By reading only one byte at a time, we expect to exploit exactly
  // 1 / kExpectedDramToL2TransactionSize of the bandwidth.
  EXPECT_THAT(EffectiveBandwidthUtilizationRatePerOperand(root, {1, 1}),
              ElementsAre(1.0 / kExpectedDramToL2TransactionSize));

  // Reading one full row won't cut it; by reading 16 bytes at a time, we expect
  // to exploit exactly 16 / kExpectedDramToL2TransactionSize of the bandwidth.
  EXPECT_THAT(EffectiveBandwidthUtilizationRatePerOperand(root, {1, 16}),
              ElementsAre(16.0 / kExpectedDramToL2TransactionSize));

  // Reading 4 rows at a time will allow us to exploit 100% of the bandwidth.
  EXPECT_THAT(EffectiveBandwidthUtilizationRatePerOperand(root, {4, 16}),
              ElementsAre(1.0));

  // Reading 8 rows at a time will allow us to exploit 100% of the bandwidth.
  EXPECT_THAT(EffectiveBandwidthUtilizationRatePerOperand(root, {8, 16}),
              ElementsAre(1.0));

  // Reading 6 rows at a time will however only allow us to exploit 75% of the
  // bandwidth; the first four rows are read fully coalesced, but the last two
  // rows use only half of the transaction size---i.e. 3/4 of the transactions
  // are coalesced.
  EXPECT_THAT(EffectiveBandwidthUtilizationRatePerOperand(root, {6, 16}),
              ElementsAre(0.75));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
