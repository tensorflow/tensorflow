/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/codegen/tiling/experimental/tiled_hlo.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/test_utils.h"
#include "xla/codegen/tiling/experimental/tile_propagation.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/util.h"

namespace xla::gpu::experimental {
namespace {

using ::mlir::MLIRContext;
using ::testing::Contains;
using ::testing::Eq;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::UnorderedElementsAre;

class TiledHloTest : public HloHardwareIndependentTestBase {
 public:
  TiledHloTest() { RegisterSymbolicExprStorage(&mlir_context_); }

  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_string);
    CHECK_OK(module_or);
    module_ = std::move(module_or.value());
    return module_->entry_computation()->root_instruction();
  }

  MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

TEST_F(TiledHloTest, TestPrinting) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10,30] parameter(0)
      ROOT broadcast = f32[10,20,30] broadcast(p0), dimensions={0,2}
    }
  )");
  auto tiling_space = TilingSpace::Create(
      *HloFusionAdaptor::ForInstruction(root), &mlir_context_);
  ASSERT_OK_AND_ASSIGN(
      Tiles tiled_operands,
      PropagateTileToInput(
          *tiling_space, *root,
          GetTestTile(*tiling_space, root->shape().dimensions()), 0));

  TiledHloInstruction tiled_hlo_instruction(root, tiled_operands[0]);
  EXPECT_THAT(tiled_hlo_instruction, MatchString(R"(
    hlo: %broadcast = f32[10,20,30]{2,1,0} broadcast(%p0), dimensions={0,2}
    tile: (tid_0, tid_1, tid_2)
      -> offsets [tid_0 * ts_0, tid_2 * ts_2]
         sizes [ts_0, ts_2]
         strides [1, 3]
         upper bounds [10, 30]
  )"));
}

MATCHER_P2(IsHloWithOperands, opcode, operand_opcodes,
           "Check if HLO has given opcode and operands with given opcodes") {
  const TiledHloInstruction& hlo = *arg;
  if (hlo.hlo()->opcode() != opcode) {
    return false;
  }
  return absl::c_all_of(llvm::zip(hlo.operands(), operand_opcodes),
                        [](const auto& pair) {
                          const auto& [operand, operand_opcode] = pair;
                          return operand->hlo()->opcode() == operand_opcode;
                        });
}

class TileAnalysisTest : public HloHardwareIndependentTestBase {
 public:
  TileAnalysisTest() { RegisterSymbolicExprStorage(&mlir_context_); }

  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_string);
    CHECK_OK(module_or);
    module_ = std::move(module_or.value());
    return module_->entry_computation()->root_instruction();
  }

  absl::StatusOr<TiledHloComputation> ParseAndTile(
      absl::string_view hlo_string, absl::Span<const int64_t> tile_sizes) {
    HloInstruction* root = ParseAndGetRoot(hlo_string);
    auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
    auto tiling_space = TilingSpace::Create(*fusion_adaptor, &mlir_context_);
    tiling_space->AssignTileSizes(tile_sizes);
    return TiledHloComputation::Tile(*fusion_adaptor, std::move(tiling_space));
  }

  mlir::MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

TEST_F(TileAnalysisTest, SingleTileReduce) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"hlo(
    max {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT maximum = f32[] maximum(x, y)
    }

    ENTRY e {
      p0 = f32[2,97]{1,0} parameter(0)
      constant = f32[] constant(-inf)
      ROOT reduce = f32[2]{0} reduce(p0, constant), dimensions={1}, to_apply=max
    })hlo",
                                    {4, 100}));

  EXPECT_THAT(tiled_computation, MatchString(R"(
Dimensions:
0 type: parallel size: 2 tile size: 4 dim ID:0 hlo: %reduce = f32[2]{0} reduce(%p0, %constant), dimensions={1}, to_apply=%max
1 type: sequential size: 97 tile size: 100 dim ID:1 hlo: %reduce = f32[2]{0} reduce(%p0, %constant), dimensions={1}, to_apply=%max
Root tiles:
0 root tile:  offsets [tid_0 * 4] sizes [4] strides [1] upper bounds [2]

Tiled HLO:
  p0.tile_0 = parameter(0)  offsets [tid_0 * 4, tid_1 * 100] sizes [4, 100] strides [1, 1] upper bounds [2, 97]
  constant.tile_0 = constant()  offsets [] sizes [] strides [] upper bounds []
  reduce.tile_0 = reduce(p0.tile_0, constant.tile_0)  offsets [tid_0 * 4] sizes [4] strides [1] upper bounds [2]
  )"));

  EXPECT_THAT(
      tiled_computation.tiled_hlo_instructions(),
      Contains(IsHloWithOperands(
          HloOpcode::kReduce, std::vector<HloOpcode>{HloOpcode::kParameter,
                                                     HloOpcode::kConstant})));
}

TEST_F(TileAnalysisTest, SimpleNormalizationDiamond) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"hlo(
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
    })hlo",
                                    {8, 128, 128}));

  EXPECT_THAT(tiled_computation, MatchString(R"(
Dimensions:
0 type: parallel size: 2 tile size: 8 dim ID:0 hlo: %subtract = f32[2,97]{1,0} subtract(%p0.1, %broadcast)
1 type: parallel size: 97 tile size: 128 dim ID:1 hlo: %subtract = f32[2,97]{1,0} subtract(%p0.1, %broadcast)
2 type: sequential size: 97 tile size: 128 dim ID:1 hlo: %reduce = f32[2]{0} reduce(%p0.1, %constant), dimensions={1}, to_apply=%max
Root tiles:
0 root tile:  offsets [tid_0 * 8, tid_1 * 128] sizes [8, 128] strides [1, 1] upper bounds [2, 97]

Tiled HLO:
  p0.2.tile_0 = parameter(0)  offsets [tid_0 * 8, tid_1 * 128] sizes [8, 128] strides [1, 1] upper bounds [2, 97]
  p0.2.tile_1 = parameter(0)  offsets [tid_0 * 8, tid_2 * 128] sizes [8, 128] strides [1, 1] upper bounds [2, 97]
  constant.tile_0 = constant()  offsets [] sizes [] strides [] upper bounds []
  reduce.tile_0 = reduce(p0.2.tile_1, constant.tile_0)  offsets [tid_0 * 8] sizes [8] strides [1] upper bounds [2]
  broadcast.tile_0 = broadcast(reduce.tile_0)  offsets [tid_0 * 8, tid_1 * 128] sizes [8, 128] strides [1, 1] upper bounds [2, 97]
  subtract.tile_0 = subtract(p0.2.tile_0, broadcast.tile_0)  offsets [tid_0 * 8, tid_1 * 128] sizes [8, 128] strides [1, 1] upper bounds [2, 97]
  )"));

  EXPECT_THAT(tiled_computation.tiled_hlo_instructions(),
              Contains(IsHloWithOperands(
                  HloOpcode::kSubtract,
                  std::vector<HloOpcode>{HloOpcode::kParameter,
                                         HloOpcode::kBroadcast})));
  EXPECT_THAT(
      tiled_computation.tiled_hlo_instructions(),
      Contains(IsHloWithOperands(HloOpcode::kBroadcast,
                                 std::vector<HloOpcode>{HloOpcode::kReduce})));
}

TEST_F(TileAnalysisTest, ConcatenateIsSupported) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(
                           R"hlo(
    concatenate {
      p0 = bf16[6] parameter(0)
      p1 = bf16[6] parameter(1)
      p2 = bf16[6] parameter(2)
      ROOT concatenate = bf16[18] concatenate(p0, p1, p2), dimensions={0}
    }

    ENTRY main {
      p0 = bf16[6] parameter(0)
      p1 = bf16[6] parameter(1)
      p2 = bf16[6] parameter(2)
      ROOT fusion = bf16[18] fusion(p0, p1, p2),
        kind=kCustom, calls=concatenate
    })hlo",
                           {3}));

  EXPECT_THAT(tiled_computation, MatchString(R"(
Dimensions:
0 type: parallel size: 18 tile size: 3 dim ID:0 hlo: %concatenate = bf16[18]{0} concatenate(%p0, %p1, %p2), dimensions={0}
Root tiles:
0 root tile:  offsets [tid_0 * 3] sizes [3] strides [1] upper bounds [18]

Tiled HLO:
  concatenate.tile_0 = concatenate(p0.1.tile_0, p1.1.tile_0, p2.1.tile_0)  offsets [tid_0 * 3] sizes [3] strides [1] upper bounds [18]
  region #0 {
    p0.1.tile_0 = parameter(0)  offsets [tid_0 * 3] sizes [3] strides [1] upper bounds [6]
  }
  region #1 {
    p1.1.tile_0 = parameter(1)  offsets [tid_0 * 3 - 6] sizes [3] strides [1] upper bounds [6]
  }
  region #2 {
    p2.1.tile_0 = parameter(2)  offsets [tid_0 * 3 - 12] sizes [3] strides [1] upper bounds [6]
  }
  )"));

  EXPECT_THAT(tiled_computation.tiled_hlo_instructions(),
              Contains(IsHloWithOperands(
                  HloOpcode::kConcatenate,
                  std::vector<HloOpcode>(3, HloOpcode::kParameter))));
}

TEST_F(TileAnalysisTest, Dot) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(
                           R"hlo(
    fusion {
      p0 = f32[4,8] parameter(0)
      p1 = f32[8,16] parameter(1)
      ROOT dot = f32[4,16] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY main {
      p0 = f32[4,8] parameter(0)
      p1 = f32[8,16] parameter(1)
      ROOT fusion = f32[4,16] fusion(p0, p1), kind=kLoop, calls=fusion
    })hlo",
                           {1, 2, 4}));

  EXPECT_THAT(tiled_computation, MatchString(R"(
     Dimensions:
       0 type: parallel size: 4 tile size: 1 dim ID:0 hlo: %dot = f32[4,16]{1,0} dot(%p0, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
       1 type: parallel size: 16 tile size: 2 dim ID:1 hlo: %dot = f32[4,16]{1,0} dot(%p0, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
       2 type: sequential size: 8 tile size: 4 dim ID:2 hlo: %dot = f32[4,16]{1,0} dot(%p0, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
       Root tiles:
       0 root tile:  offsets [tid_0, tid_1 * 2] sizes [1, 2] strides [1, 1] upper bounds [4, 16]

     Tiled HLO:
       dot.tile_0 = dot(p0.1.tile_0, p1.1.tile_0)  offsets [tid_0, tid_1 * 2] sizes [1, 2] strides [1, 1] upper bounds [4, 16]
       region #0 {
         p0.1.tile_0 = parameter(0)  offsets [tid_0, tid_2 * 4] sizes [1, 4] strides [1, 1] upper bounds [4, 8]
         p1.1.tile_0 = parameter(1)  offsets [tid_2 * 4, tid_1 * 2] sizes [4, 2] strides [1, 1] upper bounds [8, 16]
       }
  )"));

  EXPECT_THAT(
      tiled_computation.tiled_hlo_instructions(),
      Contains(IsHloWithOperands(
          HloOpcode::kDot, std::vector<HloOpcode>(2, HloOpcode::kParameter))));
}

TEST_F(TileAnalysisTest, DotWithFullContractionDimTile) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(
                           R"hlo(
    fusion {
      p0 = f32[4,8] parameter(0)
      p1 = f32[8,16] parameter(1)
      ROOT dot = f32[4,16] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY main {
      p0 = f32[4,8] parameter(0)
      p1 = f32[8,16] parameter(1)
      ROOT fusion = f32[4,16] fusion(p0, p1), kind=kLoop, calls=fusion
    })hlo",
                           {32, 32, 32}));

  EXPECT_THAT(tiled_computation, MatchString(R"(
Dimensions:
0 type: parallel size: 4 tile size: 32 dim ID:0 hlo: %dot = f32[4,16]{1,0} dot(%p0, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
1 type: parallel size: 16 tile size: 32 dim ID:1 hlo: %dot = f32[4,16]{1,0} dot(%p0, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
2 type: sequential size: 8 tile size: 32 dim ID:2 hlo: %dot = f32[4,16]{1,0} dot(%p0, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
Root tiles:
0 root tile:  offsets [tid_0 * 32, tid_1 * 32] sizes [32, 32] strides [1, 1] upper bounds [4, 16]

Tiled HLO:
  dot.tile_0 = dot(p0.1.tile_0, p1.1.tile_0)  offsets [tid_0 * 32, tid_1 * 32] sizes [32, 32] strides [1, 1] upper bounds [4, 16]
  region #0 {
    p0.1.tile_0 = parameter(0)  offsets [tid_0 * 32, tid_2 * 32] sizes [32, 32] strides [1, 1] upper bounds [4, 8]
    p1.1.tile_0 = parameter(1)  offsets [tid_2 * 32, tid_1 * 32] sizes [32, 32] strides [1, 1] upper bounds [8, 16]
  }
  )"));

  EXPECT_THAT(
      tiled_computation.tiled_hlo_instructions(),
      Contains(IsHloWithOperands(
          HloOpcode::kDot, std::vector<HloOpcode>(2, HloOpcode::kParameter))));
}

TEST_F(TileAnalysisTest, ScaledDot) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"hlo(
    fusion {
      lhs = f8e4m3fn[128,64] parameter(0)
      rhs = f8e4m3fn[64,128] parameter(1)
      lhs_scale = f8e8m0fnu[128,2] parameter(2)
      rhs_scale = f8e8m0fnu[2,128] parameter(3)
      ROOT dot = f32[128,128] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = f8e4m3fn[128,64] parameter(0)
      rhs = f8e4m3fn[64,128] parameter(1)
      lhs_scale = f8e8m0fnu[128,2] parameter(2)
      rhs_scale = f8e8m0fnu[2,128] parameter(3)
      ROOT fusion = f32[128,128] fusion(lhs, rhs, lhs_scale, rhs_scale),
        kind=kLoop, calls=fusion
    })hlo",
                                    {2, 4, 8}));

  EXPECT_THAT(tiled_computation, MatchString(R"(
Dimensions:
0 type: parallel size: 128 tile size: 2 dim ID:0 hlo: %dot = f32[128,128]{1,0} scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale), lhs_contracting_dims={1}, rhs_contracting_dims={0}
1 type: parallel size: 128 tile size: 4 dim ID:1 hlo: %dot = f32[128,128]{1,0} scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale), lhs_contracting_dims={1}, rhs_contracting_dims={0}
2 type: sequential size: 64 tile size: 8 dim ID:2 hlo: %dot = f32[128,128]{1,0} scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale), lhs_contracting_dims={1}, rhs_contracting_dims={0}
Root tiles:
0 root tile:  offsets [tid_0 * 2, tid_1 * 4] sizes [2, 4] strides [1, 1] upper bounds [128, 128]

Tiled HLO:
  dot.tile_0 = scaled-dot(lhs.1.tile_0, rhs.1.tile_0, lhs_scale.1.tile_0, rhs_scale.1.tile_0)  offsets [tid_0 * 2, tid_1 * 4] sizes [2, 4] strides [1, 1] upper bounds [128, 128]
  region #0 {
    lhs.1.tile_0 = parameter(0)  offsets [tid_0 * 2, tid_2 * 8] sizes [2, 8] strides [1, 1] upper bounds [128, 64]
    rhs.1.tile_0 = parameter(1)  offsets [tid_2 * 8, tid_1 * 4] sizes [8, 4] strides [1, 1] upper bounds [64, 128]
    lhs_scale.1.tile_0 = parameter(2)  offsets [tid_0 * 2, (tid_2 * 8) floordiv 32] sizes [2, (tid_2 * 8 + 7) floordiv 32 - tid_2 floordiv 4 + 1] strides [1, 1] upper bounds [128, 2]
    rhs_scale.1.tile_0 = parameter(3)  offsets [(tid_2 * 8) floordiv 32, tid_1 * 4] sizes [(tid_2 * 8 + 7) floordiv 32 - tid_2 floordiv 4 + 1, 4] strides [1, 1] upper bounds [2, 128]
  }
  )"));

  EXPECT_THAT(tiled_computation.tiled_hlo_instructions(),
              Contains(IsHloWithOperands(
                  HloOpcode::kScaledDot,
                  std::vector<HloOpcode>(4, HloOpcode::kParameter))));
}

TEST_F(TileAnalysisTest, RuntimeVariablesAreEmittedFirst) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"hlo(
    fusion {
      p0 = s32[64] parameter(0)
      c13 = s32[] constant(13)
      c7 = s32[] constant(7)
      add = s32[] add(c7, c13)
      ROOT r = s32[10] dynamic-slice(p0, add), dynamic_slice_sizes={10}
    }

    ENTRY main {
      p0 = s32[64] parameter(0)
      ROOT fusion = s32[10] fusion(p0), kind=kCustom, calls=fusion
    })hlo",
                                    {10}));

  EXPECT_THAT(tiled_computation, MatchString(R"(
Dimensions:
0 type: parallel size: 10 tile size: 10 dim ID:0 hlo: %r = s32[10]{0} dynamic-slice(%p0, %add), dynamic_slice_sizes={10}
Runtime variables:
0 bounds: [0, 54] hlo: %add = s32[] add(%c7, %c13)
Root tiles:
0 root tile:  offsets [tid_0 * 10] sizes [10] strides [1] upper bounds [10]

Tiled HLO:
  c7.tile_0 = constant()  offsets [] sizes [] strides [] upper bounds []
  c13.tile_0 = constant()  offsets [] sizes [] strides [] upper bounds []
  add.tile_0 = add(c7.tile_0, c13.tile_0)  offsets [] sizes [] strides [] upper bounds []
  p0.1.tile_0 = parameter(0)  offsets [rt_0 + tid_0 * 10] sizes [10] strides [1] upper bounds [rt_0 + 10]
  r.tile_0 = dynamic-slice(p0.1.tile_0, add.tile_0)  offsets [tid_0 * 10] sizes [10] strides [1] upper bounds [10]
  )"));
  EXPECT_THAT(
      tiled_computation.rt_symbol_to_tiled_hlo(),
      UnorderedElementsAre(
          Pair(::testing::_,
               Pair(Pointee(Property(&TiledHloInstruction::hlo,
                                     Pointee(Property(&HloInstruction::opcode,
                                                      Eq(HloOpcode::kAdd))))),
                    ::testing::_))));
}

TEST_F(TileAnalysisTest, CSEworksCorrectly) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"hlo(
    fusion {
      p0 = f32[64] parameter(0)
      c0 = s32[64] convert(p0)
      slice1 = s32[1] slice(c0), slice={[0:1]}
      off = s32[] reshape(slice1)
      d1 = s32[20] dynamic-slice(c0, off), dynamic_slice_sizes={20}
      slice2 = s32[1] slice(d1), slice={[1:2]}
      off2 = s32[] reshape(slice2)
      ROOT d2 = s32[10] dynamic-slice(d1, off2), dynamic_slice_sizes={10}
    }

    ENTRY main {
      p0 = f32[64] parameter(0)
      ROOT fusion = s32[10] fusion(p0), kind=kCustom, calls=fusion
    })hlo",
                                    {16}));

  EXPECT_THAT(tiled_computation, MatchString(R"(
Dimensions:
0 type: parallel size: 10 tile size: 16 dim ID:0 hlo: %d2 = s32[10]{0} dynamic-slice(%d1, %off2), dynamic_slice_sizes={10}
Runtime variables:
0 bounds: [0, 10] hlo: %off2 = s32[] reshape(%slice2)
1 bounds: [0, 44] hlo: %off = s32[] reshape(%slice1)
Root tiles:
0 root tile:  offsets [tid_0 * 16] sizes [16] strides [1] upper bounds [10]

Tiled HLO:
  p0.1.tile_0 = parameter(0)  offsets [0] sizes [1] strides [1] upper bounds [1]
  c0.tile_0 = convert(p0.1.tile_0)  offsets [0] sizes [1] strides [1] upper bounds [1]
  slice1.tile_0 = slice(c0.tile_0)  offsets [0] sizes [1] strides [1] upper bounds [1]
  off.tile_0 = reshape(slice1.tile_0)  offsets [] sizes [] strides [] upper bounds []
  p0.1.tile_1 = parameter(0)  offsets [rt_1 + 1] sizes [1] strides [1] upper bounds [rt_1 + 2]
  c0.tile_1 = convert(p0.1.tile_1)  offsets [rt_1 + 1] sizes [1] strides [1] upper bounds [rt_1 + 2]
  d1.tile_0 = dynamic-slice(c0.tile_1, off.tile_0)  offsets [1] sizes [1] strides [1] upper bounds [2]
  slice2.tile_0 = slice(d1.tile_0)  offsets [0] sizes [1] strides [1] upper bounds [1]
  off2.tile_0 = reshape(slice2.tile_0)  offsets [] sizes [] strides [] upper bounds []
  p0.1.tile_2 = parameter(0)  offsets [rt_1 + rt_0 + tid_0 * 16] sizes [16] strides [1] upper bounds [rt_0 + 10 + rt_1]
  c0.tile_2 = convert(p0.1.tile_2)  offsets [rt_1 + rt_0 + tid_0 * 16] sizes [16] strides [1] upper bounds [rt_0 + 10 + rt_1]
  d1.tile_1 = dynamic-slice(c0.tile_2, off.tile_0)  offsets [rt_0 + tid_0 * 16] sizes [16] strides [1] upper bounds [rt_0 + 10]
  d2.tile_0 = dynamic-slice(d1.tile_1, off2.tile_0)  offsets [tid_0 * 16] sizes [16] strides [1] upper bounds [10]
  )"));
}

// TODO(b/422676780): Port the remaining tests.

}  // namespace
}  // namespace xla::gpu::experimental
