/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/tile_analysis.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::AllOf;
using ::testing::DescribeMatcher;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::ExplainMatchResult;
using ::testing::Optional;
using ::testing::SizeIs;
using ::testing::StrEq;

MATCHER_P4(
    MatchSymbolicTile, affine_map_string, sizes, max_sizes,
    max_strides_and_offsets,
    absl::StrCat(
        negation ? "equals " : "doesn't equal ", "symbolic tile ",
        affine_map_string, " where sizes_ ",
        DescribeMatcher<std::vector<std::optional<int64_t>>>(sizes),
        ", max_sizes_ ", DescribeMatcher<std::vector<int64_t>>(max_sizes),
        " and ", "max_strides_and_offsets_ ",
        DescribeMatcher<std::vector<int64_t>>(max_strides_and_offsets))) {
  return ExplainMatchResult(StrEq(affine_map_string),
                            ToString(arg.affine_map()), result_listener) &&
         ExplainMatchResult(sizes, arg.sizes(), result_listener) &&
         ExplainMatchResult(max_sizes, arg.max_sizes(), result_listener) &&
         ExplainMatchResult(max_strides_and_offsets,
                            arg.max_strides_and_offsets(), result_listener);
}

class SymbolicTileTest : public HloTestBase {
 public:
  StatusOr<HloInstructionIndexing> GetOutputToInputIndexingForEntryComputation(
      absl::string_view hlo_string, int output_id = 0) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_string));
    HloInstruction* root = module->entry_computation()->root_instruction();

    for (auto* operand : root->operands()) {
      TF_RET_CHECK(operand->opcode() == HloOpcode::kParameter ||
                   operand->opcode() == HloOpcode::kConstant)
          << "If there are multiple instructions, they need to be wrapped in a "
             "fusion.";
    }
    return ComputeOutputToInputIndexing(root, output_id, &mlir_context_);
  }
  mlir::MLIRContext mlir_context_;
};

TEST_F(SymbolicTileTest, SymbolicTileConstructionIsCorrect) {
  std::vector<int64_t> shape = {182, 17, 2};
  SymbolicTile tile(shape, &mlir_context_);

  EXPECT_THAT(ToString(tile.affine_map()),
              StrEq("(d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> "
                    "(d0 * s0 + d1, d2 * s1 + d3, d4 * s2 + d5)"));
  EXPECT_THAT(tile.sizes(), AllOf(Each(std::nullopt), SizeIs(shape.size())));
  EXPECT_THAT(tile.max_sizes(), ElementsAreArray(shape));
}

TEST_F(SymbolicTileTest,
       CanPropagateTileFromDotOutputToInputsWithoutSpecializedTileSizes) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[11, 17, 19] parameter(0)
      p1 = f32[11, 19, 23] parameter(1)
      ROOT dot = f32[11, 17, 23] dot(p0, p1),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
    }
  )"));

  SymbolicTile output_tile(/*target_shape=*/{11, 17, 23}, &mlir_context_);

  EXPECT_THAT(
      output_tile.TryPropagateTileThroughIndexingMap(
          *input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTile(
          "(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3] -> "
          "(d0 * s1 + d1, d2 * s2 + d3, s0)",
          ElementsAre(19, std::nullopt, std::nullopt, std::nullopt),
          ElementsAre(19, 11, 17, 23), ElementsAre(11, 11, 17, 17, 23, 23))));

  EXPECT_THAT(
      output_tile.TryPropagateTileThroughIndexingMap(
          *input_indexing.indexing_maps[1].begin()),
      Optional(MatchSymbolicTile(
          "(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3] -> "
          "(d0 * s1 + d1, s0, d4 * s3 + d5)",
          ElementsAre(19, std::nullopt, std::nullopt, std::nullopt),
          ElementsAre(19, 11, 17, 23), ElementsAre(11, 11, 17, 17, 23, 23))));
}

TEST_F(SymbolicTileTest, CanPropagateTileThroughTrivialReshape) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[11, 17, 19] parameter(0)
      ROOT reshape = f32[1, 11, 17, 19] reshape(p0)
    }
  )"));

  std::vector<int64_t> target_shape({1, 11, 17, 19});
  SymbolicTile output_tile(target_shape, &mlir_context_);

  std::optional<SymbolicTile> operand_tile =
      output_tile.TryPropagateTileThroughIndexingMap(
          *input_indexing.indexing_maps[0].begin());

  std::optional<int64_t> undef = std::nullopt;

  EXPECT_THAT(operand_tile,
              Optional(MatchSymbolicTile(
                  "(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3] -> "
                  "(d2 * s1 + d3, d4 * s2 + d5, d6 * s3 + d7)",  // NOLINT
                  AllOf(Each(undef), SizeIs(target_shape.size())),
                  ElementsAreArray(target_shape),
                  ElementsAre(1, 1, 11, 11, 17, 17, 19, 19))));
}

TEST_F(SymbolicTileTest,
       FailsToPropagateTileThroughReshapeWithoutSpecializedTileSizes) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[12, 4, 19] parameter(0)
      ROOT reshape = f32[4, 12, 19] reshape(p0)
    }
  )"));

  std::vector<int64_t> target_shape({4, 12, 19});
  SymbolicTile output_tile(target_shape, &mlir_context_);

  EXPECT_EQ(output_tile.TryPropagateTileThroughIndexingMap(
                *input_indexing.indexing_maps[0].begin()),
            std::nullopt);
}

TEST_F(SymbolicTileTest,
       CanPropagateTileThroughElementwiseOpWithoutSpecializedTileSizes) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[150] parameter(0)
      p1 = f32[150] parameter(1)
      ROOT add = f32[150] add(p0, p1)
    }
  )"));

  SymbolicTile output_tile(/*target_shape=*/{150}, &mlir_context_);

  EXPECT_THAT(output_tile.TryPropagateTileThroughIndexingMap(
                  *input_indexing.indexing_maps[0].begin()),
              Optional(MatchSymbolicTile(
                  "(d0, d1)[s0] -> (d0 * s0 + d1)", ElementsAre(std::nullopt),
                  ElementsAre(150), ElementsAre(150, 150))));
}

TEST_F(SymbolicTileTest,
       CanPropagateTileFromBroadcastOutputToInputWithoutSpecializedTileSizes) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[150] parameter(0)
      ROOT broadcast = f32[157,150] broadcast(p0), dimensions={1}
    }
  )"));

  SymbolicTile output_tile(/*target_shape=*/{157, 150}, &mlir_context_);

  EXPECT_THAT(output_tile.TryPropagateTileThroughIndexingMap(
                  *input_indexing.indexing_maps[0].begin()),
              Optional(MatchSymbolicTile(
                  "(d0, d1, d2, d3)[s0, s1] -> (d2 * s1 + d3)",
                  ElementsAre(std::nullopt, std::nullopt),
                  ElementsAre(157, 150), ElementsAre(157, 157, 150, 150))));
}

TEST_F(SymbolicTileTest,
       CanPropagateTileFromReduceOutputToInputWithoutSpecializedTileSizes) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }

    ENTRY e {
      p0 = f32[125,150] parameter(0)
      c0 = f32[] constant(-inf)
      ROOT reduce = f32[150] reduce(p0, c0), dimensions={0}, to_apply=max
    }
  )"));

  SymbolicTile output_tile(/*target_shape=*/{150}, &mlir_context_);

  EXPECT_THAT(output_tile.TryPropagateTileThroughIndexingMap(
                  *input_indexing.indexing_maps[0].begin()),
              Optional(MatchSymbolicTile(
                  "(d0, d1)[s0, s1] -> (s0, d0 * s1 + d1)",
                  ElementsAre(125, std::nullopt), ElementsAre(125, 150),
                  ElementsAre(150, 150))));
}

TEST_F(SymbolicTileTest,
       CanPropagateTileThroughReverseWithoutSpecializedTileSizes) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[179] parameter(0)
      ROOT reverse = f32[179] reverse(p0), dimensions={0}
    }
  )"));

  SymbolicTile output_tile(/*target_shape=*/{179}, &mlir_context_);

  EXPECT_THAT(
      output_tile.TryPropagateTileThroughIndexingMap(
          *input_indexing.indexing_maps[0].begin()),
      Optional(MatchSymbolicTile("(d0, d1)[s0] -> (-(d0 * s0 + d1) + 178)",
                                 ElementsAre(std::nullopt), ElementsAre(179),
                                 ElementsAre(179, 179))));
}

TEST_F(SymbolicTileTest,
       CanPropagateTileFromSliceOutputToInputWithoutSpecializedTileSizes) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[120,142] parameter(0)
      ROOT slice = f32[10,21] slice(p0), slice={[40:60:2], [20:104:4]}
    }
  )"));

  SymbolicTile output_tile(/*target_shape=*/{10, 21}, &mlir_context_);

  EXPECT_THAT(output_tile.TryPropagateTileThroughIndexingMap(
                  *input_indexing.indexing_maps[0].begin()),
              Optional(MatchSymbolicTile(
                  "(d0, d1, d2, d3)[s0, s1] -> "
                  "((d0 * s0 + d1) * 2 + 40, (d2 * s1 + d3) * 4 + 20)",
                  ElementsAre(std::nullopt, std::nullopt), ElementsAre(10, 21),
                  ElementsAre(10, 10, 21, 21))));
}

TEST_F(SymbolicTileTest,
       CanPropagateTileThroughTransposeWithoutSpecializedTileSizes) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[21,10] parameter(0)
      ROOT transpose = f32[10,21] transpose(p0), dimensions={1,0}
    }
  )"));

  SymbolicTile output_tile(/*target_shape=*/{10, 21}, &mlir_context_);

  EXPECT_THAT(output_tile.TryPropagateTileThroughIndexingMap(
                  *input_indexing.indexing_maps[0].begin()),
              Optional(MatchSymbolicTile(
                  "(d0, d1, d2, d3)[s0, s1] -> (d2 * s1 + d3, d0 * s0 + d1)",
                  ElementsAre(std::nullopt, std::nullopt), ElementsAre(10, 21),
                  ElementsAre(10, 10, 21, 21))));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
