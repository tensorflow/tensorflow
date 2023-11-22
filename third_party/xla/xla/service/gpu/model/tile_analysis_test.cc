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

#include <vector>

#include <gmock/gmock.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/statusor.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::ExplainMatchResult;
using ::testing::HasSubstr;
using ::testing::PrintToString;

MATCHER_P2(MatchIndexingMap, affine_map_string, sizes,
           absl::StrCat(negation ? "equals " : "doesn't equal ", "affine map ",
                        affine_map_string, " with sizes ",
                        PrintToString(sizes))) {
  return ExplainMatchResult(HasSubstr(affine_map_string),
                            ToString(arg.affine_map), result_listener) &&
         ExplainMatchResult(ElementsAreArray(sizes), arg.sizes,
                            result_listener);
}

MATCHER_P2(MatchOperandIndexing, operand_id, indexing_map_matchers, "") {
  return ExplainMatchResult(Eq(operand_id), arg.operand_id, result_listener) &&
         ExplainMatchResult(indexing_map_matchers, arg.indexing_maps,
                            result_listener);
}

class TileAnalysisTest : public HloTestBase {
 public:
  StatusOr<HloInstructionIndexing> GetIndexingMapsForEntryComputation(
      absl::string_view hlo_string, int operand_id = 0) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_string));
    HloInstruction* root = module->entry_computation()->root_instruction();

    return ComputeInstructionIndexing(root, operand_id, &mlir_context_);
  }
  mlir::MLIRContext mlir_context_;
};

TEST_F(TileAnalysisTest, ElementwiseOp) {
  auto input_indexing_or = GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20] parameter(0)
      p1 = f32[10, 20] parameter(1)
      ROOT add0 = f32[10, 20] add(p0, p1)
    }
  )");
  ASSERT_IS_OK(input_indexing_or);
  EXPECT_THAT(
      input_indexing_or->operand_indexing_maps,
      ElementsAre(
          MatchOperandIndexing(
              0, ElementsAre(MatchIndexingMap("(d0, d1) -> (d0, d1)",
                                              std::vector<int>{10, 20}))),
          MatchOperandIndexing(
              1, ElementsAre(MatchIndexingMap("(d0, d1) -> (d0, d1)",
                                              std::vector<int>{10, 20})))));
}

TEST_F(TileAnalysisTest, BroadcastOp) {
  auto input_indexing_or = GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[20] parameter(0)
      ROOT bc0 = f32[10, 20, 30] broadcast(p0), dimensions={1}
    }
  )");
  ASSERT_IS_OK(input_indexing_or);
  EXPECT_THAT(
      input_indexing_or->operand_indexing_maps,
      ElementsAre(MatchOperandIndexing(
          0, ElementsAre(MatchIndexingMap("(d0, d1, d2) -> (d1)",
                                          std::vector<int>{10, 20, 30})))));
}

TEST_F(TileAnalysisTest, ReduceOp) {
  auto input_indexing_or = GetIndexingMapsForEntryComputation(R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }
    ENTRY e {
      p0 = f32[150, 20, 10, 50] parameter(0)
      p0_init = f32[] constant(-inf)
      ROOT reduce = f32[150, 10] reduce(p0, p0_init),
        dimensions={3, 1}, to_apply=max
    }
  )");
  ASSERT_IS_OK(input_indexing_or);
  EXPECT_THAT(input_indexing_or->operand_indexing_maps,
              ElementsAre(MatchOperandIndexing(
                  0, ElementsAre(MatchIndexingMap(
                         "(d0, d1)[s0, s1] -> (d0, s0, d1, s1)",
                         std::vector<int>{150, 10, 20, 50})))));
}

TEST_F(TileAnalysisTest, VariadicReduceOp) {
  absl::string_view hlo_string = R"(
    HloModule m
    min {
      tmp_0 = f32[] parameter(0)
      tmp_1 = f32[] parameter(2)
      tmp_2 = s32[] parameter(1)
      tmp_3 = s32[] parameter(3)
      cmp = pred[] compare(tmp_0, tmp_1), direction=GE
      select1 = f32[] select(cmp, tmp_0, tmp_1)
      select2 = s32[] select(cmp, tmp_2, tmp_3)
      ROOT tmp_4 = (f32[], s32[]) tuple(select1, select2)
    }
    ENTRY e {
      p0 = f32[256,10] parameter(0)
      p0_init = f32[] constant(-inf)
      p1 = s32[256,10] parameter(1)
      p1_init = s32[] constant(0)
      ROOT reduce = (f32[10], s32[10]) reduce(p0, p1, p0_init, p1_init),
        dimensions={0}, to_apply=min
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();

  auto input_indexing_0 = ComputeInstructionIndexing(root, 0, &mlir_context_);
  ASSERT_IS_OK(input_indexing_0);
  EXPECT_THAT(
      input_indexing_0->operand_indexing_maps,
      ElementsAre(
          MatchOperandIndexing(
              0, ElementsAre(MatchIndexingMap("(d0)[s0] -> (s0, d0)",
                                              std::vector<int>{10, 256}))),
          MatchOperandIndexing(
              1, ElementsAre(MatchIndexingMap("(d0)[s0] -> (s0, d0)",
                                              std::vector<int>{10, 256})))));

  auto input_indexing_1 = ComputeInstructionIndexing(root, 1, &mlir_context_);
  ASSERT_IS_OK(input_indexing_1);
  EXPECT_THAT(
      input_indexing_1->operand_indexing_maps,
      ElementsAre(
          MatchOperandIndexing(
              0, ElementsAre(MatchIndexingMap("(d0)[s0] -> (s0, d0)",
                                              std::vector<int>{10, 256}))),
          MatchOperandIndexing(
              1, ElementsAre(MatchIndexingMap("(d0)[s0] -> (s0, d0)",
                                              std::vector<int>{10, 256})))));
}

TEST_F(TileAnalysisTest, ReverseOp) {
  auto input_indexing_or = GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      %p0 = f32[1, 17, 9, 9] parameter(0)
     ROOT reverse = f32[1, 17, 9, 9] reverse(%p0), dimensions={1, 2}
    }
  )");
  ASSERT_IS_OK(input_indexing_or);
  EXPECT_THAT(input_indexing_or->operand_indexing_maps,
              ElementsAre(MatchOperandIndexing(
                  0, ElementsAre(MatchIndexingMap(
                         "(d0, d1, d2, d3) -> (d0, -d1 + 17, -d2 + 9, d3)",
                         std::vector<int>{1, 17, 9, 9})))));
}

TEST_F(TileAnalysisTest, SliceOp) {
  auto input_indexing_or = GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      %p0 = f32[10, 20, 50] parameter(0)
      ROOT %slice = f32[5, 3, 25] slice(f32[10, 20, 50] %p0),
          slice={[5:10:1], [3:20:7], [0:50:2]}
    }
  )");
  ASSERT_IS_OK(input_indexing_or);
  EXPECT_THAT(input_indexing_or->operand_indexing_maps,
              ElementsAre(MatchOperandIndexing(
                  0, ElementsAre(MatchIndexingMap(
                         "(d0, d1, d2) -> (d0 + 5, d1 * 7 + 3, d2 * 2)",
                         std::vector<int>{5, 3, 25})))));
}

TEST_F(TileAnalysisTest, TransposeOp) {
  auto input_indexing_or = GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      %p0 = f16[1, 8, 1536, 512] parameter(0)
      ROOT transpose = f16[1, 8, 512, 1536]{2, 3, 1, 0}
             transpose(%p0), dimensions={0, 1, 3, 2}
    }
  )");
  ASSERT_IS_OK(input_indexing_or);
  EXPECT_THAT(input_indexing_or->operand_indexing_maps,
              ElementsAre(MatchOperandIndexing(
                  0, ElementsAre(MatchIndexingMap(
                         "(d0, d1, d2, d3) -> (d0, d1, d3, d2)",
                         std::vector<int>{1, 8, 512, 1536})))));
}

TEST_F(TileAnalysisTest, UnsupportedOp) {
  auto input_indexing_or = GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      %p0 = f32[1, 17, 9, 9] parameter(0)
      %p1 = f32[5, 17, 9, 9] parameter(1)
      ROOT %concat = f32[6, 17, 9, 9] concatenate(%p0, %p1)
    }
  )");
  ASSERT_IS_NOT_OK(input_indexing_or);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
