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
using ::testing::UnorderedElementsAre;

MATCHER_P2(MatchIndexingMap, affine_map_string, input_dims_sizes,
           absl::StrCat(negation ? "equals " : "doesn't equal ", "affine map ",
                        affine_map_string, " with input dim sizes ",
                        PrintToString(input_dims_sizes))) {
  return ExplainMatchResult(HasSubstr(affine_map_string),
                            ToString(arg.affine_map), result_listener) &&
         ExplainMatchResult(ElementsAreArray(input_dims_sizes),
                            arg.input_dims_sizes, result_listener);
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
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20] parameter(0)
      p1 = f32[10, 20] parameter(1)
      ROOT add0 = f32[10, 20] add(p0, p1)
    }
  )"));
  EXPECT_THAT(
      input_indexing.operand_indexing_maps,
      ElementsAre(MatchOperandIndexing(
                      0, ElementsAre(MatchIndexingMap("(d0, d1) -> (d0, d1)",
                                                      std::vector<int>{}))),
                  MatchOperandIndexing(
                      1, ElementsAre(MatchIndexingMap("(d0, d1) -> (d0, d1)",
                                                      std::vector<int>{})))));
}

TEST_F(TileAnalysisTest, BroadcastOp) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[20] parameter(0)
      ROOT bc0 = f32[10, 20, 30] broadcast(p0), dimensions={1}
    }
  )"));
  EXPECT_THAT(input_indexing.operand_indexing_maps,
              ElementsAre(MatchOperandIndexing(
                  0, ElementsAre(MatchIndexingMap("(d0, d1, d2) -> (d1)",
                                                  std::vector<int>{})))));
}

TEST_F(TileAnalysisTest, FusionOpWithSingleBinaryOp) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule m
    f {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT a0 = f32[100] add(p0, p1)
    }
    ENTRY e {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT fusion = f32[100] fusion(p0, p1), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(
      input_indexing.operand_indexing_maps,
      UnorderedElementsAre(
          MatchOperandIndexing(0, ElementsAre(MatchIndexingMap(
                                      "(d0) -> (d0)", std::vector<int>{}))),
          MatchOperandIndexing(1, ElementsAre(MatchIndexingMap(
                                      "(d0) -> (d0)", std::vector<int>{})))));
}

TEST_F(TileAnalysisTest, FusionOpTensorPlusTransposedTensor) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule m
    f {
      p0 = f32[1000, 1000] parameter(0)
      transpose_p0 = f32[1000, 1000]{0, 1} transpose(p0), dimensions={1, 0}
      ROOT a0 = f32[1000, 1000] add(p0, transpose_p0)
    }
    ENTRY e {
      p0 = f32[1000,1000] parameter(0)
      ROOT fusion = f32[1000,1000] fusion(p0), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(
      input_indexing.operand_indexing_maps,
      ElementsAre(MatchOperandIndexing(
          0,
          UnorderedElementsAre(
              MatchIndexingMap("(d0, d1) -> (d1, d0)", std::vector<int>{}),
              MatchIndexingMap("(d0, d1) -> (d0, d1)", std::vector<int>{})))));
}

TEST_F(TileAnalysisTest, FusionExponentialDuplication) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule test_module
    ENTRY entry_computation {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      add0 = f32[4] add(p0, p1)
      slice1.0 = f32[3] slice(add0), slice={[0:3]}
      slice1.1 = f32[3] slice(add0), slice={[1:4]}
      add1 = f32[3]{0} add(slice1.0, slice1.1)
      slice2.0 = f32[2] slice(add1), slice={[0:2]}
      slice2.1 = f32[2] slice(add1), slice={[1:3]}
      ROOT add2 = f32[2] add(slice2.0, slice2.1)
  })"));
  EXPECT_THAT(
      input_indexing.operand_indexing_maps,
      ElementsAre(
          MatchOperandIndexing(0, ElementsAre(MatchIndexingMap(
                                      "(d0) -> (d0)", std::vector<int>{}))),
          MatchOperandIndexing(1, ElementsAre(MatchIndexingMap(
                                      "(d0) -> (d0)", std::vector<int>{})))));
}

TEST_F(TileAnalysisTest, FusionOpWithReduceOfReduce) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }
    f {
      p0 = f32[150, 20, 10, 50] parameter(0)
      p0_init = f32[] parameter(1)
      reduce_1 = f32[20, 10] reduce(p0, p0_init),
        dimensions={0, 3}, to_apply=max
      ROOT reduce_2 = f32[10] reduce(reduce_1, p0_init),
        dimensions={0}, to_apply=max
    }
    ENTRY e {
      p0 = f32[150, 20, 10, 50] parameter(0)
      p0_init = f32[] constant(-inf)
      ROOT fusion = f32[10] fusion(p0, p0_init), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.operand_indexing_maps,
              ElementsAre(MatchOperandIndexing(
                  0, ElementsAre(MatchIndexingMap(
                         "(d0)[s0, s1, s2] -> (s0, s2, d0, s1)",
                         std::vector<int>{150, 50, 20})))));
}

TEST_F(TileAnalysisTest, FusionOpWithReduceOfBroadcast) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }
    f {
      p0 = f32[15, 20] parameter(0)
      p0_init = f32[] parameter(1)
      p0_bcast = f32[15, 32, 20, 64] broadcast(p0), dimensions={0, 2}

      ROOT reduce_2 = f32[15, 64] reduce(p0_bcast, p0_init),
        dimensions={1, 2}, to_apply=max
    }
    ENTRY e {
      p0 = f32[15, 20] parameter(0)
      p0_init = f32[] constant(-inf)
      ROOT fusion = f32[15, 64] fusion(p0, p0_init), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.operand_indexing_maps,
              ElementsAre(MatchOperandIndexing(
                  0, ElementsAre(MatchIndexingMap("(d0, d1)[s0] -> (d0, s0)",
                                                  std::vector<int>{20})))));
}

TEST_F(TileAnalysisTest, FusionOpWithTransposeOfTranspose) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule m
    f {
      p0 = f32[20, 10, 50] parameter(0)

      lhs_transpose_1 = f32[10, 20, 50]
             transpose(p0), dimensions={1, 0, 2}
      lhs_e = f32[10, 20, 50] exponential(lhs_transpose_1)
      lhs_transpose_2 = f32[10, 50, 20]
             transpose(lhs_e), dimensions={0, 2, 1}

      rhs_transpose_1 = f32[50, 10, 20]
             transpose(p0), dimensions={2, 1, 0}
      rhs_log = f32[50, 10, 20] exponential(rhs_transpose_1)
      rhs_transpose_2 = f32[10, 50, 20]
             transpose(rhs_log), dimensions={1, 0, 2}

      ROOT add = f32[10, 50, 20] add(lhs_transpose_2, rhs_transpose_2)
    }
    ENTRY e {
      p0 = f32[20, 10, 50] parameter(0)
      ROOT fusion = f32[10, 50, 20] fusion(p0), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(
      input_indexing.operand_indexing_maps,
      ElementsAre(MatchOperandIndexing(
          0, ElementsAre(MatchIndexingMap("(d0, d1, d2) -> (d2, d0, d1)",
                                          std::vector<int>{})))));
}

TEST_F(TileAnalysisTest, FusionOpWithReducedSlice) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }
    f {
      p0 = f32[150, 64, 1024] parameter(0)
      p0_init = f32[] parameter(1)
      p0_slice = f32[16, 32, 128] slice(f32[150, 64, 1024] p0),
                slice={[5:21:1], [0:64:2], [50:434:3]}
      ROOT reduce = f32[32] reduce(p0_slice, p0_init),
        dimensions={0, 2}, to_apply=max
    }
    ENTRY e {
      p0 = f32[150, 64, 1024] parameter(0)
      p0_init = f32[] constant(-inf)
      ROOT fusion = f32[32] fusion(p0, p0_init), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.operand_indexing_maps,
              ElementsAre(MatchOperandIndexing(
                  0, ElementsAre(MatchIndexingMap(
                         "(d0)[s0, s1] -> (s0 + 5, d0 * 2, s1 * 3 + 50)",
                         std::vector<int>{16, 128})))));
}

TEST_F(TileAnalysisTest, FusionOpWithSliceOfSlice) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule m
    f {
      p0 = f32[150, 64, 1024] parameter(0)
      p0_slice_1 = f32[16, 32, 128] slice(f32[150, 64, 1024] p0),
                slice={[5:21:1], [0:64:2], [50:434:3]}
      ROOT p0_slice_2 = f32[7, 9, 24] slice(f32[16, 32, 128] p0_slice_1),
                slice={[3:16:2], [4:30:3], [5:100:4]}
    }
    ENTRY e {
      p0 = f32[150, 64, 1024] parameter(0)
      ROOT fusion = f32[7, 9, 24] fusion(p0), kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(
      input_indexing.operand_indexing_maps,
      ElementsAre(MatchOperandIndexing(
          0, ElementsAre(MatchIndexingMap(
                 "(d0, d1, d2) -> (d0 * 2 + 8, d1 * 6 + 8, d2 * 12 + 65)",
                 std::vector<int>{})))));
}

TEST_F(TileAnalysisTest, ReduceOp) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
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
  )"));
  EXPECT_THAT(input_indexing.operand_indexing_maps,
              ElementsAre(MatchOperandIndexing(
                  0, ElementsAre(MatchIndexingMap(
                         "(d0, d1)[s0, s1] -> (d0, s0, d1, s1)",
                         std::vector<int>{20, 50})))));
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
      ElementsAre(MatchOperandIndexing(
                      0, ElementsAre(MatchIndexingMap("(d0)[s0] -> (s0, d0)",
                                                      std::vector<int>{256}))),
                  MatchOperandIndexing(
                      1, ElementsAre(MatchIndexingMap(
                             "(d0)[s0] -> (s0, d0)", std::vector<int>{256})))));

  auto input_indexing_1 = ComputeInstructionIndexing(root, 1, &mlir_context_);
  ASSERT_IS_OK(input_indexing_1);
  EXPECT_THAT(
      input_indexing_1->operand_indexing_maps,
      ElementsAre(MatchOperandIndexing(
                      0, ElementsAre(MatchIndexingMap("(d0)[s0] -> (s0, d0)",
                                                      std::vector<int>{256}))),
                  MatchOperandIndexing(
                      1, ElementsAre(MatchIndexingMap(
                             "(d0)[s0] -> (s0, d0)", std::vector<int>{256})))));
}

TEST_F(TileAnalysisTest, ReverseOp) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1, 17, 9, 9] parameter(0)
     ROOT reverse = f32[1, 17, 9, 9] reverse(p0), dimensions={1, 2}
    }
  )"));
  EXPECT_THAT(input_indexing.operand_indexing_maps,
              ElementsAre(MatchOperandIndexing(
                  0, ElementsAre(MatchIndexingMap(
                         "(d0, d1, d2, d3) -> (d0, -d1 + 17, -d2 + 9, d3)",
                         std::vector<int>{})))));
}

TEST_F(TileAnalysisTest, SliceOp) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20, 50] parameter(0)
      ROOT slice = f32[5, 3, 25] slice(f32[10, 20, 50] p0),
          slice={[5:10:1], [3:20:7], [0:50:2]}
    }
  )"));
  EXPECT_THAT(input_indexing.operand_indexing_maps,
              ElementsAre(MatchOperandIndexing(
                  0, ElementsAre(MatchIndexingMap(
                         "(d0, d1, d2) -> (d0 + 5, d1 * 7 + 3, d2 * 2)",
                         std::vector<int>{})))));
}

TEST_F(TileAnalysisTest, TransposeOp) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f16[1, 8, 1536, 512] parameter(0)
      ROOT transpose = f16[1, 8, 512, 1536]{2, 3, 1, 0}
             transpose(p0), dimensions={0, 1, 3, 2}
    }
  )"));
  EXPECT_THAT(input_indexing.operand_indexing_maps,
              ElementsAre(MatchOperandIndexing(
                  0, ElementsAre(MatchIndexingMap(
                         "(d0, d1, d2, d3) -> (d0, d1, d3, d2)",
                         std::vector<int>{})))));
}

TEST_F(TileAnalysisTest, DotOp) {
  TF_ASSERT_OK_AND_ASSIGN(auto input_indexing,
                          GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 38, 17, 11, 18, 10] parameter(0)
      p1 = f32[17, 10, 16, 18, 22, 38] parameter(1)
      ROOT dot = f32[10, 38, 4, 11, 16, 22] dot(p0, p1),
        lhs_batch_dims={5,1}, rhs_batch_dims={1,5},
        lhs_contracting_dims={4,2}, rhs_contracting_dims={3,0}
    }
  )"));
  EXPECT_THAT(
      input_indexing.operand_indexing_maps,
      ElementsAre(
          MatchOperandIndexing(0, ElementsAre(MatchIndexingMap(
                                      "(d0, d1, d2, d3, d4, d5)[s0, s1] -> "
                                      "(d2, d1, s1, d3, s0, d0)",
                                      std::vector<int>{18, 17}))),
          MatchOperandIndexing(1, ElementsAre(MatchIndexingMap(
                                      "(d0, d1, d2, d3, d4, d5)[s0, s1] -> "
                                      "(s1, d0, d4, s0, d5, d1)",
                                      std::vector<int>{18, 17})))));
}

TEST_F(TileAnalysisTest, UnsupportedOps) {
  ASSERT_IS_NOT_OK(GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1, 17, 9, 9] parameter(0)
      p1 = f32[5, 17, 9, 9] parameter(1)
      ROOT concat = f32[6, 17, 9, 9] concatenate(p0, p1)
    }
  )"));
  ASSERT_IS_NOT_OK(GetIndexingMapsForEntryComputation(R"(
    HloModule m
    ENTRY e {
      input = s32[1,1,25,1] parameter(0)
      update = s32[1,1,2,1] parameter(1)
      start_indices = s32[4] parameter(2)
      ROOT dyn-update = s32[1,1,25,1] dynamic-update-slice(
        s32[1,1,25,1] input, s32[1,1,2,1] update, s32[4] start_indices)
    }
  )"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
