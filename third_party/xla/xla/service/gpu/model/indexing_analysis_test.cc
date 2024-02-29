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

#include "xla/service/gpu/model/indexing_analysis.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/fusions/tiling_util.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::ExplainMatchResult;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

MATCHER_P2(MatchInstrIndexing, operand_id, indexing_map_matchers, "") {
  return ExplainMatchResult(Eq(operand_id), arg.operand_id, result_listener) &&
         ExplainMatchResult(indexing_map_matchers, arg.indexing_maps,
                            result_listener);
}

class IndexingAnalysisTest : public HloTestBase {
 public:
  HloInstructionIndexing GetOutputToInputIndexingForEntryComputation(
      absl::string_view hlo_string, int output_id = 0,
      bool use_physical_layout = false) {
    return ComputeOutputToInputIndexingForEntryComputation(
        static_cast<HloTestBase*>(this), &mlir_context_, hlo_string, output_id,
        use_physical_layout);
  }

  HloInstructionIndexing GetInputToOutputIndexingForEntryComputation(
      absl::string_view hlo_string, int input_id = 0,
      bool use_physical_layout = false) {
    return ComputeInputToOutputIndexingForEntryComputation(
        static_cast<HloTestBase*>(this), &mlir_context_, hlo_string, input_id,
        use_physical_layout);
  }
  mlir::MLIRContext mlir_context_;
};

TEST_F(IndexingAnalysisTest, FuseProducerConsumerOutputToInputIndexing) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1000, 1000] parameter(0)
      transpose_p0 = f32[1000, 1000]{0, 1} transpose(p0), dimensions={1, 0}
      ROOT a0 = f32[1000, 1000] add(p0, transpose_p0)
    }
  )");
  EXPECT_TRUE(module.ok());
  const HloInstruction* root =
      module.value()->entry_computation()->root_instruction();
  const HloInstruction* parameter = root->operand(0);
  const HloInstruction* transpose = root->operand(1);

  auto root_indexing =
      ComputeOutputToInputIndexing(root, /*output_id=*/0, &mlir_context_);

  auto grouped_by_key = GroupIndexingMapsByProducers(root_indexing, root);

  EXPECT_THAT(
      grouped_by_key,
      UnorderedElementsAre(Pair(parameter, ElementsAre(MatchIndexingMap(R"(
                    (d0, d1) -> (d0, d1)
                    domain:
                    d0 in [0, 999]
                    d1 in [0, 999]
                  )"))),
                           Pair(transpose, ElementsAre(MatchIndexingMap(R"(
                    (d0, d1) -> (d0, d1)
                    domain:
                    d0 in [0, 999]
                    d1 in [0, 999]
                  )")))));
}

TEST_F(IndexingAnalysisTest, ComputeGroupedOutputToInputIndexing) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1000, 1000] parameter(0)
      transpose_p0 = f32[1000, 1000]{0, 1} transpose(p0), dimensions={1, 0}
      ROOT a0 = f32[1000, 1000] add(p0, transpose_p0)
    }
  )");
  EXPECT_TRUE(module.ok());
  const HloInstruction* root =
      (*module)->entry_computation()->root_instruction();
  const HloInstruction* parameter = root->operand(0);
  const HloInstruction* transpose = root->operand(1);

  auto fusion_adaptor = ProducerConsumerFusion(transpose, root);

  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      fusion_adaptor, fusion_adaptor.GetRoots()[0], &mlir_context_);
  EXPECT_THAT(grouped_indexing,
              UnorderedElementsAre(
                  Pair(root, ElementsAre(MatchIndexingMap(R"(
                    (d0, d1) -> (d0, d1)
                    domain:
                    d0 in [0, 999]
                    d1 in [0, 999]
                  )"))),
                  Pair(transpose, ElementsAre(MatchIndexingMap(R"(
                    (d0, d1) -> (d0, d1)
                    domain:
                    d0 in [0, 999]
                    d1 in [0, 999]
                  )"))),
                  Pair(parameter, UnorderedElementsAre(MatchIndexingMap(R"(
                        (d0, d1) -> (d0, d1)
                        domain:
                        d0 in [0, 999]
                        d1 in [0, 999]
                      )"),
                                                       MatchIndexingMap(R"(
                        (d0, d1) -> (d1, d0)
                        domain:
                        d0 in [0, 999]
                        d1 in [0, 999]
                      )")))));
}

TEST_F(IndexingAnalysisTest,
       ComputeGroupedOutputToInputIndexing_VariadicReduce) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

add {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  param_2 = f32[] parameter(2)
  param_3 = f32[] parameter(3)
  add.0 = f32[] add(param_0, param_2)
  add.1 = f32[] add(param_1, param_3)
  ROOT t = (f32[], f32[]) tuple(add.0, add.1)
}

ENTRY entry_computation {
  param_0.3 = f32[32,40]{1,0} parameter(0)
  param_1.3 = f32[32,40]{1,0} parameter(1)
  param_2.2 = f32[] parameter(2)
  constant = f32[] constant(0)
  ROOT reduce = (f32[32]{0}, f32[32]{0}) reduce(param_0.3, param_1.3, param_2.2, constant), dimensions={1}, to_apply=add
}
  )");
  EXPECT_TRUE(module.ok());
  const HloInstruction* root =
      (*module)->entry_computation()->root_instruction();

  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);

  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, fusion_adaptor->GetRoots()[0], &mlir_context_);

  EXPECT_THAT(grouped_indexing,
              UnorderedElementsAre(
                  Pair(root, ElementsAre(MatchIndexingMap(R"(
                    (d0) -> (d0)
                    domain:
                    d0 in [0, 31]
                  )"))),
                  Pair(root->operand(0), ElementsAre(MatchIndexingMap(R"(
                    (d0)[s0] -> (d0, s0)
                    domain:
                    d0 in [0, 31]
                    s0 in [0, 39]
                  )"))),
                  Pair(root->operand(1), ElementsAre(MatchIndexingMap(R"(
                    (d0)[s0] -> (d0, s0)
                    domain:
                    d0 in [0, 31]
                    s0 in [0, 39]
                  )"))),
                  Pair(root->operand(2), ElementsAre(MatchIndexingMap(R"(
                    (d0) -> ()
                    domain:
                    d0 in [0, 31]
                  )"))),
                  Pair(root->operand(3), ElementsAre(MatchIndexingMap(R"(
                    (d0) -> ()
                    domain:
                    d0 in [0, 31]
                  )")))));
}

TEST_F(IndexingAnalysisTest, ComputeGroupedOutputToInputIndexing_SingleOp) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1000, 1000] parameter(0)
      p1 = f32[1000, 1000] parameter(1)
      exp0 = f32[1000, 1000] exponential(p1)
      ROOT a0 = f32[1000, 1000] add(p0, exp0)
    }
  )");
  EXPECT_TRUE(module.ok());
  HloComputation* entry_computation = (*module)->entry_computation();
  const HloInstruction* exponential =
      entry_computation->GetInstructionWithName("exp0");
  const HloInstruction* parameter =
      entry_computation->GetInstructionWithName("p1");

  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(exponential);
  HloInstructionAdaptor parameter_adaptor(*parameter);
  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, parameter_adaptor, &mlir_context_);
  EXPECT_THAT(grouped_indexing, UnorderedElementsAre(Pair(
                                    parameter, ElementsAre(MatchIndexingMap(R"(
                                                     (d0, d1) -> (d0, d1)
                                                     domain:
                                                     d0 in [0, 999]
                                                     d1 in [0, 999]
                                                   )")))));
}

TEST_F(IndexingAnalysisTest,
       ComputeGroupedOutputToInputIndexing_StartNotAtRoot) {
  auto module = ParseAndReturnVerifiedModule(R"(
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
  )");
  EXPECT_TRUE(module.ok());

  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(
      (*module)->entry_computation()->root_instruction());
  auto root = fusion_adaptor->GetRoots()[0];
  auto bcast = root.GetOperand(0);
  auto parameter_0 = bcast.GetOperand(0);

  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, bcast, &mlir_context_);
  EXPECT_THAT(
      grouped_indexing,
      UnorderedElementsAre(
          Pair(&bcast.instruction(), ElementsAre(MatchIndexingMap(R"(
            (d0, d1, d2, d3) -> (d0, d1, d2, d3)
            domain:
            d0 in [0, 14]
            d1 in [0, 31]
            d2 in [0, 19]
            d3 in [0, 63]
          )"))),
          Pair(&parameter_0.instruction(), ElementsAre(MatchIndexingMap(R"(
            (d0, d1, d2, d3) -> (d0, d2)
            domain:
            d0 in [0, 14]
            d1 in [0, 31]
            d2 in [0, 19]
            d3 in [0, 63]
          )")))));
}

TEST_F(IndexingAnalysisTest, PhysicalLayoutTestOutputPermutation) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20, 30] parameter(0)
      ROOT add0 = f32[10, 20, 30]{1, 0, 2} exponential(p0)
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(
      ir, /*output_id=*/0, /*use_physical_layout=*/true);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d1, d2, d0)
                            domain:
                            d0 in [0, 29]
                            d1 in [0, 9]
                            d2 in [0, 19]
                          )"))));

  auto output_indexing = GetInputToOutputIndexingForEntryComputation(
      ir, /*input_id=*/0, /*use_physical_layout=*/true);
  EXPECT_THAT(output_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d2, d0, d1)
                            domain:
                            d0 in [0, 9]
                            d1 in [0, 19]
                            d2 in [0, 29]
                          )"))));
}

TEST_F(IndexingAnalysisTest, CopyNothing) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      p0 = f32[0, 0]{0,1} parameter(0)
      ROOT copy0 = f32[0, 0]{1,0} copy(p0)
    }
  )";
  auto input_indexing =
      GetOutputToInputIndexingForEntryComputation(ir, /*output_id=*/0);
  input_indexing.Simplify();
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0, d1)
                            domain:
                            d0 in [0, -1]
                            d1 in [0, -1]
                          )"))));

  auto output_indexing =
      GetInputToOutputIndexingForEntryComputation(ir, /*input_id=*/0);
  output_indexing.Simplify();
  EXPECT_THAT(output_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0, d1)
                            domain:
                            d0 in [0, -1]
                            d1 in [0, -1]
                          )"))));
}

TEST_F(IndexingAnalysisTest, PhysicalLayoutTestInputPermutation) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20, 30]{1, 0, 2} parameter(0)
      ROOT add0 = f32[10, 20, 30] exponential(p0)
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(
      ir, /*output_id=*/0, /*use_physical_layout=*/true);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d2, d0, d1)
                            domain:
                            d0 in [0, 9]
                            d1 in [0, 19]
                            d2 in [0, 29]
                          )"))));

  auto output_indexing = GetInputToOutputIndexingForEntryComputation(
      ir, /*input_id=*/0, /*use_physical_layout=*/true);
  EXPECT_THAT(output_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d1, d2, d0)
                            domain:
                            d0 in [0, 29]
                            d1 in [0, 9]
                            d2 in [0, 19]
                          )"))));
}

TEST_F(IndexingAnalysisTest, PhysicalLayoutTestInputAndOutputPermutation) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20, 30]{1, 0, 2} parameter(0)
      ROOT add0 = f32[10, 20, 30]{1, 0, 2} exponential(p0)
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(
      ir, /*output_id=*/0, /*use_physical_layout=*/true);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1, d2)
                            domain:
                            d0 in [0, 29]
                            d1 in [0, 9]
                            d2 in [0, 19]
                          )"))));

  auto output_indexing = GetInputToOutputIndexingForEntryComputation(
      ir, /*input_id=*/0, /*use_physical_layout=*/true);
  EXPECT_THAT(output_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1, d2)
                            domain:
                            d0 in [0, 29]
                            d1 in [0, 9]
                            d2 in [0, 19]
                          )"))));
}

TEST_F(IndexingAnalysisTest, ElementwiseOp) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20] parameter(0)
      p1 = f32[10, 20] parameter(1)
      ROOT add0 = f32[10, 20] add(p0, p1)
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0, d1)
                            domain:
                            d0 in [0, 9]
                            d1 in [0, 19]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0, d1)
                            domain:
                            d0 in [0, 9]
                            d1 in [0, 19]
                          )"))));

  auto output_indexing_0 =
      GetInputToOutputIndexingForEntryComputation(ir, /*input_id=*/0);
  EXPECT_THAT(output_indexing_0.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0, d1)
                            domain:
                            d0 in [0, 9]
                            d1 in [0, 19]
                          )"))));

  auto output_indexing_1 =
      GetInputToOutputIndexingForEntryComputation(ir, /*input_id=*/1);
  EXPECT_THAT(output_indexing_1.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0, d1)
                            domain:
                            d0 in [0, 9]
                            d1 in [0, 19]
                          )"))));
}

TEST_F(IndexingAnalysisTest, BitcastIsReshape) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 32] parameter(0)
      ROOT bitcast = f32[4, 8, 4] bitcast(p0)
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1 * 4 + d2)
                            domain:
                            d0 in [0, 3]
                            d1 in [0, 7]
                            d2 in [0, 3]
                          )"))));
}

TEST_F(IndexingAnalysisTest, BitcastIsTranspose) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[3, 12288, 6, 128] parameter(0)
      ROOT bitcast = f32[3, 6, 128, 12288] {2, 1, 3, 0} bitcast(p0)
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2, d3) -> (d0, d3, d1, d2)
                            domain:
                            d0 in [0, 2]
                            d1 in [0, 5]
                            d2 in [0, 127]
                            d3 in [0, 12287]
                          )"))));
}

TEST_F(IndexingAnalysisTest, BitcastIsTransposeReshapeTranspose) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      p0 = f32[16, 17, 3] parameter(0)
      ROOT bitcast = f32[51, 16] {0, 1} bitcast(p0)
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d1, d0 floordiv 3, d0 mod 3)
                            domain:
                            d0 in [0, 50]
                            d1 in [0, 15]
                          )"))));
  auto output_indexing = GetInputToOutputIndexingForEntryComputation(ir);
  EXPECT_THAT(output_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d1 * 3 + d2, d0)
                            domain:
                            d0 in [0, 15]
                            d1 in [0, 16]
                            d2 in [0, 2]
                          )"))));
}

TEST_F(IndexingAnalysisTest, BroadcastOp) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      p0 = f32[20] parameter(0)
      ROOT bc0 = f32[10, 20, 30] broadcast(p0), dimensions={1}
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d1)
                            domain:
                            d0 in [0, 9]
                            d1 in [0, 19]
                            d2 in [0, 29]
                          )"))));

  auto output_indexing = GetInputToOutputIndexingForEntryComputation(ir);
  EXPECT_THAT(output_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0)[s0, s1] -> (s0, d0, s1)
                            domain:
                            d0 in [0, 19]
                            s0 in [0, 9]
                            s1 in [0, 29]
                          )"))));
}

TEST_F(IndexingAnalysisTest, ConstantOp) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      ROOT c1 = bf16[17, 22] constant(1)
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(input_indexing.indexing_maps, IsEmpty());
}

TEST_F(IndexingAnalysisTest, ConcatenateOp) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      p0 = f32[2, 5, 7] parameter(0)
      p1 = f32[2, 11, 7] parameter(1)
      p2 = f32[2, 17, 7] parameter(2)
      ROOT concat = f32[2, 33, 7] concatenate(
        f32[2, 5, 7] p0, f32[2, 11, 7] p1, f32[2, 17, 7] p2), dimensions={1}
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1, d2)
                            domain:
                            d0 in [0, 1]
                            d1 in [0, 4]
                            d2 in [0, 6]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1 - 5, d2)
                            domain:
                            d0 in [0, 1]
                            d1 in [5, 15]
                            d2 in [0, 6]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1 - 16, d2)
                            domain:
                            d0 in [0, 1]
                            d1 in [16, 32]
                            d2 in [0, 6]
                          )"))));

  auto output_indexing_0 =
      GetInputToOutputIndexingForEntryComputation(ir, /*input_id=*/0);
  EXPECT_THAT(output_indexing_0.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1, d2)
                            domain:
                            d0 in [0, 1]
                            d1 in [0, 4]
                            d2 in [0, 6]
                          )"))));

  auto output_indexing_1 =
      GetInputToOutputIndexingForEntryComputation(ir, /*input_id=*/1);
  EXPECT_THAT(output_indexing_1.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1 + 5, d2)
                            domain:
                            d0 in [0, 1]
                            d1 in [0, 10]
                            d2 in [0, 6]
                          )"))));

  auto output_indexing_2 =
      GetInputToOutputIndexingForEntryComputation(ir, /*input_id=*/2);
  EXPECT_THAT(output_indexing_2.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1 + 16, d2)
                            domain:
                            d0 in [0, 1]
                            d1 in [0, 16]
                            d2 in [0, 6]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpWithSingleBinaryOp) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
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
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0) -> (d0)
                            domain:
                            d0 in [0, 99]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0) -> (d0)
                            domain:
                            d0 in [0, 99]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpWithDot) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    f {
      p0 = s8[3,12288,6,128]{3,2,1,0} parameter(0)
      bitcast1 = s8[3,6,128,12288]{2,1,3,0} bitcast(p0)
      copy1 = s8[3,6,128,12288]{3,2,1,0} copy(bitcast1)
      bitcast2 = s8[2304,12288]{1,0} bitcast(copy1)
      convert1 = bf16[2304,12288]{1,0} convert(bitcast2)
      bitcast3 = bf16[2304,16,768]{2,1,0} bitcast(convert1)
      p3 = bf16[16,12288]{1,0} parameter(3)
      convert2 = f32[16,12288]{1,0} convert(p3)
      p4 = bf16[16,12288]{1,0} parameter(4)
      convert3 = f32[16,12288]{1,0} convert(p4)
      add1 = f32[16,12288]{1,0} add(convert2, convert3)
      p2 = bf16[16]{0} parameter(2)
      convert15 = f32[16]{0} convert(p2)
      rsqrt = f32[16]{0} rsqrt(convert15)
      convert4 = bf16[16]{0} convert(rsqrt)
      bcast1 = bf16[16,12288]{1,0} broadcast(convert4), dimensions={0}
      convert5 = f32[16,12288]{1,0} convert(bcast1)
      multiply1 = f32[16,12288]{1,0} multiply(add1, convert5)
      p1 = bf16[12288]{0} parameter(1)
      convert6 = f32[12288]{0} convert(p1)
      c1 = bf16[] constant(1)
      bcast2 = bf16[12288]{0} broadcast(c1), dimensions={}
      convert7 = f32[12288]{0} convert(bcast2)
      add2 = f32[12288]{0} add(convert6, convert7)
      convert8 = bf16[12288]{0} convert(add2)
      bcast3 = bf16[16,12288]{1,0} broadcast(convert8), dimensions={1}
      convert9 = f32[16,12288]{1,0} convert(bcast3)
      multiply2 = f32[16,12288]{1,0} multiply(multiply1, convert9)
      convert10 = bf16[16,12288]{1,0} convert(multiply2)
      bcast4 = bf16[16,16,768]{2,1,0} bitcast(convert10)
      dot = bf16[16,2304,16]{2,1,0} dot(bitcast3, bcast4),
        lhs_batch_dims={1}, lhs_contracting_dims={2},
        rhs_batch_dims={1}, rhs_contracting_dims={2}
      bcast5 = bf16[16,3,6,128,16]{4,3,2,1,0} bitcast(dot)
      copy2 = bf16[16,3,6,128,16]{3,2,4,1,0} copy(bcast5)
      convert13 = f32[16,3,6,128,16]{3,2,4,1,0} convert(copy2)
      p5 = bf16[3,6,128]{2,1,0} parameter(5)
      bcast6 = bf16[3,6,128,16]{2,1,3,0} broadcast(p5), dimensions={0,1,2}
      convert11 = f32[3,6,128,16]{2,1,3,0} convert(bcast6)
      bcast7 = f32[16,3,6,128,16]{3,2,4,1,0} broadcast(convert11),
        dimensions={1,2,3,4}
      multiply3 = f32[16,3,6,128,16]{3,2,4,1,0} multiply(convert13, bcast7)
      convert12 = bf16[16,3,6,128,16]{3,2,4,1,0} convert(multiply3)
      ROOT bcast8 = bf16[16,16,3,1,6,128]{5,4,1,3,2,0} bitcast(convert12)
    }
    ENTRY e {
      p0 = s8[3,12288,6,128]{3,2,1,0} parameter(0)
      p1 = bf16[12288]{0} parameter(1)
      p2 = bf16[16]{0} parameter(2)
      p3 = bf16[16,12288]{1,0} parameter(3)
      p4 = bf16[16,12288]{1,0} parameter(4)
      p5 = bf16[3,6,128]{2,1,0} parameter(5)
      ROOT fusion = bf16[16,16,3,1,6,128]{5,4,1,3,2,0}
        fusion(p0, p1, p2, p3, p4, p5), kind=kLoop, calls=f
    }
  )");

  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                (d0, d1, d2, d3, d4, d5)[s0] -> (d2, d0 * 768 + s0, d4, d5)
                domain:
                d0 in [0, 15]
                d1 in [0, 15]
                d2 in [0, 2]
                d3 in [0, 0]
                d4 in [0, 5]
                d5 in [0, 127]
                s0 in [0, 767]
              )")),
                          ElementsAre(MatchIndexingMap(R"(
                (d0, d1, d2, d3, d4, d5)[s0] -> (d0 * 768 + s0)
                domain:
                d0 in [0, 15]
                d1 in [0, 15]
                d2 in [0, 2]
                d3 in [0, 0]
                d4 in [0, 5]
                d5 in [0, 127]
                s0 in [0, 767]
              )")),
                          ElementsAre(MatchIndexingMap(R"(
                (d0, d1, d2, d3, d4, d5) -> (d1)
                domain:
                d0 in [0, 15]
                d1 in [0, 15]
                d2 in [0, 2]
                d3 in [0, 0]
                d4 in [0, 5]
                d5 in [0, 127]
              )")),
                          ElementsAre(MatchIndexingMap(R"(
                (d0, d1, d2, d3, d4, d5)[s0] -> (d1, d0 * 768 + s0)
                domain:
                d0 in [0, 15]
                d1 in [0, 15]
                d2 in [0, 2]
                d3 in [0, 0]
                d4 in [0, 5]
                d5 in [0, 127]
                s0 in [0, 767]
              )")),
                          ElementsAre(MatchIndexingMap(R"(
                (d0, d1, d2, d3, d4, d5)[s0] -> (d1, d0 * 768 + s0)
                domain:
                d0 in [0, 15]
                d1 in [0, 15]
                d2 in [0, 2]
                d3 in [0, 0]
                d4 in [0, 5]
                d5 in [0, 127]
                s0 in [0, 767]
              )")),
                          ElementsAre(MatchIndexingMap(R"(
                (d0, d1, d2, d3, d4, d5) -> (d2, d4, d5)
                domain:
                d0 in [0, 15]
                d1 in [0, 15]
                d2 in [0, 2]
                d3 in [0, 0]
                d4 in [0, 5]
                d5 in [0, 127]
              )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpWithSoftmax) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    add_computation {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    max_computation {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }
    softmax {
      p0 = f32[2,65,125]{2,1,0} parameter(0)
      bitcast0 = f32[65,2,125]{2,1,0} bitcast(p0)
      constant_neg_inf_1 = f32[] constant(-inf)
      reduce0 = f32[2,65]{1,0} reduce(p0, constant_neg_inf_1),
        dimensions={2}, to_apply=max_computation
      bitcast1 = f32[130]{0} bitcast(reduce0)
      bcast1 = f32[130,125]{1,0} broadcast(bitcast1), dimensions={0}
      bitcast2 = f32[65,2,125]{2,1,0} bitcast(bcast1)
      subtract0 = f32[65,2,125]{2,1,0} subtract(bitcast0, bitcast2)
      exponential0 = f32[65,2,125]{2,1,0} exponential(subtract0)
      bitcast3 = f32[65,2,125]{2,1,0} bitcast(p0)
      reduce1 = f32[2,65]{1,0} reduce(p0, constant_neg_inf_1),
        dimensions={2}, to_apply=max_computation
      bitcast4 = f32[130]{0} bitcast(reduce1)
      bcast2 = f32[130,125]{1,0} broadcast(bitcast4), dimensions={0}
      bitcast5 = f32[65,2,125]{2,1,0} bitcast(bcast2)
      subtract1 = f32[65,2,125]{2,1,0} subtract(bitcast3, bitcast5)
      exponential1 = f32[65,2,125]{2,1,0} exponential(subtract1)
      constant_zero_1 = f32[] constant(0)
      reduce2 = f32[65,2]{1,0} reduce(exponential1, constant_zero_1),
        dimensions={2}, to_apply=add_computation
      bitcast6 = f32[130]{0} bitcast(reduce2)
      bcast3 = f32[130,125]{1,0} broadcast(bitcast6), dimensions={0}
      bitcast7 = f32[65,2,125]{2,1,0} bitcast(bcast3)
      divide = f32[65,2,125]{2,1,0} divide(exponential0, bitcast7)
      ROOT bitcast8 = f32[2,65,125]{2,1,0} bitcast(divide)
    }
    ENTRY e {
      p0 = f32[2,65,125]{2,1,0} parameter(0)
      ROOT fusion = f32[2,65,125]{2,1,0}
        fusion(p0), kind=kLoop, calls=softmax
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(UnorderedElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2)[s0] -> (d0, d1, s0)
                            domain:
                            d0 in [0, 1]
                            d1 in [0, 64]
                            d2 in [0, 124]
                            s0 in [0, 124]
                          )"),
                                               MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1, d2)
                            domain:
                            d0 in [0, 1]
                            d1 in [0, 64]
                            d2 in [0, 124]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpTensorPlusTransposedTensor) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
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
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(UnorderedElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0, d1)
                            domain:
                            d0 in [0, 999]
                            d1 in [0, 999]
                          )"),
                                               MatchIndexingMap(R"(
                            (d0, d1) -> (d1, d0)
                            domain:
                            d0 in [0, 999]
                            d1 in [0, 999]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionExponentialDuplication) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule test_module

    fused_computation {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      add0 = f32[4] add(p0, p1)
      slice1.0 = f32[3] slice(add0), slice={[0:3]}
      slice1.1 = f32[3] slice(add0), slice={[1:4]}
      add1 = f32[3]{0} add(slice1.0, slice1.1)
      slice2.0 = f32[2] slice(add1), slice={[0:2]}
      slice2.1 = f32[2] slice(add1), slice={[1:3]}
      ROOT add2 = f32[2] add(slice2.0, slice2.1)
    }

    ENTRY entry_computation {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      ROOT fusion = f32[2] fusion(p0, p1), kind=kLoop,
      calls=fused_computation
    })");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(UnorderedElementsAre(MatchIndexingMap(R"(
                            (d0) -> (d0 + 1)
                            domain:
                            d0 in [0, 1]
                          )"),
                                               MatchIndexingMap(R"(
                            (d0) -> (d0)
                            domain:
                            d0 in [0, 1]
                          )"),
                                               MatchIndexingMap(R"(
                            (d0) -> (d0 + 2)
                            domain:
                            d0 in [0, 1]
                          )")),
                          UnorderedElementsAre(MatchIndexingMap(R"(
                            (d0) -> (d0 + 2)
                            domain:
                            d0 in [0, 1]
                          )"),
                                               MatchIndexingMap(R"(
                            (d0) -> (d0 + 1)
                            domain:
                            d0 in [0, 1]
                          )"),
                                               MatchIndexingMap(R"(
                            (d0) -> (d0)
                            domain:
                            d0 in [0, 1]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpWithReduceOfReduce) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
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
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0)[s0, s1, s2] -> (s0, s2, d0, s1)
                            domain:
                            d0 in [0, 9]
                            s0 in [0, 149]
                            s1 in [0, 49]
                            s2 in [0, 19]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0) -> ()
                            domain:
                            d0 in [0, 9]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpWithReduceOfBroadcast) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
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
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1)[s0] -> (d0, s0)
                            domain:
                            d0 in [0, 14]
                            d1 in [0, 63]
                            s0 in [0, 19]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> ()
                            domain:
                            d0 in [0, 14]
                            d1 in [0, 63]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpWithTransposeOfTranspose) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
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
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d2, d0, d1)
                            domain:
                            d0 in [0, 9]
                            d1 in [0, 49]
                            d2 in [0, 19]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpWithReducedSlice) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
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
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0)[s0, s1] -> (s0 + 5, d0 * 2, s1 * 3 + 50)
                            domain:
                            d0 in [0, 31]
                            s0 in [0, 15]
                            s1 in [0, 127]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0) -> ()
                            domain:
                            d0 in [0, 31]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpWithReshape_CollapseOfExpand) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    f {
      p0 = f32[128] parameter(0)
      expand = f32[8, 16] reshape(p0)
      ROOT collapse = f32[128] reshape(expand)
    }
    ENTRY e {
      p0 = f32[128] parameter(0)
      ROOT fusion = f32[128] fusion(p0), kind=kLoop, calls=f
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0) -> (d0)
                            domain:
                            d0 in [0, 127]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpWithReshape_ExpandOfCollapse) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    f {
      p0 = f32[8, 16] parameter(0)
      collapse = f32[128] reshape(p0)
      ROOT expand = f32[8, 16] reshape(collapse)
    }
    ENTRY e {
      p0 = f32[8, 16] parameter(0)
      ROOT fusion = f32[8, 16] fusion(p0), kind=kLoop, calls=f
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0, d1)
                            domain:
                            d0 in [0, 7]
                            d1 in [0, 15]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpWithReshape_ChainedGenericReshapes) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    f {
      p0 = f32[10, 10, 10] parameter(0)
      reshape1 = f32[50, 20] reshape(p0)
      ROOT reshape2 = f32[10, 10, 10] reshape(reshape1)
    }
    ENTRY e {
      p0 = f32[10, 10, 10] parameter(0)
      ROOT fusion = f32[10, 10, 10] fusion(p0), kind=kLoop, calls=f
    }
  )");
  // TODO(jreiffers): Remove the redundant constraint.
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1, d2)
                            domain:
                            d0 in [0, 9]
                            d1 in [0, 9]
                            d2 in [0, 9]
                            d1 * 10 + d2 - (d1 floordiv 2) * 20 in [0, 19]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpWithSliceOfSlice) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
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
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0 * 2 + 8,
                                             d1 * 6 + 8,
                                             d2 * 12 + 65)
                            domain:
                            d0 in [0, 6]
                            d1 in [0, 8]
                            d2 in [0, 23]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpSliceOfAllConcatenateOpInputs) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    f {
      p0 = f32[2, 5, 7] parameter(0)
      p1 = f32[2, 11, 7] parameter(1)
      p2 = f32[2, 17, 7] parameter(2)
      concat = f32[2, 33, 7] concatenate(
        f32[2, 5, 7] p0, f32[2, 11, 7] p1, f32[2, 17, 7] p2), dimensions={1}
      ROOT slice = f32[2, 11, 7] slice(f32[2, 33, 7] concat),
        slice={[0:2:1], [0:33:3], [0:7:1]}
    }
    ENTRY e {
      p0 = f32[2, 5, 7] parameter(0)
      p1 = f32[2, 11, 7] parameter(1)
      p2 = f32[2, 17, 7] parameter(2)
      ROOT fusion = f32[2, 11, 7] fusion(p0, p1, p2), kind=kLoop, calls=f
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1 * 3, d2)
                            domain:
                            d0 in [0, 1]
                            d1 in [0, 1]
                            d2 in [0, 6]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1 * 3 - 5, d2)
                            domain:
                            d0 in [0, 1]
                            d1 in [2, 5]
                            d2 in [0, 6]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1 * 3 - 16, d2)
                            domain:
                            d0 in [0, 1]
                            d1 in [6, 10]
                            d2 in [0, 6]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpSliceOfOneOfConcatenateOpInputs) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    f {
      p0 = f32[2, 5, 7] parameter(0)
      p1 = f32[2, 11, 7] parameter(1)
      p2 = f32[2, 17, 7] parameter(2)
      concat = f32[2, 33, 7] concatenate(
        f32[2, 5, 7] p0, f32[2, 11, 7] p1, f32[2, 17, 7] p2), dimensions={1}
      ROOT slice = f32[2, 3, 7] slice(f32[2, 33, 7] concat),
        slice={[0:2:1], [0:5:2], [0:7:1]}
    }
    ENTRY e {
      p0 = f32[2, 5, 7] parameter(0)
      p1 = f32[2, 11, 7] parameter(1)
      p2 = f32[2, 17, 7] parameter(2)
      ROOT fusion = f32[2, 3, 7] fusion(p0, p1, p2), kind=kLoop, calls=f
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1 * 2, d2)
                            domain:
                            d0 in [0, 1]
                            d1 in [0, 2]
                            d2 in [0, 6]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1 * 2 - 5, d2)
                            domain:
                            d0 in [0, 1]
                            d1 in [3, 2]
                            d2 in [0, 6]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0, d1 * 2 - 16, d2)
                            domain:
                            d0 in [0, 1]
                            d1 in [8, 2]
                            d2 in [0, 6]
                          )"))));
}

TEST_F(IndexingAnalysisTest, FusionOpReshapeOfConcat) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    f {
      p0 = f32[2] parameter(0)
      p1 = f32[30] parameter(1)
      concat = f32[32] concatenate(f32[2] p0, f32[30] p1), dimensions={0}
      ROOT reshape = f32[4, 8] reshape(concat)
    }
    ENTRY e {
      p0 = f32[2] parameter(0)
      p1 = f32[30] parameter(1)
      ROOT fusion = f32[4, 8] fusion(p0, p1), kind=kLoop, calls=f
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0 * 8 + d1)
                            domain:
                            d0 in [0, 3]
                            d1 in [0, 7]
                            d0 * 8 + d1 in [0, 1]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0 * 8 + d1 - 2)
                            domain:
                            d0 in [0, 3]
                            d1 in [0, 7]
                            d0 * 8 + d1 in [2, 31]
                          )"))));
}

TEST_F(IndexingAnalysisTest, IotaOp) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      ROOT iota = s32[5,5,111,42] iota(), iota_dimension=0
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(input_indexing.indexing_maps, IsEmpty());
}

TEST_F(IndexingAnalysisTest, ReshapeOpCollapseShape) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4,8] parameter(0)
      ROOT reshape = f32[32] reshape(p0)
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0) -> (d0 floordiv 8, d0 mod 8)
                            domain:
                            d0 in [0, 31]
                          )"))));
}

TEST_F(IndexingAnalysisTest, ReshapeOpExpandShape) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[32] parameter(0)
      ROOT reshape = f32[4, 8] reshape(p0)
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0 * 8 + d1)
                            domain:
                            d0 in [0, 3]
                            d1 in [0, 7]
                          )"))));
}

TEST_F(IndexingAnalysisTest, ReshapeOpExpandAndCollapseShape) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 8, 12] parameter(0)
      ROOT reshape = f32[32, 3, 4] reshape(p0)
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                (d0, d1, d2) -> (d0 floordiv 8, d0 mod 8, d1 * 4 + d2)
                domain:
                d0 in [0, 31]
                d1 in [0, 2]
                d2 in [0, 3]
              )"))));

  auto output_indexing = GetInputToOutputIndexingForEntryComputation(ir);
  EXPECT_THAT(output_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                (d0, d1, d2) -> (d0 * 8 + d1, d2 floordiv 4, d2 mod 4)
                domain:
                d0 in [0, 3]
                d1 in [0, 7]
                d2 in [0, 11]
              )"))));
}

TEST_F(IndexingAnalysisTest, ReshapeOpExpandSubshapeOnly) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[16, 8] parameter(0)
      ROOT reshape = f32[4, 4, 8] reshape(p0)
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                (d0, d1, d2) -> (d0 * 4 + d1, d2)
                domain:
                d0 in [0, 3]
                d1 in [0, 3]
                d2 in [0, 7]
              )"))));
}

TEST_F(IndexingAnalysisTest, ReshapeOpGenericReshape2DTO3D) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4,8] parameter(0)
      ROOT reshape = f32[2, 4, 4] reshape(p0)
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                (d0, d1, d2) -> (d0 * 2 + d1 floordiv 2,
                                d1 * 4 + d2 - (d1 floordiv 2) * 8)
                domain:
                d0 in [0, 1]
                d1 in [0, 3]
                d2 in [0, 3]
              )"))));
}

TEST_F(IndexingAnalysisTest, ReshapeOpGenericReshape3DTO2D) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[2, 4, 4] parameter(0)
      ROOT reshape = f32[4, 8] reshape(p0)
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0 floordiv 2,
                                         d0 * 2 - (d0 floordiv 2) * 4 +
                                           d1 floordiv 4,
                                         d1 mod 4)
                            domain:
                            d0 in [0, 3]
                            d1 in [0, 7]
                          )"))));
}

TEST_F(IndexingAnalysisTest, PadOp) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 4] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[12, 16] pad(p0, p1), padding=1_4_1x4_8_0
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                                        (d0, d1) -> (
                                          (d0 - 1) floordiv 2,
                                          d1 - 4
                                        )
                                        domain:
                                        d0 in [1, 7]
                                        d1 in [4, 7]
                                        (d0 - 1) mod 2 in [0, 0]
                                      )")),
                          ElementsAre(MatchIndexingMap(R"(
                                        (d0, d1) -> ()
                                        domain:
                                        d0 in [0, 11]
                                        d1 in [0, 15]
                                      )"))));
}

TEST_F(IndexingAnalysisTest, PadOpNoInterior) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[2,8] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[10,8] pad(p0, p1), padding=1_7x0_0
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                                        (d0, d1) -> (d0 - 1, d1)
                                        domain:
                                        d0 in [1, 2]
                                        d1 in [0, 7]
                                      )")),
                          ElementsAre(MatchIndexingMap(R"(
                                        (d0, d1) -> ()
                                        domain:
                                        d0 in [0, 9]
                                        d1 in [0, 7]
                                      )"))));
}

TEST_F(IndexingAnalysisTest, PadOpNegativePadding) {
  // The interior padding is applied first (even with negative padding), so we
  // get a size of 5 (7 + 6 - 8).
  // in:     0 1 2 3 4 5 6
  // padded: 0 p 1 p 2 p 3 p 4 p 5 p 6
  // sliced:       p 2 p 3 p
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[7] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[5] pad(p0, p1), padding=-3_-5_1
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                                        (d0) -> ((d0 + 3) floordiv 2)
                                        domain:
                                        d0 in [0, 4]
                                        (d0 + 3) mod 2 in [0, 0]
                                      )")),
                          ElementsAre(MatchIndexingMap(R"(
                                        (d0) -> ()
                                        domain:
                                        d0 in [0, 4]
                                      )"))));
}

TEST_F(IndexingAnalysisTest, ReduceOp) {
  auto ir = R"(
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
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1)[s0, s1] -> (d0, s0, d1, s1)
                            domain:
                            d0 in [0, 149]
                            d1 in [0, 9]
                            s0 in [0, 19]
                            s1 in [0, 49]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> ()
                            domain:
                            d0 in [0, 149]
                            d1 in [0, 9]
                          )"))));

  auto output_indexing = GetInputToOutputIndexingForEntryComputation(ir);
  EXPECT_THAT(output_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2, d3) -> (d0, d2)
                            domain:
                            d0 in [0, 149]
                            d1 in [0, 19]
                            d2 in [0, 9]
                            d3 in [0, 49]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            ()[s0, s1] -> (s0, s1)
                            domain:
                            s0 in [0, 149]
                            s1 in [0, 9]
                          )"))));
}

TEST_F(IndexingAnalysisTest, VariadicReduceOp) {
  absl::string_view ir = R"(
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

  auto output_indexing_0 =
      GetOutputToInputIndexingForEntryComputation(ir, /*output_id=*/0);
  EXPECT_THAT(output_indexing_0.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0)[s0] -> (s0, d0)
                            domain:
                            d0 in [0, 9]
                            s0 in [0, 255]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0)[s0] -> (s0, d0)
                            domain:
                            d0 in [0, 9]
                            s0 in [0, 255]
                          )")),

                          ElementsAre(MatchIndexingMap(R"(
                            (d0) -> ()
                            domain:
                            d0 in [0, 9]
                          )")),

                          ElementsAre(MatchIndexingMap(R"(
                            (d0) -> ()
                            domain:
                            d0 in [0, 9]
                          )"))));

  auto output_indexing_1 =
      GetOutputToInputIndexingForEntryComputation(ir, /*output_id=*/1);
  EXPECT_THAT(output_indexing_1.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0)[s0] -> (s0, d0)
                            domain:
                            d0 in [0, 9]
                            s0 in [0, 255]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0)[s0] -> (s0, d0)
                            domain:
                            d0 in [0, 9]
                            s0 in [0, 255]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0) -> ()
                            domain:
                            d0 in [0, 9]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0) -> ()
                            domain:
                            d0 in [0, 9]
                          )"))));

  auto input_indexing_0 =
      GetInputToOutputIndexingForEntryComputation(ir, /*input_id=*/0);

  EXPECT_THAT(input_indexing_0.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d1)
                            domain:
                            d0 in [0, 255]
                            d1 in [0, 9]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d1)
                            domain:
                            d0 in [0, 255]
                            d1 in [0, 9]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            ()[s0] -> (s0)
                            domain:
                            s0 in [0, 9]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            ()[s0] -> (s0)
                            domain:
                            s0 in [0, 9]
                          )"))));

  auto input_indexing_1 =
      GetInputToOutputIndexingForEntryComputation(ir, /*input_id=*/1);
  EXPECT_THAT(input_indexing_1.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d1)
                            domain:
                            d0 in [0, 255]
                            d1 in [0, 9]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d1)
                            domain:
                            d0 in [0, 255]
                            d1 in [0, 9]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            ()[s0] -> (s0)
                            domain:
                            s0 in [0, 9]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            ()[s0] -> (s0)
                            domain:
                            s0 in [0, 9]
                          )"))));
}

TEST_F(IndexingAnalysisTest, ReduceWindowOp_NoPadding) {
  auto ir = R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }
    ENTRY e {
      c_inf = f32[] constant(-inf)
      p0 = f32[1024, 514]parameter(0)
      ROOT reduce-window = f32[1024, 3] reduce-window(p0, c_inf),
        window={size=1x512 pad=0_0x0_0}, to_apply=max
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1)[s0] -> (d0, d1 + s0)
                            domain:
                            d0 in [0, 1023]
                            d1 in [0, 2]
                            s0 in [0, 511]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> ()
                            domain:
                            d0 in [0, 1023]
                            d1 in [0, 2]
                          )"))));
}

TEST_F(IndexingAnalysisTest, ReduceWindowOp_PaddingAndWindowStride) {
  auto ir = R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }
    ENTRY e {
      c_inf = f32[] constant(-inf)
      p0 = f32[13, 17] parameter(0)
      ROOT reduce-window = f32[7, 17] reduce-window(p0, c_inf),
       window={size=3x2 stride=2x1 pad=1_1x0_1}, to_apply=max
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1)[s0, s1] -> (d0 * 2 + s0 - 1, d1 + s1)
                            domain:
                            d0 in [0, 6]
                            d1 in [0, 16]
                            s0 in [0, 2]
                            s1 in [0, 1]
                            d0 * 2 + s0 in [1, 13]
                            d1 + s1 in [0, 16]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> ()
                            domain:
                            d0 in [0, 6]
                            d1 in [0, 16]
                          )"))));
}

TEST_F(IndexingAnalysisTest, ReduceWindowOp_Dilation) {
  auto ir = R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }
    ENTRY e {
      c_inf = f32[] constant(-inf)
      p0 = f32[2, 3] parameter(0)
      ROOT reduce-window = f32[3, 5] reduce-window(p0, c_inf),
       window={size=1x1 pad=0_0x0_0 lhs_dilate=2x2}, to_apply=max
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0 floordiv 2, d1 floordiv 2)
                            domain:
                            d0 in [0, 2]
                            d1 in [0, 4]
                            d0 mod 2 in [0, 0]
                            d1 mod 2 in [0, 0]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> ()
                            domain:
                            d0 in [0, 2]
                            d1 in [0, 4]
                          )"))));
}

TEST_F(IndexingAnalysisTest, ReduceWindowOp_Variadic) {
  auto ir = R"(
    HloModule m
    combiner {
      a0 = f32[] parameter(0)
      a1 = s32[] parameter(1)
      b0 = f32[] parameter(2)
      b1 = s32[] parameter(3)
      add0 = f32[] add(a0, b0)
      add1 = s32[] add(a1, b1)
      ROOT sum2 = (f32[], s32[]) tuple(add0, add1)
    }
    ENTRY e {
      c_f32 = f32[] constant(-inf)
      c_s32 = s32[] constant(10)
      p0 = f32[2, 3] parameter(0)
      p1 = s32[2, 3] parameter(1)
      ROOT reduce-window = (f32[1, 2], s32[1, 2])
        reduce-window(p0, p1, c_f32, c_s32),
        window={size=2x2 pad=0_0x0_0}, to_apply=combiner
    }
  )";
  auto input_indexing_0 =
      GetOutputToInputIndexingForEntryComputation(ir, /*output_id=*/0);
  EXPECT_THAT(input_indexing_0.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1)[s0, s1] -> (s0, d1 + s1)
                            domain:
                            d0 in [0, 0]
                            d1 in [0, 1]
                            s0 in [0, 1]
                            s1 in [0, 1]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1)[s0, s1] -> (s0, d1 + s1)
                            domain:
                            d0 in [0, 0]
                            d1 in [0, 1]
                            s0 in [0, 1]
                            s1 in [0, 1]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> ()
                            domain:
                            d0 in [0, 0]
                            d1 in [0, 1]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                           (d0, d1) -> ()
                            domain:
                            d0 in [0, 0]
                            d1 in [0, 1]
                          )"))));
  auto input_indexing_1 =
      GetOutputToInputIndexingForEntryComputation(ir, /*output_id=*/1);
  EXPECT_THAT(input_indexing_1.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1)[s0, s1] -> (s0, d1 + s1)
                            domain:
                            d0 in [0, 0]
                            d1 in [0, 1]
                            s0 in [0, 1]
                            s1 in [0, 1]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1)[s0, s1] -> (s0, d1 + s1)
                            domain:
                            d0 in [0, 0]
                            d1 in [0, 1]
                            s0 in [0, 1]
                            s1 in [0, 1]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> ()
                            domain:
                            d0 in [0, 0]
                            d1 in [0, 1]
                          )")),
                          ElementsAre(MatchIndexingMap(R"(
                           (d0, d1) -> ()
                            domain:
                            d0 in [0, 0]
                            d1 in [0, 1]
                          )"))));
}

TEST_F(IndexingAnalysisTest, ReverseOp) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      p0 = f32[1, 17, 9, 9] parameter(0)
      ROOT reverse = f32[1, 17, 9, 9] reverse(p0), dimensions={1, 2}
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2, d3) -> (d0, -d1 + 16, -d2 + 8, d3)
                            domain:
                            d0 in [0, 0]
                            d1 in [0, 16]
                            d2 in [0, 8]
                            d3 in [0, 8]
                          )"))));

  auto output_indexing = GetInputToOutputIndexingForEntryComputation(ir);
  EXPECT_THAT(output_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2, d3) -> (d0, -d1 + 16, -d2 + 8, d3)
                            domain:
                            d0 in [0, 0]
                            d1 in [0, 16]
                            d2 in [0, 8]
                            d3 in [0, 8]
                          )"))));
}

TEST_F(IndexingAnalysisTest, ReverseReshape) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    fused_computation {
      p0 = f32[10, 11] parameter(0)
      reverse.0 = f32[10, 11] reverse(p0), dimensions={0, 1}
      reshape.0 = f32[110] reshape(reverse.0)
      reverse.1 = f32[110] reverse(reshape.0), dimensions={0}
      ROOT reshape.1 = f32[10, 11] reshape(reverse.1)
    }
    ENTRY e {
      p0 = f32[10, 11] parameter(0)
      ROOT fusion = f32[10, 11] fusion(p0), kind=kLoop,
      calls=fused_computation
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0, d1)
                            domain:
                            d0 in [0, 9]
                            d1 in [0, 10]
                          )"))));
}

TEST_F(IndexingAnalysisTest, SliceOp) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20, 50] parameter(0)
      ROOT slice = f32[5, 3, 25] slice(f32[10, 20, 50] p0),
          slice={[5:10:1], [3:20:7], [0:50:2]}
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2) -> (d0 + 5, d1 * 7 + 3, d2 * 2)
                            domain:
                            d0 in [0, 4]
                            d1 in [0, 2]
                            d2 in [0, 24]
                          )"))));
}

TEST_F(IndexingAnalysisTest, TransposeOp) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      p0 = f32[3, 12288, 6, 128] parameter(0)
      ROOT transpose = f32[3, 6, 128, 12288]
        transpose(p0), dimensions={0, 2, 3, 1}
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2, d3) -> (d0, d3, d1, d2)
                            domain:
                            d0 in [0, 2]
                            d1 in [0, 5]
                            d2 in [0, 127]
                            d3 in [0, 12287]
                          )"))));
  auto output_indexing = GetInputToOutputIndexingForEntryComputation(ir);
  EXPECT_THAT(output_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2, d3) -> (d0, d2, d3, d1)
                            domain:
                            d0 in [0, 2]
                            d1 in [0, 12287]
                            d2 in [0, 5]
                            d3 in [0, 127]
                          )"))));
}

TEST_F(IndexingAnalysisTest, TransposeOp4D) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[3, 12288, 6, 128] parameter(0)
      ROOT bitcast = f32[3, 6, 128, 12288] {2, 1, 3, 0} bitcast(p0)
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                            (d0, d1, d2, d3) -> (d0, d3, d1, d2)
                            domain:
                            d0 in [0, 2]
                            d1 in [0, 5]
                            d2 in [0, 127]
                            d3 in [0, 12287]
                          )"))));
}

TEST_F(IndexingAnalysisTest, DotOp) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 38, 17, 11, 18, 10] parameter(0)
      p1 = f32[17, 10, 16, 18, 22, 38] parameter(1)
      ROOT dot = f32[10, 38, 4, 11, 16, 22] dot(p0, p1),
        lhs_batch_dims={5,1}, rhs_batch_dims={1,5},
        lhs_contracting_dims={4,2}, rhs_contracting_dims={3,0}
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(ElementsAre(MatchIndexingMap(R"(
                (d0, d1, d2, d3, d4, d5)[s0, s1] -> (d2, d1, s1, d3, s0, d0)
                domain:
                d0 in [0, 9]
                d1 in [0, 37]
                d2 in [0, 3]
                d3 in [0, 10]
                d4 in [0, 15]
                d5 in [0, 21]
                s0 in [0, 17]
                s1 in [0, 16]
              )")),
                          ElementsAre(MatchIndexingMap(R"(
                (d0, d1, d2, d3, d4, d5)[s0, s1] -> (s1, d0, d4, s0, d5, d1)
                domain:
                d0 in [0, 9]
                d1 in [0, 37]
                d2 in [0, 3]
                d3 in [0, 10]
                d4 in [0, 15]
                d5 in [0, 21]
                s0 in [0, 17]
                s1 in [0, 16]
              )"))));
}

TEST_F(IndexingAnalysisTest, UnsupportedOps) {
  auto ir = R"(
    HloModule m
    ENTRY e {
      input = s32[1,1,25,1] parameter(0)
      update = s32[1,1,2,1] parameter(1)
      start_indices = s32[4] parameter(2)
      ROOT dyn-update = s32[1,1,25,1] dynamic-update-slice(
        input, update, start_indices)
    }
  )";
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(ir);
  EXPECT_THAT(
      input_indexing.indexing_maps,
      ElementsAre(ElementsAre(UndefinedMap()), ElementsAre(UndefinedMap()),
                  ElementsAre(UndefinedMap())));

  auto output_indexing_0 = GetInputToOutputIndexingForEntryComputation(ir, 0);
  EXPECT_THAT(output_indexing_0.indexing_maps,
              ElementsAre(ElementsAre(UndefinedMap())));

  auto output_indexing_1 = GetInputToOutputIndexingForEntryComputation(ir, 1);
  EXPECT_THAT(output_indexing_1.indexing_maps,
              ElementsAre(ElementsAre(UndefinedMap())));

  auto output_indexing_2 = GetInputToOutputIndexingForEntryComputation(ir, 2);
  EXPECT_THAT(output_indexing_2.indexing_maps,
              ElementsAre(ElementsAre(UndefinedMap())));
}

TEST_F(IndexingAnalysisTest, FusionWithUnsupportedOp) {
  auto input_indexing = GetOutputToInputIndexingForEntryComputation(R"(
    HloModule m
    fused_computation {
      input = f32[20, 20] parameter(0)
      start_indices = s32[2] parameter(1)
      lhs = f32[5, 5] dynamic-slice(f32[20,20] input, s32[2] start_indices),
          dynamic_slice_sizes={5, 5}
      rhs = f32[5, 5] slice(f32[20, 20] input),
          slice={[0:20:4], [0:5:1]}
      ROOT add = f32[5, 5] add(lhs, rhs)
    }
    ENTRY e {
      p0 = f32[20, 20] parameter(0)
      p1 = s32[2] parameter(1)
      ROOT fusion = f32[5, 5] fusion(p0, p1), kind=kLoop,
          calls=fused_computation
    }
  )");
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(UnorderedElementsAre(MatchIndexingMap(R"(
                            (d0, d1) -> (d0 * 4, d1)
                            domain:
                            d0 in [0, 4]
                            d1 in [0, 4]
                          )"),
                                               UndefinedMap()),
                          ElementsAre(UndefinedMap())));
}

TEST_F(IndexingAnalysisTest, TilingIndexing) {
  Tiling tiling{/*shape=*/{1024, 256, 16},
                /*tile_sizes=*/{8, 1, 4},
                /*num_threads=*/{1, 4, 4}};
  EXPECT_THAT(GetIndexingMapForTiling(tiling, &mlir_context_).ToString(),
              MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          (d3 floordiv 64) * 8 + s0,
          (d3 mod 64) * 4 + d0 floordiv 4,
          d0 mod 4 + s2 * 4
        )
        domain:
        d0 in [0, 15]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 8191]
        d4 in [0, 0]
        d5 in [0, 0]
        s0 in [0, 7]
        s1 in [0, 0]
        s2 in [0, 3]
      )"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
