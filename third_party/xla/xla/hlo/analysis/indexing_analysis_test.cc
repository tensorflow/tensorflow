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

#include "xla/hlo/analysis/indexing_analysis.h"

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Visitors.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/analysis/stablehlo_indexing_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/hlo/utils/hlo_traversal.h"

namespace xla {
namespace gpu {
namespace {

using ::llvm::dyn_cast;
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

class IndexingAnalysisTest : public IndexingTestBase,
                             public ::testing::WithParamInterface<bool> {
 public:
  using IndexingTestBase::GetInputToOutputIndexing;
  using IndexingTestBase::GetOutputToInputIndexing;

  HloInstructionIndexing GetOutputToInputIndexing(
      const HloInstruction* instr, int output_id,
      bool use_physical_layout) override {
    if (GetParam()) {
      // StableHLO mode
      auto module_ref =
          xla::ConvertHloToStablehlo(mlir_context_, instr->GetModule());
      if (!module_ref.ok()) {
        ADD_FAILURE() << "HLO to StableHLO conversion failed: "
                      << module_ref.status();
        return HloInstructionIndexing::FromIndexingMaps({});
      }
      stablehlo_modules_.push_back(std::move(module_ref.value()));
      auto module_op = *stablehlo_modules_.back();
      mlir::Operation* op = nullptr;
      module_op->walk([&](mlir::Operation* nested_op) {
        if (auto name_loc = dyn_cast<mlir::NameLoc>(nested_op->getLoc())) {
          if (name_loc.getName() == instr->name()) {
            op = nested_op;
            return mlir::WalkResult::interrupt();
          }
        }
        return mlir::WalkResult::advance();
      });
      if (!op) {
        ADD_FAILURE() << "Could not find corresponding StableHLO op for "
                      << instr->name();
        return HloInstructionIndexing::FromIndexingMaps({});
      }
      return ComputeOutputToInputIndexing(op, output_id);
    }
    return IndexingTestBase::GetOutputToInputIndexing(instr, output_id,
                                                      use_physical_layout);
  }

  void SetUp() override {
    IndexingTestBase::SetUp();
    mlir_context_.loadDialect<mlir::stablehlo::StablehloDialect,
                              mlir::func::FuncDialect>();

    static const auto* unsupported_tests =
        new absl::flat_hash_set<absl::string_view>{
            // StableHLO indexing analysis does not support physical layout /
            // permutations yet.
            "PhysicalLayoutTestInputPermutation/1",
            "PhysicalLayoutTestOutputPermutation/1",
            "PhysicalLayoutTestInputAndOutputPermutation/1",

            // Custom call / MHLO unknown handling.
            "ScaledDotOp/1",
        };

    if (GetParam()) {
      const testing::TestInfo* test_info =
          testing::UnitTest::GetInstance()->current_test_info();
      absl::string_view test_name = test_info->name();
      // Here we rely on exact match of "TestName/1" which corresponds to
      // GetParam() == true.
      if (unsupported_tests->contains(test_name)) {
        GTEST_SKIP() << "Skipping unsupported StableHLO test: " << test_name;
      }
    }
  }

 private:
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> stablehlo_modules_;
};

INSTANTIATE_TEST_SUITE_P(StablehloIndexingAnalysis, IndexingAnalysisTest,
                         ::testing::Values(false, true));

TEST_P(IndexingAnalysisTest, GroupIndexingMapsByProducers) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1000, 1000] parameter(0)
      transpose_p0 = f32[1000, 1000]{0, 1} transpose(p0), dimensions={1, 0}
      ROOT a0 = f32[1000, 1000] add(p0, transpose_p0)
    }
  )");
  const HloInstruction* parameter = root->operand(0);
  const HloInstruction* transpose = root->operand(1);

  auto root_indexing = GetOutputToInputIndexing(root);
  auto grouped_by_key = GroupIndexingMapsByProducers(root_indexing, root);

  EXPECT_THAT(
      grouped_by_key,
      UnorderedElementsAre(Pair(parameter, ElementsAre(MatchOperandIndexing(R"(
                    (d0, d1) -> (d0, d1),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
                  )"))),
                           Pair(transpose, ElementsAre(MatchOperandIndexing(R"(
                    (d0, d1) -> (d0, d1),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
                  )")))));
}

TEST_P(IndexingAnalysisTest, ComputeGroupedOutputToInputIndexing) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1000, 1000] parameter(0)
      transpose_p0 = f32[1000, 1000]{0, 1} transpose(p0), dimensions={1, 0}
      ROOT a0 = f32[1000, 1000] add(p0, transpose_p0)
    }
  )");
  const HloInstruction* parameter = root->operand(0);
  const HloInstruction* transpose = root->operand(1);

  auto fusion_adaptor = HloFusionAdaptor::ForProducerConsumer(transpose, root);

  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, fusion_adaptor->GetRoots()[0], &mlir_context_);
  EXPECT_THAT(grouped_indexing,
              UnorderedElementsAre(
                  Pair(root, ElementsAre(MatchOperandIndexing(R"(
                    (d0, d1) -> (d0, d1),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
                  )"))),
                  Pair(transpose, ElementsAre(MatchOperandIndexing(R"(
                    (d0, d1) -> (d0, d1),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
                  )"))),
                  Pair(parameter, UnorderedElementsAre(MatchOperandIndexing(R"(
                        (d0, d1) -> (d0, d1),
                        domain:
                        d0 in [0, 999],
                        d1 in [0, 999]
                      )"),
                                                       MatchOperandIndexing(R"(
                        (d0, d1) -> (d1, d0),
                        domain:
                        d0 in [0, 999],
                        d1 in [0, 999]
                      )")))));
}

TEST_P(IndexingAnalysisTest,
       ComputeGroupedOutputToInputIndexing_VariadicReduce) {
  auto root = ParseAndGetRoot(R"(
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
      ROOT reduce = (f32[32]{0}, f32[32]{0})
        reduce(param_0.3, param_1.3, param_2.2, constant),
        dimensions={1}, to_apply=add
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);

  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, fusion_adaptor->GetRoots()[0], &mlir_context_);

  EXPECT_THAT(grouped_indexing,
              UnorderedElementsAre(
                  Pair(root, ElementsAre(MatchOperandIndexing(R"(
                    (d0) -> (d0),
                    domain:
                    d0 in [0, 31]
                  )"))),
                  Pair(root->operand(0), ElementsAre(MatchOperandIndexing(R"(
                    (d0)[s0] -> (d0, s0),
                    domain:
                    d0 in [0, 31],
                    s0 in [0, 39]
                  )"))),
                  Pair(root->operand(1), ElementsAre(MatchOperandIndexing(R"(
                    (d0)[s0] -> (d0, s0),
                    domain:
                    d0 in [0, 31],
                    s0 in [0, 39]
                  )"))),
                  Pair(root->operand(2), ElementsAre(MatchOperandIndexing(R"(
                    (d0) -> (),
                    domain:
                    d0 in [0, 31]
                  )"))),
                  Pair(root->operand(3), ElementsAre(MatchOperandIndexing(R"(
                    (d0) -> (),
                    domain:
                    d0 in [0, 31]
                  )")))));
}

TEST_P(IndexingAnalysisTest, ComputeGroupedOutputToInputIndexing_SingleOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1000, 1000] parameter(0)
      p1 = f32[1000, 1000] parameter(1)
      exp0 = f32[1000, 1000] exponential(p1)
      ROOT a0 = f32[1000, 1000] add(p0, exp0)
    }
  )");
  HloComputation* entry_computation = root->parent();
  const HloInstruction* exponential =
      entry_computation->GetInstructionWithName("exp0");
  const HloInstruction* parameter =
      entry_computation->GetInstructionWithName("p1");

  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(exponential);
  HloInstructionAdaptor parameter_adaptor =
      fusion_adaptor->GetRoots()[0].GetOperand(0);
  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, parameter_adaptor, &mlir_context_);
  EXPECT_THAT(
      grouped_indexing,
      UnorderedElementsAre(Pair(parameter, ElementsAre(MatchOperandIndexing(R"(
                                                     (d0, d1) -> (d0, d1),
                                                     domain:
                                                     d0 in [0, 999],
                                                     d1 in [0, 999]
                                                   )")))));
}

TEST_P(IndexingAnalysisTest,
       ComputeGroupedOutputToInputIndexing_StartNotAtRoot) {
  auto root = ParseAndGetRoot(R"(
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
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  auto root_adaptor = fusion_adaptor->GetRoots()[0];

  auto bcast = root_adaptor.GetOperand(0);
  auto parameter_0 = bcast.GetOperand(0);

  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, bcast, &mlir_context_);
  EXPECT_THAT(
      grouped_indexing,
      UnorderedElementsAre(
          Pair(&bcast.instruction(), ElementsAre(MatchOperandIndexing(R"(
            (d0, d1, d2, d3) -> (d0, d1, d2, d3),
            domain:
            d0 in [0, 14],
            d1 in [0, 31],
            d2 in [0, 19],
            d3 in [0, 63]
          )"))),
          Pair(&parameter_0.instruction(), ElementsAre(MatchOperandIndexing(R"(
            (d0, d1, d2, d3) -> (d0, d2),
            domain:
            d0 in [0, 14],
            d1 in [0, 31],
            d2 in [0, 19],
            d3 in [0, 63]
          )")))));
}

TEST_P(IndexingAnalysisTest, PhysicalLayoutTestOutputPermutation) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20, 30] parameter(0)
      ROOT add0 = f32[10, 20, 30]{1, 0, 2} exponential(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root, /*output_id=*/0,
                                                 /*use_physical_layout=*/true);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d1, d2, d0),
                              domain:
                              d0 in [0, 29],
                              d1 in [0, 9],
                              d2 in [0, 19]
                          )"));

  auto output_indexing = GetInputToOutputIndexing(root, /*input_id=*/0,
                                                  /*use_physical_layout=*/true);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d2, d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19],
                              d2 in [0, 29]
                          )"));
}

TEST_P(IndexingAnalysisTest, CopyNothing) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[0, 0]{0,1} parameter(0)
      ROOT copy0 = f32[0, 0]{1,0} copy(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root, /*output_id=*/0);
  input_indexing.Simplify();
  EXPECT_THAT(input_indexing.ToString(),
              MatchIndexingString("operand id = 0 KNOWN EMPTY"));

  auto output_indexing = GetInputToOutputIndexing(root, /*input_id=*/0);
  output_indexing.Simplify();
  EXPECT_THAT(output_indexing.ToString(),
              MatchIndexingString("operand id = 0 KNOWN EMPTY"));
}

TEST_P(IndexingAnalysisTest, ReshapeNothing) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1,0,0] parameter(0)
      ROOT reshape = f32[0] reshape(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root, /*output_id=*/0);
  input_indexing.Simplify();
  EXPECT_THAT(input_indexing.ToString(),
              MatchIndexingString("operand id = 0 KNOWN EMPTY"));

  auto output_indexing = GetInputToOutputIndexing(root, /*input_id=*/0);
  output_indexing.Simplify();
  EXPECT_THAT(output_indexing.ToString(),
              MatchIndexingString("operand id = 0 KNOWN EMPTY"));
  // Even though the indexing is known empty, the rank of the map should still
  // be 1.
  EXPECT_EQ(output_indexing.indexing_maps[0]
                .begin()
                ->map()
                .GetAffineMap()
                .getNumResults(),
            1);
}

TEST_P(IndexingAnalysisTest, PhysicalLayoutTestInputPermutation) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20, 30]{1, 0, 2} parameter(0)
      ROOT add0 = f32[10, 20, 30] exponential(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root, /*output_id=*/0,
                                                 /*use_physical_layout=*/true);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d2, d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19],
                              d2 in [0, 29]
                          )"));

  auto output_indexing = GetInputToOutputIndexing(root, /*input_id=*/0,
                                                  /*use_physical_layout=*/true);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d1, d2, d0),
                              domain:
                              d0 in [0, 29],
                              d1 in [0, 9],
                              d2 in [0, 19]
                          )"));
}

TEST_P(IndexingAnalysisTest, PhysicalLayoutTestInputAndOutputPermutation) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20, 30]{1, 0, 2} parameter(0)
      ROOT add0 = f32[10, 20, 30]{1, 0, 2} exponential(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root, /*output_id=*/0,
                                                 /*use_physical_layout=*/true);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1, d2),
                              domain:
                              d0 in [0, 29],
                              d1 in [0, 9],
                              d2 in [0, 19]
                          )"));

  auto output_indexing = GetInputToOutputIndexing(root, /*input_id=*/0,
                                                  /*use_physical_layout=*/true);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1, d2),
                              domain:
                              d0 in [0, 29],
                              d1 in [0, 9],
                              d2 in [0, 19]
                          )"));
}

TEST_P(IndexingAnalysisTest, ElementwiseOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20] parameter(0)
      p1 = f32[10, 20] parameter(1)
      ROOT add0 = f32[10, 20] add(p0, p1)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                            operand id = 1
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                          )"));

  auto output_indexing_0 = GetInputToOutputIndexing(root, /*input_id=*/0);
  EXPECT_THAT(output_indexing_0.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                          )"));

  auto output_indexing_1 = GetInputToOutputIndexing(root, /*input_id=*/1);
  EXPECT_THAT(output_indexing_1.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                          )"));
}

TEST_P(IndexingAnalysisTest, Map) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    mapper {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add = f32[] add(a, b)
    }
    ENTRY e {
      p0 = f32[10, 20] parameter(0)
      p1 = f32[10, 20] parameter(1)
      ROOT add0 = f32[10, 20] map(%p0, %p1), dimensions={}, to_apply=mapper
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                            operand id = 1
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                          )"));

  auto output_indexing_0 = GetInputToOutputIndexing(root, /*input_id=*/0);
  EXPECT_THAT(output_indexing_0.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                          )"));

  auto output_indexing_1 = GetInputToOutputIndexing(root, /*input_id=*/1);
  EXPECT_THAT(output_indexing_1.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d0, d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19]
                          )"));
}

TEST_P(IndexingAnalysisTest, BitcastIsReshape) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 32] parameter(0)
      ROOT bitcast = f32[4, 8, 4] bitcast(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1 * 4 + d2),
                              domain:
                              d0 in [0, 3],
                              d1 in [0, 7],
                              d2 in [0, 3]
                          )"));
}

TEST_P(IndexingAnalysisTest, BitcastIsTranspose) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[3, 12288, 6, 128] parameter(0)
      ROOT bitcast = f32[3, 6, 128, 12288] {2, 1, 3, 0} bitcast(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2, d3) -> (d0, d3, d1, d2),
                              domain:
                              d0 in [0, 2],
                              d1 in [0, 5],
                              d2 in [0, 127],
                              d3 in [0, 12287]
                          )"));
}

TEST_P(IndexingAnalysisTest, BitcastIsTransposeReshapeTranspose) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[16, 17, 3] parameter(0)
      ROOT bitcast = f32[51, 16] {0, 1} bitcast(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d1, d0 floordiv 3, d0 mod 3),
                              domain:
                              d0 in [0, 50],
                              d1 in [0, 15]
                          )"));
  auto output_indexing = GetInputToOutputIndexing(root);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d1 * 3 + d2, d0),
                              domain:
                              d0 in [0, 15],
                              d1 in [0, 16],
                              d2 in [0, 2]
                          )"));
}

TEST_P(IndexingAnalysisTest, BroadcastOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[20] parameter(0)
      ROOT bc0 = f32[10, 20, 30] broadcast(p0), dimensions={1}
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d1),
                              domain:
                              d0 in [0, 9],
                              d1 in [0, 19],
                              d2 in [0, 29]
                          )"));
  auto output_indexing = GetInputToOutputIndexing(root);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0)[s0, s1] -> (s0, d0, s1),
                              domain:
                              d0 in [0, 19],
                              s0 in [0, 9],
                              s1 in [0, 29]
                          )"));
}

TEST_P(IndexingAnalysisTest, ConstantOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      ROOT c1 = bf16[17, 22] constant(1)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), IsEmpty());
}

TEST_P(IndexingAnalysisTest, ConcatenateOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[2, 5, 7] parameter(0)
      p1 = f32[2, 11, 7] parameter(1)
      p2 = f32[2, 17, 7] parameter(2)
      ROOT concat = f32[2, 33, 7] concatenate(
        f32[2, 5, 7] p0, f32[2, 11, 7] p1, f32[2, 17, 7] p2), dimensions={1}
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1, d2),
                              domain:
                              d0 in [0, 1],
                              d1 in [0, 4],
                              d2 in [0, 6]
                            operand id = 1
                              (d0, d1, d2) -> (d0, d1 - 5, d2),
                              domain:
                              d0 in [0, 1],
                              d1 in [5, 15],
                              d2 in [0, 6]
                            operand id = 2
                              (d0, d1, d2) -> (d0, d1 - 16, d2),
                              domain:
                              d0 in [0, 1],
                              d1 in [16, 32],
                              d2 in [0, 6]
                          )"));

  auto output_indexing_0 = GetInputToOutputIndexing(root, /*input_id=*/0);
  EXPECT_THAT(output_indexing_0.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1, d2),
                              domain:
                              d0 in [0, 1],
                              d1 in [0, 4],
                              d2 in [0, 6]
                          )"));

  auto output_indexing_1 = GetInputToOutputIndexing(root, /*input_id=*/1);
  EXPECT_THAT(output_indexing_1.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1 + 5, d2),
                              domain:
                              d0 in [0, 1],
                              d1 in [0, 10],
                              d2 in [0, 6]
                          )"));

  auto output_indexing_2 = GetInputToOutputIndexing(root, /*input_id=*/2);
  EXPECT_THAT(output_indexing_2.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2) -> (d0, d1 + 16, d2),
                              domain:
                              d0 in [0, 1],
                              d1 in [0, 16],
                              d2 in [0, 6]
                          )"));
}

TEST_P(IndexingAnalysisTest, DynamicSliceOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      %src = s32[2,2,258] parameter(0)
      %of1 = s32[] parameter(1)
      %of2 = s32[] parameter(2)
      %of3 = s32[] parameter(3)
      ROOT %ds = s32[1,2,32] dynamic-slice(s32[2,2,258] %src,
        s32[] %of1, s32[] %of2, s32[] %of3),
        dynamic_slice_sizes={1, 2, 32}
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
      (d0, d1, d2){rt0, rt1, rt2} -> (d0 + rt0, d1 + rt1, d2 + rt2),
      domain:
        d0 in [0, 0],
        d1 in [0, 1],
        d2 in [0, 31],
        rt0 in [0, 1],
        rt1 in [0, 0],
        rt2 in [0, 226]
      runtime variables:
        rt0: parameter(1); (d0, d1, d2) -> (),
          domain: d0 in [0, 0], d1 in [0, 1], d2 in [0, 31]
        rt1: parameter(2); (d0, d1, d2) -> (),
          domain: d0 in [0, 0], d1 in [0, 1], d2 in [0, 31]
        rt2: parameter(3); (d0, d1, d2) -> (),
          domain: d0 in [0, 0], d1 in [0, 1], d2 in [0, 31]
    operand id = 1
      (d0, d1, d2) -> (),
      domain:
      d0 in [0, 0],
      d1 in [0, 1],
      d2 in [0, 31]
    operand id = 2
      (d0, d1, d2) -> (),
      domain:
      d0 in [0, 0],
      d1 in [0, 1],
      d2 in [0, 31]
    operand id = 3
      (d0, d1, d2) -> (),
      domain:
      d0 in [0, 0],
      d1 in [0, 1],
      d2 in [0, 31]
    )"));
}

TEST_P(IndexingAnalysisTest, DynamicUpdateSliceOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      %src = s32[20,30] parameter(0)
      %upd = s32[5,10] parameter(1)
      %of1 = s32[] parameter(2)
      %of2 = s32[] parameter(3)
      ROOT %dus = s32[20,30] dynamic-update-slice(
          s32[20,30] %src, s32[5,10] %upd, s32[] %of1, s32[] %of2)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
      (d0, d1) -> (d0, d1),
      domain:
        d0 in [0, 19],
        d1 in [0, 29]
    operand id = 1
      (d0, d1){rt0, rt1} -> (d0 - rt0, d1 - rt1),
      domain:
        d0 in [0, 19],
        d1 in [0, 29],
        rt0 in [0, 15],
        rt1 in [0, 20]
      runtime variables:
        rt0: parameter(2); (d0, d1) -> (),
          domain: d0 in [0, 19], d1 in [0, 29]
        rt1: parameter(3); (d0, d1) -> (),
          domain: d0 in [0, 19], d1 in [0, 29]
    operand id = 2
      (d0, d1) -> (),
      domain:
        d0 in [0, 19],
        d1 in [0, 29]
    operand id = 3
      (d0, d1) -> (),
      domain:
        d0 in [0, 19],
        d1 in [0, 29]
      )"));
}

TEST_P(IndexingAnalysisTest, FusionOpWithSingleBinaryOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0) -> (d0),
                              domain:
                              d0 in [0, 99]
                            operand id = 1
                              (d0) -> (d0),
                              domain:
                              d0 in [0, 99]
                          )"));
}

TEST_P(IndexingAnalysisTest, FusionOpWithDot) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  )"));

  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                operand id = 0
                  (d0, d1, d2, d3, d4, d5)[s0] -> (d2, d0 * 768 + s0, d4, d5),
                  domain:
                  d0 in [0, 15],
                  d1 in [0, 15],
                  d2 in [0, 2],
                  d3 in [0, 0],
                  d4 in [0, 5],
                  d5 in [0, 127],
                  s0 in [0, 767]
                operand id = 1
                  (d0, d1, d2, d3, d4, d5)[s0] -> (d0 * 768 + s0),
                  domain:
                  d0 in [0, 15],
                  d1 in [0, 15],
                  d2 in [0, 2],
                  d3 in [0, 0],
                  d4 in [0, 5],
                  d5 in [0, 127],
                  s0 in [0, 767]
                operand id = 2
                  (d0, d1, d2, d3, d4, d5) -> (d1),
                  domain:
                  d0 in [0, 15],
                  d1 in [0, 15],
                  d2 in [0, 2],
                  d3 in [0, 0],
                  d4 in [0, 5],
                  d5 in [0, 127]
                operand id = 3
                  (d0, d1, d2, d3, d4, d5)[s0] -> (d1, d0 * 768 + s0),
                  domain:
                  d0 in [0, 15],
                  d1 in [0, 15],
                  d2 in [0, 2],
                  d3 in [0, 0],
                  d4 in [0, 5],
                  d5 in [0, 127],
                  s0 in [0, 767]
                operand id = 4
                  (d0, d1, d2, d3, d4, d5)[s0] -> (d1, d0 * 768 + s0),
                  domain:
                  d0 in [0, 15],
                  d1 in [0, 15],
                  d2 in [0, 2],
                  d3 in [0, 0],
                  d4 in [0, 5],
                  d5 in [0, 127],
                  s0 in [0, 767]
                operand id = 5
                  (d0, d1, d2, d3, d4, d5) -> (d2, d4, d5),
                  domain:
                  d0 in [0, 15],
                  d1 in [0, 15],
                  d2 in [0, 2],
                  d3 in [0, 0],
                  d4 in [0, 5],
                  d5 in [0, 127]
              )"));
}

TEST_P(IndexingAnalysisTest, FusionOpWithSoftmax) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  )"));
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(UnorderedElementsAre(MatchOperandIndexing(R"(
                            (d0, d1, d2)[s0] -> (d0, d1, s0),
                            domain:
                            d0 in [0, 1],
                            d1 in [0, 64],
                            d2 in [0, 124],
                            s0 in [0, 124]
                          )"),
                                               MatchOperandIndexing(R"(
                            (d0, d1, d2) -> (d0, d1, d2),
                            domain:
                            d0 in [0, 1],
                            d1 in [0, 64],
                            d2 in [0, 124]
                          )"))));
}

TEST_P(IndexingAnalysisTest, FusionOpTensorPlusTransposedTensor) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(UnorderedElementsAre(MatchOperandIndexing(R"(
                            (d0, d1) -> (d0, d1),
                            domain:
                            d0 in [0, 999],
                            d1 in [0, 999]
                          )"),
                                               MatchOperandIndexing(R"(
                            (d0, d1) -> (d1, d0),
                            domain:
                            d0 in [0, 999],
                            d1 in [0, 999]
                          )"))));
}

TEST_P(IndexingAnalysisTest, FusionExponentialDuplication) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
    })"));
  EXPECT_THAT(input_indexing.indexing_maps,
              ElementsAre(UnorderedElementsAre(MatchOperandIndexing(R"(
                            (d0) -> (d0 + 1),
                            domain:
                            d0 in [0, 1]
                          )"),
                                               MatchOperandIndexing(R"(
                            (d0) -> (d0),
                            domain:
                            d0 in [0, 1]
                          )"),
                                               MatchOperandIndexing(R"(
                            (d0) -> (d0 + 2),
                            domain:
                            d0 in [0, 1]
                          )")),
                          UnorderedElementsAre(MatchOperandIndexing(R"(
                            (d0) -> (d0 + 2),
                            domain:
                            d0 in [0, 1]
                          )"),
                                               MatchOperandIndexing(R"(
                            (d0) -> (d0 + 1),
                            domain:
                            d0 in [0, 1]
                          )"),
                                               MatchOperandIndexing(R"(
                            (d0) -> (d0),
                            domain:
                            d0 in [0, 1]
                          )"))));
}

TEST_P(IndexingAnalysisTest, GatherOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY main {
      operand = f32[33,76,70] parameter(0)
      indices = s32[1806,2] parameter(1)
      ROOT r = f32[1806,7,8,4] gather(operand, indices), offset_dims={1,2,3},
                                 collapsed_slice_dims={}, start_index_map={0,1},
                                 index_vector_dim=1, slice_sizes={7,8,4}
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
      (d0, d1, d2, d3){rt0, rt1} -> (d1 + rt0, d2 + rt1, d3),
      domain:
        d0 in [0, 1805],
        d1 in [0, 6],
        d2 in [0, 7],
        d3 in [0, 3],
        rt0 in [0, 26],
        rt1 in [0, 68]
      runtime variables:
        rt0: parameter(1); (d0, d1, d2, d3) -> (d0, 0),
          domain: d0 in [0, 1805], d1 in [0, 6], d2 in [0, 7], d3 in [0, 3]
        rt1: parameter(1); (d0, d1, d2, d3) -> (d0, 1),
          domain: d0 in [0, 1805], d1 in [0, 6], d2 in [0, 7], d3 in [0, 3]
    operand id = 1
      (d0, d1, d2, d3)[s0] -> (d0, s0),
      domain:
        d0 in [0, 1805],
        d1 in [0, 6],
        d2 in [0, 7],
        d3 in [0, 3],
        s0 in [0, 1]
    )"));
}

TEST_P(IndexingAnalysisTest, GatherOpWithShuffledStartIndexMap) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY main {
      operand = f32[33,76,70] parameter(0)
      indices = s32[1806,2] parameter(1)
      ROOT r = f32[1806,7,8,4] gather(operand, indices), offset_dims={1,2,3},
                                 collapsed_slice_dims={}, start_index_map={1,0},
                                 index_vector_dim=1, slice_sizes={7,8,4}
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
      (d0, d1, d2, d3){rt0, rt1} -> (d1 + rt0, d2 + rt1, d3),
      domain:
        d0 in [0, 1805],
        d1 in [0, 6],
        d2 in [0, 7],
        d3 in [0, 3],
        rt0 in [0, 26],
        rt1 in [0, 68]
      runtime variables:
        rt0: parameter(1); (d0, d1, d2, d3) -> (d0, 1),
          domain: d0 in [0, 1805], d1 in [0, 6], d2 in [0, 7], d3 in [0, 3]
        rt1: parameter(1); (d0, d1, d2, d3) -> (d0, 0),
          domain: d0 in [0, 1805], d1 in [0, 6], d2 in [0, 7], d3 in [0, 3]
    operand id = 1
      (d0, d1, d2, d3)[s0] -> (d0, s0),
      domain:
        d0 in [0, 1805],
        d1 in [0, 6],
        d2 in [0, 7],
        d3 in [0, 3],
        s0 in [0, 1]
    )"));
}

TEST_P(IndexingAnalysisTest, FusionOpWithReduceOfReduce) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0)[s0, s1, s2] -> (s0, s2, d0, s1),
                            domain:
                            d0 in [0, 9],
                            s0 in [0, 149],
                            s1 in [0, 49],
                            s2 in [0, 19]
                          operand id = 1
                            (d0) -> (),
                            domain:
                            d0 in [0, 9]
                          )"));
}

TEST_P(IndexingAnalysisTest, FusionOpWithReduceOfBroadcast) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0] -> (d0, s0),
                            domain:
                            d0 in [0, 14],
                            d1 in [0, 63],
                            s0 in [0, 19]
                          operand id = 1
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 14],
                            d1 in [0, 63]
                          )"));
}

TEST_P(IndexingAnalysisTest, FusionOpWithTransposeOfTranspose) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2) -> (d2, d0, d1),
                            domain:
                            d0 in [0, 9],
                            d1 in [0, 49],
                            d2 in [0, 19]
                          )"));
}

TEST_P(IndexingAnalysisTest, FusionOpWithReducedSlice) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0)[s0, s1] -> (s0 + 5, d0 * 2, s1 * 3 + 50),
                            domain:
                            d0 in [0, 31],
                            s0 in [0, 15],
                            s1 in [0, 127]
                          operand id = 1
                            (d0) -> (),
                            domain:
                            d0 in [0, 31]
                          )"));
}

TEST_P(IndexingAnalysisTest, FusionOpWithReshape_CollapseOfExpand) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0) -> (d0),
                            domain:
                            d0 in [0, 127]
                          )"));
}

TEST_P(IndexingAnalysisTest, FusionOpWithReshape_ExpandOfCollapse) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1) -> (d0, d1),
                            domain:
                            d0 in [0, 7],
                            d1 in [0, 15]
                          )"));
}

TEST_P(IndexingAnalysisTest, FusionOpWithReshape_ChainedGenericReshapes) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2) -> (d0, d1, d2),
                            domain:
                            d0 in [0, 9],
                            d1 in [0, 9],
                            d2 in [0, 9]
                          )"));
}

TEST_P(IndexingAnalysisTest, FusionOpWithSliceOfSlice) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
              operand id = 0
                (d0, d1, d2) -> (d0 * 2 + 8, d1 * 6 + 8, d2 * 12 + 65),
                domain:
                d0 in [0, 6],
                d1 in [0, 8],
                d2 in [0, 23]
              )"));
}

TEST_P(IndexingAnalysisTest, FusionOpWithDynSliceOfDynSlice) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    f {
      %src = s32[150, 64] parameter(0)
      %of11 = s32[] parameter(1)
      %of12 = s32[] parameter(2)
      %of21 = s32[] parameter(3)
      %of22 = s32[] parameter(4)

      %ds1 = s32[50, 32] dynamic-slice(s32[150, 64] %src,
        s32[] %of11, s32[] %of12), dynamic_slice_sizes={50, 32}

      ROOT %ds2 = s32[25, 16] dynamic-slice(s32[50, 32] %ds1,
        s32[] %of21, s32[] %of22), dynamic_slice_sizes={25, 16}
    }
    ENTRY e {
      %p0 = s32[150, 64] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = s32[] parameter(2)
      %p3 = s32[] parameter(3)
      %p4 = s32[] parameter(4)
      ROOT fusion = s32[25, 16] fusion(p0, p1, p2, p3, p4),
        kind=kLoop, calls=f
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
      (d0, d1){rt0, rt1, rt2, rt3} -> (d0 + rt0 + rt2, d1 + rt1 + rt3),
      domain:
        d0 in [0, 24],
        d1 in [0, 15],
        rt0 in [0, 100],
        rt1 in [0, 32],
        rt2 in [0, 25],
        rt3 in [0, 16]
      runtime variables:
        rt0: parameter(3); (d0, d1) -> (),
          domain: d0 in [0, 24], d1 in [0, 15]
        rt1: parameter(4); (d0, d1) -> (),
          domain: d0 in [0, 24], d1 in [0, 15]
        rt2: parameter(1); (d0, d1){rt0, rt1} -> (),
          domain:
            d0 in [0, 24],
            d1 in [0, 15],
            rt0 in [0, 25],
            rt1 in [0, 16],
            d0 + rt0 in [0, 49],
            d1 + rt1 in [0, 31]
        rt3: parameter(2); (d0, d1){rt0, rt1} -> (),
          domain:
            d0 in [0, 24], d1 in [0, 15],
            rt0 in [0, 25], rt1 in [0, 16],
            d0 + rt0 in [0, 49],
            d1 + rt1 in [0, 31]
    operand id = 1
      (d0, d1) -> (),
      domain:
        d0 in [0, 24],
        d1 in [0, 15]
    operand id = 2
      (d0, d1) -> (),
      domain:
        d0 in [0, 24],
        d1 in [0, 15]
    operand id = 3
      (d0, d1) -> (),
      domain:
        d0 in [0, 24],
        d1 in [0, 15]
    operand id = 4
      (d0, d1) -> (),
      domain:
        d0 in [0, 24],
        d1 in [0, 15]
    )"));
}

TEST_P(IndexingAnalysisTest, FusionOpSliceOfAllConcatenateOpInputs) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2) -> (d0, d1 * 3, d2),
                            domain:
                            d0 in [0, 1],
                            d1 in [0, 1],
                            d2 in [0, 6]
                          operand id = 1
                            (d0, d1, d2) -> (d0, d1 * 3 - 5, d2),
                            domain:
                            d0 in [0, 1],
                            d1 in [2, 5],
                            d2 in [0, 6]
                          operand id = 2
                            (d0, d1, d2) -> (d0, d1 * 3 - 16, d2),
                            domain:
                            d0 in [0, 1],
                            d1 in [6, 10],
                            d2 in [0, 6]
                          )"));
}

TEST_P(IndexingAnalysisTest, FusionOpSliceOfOneOfConcatenateOpInputs) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2) -> (d0, d1 * 2, d2),
                            domain:
                            d0 in [0, 1],
                            d1 in [0, 2],
                            d2 in [0, 6]
                          operand id = 1
                            KNOWN EMPTY
                          operand id = 2
                            KNOWN EMPTY
                          )"));
}

TEST_P(IndexingAnalysisTest, FusionOpReshapeOfConcat) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1) -> (d0 * 8 + d1),
                            domain:
                            d0 in [0, 3],
                            d1 in [0, 7],
                            d0 * 8 + d1 in [0, 1]
                          operand id = 1
                            (d0, d1) -> (d0 * 8 + d1 - 2),
                            domain:
                            d0 in [0, 3],
                            d1 in [0, 7],
                            d0 * 8 + d1 in [2, 31]
                          )"));
}

TEST_P(IndexingAnalysisTest, IotaOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      ROOT iota = s32[5,5,111,42] iota(), iota_dimension=0
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.indexing_maps, IsEmpty());
}

TEST_P(IndexingAnalysisTest, ReshapeOpCollapseShape) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4,8] parameter(0)
      ROOT reshape = f32[32] reshape(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0) -> (d0 floordiv 8, d0 mod 8),
                            domain:
                            d0 in [0, 31]
                          )"));
}

TEST_P(IndexingAnalysisTest, ReshapeOpExpandShape) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[32] parameter(0)
      ROOT reshape = f32[4, 8] reshape(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1) -> (d0 * 8 + d1),
                            domain:
                            d0 in [0, 3],
                            d1 in [0, 7]
                          )"));
}

TEST_P(IndexingAnalysisTest, ReshapeOpExpandAndCollapseShape) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 8, 12] parameter(0)
      ROOT reshape = f32[32, 3, 4] reshape(p0)
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
              operand id = 0
                (d0, d1, d2) -> (d0 floordiv 8, d0 mod 8, d1 * 4 + d2),
                domain:
                d0 in [0, 31],
                d1 in [0, 2],
                d2 in [0, 3]
              )"));

  auto output_indexing = GetInputToOutputIndexing(root);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
              operand id = 0
                (d0, d1, d2) -> (d0 * 8 + d1, d2 floordiv 4, d2 mod 4),
                domain:
                d0 in [0, 3],
                d1 in [0, 7],
                d2 in [0, 11]
              )"));
}

TEST_P(IndexingAnalysisTest, ReshapeOpExpandSubshapeOnly) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[16, 8] parameter(0)
      ROOT reshape = f32[4, 4, 8] reshape(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
              operand id = 0
                (d0, d1, d2) -> (d0 * 4 + d1, d2),
                domain:
                d0 in [0, 3],
                d1 in [0, 3],
                d2 in [0, 7]
              )"));
}

TEST_P(IndexingAnalysisTest, ReshapeOpGenericReshape2DTo3D) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4,8] parameter(0)
      ROOT reshape = f32[2, 4, 4] reshape(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
              operand id = 0
                (d0, d1, d2) -> (d0 * 2 + d1 floordiv 2, (d1 mod 2) * 4 + d2),
                domain:
                d0 in [0, 1],
                d1 in [0, 3],
                d2 in [0, 3]
              )"));
}

TEST_P(IndexingAnalysisTest, ReshapeOpGenericReshape3DTo2D) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[2, 4, 4] parameter(0)
      ROOT reshape = f32[4, 8] reshape(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1) -> (d0 floordiv 2,
                                        (d0 mod 2) * 2 + d1 floordiv 4,
                                        d1 mod 4),
                            domain:
                            d0 in [0, 3],
                            d1 in [0, 7]
                          )"));
}

TEST_P(IndexingAnalysisTest, PadOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 4] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[12, 16] pad(p0, p1), padding=1_4_1x4_8_0
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                                  operand id = 0
                                    (d0, d1) -> ((d0 - 1) floordiv 2, d1 - 4),
                                    domain:
                                    d0 in [1, 7],
                                    d1 in [4, 7],
                                    (d0 - 1) mod 2 in [0, 0]
                                  operand id = 1
                                    (d0, d1) -> (),
                                    domain:
                                    d0 in [0, 11],
                                    d1 in [0, 15]
                                )"));
}

TEST_P(IndexingAnalysisTest, PadOpNoInterior) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[2,8] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[10,8] pad(p0, p1), padding=1_7x0_0
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                                  operand id = 0
                                    (d0, d1) -> (d0 - 1, d1),
                                    domain:
                                    d0 in [1, 2],
                                    d1 in [0, 7]
                                  operand id = 1
                                    (d0, d1) -> (),
                                    domain:
                                    d0 in [0, 9],
                                    d1 in [0, 7]
                                )"));
}

TEST_P(IndexingAnalysisTest, PadOpNegativePadding) {
  // The interior padding is applied first (even with negative padding), so we
  // get a size of 5 (7 + 6 - 8).
  // in:     0 1 2 3 4 5 6
  // padded: 0 p 1 p 2 p 3 p 4 p 5 p 6
  // sliced:       p 2 p 3 p
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[7] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[5] pad(p0, p1), padding=-3_-5_1
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                                  operand id = 0
                                    (d0) -> ((d0 + 3) floordiv 2),
                                    domain:
                                    d0 in [0, 4],
                                    (d0 + 3) mod 2 in [0, 0]
                                  operand id = 1
                                    (d0) -> (),
                                    domain:
                                    d0 in [0, 4]
                                )"));
}

TEST_P(IndexingAnalysisTest, ReduceOp) {
  auto root = ParseAndGetRoot(R"(
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
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0, s1] -> (d0, s0, d1, s1),
                            domain:
                            d0 in [0, 149],
                            d1 in [0, 9],
                            s0 in [0, 19],
                            s1 in [0, 49]
                          operand id = 1
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 149],
                            d1 in [0, 9]
                          )"));

  auto output_indexing_0 = GetInputToOutputIndexing(root, 0);
  EXPECT_THAT(output_indexing_0.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3) -> (d0, d2),
                            domain:
                            d0 in [0, 149],
                            d1 in [0, 19],
                            d2 in [0, 9],
                            d3 in [0, 49]
                          )"));
  auto output_indexing_1 = GetInputToOutputIndexing(root, 1);
  EXPECT_THAT(output_indexing_1.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            ()[s0, s1] -> (s0, s1),
                            domain:
                            s0 in [0, 149],
                            s1 in [0, 9]
                          )"));
}

TEST_P(IndexingAnalysisTest, VariadicReduceOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
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
  )");

  auto output_indexing_0 = GetOutputToInputIndexing(root, /*output_id=*/0);
  EXPECT_THAT(output_indexing_0.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0)[s0] -> (s0, d0),
                            domain:
                            d0 in [0, 9],
                            s0 in [0, 255]
                          operand id = 1
                            (d0)[s0] -> (s0, d0),
                            domain:
                            d0 in [0, 9],
                            s0 in [0, 255]
                          operand id = 2
                            (d0) -> (),
                            domain:
                            d0 in [0, 9]
                          operand id = 3
                            (d0) -> (),
                            domain:
                            d0 in [0, 9]
                          )"));

  auto output_indexing_1 = GetOutputToInputIndexing(root, /*output_id=*/1);
  EXPECT_THAT(output_indexing_1.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0)[s0] -> (s0, d0),
                            domain:
                            d0 in [0, 9],
                            s0 in [0, 255]
                          operand id = 1
                            (d0)[s0] -> (s0, d0),
                            domain:
                            d0 in [0, 9],
                            s0 in [0, 255]
                          operand id = 2
                            (d0) -> (),
                            domain:
                            d0 in [0, 9]
                          operand id = 3
                            (d0) -> (),
                            domain:
                            d0 in [0, 9]
                          )"));

  constexpr absl::string_view kInputToOutputIndexing = R"(
      (d0, d1) -> (d1),
      domain:
      d0 in [0, 255],
      d1 in [0, 9]
  )";
  auto input_indexing_0 = GetInputToOutputIndexing(root, /*input_id=*/0);
  EXPECT_THAT(
      input_indexing_0.indexing_maps,
      ElementsAre(ElementsAre(MatchOperandIndexing(kInputToOutputIndexing)),
                  ElementsAre(MatchOperandIndexing(kInputToOutputIndexing))));

  auto input_indexing_1 = GetInputToOutputIndexing(root, /*input_id=*/1);
  EXPECT_THAT(
      input_indexing_1.indexing_maps,
      ElementsAre(ElementsAre(MatchOperandIndexing(kInputToOutputIndexing)),
                  ElementsAre(MatchOperandIndexing(kInputToOutputIndexing))));

  constexpr absl::string_view kInitToOutputIndexing = R"(
      ()[s0] -> (s0),
      domain:
      s0 in [0, 9]
  )";
  auto input_indexing_2 = GetInputToOutputIndexing(root, /*input_id=*/2);
  EXPECT_THAT(
      input_indexing_2.indexing_maps,
      ElementsAre(ElementsAre(MatchOperandIndexing(kInitToOutputIndexing)),
                  ElementsAre(MatchOperandIndexing(kInitToOutputIndexing))));
  auto input_indexing_3 = GetInputToOutputIndexing(root, /*input_id=*/2);
  EXPECT_THAT(
      input_indexing_3.indexing_maps,
      ElementsAre(ElementsAre(MatchOperandIndexing(kInitToOutputIndexing)),
                  ElementsAre(MatchOperandIndexing(kInitToOutputIndexing))));
}

TEST_P(IndexingAnalysisTest, ReduceWindowOp_NoPadding) {
  auto root = ParseAndGetRoot(R"(
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
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0] -> (d0, d1 + s0),
                            domain:
                            d0 in [0, 1023],
                            d1 in [0, 2],
                            s0 in [0, 511]
                          operand id = 1
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 1023],
                            d1 in [0, 2]
                          )"));
}

TEST_P(IndexingAnalysisTest, ReduceWindowOp_4DWithTrivalDims_NoPadding) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }
    ENTRY e {
      c_inf = f32[] constant(-inf)
      p0 = f32[1024, 514, 1, 1]parameter(0)
      ROOT reduce-window = f32[1024, 3, 1, 1] reduce-window(p0, c_inf),
        window={size=1x512x1x1 pad=0_0x0_0x0_0x0_0}, to_apply=max
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3)[s0] -> (d0, d1 + s0, 0, 0),
                            domain:
                            d0 in [0, 1023],
                            d1 in [0, 2],
                            d2 in [0, 0],
                            d3 in [0, 0],
                            s0 in [0, 511]
                          operand id = 1
                            (d0, d1, d2, d3) -> (),
                            domain:
                            d0 in [0, 1023],
                            d1 in [0, 2],
                            d2 in [0, 0],
                            d3 in [0, 0]
                          )"));
}

TEST_P(IndexingAnalysisTest, ReduceWindowOp_PaddingAndWindowStride) {
  auto root = ParseAndGetRoot(R"(
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
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0, s1] -> (d0 * 2 + s0 - 1, d1 + s1),
                            domain:
                            d0 in [0, 6],
                            d1 in [0, 16],
                            s0 in [0, 2],
                            s1 in [0, 1],
                            d0 * 2 + s0 in [1, 13],
                            d1 + s1 in [0, 16]
                          operand id = 1
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 6],
                            d1 in [0, 16]
                          )"));
}

TEST_P(IndexingAnalysisTest, ReduceWindowOp_BaseDilation) {
  auto root = ParseAndGetRoot(R"(
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
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1) -> (d0 floordiv 2, d1 floordiv 2),
                            domain:
                            d0 in [0, 2],
                            d1 in [0, 4],
                            d0 mod 2 in [0, 0],
                            d1 mod 2 in [0, 0]
                          operand id = 1
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 2],
                            d1 in [0, 4]
                          )"));
}

TEST_P(IndexingAnalysisTest, ReduceWindowOp_WindowDilation) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }
    ENTRY e {
      c_inf = f32[] constant(-inf)
      p0 = f32[7, 3] parameter(0)
      ROOT reduce-window = f32[4, 3] reduce-window(p0, c_inf),
       window={size=2x1 pad=0_0x0_0 rhs_dilate=3x1}, to_apply=max
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0] -> (d0 + s0 * 3, d1),
                            domain:
                            d0 in [0, 3],
                            d1 in [0, 2],
                            s0 in [0, 1]
                          operand id = 1
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 3],
                            d1 in [0, 2]
                          )"));
}

TEST_P(IndexingAnalysisTest, ReduceWindowOp_Variadic) {
  auto root = ParseAndGetRoot(R"(
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
  )");
  auto input_indexing_0 = GetOutputToInputIndexing(root, /*output_id=*/0);
  EXPECT_THAT(input_indexing_0.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0, s1] -> (s0, d1 + s1),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1],
                            s0 in [0, 1],
                            s1 in [0, 1]
                          operand id = 1
                            (d0, d1)[s0, s1] -> (s0, d1 + s1),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1],
                            s0 in [0, 1],
                            s1 in [0, 1]
                          operand id = 2
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1]
                          operand id = 3
                           (d0, d1) -> (),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1]
                          )"));
  auto input_indexing_1 = GetOutputToInputIndexing(root, /*output_id=*/1);
  EXPECT_THAT(input_indexing_1.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1)[s0, s1] -> (s0, d1 + s1),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1],
                            s0 in [0, 1],
                            s1 in [0, 1]
                          operand id = 1
                            (d0, d1)[s0, s1] -> (s0, d1 + s1),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1],
                            s0 in [0, 1],
                            s1 in [0, 1]
                          operand id = 2
                            (d0, d1) -> (),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1]
                          operand id = 3
                           (d0, d1) -> (),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 1]
                          )"));
}

TEST_P(IndexingAnalysisTest, ConvolutionOp_NoPadding) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1,12,10,4] parameter(0)
      p1 = f32[4,3,5,8] parameter(1)
      ROOT conv = f32[1,10,6,8] convolution(p0, p1),
        window={size=3x5 pad=0_0x0_0}, dim_labels=b01f_i01o->b01f
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3)[s0, s1, s2] -> (0, d1 + s0, d2 + s1, s2),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 9],
                            d2 in [0, 5],
                            d3 in [0, 7],
                            s0 in [0, 2],
                            s1 in [0, 4],
                            s2 in [0, 3]
                          operand id = 1
                            (d0, d1, d2, d3)[s0, s1, s2] -> (s2, s0, s1, d3),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 9],
                            d2 in [0, 5],
                            d3 in [0, 7],
                            s0 in [0, 2],
                            s1 in [0, 4],
                            s2 in [0, 3]
                          )"));
}

TEST_P(IndexingAnalysisTest, ConvolutionOp_4DWithTrivialDims_NoPadding) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f64[1,1,16,16,2,2]{5,4,3,2,1,0} parameter(0)
      p1 = f64[1,1,3,3,1,1]{5,4,3,2,1,0} parameter(1)
      ROOT conv = f64[1,1,14,14,2,2]{5,4,3,2,1,0} convolution(p0, p1),
        window={size=3x3x1x1}, dim_labels=bf0123_oi0123->bf0123
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3, d4, d5)[s0, s1] -> (0, 0, d2 + s0, d3 + s1, d4, d5),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 0],
                            d2 in [0, 13],
                            d3 in [0, 13],
                            d4 in [0, 1],
                            d5 in [0, 1],
                            s0 in [0, 2],
                            s1 in [0, 2]
                          operand id = 1
                            (d0, d1, d2, d3, d4, d5)[s0, s1] -> (0, 0, s0, s1, 0, 0),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 0],
                            d2 in [0, 13],
                            d3 in [0, 13],
                            d4 in [0, 1],
                            d5 in [0, 1],
                            s0 in [0, 2],
                            s1 in [0, 2]
                          )"));
}

TEST_P(IndexingAnalysisTest, ConvolutionOp_PaddingAndWindowStride) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1,12,10,4] parameter(0)
      p1 = f32[4,3,5,8] parameter(1)
      ROOT conv = f32[1,6,5,8] convolution(p0, p1),
        window={size=3x5 stride=2x2 pad=1_1x2_2}, dim_labels=b01f_i01o->b01f
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3)[s0, s1, s2] -> (0, d1 * 2 + s0 - 1, d2 * 2 + s1 - 2, s2),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 5],
                            d2 in [0, 4],
                            d3 in [0, 7],
                            s0 in [0, 2],
                            s1 in [0, 4],
                            s2 in [0, 3],
                            d1 * 2 + s0 in [1, 12],
                            d2 * 2 + s1 in [2, 11]
                          operand id = 1
                            (d0, d1, d2, d3)[s0, s1, s2] -> (s2, s0, s1, d3),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 5],
                            d2 in [0, 4],
                            d3 in [0, 7],
                            s0 in [0, 2],
                            s1 in [0, 4],
                            s2 in [0, 3]
                          )"));
}

TEST_P(IndexingAnalysisTest, ConvolutionOp_LhsDilation) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1,12,10,4] parameter(0)
      p1 = f32[4,3,5,8] parameter(1)
      ROOT conv = f32[1,21,15,8] convolution(p0, p1),
        window={size=3x5 pad=0_0x0_0 lhs_dilate=2x2}, dim_labels=b01f_i01o->b01f
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3)[s0, s1, s2] -> (0, (d1 + s0) floordiv 2, (d2 + s1) floordiv 2, s2),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 20],
                            d2 in [0, 14],
                            d3 in [0, 7],
                            s0 in [0, 2],
                            s1 in [0, 4],
                            s2 in [0, 3],
                            (d1 + s0) mod 2 in [0, 0],
                            (d2 + s1) mod 2 in [0, 0]
                          operand id = 1
                            (d0, d1, d2, d3)[s0, s1, s2] -> (s2, s0, s1, d3),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 20],
                            d2 in [0, 14],
                            d3 in [0, 7],
                            s0 in [0, 2],
                            s1 in [0, 4],
                            s2 in [0, 3]
                          )"));
}

TEST_P(IndexingAnalysisTest, ConvolutionOp_RhsDilation) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1,12,10,4] parameter(0)
      p1 = f32[4,3,5,8] parameter(1)
      ROOT conv = f32[1,8,2,8] convolution(p0, p1),
        window={size=3x5 pad=0_0x0_0 rhs_dilate=2x2}, dim_labels=b01f_i01o->b01f
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3)[s0, s1, s2] -> (0, d1 + s0 * 2, d2 + s1 * 2, s2),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 7],
                            d2 in [0, 1],
                            d3 in [0, 7],
                            s0 in [0, 2],
                            s1 in [0, 4],
                            s2 in [0, 3]
                          operand id = 1
                            (d0, d1, d2, d3)[s0, s1, s2] -> (s2, s0, s1, d3),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 7],
                            d2 in [0, 1],
                            d3 in [0, 7],
                            s0 in [0, 2],
                            s1 in [0, 4],
                            s2 in [0, 3]
                          )"));
}

TEST_P(IndexingAnalysisTest, ConvolutionOp_FeatureGroups) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1,12,10,24] parameter(0)
      p1 = f32[4,3,5,48] parameter(1)
      ROOT conv = f32[1,10,6,48] convolution(p0, p1),
        window={size=3x5 pad=0_0x0_0}, dim_labels=b01f_i01o->b01f, feature_group_count=6
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3)[s0, s1, s2] -> (0, d1 + s0, d2 + s1, (d3 floordiv 8) * 4 + s2),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 9],
                            d2 in [0, 5],
                            d3 in [0, 47],
                            s0 in [0, 2],
                            s1 in [0, 4],
                            s2 in [0, 3]
                          operand id = 1
                            (d0, d1, d2, d3)[s0, s1, s2] -> (s2, s0, s1, d3),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 9],
                            d2 in [0, 5],
                            d3 in [0, 47],
                            s0 in [0, 2],
                            s1 in [0, 4],
                            s2 in [0, 3]
                          )"));
}

TEST_P(IndexingAnalysisTest, ConvolutionOp_BatchGroups) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[14,12,10,4] parameter(0)
      p1 = f32[4,3,5,21] parameter(1)
      ROOT conv = f32[2,10,6,21] convolution(p0, p1),
        window={size=3x5 pad=0_0x0_0}, dim_labels=b01f_i01o->b01f, batch_group_count=7
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 + s3 * 2, d1 + s0, d2 + s1, s2),
                            domain:
                            d0 in [0, 1],
                            d1 in [0, 9],
                            d2 in [0, 5],
                            d3 in [0, 20],
                            s0 in [0, 2],
                            s1 in [0, 4],
                            s2 in [0, 3],
                            s3 in [0, 6]
                          operand id = 1
                            (d0, d1, d2, d3)[s0, s1, s2] -> (s2, s0, s1, d3),
                            domain:
                            d0 in [0, 1],
                            d1 in [0, 9],
                            d2 in [0, 5],
                            d3 in [0, 20],
                            s0 in [0, 2],
                            s1 in [0, 4],
                            s2 in [0, 3]
                          )"));
}

TEST_P(IndexingAnalysisTest, ReverseOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[1, 17, 9, 9] parameter(0)
      ROOT reverse = f32[1, 17, 9, 9] reverse(p0), dimensions={1, 2}
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3) -> (d0, -d1 + 16, -d2 + 8, d3),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 16],
                            d2 in [0, 8],
                            d3 in [0, 8]
                          )"));

  auto output_indexing = GetInputToOutputIndexing(root);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2, d3) -> (d0, -d1 + 16, -d2 + 8, d3),
                            domain:
                            d0 in [0, 0],
                            d1 in [0, 16],
                            d2 in [0, 8],
                            d3 in [0, 8]
                          )"));
}

TEST_P(IndexingAnalysisTest, ReverseReshape) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
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
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1) -> (d0, d1),
                            domain:
                            d0 in [0, 9],
                            d1 in [0, 10]
                          )"));
}

TEST_P(IndexingAnalysisTest, SliceOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20, 50] parameter(0)
      ROOT slice = f32[5, 3, 25] slice(f32[10, 20, 50] p0),
          slice={[5:10:1], [3:20:7], [0:50:2]}
    }
  )");
  auto input_indexing = GetOutputToInputIndexing(root);
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2) -> (d0 + 5, d1 * 7 + 3, d2 * 2),
                            domain:
                            d0 in [0, 4],
                            d1 in [0, 2],
                            d2 in [0, 24]
                          )"));
  auto output_indexing = GetInputToOutputIndexing(root);
  EXPECT_THAT(output_indexing.ToString(), MatchIndexingString(R"(
                          operand id = 0
                            (d0, d1, d2) -> (
                              d0 - 5,
                              (d1 - 3) floordiv 7,
                              d2 floordiv 2
                            ),
                            domain:
                            d0 in [5, 9],
                            d1 in [3, 17],
                            d2 in [0, 48],
                            (d1 - 3) mod 7 in [0, 0],
                            d2 mod 2 in [0, 0]
                          )"));
}

TEST_P(IndexingAnalysisTest, TransposeOp) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[3, 12288, 6, 128] parameter(0)
      ROOT transpose = f32[3, 6, 128, 12288]
        transpose(p0), dimensions={0, 2, 3, 1}
    }
  )");
  EXPECT_THAT(GetOutputToInputIndexing(root).ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2, d3) -> (d0, d3, d1, d2),
                              domain:
                              d0 in [0, 2],
                              d1 in [0, 5],
                              d2 in [0, 127],
                              d3 in [0, 12287]
                          )"));
  EXPECT_THAT(GetInputToOutputIndexing(root).ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2, d3) -> (d0, d2, d3, d1),
                              domain:
                              d0 in [0, 2],
                              d1 in [0, 12287],
                              d2 in [0, 5],
                              d3 in [0, 127]
                          )"));
}

TEST_P(IndexingAnalysisTest, TransposeOp4D) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[3, 12288, 6, 128] parameter(0)
      ROOT bitcast = f32[3, 6, 128, 12288] {2, 1, 3, 0} bitcast(p0)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1, d2, d3) -> (d0, d3, d1, d2),
                              domain:
                              d0 in [0, 2],
                              d1 in [0, 5],
                              d2 in [0, 127],
                              d3 in [0, 12287]
                          )"));
}

TEST_P(IndexingAnalysisTest, DotOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 38, 17, 11, 18, 10] parameter(0)
      p1 = f32[17, 10, 16, 18, 22, 38] parameter(1)
      ROOT dot = f32[10, 38, 4, 11, 16, 22] dot(p0, p1),
        lhs_batch_dims={5,1}, rhs_batch_dims={1,5},
        lhs_contracting_dims={4,2}, rhs_contracting_dims={3,0}
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                operand id = 0
                  (d0, d1, d2, d3, d4, d5)[s0, s1] -> (d2, d1, s1, d3, s0, d0),
                  domain:
                  d0 in [0, 9],
                  d1 in [0, 37],
                  d2 in [0, 3],
                  d3 in [0, 10],
                  d4 in [0, 15],
                  d5 in [0, 21],
                  s0 in [0, 17],
                  s1 in [0, 16]
                operand id = 1
                  (d0, d1, d2, d3, d4, d5)[s0, s1] -> (s1, d0, d4, s0, d5, d1),
                  domain:
                  d0 in [0, 9],
                  d1 in [0, 37],
                  d2 in [0, 3],
                  d3 in [0, 10],
                  d4 in [0, 15],
                  d5 in [0, 21],
                  s0 in [0, 17],
                  s1 in [0, 16]
              )"));
}

TEST_P(IndexingAnalysisTest, ScaledDotOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      a = f32[2,10] parameter(0)
      b = f32[10,2] parameter(1)
      a_scale = f32[2,2] parameter(2)
      b_scale = f32[2,2] parameter(3)
      ROOT dot = f32[2,2] scaled-dot(a, b, a_scale, b_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
      (d0, d1)[s0] -> (d0, s0),
      domain:
      d0 in [0, 1],
      d1 in [0, 1],
      s0 in [0, 9]
    operand id = 1
      (d0, d1)[s0] -> (s0, d1),
      domain:
      d0 in [0, 1],
      d1 in [0, 1],
      s0 in [0, 9]
    operand id = 2
      (d0, d1)[s0] -> (d0, s0 floordiv 5),
      domain:
      d0 in [0, 1],
      d1 in [0, 1],
      s0 in [0, 9]
    operand id = 3
      (d0, d1)[s0] -> (s0 floordiv 5, d1),
      domain:
      d0 in [0, 1],
      d1 in [0, 1],
      s0 in [0, 9]
  )"));
}

TEST_P(IndexingAnalysisTest, UnsupportedOps) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[20, 20] parameter(0)
      p1 = f32[4,4] parameter(1)
      p2 = f32[4,3] parameter(2)
      ROOT out = f32[4,3] triangular-solve(f32[4,4] p1, f32[4,3] p2),
        left_side=true,
        lower=true,
        transpose_a=NO_TRANSPOSE,
        unit_diagonal=true
    }
  )");
  EXPECT_THAT(GetOutputToInputIndexing(root).ToString(), MatchIndexingString(R"(
                            operand id = 0 unknown indexing
                            operand id = 1 unknown indexing
                          )"));
  EXPECT_THAT(GetInputToOutputIndexing(root, 0).ToString(),
              MatchIndexingString(R"(
                            operand id = 0 unknown indexing
                          )"));
  EXPECT_THAT(GetInputToOutputIndexing(root, 1).ToString(),
              MatchIndexingString(R"(
                            operand id = 0 unknown indexing
                          )"));
}

TEST_P(IndexingAnalysisTest, FusionWithUnsupportedOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    fused_computation {
      p0 = f32[20, 20] parameter(0)
      p1 = f32[4,4] parameter(1)
      p2 = f32[4,3] parameter(2)
      lhs =  f32[4,3] triangular-solve(f32[4,4] p1, f32[4,3] p2),
        left_side=true,
        lower=true,
        transpose_a=NO_TRANSPOSE,
        unit_diagonal=true
      rhs = f32[4, 3] slice(f32[20, 20] p0),
          slice={[0:20:6], [0:5:2]}
      ROOT add = f32[4, 3] add(lhs, rhs)
    }
    ENTRY e {
      p0 = f32[20, 20] parameter(0)
      p1 = f32[4, 4] parameter(1)
      p2 = f32[4, 3] parameter(2)
      ROOT fusion = f32[4, 3] fusion(p0, p1, p2), kind=kLoop,
          calls=fused_computation
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                            operand id = 0
                              (d0, d1) -> (d0 * 6, d1 * 2),
                              domain:
                              d0 in [0, 3],
                              d1 in [0, 2]
                            operand id = 1
                              unknown indexing
                            operand id = 2
                              unknown indexing
                          )"));
}

TEST_P(IndexingAnalysisTest, EpilogueIndexing) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m
    fused_computation {
      p0 = f32[1000, 1000] parameter(0)
      t = f32[1000, 1000]{0, 1} transpose(p0), dimensions={1, 0}
      a0 = f32[1000000] bitcast(t)
      ROOT log = f32[1000000] log(a0)
    }
    ENTRY e {
      p0 = f32[1000, 1000] parameter(0)
      ROOT fusion = f32[1000000] fusion(p0), kind=kLoop,
          calls=fused_computation
    }
  )");
  ASSERT_TRUE(module.ok());
  auto* computation = (*module)->GetComputationWithName("fused_computation");
  auto fusion = HloFusionAdaptor::ForComputation(computation);
  HloInstructionAdaptor transpose(*computation->GetInstructionWithName("t"),
                                  fusion.get());
  HloInstructionAdaptor log(*computation->GetInstructionWithName("log"),
                            fusion.get());

  EXPECT_THAT(ToString(ComputeEpilogueInputToOutputIndexing(transpose, log,
                                                            &mlir_context_)),
              MatchIndexingString(R"(
                  (d0, d1) -> (d1 * 1000 + d0),
                  domain:
                  d0 in [0, 999],
                  d1 in [0, 999]
              )"));
}

TEST_P(IndexingAnalysisTest, EpilogueIndexing_NoEpilogue) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m
    fused_computation {
      p0 = f32[1000, 1000] parameter(0)
      ROOT t = f32[1000, 1000]{0, 1} transpose(p0), dimensions={1, 0}
    }
    ENTRY e {
      p0 = f32[1000, 1000] parameter(0)
      ROOT fusion = f32[1000, 1000] fusion(p0), kind=kLoop,
          calls=fused_computation
    }
  )");
  ASSERT_TRUE(module.ok());
  auto* computation = (*module)->GetComputationWithName("fused_computation");
  auto fusion = HloFusionAdaptor::ForComputation(computation);
  HloInstructionAdaptor transpose(*computation->GetInstructionWithName("t"),
                                  fusion.get());

  EXPECT_THAT(ToString(ComputeEpilogueInputToOutputIndexing(
                  transpose, transpose, &mlir_context_)),
              MatchIndexingString(R"(
                  (d0, d1) -> (d0, d1),
                  domain:
                  d0 in [0, 999],
                  d1 in [0, 999]
              )"));
}

TEST_P(IndexingAnalysisTest, BroadcastingElementwise) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = pred[] parameter(0)
      p1 = f32[1000, 1000] parameter(1)
      p2 = f32[1000, 1000] parameter(2)
      ROOT select = f32[1000, 1000] select(p0, p1, p2)
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
                  operand id = 0 (d0, d1) -> (),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
                  operand id = 1 (d0, d1) -> (d0, d1),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
                  operand id = 2 (d0, d1) -> (d0, d1),
                    domain:
                    d0 in [0, 999],
                    d1 in [0, 999]
              )"));
}

TEST_P(IndexingAnalysisTest, FusionWithRTVarsSimplification_ScalarConstant) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        p0 = s32[4096] parameter(0)
        offset = s64[] constant(42)
        ROOT dynamic-slice = s32[10]
          dynamic-slice(p0, offset), dynamic_slice_sizes={10}
      }
      ENTRY main {
        p0 = s32[4096] parameter(0)
        ROOT fusion = s32[10] fusion(p0), kind=kInput, calls=fused_computation
      }
    )hlo"));

  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
      (d0) -> (d0 + 42),
      domain:
      d0 in [0, 9]
  )"));
}

TEST_P(IndexingAnalysisTest,
       FusionWithRTVarsSimplification_ScalarConstantOutsideOfRangeIsKept) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        p0 = s32[100] parameter(0)
        offset = s64[] constant(99)
        ROOT dynamic-slice = s32[10]
          dynamic-slice(p0, offset), dynamic_slice_sizes={10}
      }
      ENTRY main {
        p0 = s32[100] parameter(0)
        ROOT fusion = s32[10] fusion(p0), kind=kInput, calls=fused_computation
      }
    )hlo"));

  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
      (d0){rt0} -> (d0 + rt0),
      domain:
        d0 in [0, 9],
        rt0 in [0, 90]
      runtime variables:
        rt0: constant(99); (d0) -> (),
          domain: d0 in [0, 9]
  )"));
}

TEST_P(IndexingAnalysisTest, FusionWithRTVarsSimplification_Iota) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        p0 = f32[33,76] parameter(0)
        iota = s64[42,1] iota(), iota_dimension=0
        ROOT gather = f32[42,1,1] gather(p0, iota),
          offset_dims={1,2},
          collapsed_slice_dims={},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1,1}
      }
      ENTRY main {
        p0 = f32[33,76] parameter(0)
        ROOT fusion = f32[42,1,1] fusion(p0), kind=kInput, calls=fused_computation
      }
    )hlo"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
    (d0, d1, d2) -> (d0, 0),
     domain:
     d0 in [0, 41],
     d1 in [0, 0],
     d2 in [0, 0]
  )"));
}

TEST_P(IndexingAnalysisTest, FusionWithRTVarsSimplification_IotaAsConstant) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        p0 = f32[33,76] parameter(0)
        iota = s64[42,1] iota(), iota_dimension=1
        ROOT gather = f32[42,1,1] gather(p0, iota),
          offset_dims={1,2},
          collapsed_slice_dims={},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1,1}
      }
      ENTRY main {
        p0 = f32[33,76] parameter(0)
        ROOT fusion = f32[42,1,1] fusion(p0), kind=kInput, calls=fused_computation
      }
    )hlo"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
    (d0, d1, d2) -> (0, 0),
     domain:
     d0 in [0, 41],
     d1 in [0, 0],
     d2 in [0, 0]
  )"));
}

TEST_P(IndexingAnalysisTest, FusionOpWithPadAndDynamicSlice) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"hlo(
      HloModule m
      fused_computation {
        bitcast = s32[1,4096]{1,0} parameter(0)
        constant = s32[] constant(0)
        pad = s32[1,8192]{1,0} pad(bitcast, constant), padding=0_0x4096_0
        slice = s32[1]{0} parameter(1)
        bitcast.4 = s32[] bitcast(slice)
        ROOT dynamic-slice = s32[1,4096]{1,0}
          dynamic-slice(pad, constant, bitcast.4), dynamic_slice_sizes={1,4096}
      }
      ENTRY main {
        param_0 = s32[1,4096]{1,0} parameter(0)
        param_1 = s32[1]{0} parameter(1)
        ROOT fusion = s32[1,4096]{1,0} fusion(param_0, param_1), kind=kInput,
          calls=fused_computation
      }
    )hlo"));
  // HLO uses instruction name, StableHLO uses op name
  EXPECT_THAT(input_indexing.ToString(),
              MatchIndexingString(
                  absl::StrFormat(R"(
    operand id = 0
      (d0, d1){rt0} -> (0, d1 + rt0 - 4096),
      domain:
        d0 in [0, 0],
        d1 in [0, 4095],
        rt0 in [0, 4096],
        d1 + rt0 in [4096, 8191]
      runtime variables:
        rt0: %s; (d0, d1) -> (),
          domain: d0 in [0, 0], d1 in [0, 4095]
    operand id = 1
      (d0, d1) -> (0),
      domain:
        d0 in [0, 0],
        d1 in [0, 4095]
    )",
                                  GetParam() ? "mhlo.bitcast" : "bitcast.4")));
}

TEST_P(IndexingAnalysisTest, NestedDotFusionWithDynamicUpdateSlice) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule t

    dot_lhs {
      ROOT p0 = f32[2,4]{1,0} parameter(0)
    }

    dot_rhs {
      p0 = f32[4,5,2]{2,1,0} parameter(0)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      dynamic_slice = f32[1,5,2]{2,1,0} dynamic-slice(p0, p1, p2, p3), dynamic_slice_sizes={1,5,2}
      ROOT bitcast = f32[5,2]{1,0} bitcast(dynamic_slice)
    }

    triton_gemm {
      dot_lhs_param = f32[2,4]{1,0} parameter(0)
      lhs = f32[2,4]{1,0} fusion(dot_lhs_param), kind=kCustom, calls=dot_lhs
      dynamic_slice_input = f32[4,5,2]{2,1,0} parameter(1)
      start_index0 = s32[] parameter(2)
      start_index1 = s32[] parameter(3)
      start_index2 = s32[] parameter(4)
      rhs = f32[5,2]{1,0} fusion(dynamic_slice_input, start_index0, start_index1, start_index2), kind=kCustom, calls=dot_rhs
      ROOT dot = f32[4,5]{1,0} dot(lhs, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={1}
    }

    ENTRY e {
      p0 = f32[2,4]{1,0} parameter(0)
      p1 = f32[4,5,2]{2,1,0} parameter(1)
      p2 = s32[] parameter(2)
      c0 = s32[] constant(0)
      c1 = s32[] constant(0)
      ROOT fusion = f32[4,5]{1,0} fusion(p0, p1, p2, c0, c1), kind=kCustom, calls=triton_gemm
    }
    )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0 (d0, d1)[s0] -> (s0, d0),
      domain: d0 in [0, 3], d1 in [0, 4], s0 in [0, 1]
    operand id = 1 (d0, d1)[s0]{rt0} -> (rt0, d1, s0),
      domain: d0 in [0, 3], d1 in [0, 4], s0 in [0, 1], rt0 in [0, 3]
    runtime variables:
      rt0: parameter(1);
        (d0, d1)[s0] -> (), domain: d0 in [0, 3], d1 in [0, 4], s0 in [0, 1]
    operand id = 2 (d0, d1) -> (),
      domain: d0 in [0, 3], d1 in [0, 4]
    operand id = 3 (d0, d1) -> (),
      domain: d0 in [0, 3], d1 in [0, 4]
    operand id = 4 (d0, d1) -> (),
      domain: d0 in [0, 3], d1 in [0, 4]
  )"));
}

TEST_P(IndexingAnalysisTest, AllGatherOp) {
  auto input_indexing = GetOutputToInputIndexing(ParseAndGetRoot(R"(
    HloModule m, replica_count=4
    ENTRY e {
      p0 = f32[128, 128] parameter(0)
      ROOT all-gather = f32[128, 256] all-gather(p0),
        replica_groups={{0,1}, {2,3}}, dimensions={1}
    }
  )"));
  EXPECT_THAT(input_indexing.ToString(), MatchIndexingString(R"(
    operand id = 0
      (d0, d1) -> (d0, d1 mod 128),
      domain:
        d0 in [0, 127],
        d1 in [0, 255]
      replica id:
        (d0, d1) -> (d1 floordiv 128),
        domain:
          d0 in [0, 127],
          d1 in [0, 255]
  )"));
}

TEST_P(IndexingAnalysisTest, AllGatherFusionWithTranspose) {
  auto root = ParseAndGetRoot(R"(
    HloModule m

    fusion_computation {
      p0 = f32[16384] parameter(0)
      p1 = f32[128, 256] parameter(1)
      bitcast = f32[128, 128] bitcast(p0)
      all-gather = f32[256, 128] all-gather(bitcast), replica_groups={{0,1}},
        dimensions={0}
      transpose = f32[128, 256] transpose(all-gather), dimensions={1, 0}
      ROOT add = f32[128, 256] add(transpose, p1)
    }

    ENTRY e {
      p0 = f32[16384] parameter(0)
      p1 = f32[128, 256] parameter(1)
      ROOT fusion = f32[128, 256] fusion(p0, p1), kind=kLoop, calls=fusion_computation
    }
  )");
  EXPECT_THAT(GetOutputToInputIndexing(root).ToString(), MatchIndexingString(R"(
    operand id = 0
      (d0, d1) -> ((d1 mod 128) * 128 + d0),
      domain:
        d0 in [0, 127],
        d1 in [0, 255]
      replica id:
        (d0, d1) -> (d1 floordiv 128),
        domain:
          d0 in [0, 127],
          d1 in [0, 255]
    operand id = 1
      (d0, d1) -> (d0, d1),
      domain:
        d0 in [0, 127],
        d1 in [0, 255]
  )"));
}

TEST_P(IndexingAnalysisTest, AllGatherFusionWithReshape) {
  auto root = ParseAndGetRoot(R"(
    HloModule m

    ENTRY e {
      p0 = f32[128, 128] parameter(0)
      all-gather = f32[256, 128] all-gather(p0), replica_groups={{0,1}}, dimensions={0}
      ROOT reshape = f32[32768] reshape(all-gather)
    }
  )");

  auto all_gather = root->operand(0);
  auto parameter = all_gather->operand(0);

  auto fusion_adaptor = HloFusionAdaptor::ForProducerConsumer(all_gather, root);

  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, fusion_adaptor->GetRoots()[0], &mlir_context_);

  EXPECT_THAT(grouped_indexing[root], ElementsAre(MatchOperandIndexing(R"(
    (d0) -> (d0),
    domain:
      d0 in [0, 32767]
  )")));

  EXPECT_THAT(grouped_indexing[all_gather], ElementsAre(MatchOperandIndexing(R"(
    (d0) -> (d0 floordiv 128, d0 mod 128),
    domain:
      d0 in [0, 32767]
  )")));

  EXPECT_THAT(grouped_indexing[parameter], ElementsAre(MatchOperandIndexing(R"(
    (d0) -> ((d0 floordiv 128) mod 128, d0 mod 128),
    domain:
      d0 in [0, 32767]
    replica id:
      (d0) -> ((d0 floordiv 128) floordiv 128),
      domain:
        d0 in [0, 32767],
        d0 floordiv 128 in [0, 255],
        d0 mod 128 in [0, 127]
  )")));
}

TEST_P(IndexingAnalysisTest, ChainedAllGatherFusion) {
  auto root = ParseAndGetRoot(R"(
    HloModule m

    ENTRY e {
      p0 = f32[128] parameter(0)
      all-gather.0 = f32[256] all-gather(p0), replica_groups={{0,1}}, dimensions={0}
      ROOT all-gather.1 = f32[512] all-gather(all-gather.0), replica_groups={{0,1}}, dimensions={0}
    }
  )");

  auto all_gather = root->operand(0);
  auto parameter = all_gather->operand(0);

  auto fusion_adaptor = HloFusionAdaptor::ForProducerConsumer(all_gather, root);

  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, fusion_adaptor->GetRoots()[0], &mlir_context_);

  EXPECT_THAT(grouped_indexing[parameter],
              ElementsAre(UndefinedOperandIndexing()));
}

TEST_P(IndexingAnalysisTest, AllGatherDotFusion_GatherNonContractingDim) {
  auto root = ParseAndGetRoot(R"(
    HloModule m

    ENTRY e {
      p0 = f32[64, 256] parameter(0)
      p1 = f32[256, 128] parameter(1)
      all-gather = f32[128, 256] all-gather(p0), replica_groups={{0,1}}, dimensions={0}
      ROOT dot = f32[128, 128] dot(all-gather, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )");

  auto all_gather = root->operand(0);
  auto parameter = all_gather->operand(0);

  auto fusion_adaptor = HloFusionAdaptor::ForProducerConsumer(all_gather, root);

  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, fusion_adaptor->GetRoots()[0], &mlir_context_);

  EXPECT_THAT(grouped_indexing[parameter], ElementsAre(MatchOperandIndexing(R"(
    (d0, d1)[s0] -> (d0 mod 64, s0),
    domain:
      d0 in [0, 127],
      d1 in [0, 127],
      s0 in [0, 255]
    replica id:
      (d0, d1)[s0] -> (d0 floordiv 64),
      domain:
        d0 in [0, 127],
        d1 in [0, 127],
        s0 in [0, 255]
  )")));
}

TEST_P(IndexingAnalysisTest, AllGatherDotFusion_GatherContractingDim) {
  auto root = ParseAndGetRoot(R"(
    HloModule m

    ENTRY e {
      p0 = f32[128, 128] parameter(0)
      p1 = f32[256, 128] parameter(1)
      all-gather = f32[128, 256] all-gather(p0), replica_groups={{0,1}}, dimensions={1}
      ROOT dot = f32[128, 128] dot(all-gather, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )");

  auto all_gather = root->operand(0);
  auto parameter = all_gather->operand(0);

  auto fusion_adaptor = HloFusionAdaptor::ForProducerConsumer(all_gather, root);

  auto grouped_indexing = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, fusion_adaptor->GetRoots()[0], &mlir_context_);

  EXPECT_THAT(grouped_indexing[parameter], ElementsAre(MatchOperandIndexing(R"(
    (d0, d1)[s0] -> (d0, s0 mod 128),
    domain:
      d0 in [0, 127],
      d1 in [0, 127],
      s0 in [0, 255]
    replica id:
      (d0, d1)[s0] -> (s0 floordiv 128),
      domain:
        d0 in [0, 127],
        d1 in [0, 127],
        s0 in [0, 255]
  )")));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
