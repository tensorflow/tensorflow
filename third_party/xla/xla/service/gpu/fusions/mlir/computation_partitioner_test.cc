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
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace mlir_converter {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;

using ComputationPartitionerTest = HloTestBase;

TEST_F(ComputationPartitionerTest, PartitionDiamonds) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    fused_computation {
      %param = f32[6] parameter(0)
      %slice0.1 = f32[5] slice(f32[6]{0} %param), slice={[0:5]}
      %slice0.2 = f32[5] slice(f32[6]{0} %param), slice={[1:6]}
      %add0 = f32[5] add(f32[5]{0} %slice0.1, f32[5]{0} %slice0.2)
      %slice1.1 = f32[4] slice(f32[5]{0} %add0), slice={[0:4]}
      %slice1.2 = f32[4] slice(f32[5]{0} %add0), slice={[1:5]}
      %add1 = f32[4] add(f32[4]{0} %slice1.1, f32[4]{0} %slice1.2)
      %slice2.1 = f32[3] slice(f32[4]{0} %add1), slice={[0:3]}
      %slice2.2 = f32[3] slice(f32[4]{0} %add1), slice={[1:4]}
      %add2 = f32[3] add(f32[3]{0} %slice2.1, f32[3]{0} %slice2.2)
      %slice3.1 = f32[2] slice(f32[3]{0} %add2), slice={[0:2]}
      %slice3.2 = f32[2] slice(f32[3]{0} %add2), slice={[1:3]}
      ROOT %add3 = f32[2] add(f32[2]{0} %slice3.1, f32[2]{0} %slice3.2)
    })")
                    .value();

  auto* fusion = module->GetComputationWithName("fused_computation");
  ASSERT_NE(fusion, nullptr);
  PartitionedComputation computation(fusion);
  auto slice01 = fusion->GetInstructionWithName("slice0.1");
  auto slice02 = fusion->GetInstructionWithName("slice0.2");
  auto add0 = fusion->GetInstructionWithName("add0");
  auto slice11 = fusion->GetInstructionWithName("slice1.1");
  auto slice12 = fusion->GetInstructionWithName("slice1.2");
  auto add1 = fusion->GetInstructionWithName("add1");
  auto slice21 = fusion->GetInstructionWithName("slice2.1");
  auto slice22 = fusion->GetInstructionWithName("slice2.2");
  auto add2 = fusion->GetInstructionWithName("add2");
  auto slice31 = fusion->GetInstructionWithName("slice3.1");
  auto slice32 = fusion->GetInstructionWithName("slice3.2");
  auto add3 = fusion->GetInstructionWithName("add3");

  const auto& graphs = computation.subgraphs();
  ASSERT_THAT(graphs, SizeIs(4));
  EXPECT_THAT(graphs[0].instructions_post_order,
              ElementsAre(slice01, slice02, add0));
  EXPECT_THAT(graphs[1].instructions_post_order,
              ElementsAre(slice11, slice12, add1));
  EXPECT_THAT(graphs[2].instructions_post_order,
              ElementsAre(slice21, slice22, add2));
  EXPECT_THAT(graphs[3].instructions_post_order,
              ElementsAre(slice31, slice32, add3));

  EXPECT_THAT(graphs[0].roots, ElementsAre(add0));
  EXPECT_THAT(graphs[1].roots, ElementsAre(add1));
  EXPECT_THAT(graphs[2].roots, ElementsAre(add2));
  EXPECT_THAT(graphs[3].roots, ElementsAre(add3));

  EXPECT_EQ(&computation.GetRootSubgraph(), &graphs[3]);
  EXPECT_EQ(&computation.FindSubgraph(slice21), &graphs[2]);
}

TEST_F(ComputationPartitionerTest, TupleRoot) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    fused_computation {
      %p0 = f32[6] parameter(0)
      %p1 = f32[6] parameter(1)
      %add = f32[6] add(p0, p1)
      %sub = f32[6] subtract(p0, p1)
      ROOT %root = (f32[6], f32[6]) tuple(%add, %sub)
    })")
                    .value();

  auto* fusion = module->GetComputationWithName("fused_computation");
  ASSERT_NE(fusion, nullptr);
  PartitionedComputation computation(fusion);

  ASSERT_THAT(computation.subgraphs(), SizeIs(1));
  EXPECT_THAT(computation.GetRootSubgraph().roots, SizeIs(1));
  EXPECT_THAT(computation.GetRootSubgraph().instructions_post_order, SizeIs(3));
}

TEST_F(ComputationPartitionerTest, EnforcePartitioning) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    fused_computation {
      %p0 = f32[64, 32] parameter(0)
      %p1 = f32[64, 32] parameter(1)
      %add = f32[64, 32] add(p0, p1)
      %transpose = f32[32, 64] transpose(%add), dimensions={1, 0}
      %exp = f32[32, 64] exponential(%transpose)
      ROOT %root = f32[32, 64] tanh(%exp)
    })")
                    .value();

  auto* fusion = module->GetComputationWithName("fused_computation");
  ASSERT_NE(fusion, nullptr);
  PartitionedComputation computation(fusion, [](const HloInstruction* instr) {
    return instr->opcode() == HloOpcode::kTranspose;
  });
  ASSERT_THAT(computation.subgraphs(), SizeIs(2));
  EXPECT_THAT(computation.GetRootSubgraph().roots, SizeIs(1));
  EXPECT_THAT(computation.GetRootSubgraph().instructions_post_order, SizeIs(2));
}

TEST_F(ComputationPartitionerTest, PartiallyMergable) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    fused_computation {
      %p0 = f32[10,10] parameter(0)
      %p1 = f32[10,10] parameter(1)
      %add = f32[10,10] add(%p0, %p1)
      %transpose = f32[10,10] transpose(%add), dimensions={1,0}
      ROOT %sub = f32[10,10] subtract(%add, %transpose)
    })")
                    .value();

  auto* fusion = module->GetComputationWithName("fused_computation");
  ASSERT_NE(fusion, nullptr);
  PartitionedComputation computation(fusion);

  auto transpose = fusion->GetInstructionWithName("transpose");
  auto sub = fusion->GetInstructionWithName("sub");

  ASSERT_THAT(computation.subgraphs(), SizeIs(2));
  EXPECT_THAT(computation.GetRootSubgraph().instructions_post_order,
              ElementsAre(transpose, sub));
}

std::string PrintAndErase(mlir::func::FuncOp func) {
  // Set visibility to private so the function verifies.
  func.setSymVisibility("private");
  std::string out;
  llvm::raw_string_ostream os(out);
  os << func;
  // Erase the function so we don't leak memory.
  func.erase();
  return out;
}

TEST_F(ComputationPartitionerTest, SubgraphSignatures) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      %p0 = f32[] parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %add = f32[] add(%p0, %p1)
    }

    fusion {
      %p0 = f32[10,10]{0,1} parameter(0)
      %p1 = f32[10,10]{1,0} parameter(1)
      %c0 = f32[] constant(2)
      %bc = f32[10,10]{0,1} bitcast(%p1)
      %add = f32[10,10] add(%p0, %bc)
      ROOT %reduce = f32[10] reduce(%add, %c0), dimensions={1}, to_apply=add
    }

    ENTRY main {
      %p0 = f32[10,10] parameter(0)
      %p1 = f32[10,10] parameter(1)
      ROOT %fusion = f32[10] fusion(%p0, %p1), kind=kLoop, calls=fusion
    })")
                    .value();

  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(&context), &context);

  PartitionedComputation fusion(module->GetComputationWithName("fusion"));
  EXPECT_EQ(
      PrintAndErase(
          CreateSubgraphMlirFunction(fusion.GetRootSubgraph(), builder)),
      "func.func private @fusion_reduce(tensor<10x10xf32, dense<[0, 1]> : "
      "tensor<2xi64>>, tensor<10x10xf32>, index {xla.range = [0 : index, 9 : "
      "index]}) -> f32");

  PartitionedComputation add(module->GetComputationWithName("add"));
  EXPECT_EQ(
      PrintAndErase(CreateSubgraphMlirFunction(add.GetRootSubgraph(), builder)),
      "func.func private @add_add(f32, f32) -> f32");
}

TEST_F(ComputationPartitionerTest, SubgraphSignaturesWithInjectedValues) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    %fused_computation {
      %p0 = f32[2,16,17] parameter(0)
      %log = f32[2,16,17] log(%p0)
      %transpose = f32[2,17,16] transpose(%log), dimensions={0,2,1}
      %p1 = f32[] parameter(1)
      %bc = f32[2,17,16] broadcast(%p1), dimensions={}
      ROOT %add = f32[2,17,16] add(%transpose, %bc)
    }

    ENTRY main {
      %p0 = f32[2,16,17] parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %fusion = f32[2,17,16] fusion(%p0, %p1), kind=kInput,
            calls=%fused_computation
    }
  )")
                    .value();

  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(&context), &context);

  // We force a split at the transpose (like the transpose emitter would do) and
  // enforce that the transpose is injected as a parameter into the epilogue.
  auto* fused_computation = module->GetComputationWithName("fused_computation");
  PartitionedComputation fusion(
      fused_computation,
      [](const HloInstruction* instr) {
        // Make the transpose a new root.
        return instr->opcode() == HloOpcode::kTranspose;
      },
      [](const HloInstruction* instr, int operand) {
        // Inject the transpose argument.
        return instr->operand(operand)->opcode() == HloOpcode::kTranspose;
      });
  auto& injected_params = fusion.GetRootSubgraph().injected_param_indices;
  EXPECT_EQ(injected_params.size(), 1);
  std::pair<const HloInstruction*, int> injected_operand_key(
      fused_computation->root_instruction(), 0);
  ASSERT_TRUE(injected_params.contains(injected_operand_key));
  EXPECT_EQ(injected_params.at(injected_operand_key), 0);
  EXPECT_EQ(PrintAndErase(
                CreateSubgraphMlirFunction(fusion.GetRootSubgraph(), builder)),
            "func.func private @fused_computation_add(tensor<2x16x17xf32>, "
            "tensor<f32>, index {xla.range = [0 : index, 1 : index]}, index "
            "{xla.range = [0 : index, 16 : index]}, index {xla.range = [0 : "
            "index, 15 : index]}, f32) -> f32");
}

}  // namespace
}  // namespace mlir_converter
}  // namespace gpu
}  // namespace xla
