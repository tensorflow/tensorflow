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
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace mlir_converter {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class ComputationPartitionerTest : public HloTestBase {
 protected:
  ComputationPartitionerTest() {
    mlir_context_.loadDialect<mlir::func::FuncDialect>();
  }

  mlir::MLIRContext mlir_context_;
};

std::string PrintAndErase(mlir::func::FuncOp func) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << func;
  // Erase the function so we don't leak memory.
  func.erase();
  return out;
}

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
  PartitionedComputation computation(fusion, &mlir_context_);
  auto param = fusion->GetInstructionWithName("param");
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
  ASSERT_THAT(graphs, SizeIs(5));
  EXPECT_THAT(graphs[0].instructions, UnorderedElementsAre(param));
  EXPECT_THAT(graphs[1].instructions,
              UnorderedElementsAre(slice01, slice02, add0));
  EXPECT_THAT(graphs[2].instructions,
              UnorderedElementsAre(slice11, slice12, add1));
  EXPECT_THAT(graphs[3].instructions,
              UnorderedElementsAre(slice21, slice22, add2));
  EXPECT_THAT(graphs[4].instructions,
              UnorderedElementsAre(slice31, slice32, add3));

  EXPECT_THAT(graphs[1].roots, ElementsAre(add0));
  EXPECT_THAT(graphs[2].roots, ElementsAre(add1));
  EXPECT_THAT(graphs[3].roots, ElementsAre(add2));
  EXPECT_THAT(graphs[4].roots, ElementsAre(add3));

  EXPECT_EQ(&computation.GetRootSubgraph(), &graphs[4]);
  EXPECT_EQ(&computation.FindSubgraph(slice21), &graphs[3]);
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
  PartitionedComputation computation(fusion, &mlir_context_);

  ASSERT_THAT(computation.subgraphs(), SizeIs(5)) << computation.ToString();
}

TEST_F(ComputationPartitionerTest, Epilogue) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    fused_computation {
      p0 = f32[4] parameter(0)
      c0 = f32[] constant(0)
      reduce = f32[] reduce(p0, c0), dimensions={0}, to_apply=add
      bitcast = f32[1] bitcast(reduce)
      abs = f32[1] abs(bitcast)
      log = f32[1] log(abs)
      sign = f32[1] sign(bitcast)
      ROOT tuple = (f32[1], f32[1]) tuple(log, sign)
    })")
                    .value();

  auto* fused_computation = module->GetComputationWithName("fused_computation");
  EpilogueSpecification epilogue{
      /*heroes=*/{fused_computation->GetInstructionWithName("reduce")},
      /*roots=*/
      {fused_computation->GetInstructionWithName("log"),
       fused_computation->GetInstructionWithName("sign")},
      /*index_ranges=*/{1, 42},
      {CreateIdentityMap(
          fused_computation->root_instruction()->shape().tuple_shapes(0),
          &mlir_context_)}};
  PartitionedComputations fusion(fused_computation, &mlir_context_, {epilogue});

  mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(&mlir_context_),
                                     &mlir_context_);
  EXPECT_EQ(
      PrintAndErase(
          CreateSubgraphMlirFunction(fusion.epilogues().front(), builder)),
      "func.func private @fused_computation__epilogue__log_sign(tensor<4xf32>, "
      "index {xla.range = [0 : index, 0 : index]}, "
      "index {xla.range = [0 : index, 41 : index]}, "
      "f32) -> (f32, f32)");
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
  PartitionedComputation computation(
      fusion, &mlir_context_, [](const HloInstruction* instr) {
        return instr->opcode() == HloOpcode::kTranspose;
      });
  ASSERT_THAT(computation.subgraphs(), SizeIs(2));
  EXPECT_THAT(computation.GetRootSubgraph().roots, SizeIs(1));
  EXPECT_THAT(computation.GetRootSubgraph().instructions, SizeIs(2));
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
  PartitionedComputation computation(fusion, &mlir_context_);

  auto transpose = fusion->GetInstructionWithName("transpose");
  auto sub = fusion->GetInstructionWithName("sub");

  ASSERT_THAT(computation.subgraphs(), SizeIs(2));
  EXPECT_THAT(computation.GetRootSubgraph().instructions,
              UnorderedElementsAre(transpose, sub));
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

  PartitionedComputation fusion(module->GetComputationWithName("fusion"),
                                &mlir_context_);
  EXPECT_EQ(
      PrintAndErase(
          CreateSubgraphMlirFunction(fusion.GetRootSubgraph(), builder)),
      "func.func private @fusion_reduce(tensor<10x10xf32, dense<[0, 1]> : "
      "tensor<2xi64>>, tensor<10x10xf32>, index {xla.range = [0 : index, 9 : "
      "index]}) -> f32");

  PartitionedComputation add(module->GetComputationWithName("add"),
                             &mlir_context_);
  EXPECT_EQ(
      PrintAndErase(CreateSubgraphMlirFunction(add.GetRootSubgraph(), builder)),
      "func.func private @add_add(f32, f32) -> f32");
}

}  // namespace
}  // namespace mlir_converter
}  // namespace gpu
}  // namespace xla
