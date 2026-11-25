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

#include "xla/backends/gpu/codegen/emitters/nontemporal_hints.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialectDecl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::IsNull;
using ::testing::NotNull;
using ::xla::emitters::EmitKernelApi;
using ::xla::emitters::kXlaContiguousOutputAttr;
using ::xla::emitters::kXlaNotBroadcastAttr;
using ::xla::emitters::kXlaStreamedOnceAttr;

class NontemporalHintsTest : public HloHardwareIndependentTestBase {
 protected:
  const HloFusionInstruction* GetFusion(const HloModule* module,
                                        absl::string_view name) {
    return Cast<HloFusionInstruction>(FindInstruction(module, name));
  }

  // Uses the dataflow analysis settings of buffer assignment.
  absl::StatusOr<std::unique_ptr<HloDataflowAnalysis>> Dataflow(
      const HloModule* module) {
    return HloDataflowAnalysis::Run(*module, /*ssa_form=*/true);
  }

  absl::StatusOr<std::vector<OperandReads>> Reads(
      const HloModule* module, absl::string_view fusion_name) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloDataflowAnalysis> dataflow,
                     Dataflow(module));
    mlir::MLIRContext ctx;
    return ComputeOperandReads(*GetFusion(module, fusion_name), dataflow.get(),
                               &ctx);
  }
};

TEST_F(NontemporalHintsTest, ElementwiseOperandsAreStreamedOnce) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
add_comp {
  p0 = f32[1024] parameter(0)
  p1 = f32[1024] parameter(1)
  ROOT a = f32[1024] add(p0, p1)
}

ENTRY e {
  a = f32[1024] parameter(0)
  b = f32[1024] parameter(1)
  ROOT f = f32[1024] fusion(a, b), kind=kLoop, calls=add_comp
})"));
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> reads,
                       Reads(module.get(), "f"));
  ASSERT_EQ(reads.size(), 2);
  for (const OperandReads& r : reads) {
    EXPECT_TRUE(r.streamed_once);
    EXPECT_TRUE(r.not_broadcast);
    EXPECT_FALSE(r.read_elsewhere);
  }
}

TEST_F(NontemporalHintsTest, BroadcastOperandGetsNeither) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
bc_comp {
  p0 = f32[] parameter(0)
  ROOT b = f32[1024] broadcast(p0), dimensions={}
}

ENTRY e {
  a = f32[] parameter(0)
  ROOT f = f32[1024] fusion(a), kind=kLoop, calls=bc_comp
})"));
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> reads,
                       Reads(module.get(), "f"));
  ASSERT_EQ(reads.size(), 1);
  EXPECT_FALSE(reads[0].streamed_once);
  EXPECT_FALSE(reads[0].not_broadcast);
}

TEST_F(NontemporalHintsTest, BroadcastThroughReshapeGetsNeither) {
  // Uses every output dim, (d0) -> (d0 floordiv 8), but reads elements 8x.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
bc_comp {
  p0 = f32[1024] parameter(0)
  b = f32[1024,8] broadcast(p0), dimensions={0}
  ROOT r = f32[8192] reshape(b)
}

ENTRY e {
  a = f32[1024] parameter(0)
  ROOT f = f32[8192] fusion(a), kind=kLoop, calls=bc_comp
})"));
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> reads,
                       Reads(module.get(), "f"));
  ASSERT_EQ(reads.size(), 1);
  EXPECT_FALSE(reads[0].streamed_once);
  EXPECT_FALSE(reads[0].not_broadcast);
}

TEST_F(NontemporalHintsTest, OverlappingWindowsGetNeither) {
  // (d0)[s0] -> (d0 + s0): neighboring windows read the same elements.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

rw_comp {
  p0 = f32[1024] parameter(0)
  c = f32[] constant(0)
  ROOT rw = f32[1021] reduce-window(p0, c), window={size=4}, to_apply=add
}

ENTRY e {
  a = f32[1024] parameter(0)
  ROOT f = f32[1021] fusion(a), kind=kLoop, calls=rw_comp
})"));
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> reads,
                       Reads(module.get(), "f"));
  ASSERT_EQ(reads.size(), 1);
  EXPECT_FALSE(reads[0].streamed_once);
  EXPECT_FALSE(reads[0].not_broadcast);
}

TEST_F(NontemporalHintsTest, ReduceInputIsNotBroadcastButNotStreamedOnce) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
max_comp {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT m = f32[] maximum(a, b)
}

reduce_comp {
  p0 = f32[1024,1024] parameter(0)
  c = f32[] constant(-inf)
  ROOT r = f32[1024] reduce(p0, c), dimensions={1}, to_apply=max_comp
}

ENTRY e {
  a = f32[1024,1024] parameter(0)
  ROOT f = f32[1024] fusion(a), kind=kInput, calls=reduce_comp
})"));
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> reads,
                       Reads(module.get(), "f"));
  ASSERT_EQ(reads.size(), 1);
  EXPECT_TRUE(reads[0].not_broadcast);
  EXPECT_FALSE(reads[0].streamed_once);
}

TEST_F(NontemporalHintsTest, ScalarReductions) {
  // 0-D outputs have no dims: only the range vars tell the reads apart.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

broadcast_comp {
  p0 = f32[1024] parameter(0)
  b = f32[1024,8] broadcast(p0), dimensions={0}
  r = f32[8192] reshape(b)
  c = f32[] constant(0)
  ROOT s = f32[] reduce(r, c), dimensions={0}, to_apply=add
}

plain_comp {
  p0 = f32[1024] parameter(0)
  c = f32[] constant(0)
  ROOT s = f32[] reduce(p0, c), dimensions={0}, to_apply=add
}

ENTRY e {
  a = f32[1024] parameter(0)
  b = f32[1024] parameter(1)
  broadcast = f32[] fusion(a), kind=kInput, calls=broadcast_comp
  plain = f32[] fusion(b), kind=kInput, calls=plain_comp
  ROOT t = (f32[], f32[]) tuple(broadcast, plain)
})"));
  // ()[s0] -> (s0 floordiv 8): each element is read 8 times.
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> broadcast,
                       Reads(module.get(), "broadcast"));
  ASSERT_EQ(broadcast.size(), 1);
  EXPECT_FALSE(broadcast[0].not_broadcast);
  EXPECT_FALSE(broadcast[0].streamed_once);

  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> plain,
                       Reads(module.get(), "plain"));
  ASSERT_EQ(plain.size(), 1);
  EXPECT_TRUE(plain[0].not_broadcast);
  EXPECT_FALSE(plain[0].streamed_once);
}

TEST_F(NontemporalHintsTest, ConcatenateOfSameOperandIsNotStreamedOnce) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
concat_comp {
  p0 = f32[512] parameter(0)
  ROOT c = f32[1024] concatenate(p0, p0), dimensions={0}
}

ENTRY e {
  a = f32[512] parameter(0)
  ROOT f = f32[1024] fusion(a), kind=kLoop, calls=concat_comp
})"));
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> reads,
                       Reads(module.get(), "f"));
  ASSERT_EQ(reads.size(), 1);
  // Two maps (one per slice): read more than once.
  EXPECT_FALSE(reads[0].streamed_once);
}

TEST_F(NontemporalHintsTest, DynamicSliceDataOperandGetsNeither) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
ds_comp {
  p0 = f32[1024] parameter(0)
  p1 = s32[] parameter(1)
  ROOT d = f32[64] dynamic-slice(p0, p1), dynamic_slice_sizes={64}
}

ENTRY e {
  a = f32[1024] parameter(0)
  o = s32[] parameter(1)
  ROOT f = f32[64] fusion(a, o), kind=kLoop, calls=ds_comp
})"));
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> reads,
                       Reads(module.get(), "f"));
  ASSERT_EQ(reads.size(), 2);
  // The data operand is read through a runtime offset.
  EXPECT_FALSE(reads[0].streamed_once);
  EXPECT_FALSE(reads[0].not_broadcast);
}

TEST_F(NontemporalHintsTest, VariadicReduceDoesNotCrash) {
  // Regression: a variadic reduce is one root with several outputs.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
add_pair {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f32[] parameter(2)
  d = f32[] parameter(3)
  add0 = f32[] add(a, c)
  add1 = f32[] add(b, d)
  ROOT t = (f32[], f32[]) tuple(add0, add1)
}

vreduce_comp {
  p0 = f32[1024,1024] parameter(0)
  p1 = f32[1024,1024] parameter(1)
  c0 = f32[] constant(0)
  c1 = f32[] constant(0)
  reduce = (f32[1024], f32[1024]) reduce(p0, p1, c0, c1), dimensions={1},
      to_apply=add_pair
  gte0 = f32[1024] get-tuple-element(reduce), index=0
  gte1 = f32[1024] get-tuple-element(reduce), index=1
  ROOT t = (f32[1024], f32[1024]) tuple(gte0, gte1)
}

ENTRY e {
  a = f32[1024,1024] parameter(0)
  b = f32[1024,1024] parameter(1)
  ROOT f = (f32[1024], f32[1024]) fusion(a, b), kind=kInput, calls=vreduce_comp
})"));
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> reads,
                       Reads(module.get(), "f"));
  ASSERT_EQ(reads.size(), 2);
  for (const OperandReads& r : reads) {
    EXPECT_TRUE(r.not_broadcast);
    EXPECT_FALSE(r.streamed_once);
  }
}

TEST_F(NontemporalHintsTest, ScatterFusionReturnsEmpty) {
  // Regression (torax): scatter operands have no indexing map.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
update_comp {
  a = f32[] parameter(0)
  ROOT b = f32[] parameter(1)
}

scatter_comp {
  operand = f32[1024,1024] parameter(0)
  indices = s32[16,1] parameter(1)
  updates = f32[16,1024] parameter(2)
  offset = s32[] parameter(3)
  sliced = f32[16,1024] dynamic-slice(updates, offset, offset),
      dynamic_slice_sizes={16,1024}
  ROOT s = f32[1024,1024] scatter(operand, indices, sliced),
      update_window_dims={1}, inserted_window_dims={0},
      scatter_dims_to_operand_dims={0}, index_vector_dim=1,
      to_apply=update_comp
}

ENTRY e {
  operand = f32[1024,1024] parameter(0)
  indices = s32[16,1] parameter(1)
  updates = f32[16,1024] parameter(2)
  offset = s32[] parameter(3)
  ROOT f = f32[1024,1024] fusion(operand, indices, updates, offset),
      kind=kInput, calls=scatter_comp
})"));
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> reads,
                       Reads(module.get(), "f"));
  EXPECT_TRUE(reads.empty());
}

TEST_F(NontemporalHintsTest, MultiOutputConcatenateUnionsRoots) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
mo_concat_comp {
  p0 = f32[512] parameter(0)
  p1 = f32[512] parameter(1)
  c0 = f32[1024] concatenate(p0, p1), dimensions={0}
  c1 = f32[1024] concatenate(p0, p1), dimensions={0}
  ROOT t = (f32[1024], f32[1024]) tuple(c0, c1)
}

ENTRY e {
  a = f32[512] parameter(0)
  b = f32[512] parameter(1)
  ROOT f = (f32[1024], f32[1024]) fusion(a, b), kind=kLoop,
      calls=mo_concat_comp
})"));
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> reads,
                       Reads(module.get(), "f"));
  ASSERT_EQ(reads.size(), 2);
  // Both roots read the same slice; the maps dedupe to one.
  for (const OperandReads& r : reads) {
    EXPECT_TRUE(r.not_broadcast);
    EXPECT_TRUE(r.streamed_once);
  }
}

TEST_F(NontemporalHintsTest, ReadElsewhereViaBitcastAndTuple) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
negate_a {
  p = f32[1024] parameter(0)
  ROOT n = f32[1024] negate(p)
}

negate_b {
  p = f32[1024] parameter(0)
  ROOT n = f32[1024] negate(p)
}

ENTRY e {
  a = f32[1024] parameter(0)
  b = f32[1024] parameter(1)
  shared = f32[1024] fusion(a), kind=kLoop, calls=negate_a
  bitcast = f32[32,32] bitcast(a)
  other = f32[32,32] negate(bitcast)
  sole = f32[1024] fusion(b), kind=kLoop, calls=negate_b
  ROOT t = (f32[1024], f32[32,32], f32[1024], f32[1024])
      tuple(shared, other, sole, b)
})"));
  // `a` is also read through a bitcast; the root tuple only forwards `b`.
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> shared,
                       Reads(module.get(), "shared"));
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> sole,
                       Reads(module.get(), "sole"));
  ASSERT_EQ(shared.size(), 1);
  ASSERT_EQ(sole.size(), 1);
  EXPECT_TRUE(shared[0].read_elsewhere);
  EXPECT_FALSE(sole[0].read_elsewhere);
}

TEST_F(NontemporalHintsTest, BufferReadThroughTwoOperandsIsNotStreamedOnce) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
add_comp {
  p0 = f32[1024] parameter(0)
  p1 = f32[32,32] parameter(1)
  b = f32[1024] bitcast(p1)
  ROOT a = f32[1024] add(p0, b)
}

ENTRY e {
  a = f32[1024] parameter(0)
  bitcast = f32[32,32] bitcast(a)
  ROOT f = f32[1024] fusion(a, bitcast), kind=kLoop, calls=add_comp
})"));
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> reads,
                       Reads(module.get(), "f"));
  ASSERT_EQ(reads.size(), 2);
  for (const OperandReads& r : reads) {
    EXPECT_TRUE(r.not_broadcast);
    EXPECT_FALSE(r.streamed_once);
    EXPECT_TRUE(r.read_elsewhere);
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloDataflowAnalysis> dataflow,
                       Dataflow(module.get()));
  EXPECT_THAT(
      ComputeReadElsewhere(*GetFusion(module.get(), "f"), dataflow.get()),
      ElementsAre(true, true));
}

TEST_F(NontemporalHintsTest, ReadElsewhereThroughWhile) {
  // The tuple passes `a` to a loop that reads it.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
negate_comp {
  p = f32[1024] parameter(0)
  ROOT n = f32[1024] negate(p)
}

body {
  p = (f32[1024], s32[]) parameter(0)
  x = f32[1024] get-tuple-element(p), index=0
  i = s32[] get-tuple-element(p), index=1
  n = f32[1024] negate(x)
  one = s32[] constant(1)
  next = s32[] add(i, one)
  ROOT t = (f32[1024], s32[]) tuple(n, next)
}

cond {
  p = (f32[1024], s32[]) parameter(0)
  i = s32[] get-tuple-element(p), index=1
  limit = s32[] constant(10)
  ROOT lt = pred[] compare(i, limit), direction=LT
}

ENTRY e {
  a = f32[1024] parameter(0)
  f = f32[1024] fusion(a), kind=kLoop, calls=negate_comp
  zero = s32[] constant(0)
  init = (f32[1024], s32[]) tuple(a, zero)
  w = (f32[1024], s32[]) while(init), condition=cond, body=body
  r = f32[1024] get-tuple-element(w), index=0
  ROOT t = (f32[1024], f32[1024]) tuple(f, r)
})"));
  ASSERT_OK_AND_ASSIGN(std::vector<OperandReads> reads,
                       Reads(module.get(), "f"));
  ASSERT_EQ(reads.size(), 1);
  EXPECT_TRUE(reads[0].read_elsewhere);
  EXPECT_TRUE(reads[0].streamed_once);
}

TEST_F(NontemporalHintsTest, NoDataflowAssumesReadElsewhere) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
negate_comp {
  p = f32[1024] parameter(0)
  ROOT n = f32[1024] negate(p)
}

ENTRY e {
  a = f32[1024] parameter(0)
  ROOT f = f32[1024] fusion(a), kind=kLoop, calls=negate_comp
})"));
  mlir::MLIRContext ctx;
  std::vector<OperandReads> reads = ComputeOperandReads(
      *GetFusion(module.get(), "f"), /*dataflow=*/nullptr, &ctx);
  ASSERT_EQ(reads.size(), 1);
  EXPECT_TRUE(reads[0].not_broadcast);
  EXPECT_FALSE(reads[0].streamed_once);
  EXPECT_TRUE(reads[0].read_elsewhere);
  EXPECT_THAT(ComputeReadElsewhere(*GetFusion(module.get(), "f"),
                                   /*dataflow=*/nullptr),
              ElementsAre(true));
}

TEST_F(NontemporalHintsTest, AnnotateSetsFactsOnOperandsAndContiguousOutput) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
add_comp {
  p0 = f32[1024] parameter(0)
  p1 = f32[1024] parameter(1)
  ROOT a = f32[1024] add(p0, p1)
}

ENTRY e {
  a = f32[1024] parameter(0)
  b = f32[1024] parameter(1)
  ROOT f = f32[1024] fusion(a, b), kind=kLoop, calls=add_comp
})"));
  AliasInfo alias_info;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(
          module.get(), std::make_unique<DependencyHloOrdering>(module.get()),
          [](const BufferValue& buffer) {
            return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
          },
          &alias_info, [](LogicalBuffer::Color) { return 0; },
          BufferAssigner::Options{.allocate_buffers_for_constants = true}));

  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::func::FuncDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(&ctx));
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp entry,
                       EmitKernelApi(*mlir_module, *fusion, assignment.get(),
                                     {16, 16, 16}, "entry"));

  AnnotateNontemporalHintFacts(*mlir_module, "entry", *fusion,
                               &assignment->dataflow_analysis(),
                               OutputContiguity::kContiguous);

  EXPECT_THAT(entry->getAttr(kXlaContiguousOutputAttr), NotNull());
  EXPECT_THAT(entry.getArgAttr(0, kXlaStreamedOnceAttr), NotNull());
  EXPECT_THAT(entry.getArgAttr(0, kXlaNotBroadcastAttr), NotNull());
  EXPECT_THAT(entry.getArgAttr(1, kXlaStreamedOnceAttr), NotNull());
  // The result argument (index 2) gets no read facts.
  EXPECT_THAT(entry.getArgAttr(2, kXlaStreamedOnceAttr), IsNull());
  EXPECT_THAT(entry.getArgAttr(2, kXlaNotBroadcastAttr), IsNull());
}

TEST_F(NontemporalHintsTest, AnnotateWithoutContiguousOutputOmitsAttr) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
neg_comp {
  p0 = f32[1024] parameter(0)
  ROOT n = f32[1024] negate(p0)
}

ENTRY e {
  a = f32[1024] parameter(0)
  ROOT f = f32[1024] fusion(a), kind=kLoop, calls=neg_comp
})"));
  AliasInfo alias_info;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(
          module.get(), std::make_unique<DependencyHloOrdering>(module.get()),
          [](const BufferValue& buffer) {
            return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
          },
          &alias_info, [](LogicalBuffer::Color) { return 0; },
          BufferAssigner::Options{.allocate_buffers_for_constants = true}));

  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::func::FuncDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(&ctx));
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp entry,
                       EmitKernelApi(*mlir_module, *fusion, assignment.get(),
                                     {16, 16, 16}, "entry"));

  AnnotateNontemporalHintFacts(*mlir_module, "entry", *fusion,
                               &assignment->dataflow_analysis(),
                               OutputContiguity::kNonContiguous);
  EXPECT_THAT(entry->getAttr(kXlaContiguousOutputAttr), IsNull());
  EXPECT_THAT(entry.getArgAttr(0, kXlaStreamedOnceAttr), NotNull());
}

class ReadsEachElementAtMostOnceTest : public testing::Test {
 protected:
  ReadsEachElementAtMostOnceTest() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  bool ReadsOnce(absl::string_view map, const Shape& shape) {
    std::optional<IndexingMap> indexing_map =
        ParseIndexingMap(map, &mlir_context_);
    EXPECT_TRUE(indexing_map.has_value()) << map;
    return indexing_map.has_value() &&
           ReadsEachElementAtMostOnce(*indexing_map, shape);
  }

  mlir::MLIRContext mlir_context_;
};

TEST_F(ReadsEachElementAtMostOnceTest, OneToOne) {
  // Transpose, reverse, reshape, strided slice and reduction.
  EXPECT_TRUE(ReadsOnce(R"(
    (d0, d1) -> (d1, d0),
    domain: d0 in [0, 199], d1 in [0, 99]
  )",
                        ShapeUtil::MakeShape(F32, {100, 200})));
  EXPECT_TRUE(ReadsOnce(R"(
    (d0) -> (-d0 + 99),
    domain: d0 in [0, 99]
  )",
                        ShapeUtil::MakeShape(F32, {100})));
  EXPECT_TRUE(ReadsOnce(R"(
    (d0) -> (d0 floorDiv 64, d0 mod 64),
    domain: d0 in [0, 4095]
  )",
                        ShapeUtil::MakeShape(F32, {64, 64})));
  EXPECT_TRUE(ReadsOnce(R"(
    (d0) -> (d0 * 2),
    domain: d0 in [0, 49]
  )",
                        ShapeUtil::MakeShape(F32, {100})));
  EXPECT_TRUE(ReadsOnce(R"(
    (d0)[s0] -> (d0, s0),
    domain: d0 in [0, 99], s0 in [0, 63]
  )",
                        ShapeUtil::MakeShape(F32, {100, 64})));
}

TEST_F(ReadsEachElementAtMostOnceTest, SplitDimensions) {
  // A grouped-query attention transpose and a reduction whose output was
  // reshaped: d floorDiv c and d mod c do not recombine.
  EXPECT_TRUE(ReadsOnce(R"(
    (d0, d1, d2, d3) -> (d0, d1 floorDiv 4, d3, d2 * 4 + d1 mod 4),
    domain: d0 in [0, 1], d1 in [0, 31], d2 in [0, 15], d3 in [0, 15]
  )",
                        ShapeUtil::MakeShape(F32, {2, 8, 16, 64})));
  EXPECT_TRUE(ReadsOnce(R"(
    (d0)[s0] -> (d0 floorDiv 16, s0, d0 mod 16),
    domain: d0 in [0, 127], s0 in [0, 31]
  )",
                        ShapeUtil::MakeShape(F32, {8, 32, 16})));
  // d0 and d0 + 8 read the same element.
  EXPECT_FALSE(ReadsOnce(R"(
    (d0)[s0] -> (d0 floorDiv 16, s0, d0 mod 8),
    domain: d0 in [0, 127], s0 in [0, 31]
  )",
                         ShapeUtil::MakeShape(F32, {8, 32, 8})));
  // (d0, d1) = (4, 0) and (0, 1) read the same element.
  EXPECT_FALSE(ReadsOnce(R"(
    (d0, d1) -> (d0 floorDiv 4 + d1, d0 mod 4),
    domain: d0 in [0, 15], d1 in [0, 3]
  )",
                         ShapeUtil::MakeShape(F32, {7, 4})));
}

TEST_F(ReadsEachElementAtMostOnceTest, BitcastOfNonDefaultLayout) {
  // A reduction of rows of 4 of the memory of an f32[2,4,3]{2,0,1}: the
  // floorDiv and mod recombine in the physical order only.
  EXPECT_TRUE(ReadsOnce(
      R"(
    (d0)[s0] -> (((d0 * 4 + s0) floorDiv 3) mod 2, (d0 * 4 + s0) floorDiv 6,
                 (d0 * 4 + s0) mod 3),
    domain: d0 in [0, 5], s0 in [0, 3]
  )",
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 4, 3}, {2, 0, 1})));
}

TEST_F(ReadsEachElementAtMostOnceTest, BitcastOfTranspose) {
  // A reduction of rows of 4 of transpose(f32[4,2,6], {1,0,2}): the floorDiv
  // and mod recombine in the order of the transpose only.
  EXPECT_TRUE(ReadsOnce(R"(
    (d0)[s0] -> (((d0 * 4 + s0) floorDiv 6) mod 4, (d0 * 4 + s0) floorDiv 24,
                 (d0 * 4 + s0) mod 6),
    domain: d0 in [0, 11], s0 in [0, 3]
  )",
                        ShapeUtil::MakeShape(F32, {4, 2, 6})));
}

TEST_F(ReadsEachElementAtMostOnceTest, ManyToOne) {
  // Reshaped broadcast, sum of dims, unused dim and overlapping windows.
  EXPECT_FALSE(ReadsOnce(R"(
    (d0) -> (d0 floorDiv 8),
    domain: d0 in [0, 8191]
  )",
                         ShapeUtil::MakeShape(F32, {1024})));
  EXPECT_FALSE(ReadsOnce(R"(
    (d0, d1) -> (d0 + d1),
    domain: d0 in [0, 99], d1 in [0, 99]
  )",
                         ShapeUtil::MakeShape(F32, {199})));
  EXPECT_FALSE(ReadsOnce(R"(
    (d0, d1) -> (d0),
    domain: d0 in [0, 99], d1 in [0, 7]
  )",
                         ShapeUtil::MakeShape(F32, {100})));
  EXPECT_FALSE(ReadsOnce(R"(
    (d0)[s0] -> (d0 * 2 + s0),
    domain: d0 in [0, 49], s0 in [0, 2]
  )",
                         ShapeUtil::MakeShape(F32, {101})));
}

TEST_F(ReadsEachElementAtMostOnceTest, ZeroDimensionalOutput) {
  EXPECT_TRUE(ReadsOnce("() -> ()", ShapeUtil::MakeShape(F32, {})));
  EXPECT_TRUE(ReadsOnce(R"(
    ()[s0] -> (s0),
    domain: s0 in [0, 63]
  )",
                        ShapeUtil::MakeShape(F32, {64})));
  EXPECT_FALSE(ReadsOnce(R"(
    ()[s0] -> (s0 floorDiv 2),
    domain: s0 in [0, 63]
  )",
                         ShapeUtil::MakeShape(F32, {32})));
}

// Maps from (th_x, th_y, th_z, bl_x, bl_y, bl_z) to the element index.
class ReadsLinesInPiecesTest : public testing::Test {
 protected:
  ReadsLinesInPiecesTest() { RegisterSymbolicExprStorage(&mlir_context_); }

  bool ReadsInPieces(absl::string_view map, int64_t element_bytes,
                     int64_t buffer_bytes) {
    std::optional<IndexingMap> indexing_map =
        ParseIndexingMap(map, &mlir_context_);
    EXPECT_TRUE(indexing_map.has_value()) << map;
    return indexing_map.has_value() &&
           ReadsLinesInPieces(*indexing_map, /*wavefront_size=*/64,
                              element_bytes, buffer_bytes);
  }

  mlir::MLIRContext mlir_context_;
};

TEST_F(ReadsLinesInPiecesTest, VectorLoadsOfWholeLines) {
  // Four consecutive f32 per lane: one 16-byte load.
  EXPECT_FALSE(ReadsInPieces(R"(
    (d0, d1, d2, d3, d4, d5)[s0] -> (d3 * 512 + d0 * 4 + s0),
    domain:
    d0 in [0, 127], d1 in [0, 0], d2 in [0, 0],
    d3 in [0, 999], d4 in [0, 0], d5 in [0, 0],
    s0 in [0, 3]
  )",
                             /*element_bytes=*/4, /*buffer_bytes=*/2048000));
}

TEST_F(ReadsLinesInPiecesTest, UnrolledScalarLoadsOfWholeLines) {
  // Four unrolled loads, each 64 consecutive f32.
  EXPECT_FALSE(ReadsInPieces(R"(
    (d0, d1, d2, d3, d4, d5)[s0] -> (d3 * 512 + s0 * 128 + d0),
    domain:
    d0 in [0, 127], d1 in [0, 0], d2 in [0, 0],
    d3 in [0, 999], d4 in [0, 0], d5 in [0, 0],
    s0 in [0, 3]
  )",
                             /*element_bytes=*/4, /*buffer_bytes=*/2048000));
}

TEST_F(ReadsLinesInPiecesTest, PadShiftsRowsOffLines) {
  // Pad f32[R, 112] to [R, 113]: a line is read in two pieces.
  EXPECT_TRUE(ReadsInPieces(R"(
    (d0, d1, d2, d3, d4, d5) -> (((d3 * 128 + d0) floorDiv 113) * 112
                                 + (d3 * 128 + d0) mod 113),
    domain:
    d0 in [0, 127], d1 in [0, 0], d2 in [0, 0],
    d3 in [0, 999], d4 in [0, 0], d5 in [0, 0],
    (d3 * 128 + d0) mod 113 in [0, 111]
  )",
                            /*element_bytes=*/4, /*buffer_bytes=*/50176000));
}

TEST_F(ReadsLinesInPiecesTest, LeadingPadSkipsInactiveBlocks) {
  // Pad f32[R, 112] to [R + 8, 114]: the first three blocks are padding.
  EXPECT_TRUE(ReadsInPieces(R"(
    (d0, d1, d2, d3, d4, d5) -> (((d3 * 128 + d0) floorDiv 114 - 4) * 112
                                 + (d3 * 128 + d0) mod 114 - 1),
    domain:
    d0 in [0, 127], d1 in [0, 0], d2 in [0, 0],
    d3 in [0, 999], d4 in [0, 0], d5 in [0, 0],
    (d3 * 128 + d0) floorDiv 114 in [4, 1000],
    (d3 * 128 + d0) mod 114 in [1, 112]
  )",
                            /*element_bytes=*/4, /*buffer_bytes=*/50176000));
}

TEST_F(ReadsLinesInPiecesTest, ColumnReads) {
  // One f32 per row: 4 bytes of each line.
  EXPECT_TRUE(ReadsInPieces(R"(
    (d0, d1, d2, d3, d4, d5) -> (d0 * 1024 + d3),
    domain:
    d0 in [0, 63], d1 in [0, 0], d2 in [0, 0],
    d3 in [0, 1023], d4 in [0, 0], d5 in [0, 0]
  )",
                            /*element_bytes=*/4, /*buffer_bytes=*/262144));
}

TEST_F(ReadsLinesInPiecesTest, PartialLineAtTheEndOfTheBufferIsWhole) {
  // f32[100] is 400 bytes: the second block ends mid-line.
  EXPECT_FALSE(ReadsInPieces(R"(
    (d0, d1, d2, d3, d4, d5) -> (d3 * 64 + d0),
    domain:
    d0 in [0, 63], d1 in [0, 0], d2 in [0, 0],
    d3 in [0, 1], d4 in [0, 0], d5 in [0, 0],
    d3 * 64 + d0 in [0, 99]
  )",
                             /*element_bytes=*/4, /*buffer_bytes=*/400));
}

TEST_F(ReadsLinesInPiecesTest, NegativeOffsetsSplitLines) {
  // Elements [-16, 15] read half of two lines.
  EXPECT_TRUE(ReadsInPieces(R"(
    (d0, d1, d2, d3, d4, d5) -> (d0 - 16),
    domain:
    d0 in [0, 31], d1 in [0, 0], d2 in [0, 0],
    d3 in [0, 0], d4 in [0, 0], d5 in [0, 0]
  )",
                            /*element_bytes=*/4, /*buffer_bytes=*/1024));
}

TEST_F(ReadsLinesInPiecesTest, UndefinedMap) {
  EXPECT_FALSE(ReadsLinesInPieces(IndexingMap::GetUndefined(),
                                  /*wavefront_size=*/64, /*element_bytes=*/4,
                                  /*buffer_bytes=*/1024));
}

TEST_F(ReadsLinesInPiecesTest, LargeInnerLoopWithLeadingPadIsBounded) {
  // Leading pad keeps blocks 0..1023 inactive while inner range var s0 is 1e6.
  EXPECT_TRUE(ReadsInPieces(R"(
    (d0, d1, d2, d3, d4, d5)[s0] -> (
      (((d3 * 128 + d0) floorDiv 114 - 1500) * 112
       + (d3 * 128 + d0) mod 114 - 1) * 1000000 + s0),
    domain:
    d0 in [0, 127], d1 in [0, 0], d2 in [0, 0],
    d3 in [0, 2999], d4 in [0, 0], d5 in [0, 0],
    s0 in [0, 999999],
    (d3 * 128 + d0) floorDiv 114 in [1500, 3547],
    (d3 * 128 + d0) mod 114 in [1, 112]
  )",
                            /*element_bytes=*/4,
                            /*buffer_bytes=*/1000000000000LL));
}

}  // namespace
}  // namespace xla::gpu
