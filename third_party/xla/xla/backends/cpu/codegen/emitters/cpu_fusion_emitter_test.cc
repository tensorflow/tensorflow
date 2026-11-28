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

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/fusion_emitter.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/logical_buffer.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"

namespace xla {
namespace cpu {
namespace {

using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAreArray;

bool ScatterConstraintsSatisfied(const IndexingMap& map, int64_t thread_id,
                                 int64_t index_id, int64_t vector_id,
                                 int64_t vector_element_id) {
  mlir::MLIRContext* ctx = map.GetMLIRContext();
  return map.ConstraintsSatisfied(
      {CreateSymbolicConstant(thread_id, ctx)},
      {CreateSymbolicConstant(index_id, ctx),
       CreateSymbolicConstant(vector_id, ctx),
       CreateSymbolicConstant(vector_element_id, ctx)});
}

llvm::SmallVector<int64_t, 4> EvaluateScatterMap(const IndexingMap& map,
                                                 int64_t thread_id,
                                                 int64_t index_id,
                                                 int64_t vector_id,
                                                 int64_t vector_element_id) {
  mlir::MLIRContext* ctx = map.GetMLIRContext();
  return map.Evaluate({CreateSymbolicConstant(thread_id, ctx)},
                      {CreateSymbolicConstant(index_id, ctx),
                       CreateSymbolicConstant(vector_id, ctx),
                       CreateSymbolicConstant(vector_element_id, ctx)});
}

struct GetWorkDimensionsTestCase {
  std::string test_name;
  absl::string_view shape_and_layout;
  int64_t outer_dimension_partitions;
  uint64_t expected_num_work_groups;
  std::vector<int64_t> expected_work_tile_size;
  absl::string_view root_op = "copy(p0)";
};

class CpuFusionEmitterTest
    : public HloHardwareIndependentTestBase,
      public ::testing::WithParamInterface<GetWorkDimensionsTestCase> {
 protected:
  absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
      const HloModule& hlo) {
    return BufferAssigner::Run(
        &hlo, std::make_unique<DependencyHloOrdering>(&hlo),
        [](const BufferValue& buffer) {
          return CpuExecutable::ShapeSizeBytes(buffer.shape());
        },
        &alias_info_, [](LogicalBuffer::Color) { return /*alignment=*/1; },
        BufferAssigner::Options{});
  }

  AliasInfo alias_info_;
};

static constexpr absl::string_view kScatterHlo = R"(
  add {
    %lhs = f32[] parameter(0)
    %rhs = f32[] parameter(1)
    ROOT %add.2 = f32[] add(%lhs, %rhs)
  }

  scatter_computation {
    %operand = f32[50,64,8] parameter(0)
    %indices = s32[500,1]{1,0} parameter(1)
    %updates = f32[500,1,64,8] parameter(2)
    ROOT %scatter = f32[50,64,8] scatter(%operand, %indices, %updates),
      update_window_dims={1,2,3},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      to_apply=add
  }

  ENTRY main {
    %p = f32[50,64,8]{2,1,0} parameter(0)
    %p.1 = s32[500,1]{1,0} parameter(1)
    %p.2 = f32[500,1,64,8]{3,2,1,0} parameter(2)
    ROOT %wrapped_scatter = f32[50,64,8]{2,1,0} fusion(%p, %p.1, %p.2),
      kind=kLoop,
      calls=%scatter_computation
  }
)";

TEST_F(CpuFusionEmitterTest, ScatterMlir) {
  constexpr absl::string_view kExpected = R"(
    CHECK:       module @wrapped_scatter attributes {{{.*}}xla.extra_backend_options = #xla<extra_backend_options["xla_cpu_disable_loop_unrolling"]>{{.*}}}
    CHECK:       @wrapped_scatter(
    CHECK-SAME:    xla.entry
    CHECK:           %[[XLA_LOOP:.+]] = xla.loop
    CHECK:           xla.pure_call
    CHECK:           scf.if
    CHECK:             xla.pure_call
    CHECK:             tensor.extract
    CHECK:             arith.addf
    CHECK:           return %[[XLA_LOOP]]
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kScatterHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
  TF_ASSERT_OK_AND_ASSIGN(KernelDefinition kernel_definition,
                          emitter.EmitKernelDefinition());
  const auto& mlir_source = kernel_definition.source();
  auto mlir_dump = mlir_source.ToString();
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(mlir_dump, kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(CpuFusionEmitterTest, ScatterLlvm) {
  constexpr absl::string_view kExpected = R"(
    CHECK-NOT:  @wrapped_scatter_entry(
    CHECK-NOT:  @wrapped_scatter_kernel(
    CHECK:      @wrapped_scatter(
    CHECK:      uwtable "frame-pointer"="all"
    CHECK-SAME: "prefer-vector-width"="512"
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kScatterHlo));
  auto& debug_options = hlo_module->mutable_config().mutable_debug_options();
  debug_options.set_xla_cpu_prefer_vector_width(512);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
  TF_ASSERT_OK_AND_ASSIGN(KernelDefinition kernel_definition,
                          emitter.EmitKernelDefinition());
  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  TF_ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  auto llvm_dump = llvm_source.ToString();
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(llvm_dump, kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(CpuFusionEmitterTest,
       ScatterReorderedOperandsWritesToResultSliceAndHonorsNonInvariant) {
  constexpr absl::string_view kHlo = R"(
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    scatter_computation {
      indices = s32[2,1] parameter(0)
      updates = s32[2,1] parameter(1)
      operand = s32[4] parameter(2)
      ROOT s = s32[4] scatter(operand, indices, updates),
        update_window_dims={1},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add
    }

    ENTRY main {
      p0 = s32[2,1] parameter(0)
      p1 = s32[2,1] parameter(1)
      p2 = s32[4] parameter(2)
      ROOT wrapped_scatter = s32[4] fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation,
        frontend_attributes={xla.no_invariant_operands="0"}
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> kernel_definition,
                       emitter.EmitKernelDefinition());
  EXPECT_FALSE(kernel_definition.spec().invariant_arguments().contains(0));
  EXPECT_TRUE(kernel_definition.spec().invariant_arguments().contains(1));
  EXPECT_FALSE(kernel_definition.spec().invariant_arguments().contains(2));
  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  const llvm::Function* wrapped =
      llvm_source.module()->getFunction("wrapped_scatter_wrapped");
  ASSERT_NE(wrapped, nullptr);
  // Operand 0 is marked not invariant, operand 1 is read only, operand 2 and
  // arg 3 share the writable result slice, and args 4..6 are the workgroup IDs.
  ASSERT_EQ(wrapped->arg_size(), 7);
  EXPECT_FALSE(wrapped->getArg(0)->onlyReadsMemory());
  EXPECT_FALSE(wrapped->getArg(0)->hasNoAliasAttr());
  EXPECT_TRUE(wrapped->getArg(1)->onlyReadsMemory());
  EXPECT_TRUE(wrapped->getArg(1)->hasNoAliasAttr());
  EXPECT_FALSE(wrapped->getArg(2)->onlyReadsMemory());
  EXPECT_TRUE(wrapped->getArg(2)->hasNoAliasAttr());
  EXPECT_FALSE(wrapped->getArg(2)->use_empty());
  EXPECT_TRUE(wrapped->getArg(3)->use_empty());
}

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapBounds) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  struct Point {
    int64_t thread_id;
    int64_t index_id;
    int64_t vector_id;
    int64_t vector_element_id;
  };
  struct Case {
    absl::string_view name;
    std::vector<int64_t> updates_shape;
    int64_t num_threads;
    int64_t vector_size;
    Point last_valid;
    std::vector<int64_t> expected_last_valid_index;
    std::vector<Point> rejected;
    Interval expected_thread_bound;
    std::vector<Interval> expected_symbol_bounds;
    std::vector<Interval> expected_constraints;
    std::vector<Interval> expected_result_ranges;
  };

  const Case cases[] = {
      {"exact_divisibility",
       {6, 4},
       /*num_threads=*/3,
       /*vector_size=*/4,
       /*last_valid=*/{2, 1, 0, 3},
       /*expected_last_valid_index=*/{5, 3},
       /*rejected=*/{},
       /*expected_thread_bound=*/{0, 2},
       /*expected_symbol_bounds=*/{{0, 1}, {0, 0}, {0, 3}},
       /*expected_constraints=*/{},
       /*expected_result_ranges=*/{{0, 5}, {0, 3}}},
      {"partial_tail_workgroup",
       {5, 4},
       /*num_threads=*/3,
       /*vector_size=*/4,
       /*last_valid=*/{2, 0, 0, 3},
       /*expected_last_valid_index=*/{4, 3},
       /*rejected=*/{{2, 1, 0, 0}},
       /*expected_thread_bound=*/{0, 2},
       /*expected_symbol_bounds=*/{{0, 1}, {0, 0}, {0, 3}},
       /*expected_constraints=*/{{0, 4}},
       /*expected_result_ranges=*/{{0, 4}, {0, 3}}},
      {"partial_tail_two_threads",
       {5, 4},
       /*num_threads=*/2,
       /*vector_size=*/4,
       /*last_valid=*/{1, 1, 0, 3},
       /*expected_last_valid_index=*/{4, 3},
       /*rejected=*/{{1, 2, 0, 0}},
       /*expected_thread_bound=*/{0, 1},
       /*expected_symbol_bounds=*/{{0, 2}, {0, 0}, {0, 3}},
       /*expected_constraints=*/{{0, 4}},
       /*expected_result_ranges=*/{{0, 4}, {0, 3}}},
      {"full_plus_idle_workgroup",
       {4, 4},
       /*num_threads=*/3,
       /*vector_size=*/4,
       /*last_valid=*/{1, 1, 0, 3},
       /*expected_last_valid_index=*/{3, 3},
       /*rejected=*/{{2, 0, 0, 0}, {2, 1, 0, 0}},
       /*expected_thread_bound=*/{0, 1},
       /*expected_symbol_bounds=*/{{0, 1}, {0, 0}, {0, 3}},
       /*expected_constraints=*/{},
       /*expected_result_ranges=*/{{0, 3}, {0, 3}}},
      {"full_partial_and_idle_workgroups",
       {5, 4},
       /*num_threads=*/4,
       /*vector_size=*/4,
       /*last_valid=*/{2, 0, 0, 3},
       /*expected_last_valid_index=*/{4, 3},
       /*rejected=*/{{2, 1, 0, 0}, {3, 0, 0, 0}},
       /*expected_thread_bound=*/{0, 3},
       /*expected_symbol_bounds=*/{{0, 1}, {0, 0}, {0, 3}},
       /*expected_constraints=*/{{0, 4}},
       /*expected_result_ranges=*/{{0, 4}, {0, 3}}},
      {"more_threads_than_updates",
       {3, 4},
       /*num_threads=*/8,
       /*vector_size=*/4,
       /*last_valid=*/{2, 0, 0, 3},
       /*expected_last_valid_index=*/{2, 3},
       /*rejected=*/{{3, 0, 0, 0}, {7, 0, 0, 0}},
       /*expected_thread_bound=*/{0, 2},
       /*expected_symbol_bounds=*/{{0, 0}, {0, 0}, {0, 3}},
       /*expected_constraints=*/{},
       /*expected_result_ranges=*/{{0, 2}, {0, 3}}},
      {"non_divisible_vector_size",
       {5, 3, 2},
       /*num_threads=*/3,
       /*vector_size=*/4,
       /*last_valid=*/{2, 0, 1, 1},
       /*expected_last_valid_index=*/{4, 2, 1},
       /*rejected=*/{{0, 0, 1, 2}, {0, 0, 1, 3}, {2, 1, 0, 0}},
       /*expected_thread_bound=*/{0, 2},
       /*expected_symbol_bounds=*/{{0, 1}, {0, 1}, {0, 3}},
       /*expected_constraints=*/{{0, 4}, {0, 5}},
       /*expected_result_ranges=*/{{0, 4}, {0, 3}, {0, 1}}},
      {"vector_wider_than_slice",
       {5, 3},
       /*num_threads=*/3,
       /*vector_size=*/4,
       /*last_valid=*/{2, 0, 0, 2},
       /*expected_last_valid_index=*/{4, 2},
       /*rejected=*/{{0, 0, 0, 3}, {2, 1, 0, 0}},
       /*expected_thread_bound=*/{0, 2},
       /*expected_symbol_bounds=*/{{0, 1}, {0, 0}, {0, 2}},
       /*expected_constraints=*/{{0, 4}},
       /*expected_result_ranges=*/{{0, 4}, {0, 2}}},
      {"rank_zero_scalar_vec1",
       {5},
       /*num_threads=*/3,
       /*vector_size=*/1,
       /*last_valid=*/{2, 0, 0, 0},
       /*expected_last_valid_index=*/{4},
       /*rejected=*/{{2, 1, 0, 0}},
       /*expected_thread_bound=*/{0, 2},
       /*expected_symbol_bounds=*/{{0, 1}, {0, 0}, {0, 0}},
       /*expected_constraints=*/{{0, 4}},
       /*expected_result_ranges=*/{{0, 4}}},
      {"rank_zero_scalar_vec4",
       {5},
       /*num_threads=*/3,
       /*vector_size=*/4,
       /*last_valid=*/{2, 0, 0, 0},
       /*expected_last_valid_index=*/{4},
       /*rejected=*/{{0, 0, 0, 1}, {2, 1, 0, 0}},
       /*expected_thread_bound=*/{0, 2},
       /*expected_symbol_bounds=*/{{0, 1}, {0, 0}, {0, 0}},
       /*expected_constraints=*/{{0, 4}},
       /*expected_result_ranges=*/{{0, 4}}},
  };

  for (const Case& tc : cases) {
    SCOPED_TRACE(tc.name);
    IndexingMap map = GetScatterIndexingMap(tc.updates_shape, tc.num_threads,
                                            tc.vector_size, ctx.get());
    EXPECT_FALSE(map.IsKnownEmpty());
    EXPECT_THAT(EvaluateScatterMap(
                    map, tc.last_valid.thread_id, tc.last_valid.index_id,
                    tc.last_valid.vector_id, tc.last_valid.vector_element_id),
                ElementsAreArray(tc.expected_last_valid_index));
    EXPECT_TRUE(ScatterConstraintsSatisfied(
        map, tc.last_valid.thread_id, tc.last_valid.index_id,
        tc.last_valid.vector_id, tc.last_valid.vector_element_id));
    for (const Point& p : tc.rejected) {
      EXPECT_FALSE(ScatterConstraintsSatisfied(
          map, p.thread_id, p.index_id, p.vector_id, p.vector_element_id));
    }

    map.Simplify();
    EXPECT_FALSE(map.IsKnownEmpty());
    EXPECT_EQ(map.GetDimensionBound(0), tc.expected_thread_bound);
    EXPECT_THAT(map.GetSymbolBounds(),
                ElementsAreArray(tc.expected_symbol_bounds));
    std::vector<Interval> constraints;
    for (const auto& [expr, interval] : map.GetSymbolicConstraints()) {
      constraints.push_back(interval);
    }
    EXPECT_THAT(constraints,
                UnorderedElementsAreArray(tc.expected_constraints));
    EXPECT_THAT(map.ComputeResultRanges(),
                ElementsAreArray(tc.expected_result_ranges));
  }
}

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapEmptyShapes) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  struct EmptyCase {
    absl::string_view name;
    std::vector<int64_t> updates_shape;
    int64_t num_threads;
    int64_t vector_size;
  };
  for (const EmptyCase& tc : std::initializer_list<EmptyCase>{
           {"zero_updates_single_thread", {0, 4, 8}, 1, 4},
           {"zero_updates_multi_thread", {0, 4, 8}, 4, 4},
           {"zero_updates_rank_zero", {0}, 2, 1},
           {"zero_slice_single_thread", {5, 0, 8}, 1, 1},
           {"zero_slice_multi_thread", {5, 4, 0}, 3, 4},
           {"zero_updates_and_slice", {0, 0}, 2, 1}}) {
    SCOPED_TRACE(tc.name);
    IndexingMap map = GetScatterIndexingMap(tc.updates_shape, tc.num_threads,
                                            tc.vector_size, ctx.get());
    EXPECT_TRUE(map.IsKnownEmpty());
    EXPECT_FALSE(ScatterConstraintsSatisfied(map, 0, 0, 0, 0));
    map.Simplify();
    EXPECT_TRUE(map.IsKnownEmpty());
    EXPECT_THAT(map.GetSymbolicConstraints(), IsEmpty());
  }
}

TEST_F(CpuFusionEmitterTest, ScatterEmptyShapesShortCircuit) {
  constexpr absl::string_view kZeroUpdatesHlo = R"hlo(
    HloModule scatter_zero_updates
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = f32[2,2]{1,0} parameter(0)
      indices = s32[0,1]{1,0} parameter(1)
      updates = f32[0,1,2]{2,1,0} parameter(2)
      ROOT s = f32[2,2]{1,0} scatter(operand, indices, updates),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = f32[2,2]{1,0} parameter(0)
      p1 = s32[0,1]{1,0} parameter(1)
      p2 = f32[0,1,2]{2,1,0} parameter(2)
      ROOT wrapped_scatter = f32[2,2]{1,0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  constexpr absl::string_view kZeroSliceHlo = R"hlo(
    HloModule scatter_zero_slice
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = f32[2,2]{1,0} parameter(0)
      indices = s32[2,2]{1,0} parameter(1)
      updates = f32[2,1,0]{2,1,0} parameter(2)
      ROOT s = f32[2,2]{1,0} scatter(operand, indices, updates),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0,1},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = f32[2,2]{1,0} parameter(0)
      p1 = s32[2,2]{1,0} parameter(1)
      p2 = f32[2,1,0]{2,1,0} parameter(2)
      ROOT wrapped_scatter = f32[2,2]{1,0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  for (const auto& [name, hlo] :
       {std::pair<absl::string_view, absl::string_view>{"zero_updates",
                                                        kZeroUpdatesHlo},
        std::pair<absl::string_view, absl::string_view>{"zero_slice",
                                                        kZeroSliceHlo}}) {
    SCOPED_TRACE(name);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                         RunBufferAssignment(*module));
    const auto* fusion = Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
    std::unique_ptr<mlir::MLIRContext> mlir_context =
        FusionCompiler::CreateContext();
    CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
    ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> kernel_definition,
                         emitter.EmitKernelDefinition());
    EXPECT_THAT(RunFileCheck(kernel_definition.source().ToString(), R"(
      CHECK:     func.func @wrapped_scatter(
      CHECK-NOT: xla.loop
      CHECK:     return
    )"),
                absl_testing::IsOkAndHolds(true));
  }
}

TEST_F(CpuFusionEmitterTest, ScatterRankZeroOperandEmitsLoop) {
  constexpr absl::string_view kMultiUpdateScalarHlo = R"hlo(
    HloModule scatter_rank0_multi
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = s32[] parameter(0)
      indices = s32[3,0]{1,0} parameter(1)
      updates = s32[3]{0} parameter(2)
      ROOT s = s32[] scatter(operand, indices, updates),
        update_window_dims={},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = s32[] parameter(0)
      p1 = s32[3,0]{1,0} parameter(1)
      p2 = s32[3]{0} parameter(2)
      ROOT wrapped_scatter = s32[] fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kMultiUpdateScalarHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*module));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_THAT(RunFileCheck(kernel_definition.source().ToString(), R"(
    CHECK: func.func @wrapped_scatter(
    CHECK:   xla.loop
    CHECK:     tensor.extract %{{.+}}[] : tensor<i32>
    CHECK:     arith.addi
    CHECK:     tensor.insert %{{.+}} into %{{.+}}[] : tensor<i32>
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, ScatterVectorSizeUsesWidestOperandOrUpdateType) {
  // Operand 0 is s8 (1 byte) and operand 1 is c128 (16 bytes). With a 64 byte
  // vectorization budget, max_vectorized_elements must be 64 / 16 = 4 (giving
  // vector_id in [0, 3], vector_element_id in [0, 3] for slice size 16), not 16
  // from s8 alone.
  constexpr absl::string_view kVariadicHlo = R"hlo(
    HloModule variadic_scatter_vector_size
    reducer {
      lhs0 = s8[] parameter(0)
      lhs1 = c128[] parameter(1)
      rhs0 = s8[] parameter(2)
      rhs1 = c128[] parameter(3)
      sum0 = s8[] add(lhs0, rhs0)
      sum1 = c128[] add(lhs1, rhs1)
      ROOT out = (s8[], c128[]) tuple(sum0, sum1)
    }
    scatter_computation {
      op0 = s8[4,16]{1,0} parameter(0)
      op1 = c128[4,16]{1,0} parameter(1)
      indices = s32[2,1]{1,0} parameter(2)
      upd0 = s8[2,1,16]{2,1,0} parameter(3)
      upd1 = c128[2,1,16]{2,1,0} parameter(4)
      ROOT s = (s8[4,16]{1,0}, c128[4,16]{1,0}) scatter(
          op0, op1, indices, upd0, upd1),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=reducer
    }
    ENTRY main {
      p0 = s8[4,16]{1,0} parameter(0)
      p1 = c128[4,16]{1,0} parameter(1)
      p2 = s32[2,1]{1,0} parameter(2)
      p3 = s8[2,1,16]{2,1,0} parameter(3)
      p4 = c128[2,1,16]{2,1,0} parameter(4)
      ROOT wrapped_scatter = (s8[4,16]{1,0}, c128[4,16]{1,0}) fusion(
          p0, p1, p2, p3, p4),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  // Mixed floating point precision where updates are wider than the operand:
  // operand is bf16 (2 bytes) and updates are f64 (8 bytes), so
  // max_vectorized_elements must be 64 / 8 = 8 (vector_id in [0, 1],
  // vector_element_id in [0, 7]), not 16 from bf16 alone.
  constexpr absl::string_view kWiderUpdateHlo = R"hlo(
    HloModule wider_update_scatter_vector_size
    reducer {
      lhs = bf16[] parameter(0)
      rhs = f64[] parameter(1)
      lhs_f64 = f64[] convert(lhs)
      sum = f64[] add(lhs_f64, rhs)
      ROOT out = bf16[] convert(sum)
    }
    scatter_computation {
      op = bf16[4,16]{1,0} parameter(0)
      indices = s32[2,1]{1,0} parameter(1)
      upd = f64[2,1,16]{2,1,0} parameter(2)
      ROOT s = bf16[4,16]{1,0} scatter(op, indices, upd),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=reducer
    }
    ENTRY main {
      p0 = bf16[4,16]{1,0} parameter(0)
      p1 = s32[2,1]{1,0} parameter(1)
      p2 = f64[2,1,16]{2,1,0} parameter(2)
      ROOT wrapped_scatter = bf16[4,16]{1,0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  struct VectorSizeCase {
    absl::string_view name;
    absl::string_view hlo;
    absl::string_view expected_domain;
  };
  for (const VectorSizeCase& tc : std::initializer_list<VectorSizeCase>{
           {"variadic_s8_c128", kVariadicHlo,
            "CHECK: domain: thread_id in [0, 0], index_id in [0, 1], "
            "vector_id in [0, 3], vector_element_id in [0, 3]"},
           {"wider_update_bf16_f64", kWiderUpdateHlo,
            "CHECK: domain: thread_id in [0, 0], index_id in [0, 1], "
            "vector_id in [0, 1], vector_element_id in [0, 7]"}}) {
    SCOPED_TRACE(tc.name);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(tc.hlo));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                         RunBufferAssignment(*module));
    const auto* fusion = Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
    std::unique_ptr<mlir::MLIRContext> mlir_context =
        FusionCompiler::CreateContext();
    CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
    ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> kernel_definition,
                         emitter.EmitKernelDefinition());
    EXPECT_THAT(
        RunFileCheck(kernel_definition.source().ToString(), tc.expected_domain),
        absl_testing::IsOkAndHolds(true));
  }
}

TEST_F(CpuFusionEmitterTest, ScatterEmitterRejectsUnsupportedScatters) {
  constexpr absl::string_view kNonSimplified1DHlo = R"hlo(
    HloModule non_simplified_1d
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = s32[4]{0} parameter(0)
      indices = s32[2,1]{1,0} parameter(1)
      updates = s32[2]{0} parameter(2)
      ROOT s = s32[4]{0} scatter(operand, indices, updates),
        update_window_dims={},
        inserted_window_dims={0},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = s32[4]{0} parameter(0)
      p1 = s32[2,1]{1,0} parameter(1)
      p2 = s32[2]{0} parameter(2)
      ROOT wrapped_scatter = s32[4]{0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  constexpr absl::string_view kBatchingDimsHlo = R"hlo(
    HloModule batching_dims_scatter
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = s32[2,4]{1,0} parameter(0)
      indices = s32[2,1]{1,0} parameter(1)
      updates = s32[2,1]{1,0} parameter(2)
      ROOT s = s32[2,4]{1,0} scatter(operand, indices, updates),
        update_window_dims={1},
        inserted_window_dims={},
        input_batching_dims={0},
        scatter_indices_batching_dims={0},
        scatter_dims_to_operand_dims={1},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = s32[2,4]{1,0} parameter(0)
      p1 = s32[2,1]{1,0} parameter(1)
      p2 = s32[2,1]{1,0} parameter(2)
      ROOT wrapped_scatter = s32[2,4]{1,0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  constexpr absl::string_view kMismatchedVariadicDimsHlo = R"hlo(
    HloModule mismatched_variadic_dims
    reducer {
      lhs0 = s32[] parameter(0)
      lhs1 = s32[] parameter(1)
      rhs0 = s32[] parameter(2)
      rhs1 = s32[] parameter(3)
      sum0 = s32[] add(lhs0, rhs0)
      sum1 = s32[] add(lhs1, rhs1)
      ROOT out = (s32[], s32[]) tuple(sum0, sum1)
    }
    scatter_computation {
      op0 = s32[4]{0} parameter(0)
      op1 = s32[8]{0} parameter(1)
      indices = s32[2,1]{1,0} parameter(2)
      upd0 = s32[2,1]{1,0} parameter(3)
      upd1 = s32[2,1]{1,0} parameter(4)
      ROOT s = (s32[4]{0}, s32[8]{0}) scatter(op0, op1, indices, upd0, upd1),
        update_window_dims={1},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=reducer
    }
    ENTRY main {
      p0 = s32[4]{0} parameter(0)
      p1 = s32[8]{0} parameter(1)
      p2 = s32[2,1]{1,0} parameter(2)
      p3 = s32[2,1]{1,0} parameter(3)
      p4 = s32[2,1]{1,0} parameter(4)
      ROOT wrapped_scatter = (s32[4]{0}, s32[8]{0}) fusion(p0, p1, p2, p3, p4),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  struct RejectCase {
    absl::string_view name;
    absl::string_view hlo;
    absl::string_view expected_error_substr;
  };
  for (const RejectCase& tc : std::initializer_list<RejectCase>{
           {"non_simplified_1d", kNonSimplified1DHlo,
            "Non-simplified HLO Scatter is not supported."},
           {"batching_dims", kBatchingDimsHlo,
            "Non-simplified HLO Scatter is not supported."},
           {"mismatched_variadic_dims", kMismatchedVariadicDimsHlo,
            "Variadic scatter operands must have matching dimensions."}}) {
    SCOPED_TRACE(tc.name);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(tc.hlo));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                         RunBufferAssignment(*module));
    const auto* fusion = Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
    std::unique_ptr<mlir::MLIRContext> mlir_context =
        FusionCompiler::CreateContext();
    CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
    EXPECT_THAT(emitter.EmitKernelDefinition(),
                absl_testing::StatusIs(absl::StatusCode::kInternal,
                                       HasSubstr(tc.expected_error_substr)));
  }
}

TEST_F(CpuFusionEmitterTest, SortComparatorMlirAndLlvm) {
  constexpr absl::string_view kSortHlo = R"(
    HloModule sort_module
    compare {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT lt = pred[] compare(lhs, rhs), direction=LT
    }
    ENTRY main {
      p0 = f32[8] parameter(0)
      ROOT sorted = f32[8] sort(p0), dimensions={0}, to_apply=compare
    }
  )";
  constexpr absl::string_view kExpectedMlir = R"(
    CHECK:      func.func @compare(
    CHECK-SAME:   %[[ARG0:.+]]: tensor<f32>, %[[ARG1:.+]]: tensor<f32>) -> i1
    CHECK-SAME:   xla.cpu.is_comparator
    CHECK-NOT:    xla.entry
    CHECK:        %[[LHS:.+]] = tensor.extract %[[ARG0]][] : tensor<f32>
    CHECK:        %[[RHS:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
    CHECK:        %[[CMP:.+]] = xla.pure_call @compare_lt(%[[LHS]], %[[RHS]]) : (f32, f32) -> i8
    CHECK:        %[[RES:.+]] = arith.trunci %[[CMP]] : i8 to i1
    CHECK:        return %[[RES]] : i1
  )";
  constexpr absl::string_view kExpectedLlvm = R"(
    CHECK:      define noundef zeroext i1 @compare(ptr noundef nonnull %{{.+}}) #[[ATTR:[0-9]+]]
    CHECK:      attributes #[[ATTR]] = {{{.*}}uwtable{{[ }]}}
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kSortHlo));
  const HloComputation* comparator =
      hlo_module->entry_computation()->root_instruction()->to_apply();
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource mlir_source,
      EmitSortComparator(*mlir_context, *comparator, "compare"));
  EXPECT_THAT(RunFileCheck(mlir_source.ToString(), kExpectedMlir),
              absl_testing::IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(mlir_source)));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), kExpectedLlvm),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, SortComparatorWeakAndTotalOrder) {
  constexpr absl::string_view kWeakSortHlo = R"hlo(
    HloModule sort_weak_module
    compare {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT lt = pred[] compare(lhs, rhs), direction=LT, order=WEAK
    }
    ENTRY main {
      p0 = bf16[8] parameter(0)
      ROOT sorted = bf16[8] sort(p0), dimensions={0}, to_apply=compare
    }
  )hlo";
  constexpr absl::string_view kExpectedWeakMlir = R"(
    CHECK:      func.func private @compare_lt(
    CHECK-DAG:    arith.cmpf ult
    CHECK-DAG:    arith.cmpf ord
    CHECK:        arith.andi
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> weak_module,
                       ParseAndReturnVerifiedModule(kWeakSortHlo));
  const HloComputation* weak_comparator =
      weak_module->entry_computation()->root_instruction()->to_apply();
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource weak_mlir,
      EmitSortComparator(*mlir_context, *weak_comparator, "compare_weak"));
  EXPECT_THAT(RunFileCheck(weak_mlir.ToString(), kExpectedWeakMlir),
              absl_testing::IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource weak_llvm,
                       compiler.Compile(std::move(weak_mlir)));
  EXPECT_THAT(RunFileCheck(weak_llvm.ToString(),
                           "CHECK: define noundef zeroext i1 @compare_weak("),
              absl_testing::IsOkAndHolds(true));

  constexpr absl::string_view kTotalSortHlo = R"hlo(
    HloModule sort_total_module
    compare {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT lt = pred[] compare(lhs, rhs), direction=LT, order=TOTAL
    }
    ENTRY main {
      p0 = bf16[8] parameter(0)
      ROOT sorted = bf16[8] sort(p0), dimensions={0}, to_apply=compare
    }
  )hlo";
  constexpr absl::string_view kExpectedTotalMlir = R"(
    CHECK:      func.func private @compare_lt(
    CHECK:        arith.bitcast %{{.+}} : bf16 to i16
    CHECK:        arith.cmpi slt
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> total_module,
                       ParseAndReturnVerifiedModule(kTotalSortHlo));
  const HloComputation* total_comparator =
      total_module->entry_computation()->root_instruction()->to_apply();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource total_mlir,
      EmitSortComparator(*mlir_context, *total_comparator, "compare_total"));
  EXPECT_THAT(RunFileCheck(total_mlir.ToString(), kExpectedTotalMlir),
              absl_testing::IsOkAndHolds(true));
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource total_llvm,
                       compiler.Compile(std::move(total_mlir)));
  EXPECT_THAT(RunFileCheck(total_llvm.ToString(),
                           "CHECK: define noundef zeroext i1 @compare_total("),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, SortComparatorUnsupportedOp) {
  constexpr absl::string_view kSortHlo = R"(
    HloModule sort_module
    cond {
      state = s32[] parameter(0)
      limit = s32[] constant(4)
      ROOT lt = pred[] compare(state, limit), direction=LT
    }
    body {
      state = s32[] parameter(0)
      one = s32[] constant(1)
      ROOT next = s32[] add(state, one)
    }
    compare {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      loop = s32[] while(lhs), condition=cond, body=body
      ROOT lt = pred[] compare(loop, rhs), direction=LT
    }
    ENTRY main {
      p0 = s32[8] parameter(0)
      ROOT sorted = s32[8] sort(p0), dimensions={0}, to_apply=compare
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kSortHlo));
  const HloComputation* comparator =
      hlo_module->entry_computation()->root_instruction()->to_apply();
  auto mlir_context = FusionCompiler::CreateContext();
  EXPECT_THAT(EmitSortComparator(*mlir_context, *comparator, "compare"),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_P(CpuFusionEmitterTest, WorkDimensionsTest) {
  const GetWorkDimensionsTestCase& tc = GetParam();
  constexpr absl::string_view kHloTemplate = R"(
    HloModule test
    fused_computation {
      p0 = $0 parameter(0)
      ROOT res = $0 $2
    }
    ENTRY main {
      p0 = $0 parameter(0)
      ROOT fusion = $0 fusion(p0), kind=kLoop,
        calls=fused_computation,
        backend_config={"outer_dimension_partitions":["$1"]}
    }
  )";
  std::string hlo = absl::Substitute(kHloTemplate, tc.shape_and_layout,
                                     tc.outer_dimension_partitions, tc.root_op);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*module));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  const WorkDimensions& dims = kernel_definition.spec().work_dimensions();
  EXPECT_EQ(dims.num_work_groups.x, tc.expected_num_work_groups);
  EXPECT_THAT(dims.work_tile_size.dimensions,
              ElementsAreArray(tc.expected_work_tile_size));
}

INSTANTIATE_TEST_SUITE_P(
    WorkDimensions, CpuFusionEmitterTest,
    ::testing::ValuesIn<GetWorkDimensionsTestCase>({
        {
            /*test_name=*/"NonSubByteDoesNotRoundOddTile",
            /*shape_and_layout=*/"f32[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{3, 3},
        },
        {
            // CeilOfRatio(6, 2) = 3 is rounded up to 4 (multiple of 2
            // elements/byte), giving 4 * 3 = 12 elements (6 whole bytes) per
            // workgroup.
            /*test_name=*/"SubByteS4RoundsOddSplitTileToMultipleOfTwo",
            /*shape_and_layout=*/"s4[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // CeilOfRatio(5, 2) = 3 is also rounded up to 4, so workgroup 0
            // writes 12 elements (6 whole bytes) and workgroup 1 writes the
            // remaining 3 elements starting at byte 6.
            /*test_name=*/
            "SubByteS4OddRowsStillRoundsSplitTileToMultipleOfTwo",
            /*shape_and_layout=*/"s4[5,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // CeilOfRatio(6, 2) = 3 is rounded up to 4 (multiple of 4
            // elements/byte), giving 4 * 3 = 12 elements (3 whole bytes) per
            // workgroup.
            /*test_name=*/"SubByteS2RoundsSplitTileToMultipleOfFour",
            /*shape_and_layout=*/"s2[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // CeilOfRatio(12, 2) = 6 is rounded up to 8 (multiple of 8
            // elements/byte), giving 8 * 3 = 24 elements (3 whole bytes) per
            // workgroup.
            /*test_name=*/"SubByteS1RoundsSplitTileToMultipleOfEight",
            /*shape_and_layout=*/"s1[12,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{8, 3},
        },
        {
            // PRED is stored 1 element per byte, so odd split tiles are
            // not rounded.
            /*test_name=*/"PredIsUnpackedAndDoesNotRoundOddTile",
            /*shape_and_layout=*/"pred[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{3, 3},
        },
        {
            // CeilOfRatio(6, 4) = 2 (already a multiple of 2 elems/byte);
            // 3 workgroups of size 2 cover all 6 rows, so num_work_groups is
            // shrunk from 4 to 3.
            /*test_name=*/"SubByteRecomputesNumWorkGroupsAfterRounding",
            /*shape_and_layout=*/"s4[6,3]{1,0}",
            /*outer_dimension_partitions=*/4,
            /*expected_num_work_groups=*/3,
            /*expected_work_tile_size=*/{2, 3},
        },
        {
            // Concatenate fusions with packed sub-byte outputs must run in a
            // single workgroup covering the entire shape, because concatenated
            // operands can start at non-byte-aligned offsets.
            /*test_name=*/"ConcatenateSubByteForcesSingleWorkGroup",
            /*shape_and_layout=*/"s4[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/1,
            /*expected_work_tile_size=*/{6, 3},
            /*root_op=*/"concatenate(p0), dimensions={0}",
        },
        {
            // DynamicUpdateSlice fusions with packed sub-byte outputs must also
            // run in a single workgroup covering the entire shape, because the
            // update index may land at a non-byte aligned offset.
            /*test_name=*/"DynamicUpdateSliceSubByteForcesSingleWorkGroup",
            /*shape_and_layout=*/"s4[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/1,
            /*expected_work_tile_size=*/{6, 3},
            /*root_op=*/
            "dynamic-update-slice(p0, p0, s32[] constant(0), s32[] "
            "constant(0))",
        },
        {
            // 3D shape where the leading dim (2) is smaller than 4 workgroups:
            // dim 0 (2) is folded with dim 1 (6) into 12 slices of 3 elements;
            // CeilOfRatio(12, 4) = 3 rounds up to 4, leaving 3 workgroups of
            // tile {4, 3}.
            /*test_name=*/"ThreeDimsFoldsLeadingDimAndRoundsSplitTile",
            /*shape_and_layout=*/"s4[2,6,3]{2,1,0}",
            /*outer_dimension_partitions=*/4,
            /*expected_num_work_groups=*/3,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // 3D shape where the leading dim (6) is large enough to split
            // directly: CeilOfRatio(6, 2) = 3 rounds up to 4, keeping both
            // minor dims (4 and 3) intact.
            /*test_name=*/"ThreeDimsSplitsOutermostDimAndKeepsMinorDims",
            /*shape_and_layout=*/"s4[6,4,3]{2,1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 4, 3},
        },
        {
            // More workgroups (8) than the product of the first two dimensions
            // (2 * 2 = 4): all three dimensions fold together (2 * 2 * 3 = 12),
            // and CeilOfRatio(12, 8) = 2 yields 6 workgroups of 1D tile {2}.
            /*test_name=*/"FoldsAllDimensionsIntoSingleTileDim",
            /*shape_and_layout=*/"s4[2,2,3]{2,1,0}",
            /*outer_dimension_partitions=*/8,
            /*expected_num_work_groups=*/6,
            /*expected_work_tile_size=*/{2},
        },
        {
            // As many workgroups (12) as total elements (2 * 2 * 3 = 12): all
            // three dimensions fold together, and CeilOfRatio(12, 12) = 1 is
            // rounded up to 2 elements (1 byte) per workgroup, shrinking
            // num_work_groups from 12 to 6.
            /*test_name=*/"OneWorkGroupPerElementRoundsToWholeByte",
            /*shape_and_layout=*/"s4[2,2,3]{2,1,0}",
            /*outer_dimension_partitions=*/12,
            /*expected_num_work_groups=*/6,
            /*expected_work_tile_size=*/{2},
        },
        {
            // Column-major layout {0,1}: dim 1 (size 6) is the most-major
            // physical dimension and gets split and rounded from 3 to 4, while
            // dim 0 (size 3) is minor.
            /*test_name=*/"SubByteColumnMajorLayoutSplitsPhysicalMajorDim",
            /*shape_and_layout=*/"s4[3,6]{0,1}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // Single workgroup does not round odd dimensions.
            /*test_name=*/"SingleWorkGroupDoesNotRoundOddSubByteDims",
            /*shape_and_layout=*/"s4[5,3]{1,0}",
            /*outer_dimension_partitions=*/1,
            /*expected_num_work_groups=*/1,
            /*expected_work_tile_size=*/{5, 3},
        },
        {
            // f6 types are stored one element per byte and are not rounded.
            /*test_name=*/"NonPowerOfTwoSubByteF6DoesNotRoundOddTile",
            /*shape_and_layout=*/"f6e3m2fn[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{3, 3},
        },
        {
            // More workgroups (16) than elements (12): clamped to 12, then
            // rounded to 2 elements (1 byte) per workgroup.
            /*test_name=*/"MoreWorkGroupsThanElementsSubByteRoundsToWholeByte",
            /*shape_and_layout=*/"s4[2,2,3]{2,1,0}",
            /*outer_dimension_partitions=*/16,
            /*expected_num_work_groups=*/6,
            /*expected_work_tile_size=*/{2},
        },
        {
            // More workgroups (8) than elements (6): clamped to 6.
            /*test_name=*/
            "MoreWorkGroupsThanElementsNonSubByteClampsToElements",
            /*shape_and_layout=*/"f32[2,3]{1,0}",
            /*outer_dimension_partitions=*/8,
            /*expected_num_work_groups=*/6,
            /*expected_work_tile_size=*/{1},
        },
        {
            // Zero-element shapes get a single workgroup and an empty tile.
            /*test_name=*/"ZeroElementsUsesSingleWorkGroup",
            /*shape_and_layout=*/"f32[0,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/1,
            /*expected_work_tile_size=*/{},
        },
    }),
    [](const ::testing::TestParamInfo<GetWorkDimensionsTestCase>& info) {
      return info.param.test_name;
    });

TEST_F(CpuFusionEmitterTest, ConcatenateMatchingLayoutEmitsMemcpy) {
  constexpr absl::string_view kHlo = R"(
    HloModule concat_module
    fused_computation {
      p0 = f32[4,2,8]{2,1,0} parameter(0)
      p1 = f32[4,3,8]{2,1,0} parameter(1)
      ROOT concat = f32[4,5,8]{2,1,0} concatenate(p0, p1), dimensions={1}
    }
    ENTRY main {
      p0 = f32[4,2,8]{2,1,0} parameter(0)
      p1 = f32[4,3,8]{2,1,0} parameter(1)
      ROOT wrapped_concat = f32[4,5,8]{2,1,0} fusion(p0, p1),
        kind=kLoop, calls=fused_computation,
        backend_config={"outer_dimension_partitions":["2"]}
    }
  )";
  constexpr absl::string_view kExpectedMlir = R"(
    CHECK:      func.func @wrapped_concat(
    CHECK-SAME:   xla.entry
    CHECK-NOT:    xla.loop
    CHECK:        scf.for
    CHECK:          xla.copy_slice {{.*}} count(16)
    CHECK:          xla.copy_slice {{.*}} count(24)
  )";
  std::string expected_llvm = absl::Substitute(R"(
    CHECK:      define noundef ptr @wrapped_concat(
    CHECK-NOT:    call void @llvm.memcpy
    CHECK:        call void @llvm.assume(i1 true) [ "align"(ptr %{{.*}}, i64 $0) ]
    CHECK:      define internal void @wrapped_concat_wrapped(
    CHECK:        call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %{{.*}}, i64 64, i1 false)
    CHECK:        call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %{{.*}}, i64 96, i1 false)
  )",
                                               MinAlign());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_EQ(kernel_definition.spec().work_dimensions().num_work_groups.x, 2);
  EXPECT_THAT(
      RunFileCheck(kernel_definition.source().ToString(), kExpectedMlir),
      absl_testing::IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), expected_llvm),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest,
       ConcatenateNoOuterDimsForcesSingleWorkGroupAndEmitsMemcpy) {
  constexpr absl::string_view kHlo = R"(
    HloModule concat_module
    fused_computation {
      p0 = f32[8,16]{1,0} parameter(0)
      p1 = f32[12,16]{1,0} parameter(1)
      ROOT concat = f32[20,16]{1,0} concatenate(p0, p1), dimensions={0}
    }
    ENTRY main {
      p0 = f32[8,16]{1,0} parameter(0)
      p1 = f32[12,16]{1,0} parameter(1)
      ROOT wrapped_concat = f32[20,16]{1,0} fusion(p0, p1),
        kind=kLoop, calls=fused_computation,
        backend_config={"outer_dimension_partitions":["4"]}
    }
  )";
  std::string expected_llvm = absl::Substitute(R"(
    CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align $0 %{{.*}}, ptr align $0 %{{.*}}, i64 512, i1 false)
    CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align $0 %{{.*}}, ptr align $0 %{{.*}}, i64 768, i1 false)
  )",
                                               MinAlign());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_EQ(kernel_definition.spec().work_dimensions().num_work_groups.x, 1);

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), expected_llvm),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, ConcatenateMismatchedLayoutFallsBackToLoop) {
  constexpr absl::string_view kHlo = R"(
    HloModule concat_module
    fused_computation {
      p0 = f32[4,8]{0,1} parameter(0)
      p1 = f32[6,8]{1,0} parameter(1)
      ROOT concat = f32[10,8]{1,0} concatenate(p0, p1), dimensions={0}
    }
    ENTRY main {
      p0 = f32[4,8]{0,1} parameter(0)
      p1 = f32[6,8]{1,0} parameter(1)
      ROOT wrapped_concat = f32[10,8]{1,0} fusion(p0, p1),
        kind=kLoop, calls=fused_computation
    }
  )";
  constexpr absl::string_view kExpectedMlir = R"(
    CHECK-NOT: xla.copy_slice
    CHECK:     xla.loop
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_THAT(
      RunFileCheck(kernel_definition.source().ToString(), kExpectedMlir),
      absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest,
       ConcatenateNonDefaultMatchingLayoutUsesPhysicalOrderMemcpy) {
  constexpr absl::string_view kHlo = R"(
    HloModule concat_module
    fused_computation {
      p0 = f32[2,4,3]{0,2,1} parameter(0)
      p1 = f32[2,4,5]{0,2,1} parameter(1)
      ROOT concat = f32[2,4,8]{0,2,1} concatenate(p0, p1), dimensions={2}
    }
    ENTRY main {
      p0 = f32[2,4,3]{0,2,1} parameter(0)
      p1 = f32[2,4,5]{0,2,1} parameter(1)
      ROOT wrapped_concat = f32[2,4,8]{0,2,1} fusion(p0, p1),
        kind=kLoop, calls=fused_computation,
        backend_config={"outer_dimension_partitions":["2"]}
    }
  )";
  constexpr absl::string_view kExpectedMlir = R"(
    CHECK:      func.func @wrapped_concat(
    CHECK-NOT:    xla.loop
    CHECK:        scf.for
    CHECK:          xla.copy_slice {{.*}} count(6)
    CHECK:          xla.copy_slice {{.*}} count(10)
  )";
  constexpr absl::string_view kExpectedLlvm = R"(
    CHECK: call void @llvm.memcpy{{.*}}i64 24
    CHECK: call void @llvm.memcpy{{.*}}i64 40
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_EQ(kernel_definition.spec().work_dimensions().num_work_groups.x, 2);
  EXPECT_THAT(
      RunFileCheck(kernel_definition.source().ToString(), kExpectedMlir),
      absl_testing::IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), kExpectedLlvm),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, ConcatenateComplexAndNarrowFloatCompileToMemcpy) {
  for (absl::string_view elem_type : {"c64", "f6e3m2fn"}) {
    std::string hlo = absl::Substitute(R"(
      HloModule concat_module
      fused_computation {
        p0 = $0[4,3]{1,0} parameter(0)
        p1 = $0[4,5]{1,0} parameter(1)
        ROOT concat = $0[4,8]{1,0} concatenate(p0, p1), dimensions={1}
      }
      ENTRY main {
        p0 = $0[4,3]{1,0} parameter(0)
        p1 = $0[4,5]{1,0} parameter(1)
        ROOT wrapped_concat = $0[4,8]{1,0} fusion(p0, p1),
          kind=kLoop, calls=fused_computation
      }
    )",
                                       elem_type);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                         ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                         RunBufferAssignment(*hlo_module));
    const auto* fusion = Cast<HloFusionInstruction>(
        hlo_module->entry_computation()->root_instruction());
    auto mlir_context = FusionCompiler::CreateContext();
    ASSERT_OK_AND_ASSIGN(
        KernelDefinition<MlirKernelSource> kernel_definition,
        EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                         /*use_unique_c_name=*/false,
                         /*enable_tiled_emitter=*/false));
    FusionCompiler compiler(mlir_context.get(),
                            FusionCompiler::Options{512, 1, true});
    ASSERT_OK_AND_ASSIGN(
        LlvmKernelSource llvm_source,
        compiler.Compile(std::move(kernel_definition).TakeSource()));
    EXPECT_THAT(
        RunFileCheck(llvm_source.ToString(),
                     elem_type == "c64"
                         ? "CHECK: call void @llvm.memcpy{{.*}}i64 24\nCHECK: "
                           "call void @llvm.memcpy{{.*}}i64 40"
                         : "CHECK: call void @llvm.memcpy{{.*}}i64 3\nCHECK: "
                           "call void @llvm.memcpy{{.*}}i64 5"),
        absl_testing::IsOkAndHolds(true));
  }
}

ABSL_ATTRIBUTE_NO_SANITIZE_CFI XLA_CPU_KernelError* CallCpuKernel(
    FunctionLibrary::Kernel* kernel, XLA_CPU_KernelCallFrame* frame) {
  return (*kernel)(frame);
}

TEST_F(CpuFusionEmitterTest,
       AtomicCasEmitsAlignedUnorderedLoadsAndJitExecutes) {
  constexpr absl::string_view kMlir = R"(
    module attributes {xla.cpu_memory_region_name = "cpu_fusion_emitter_test"} {
      func.func @cpu_atomic_cas(
          %s4_buf: tensor<4xi4> {llvm.align = 16 : index, xla.slice_index = 0},
          %f16_buf: tensor<2xf16> {llvm.align = 16 : index, xla.slice_index = 1},
          %f32_buf: tensor<2xf32> {llvm.align = 16 : index, xla.slice_index = 2},
          %c64_buf: tensor<2xcomplex<f32>> {llvm.align = 16 : index, xla.slice_index = 3})
          -> (tensor<4xi4>, tensor<2xf16>, tensor<2xf32>,
              tensor<2xcomplex<f32>>)
          attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %v_i4 = arith.constant 5 : i4
        %v_f16 = arith.constant 2.5 : f16
        %v_f32 = arith.constant 3.5 : f32
        %v_c64 = complex.constant [4.0 : f32, 6.0 : f32] : complex<f32>

        %out_s4 = xla.atomic_store %v_i4 into %s4_buf[%c1] : tensor<4xi4>
        %out_f16 = xla.atomic_rmw %f16_buf[%c1] : tensor<2xf16> {
          ^bb0(%cur: f16):
            %sum = arith.addf %cur, %v_f16 : f16
            xla.yield %sum : f16
        }
        %out_f32 = xla.atomic_rmw %f32_buf[%c0] : tensor<2xf32> {
          ^bb0(%cur: f32):
            %max = arith.maximumf %cur, %v_f32 : f32
            xla.yield %max : f32
        }
        %out_c64 = xla.atomic_rmw %c64_buf[%c1] : tensor<2xcomplex<f32>> {
          ^bb0(%cur: complex<f32>):
            %sum = complex.add %cur, %v_c64 : complex<f32>
            xla.yield %sum : complex<f32>
        }
        return %out_s4, %out_f16, %out_f32, %out_c64
            : tensor<4xi4>, tensor<2xf16>, tensor<2xf32>,
              tensor<2xcomplex<f32>>
      }
    }
  )";

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  mlir::MLIRContext* context_ptr = mlir_context.get();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource mlir_source,
      MlirKernelSource::ParseFromString(kMlir, std::move(mlir_context)));
  FusionCompiler compiler(context_ptr, FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(mlir_source)));

  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), R"(
    CHECK: load atomic i8, ptr %[[S4_ADDR:.*]] monotonic, align 1
    CHECK: cmpxchg ptr %[[S4_ADDR]], i8 %{{.*}}, i8 %{{.*}} monotonic monotonic
    CHECK: load atomic i16, ptr %[[F16_ADDR:.*]] monotonic, align 2
    CHECK: cmpxchg ptr %[[F16_ADDR]], i16 %{{.*}}, i16 %{{.*}} monotonic monotonic
    CHECK: load atomic i32, ptr %[[F32_ADDR:.*]] monotonic, align 4
    CHECK: cmpxchg ptr %[[F32_ADDR]], i32 %{{.*}}, i32 %{{.*}} monotonic monotonic
    CHECK: load atomic i64, ptr %[[C64_ADDR:.*]] monotonic, align 8
    CHECK: cmpxchg ptr %[[C64_ADDR]], i64 %{{.*}}, i64 %{{.*}} monotonic monotonic
  )"),
              absl_testing::IsOkAndHolds(true));

  std::unique_ptr<IrCompiler> ir_compiler =
      IrCompiler::Create({},
                         IrCompiler::Options{
                             /*opt_level=*/llvm::CodeGenOptLevel::Default,
                             /*optimize_for_size=*/false,
                             TargetMachineOptions(GetDebugOptionsFromFlags()),
                         },
                         IrCompiler::CompilationHooks());
  ASSERT_OK_AND_ASSIGN(JitCompiler jit,
                       JitCompiler::Create({}, std::move(ir_compiler)));
  ASSERT_OK(jit.AddModule(std::move(llvm_source).thread_safe_module()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FunctionLibrary> library,
      std::move(jit).Compile(
          {FunctionLibrary::Sym<FunctionLibrary::Kernel>("cpu_atomic_cas")}));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("cpu_atomic_cas"));

  alignas(16) std::array<uint8_t, 2> s4_bytes = {0xa3, 0x72};
  alignas(16) std::array<half, 2> f16_vals = {half(1.0f), half(1.5f)};
  alignas(16) std::array<float, 2> f32_vals = {1.25f, 9.0f};
  alignas(16) std::array<complex64, 2> c64_vals = {complex64(1.0f, 2.0f),
                                                   complex64(3.0f, 5.0f)};
  std::vector<XLA_CPU_KernelArg> args = {
      {s4_bytes.data(), s4_bytes.size()},
      {f16_vals.data(), f16_vals.size() * sizeof(half)},
      {f32_vals.data(), f32_vals.size() * sizeof(float)},
      {c64_vals.data(), c64_vals.size() * sizeof(complex64)},
  };
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ(CallCpuKernel(kernel, &frame), nullptr);

  EXPECT_EQ(s4_bytes[0], 0x53);
  EXPECT_EQ(s4_bytes[1], 0x72);
  EXPECT_EQ(f16_vals[0], half(1.0f));
  EXPECT_EQ(f16_vals[1], half(4.0f));
  EXPECT_EQ(f32_vals[0], 3.5f);
  EXPECT_EQ(f32_vals[1], 9.0f);
  EXPECT_EQ(c64_vals[0], complex64(1.0f, 2.0f));
  EXPECT_EQ(c64_vals[1], complex64(7.0f, 11.0f));
}

TEST_F(CpuFusionEmitterTest,
       AtomicStoreContiguousLoopEmitsMonotonicAndExecutesInJit) {
  constexpr absl::string_view kMlir = R"(
    module attributes {xla.cpu_memory_region_name = "cpu_atomic_store_loop"} {
      func.func @cpu_atomic_store_loop(
          %src: tensor<128xf32> {llvm.align = 16 : index, xla.invariant, xla.slice_index = 0},
          %dst: tensor<128xf32> {llvm.align = 16 : index, xla.slice_index = 1})
          -> tensor<128xf32>
          attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c128 = arith.constant 128 : index
        %res = scf.for %i = %c0 to %c128 step %c1 iter_args(%cur = %dst) -> (tensor<128xf32>) {
          %v = tensor.extract %src[%i] : tensor<128xf32>
          %next = xla.atomic_store %v into %cur[%i] : tensor<128xf32>
          scf.yield %next : tensor<128xf32>
        }
        return %res : tensor<128xf32>
      }
    }
  )";

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  mlir::MLIRContext* context_ptr = mlir_context.get();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource mlir_source,
      MlirKernelSource::ParseFromString(kMlir, std::move(mlir_context)));
  FusionCompiler compiler(context_ptr, FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(mlir_source)));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), R"(
    CHECK: store atomic float %{{.*}}, ptr %{{.*}} monotonic, align 4
  )"),
              absl_testing::IsOkAndHolds(true));

  // Optimizing and JIT linking fails with an unresolved
  // __llvm_memcpy_element_unordered_atomic_4 symbol on x86_64 if unordered
  // ordering is used instead of monotonic.
  std::unique_ptr<IrCompiler> ir_compiler =
      IrCompiler::Create({},
                         IrCompiler::Options{
                             /*opt_level=*/llvm::CodeGenOptLevel::Default,
                             /*optimize_for_size=*/false,
                             TargetMachineOptions(GetDebugOptionsFromFlags()),
                         },
                         IrCompiler::CompilationHooks());
  ASSERT_OK_AND_ASSIGN(JitCompiler jit,
                       JitCompiler::Create({}, std::move(ir_compiler)));
  ASSERT_OK(jit.AddModule(std::move(llvm_source).thread_safe_module()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FunctionLibrary> library,
      std::move(jit).Compile({FunctionLibrary::Sym<FunctionLibrary::Kernel>(
          "cpu_atomic_store_loop")}));
  ASSERT_OK_AND_ASSIGN(FunctionLibrary::Kernel * kernel,
                       library->ResolveFunction<FunctionLibrary::Kernel>(
                           "cpu_atomic_store_loop"));

  alignas(16) std::array<float, 128> src;
  alignas(16) std::array<float, 128> dst;
  for (int i = 0; i < 128; ++i) {
    src[i] = static_cast<float>(i * 3 + 1);
    dst[i] = -1.0f;
  }
  std::array<XLA_CPU_KernelArg, 2> args = {{
      {src.data(), sizeof(src)},
      {dst.data(), sizeof(dst)},
  }};
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ(CallCpuKernel(kernel, &frame), nullptr);
  EXPECT_EQ(dst, src);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
