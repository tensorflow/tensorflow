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

#include "xla/backends/cpu/codegen/emitters/cpu_fusion_emitter.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/logical_buffer.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {

std::string LlvmModuleToString(const llvm::Module& module) {
  std::string dump;
  llvm::raw_string_ostream stream(dump);
  stream << module;
  return dump;
}

std::string MlirModuleToString(const mlir::ModuleOp& module) {
  std::string mlir_dump;
  llvm::raw_string_ostream mlir_stream(mlir_dump);
  module->print(mlir_stream);
  return mlir_dump;
}

class CpuFusionEmitterTest : public HloTestBase {
 protected:
  CpuFusionEmitterTest() : mlir_context_(FusionCompiler::CreateContext()) {}

  absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
      const HloModule& hlo) {
    return BufferAssigner::Run(
        &hlo, std::make_unique<DependencyHloOrdering>(&hlo),
        backend().compiler()->BufferSizeBytesFunction(),
        [](LogicalBuffer::Color) { return /*alignment=*/1; });
  }

  std::unique_ptr<mlir::MLIRContext> mlir_context_;
  llvm::LLVMContext llvm_context_;
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
    CHECK:       @wrapped_scatter_entry(
    CHECK-SAME:    xla.entry
    CHECK:           %[[XLA_LOOP:.+]] = xla.loop
    CHECK:           xla.pure_call
    CHECK:           scf.if
    CHECK:             xla.pure_call
    CHECK:             xla.pure_call
    CHECK:             arith.addf
    CHECK:           return %[[XLA_LOOP]]
    CHECK:       @wrapped_scatter(
    CHECK-SAME:    %[[CALL_FRAME:.+]]: !xla_cpu.call_frame)
    CHECK-SAME:    -> !xla_cpu.error
    CHECK-DAG:       xla_cpu.thread_id %[[CALL_FRAME]]
    CHECK-DAG:       xla_cpu.load %[[CALL_FRAME]], 0
    CHECK-DAG:       xla_cpu.load %[[CALL_FRAME]], 1
    CHECK-DAG:       xla_cpu.load %[[CALL_FRAME]], 2
    CHECK-DAG:       xla_cpu.load %[[CALL_FRAME]], 3
    CHECK:           xla.pure_call @wrapped_scatter_entry({{.*}}) {noinline}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kScatterHlo));
  auto& debug_options = hlo_module->mutable_config().mutable_debug_options();
  debug_options.set_xla_cpu_use_thunk_runtime(true);
  debug_options.set_xla_cpu_use_fusion_emitters(true);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  CpuScatterFusion emitter(mlir_context_.get(), &llvm_context_,
                           *buffer_assignment, fusion);
  TF_ASSERT_OK_AND_ASSIGN(
      auto mlir_module,
      emitter.CreateMLIRModule(*mlir_context_, *fusion,
                               std::string(fusion->name()) + "_entry",
                               *buffer_assignment));
  auto mlir_dump = MlirModuleToString(*mlir_module);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(mlir_dump, kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(CpuFusionEmitterTest, ScatterLlvm) {
  constexpr absl::string_view kExpected = R"(
    CHECK-NOT:  @wrapped_scatter_entry(
    CHECK:      @wrapped_scatter(
    CHECK:      uwtable "frame-pointer"="all"
    CHECK-SAME: "prefer-vector-width"="512"
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kScatterHlo));
  auto& debug_options = hlo_module->mutable_config().mutable_debug_options();
  debug_options.set_xla_cpu_use_thunk_runtime(true);
  debug_options.set_xla_cpu_use_fusion_emitters(true);
  debug_options.set_xla_cpu_prefer_vector_width(512);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  CpuScatterFusion emitter(mlir_context_.get(), &llvm_context_,
                           *buffer_assignment, fusion);
  TF_ASSERT_OK_AND_ASSIGN(auto result, emitter.Emit());
  auto llvm_dump = LlvmModuleToString(*result.llvm_module);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(llvm_dump, kExpected));
  EXPECT_TRUE(filecheck_matched);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
