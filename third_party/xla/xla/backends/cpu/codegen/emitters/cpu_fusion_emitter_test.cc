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

#include <memory>
#include <string>
#include <utility>

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
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/logical_buffer.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

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

class CpuFusionEmitterTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
      const HloModule& hlo) {
    return BufferAssigner::Run(
        &hlo, std::make_unique<DependencyHloOrdering>(&hlo),
        [](const BufferValue& buffer) {
          return CpuExecutable::ShapeSizeBytes(buffer.shape());
        },
        &alias_info_, [](LogicalBuffer::Color) { return /*alignment=*/1; });
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
    CHECK:             xla.pure_call
    CHECK:             arith.addf
    CHECK:           return %[[XLA_LOOP]]
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
  CpuScatterFusion emitter(*buffer_assignment, fusion);
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
  debug_options.set_xla_cpu_use_thunk_runtime(true);
  debug_options.set_xla_cpu_use_fusion_emitters(true);
  debug_options.set_xla_cpu_prefer_vector_width(512);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  CpuScatterFusion emitter(*buffer_assignment, fusion);
  TF_ASSERT_OK_AND_ASSIGN(KernelDefinition kernel_definition,
                          emitter.EmitKernelDefinition());
  auto [spec, source] = std::move(kernel_definition).ReleaseStorage();
  FusionCompiler compiler(FusionCompiler::Options{512});
  TF_ASSERT_OK_AND_ASSIGN(LlvmIrKernelSource llvm_source,
                          compiler.Compile(std::move(source)));
  auto llvm_dump = llvm_source.ToString();
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(llvm_dump, kExpected));
  EXPECT_TRUE(filecheck_matched);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
