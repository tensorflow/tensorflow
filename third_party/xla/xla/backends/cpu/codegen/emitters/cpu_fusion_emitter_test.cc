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
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "xla/backends/cpu/codegen/emitters/cpu_loop_emitter.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
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
  CpuFusionEmitterTest() {
    mlir_context_
        .loadDialect<mlir::tensor::TensorDialect, mlir::func::FuncDialect,
                     mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                     mlir::complex::ComplexDialect, mlir::math::MathDialect,
                     mlir::scf::SCFDialect, mlir::mhlo::MhloDialect>();
    mlir::DialectRegistry registry;
    mlir::func::registerInlinerExtension(registry);
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir_context_.appendDialectRegistry(registry);
  }

  absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
      const HloModule& hlo) {
    return BufferAssigner::Run(
        &hlo, std::make_unique<DependencyHloOrdering>(&hlo),
        backend().compiler()->BufferSizeBytesFunction(),
        [](LogicalBuffer::Color) { return /*alignment=*/1; });
  }

  mlir::MLIRContext mlir_context_;
  llvm::LLVMContext llvm_context_;
};

TEST_F(CpuFusionEmitterTest, LoopMlir) {
  constexpr absl::string_view kHlo = R"(
    fused_computation {
      ROOT %p0 = f32[100] parameter(0)
    }
    ENTRY main {
      %p0 = f32[100] parameter(0)
      ROOT fusion = f32[100] fusion(%p0), kind=kLoop, calls=fused_computation
    })";
  constexpr absl::string_view kExpected = R"(
    CHECK:       @fusion_entry(
    CHECK-SAME:    xla.entry
    CHECK:           %[[XLA_LOOP:.+]] = xla.loop
    CHECK:           xla.pure_call
    CHECK:           return %[[XLA_LOOP]]
    CHECK:       @fusion(
    CHECK-SAME:    %[[CALL_FRAME:.+]]: !xla_cpu.call_frame)
    CHECK-SAME:    -> !xla_cpu.error
    CHECK-DAG:       xla_cpu.thread_id %[[CALL_FRAME]]
    CHECK-DAG:       xla_cpu.load %[[CALL_FRAME]], 0
    CHECK-DAG:       xla_cpu.load %[[CALL_FRAME]], 1
    CHECK:           xla.pure_call @fusion_entry({{.*}}) {noinline}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  CpuLoopFusion emitter(&mlir_context_, &llvm_context_, *buffer_assignment,
                        fusion);
  TF_ASSERT_OK_AND_ASSIGN(
      auto mlir_module,
      emitter.CreateMLIRModule(mlir_context_, *fusion,
                               std::string(fusion->name()) + "_entry",
                               *buffer_assignment));
  auto mlir_dump = MlirModuleToString(*mlir_module);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(mlir_dump, kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(CpuFusionEmitterTest, LoopLlvm) {
  constexpr absl::string_view kHlo = R"(
    fused_computation {
      ROOT %p0 = f32[100] parameter(0)
    }
    ENTRY main {
      %p0 = f32[100] parameter(0)
      ROOT fusion = f32[100] fusion(%p0), kind=kLoop, calls=fused_computation
    })";
  constexpr absl::string_view kExpected = R"(
    CHECK:      @fusion(
    CHECK-NOT:  @fusion_entry(
    CHECK:      uwtable "frame-pointer"="all"
    CHECK-SAME: "prefer-vector-width"="512"
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHlo));
  hlo_module->mutable_config()
      .mutable_debug_options()
      .set_xla_cpu_prefer_vector_width(512);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  CpuLoopFusion emitter(&mlir_context_, &llvm_context_, *buffer_assignment,
                        fusion);
  TF_ASSERT_OK_AND_ASSIGN(auto result, emitter.Emit());
  auto llvm_dump = LlvmModuleToString(*result.llvm_module);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(llvm_dump, kExpected));
  EXPECT_TRUE(filecheck_matched);
  EXPECT_EQ(absl::StrJoin(result.invariant_arguments, ","), "0");
}

}  // namespace
}  // namespace cpu
}  // namespace xla
