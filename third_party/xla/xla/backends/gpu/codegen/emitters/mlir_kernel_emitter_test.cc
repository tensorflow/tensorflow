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
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "xla/backends/gpu/codegen/cubin_custom_kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class DummyCopyEmitter : public MlirKernelEmitter {
 public:
  LaunchDimensions launch_dimensions() const final { return {1, 100}; }

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t, mlir::MLIRContext*) const final {
    return std::nullopt;
  }

  std::optional<std::vector<IndexingMap>> ComputeThreadIdToInputIndexing(
      int64_t, mlir::MLIRContext*) const final {
    return std::nullopt;
  }

 protected:
  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override {
    mlir::ImplicitLocOpBuilder b(entry_function.getLoc(), entry_function);
    b.setInsertionPointToStart(entry_function.addEntryBlock());
    auto thread_id = EmitThreadId(b, 0);
    auto value = mlir::tensor::ExtractOp::create(
        b, entry_function.getArgument(0), mlir::ValueRange{thread_id});
    auto result = mlir::tensor::InsertOp::create(
        b, value, entry_function.getArgument(1), mlir::ValueRange{thread_id});
    mlir::func::ReturnOp::create(b, result->getResults());
    return absl::OkStatus();
  }
};

class MlirKernelFusionTest : public HloHardwareIndependentTestBase {
 protected:
  MlirKernelFusionTest() {
    mlir_context_.appendDialectRegistry(
        MlirKernelEmitter::GetDialectRegistry());
    mlir_context_.loadAllAvailableDialects();
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  mlir::MLIRContext mlir_context_;
  stream_executor::DeviceDescription device_info_ =
      TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
};

constexpr absl::string_view kModule = R"(
    fused_computation {
      ROOT %p0 = f32[100] parameter(0)
    }

    ENTRY main {
      %p0 = f32[100] parameter(0)
      ROOT fusion = f32[100] fusion(%p0), kind=kLoop, calls=fused_computation
    })";

TEST_F(MlirKernelFusionTest, CreateMlirModule) {
  auto module = ParseAndReturnVerifiedModule(kModule).value();
  DummyCopyEmitter emitter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto mlir_module,
      emitter.CreateMLIRModule(
          mlir_context_,
          *Cast<HloFusionInstruction>(
              module->entry_computation()->root_instruction()),
          "fusion",
          /*buffer_assignment=*/nullptr));

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *mlir_module;

  TF_ASSERT_OK_AND_ASSIGN(auto filecheck_result, RunFileCheck(out, R"(
    // CHECK:      func.func @fusion(
    // CHECK-SAME:     %[[IN:.*]]: tensor<100xf32> {xla.slice_index = 0
    // CHECK-SAME:     %[[OUT:.*]]: tensor<100xf32> {xla.slice_index = 1
    // CHECK:        %[[TID:.*]] = gpu.thread_id x
    // CHECK:        %[[VAL:.*]] = tensor.extract %[[IN]][%[[TID]]]
    // CHECK:        %[[RET:.*]] = tensor.insert %[[VAL]]
    // CHECK-SAME:     into %[[OUT]][%[[TID]]]
    // CHECK:        return %[[RET]]
  )"));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(MlirKernelFusionTest, CreateLLVMModule) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kModule));
  CubinCustomKernelCompiler kernel_compiler(
      [](llvm::Module& llvm_module, const se::DeviceDescription& descr,
         const DebugOptions& opts) { return std::vector<uint8_t>{}; },
      device_info_, module->config().debug_options());

  ObjectPool<std::unique_ptr<mlir::MLIRContext>> mlir_context_pool(
      []() { return CreateMlirContext(); });
  MlirKernelFusion emitter(std::make_unique<DummyCopyEmitter>());
  TF_ASSERT_OK_AND_ASSIGN(BorrowedMlirContext borrowed_context,
                          mlir_context_pool.GetOrCreate());
  TF_ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource source,
      emitter
          .CreateLLVMModule(
              device_info_,
              *Cast<HloFusionInstruction>(
                  module->entry_computation()->root_instruction()),
              "fusion",
              /*buffer_assignment=*/nullptr, &kernel_compiler,
              std::move(borrowed_context))
          .Await());
  auto llvm_module = std::move(source).thread_safe_module();

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *llvm_module.getModuleUnlocked();

  TF_ASSERT_OK_AND_ASSIGN(
      auto filecheck_result,
      RunFileCheck(
          out, absl::StrReplaceAll(
                   R"(
    // CHECK: define void @fusion(ptr noalias %[[IN:.*]], ptr noalias %[[OUT:.*]])
    // CHECK:   %[[TID:.*]] = call i32 TIDX()
    // CHECK:   %[[IN_PTR:.*]] = getelementptr inbounds [100 x float], ptr %[[IN]], i32 0, i32 %[[TID]]
    // CHECK:   %[[VAL:.*]] = load float, ptr %[[IN_PTR]], align 4
    // CHECK:   %[[OUT_PTR:.*]] = getelementptr inbounds [100 x float], ptr %[[OUT]], i32 0, i32 %[[TID]]
    // CHECK:   store float %[[VAL]], ptr %[[OUT_PTR]], align 4
    // CHECK:   ret void
  )",
                   {{"TIDX", device_info_.cuda_compute_capability().major == -1
                                 ? "@llvm.amdgcn.workitem.id.x"
                                 : "@llvm.nvvm.read.ptx.sreg.tid.x"}})));
  EXPECT_TRUE(filecheck_result);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
