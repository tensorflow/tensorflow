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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
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
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/mlir_context_pool.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

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
  ASSERT_OK_AND_ASSIGN(auto mlir_module,
                       emitter.CreateMLIRModule(
                           mlir_context_,
                           *Cast<HloFusionInstruction>(
                               module->entry_computation()->root_instruction()),
                           "fusion",
                           /*buffer_assignment=*/nullptr));

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *mlir_module;

  ASSERT_OK_AND_ASSIGN(auto filecheck_result, RunFileCheck(out, R"(
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kModule));
  GpuTopology gpu_topology(
      /*platform_version=*/"", /*num_partitions=*/1,
      /*num_hosts_per_partition=*/1, /*num_devices_per_host=*/1);
  CubinCustomKernelCompiler kernel_compiler(
      [](llvm::Module& llvm_module, const se::DeviceDescription& descr,
         const DebugOptions& opts) { return std::vector<uint8_t>{}; },
      device_info_, module->config().debug_options(), gpu_topology);

  MlirContextPool mlir_context_pool([]() { return CreateMlirContext(); });
  MlirKernelFusion emitter(std::make_unique<DummyCopyEmitter>());
  ASSERT_OK_AND_ASSIGN(BorrowedMlirContext borrowed_context,
                       mlir_context_pool.GetOrCreate());
  ASSERT_OK_AND_ASSIGN(
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

  ASSERT_OK_AND_ASSIGN(
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

TEST_F(MlirKernelFusionTest, CreateLLVMModuleRejectsUnsupportedFmaTypes) {
  auto check_rejected = [&](absl::string_view dtype,
                            const se::DeviceDescription& dev_info) {
    const std::string hlo = absl::StrReplaceAll(
        R"(
          fused_computation {
            %p0 = DTYPE[100] parameter(0)
            %p1 = DTYPE[100] parameter(1)
            %p2 = DTYPE[100] parameter(2)
            ROOT %out = DTYPE[100] fma(%p0, %p1, %p2)
          }

          ENTRY main {
            %p0 = DTYPE[100] parameter(0)
            %p1 = DTYPE[100] parameter(1)
            %p2 = DTYPE[100] parameter(2)
            ROOT fusion = DTYPE[100] fusion(%p0, %p1, %p2), kind=kLoop,
              calls=fused_computation
          })",
        {{"DTYPE", dtype}});
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    GpuTopology gpu_topology(
        /*platform_version=*/"", /*num_partitions=*/1,
        /*num_hosts_per_partition=*/1, /*num_devices_per_host=*/1);
    CubinCustomKernelCompiler kernel_compiler(
        [](llvm::Module&, const se::DeviceDescription&, const DebugOptions&) {
          return std::vector<uint8_t>{};
        },
        dev_info, module->config().debug_options(), gpu_topology);
    MlirContextPool mlir_context_pool([]() { return CreateMlirContext(); });
    MlirKernelFusion emitter(std::make_unique<DummyCopyEmitter>());
    ASSERT_OK_AND_ASSIGN(BorrowedMlirContext borrowed_context,
                         mlir_context_pool.GetOrCreate());
    EXPECT_THAT(emitter
                    .CreateLLVMModule(
                        dev_info,
                        *Cast<HloFusionInstruction>(
                            module->entry_computation()->root_instruction()),
                        "fusion", /*buffer_assignment=*/nullptr,
                        &kernel_compiler, std::move(borrowed_context))
                    .Await(),
                absl_testing::StatusIs(absl::StatusCode::kUnimplemented))
        << "dtype=" << dtype;
  };

  check_rejected(
      "f16", TestGpuDeviceInfo::A100SXMDeviceInfo(
                 se::GpuComputeCapability{se::CudaComputeCapability(5, 2)}));
  check_rejected("bf16",
                 TestGpuDeviceInfo::A100SXMDeviceInfo(se::GpuComputeCapability{
                     se::CudaComputeCapability::Volta()}));
  check_rejected("bf16", TestGpuDeviceInfo::AMDMI300DeviceInfo());
  check_rejected("f8e4m3fn", TestGpuDeviceInfo::H100SXMDeviceInfo());
}

// Emits a kernel that computes the root of the fused computation elementwise,
// so that the emitted code contains the fused instructions.
class DummyElementwiseEmitter : public DummyCopyEmitter {
 protected:
  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override {
    mlir::ImplicitLocOpBuilder b(entry_function.getLoc(), entry_function);
    b.setInsertionPointToStart(entry_function.addEntryBlock());
    mlir::Value thread_id = EmitThreadId(b, 0);
    llvm::SmallVector<mlir::Value> operands(
        entry_function.getArguments().drop_back());
    operands.push_back(thread_id);
    mlir::Value value =
        mlir::func::CallOp::create(
            b, call_targets(fusion.fused_expression_root()), operands)
            .getResult(0);
    auto result = mlir::tensor::InsertOp::create(
        b, value, entry_function.getArguments().back(),
        mlir::ValueRange{thread_id});
    mlir::func::ReturnOp::create(b, result->getResults());
    return absl::OkStatus();
  }
};

constexpr absl::string_view kModuleWithMetadata = R"(
    fused_computation {
      %p0 = f32[100] parameter(0)
      %p1 = f32[100] parameter(1)
      %add = f32[100] add(%p0, %p1),
        metadata={op_name="add" source_file="dir/model.py" source_line=10}
      ROOT %mul = f32[100] multiply(%add, %p1),
        metadata={op_name="mul" source_file="dir/model.py" source_line=20}
    }

    ENTRY main {
      %p0 = f32[100] parameter(0)
      %p1 = f32[100] parameter(1)
      ROOT fusion = f32[100] fusion(%p0, %p1), kind=kLoop,
        calls=fused_computation,
        metadata={op_name="fusion" source_file="dir/model.py" source_line=30}
    })";

absl::StatusOr<std::string> EmitLlvmIr(
    const HloModule& module, const se::DeviceDescription& device_info) {
  GpuTopology gpu_topology(
      /*platform_version=*/"", /*num_partitions=*/1,
      /*num_hosts_per_partition=*/1, /*num_devices_per_host=*/1);
  CubinCustomKernelCompiler kernel_compiler(
      [](llvm::Module&, const se::DeviceDescription&, const DebugOptions&) {
        return std::vector<uint8_t>{};
      },
      device_info, module.config().debug_options(), gpu_topology);
  MlirContextPool mlir_context_pool([]() { return CreateMlirContext(); });
  MlirKernelFusion emitter(std::make_unique<DummyElementwiseEmitter>());
  ABSL_ASSIGN_OR_RETURN(BorrowedMlirContext borrowed_context,
                   mlir_context_pool.GetOrCreate());
  ABSL_ASSIGN_OR_RETURN(
      LlvmKernelSource source,
      emitter
          .CreateLLVMModule(device_info,
                            *Cast<HloFusionInstruction>(
                                module.entry_computation()->root_instruction()),
                            "fusion", /*buffer_assignment=*/nullptr,
                            &kernel_compiler, std::move(borrowed_context))
          .Await());
  auto llvm_module = std::move(source).thread_safe_module();
  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *llvm_module.getModuleUnlocked();
  return out;
}

TEST_F(MlirKernelFusionTest, CreateLLVMModuleEmitsLineInfoWhenEnabled) {
  for (bool debug_info : {false, true}) {
    SCOPED_TRACE(debug_info ? "debug info" : "line info");
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(kModuleWithMetadata));
    DebugOptions& debug_options =
        module->mutable_config().mutable_debug_options();
    debug_options.set_xla_gpu_generate_line_info(!debug_info);
    debug_options.set_xla_gpu_generate_debug_info(debug_info);
    ASSERT_OK_AND_ASSIGN(std::string llvm_ir,
                         EmitLlvmIr(*module, device_info_));
    EXPECT_THAT(RunFileCheck(llvm_ir, R"(
      // CHECK: define void @fusion({{.*}}) {{.*}}!dbg ![[SP:[0-9]+]]
      // CHECK: fadd {{.*}}, !dbg ![[ADD_LOC:[0-9]+]]
      // CHECK: fmul {{.*}}, !dbg ![[MUL_LOC:[0-9]+]]
      // CHECK-DAG: !DICompileUnit({{.*}}emissionKind: DebugDirectivesOnly
      // CHECK-DAG: !DIFile(filename: "model.py", directory: "dir")
      // CHECK-DAG: ![[SP]] = distinct !DISubprogram(name: "fusion"{{.*}}line: 30
      // CHECK-DAG: ![[ADD_LOC]] = !DILocation(line: 10,
      // CHECK-DAG: ![[MUL_LOC]] = !DILocation(line: 20,
    )"),
                absl_testing::IsOkAndHolds(true));
  }
}

TEST_F(MlirKernelFusionTest, CreateLLVMModuleEmitsNoLineInfoByDefault) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kModuleWithMetadata));
  ASSERT_OK_AND_ASSIGN(std::string llvm_ir, EmitLlvmIr(*module, device_info_));
  EXPECT_THAT(RunFileCheck(llvm_ir, R"(
    // CHECK: define void @fusion(
    // CHECK-NOT: !dbg
    // CHECK-NOT: !DICompileUnit
  )"),
              absl_testing::IsOkAndHolds(true));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
