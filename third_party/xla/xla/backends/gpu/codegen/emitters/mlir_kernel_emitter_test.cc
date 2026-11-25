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
#include "llvm/TargetParser/Triple.h"
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
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/codegen/emitters/concatenate.h"
#include "xla/backends/gpu/codegen/emitters/loop.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/triton/triton_kernel_source.h"
#include "xla/backends/gpu/codegen/triton/triton_wrapper_result.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/future.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/mlir_context_pool.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

class MlirKernelFusionTestPeer {
 public:
  static absl::StatusOr<LlvmKernelSource> CreateLLVMModule(
      const MlirKernelFusion& emitter, const se::DeviceDescription& device,
      const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment* buffer_assignment,
      mlir::MLIRContext& mlir_context) {
    ABSL_ASSIGN_OR_RETURN(MlirKernelSource source,
                     emitter.EmitMlirSource(device, fusion, entry_function_name,
                                            buffer_assignment, mlir_context));
    return CompileMlirToLlvm(device, *fusion.GetModule(), entry_function_name,
                             emitter.emitter_->unroll_factor(), mlir_context,
                             std::move(source));
  }
};

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
  MlirKernelFusionTest() : mlir_context_(CreateMlirContext()) {
    mlir_context_->appendDialectRegistry(
        MlirKernelEmitter::GetDialectRegistry());
    mlir_context_->loadAllAvailableDialects();
  }

  std::unique_ptr<mlir::MLIRContext> mlir_context_;
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
                           *mlir_context_,
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
  MlirKernelFusion emitter(std::make_unique<DummyCopyEmitter>());
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource source,
                       MlirKernelFusionTestPeer::CreateLLVMModule(
                           emitter, device_info_,
                           *Cast<HloFusionInstruction>(
                               module->entry_computation()->root_instruction()),
                           "fusion",
                           /*buffer_assignment=*/nullptr, *mlir_context_));
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
    MlirKernelFusion emitter(std::make_unique<DummyCopyEmitter>());
    EXPECT_THAT(MlirKernelFusionTestPeer::CreateLLVMModule(
                    emitter, dev_info,
                    *Cast<HloFusionInstruction>(
                        module->entry_computation()->root_instruction()),
                    "fusion", /*buffer_assignment=*/nullptr, *mlir_context_),
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
  std::unique_ptr<mlir::MLIRContext> mlir_context = CreateMlirContext();
  MlirKernelFusion emitter(std::make_unique<DummyElementwiseEmitter>());
  ABSL_ASSIGN_OR_RETURN(LlvmKernelSource source,
                   MlirKernelFusionTestPeer::CreateLLVMModule(
                       emitter, device_info,
                       *Cast<HloFusionInstruction>(
                           module.entry_computation()->root_instruction()),
                       "fusion", /*buffer_assignment=*/nullptr, *mlir_context));
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

// Copy emitter reporting contiguous output writes, so stores can be hinted.
class ContiguousCopyEmitter : public DummyCopyEmitter {
 public:
  bool WritesOutputsContiguously() const override { return true; }
};

// Hints need buffers above 16x L2 (64 MiB on the MI300 test device).
constexpr absl::string_view kLargeCopyModule = R"(
    fused_computation {
      ROOT %p0 = f32[20000000] parameter(0)
    }

    ENTRY main {
      %p0 = f32[20000000] parameter(0)
      ROOT fusion = f32[20000000] fusion(%p0), kind=kLoop,
        calls=fused_computation
    })";

absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
    HloModule& module) {
  AliasInfo alias_info;
  return BufferAssigner::Run(
      &module, std::make_unique<DependencyHloOrdering>(&module),
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
      },
      &alias_info, [](LogicalBuffer::Color) { return 0; },
      BufferAssigner::Options{.allocate_buffers_for_constants = true});
}

absl::StatusOr<std::string> EmitLlvmIrWithBufferAssignment(
    HloModule& module, const se::DeviceDescription& device_info,
    std::unique_ptr<MlirKernelEmitter> kernel_emitter =
        std::make_unique<ContiguousCopyEmitter>()) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> assignment,
                   RunBufferAssignment(module));

  std::unique_ptr<mlir::MLIRContext> mlir_context = CreateMlirContext();
  MlirKernelFusion emitter(std::move(kernel_emitter));
  ABSL_ASSIGN_OR_RETURN(LlvmKernelSource source,
                   MlirKernelFusionTestPeer::CreateLLVMModule(
                       emitter, device_info,
                       *Cast<HloFusionInstruction>(
                           module.entry_computation()->root_instruction()),
                       "fusion", assignment.get(), *mlir_context));
  auto llvm_module = std::move(source).thread_safe_module();
  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *llvm_module.getModuleUnlocked();
  return out;
}

TEST_F(MlirKernelFusionTest, NoNontemporalLoadsWithoutInputIndexingOnMI300) {
  // The emitter has no thread id to input maps, so loads stay temporal.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kLargeCopyModule));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_nontemporal_hints(true);
  ASSERT_OK_AND_ASSIGN(std::string llvm_ir,
                       EmitLlvmIrWithBufferAssignment(
                           *module, TestGpuDeviceInfo::AMDMI300DeviceInfo()));
  EXPECT_THAT(RunFileCheck(llvm_ir, R"(
    // CHECK-NOT: load {{.*}}!nontemporal
  )"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(llvm_ir, R"(
    // CHECK: store float {{.*}}!nontemporal
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(MlirKernelFusionTest, NoNontemporalHintsWhenFlagDisabled) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kLargeCopyModule));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_nontemporal_hints(false);
  ASSERT_OK_AND_ASSIGN(std::string llvm_ir,
                       EmitLlvmIrWithBufferAssignment(
                           *module, TestGpuDeviceInfo::AMDMI300DeviceInfo()));
  EXPECT_THAT(RunFileCheck(llvm_ir, R"(
    // CHECK-NOT: !nontemporal
  )"),
              absl_testing::IsOkAndHolds(true));
}

// Emits the root fusion of `module` with an `Emitter` on MI300.
template <typename Emitter>
absl::StatusOr<std::string> EmitFusionOnMI300(HloModule& module) {
  const se::DeviceDescription device_info =
      TestGpuDeviceInfo::AMDMI300DeviceInfo();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *Cast<HloFusionInstruction>(
          module.entry_computation()->root_instruction()),
      device_info);
  return EmitLlvmIrWithBufferAssignment(module, device_info,
                                        std::make_unique<Emitter>(analysis));
}

TEST_F(MlirKernelFusionTest, NontemporalLoadsOfWholeLinesOnMI300) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    fused_computation {
      %p0 = f32[32,96,112,112] parameter(0)
      ROOT %neg = f32[32,96,112,112] negate(%p0)
    }

    ENTRY main {
      %p0 = f32[32,96,112,112] parameter(0)
      ROOT fusion = f32[32,96,112,112] fusion(%p0), kind=kLoop,
        calls=fused_computation
    })"));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_nontemporal_hints(true);
  ASSERT_OK_AND_ASSIGN(std::string llvm_ir,
                       EmitFusionOnMI300<LoopFusion>(*module));
  EXPECT_THAT(RunFileCheck(llvm_ir, R"(
    // CHECK: load {{.*}}!nontemporal
    // CHECK: store {{.*}}!nontemporal
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(MlirKernelFusionTest, NoNontemporalLoadsOfLinesReadInPiecesOnMI300) {
  // The pad shifts rows off the 128-byte lines: several loads per line.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    fused_computation {
      %p0 = f32[32,96,112,112] parameter(0)
      %zero = f32[] constant(0)
      ROOT %pad = f32[32,96,113,113] pad(%p0, %zero), padding=0_0x0_0x0_1x0_1
    }

    ENTRY main {
      %p0 = f32[32,96,112,112] parameter(0)
      ROOT fusion = f32[32,96,113,113] fusion(%p0), kind=kLoop,
        calls=fused_computation
    })"));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_nontemporal_hints(true);
  ASSERT_OK_AND_ASSIGN(std::string llvm_ir,
                       EmitFusionOnMI300<LoopFusion>(*module));
  EXPECT_THAT(RunFileCheck(llvm_ir, R"(
    // CHECK-NOT: load {{.*}}!nontemporal
  )"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(llvm_ir, R"(
    // CHECK: store {{.*}}!nontemporal
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(MlirKernelFusionTest, NoNontemporalLoadsAfterLeadingPadOnMI300) {
  // The first blocks write only padding; later rows are read in pieces.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    fused_computation {
      %p0 = f32[32,96,112,112] parameter(0)
      %zero = f32[] constant(0)
      ROOT %pad = f32[32,98,114,114] pad(%p0, %zero), padding=0_0x1_1x1_1x1_1
    }

    ENTRY main {
      %p0 = f32[32,96,112,112] parameter(0)
      ROOT fusion = f32[32,98,114,114] fusion(%p0), kind=kLoop,
        calls=fused_computation
    })"));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_nontemporal_hints(true);
  ASSERT_OK_AND_ASSIGN(std::string llvm_ir,
                       EmitFusionOnMI300<LoopFusion>(*module));
  EXPECT_THAT(RunFileCheck(llvm_ir, R"(
    // CHECK-NOT: load {{.*}}!nontemporal
  )"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(llvm_ir, R"(
    // CHECK: store {{.*}}!nontemporal
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(MlirKernelFusionTest, NoNontemporalLoadsOfSubByteElementsOnMI300) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    fused_computation {
      %p0 = s4[200000000] parameter(0)
      ROOT %neg = s4[200000000] negate(%p0)
    }

    ENTRY main {
      %p0 = s4[200000000] parameter(0)
      ROOT fusion = s4[200000000] fusion(%p0), kind=kLoop,
        calls=fused_computation
    })"));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_nontemporal_hints(true);
  ASSERT_OK_AND_ASSIGN(std::string llvm_ir,
                       EmitFusionOnMI300<LoopFusion>(*module));
  EXPECT_THAT(RunFileCheck(llvm_ir, R"(
    // CHECK-NOT: load {{.*}}!nontemporal
  )"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(llvm_ir, R"(
    // CHECK: store {{.*}}!nontemporal
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(MlirKernelFusionTest,
       NontemporalLoadsOfNarrowerConcatenateOperandOnMI300) {
  // The emitter indexes both operands like the wider %p0. Within its own rows,
  // %p1 is read in whole lines too.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    fused_computation {
      %p0 = f32[524288,64] parameter(0)
      %p1 = f32[524288,48] parameter(1)
      ROOT %concat = f32[524288,112] concatenate(%p0, %p1), dimensions={1}
    }

    ENTRY main {
      %p0 = f32[524288,64] parameter(0)
      %p1 = f32[524288,48] parameter(1)
      ROOT fusion = f32[524288,112] fusion(%p0, %p1), kind=kLoop,
        calls=fused_computation
    })"));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_nontemporal_hints(true);
  ASSERT_OK_AND_ASSIGN(std::string llvm_ir,
                       EmitFusionOnMI300<ConcatenateFusion>(*module));
  EXPECT_THAT(RunFileCheck(llvm_ir, R"(
    // CHECK: define void @fusion(ptr {{.*}}%[[P0:[0-9]+]], ptr {{.*}}%[[P1:[0-9]+]], ptr
    // CHECK: %[[P0_PTR:[0-9]+]] = getelementptr {{.*}}, ptr %[[P0]],
    // CHECK-NEXT: load {{.*}}, ptr %[[P0_PTR]], {{.*}}!nontemporal
    // CHECK: %[[P1_PTR:[0-9]+]] = getelementptr {{.*}}, ptr %[[P1]],
    // CHECK-NEXT: load {{.*}}, ptr %[[P1_PTR]], {{.*}}!nontemporal
  )"),
              absl_testing::IsOkAndHolds(true));
}

// Counts the kernels it compiles, to empty binaries.
class CountingKernelCompiler : public KernelCompiler {
 public:
  Future<std::unique_ptr<Thunk>> Compile(Thunk::ThunkInfo, MlirKernelDefinition,
                                         absl::string_view, BorrowedMlirContext,
                                         const HloModule&, int) override {
    return absl::UnimplementedError("Compile");
  }

  Future<TritonWrapperResult> CompileTritonToLlvm(
      absl::string_view, const HloModule&, const se::DeviceDescription&,
      const xtile::BlockLevelParameters&, const llvm::Triple&,
      const std::string&, TritonKernelSource, BorrowedMlirContext,
      bool) override {
    return absl::UnimplementedError("CompileTritonToLlvm");
  }

  Future<std::vector<uint8_t>> CompileToTargetBinary(MlirKernelDefinition,
                                                     absl::string_view,
                                                     BorrowedMlirContext,
                                                     const HloModule&,
                                                     int) override {
    ++num_compiled_kernels_;
    return std::vector<uint8_t>();
  }

  Future<std::vector<uint8_t>> CompileToTargetBinary(
      LlvmKernelSource, absl::string_view,
      std::optional<GpuKernelSpec>) override {
    return absl::UnimplementedError("CompileToTargetBinary");
  }

  int num_compiled_kernels() const { return num_compiled_kernels_; }

 private:
  int num_compiled_kernels_ = 0;
};

// Emits the fusions of `module` on MI300 and returns how many kernels they
// compile; the other fusions reuse those kernels.
absl::StatusOr<int> CountCompiledKernelsOnMI300(HloModule& module) {
  se::GpuTargetConfigProto target_config_proto;
  *target_config_proto.mutable_gpu_device_info() =
      TestGpuDeviceInfo::AMDMI300DeviceInfo().ToProto();
  ABSL_ASSIGN_OR_RETURN(GpuTargetConfig target_config,
                   GpuTargetConfig::FromProto(target_config_proto));
  const GpuTopology gpu_topology(
      /*platform_version=*/"", /*num_partitions=*/1,
      /*num_hosts_per_partition=*/1, /*num_devices_per_host=*/1,
      std::move(target_config));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> assignment,
                   RunBufferAssignment(module));
  std::unique_ptr<mlir::MLIRContext> mlir_context = CreateMlirContext();
  MlirContextPool mlir_context_pool([]() { return CreateMlirContext(); });
  CountingKernelCompiler kernel_compiler;
  IrEmitterContext ir_emitter_context(
      &module, assignment.get(), /*execution_stream_assignment=*/nullptr,
      /*platform_name=*/"ROCM", gpu_topology, mlir_context.get(),
      llvm::Triple("amdgcn-amd-amdhsa"), /*data_layout=*/"", &kernel_compiler,
      xla::cpu::TargetMachineOptions{}, &mlir_context_pool);
  MlirKernelFusion emitter(std::make_unique<DummyCopyEmitter>());
  for (const HloInstruction* instruction :
       module.entry_computation()->instructions()) {
    if (const auto* fusion = DynCast<HloFusionInstruction>(instruction)) {
      ABSL_RETURN_IF_ERROR(
          emitter.Emit(ir_emitter_context, *fusion).Await().status());
    }
  }
  return kernel_compiler.num_compiled_kernels();
}

// Three copies of a fusion. Another instruction also reads the operand of
// %fusion2.
constexpr absl::string_view kFusionsWithOtherReaders = R"(
    fused_computation0 {
      ROOT %p0 = f32[100] parameter(0)
    }

    fused_computation1 {
      ROOT %p0 = f32[100] parameter(0)
    }

    fused_computation2 {
      ROOT %p0 = f32[100] parameter(0)
    }

    ENTRY main {
      %p0 = f32[100] parameter(0)
      %p1 = f32[100] parameter(1)
      %p2 = f32[100] parameter(2)
      %fusion0 = f32[100] fusion(%p0), kind=kLoop, calls=fused_computation0
      %fusion1 = f32[100] fusion(%p1), kind=kLoop, calls=fused_computation1
      %fusion2 = f32[100] fusion(%p2), kind=kLoop, calls=fused_computation2
      %neg = f32[100] negate(%p2)
      ROOT %tuple = (f32[100], f32[100], f32[100], f32[100])
        tuple(%fusion0, %fusion1, %fusion2, %neg)
    })";

TEST_F(MlirKernelFusionTest, OperandsReadElsewhereGetOwnKernelOnMI300) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kFusionsWithOtherReaders));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_nontemporal_hints(true);
  // %fusion0 and %fusion1 share a kernel.
  EXPECT_THAT(CountCompiledKernelsOnMI300(*module),
              absl_testing::IsOkAndHolds(2));
}

TEST_F(MlirKernelFusionTest, OtherReadersShareKernelWhenHintsDisabled) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kFusionsWithOtherReaders));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_nontemporal_hints(false);
  EXPECT_THAT(CountCompiledKernelsOnMI300(*module),
              absl_testing::IsOkAndHolds(1));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
