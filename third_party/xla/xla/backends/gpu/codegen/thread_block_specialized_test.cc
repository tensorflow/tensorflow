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
#include "xla/backends/gpu/codegen/thread_block_specialized.h"

#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"

namespace xla::gpu {
namespace {

using ::testing::Optional;

class ThreadBlockSpecializedFusionTest
    : public HloInterpreterReferenceMixin<GpuPjRtCodegenTest> {
 protected:
  ThreadBlockSpecializedFusionTest() : mlir_context_(CreateMlirContext()) {
    mlir_context_->appendDialectRegistry(
        MlirKernelEmitter::GetDialectRegistry());
    mlir_context_->loadAllAvailableDialects();
  }

  bool HasHopperClusterSupport() const {
    const auto* cuda_cc =
        device_description().gpu_compute_capability().cuda_compute_capability();
    return cuda_cc != nullptr && cuda_cc->IsAtLeastHopper();
  }

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitMlirModule(
      absl::string_view hlo) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                     ParseAndReturnVerifiedModule(hlo));
    const auto* fusion = Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
    return ThreadBlockSpecializedFusion(
               HloFusionAnalysis::Create(*fusion, device_info_))
        .CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                  /*buffer_assignment=*/nullptr);
  }

  std::unique_ptr<mlir::MLIRContext> mlir_context_;
  se::DeviceDescription device_info_ = TestGpuDeviceInfo::RTXA6000DeviceInfo();
};

TEST_F(ThreadBlockSpecializedFusionTest, LoopPlusLoop) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m

nested_0 {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

nested_1 {
  p0 = f32[2048]{0} parameter(0)
  ROOT exp = f32[2048]{0} exponential(p0)
}

horizontal_comp {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[2048]{0} parameter(1)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=nested_0
  f1 = f32[2048]{0} fusion(p1), kind=kLoop, calls=nested_1
  ROOT t = (f32[1024]{0}, f32[2048]{0}) tuple(f0, f1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[2048]{0} parameter(1)
  ROOT fusion = (f32[1024]{0}, f32[2048]{0}) fusion(p0, p1), kind=kCustom,
    calls=horizontal_comp,
    backend_config={"fusion_backend_config":{"kind":"__thread_block_specialized"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info_);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kThreadBlockSpecialized);

  ThreadBlockSpecializedFusion fusion_emitter(analysis);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_blocks(), 24);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_threads_per_block(), 128);

  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      fusion_emitter.CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                              /*buffer_assignment=*/nullptr));
  EXPECT_EQ((*mlir_module)
                ->getAttrOfType<mlir::IntegerAttr>("xla.shared_memory_bytes")
                .getInt(),
            0);

  std::string mlir_str;
  llvm::raw_string_ostream mlir_os(mlir_str);
  mlir_os << *mlir_module;

  ASSERT_OK_AND_ASSIGN(bool mlir_check, RunFileCheck(mlir_str, R"(
    // CHECK-LABEL: llvm.func @fusion(
    // CHECK-SAME:    %[[ARG0:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[ARG1:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[ARG2:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[ARG3:[^:]+]]: !llvm.ptr {llvm.noalias})
    // CHECK-SAME:    attributes {xla.backend_kind = #xla.backend_kind<gpu>, xla.entry}
    // CHECK-DAG:     %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:     %[[C24:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK-DAG:     %[[BID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK:         %[[IN_0:.*]] = llvm.icmp "ult" %[[BID]], %[[C8]] : i32
    // CHECK:         llvm.cond_br %[[IN_0]], ^[[THEN0:bb[0-9]+]], ^[[ELSE0:bb[0-9]+]]
    // CHECK:       ^[[THEN0]]:
    // CHECK:         llvm.call @fusion_nested_0(%[[ARG0]], %[[ARG2]], %[[BID]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         llvm.br ^[[EXIT:bb[0-9]+]]
    // CHECK:       ^[[ELSE0]]:
    // CHECK:         %[[IN_1:.*]] = llvm.icmp "ult" %[[BID]], %[[C24]] : i32
    // CHECK:         llvm.cond_br %[[IN_1]], ^[[THEN1:bb[0-9]+]], ^[[ELSE1:bb[0-9]+]]
    // CHECK:       ^[[THEN1]]:
    // CHECK:         %[[REL_BID:.*]] = llvm.sub %[[BID]], %[[C8]] : i32
    // CHECK:         llvm.call @fusion_nested_1(%[[ARG1]], %[[ARG3]], %[[REL_BID]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         llvm.br ^[[ELSE1]]
    // CHECK:       ^[[ELSE1]]:
    // CHECK:         llvm.br ^[[EXIT]]
    // CHECK:       ^[[EXIT]]:
    // CHECK:         llvm.return
    // CHECK:       llvm.func internal @fusion_nested_0(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32) attributes {xla.backend_kind = #xla.backend_kind<gpu>, passthrough = ["alwaysinline"]}
    // CHECK:       llvm.func internal @fusion_nested_1(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32) attributes {xla.backend_kind = #xla.backend_kind<gpu>, passthrough = ["alwaysinline"]}
  )"));
  EXPECT_TRUE(mlir_check);

  ASSERT_OK_AND_ASSIGN(
      ThreadBlockSpecializedFusion::LlvmKernelSourceWrapper llvm_source_wrapper,
      fusion_emitter.CreateLLVMModule(device_info_, *fusion, "fusion",
                                      /*buffer_assignment=*/nullptr,
                                      *mlir_context_));
  EXPECT_EQ(llvm_source_wrapper.max_shmem_bytes, 0);

  std::string llvm_str;
  llvm::raw_string_ostream llvm_os(llvm_str);
  llvm_os << *llvm_source_wrapper.source.module();

  ASSERT_OK_AND_ASSIGN(bool llvm_check, RunFileCheck(llvm_str, R"(
    // CHECK-DAG: define void @fusion(
    // CHECK-DAG: define internal void @fusion_nested_0(ptr %{{.*}}, ptr %{{.*}}, i32 %{{.*}})
    // CHECK-DAG: define internal void @fusion_nested_1(ptr %{{.*}}, ptr %{{.*}}, i32 %{{.*}})
  )"));
  EXPECT_TRUE(llvm_check);

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ThreadBlockSpecializedFusionTest,
       LoopWithThreadCountGuardAndSharedInput) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m

nested_0 {
  lhs = f32[64]{0} parameter(0)
  rhs = f32[64]{0} parameter(1)
  ROOT add = f32[64]{0} add(lhs, rhs)
}

nested_1 {
  p0 = f32[2048]{0} parameter(0)
  ROOT exp = f32[2048]{0} exponential(p0)
}

horizontal_comp {
  p0 = f32[64]{0} parameter(0)
  p1 = f32[2048]{0} parameter(1)
  f0 = f32[64]{0} fusion(p0, p0), kind=kLoop, calls=nested_0
  f1 = f32[2048]{0} fusion(p1), kind=kLoop, calls=nested_1
  ROOT t = (f32[64]{0}, f32[2048]{0}) tuple(f0, f1)
}

ENTRY main {
  p0 = f32[64]{0} parameter(0)
  p1 = f32[2048]{0} parameter(1)
  ROOT fusion = (f32[64]{0}, f32[2048]{0}) fusion(p0, p1), kind=kCustom,
    calls=horizontal_comp,
    backend_config={"fusion_backend_config":{"kind":"__thread_block_specialized"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info_);
  ThreadBlockSpecializedFusion fusion_emitter(analysis);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_blocks(), 17);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_threads_per_block(), 128);

  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      fusion_emitter.CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                              /*buffer_assignment=*/nullptr));

  std::string mlir_str;
  llvm::raw_string_ostream mlir_os(mlir_str);
  mlir_os << *mlir_module;

  ASSERT_OK_AND_ASSIGN(bool mlir_check, RunFileCheck(mlir_str, R"(
    // CHECK-LABEL: llvm.func @fusion(
    // CHECK-SAME:    %[[ARG0:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[ARG1:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[ARG2:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[ARG3:[^:]+]]: !llvm.ptr {llvm.noalias})
    // CHECK-DAG:     %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:     %[[C64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG:     %[[C17:.*]] = llvm.mlir.constant(17 : i32) : i32
    // CHECK-DAG:     %[[BID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK-DAG:     %[[TID:.*]] = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK:         %[[IN_0:.*]] = llvm.icmp "ult" %[[BID]], %[[C1]] : i32
    // CHECK:         llvm.cond_br %[[IN_0]], ^[[THEN0:bb[0-9]+]], ^[[ELSE0:bb[0-9]+]]
    // CHECK:       ^[[THEN0]]:
    // CHECK:         %[[TID_OK:.*]] = llvm.icmp "ult" %[[TID]], %[[C64]] : i32
    // CHECK:         llvm.cond_br %[[TID_OK]], ^[[CALL0:bb[0-9]+]], ^[[JOIN0:bb[0-9]+]]
    // CHECK:       ^[[CALL0]]:
    // CHECK:         llvm.call @fusion_nested_0(%[[ARG0]], %[[ARG0]], %[[ARG2]], %[[BID]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         llvm.br ^[[JOIN0]]
    // CHECK:       ^[[JOIN0]]:
    // CHECK:         llvm.br ^[[EXIT:bb[0-9]+]]
    // CHECK:       ^[[ELSE0]]:
    // CHECK:         %[[IN_1:.*]] = llvm.icmp "ult" %[[BID]], %[[C17]] : i32
    // CHECK:         llvm.cond_br %[[IN_1]], ^[[THEN1:bb[0-9]+]], ^[[ELSE1:bb[0-9]+]]
    // CHECK:       ^[[THEN1]]:
    // CHECK:         %[[REL_BID:.*]] = llvm.sub %[[BID]], %[[C1]] : i32
    // CHECK:         llvm.call @fusion_nested_1(%[[ARG1]], %[[ARG3]], %[[REL_BID]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:       llvm.func internal @fusion_nested_0(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32) attributes {xla.backend_kind = #xla.backend_kind<gpu>, passthrough = ["alwaysinline"]}
  )"));
  EXPECT_TRUE(mlir_check);

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ThreadBlockSpecializedFusionTest,
       LoopPlusReductionWithThreadGuardAndBarrier) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

nested_tr {
  p0 = f32[64,64]{1,0} parameter(0)
  ROOT tr = f32[64,64]{1,0} transpose(p0), dimensions={1,0}
}

nested_reduce {
  p0 = f32[64,512]{1,0} parameter(0)
  c0 = f32[] constant(0)
  ROOT red = f32[64]{0} reduce(p0, c0), dimensions={1}, to_apply=add_f32
}

horizontal_comp {
  p0 = f32[64,64]{1,0} parameter(0)
  p1 = f32[64,512]{1,0} parameter(1)
  f0 = f32[64,64]{1,0} fusion(p0), kind=kInput, calls=nested_tr
  f1 = f32[64]{0} fusion(p1), kind=kInput, calls=nested_reduce
  ROOT t = (f32[64,64]{1,0}, f32[64]{0}) tuple(f0, f1)
}

ENTRY main {
  p0 = f32[64,64]{1,0} parameter(0)
  p1 = f32[64,512]{1,0} parameter(1)
  ROOT fusion = (f32[64,64]{1,0}, f32[64]{0}) fusion(p0, p1), kind=kCustom,
    calls=horizontal_comp,
    backend_config={"fusion_backend_config":{"kind":"__thread_block_specialized"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info_);
  ThreadBlockSpecializedFusion fusion_emitter(analysis);

  EXPECT_EQ(fusion_emitter.launch_dimensions().num_threads_per_block(), 256);

  ASSERT_OK_AND_ASSIGN(
      ThreadBlockSpecializedFusion::LlvmKernelSourceWrapper llvm_source_wrapper,
      fusion_emitter.CreateLLVMModule(device_info_, *fusion, "fusion",
                                      /*buffer_assignment=*/nullptr,
                                      *mlir_context_));
  EXPECT_GT(llvm_source_wrapper.max_shmem_bytes, 0);

  std::string llvm_str;
  llvm::raw_string_ostream llvm_os(llvm_str);
  llvm_os << *llvm_source_wrapper.source.module();

  ASSERT_OK_AND_ASSIGN(bool llvm_check, RunFileCheck(llvm_str, R"(
    // CHECK: @global_smem = external addrspace(3) global [0 x i8], align 16
    // CHECK: define void @fusion(
    // CHECK: icmp ult i32 %{{.*}}, 128
    // CHECK: call void @fusion_nested_0(
    // CHECK: call void @fusion_nested_1(
    // CHECK: define internal void @fusion_nested_0(
    // CHECK: call void @llvm.nvvm.barrier{{.*}}(i32 0, i32 128)
    // CHECK: define internal void @fusion_nested_1(
  )"));
  EXPECT_TRUE(llvm_check);

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(ThreadBlockSpecializedFusionTest,
       TransposePlusTransposeSharedMemoryAliasing) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m

nested_tr0 {
  p0 = f32[64,64]{1,0} parameter(0)
  ROOT tr = f32[64,64]{1,0} transpose(p0), dimensions={1,0}
}

nested_tr1 {
  p0 = f32[128,64]{1,0} parameter(0)
  ROOT tr = f32[64,128]{1,0} transpose(p0), dimensions={1,0}
}

horizontal_comp {
  p0 = f32[64,64]{1,0} parameter(0)
  p1 = f32[128,64]{1,0} parameter(1)
  f0 = f32[64,64]{1,0} fusion(p0), kind=kInput, calls=nested_tr0
  f1 = f32[64,128]{1,0} fusion(p1), kind=kInput, calls=nested_tr1
  ROOT t = (f32[64,64]{1,0}, f32[64,128]{1,0}) tuple(f0, f1)
}

ENTRY main {
  p0 = f32[64,64]{1,0} parameter(0)
  p1 = f32[128,64]{1,0} parameter(1)
  ROOT fusion = (f32[64,64]{1,0}, f32[64,128]{1,0}) fusion(p0, p1),
    kind=kCustom, calls=horizontal_comp,
    backend_config={"fusion_backend_config":{"kind":"__thread_block_specialized"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info_);
  ThreadBlockSpecializedFusion fusion_emitter(analysis);

  ASSERT_OK_AND_ASSIGN(
      ThreadBlockSpecializedFusion::LlvmKernelSourceWrapper llvm_source_wrapper,
      fusion_emitter.CreateLLVMModule(device_info_, *fusion, "fusion",
                                      /*buffer_assignment=*/nullptr,
                                      *mlir_context_));
  EXPECT_GT(llvm_source_wrapper.max_shmem_bytes, 0);
  // Each 32x33xf32 transpose tile is ~4224 bytes (rounded to 16 = 4224).
  // Verify that max_shmem_bytes is the max of a single nested kernel, not their
  // sum (< 6000 bytes).
  EXPECT_LT(llvm_source_wrapper.max_shmem_bytes, 6000);

  std::string llvm_str;
  llvm::raw_string_ostream llvm_os(llvm_str);
  llvm_os << *llvm_source_wrapper.source.module();

  ASSERT_OK_AND_ASSIGN(bool llvm_check, RunFileCheck(llvm_str, R"(
    // CHECK: @global_smem = external addrspace(3) global [0 x i8], align 16
    // CHECK-NOT: addrspace(3) global [
    // CHECK-DAG: define internal void @fusion_nested_0(
    // CHECK-DAG: define internal void @fusion_nested_1(
  )"));
  EXPECT_TRUE(llvm_check);

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ThreadBlockSpecializedFusionTest, MultiOutputNestedFusion) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m

nested_mof {
  p0 = f32[1024]{0} parameter(0)
  n0 = f32[1024]{0} negate(p0)
  e0 = f32[1024]{0} exponential(p0)
  ROOT t = (f32[1024]{0}, f32[1024]{0}) tuple(n0, e0)
}

nested_single {
  p0 = f32[2048]{0} parameter(0)
  ROOT c0 = f32[2048]{0} cosine(p0)
}

horizontal_comp {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[2048]{0} parameter(1)
  f0 = (f32[1024]{0}, f32[1024]{0}) fusion(p0), kind=kLoop, calls=nested_mof
  gte0 = f32[1024]{0} get-tuple-element(f0), index=0
  gte1 = f32[1024]{0} get-tuple-element(f0), index=1
  f1 = f32[2048]{0} fusion(p1), kind=kLoop, calls=nested_single
  ROOT t = (f32[1024]{0}, f32[1024]{0}, f32[2048]{0}) tuple(gte0, gte1, f1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[2048]{0} parameter(1)
  ROOT fusion = (f32[1024]{0}, f32[1024]{0}, f32[2048]{0}) fusion(p0, p1),
    kind=kCustom, calls=horizontal_comp,
    backend_config={"fusion_backend_config":{"kind":"__thread_block_specialized"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info_);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kThreadBlockSpecialized);

  ThreadBlockSpecializedFusion fusion_emitter(analysis);
  ASSERT_OK_AND_ASSIGN(
      ThreadBlockSpecializedFusion::LlvmKernelSourceWrapper llvm_source_wrapper,
      fusion_emitter.CreateLLVMModule(device_info_, *fusion, "fusion",
                                      /*buffer_assignment=*/nullptr,
                                      *mlir_context_));
  std::string llvm_str;
  llvm::raw_string_ostream llvm_os(llvm_str);
  llvm_os << *llvm_source_wrapper.source.module();
  ASSERT_OK_AND_ASSIGN(bool llvm_check, RunFileCheck(llvm_str, R"(
    // CHECK-DAG: define void @fusion(
    // CHECK-DAG: define internal void @fusion_nested_0(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i32 %{{.*}})
    // CHECK-DAG: define internal void @fusion_nested_1(ptr %{{.*}}, ptr %{{.*}}, i32 %{{.*}})
  )"));
  EXPECT_TRUE(llvm_check);

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ThreadBlockSpecializedFusionTest,
       PhasedBlockClusterSize4PlusPhasedBlockClusterSize8) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m

c0_phase0 {
  p0 = f32[512]{0} parameter(0)
  ROOT neg = f32[512]{0} negate(p0)
}

c0_phase1 {
  p0 = f32[512]{0} parameter(0)
  ROOT rev = f32[512]{0} reverse(p0), dimensions={0}
}

cluster_0 {
  p0 = f32[512]{0} parameter(0)
  f0 = f32[512]{0} fusion(p0), kind=kLoop, calls=c0_phase0
  f1 = f32[512]{0} fusion(f0), kind=kLoop, calls=c0_phase1
  ROOT t = (f32[512]{0}, f32[512]{0}) tuple(f0, f1)
}

c1_phase0 {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

c1_phase1 {
  p0 = f32[1024]{0} parameter(0)
  ROOT rev = f32[1024]{0} reverse(p0), dimensions={0}
}

cluster_1 {
  p0 = f32[1024]{0} parameter(0)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=c1_phase0
  f1 = f32[1024]{0} fusion(f0), kind=kLoop, calls=c1_phase1
  ROOT t = (f32[1024]{0}, f32[1024]{0}) tuple(f0, f1)
}

horizontal_comp {
  p0 = f32[512]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  c0 = (f32[512]{0}, f32[512]{0}) fusion(p0), kind=kCustom, calls=cluster_0,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
  c0_0 = f32[512]{0} get-tuple-element(c0), index=0
  c0_1 = f32[512]{0} get-tuple-element(c0), index=1
  c1 = (f32[1024]{0}, f32[1024]{0}) fusion(p1), kind=kCustom, calls=cluster_1,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
  c1_0 = f32[1024]{0} get-tuple-element(c1), index=0
  c1_1 = f32[1024]{0} get-tuple-element(c1), index=1
  ROOT t = (f32[512]{0}, f32[512]{0}, f32[1024]{0}, f32[1024]{0})
    tuple(c0_0, c0_1, c1_0, c1_1)
}

ENTRY main {
  p0 = f32[512]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  ROOT fusion = (f32[512]{0}, f32[512]{0}, f32[1024]{0}, f32[1024]{0})
    fusion(p0, p1), kind=kCustom, calls=horizontal_comp,
    backend_config={"fusion_backend_config":{"kind":"__thread_block_specialized"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info_);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kThreadBlockSpecialized);

  ThreadBlockSpecializedFusion fusion_emitter(analysis);
  // cluster_0 needs 4 blocks (padded to 8) and cluster_1 needs 8 blocks -> 16.
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_blocks(), 16);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_threads_per_block(), 128);
  EXPECT_THAT(fusion_emitter.cluster_dim(), Optional(se::ClusterDim(8, 1, 1)));

  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      fusion_emitter.CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                              /*buffer_assignment=*/nullptr));

  std::string mlir_str;
  llvm::raw_string_ostream mlir_os(mlir_str);
  mlir_os << *mlir_module;

  ASSERT_OK_AND_ASSIGN(bool mlir_check, RunFileCheck(mlir_str, R"(
    // CHECK-LABEL: llvm.func @fusion(
    // CHECK-SAME:    %[[ARG0:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[ARG1:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[OUT0:[^:]+]]: !llvm.ptr,
    // CHECK-SAME:    %[[OUT1:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[OUT2:[^:]+]]: !llvm.ptr,
    // CHECK-SAME:    %[[OUT3:[^:]+]]: !llvm.ptr {llvm.noalias})
    // CHECK-SAME:    nvvm.cluster_dim = array<i32: 8, 1, 1>
    // CHECK-DAG:     %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:     %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:     %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:     %[[BID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK:         %[[IN_0:.*]] = llvm.icmp "ult" %[[BID]], %[[C8]] : i32
    // CHECK:         llvm.cond_br %[[IN_0]], ^[[THEN0:bb[0-9]+]], ^[[ELSE0:bb[0-9]+]]
    // CHECK:       ^[[THEN0]]:
    // CHECK:         llvm.icmp "ult" %[[BID]], %[[C4]] : i32
    // CHECK:         llvm.call @fusion_nested_0(%[[ARG0]], %[[OUT0]], %[[BID]])
    // CHECK:         nvvm.cluster.arrive aligned
    // CHECK:         nvvm.cluster.wait aligned
    // CHECK:         llvm.icmp "ult" %[[BID]], %[[C4]] : i32
    // CHECK:         llvm.call @fusion_nested_1(%[[OUT0]], %[[OUT1]], %[[BID]])
    // CHECK:       ^[[ELSE0]]:
    // CHECK:         %[[IN_1:.*]] = llvm.icmp "ult" %[[BID]], %[[C16]] : i32
    // CHECK:         llvm.cond_br %[[IN_1]], ^[[THEN1:bb[0-9]+]], ^[[ELSE1:bb[0-9]+]]
    // CHECK:       ^[[THEN1]]:
    // CHECK:         %[[REL_BID:.*]] = llvm.sub %[[BID]], %[[C8]] : i32
    // CHECK-NEXT:    llvm.call @fusion_nested_2(%[[ARG1]], %[[OUT2]], %[[REL_BID]])
    // CHECK-NEXT:    nvvm.cluster.arrive aligned
    // CHECK-NEXT:    nvvm.cluster.wait aligned
    // CHECK-NEXT:    llvm.call @fusion_nested_3(%[[OUT2]], %[[OUT3]], %[[REL_BID]])
  )"));
  EXPECT_TRUE(mlir_check);

  if (HasHopperClusterSupport()) {
    EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-5, 1e-5}));
  }
}

TEST_F(ThreadBlockSpecializedFusionTest,
       PhasedBlockClusterPlusReductionWithBlockPaddingAndThreadGuard) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

c0_phase0 {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

c0_phase1 {
  p0 = f32[1024]{0} parameter(0)
  ROOT rev = f32[1024]{0} reverse(p0), dimensions={0}
}

cluster_0 {
  p0 = f32[1024]{0} parameter(0)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=c0_phase0
  f1 = f32[1024]{0} fusion(f0), kind=kLoop, calls=c0_phase1
  ROOT t = (f32[1024]{0}, f32[1024]{0}) tuple(f0, f1)
}

nested_reduce {
  p0 = f32[24,512]{1,0} parameter(0)
  c0 = f32[] constant(0)
  ROOT red = f32[24]{0} reduce(p0, c0), dimensions={1}, to_apply=add_f32
}

horizontal_comp {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[24,512]{1,0} parameter(1)
  c0 = (f32[1024]{0}, f32[1024]{0}) fusion(p0), kind=kCustom, calls=cluster_0,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
  c0_0 = f32[1024]{0} get-tuple-element(c0), index=0
  c0_1 = f32[1024]{0} get-tuple-element(c0), index=1
  f1 = f32[24]{0} fusion(p1), kind=kInput, calls=nested_reduce
  ROOT t = (f32[1024]{0}, f32[1024]{0}, f32[24]{0}) tuple(c0_0, c0_1, f1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[24,512]{1,0} parameter(1)
  ROOT fusion = (f32[1024]{0}, f32[1024]{0}, f32[24]{0}) fusion(p0, p1),
    kind=kCustom, calls=horizontal_comp,
    backend_config={"fusion_backend_config":{"kind":"__thread_block_specialized"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info_);
  ThreadBlockSpecializedFusion fusion_emitter(analysis);
  // cluster_0 uses 8 blocks x 128 threads; nested_reduce uses 3 blocks x 256
  // threads. Total blocks = RoundUpTo(8 + 3, 8) = 16.
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_blocks(), 16);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_threads_per_block(), 256);
  EXPECT_THAT(fusion_emitter.cluster_dim(), Optional(se::ClusterDim(8, 1, 1)));

  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      fusion_emitter.CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                              /*buffer_assignment=*/nullptr));

  std::string mlir_str;
  llvm::raw_string_ostream mlir_os(mlir_str);
  mlir_os << *mlir_module;

  ASSERT_OK_AND_ASSIGN(bool mlir_check, RunFileCheck(mlir_str, R"(
    // CHECK-LABEL: llvm.func @fusion(
    // CHECK-SAME:    nvvm.cluster_dim = array<i32: 8, 1, 1>
    // CHECK-DAG:     %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:     %[[C11:.*]] = llvm.mlir.constant(11 : i32) : i32
    // CHECK-DAG:     %[[C128:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK-DAG:     %[[BID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK-DAG:     %[[TID:.*]] = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK:         %[[IN_0:.*]] = llvm.icmp "ult" %[[BID]], %[[C8]] : i32
    // CHECK:         llvm.cond_br %[[IN_0]], ^[[THEN0:bb[0-9]+]], ^[[ELSE0:bb[0-9]+]]
    // CHECK:       ^[[THEN0]]:
    // CHECK:         llvm.icmp "ult" %[[TID]], %[[C128]] : i32
    // CHECK:         llvm.call @fusion_nested_0(
    // CHECK:         nvvm.cluster.arrive aligned
    // CHECK:         nvvm.cluster.wait aligned
    // CHECK:         llvm.icmp "ult" %[[TID]], %[[C128]] : i32
    // CHECK:         llvm.call @fusion_nested_1(
    // CHECK:       ^[[ELSE0]]:
    // CHECK:         %[[IN_1:.*]] = llvm.icmp "ult" %[[BID]], %[[C11]] : i32
    // CHECK:         llvm.cond_br %[[IN_1]], ^[[THEN1:bb[0-9]+]], ^[[ELSE1:bb[0-9]+]]
    // CHECK:       ^[[THEN1]]:
    // CHECK:         %[[REL_BID:.*]] = llvm.sub %[[BID]], %[[C8]] : i32
    // CHECK-NOT:     llvm.icmp "ult" %[[REL_BID]]
    // CHECK:         llvm.call @fusion_nested_2({{.*}}, %[[REL_BID]])
  )"));
  EXPECT_TRUE(mlir_check);

  if (HasHopperClusterSupport()) {
    EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-4, 1e-4}));
  }
}

TEST_F(ThreadBlockSpecializedFusionTest,
       TightBlockPackingWithClusteredAndNonClusteredSubKernels) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m

c0_phase0 {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

c0_phase1 {
  p0 = f32[1024]{0} parameter(0)
  ROOT rev = f32[1024]{0} reverse(p0), dimensions={0}
}

cluster_0 {
  p0 = f32[1024]{0} parameter(0)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=c0_phase0
  f1 = f32[1024]{0} fusion(f0), kind=kLoop, calls=c0_phase1
  ROOT t = (f32[1024]{0}, f32[1024]{0}) tuple(f0, f1)
}

c1_phase0 {
  p0 = f32[512]{0} parameter(0)
  ROOT exp = f32[512]{0} exponential(p0)
}

c1_phase1 {
  p0 = f32[512]{0} parameter(0)
  ROOT rev = f32[512]{0} reverse(p0), dimensions={0}
}

cluster_1 {
  p0 = f32[512]{0} parameter(0)
  f0 = f32[512]{0} fusion(p0), kind=kLoop, calls=c1_phase0
  f1 = f32[512]{0} fusion(f0), kind=kLoop, calls=c1_phase1
  ROOT t = (f32[512]{0}, f32[512]{0}) tuple(f0, f1)
}

loop_3_blocks {
  p0 = f32[384]{0} parameter(0)
  rev = f32[384]{0} reverse(p0), dimensions={0}
  ROOT abs = f32[384]{0} abs(rev)
}

loop_4_blocks {
  p0 = f32[512]{0} parameter(0)
  rev = f32[512]{0} reverse(p0), dimensions={0}
  ROOT cos = f32[512]{0} cosine(rev)
}

horizontal_comp {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  p2 = f32[384]{0} parameter(2)
  p3 = f32[512]{0} parameter(3)
  c0 = (f32[1024]{0}, f32[1024]{0}) fusion(p0), kind=kCustom, calls=cluster_0,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
  c0_0 = f32[1024]{0} get-tuple-element(c0), index=0
  c0_1 = f32[1024]{0} get-tuple-element(c0), index=1
  l0 = f32[384]{0} fusion(p2), kind=kLoop, calls=loop_3_blocks
  c1 = (f32[512]{0}, f32[512]{0}) fusion(p1), kind=kCustom, calls=cluster_1,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
  c1_0 = f32[512]{0} get-tuple-element(c1), index=0
  c1_1 = f32[512]{0} get-tuple-element(c1), index=1
  l1 = f32[512]{0} fusion(p3), kind=kLoop, calls=loop_4_blocks
  ROOT t = (f32[1024]{0}, f32[1024]{0}, f32[384]{0}, f32[512]{0}, f32[512]{0},
            f32[512]{0}) tuple(c0_0, c0_1, l0, c1_0, c1_1, l1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  p2 = f32[384]{0} parameter(2)
  p3 = f32[512]{0} parameter(3)
  ROOT fusion = (f32[1024]{0}, f32[1024]{0}, f32[384]{0}, f32[512]{0},
                 f32[512]{0}, f32[512]{0}) fusion(p0, p1, p2, p3),
    kind=kCustom, calls=horizontal_comp,
    backend_config={"fusion_backend_config":{"kind":"__thread_block_specialized"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info_);
  ThreadBlockSpecializedFusion fusion_emitter(analysis);
  // [0..7: cluster_0 (8 blocks)]
  // [8..15: cluster_1 (4 blocks padded to 8)]
  // [16..18: loop_3_blocks (3 blocks)]
  // [19..22: loop_4_blocks (4 blocks)]
  // [23..23: trailing grid padding to multiple of 8] -> 24 blocks total.
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_blocks(), 24);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_threads_per_block(), 128);
  EXPECT_THAT(fusion_emitter.cluster_dim(), Optional(se::ClusterDim(8, 1, 1)));

  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      fusion_emitter.CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                              /*buffer_assignment=*/nullptr));

  std::string mlir_str;
  llvm::raw_string_ostream mlir_os(mlir_str);
  mlir_os << *mlir_module;

  ASSERT_OK_AND_ASSIGN(bool mlir_check, RunFileCheck(mlir_str, R"(
    // CHECK-LABEL: llvm.func @fusion(
    // CHECK-SAME:    nvvm.cluster_dim = array<i32: 8, 1, 1>
    // CHECK-DAG:     %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:     %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:     %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:     %[[C19:.*]] = llvm.mlir.constant(19 : i32) : i32
    // CHECK-DAG:     %[[C23:.*]] = llvm.mlir.constant(23 : i32) : i32
    // CHECK-DAG:     %[[BID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK:         %[[IN_0:.*]] = llvm.icmp "ult" %[[BID]], %[[C8]] : i32
    // CHECK:         llvm.cond_br %[[IN_0]], ^[[THEN0:bb[0-9]+]], ^[[ELSE0:bb[0-9]+]]
    // CHECK:       ^[[THEN0]]:
    // CHECK:         llvm.call @fusion_nested_0(
    // CHECK:         nvvm.cluster.arrive aligned
    // CHECK:         nvvm.cluster.wait aligned
    // CHECK:         llvm.call @fusion_nested_1(
    // CHECK:       ^[[ELSE0]]:
    // CHECK:         %[[IN_1:.*]] = llvm.icmp "ult" %[[BID]], %[[C16]] : i32
    // CHECK:         llvm.cond_br %[[IN_1]], ^[[THEN1:bb[0-9]+]], ^[[ELSE1:bb[0-9]+]]
    // CHECK:       ^[[THEN1]]:
    // CHECK:         %[[REL_BID1:.*]] = llvm.sub %[[BID]], %[[C8]] : i32
    // CHECK-NEXT:    llvm.icmp "ult" %[[REL_BID1]], %[[C4]] : i32
    // CHECK:         llvm.call @fusion_nested_2({{.*}}, %[[REL_BID1]])
    // CHECK:         nvvm.cluster.arrive aligned
    // CHECK:         nvvm.cluster.wait aligned
    // CHECK:         llvm.icmp "ult" %[[REL_BID1]], %[[C4]] : i32
    // CHECK:         llvm.call @fusion_nested_3({{.*}}, %[[REL_BID1]])
    // CHECK:       ^[[ELSE1]]:
    // CHECK:         %[[IN_2:.*]] = llvm.icmp "ult" %[[BID]], %[[C19]] : i32
    // CHECK:         llvm.cond_br %[[IN_2]], ^[[THEN2:bb[0-9]+]], ^[[ELSE2:bb[0-9]+]]
    // CHECK:       ^[[THEN2]]:
    // CHECK:         %[[REL_BID2:.*]] = llvm.sub %[[BID]], %[[C16]] : i32
    // CHECK-NEXT:    llvm.call @fusion_nested_4({{.*}}, %[[REL_BID2]])
    // CHECK:       ^[[ELSE2]]:
    // CHECK:         %[[IN_3:.*]] = llvm.icmp "ult" %[[BID]], %[[C23]] : i32
    // CHECK:         llvm.cond_br %[[IN_3]], ^[[THEN3:bb[0-9]+]], ^[[ELSE3:bb[0-9]+]]
    // CHECK:       ^[[THEN3]]:
    // CHECK:         %[[REL_BID3:.*]] = llvm.sub %[[BID]], %[[C19]] : i32
    // CHECK-NEXT:    llvm.call @fusion_nested_5({{.*}}, %[[REL_BID3]])
  )"));
  EXPECT_TRUE(mlir_check);

  if (HasHopperClusterSupport()) {
    EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-4, 1e-4}));
  }
}

TEST_F(ThreadBlockSpecializedFusionTest,
       TwoPhaseProducerConsumerWithCrossCtaRead) {
  // `phase_0` uses 4 CTAs (`f32[512]` exponential) while `phase_1` uses all 8
  // CTAs (`f32[1024]`) and `reverse`s `phase_0`'s output across CTAs,
  // exercising both CTA range guarding and global-memory visibility across the
  // barrier.
  constexpr absl::string_view kHlo = R"hlo(
phase_0 { ROOT exp = f32[512] exponential(f32[512] parameter(0)) }
phase_1 {
  bc = f32[2,512] broadcast(f32[512] parameter(0)), dimensions={1}
  rev = f32[1024] reverse(f32[1024] reshape(bc)), dimensions={0}
  ROOT add = f32[1024] add(rev, f32[1024] parameter(1))
}
phased_comp {
  f0 = f32[512] fusion(f32[512] parameter(0)), kind=kLoop, calls=phase_0
  f1 = f32[1024] fusion(f0, f32[1024] parameter(1)), kind=kLoop, calls=phase_1
  ROOT t = (f32[512], f32[1024]) tuple(f0, f1)
}
ENTRY main {
  ROOT fusion = (f32[512], f32[1024]) fusion(f32[512] parameter(0), f32[1024] parameter(1)),
    kind=kCustom, calls=phased_comp,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info_);
  EXPECT_EQ(analysis.emitter_fusion_kind(),
            HloFusionAnalysis::EmitterFusionKind::kThreadBlockSpecialized);

  ThreadBlockSpecializedFusion fusion_emitter(analysis);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_blocks(), 8);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_threads_per_block(), 128);
  EXPECT_THAT(fusion_emitter.cluster_dim(), Optional(se::ClusterDim(8, 1, 1)));

  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      fusion_emitter.CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                              /*buffer_assignment=*/nullptr));
  std::string mlir_str;
  llvm::raw_string_ostream(mlir_str) << *mlir_module;
  ASSERT_OK_AND_ASSIGN(bool mlir_check, RunFileCheck(mlir_str, R"(
    // CHECK-LABEL: llvm.func @fusion(
    // CHECK-SAME:    %[[ARG0:[^:]+]]: !llvm.ptr {llvm.noalias}, %[[ARG1:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[ARG2:[^:]+]]: !llvm.ptr, %[[ARG3:[^:]+]]: !llvm.ptr {llvm.noalias})
    // CHECK-SAME:    nvvm.cluster_dim = array<i32: 8, 1, 1>
    // CHECK-DAG:     %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:     %[[BID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK:         %[[IN:.*]] = llvm.icmp "ult" %[[BID]], %[[C4]] : i32
    // CHECK:         llvm.cond_br %[[IN]], ^[[PHASE0_CALL:bb[0-9]+]], ^[[BARRIER:bb[0-9]+]]
    // CHECK:       ^[[PHASE0_CALL]]:
    // CHECK:         llvm.call @fusion_nested_0(%[[ARG0]], %[[ARG2]], %[[BID]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         llvm.br ^[[BARRIER]]
    // CHECK:       ^[[BARRIER]]:
    // CHECK:         nvvm.cluster.arrive aligned
    // CHECK:         nvvm.cluster.wait aligned
    // CHECK:         llvm.call @fusion_nested_1(%[[ARG2]], %[[ARG1]], %[[ARG3]], %[[BID]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         llvm.return
    // CHECK:       llvm.func internal @fusion_nested_0(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32) attributes {xla.backend_kind = #xla.backend_kind<gpu>, passthrough = ["alwaysinline"]}
    // CHECK:       llvm.func internal @fusion_nested_1(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32) attributes {xla.backend_kind = #xla.backend_kind<gpu>, passthrough = ["alwaysinline"]}
  )"));
  EXPECT_TRUE(mlir_check);

  if (HasHopperClusterSupport()) {
    EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-5, 1e-5}));
  }
}

TEST_F(ThreadBlockSpecializedFusionTest,
       HorizontalNestedInsidePhasedBlockCluster) {
  // Phase 0 is a `__thread_block_specialized` horizontal fusion running two
  // leaf sub-kernels (`branch_a` on CTA 0 with 64 threads, `branch_b` on CTAs
  // 1..2 with 128 threads, while CTA 3 skips to the barrier). After the cluster
  // barrier, Phase 1 (`merge_phase`) runs across all 4 CTAs and reads both
  // Phase 0 outputs across CTAs.
  constexpr absl::string_view kHlo = R"hlo(
HloModule m

branch_a {
  p0 = f32[64]{0} parameter(0)
  ROOT exp = f32[64]{0} exponential(p0)
}

branch_b {
  p0 = f32[256]{0} parameter(0)
  rev = f32[256]{0} reverse(p0), dimensions={0}
  ROOT neg = f32[256]{0} negate(rev)
}

horiz_phase_0 {
  p0 = f32[64]{0} parameter(0)
  p1 = f32[256]{0} parameter(1)
  f0 = f32[64]{0} fusion(p0), kind=kLoop, calls=branch_a
  f1 = f32[256]{0} fusion(p1), kind=kLoop, calls=branch_b
  ROOT t = (f32[64]{0}, f32[256]{0}) tuple(f0, f1)
}

merge_phase {
  a = f32[64]{0} parameter(0)
  b = f32[256]{0} parameter(1)
  a_bc = f32[8,64]{1,0} broadcast(a), dimensions={1}
  b_bc = f32[2,256]{1,0} broadcast(b), dimensions={1}
  b_rev = f32[512]{0} reverse(f32[512]{0} reshape(b_bc)), dimensions={0}
  ROOT add = f32[512]{0} add(f32[512]{0} reshape(a_bc), b_rev)
}

phased_comp {
  p0 = f32[64]{0} parameter(0)
  p1 = f32[256]{0} parameter(1)
  h0 = (f32[64]{0}, f32[256]{0}) fusion(p0, p1), kind=kCustom,
    calls=horiz_phase_0,
    backend_config={"fusion_backend_config":{"kind":"__thread_block_specialized"}}
  h0_0 = f32[64]{0} get-tuple-element(h0), index=0
  h0_1 = f32[256]{0} get-tuple-element(h0), index=1
  m1 = f32[512]{0} fusion(h0_0, h0_1), kind=kLoop, calls=merge_phase
  ROOT t = (f32[64]{0}, f32[256]{0}, f32[512]{0}) tuple(h0_0, h0_1, m1)
}

ENTRY main {
  p0 = f32[64]{0} parameter(0)
  p1 = f32[256]{0} parameter(1)
  ROOT fusion = (f32[64]{0}, f32[256]{0}, f32[512]{0}) fusion(p0, p1),
    kind=kCustom, calls=phased_comp,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info_);
  ThreadBlockSpecializedFusion fusion_emitter(analysis);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_blocks(), 4);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_threads_per_block(), 128);
  EXPECT_THAT(fusion_emitter.cluster_dim(), Optional(se::ClusterDim(4, 1, 1)));

  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      fusion_emitter.CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                              /*buffer_assignment=*/nullptr));
  std::string mlir_str;
  llvm::raw_string_ostream(mlir_str) << *mlir_module;
  ASSERT_OK_AND_ASSIGN(bool mlir_check, RunFileCheck(mlir_str, R"(
    // CHECK-LABEL: llvm.func @fusion(
    // CHECK-SAME:    %[[ARG0:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[ARG1:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[OUT0:[^:]+]]: !llvm.ptr,
    // CHECK-SAME:    %[[OUT1:[^:]+]]: !llvm.ptr,
    // CHECK-SAME:    %[[OUT2:[^:]+]]: !llvm.ptr {llvm.noalias})
    // CHECK-SAME:    nvvm.cluster_dim = array<i32: 4, 1, 1>
    // CHECK-DAG:     %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:     %[[C3:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG:     %[[C64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG:     %[[BID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK-DAG:     %[[TID:.*]] = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK:         %[[IN_0:.*]] = llvm.icmp "ult" %[[BID]], %[[C1]] : i32
    // CHECK:         llvm.cond_br %[[IN_0]], ^[[THEN0:bb[0-9]+]], ^[[ELSE0:bb[0-9]+]]
    // CHECK:       ^[[THEN0]]:
    // CHECK:         %[[TID_OK:.*]] = llvm.icmp "ult" %[[TID]], %[[C64]] : i32
    // CHECK:         llvm.call @fusion_nested_0(%[[ARG0]], %[[OUT0]], %[[BID]])
    // CHECK:       ^[[ELSE0]]:
    // CHECK:         %[[IN_1:.*]] = llvm.icmp "ult" %[[BID]], %[[C3]] : i32
    // CHECK:         llvm.cond_br %[[IN_1]], ^[[THEN1:bb[0-9]+]], ^[[ELSE1:bb[0-9]+]]
    // CHECK:       ^[[THEN1]]:
    // CHECK:         %[[REL_BID:.*]] = llvm.sub %[[BID]], %[[C1]] : i32
    // CHECK-NEXT:    llvm.call @fusion_nested_1(%[[ARG1]], %[[OUT1]], %[[REL_BID]])
    // CHECK:         nvvm.cluster.arrive aligned
    // CHECK:         nvvm.cluster.wait aligned
    // CHECK:         llvm.call @fusion_nested_2(%[[OUT0]], %[[OUT1]], %[[OUT2]], %[[BID]])
  )"));
  EXPECT_TRUE(mlir_check);

  if (HasHopperClusterSupport()) {
    EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-5, 1e-5}));
  }
}

TEST_F(ThreadBlockSpecializedFusionTest,
       ReductionPhaseFollowedByLoopPhaseInCluster) {
  constexpr absl::string_view kHlo = R"hlo(
add_scalar { ROOT add = f32[] add(f32[] parameter(0), f32[] parameter(1)) }
reduce_phase {
  ROOT red = f32[32] reduce(f32[32,128] parameter(0), f32[] constant(0.0)),
    dimensions={1}, to_apply=add_scalar
}
loop_phase {
  rev = f32[32] reverse(f32[32] parameter(0)), dimensions={0}
  ROOT exp = f32[32] exponential(rev)
}
phased_comp {
  f0 = f32[32] fusion(f32[32,128] parameter(0)), kind=kInput, calls=reduce_phase
  f1 = f32[32] fusion(f0), kind=kLoop, calls=loop_phase
  ROOT t = (f32[32], f32[32]) tuple(f0, f1)
}
ENTRY main {
  ROOT fusion = (f32[32], f32[32]) fusion(f32[32,128] parameter(0)),
    kind=kCustom, calls=phased_comp,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       EmitMlirModule(kHlo));
  if (HasHopperClusterSupport()) {
    EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-4, 1e-4}));
  }
}

TEST_F(ThreadBlockSpecializedFusionTest,
       DoesNotOverwriteSharedOuterOperandAcrossPhases) {
  if (!HasHopperClusterSupport()) {
    GTEST_SKIP() << "Requires Hopper (SM90+) thread-block clusters.";
  }
  constexpr absl::string_view kHlo = R"hlo(
phase_0 { ROOT neg = f32[256] negate(f32[256] parameter(0)) }
phase_1 { ROOT add = f32[256] add(f32[256] parameter(0), f32[256] parameter(1)) }
phased_comp {
  p0 = f32[256] parameter(0)
  f0 = f32[256] fusion(p0), kind=kLoop, calls=phase_0
  f1 = f32[256] fusion(f0, p0), kind=kLoop, calls=phase_1
  ROOT t = (f32[256], f32[256]) tuple(f0, f1)
}
ENTRY main {
  fused = (f32[256], f32[256]) fusion(f32[256] negate(f32[256] parameter(0))),
    kind=kCustom, calls=phased_comp,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
  out0 = f32[256] get-tuple-element(fused), index=0
  out1 = f32[256] get-tuple-element(fused), index=1
  ROOT sum = f32[256] add(out0, out1)
}
)hlo";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ThreadBlockSpecializedFusionTest,
       PhasedBlockClusterWithReversedRootTupleOrder) {
  constexpr absl::string_view kHlo = R"hlo(
phase_0 { ROOT exp = f32[512] exponential(f32[512] parameter(0)) }
phase_1 {
  bc = f32[2,512] broadcast(f32[512] parameter(0)), dimensions={1}
  rev = f32[1024] reverse(f32[1024] reshape(bc)), dimensions={0}
  ROOT add = f32[1024] add(rev, f32[1024] parameter(1))
}
phased_comp {
  f0 = f32[512] fusion(f32[512] parameter(0)), kind=kLoop, calls=phase_0
  f1 = f32[1024] fusion(f0, f32[1024] parameter(1)), kind=kLoop, calls=phase_1
  ROOT t = (f32[1024], f32[512]) tuple(f1, f0)
}
ENTRY main {
  ROOT fusion = (f32[1024], f32[512]) fusion(f32[512] parameter(0), f32[1024] parameter(1)),
    kind=kCustom, calls=phased_comp,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       EmitMlirModule(kHlo));
  std::string mlir_str;
  llvm::raw_string_ostream(mlir_str) << *mlir_module;
  ASSERT_OK_AND_ASSIGN(bool mlir_check, RunFileCheck(mlir_str, R"(
    // CHECK-LABEL: llvm.func @fusion(
    // CHECK-SAME:    %[[ARG0:[^:]+]]: !llvm.ptr {llvm.noalias}, %[[ARG1:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[ARG2:[^:]+]]: !llvm.ptr {llvm.noalias}, %[[ARG3:[^:]+]]: !llvm.ptr)
    // CHECK-DAG:     %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:     %[[BID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK:         %[[IN:.*]] = llvm.icmp "ult" %[[BID]], %[[C4]] : i32
    // CHECK:         llvm.cond_br %[[IN]], ^[[PHASE0_CALL:bb[0-9]+]], ^[[BARRIER:bb[0-9]+]]
    // CHECK:       ^[[PHASE0_CALL]]:
    // CHECK:         llvm.call @fusion_nested_0(%[[ARG0]], %[[ARG3]], %[[BID]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         llvm.br ^[[BARRIER]]
    // CHECK:       ^[[BARRIER]]:
    // CHECK:         nvvm.cluster.arrive aligned
    // CHECK:         nvvm.cluster.wait aligned
    // CHECK:         llvm.call @fusion_nested_1(%[[ARG3]], %[[ARG1]], %[[ARG2]], %[[BID]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         llvm.return
  )"));
  EXPECT_TRUE(mlir_check);

  if (HasHopperClusterSupport()) {
    EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-5, 1e-5}));
  }
}

TEST_F(ThreadBlockSpecializedFusionTest,
       DoesNotReuseKernelAcrossHorizontalAndPhasedSchedules) {
  if (!HasHopperClusterSupport()) {
    GTEST_SKIP() << "Requires Hopper (SM90+) thread-block clusters.";
  }
  constexpr absl::string_view kHlo = R"hlo(
HloModule m

h_leaf_0 { ROOT neg = f32[512]{0} negate(f32[512]{0} parameter(0)) }
h_leaf_1 { ROOT exp = f32[512]{0} exponential(f32[512]{0} parameter(0)) }
horiz_comp {
  p0 = f32[512]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = f32[512]{0} fusion(p0), kind=kLoop, calls=h_leaf_0
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=h_leaf_1
  ROOT t = (f32[512]{0}, f32[512]{0}) tuple(f0, f1)
}

p_leaf_0 { ROOT neg = f32[512]{0} negate(f32[512]{0} parameter(0)) }
p_leaf_1 { ROOT exp = f32[512]{0} exponential(f32[512]{0} parameter(0)) }
phased_comp {
  p0 = f32[512]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = f32[512]{0} fusion(p0), kind=kLoop, calls=p_leaf_0
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=p_leaf_1
  ROOT t = (f32[512]{0}, f32[512]{0}) tuple(f0, f1)
}

ENTRY main {
  p0 = f32[512]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  horiz_fusion = (f32[512]{0}, f32[512]{0}) fusion(p0, p1), kind=kCustom,
    calls=horiz_comp,
    backend_config={"fusion_backend_config":{"kind":"__thread_block_specialized"}}
  h0 = f32[512]{0} get-tuple-element(horiz_fusion), index=0
  h1 = f32[512]{0} get-tuple-element(horiz_fusion), index=1
  phased_fusion = (f32[512]{0}, f32[512]{0}) fusion(p0, p1), kind=kCustom,
    calls=phased_comp,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
  c0 = f32[512]{0} get-tuple-element(phased_fusion), index=0
  c1 = f32[512]{0} get-tuple-element(phased_fusion), index=1
  ROOT out = (f32[512]{0}, f32[512]{0}, f32[512]{0}, f32[512]{0})
    tuple(h0, h1, c0, c1)
}
)hlo";

  EXPECT_OK(CompileAndVerifyIr(kHlo,
                               R"(
        ; CHECK-DAG: define ptx_kernel void @horiz_fusion(
        ; CHECK-DAG: define ptx_kernel void @phased_fusion(
      )",
                               /*match_optimized_ir=*/false,
                               /*run_optimization_passes=*/false));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ThreadBlockSpecializedFusionTest,
       SingleBlockPhasesOmitClusterAndSynchronizeWithBlockBarrier) {
  // All three phases run on 1 block (`f32[128]` -> `f32[64]` -> `f32[128]`)
  // with cross-warp `reverse` reads and mixed thread counts (128 vs. 64),
  // verifying that pure single-block fusions omit cluster dimensions and
  // synchronize via block barriers outside the `tid < 64` guard.
  constexpr absl::string_view kHlo = R"hlo(
phase_0 { ROOT exp = f32[128] exponential(f32[128] parameter(0)) }
phase_1 {
  p0 = f32[128] parameter(0)
  rev = f32[128] reverse(p0), dimensions={0}
  lo = f32[64] slice(p0), slice={[0:64]}
  hi = f32[64] slice(rev), slice={[0:64]}
  ROOT sum = f32[64] add(lo, hi)
}
phase_2 {
  bc = f32[2,64] broadcast(f32[64] parameter(0)), dimensions={1}
  rev = f32[128] reverse(f32[128] reshape(bc)), dimensions={0}
  ROOT sub = f32[128] subtract(rev, f32[128] parameter(1))
}
phased_comp {
  f0 = f32[128] fusion(f32[128] parameter(0)), kind=kLoop, calls=phase_0
  f1 = f32[64] fusion(f0), kind=kLoop, calls=phase_1
  f2 = f32[128] fusion(f1, f32[128] parameter(1)), kind=kLoop, calls=phase_2
  ROOT t = (f32[128], f32[64], f32[128]) tuple(f0, f1, f2)
}
ENTRY main {
  ROOT fusion = (f32[128], f32[64], f32[128]) fusion(f32[128] parameter(0), f32[128] parameter(1)),
    kind=kCustom, calls=phased_comp,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  ThreadBlockSpecializedFusion fusion_emitter(
      HloFusionAnalysis::Create(*fusion, device_info_));
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_blocks(), 1);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_threads_per_block(), 128);
  EXPECT_EQ(fusion_emitter.cluster_dim(), std::nullopt);

  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      fusion_emitter.CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                              /*buffer_assignment=*/nullptr));
  std::string mlir_str;
  llvm::raw_string_ostream(mlir_str) << *mlir_module;
  ASSERT_OK_AND_ASSIGN(bool mlir_check, RunFileCheck(mlir_str, R"(
    // CHECK-LABEL: llvm.func @fusion(
    // CHECK-SAME:    %[[ARG0:[^:]+]]: !llvm.ptr {llvm.noalias}, %[[ARG1:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[ARG2:[^:]+]]: !llvm.ptr, %[[ARG3:[^:]+]]: !llvm.ptr,
    // CHECK-SAME:    %[[ARG4:[^:]+]]: !llvm.ptr {llvm.noalias})
    // CHECK-NOT:     nvvm.cluster_dim
    // CHECK-SAME:  {
    // CHECK-NOT:     ctaid.x
    // CHECK-NOT:     nvvm.cluster
    // CHECK-DAG:     %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:     %[[C64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG:     %[[TID:.*]] = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK:         llvm.call @fusion_nested_0(%[[ARG0]], %[[ARG2]], %[[C0]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         nvvm.barrier
    // CHECK:         %[[TID_IN:.*]] = llvm.icmp "ult" %[[TID]], %[[C64]] : i32
    // CHECK:         llvm.cond_br %[[TID_IN]], ^[[PHASE1_CALL:bb[0-9]+]], ^[[BARRIER2:bb[0-9]+]]
    // CHECK:       ^[[PHASE1_CALL]]:
    // CHECK:         llvm.call @fusion_nested_1(%[[ARG2]], %[[ARG3]], %[[C0]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         llvm.br ^[[BARRIER2]]
    // CHECK:       ^[[BARRIER2]]:
    // CHECK:         nvvm.barrier
    // CHECK:         llvm.call @fusion_nested_2(%[[ARG3]], %[[ARG1]], %[[ARG4]], %[[C0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         llvm.return
    // CHECK-NOT:     nvvm.cluster
    // CHECK:       llvm.func internal @fusion_nested_0(
  )"));
  EXPECT_TRUE(mlir_check) << mlir_str;

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ThreadBlockSpecializedFusionTest,
       MixedMultiBlockAndContiguousSingleBlockPhasesShareBlockGuard) {
  // 4-phase chain (`4 blocks` -> `1 block` -> `1 block` -> `8 blocks`):
  //   * `phase_0` (`f32[512]`, 4 blocks) -> cluster barrier
  //   * `phase_1` (`f32[128]`, 1 block, 128 threads, reads all 4 blocks) and
  //     `phase_2` (`f32[64]`, 1 block, 64 threads, cross-warp `reverse`) share
  //     one `bid < 1` branch separated by a block barrier outside `tid < 64`
  //   * cluster barrier -> `phase_3` (`f32[1024]`, 8 blocks)
  constexpr absl::string_view kHlo = R"hlo(
phase_0 { ROOT neg = f32[512] negate(f32[512] parameter(0)) }
phase_1 {
  p0 = f32[512] parameter(0)
  c0 = f32[128] slice(p0), slice={[0:128]}
  c1 = f32[128] slice(p0), slice={[128:256]}
  c2 = f32[128] slice(p0), slice={[256:384]}
  c3 = f32[128] slice(p0), slice={[384:512]}
  ROOT sum = f32[128] add(add(c0, c1), add(c2, c3))
}
phase_2 {
  rev = f32[128] reverse(f32[128] parameter(0)), dimensions={0}
  ROOT hi = f32[64] slice(rev), slice={[0:64]}
}
phase_3 {
  bc = f32[16,64] broadcast(f32[64] parameter(0)), dimensions={1}
  rev = f32[1024] reverse(f32[1024] reshape(bc)), dimensions={0}
  ROOT add = f32[1024] add(rev, f32[1024] parameter(1))
}
phased_comp {
  f0 = f32[512] fusion(f32[512] parameter(0)), kind=kLoop, calls=phase_0
  f1 = f32[128] fusion(f0), kind=kLoop, calls=phase_1
  f2 = f32[64] fusion(f1), kind=kLoop, calls=phase_2
  f3 = f32[1024] fusion(f2, f32[1024] parameter(1)), kind=kLoop, calls=phase_3
  ROOT t = (f32[512], f32[128], f32[64], f32[1024]) tuple(f0, f1, f2, f3)
}
ENTRY main {
  ROOT fusion = (f32[512], f32[128], f32[64], f32[1024]) fusion(f32[512] parameter(0), f32[1024] parameter(1)),
    kind=kCustom, calls=phased_comp,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  ThreadBlockSpecializedFusion fusion_emitter(
      HloFusionAnalysis::Create(*fusion, device_info_));
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_blocks(), 8);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_threads_per_block(), 128);
  EXPECT_THAT(fusion_emitter.cluster_dim(), Optional(se::ClusterDim(8, 1, 1)));

  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      fusion_emitter.CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                              /*buffer_assignment=*/nullptr));
  std::string mlir_str;
  llvm::raw_string_ostream(mlir_str) << *mlir_module;
  ASSERT_OK_AND_ASSIGN(bool mlir_check, RunFileCheck(mlir_str, R"(
    // CHECK-LABEL: llvm.func @fusion(
    // CHECK-SAME:    %[[ARG0:[^:]+]]: !llvm.ptr {llvm.noalias}, %[[ARG1:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[ARG2:[^:]+]]: !llvm.ptr, %[[ARG3:[^:]+]]: !llvm.ptr,
    // CHECK-SAME:    %[[ARG4:[^:]+]]: !llvm.ptr, %[[ARG5:[^:]+]]: !llvm.ptr {llvm.noalias})
    // CHECK-SAME:    nvvm.cluster_dim = array<i32: 8, 1, 1>
    // CHECK-DAG:     %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:     %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:     %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:     %[[C64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG:     %[[BID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK-DAG:     %[[TID:.*]] = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK:         %[[IN0:.*]] = llvm.icmp "ult" %[[BID]], %[[C4]] : i32
    // CHECK:         llvm.cond_br %[[IN0]], ^[[P0_CALL:bb[0-9]+]], ^[[CB1:bb[0-9]+]]
    // CHECK:       ^[[P0_CALL]]:
    // CHECK:         llvm.call @fusion_nested_0(%[[ARG0]], %[[ARG2]], %[[BID]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         llvm.br ^[[CB1]]
    // CHECK:       ^[[CB1]]:
    // CHECK:         nvvm.cluster.arrive aligned
    // CHECK:         nvvm.cluster.wait aligned
    // CHECK:         %[[IN12:.*]] = llvm.icmp "ult" %[[BID]], %[[C1]] : i32
    // CHECK:         llvm.cond_br %[[IN12]], ^[[SINGLE_BLK:bb[0-9]+]], ^[[CB2:bb[0-9]+]]
    // CHECK:       ^[[SINGLE_BLK]]:
    // CHECK:         llvm.call @fusion_nested_1(%[[ARG2]], %[[ARG3]], %[[C0]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         nvvm.barrier
    // CHECK:         %[[TID_IN:.*]] = llvm.icmp "ult" %[[TID]], %[[C64]] : i32
    // CHECK:         llvm.cond_br %[[TID_IN]], ^[[P2_CALL:bb[0-9]+]], ^[[P2_MERGE:bb[0-9]+]]
    // CHECK:       ^[[P2_CALL]]:
    // CHECK:         llvm.call @fusion_nested_2(%[[ARG3]], %[[ARG4]], %[[C0]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         llvm.br ^[[P2_MERGE]]
    // CHECK:       ^[[P2_MERGE]]:
    // CHECK:         llvm.br ^[[CB2]]
    // CHECK:       ^[[CB2]]:
    // CHECK:         nvvm.cluster.arrive aligned
    // CHECK:         nvvm.cluster.wait aligned
    // CHECK:         llvm.call @fusion_nested_3(%[[ARG4]], %[[ARG1]], %[[ARG5]], %[[BID]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         llvm.return
  )"));
  EXPECT_TRUE(mlir_check) << mlir_str;

  if (HasHopperClusterSupport()) {
    EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-5, 1e-5}));
  }
}

TEST_F(ThreadBlockSpecializedFusionTest,
       SingleBlockPhasedBlockClusterPlusReductionOmitsCluster) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

c0_phase0 {
  p0 = f32[128]{0} parameter(0)
  ROOT exp = f32[128]{0} exponential(p0)
}

c0_phase1 {
  p0 = f32[128]{0} parameter(0)
  rev = f32[128]{0} reverse(p0), dimensions={0}
  ROOT hi = f32[64]{0} slice(rev), slice={[0:64]}
}

cluster_0 {
  p0 = f32[128]{0} parameter(0)
  f0 = f32[128]{0} fusion(p0), kind=kLoop, calls=c0_phase0
  f1 = f32[64]{0} fusion(f0), kind=kLoop, calls=c0_phase1
  ROOT t = (f32[128]{0}, f32[64]{0}) tuple(f0, f1)
}

nested_reduce {
  p0 = f32[24,512]{1,0} parameter(0)
  c0 = f32[] constant(0)
  ROOT red = f32[24]{0} reduce(p0, c0), dimensions={1}, to_apply=add_f32
}

horizontal_comp {
  p0 = f32[128]{0} parameter(0)
  p1 = f32[24,512]{1,0} parameter(1)
  c0 = (f32[128]{0}, f32[64]{0}) fusion(p0), kind=kCustom, calls=cluster_0,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
  c0_0 = f32[128]{0} get-tuple-element(c0), index=0
  c0_1 = f32[64]{0} get-tuple-element(c0), index=1
  f1 = f32[24]{0} fusion(p1), kind=kInput, calls=nested_reduce
  ROOT t = (f32[128]{0}, f32[64]{0}, f32[24]{0}) tuple(c0_0, c0_1, f1)
}

ENTRY main {
  p0 = f32[128]{0} parameter(0)
  p1 = f32[24,512]{1,0} parameter(1)
  ROOT fusion = (f32[128]{0}, f32[64]{0}, f32[24]{0}) fusion(p0, p1),
    kind=kCustom, calls=horizontal_comp,
    backend_config={"fusion_backend_config":{"kind":"__thread_block_specialized"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info_);
  ThreadBlockSpecializedFusion fusion_emitter(analysis);
  // cluster_0 uses 1 block x 128 threads without a cluster; nested_reduce uses
  // 3 blocks x 256 threads. Total blocks = 1 + 3 = 4 with no cluster padding.
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_blocks(), 4);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_threads_per_block(), 256);
  EXPECT_EQ(fusion_emitter.cluster_dim(), std::nullopt);

  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      fusion_emitter.CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                              /*buffer_assignment=*/nullptr));

  std::string mlir_str;
  llvm::raw_string_ostream mlir_os(mlir_str);
  mlir_os << *mlir_module;

  ASSERT_OK_AND_ASSIGN(bool mlir_check, RunFileCheck(mlir_str, R"(
    // CHECK-LABEL: llvm.func @fusion(
    // CHECK-SAME:    %[[ARG0:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[ARG1:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[OUT0:[^:]+]]: !llvm.ptr,
    // CHECK-SAME:    %[[OUT1:[^:]+]]: !llvm.ptr {llvm.noalias},
    // CHECK-SAME:    %[[OUT2:[^:]+]]: !llvm.ptr {llvm.noalias})
    // CHECK-NOT:     nvvm.cluster_dim
    // CHECK-SAME:  {
    // CHECK-NOT:     nvvm.cluster
    // CHECK-DAG:     %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:     %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:     %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:     %[[C128:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK-DAG:     %[[C64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG:     %[[BID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK-DAG:     %[[TID:.*]] = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK:         %[[IN_0:.*]] = llvm.icmp "ult" %[[BID]], %[[C1]] : i32
    // CHECK:         llvm.cond_br %[[IN_0]], ^[[THEN0:bb[0-9]+]], ^[[ELSE0:bb[0-9]+]]
    // CHECK:       ^[[THEN0]]:
    // CHECK:         %[[TID0:.*]] = llvm.icmp "ult" %[[TID]], %[[C128]] : i32
    // CHECK:         llvm.cond_br %[[TID0]], ^[[P0_CALL:bb[0-9]+]], ^[[BAR:bb[0-9]+]]
    // CHECK:       ^[[P0_CALL]]:
    // CHECK:         llvm.call @fusion_nested_0(%[[ARG0]], %[[OUT0]], %[[C0]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:         llvm.br ^[[BAR]]
    // CHECK:       ^[[BAR]]:
    // CHECK:         nvvm.barrier{{$}}
    // CHECK:         %[[TID1:.*]] = llvm.icmp "ult" %[[TID]], %[[C64]] : i32
    // CHECK:         llvm.cond_br %[[TID1]], ^[[P1_CALL:bb[0-9]+]], ^[[P1_MERGE:bb[0-9]+]]
    // CHECK:       ^[[P1_CALL]]:
    // CHECK:         llvm.call @fusion_nested_1(%[[OUT0]], %[[OUT1]], %[[C0]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK:       ^[[ELSE0]]:
    // CHECK:         %[[IN_1:.*]] = llvm.icmp "ult" %[[BID]], %[[C4]] : i32
    // CHECK:         llvm.cond_br %[[IN_1]], ^[[THEN1:bb[0-9]+]], ^[[ELSE1:bb[0-9]+]]
    // CHECK:       ^[[THEN1]]:
    // CHECK:         %[[REL_BID:.*]] = llvm.sub %[[BID]], %[[C1]] : i32
    // CHECK-NEXT:    llvm.call @fusion_nested_2(%[[ARG1]], %[[OUT2]], %[[REL_BID]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    // CHECK-NOT:     nvvm.cluster
    // CHECK:       llvm.func internal @fusion_nested_0(
  )"));
  EXPECT_TRUE(mlir_check) << mlir_str;

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(ThreadBlockSpecializedFusionTest,
       HopperAndBlackwellVectorizeSmallLoopPhases) {
  // Without small-input vectorization (`A100SXMDeviceInfo`), `s8[1024]` uses 8
  // blocks x 128 threads (`unroll_factor = 1`). On Hopper
  // (`H100SXMDeviceInfo`), 4x unrolling shrinks it to a 2-block cluster with
  // `vector<4xi8>` accesses; on Blackwell (`B200SXMDeviceInfo`), 16x unrolling
  // shrinks it to a single 64-thread block (`cluster_dim() == nullopt`) with
  // `vector<16xi8>` accesses synchronized by a block barrier.
  constexpr absl::string_view kPhasedHlo = R"hlo(
phase_0 { ROOT neg = s8[1024] negate(s8[1024] parameter(0)) }
phase_1 {
  halves = s8[2,512] reshape(s8[1024] parameter(0))
  swapped = s8[2,512] reverse(halves), dimensions={0}
  ROOT sum = s8[1024] add(s8[1024] reshape(swapped), s8[1024] parameter(1))
}
phased_comp {
  f0 = s8[1024] fusion(s8[1024] parameter(0)), kind=kLoop, calls=phase_0
  f1 = s8[1024] fusion(f0, s8[1024] parameter(1)), kind=kLoop, calls=phase_1
  ROOT t = (s8[1024], s8[1024]) tuple(f0, f1)
}
ENTRY main {
  ROOT fusion = (s8[1024], s8[1024]) fusion(s8[1024] parameter(0), s8[1024] parameter(1)),
    kind=kCustom, calls=phased_comp,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kPhasedHlo));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());

  se::DeviceDescription a100 = TestGpuDeviceInfo::A100SXMDeviceInfo();
  ThreadBlockSpecializedFusion a100_emitter(
      HloFusionAnalysis::Create(*fusion, a100));
  EXPECT_EQ(a100_emitter.launch_dimensions().num_blocks(), 8);
  EXPECT_EQ(a100_emitter.launch_dimensions().num_threads_per_block(), 128);
  EXPECT_THAT(a100_emitter.cluster_dim(), Optional(se::ClusterDim(8, 1, 1)));
  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> a100_mlir_module,
      a100_emitter.CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                            /*buffer_assignment=*/nullptr));
  std::string a100_mlir_str;
  llvm::raw_string_ostream(a100_mlir_str) << *a100_mlir_module;
  ASSERT_OK_AND_ASSIGN(bool a100_check, RunFileCheck(a100_mlir_str, R"(
    // CHECK-LABEL: llvm.func internal @fusion_nested_0(
    // CHECK-NOT:     vector<
  )"));
  EXPECT_TRUE(a100_check) << a100_mlir_str;

  se::DeviceDescription h100 = TestGpuDeviceInfo::H100SXMDeviceInfo();
  ThreadBlockSpecializedFusion h100_emitter(
      HloFusionAnalysis::Create(*fusion, h100));
  EXPECT_EQ(h100_emitter.launch_dimensions().num_blocks(), 2);
  EXPECT_EQ(h100_emitter.launch_dimensions().num_threads_per_block(), 128);
  EXPECT_THAT(h100_emitter.cluster_dim(), Optional(se::ClusterDim(2, 1, 1)));
  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> h100_mlir_module,
      h100_emitter.CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                            /*buffer_assignment=*/nullptr));
  std::string h100_mlir_str;
  llvm::raw_string_ostream(h100_mlir_str) << *h100_mlir_module;
  ASSERT_OK_AND_ASSIGN(bool h100_check, RunFileCheck(h100_mlir_str, R"(
    // CHECK-LABEL: llvm.func @fusion(
    // CHECK-SAME:    nvvm.cluster_dim = array<i32: 2, 1, 1>
    // CHECK:         nvvm.cluster.arrive aligned
    // CHECK:         nvvm.cluster.wait aligned
    // CHECK-LABEL: llvm.func internal @fusion_nested_0(
    // CHECK:         vector<4xi8>
    // CHECK-LABEL: llvm.func internal @fusion_nested_1(
    // CHECK:         vector<4xi8>
  )"));
  EXPECT_TRUE(h100_check) << h100_mlir_str;

  se::DeviceDescription b200 = TestGpuDeviceInfo::B200SXMDeviceInfo();
  ThreadBlockSpecializedFusion b200_emitter(
      HloFusionAnalysis::Create(*fusion, b200));
  EXPECT_EQ(b200_emitter.launch_dimensions().num_blocks(), 1);
  EXPECT_EQ(b200_emitter.launch_dimensions().num_threads_per_block(), 64);
  EXPECT_EQ(b200_emitter.cluster_dim(), std::nullopt);
  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> b200_mlir_module,
      b200_emitter.CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                            /*buffer_assignment=*/nullptr));
  std::string b200_mlir_str;
  llvm::raw_string_ostream(b200_mlir_str) << *b200_mlir_module;
  ASSERT_OK_AND_ASSIGN(bool b200_check, RunFileCheck(b200_mlir_str, R"(
    // CHECK-LABEL: llvm.func @fusion(
    // CHECK-NOT:     nvvm.cluster_dim
    // CHECK-SAME:  {
    // CHECK-NOT:     nvvm.cluster
    // CHECK:         llvm.call @fusion_nested_0(
    // CHECK:         nvvm.barrier
    // CHECK:         llvm.call @fusion_nested_1(
    // CHECK-LABEL: llvm.func internal @fusion_nested_0(
    // CHECK:         vector<16xi8>
    // CHECK-LABEL: llvm.func internal @fusion_nested_1(
    // CHECK:         vector<16xi8>
  )"));
  EXPECT_TRUE(b200_check) << b200_mlir_str;

  // Pure horizontal `__thread_block_specialized` fusions keep default
  // small-input unrolling disabled (`8 + 8 = 16` blocks on Blackwell).
  constexpr absl::string_view kHorizontalHlo = R"hlo(
leaf_0 { ROOT neg = s8[1024] negate(s8[1024] parameter(0)) }
leaf_1 { ROOT abs = s8[1024] abs(s8[1024] parameter(0)) }
horiz_comp {
  f0 = s8[1024] fusion(s8[1024] parameter(0)), kind=kLoop, calls=leaf_0
  f1 = s8[1024] fusion(s8[1024] parameter(1)), kind=kLoop, calls=leaf_1
  ROOT t = (s8[1024], s8[1024]) tuple(f0, f1)
}
ENTRY main {
  ROOT fusion = (s8[1024], s8[1024]) fusion(s8[1024] parameter(0), s8[1024] parameter(1)),
    kind=kCustom, calls=horiz_comp,
    backend_config={"fusion_backend_config":{"kind":"__thread_block_specialized"}}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> horiz_module,
                       ParseAndReturnVerifiedModule(kHorizontalHlo));
  const auto* horiz_fusion = Cast<HloFusionInstruction>(
      horiz_module->entry_computation()->root_instruction());
  ThreadBlockSpecializedFusion horiz_b200_emitter(
      HloFusionAnalysis::Create(*horiz_fusion, b200));
  EXPECT_EQ(horiz_b200_emitter.launch_dimensions().num_blocks(), 16);

  if (HasHopperClusterSupport()) {
    EXPECT_TRUE(RunAndCompareNoHloPasses(kPhasedHlo, ErrorSpec{0, 0}));
  }
}

}  // namespace
}  // namespace xla::gpu
