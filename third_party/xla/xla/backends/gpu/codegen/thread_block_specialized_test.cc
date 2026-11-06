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
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
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
    : public HloPjRtInterpreterReferenceMixin<GpuPjRtCodegenTest> {
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
    // CHECK-DAG:     %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:     %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:     %[[BID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK:         %[[IN_0:.*]] = llvm.icmp "ult" %[[BID]], %[[C8]] : i32
    // CHECK:         llvm.cond_br %[[IN_0]], ^[[THEN0:bb[0-9]+]], ^[[ELSE0:bb[0-9]+]]
    // CHECK:       ^[[THEN0]]:
    // CHECK:         llvm.call @fusion_nested_0(%[[ARG0]], %[[OUT0]], %[[OUT1]], %[[BID]])
    // CHECK:       ^[[ELSE0]]:
    // CHECK:         %[[IN_1:.*]] = llvm.icmp "ult" %[[BID]], %[[C16]] : i32
    // CHECK:         llvm.cond_br %[[IN_1]], ^[[THEN1:bb[0-9]+]], ^[[ELSE1:bb[0-9]+]]
    // CHECK:       ^[[THEN1]]:
    // CHECK:         %[[REL_BID:.*]] = llvm.sub %[[BID]], %[[C8]] : i32
    // CHECK:         llvm.call @fusion_nested_1(%[[ARG1]], %[[OUT2]], %[[OUT3]], %[[REL_BID]])
    // CHECK-LABEL: llvm.func internal @fusion_nested_0(
    // CHECK:         llvm.mlir.constant(4 : i32) : i32
    // CHECK:         %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT:    llvm.icmp "ult" %{{.*}}, %[[C4]] : i32
    // CHECK:         llvm.call @fusion_nested_0_fusion_nested_0_nested_0(
    // CHECK:         nvvm.cluster.arrive aligned
    // CHECK:         nvvm.cluster.wait aligned
    // CHECK:         llvm.icmp "ult" %{{.*}}, %[[C4]] : i32
    // CHECK:         llvm.call @fusion_nested_0_fusion_nested_0_nested_1(
    // CHECK-LABEL: llvm.func internal @fusion_nested_1(
    // CHECK:         llvm.call @fusion_nested_1_fusion_nested_1_nested_0(
    // CHECK:         nvvm.cluster.arrive aligned
    // CHECK:         nvvm.cluster.wait aligned
    // CHECK:         llvm.call @fusion_nested_1_fusion_nested_1_nested_1(
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
    // CHECK-DAG:     %[[BID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK:         %[[IN_0:.*]] = llvm.icmp "ult" %[[BID]], %[[C8]] : i32
    // CHECK:         llvm.cond_br %[[IN_0]], ^[[THEN0:bb[0-9]+]], ^[[ELSE0:bb[0-9]+]]
    // CHECK:       ^[[THEN0]]:
    // CHECK:         llvm.call @fusion_nested_0(
    // CHECK:       ^[[ELSE0]]:
    // CHECK:         %[[IN_1:.*]] = llvm.icmp "ult" %[[BID]], %[[C11]] : i32
    // CHECK:         llvm.cond_br %[[IN_1]], ^[[THEN1:bb[0-9]+]], ^[[ELSE1:bb[0-9]+]]
    // CHECK:       ^[[THEN1]]:
    // CHECK:         %[[REL_BID:.*]] = llvm.sub %[[BID]], %[[C8]] : i32
    // CHECK-NOT:     llvm.icmp "ult" %[[REL_BID]]
    // CHECK:         llvm.call @fusion_nested_1({{.*}}, %[[REL_BID]])
    // CHECK-LABEL: llvm.func internal @fusion_nested_0(
    // CHECK:         llvm.mlir.constant(128 : i32) : i32
    // CHECK:         %[[C128:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK-NEXT:    llvm.icmp "ult" %{{.*}}, %[[C128]] : i32
    // CHECK:         llvm.call @fusion_nested_0_fusion_nested_0_nested_0(
    // CHECK:         nvvm.cluster.arrive aligned
    // CHECK:         nvvm.cluster.wait aligned
    // CHECK:         llvm.icmp "ult" %{{.*}}, %[[C128]] : i32
    // CHECK:         llvm.call @fusion_nested_0_fusion_nested_0_nested_1(
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
    // CHECK-DAG:     %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:     %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:     %[[C19:.*]] = llvm.mlir.constant(19 : i32) : i32
    // CHECK-DAG:     %[[C23:.*]] = llvm.mlir.constant(23 : i32) : i32
    // CHECK-DAG:     %[[BID:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK:         %[[IN_0:.*]] = llvm.icmp "ult" %[[BID]], %[[C8]] : i32
    // CHECK:         llvm.cond_br %[[IN_0]], ^[[THEN0:bb[0-9]+]], ^[[ELSE0:bb[0-9]+]]
    // CHECK:       ^[[THEN0]]:
    // CHECK:         llvm.call @fusion_nested_0(
    // CHECK:       ^[[ELSE0]]:
    // CHECK:         %[[IN_1:.*]] = llvm.icmp "ult" %[[BID]], %[[C16]] : i32
    // CHECK:         llvm.cond_br %[[IN_1]], ^[[THEN1:bb[0-9]+]], ^[[ELSE1:bb[0-9]+]]
    // CHECK:       ^[[THEN1]]:
    // CHECK:         %[[REL_BID1:.*]] = llvm.sub %[[BID]], %[[C8]] : i32
    // CHECK-NEXT:    llvm.call @fusion_nested_1({{.*}}, %[[REL_BID1]])
    // CHECK:       ^[[ELSE1]]:
    // CHECK:         %[[IN_2:.*]] = llvm.icmp "ult" %[[BID]], %[[C19]] : i32
    // CHECK:         llvm.cond_br %[[IN_2]], ^[[THEN2:bb[0-9]+]], ^[[ELSE2:bb[0-9]+]]
    // CHECK:       ^[[THEN2]]:
    // CHECK:         %[[REL_BID2:.*]] = llvm.sub %[[BID]], %[[C16]] : i32
    // CHECK-NEXT:    llvm.call @fusion_nested_2({{.*}}, %[[REL_BID2]])
    // CHECK:       ^[[ELSE2]]:
    // CHECK:         %[[IN_3:.*]] = llvm.icmp "ult" %[[BID]], %[[C23]] : i32
    // CHECK:         llvm.cond_br %[[IN_3]], ^[[THEN3:bb[0-9]+]], ^[[ELSE3:bb[0-9]+]]
    // CHECK:       ^[[THEN3]]:
    // CHECK:         %[[REL_BID3:.*]] = llvm.sub %[[BID]], %[[C19]] : i32
    // CHECK-NEXT:    llvm.call @fusion_nested_3({{.*}}, %[[REL_BID3]])
  )"));
  EXPECT_TRUE(mlir_check);

  if (HasHopperClusterSupport()) {
    EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-4, 1e-4}));
  }
}

}  // namespace
}  // namespace xla::gpu
