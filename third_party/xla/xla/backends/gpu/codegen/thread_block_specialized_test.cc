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
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"

namespace xla::gpu {
namespace {

class ThreadBlockSpecializedFusionTest
    : public HloPjRtInterpreterReferenceMixin<GpuPjRtCodegenTest> {
 protected:
  ThreadBlockSpecializedFusionTest() : mlir_context_(CreateMlirContext()) {
    mlir_context_->appendDialectRegistry(
        MlirKernelEmitter::GetDialectRegistry());
    mlir_context_->loadAllAvailableDialects();
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
  ROOT fusion = (f32[1024]{0}, f32[2048]{0}) fusion(p0, p1), kind=kLoop,
    calls=horizontal_comp
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

  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       fusion_emitter.CreateLLVMModule(
                           device_info_, *fusion, "fusion",
                           /*buffer_assignment=*/nullptr, *mlir_context_));

  std::string llvm_str;
  llvm::raw_string_ostream llvm_os(llvm_str);
  llvm_os << *llvm_source.module();

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
  ROOT fusion = (f32[64]{0}, f32[2048]{0}) fusion(p0, p1), kind=kLoop,
    calls=horizontal_comp
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

}  // namespace
}  // namespace xla::gpu
