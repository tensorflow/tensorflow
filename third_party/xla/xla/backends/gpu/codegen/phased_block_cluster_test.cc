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
#include "xla/backends/gpu/codegen/phased_block_cluster.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
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

using ::absl_testing::StatusIs;

class PhasedBlockClusterFusionTest
    : public HloPjRtInterpreterReferenceMixin<GpuPjRtCodegenTest> {
 protected:
  PhasedBlockClusterFusionTest() : mlir_context_(CreateMlirContext()) {
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
    return PhasedBlockClusterFusion(
               HloFusionAnalysis::Create(*fusion, device_info_))
        .CreateAndLowerMLIRModule(*mlir_context_, *fusion, "fusion",
                                  /*buffer_assignment=*/nullptr);
  }

  absl::Status CompileHlo(absl::string_view hlo) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                     ParseAndReturnVerifiedModule(hlo));
    return CompileToExecutable(std::move(module),
                               /*run_optimization_passes=*/false)
        .status();
  }

  std::unique_ptr<mlir::MLIRContext> mlir_context_;
  se::DeviceDescription device_info_ = TestGpuDeviceInfo::H100SXMDeviceInfo();
};

TEST_F(PhasedBlockClusterFusionTest, TwoPhaseProducerConsumerWithCrossCtaRead) {
  // `phase_0` uses 4 CTAs (`f32[512]`) while `phase_1` uses all 8 CTAs
  // (`f32[1024]`) and `reverse`s `phase_0`'s output across CTAs, exercising
  // both CTA range guarding and global-memory visibility across the barrier.
  constexpr absl::string_view kHlo = R"hlo(
phase_0 { ROOT neg = f32[512] negate(f32[512] parameter(0)) }
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
            HloFusionAnalysis::EmitterFusionKind::kPhasedBlockCluster);

  PhasedBlockClusterFusion fusion_emitter(analysis);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_blocks(), 8);
  EXPECT_EQ(fusion_emitter.launch_dimensions().num_threads_per_block(), 128);
  EXPECT_EQ(fusion_emitter.cluster_dim(), se::ClusterDim(8, 1, 1));

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

  if (!HasHopperClusterSupport()) {
    GTEST_SKIP() << "Execution requires Hopper (SM90+) thread-block clusters.";
  }
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(PhasedBlockClusterFusionTest,
       ParallelPhaseWithCtaSpecializationAndThreadGuardFollowedByConsumer) {
  // Concurrent sub-kernels in the same phase (general DAGs) are not supported.
  constexpr absl::string_view kHlo = R"hlo(
branch_a { ROOT exp = f32[64] exponential(f32[64] parameter(0)) }
branch_b { ROOT neg = f32[256] negate(f32[256] parameter(0)) }
merge_phase {
  a_bc = f32[4,64] broadcast(f32[64] parameter(0)), dimensions={1}
  b_rev = f32[256] reverse(f32[256] parameter(1)), dimensions={0}
  ROOT sub = f32[256] subtract(f32[256] reshape(a_bc), b_rev)
}
phased_comp {
  f0 = f32[64] fusion(f32[64] parameter(0)), kind=kLoop, calls=branch_a
  f1 = f32[256] fusion(f32[256] parameter(1)), kind=kLoop, calls=branch_b
  f2 = f32[256] fusion(f0, f1), kind=kLoop, calls=merge_phase
  ROOT t = (f32[64], f32[256], f32[256]) tuple(f0, f1, f2)
}
ENTRY main {
  ROOT fusion = (f32[64], f32[256], f32[256]) fusion(f32[64] parameter(0), f32[256] parameter(1)),
    kind=kCustom, calls=phased_comp,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
}
)hlo";
  EXPECT_THAT(EmitMlirModule(kHlo), StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(PhasedBlockClusterFusionTest,
       ThreeSequentialPhasesOversubscribedLoopAndReversedTupleOrder) {
  // Nested fusions exceeding one cluster and non-topological ROOT are rejected.
  constexpr absl::string_view kHlo = R"hlo(
phase_0 { ROOT exp = f32[4096] exponential(f32[4096] parameter(0)) }
phase_1 { ROOT rev = f32[4096] reverse(f32[4096] parameter(0)), dimensions={0} }
phase_2 { ROOT neg = f32[4096] negate(f32[4096] parameter(0)) }
phased_comp {
  f0 = f32[4096] fusion(f32[4096] parameter(0)), kind=kLoop, calls=phase_0
  f1 = f32[4096] fusion(f0), kind=kLoop, calls=phase_1
  f2 = f32[4096] fusion(f1), kind=kLoop, calls=phase_2
  ROOT t = (f32[4096], f32[4096], f32[4096]) tuple(f2, f1, f0)
}
ENTRY main {
  ROOT fusion = (f32[4096], f32[4096], f32[4096]) fusion(f32[4096] parameter(0)),
    kind=kCustom, calls=phased_comp,
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
}
)hlo";
  EXPECT_THAT(EmitMlirModule(kHlo), StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(PhasedBlockClusterFusionTest, RejectsNonLoopNestedFusion) {
  constexpr absl::string_view kHlo = R"hlo(
add_scalar { ROOT add = f32[] add(f32[] parameter(0), f32[] parameter(1)) }
reduce_phase { ROOT red = f32[32] reduce(f32[32,128] parameter(0), f32[] constant(0.0)), dimensions={1}, to_apply=add_scalar }
loop_phase { ROOT exp = f32[32] exponential(f32[32] parameter(0)) }
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
  EXPECT_THAT(EmitMlirModule(kHlo), StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(PhasedBlockClusterFusionTest,
       DoesNotOverwriteSharedOuterOperandAcrossPhases) {
  if (!HasHopperClusterSupport()) {
    GTEST_SKIP() << "Requires Hopper (SM90+) thread-block clusters.";
  }
  // `GpuAliasInfo` disables implicit output-to-operand slice sharing so phase 0
  // does not overwrite `p0` before phase 1 reads it.
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

TEST_F(PhasedBlockClusterFusionTest,
       StripsNoAliasFromMergedSliceAndRejectsUnsafeExplicitAliasing) {
  if (!HasHopperClusterSupport()) {
    GTEST_SKIP() << "Requires Hopper (SM90+) thread-block clusters.";
  }
  // Explicit `output_to_operand_aliasing` is not yet supported.
  constexpr absl::string_view kValidHlo = R"hlo(
phase_0 { ROOT neg = f32[1024] negate(f32[1024] parameter(0)) }
phase_1 { ROOT rev = f32[1024] reverse(f32[1024] parameter(0)), dimensions={0} }
phased_comp {
  f0 = f32[1024] fusion(f32[1024] parameter(0)), kind=kLoop, calls=phase_0
  f1 = f32[1024] fusion(f0), kind=kLoop, calls=phase_1
  ROOT t = (f32[1024], f32[1024]) tuple(f0, f1)
}
ENTRY main {
  ROOT f = (f32[1024], f32[1024]) fusion(f32[1024] negate(f32[1024] parameter(0))),
    kind=kCustom, calls=phased_comp, output_to_operand_aliasing={{0}: (0, {})},
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
}
)hlo";
  EXPECT_THAT(CompileHlo(kValidHlo),
              StatusIs(absl::StatusCode::kUnimplemented));

  constexpr absl::string_view kUnsafeHlo = R"hlo(
phase_0 { ROOT neg = f32[256] negate(f32[256] parameter(0)) }
phase_1 { ROOT add = f32[256] add(f32[256] parameter(0), f32[256] parameter(1)) }
phased_comp {
  p0 = f32[256] parameter(0)
  f0 = f32[256] fusion(p0), kind=kLoop, calls=phase_0
  f1 = f32[256] fusion(f0, p0), kind=kLoop, calls=phase_1
  ROOT t = (f32[256], f32[256]) tuple(f0, f1)
}
ENTRY main {
  ROOT f = (f32[256], f32[256]) fusion(f32[256] negate(f32[256] parameter(0))),
    kind=kCustom, calls=phased_comp, output_to_operand_aliasing={{0}: (0, {})},
    backend_config={"fusion_backend_config":{"kind":"__phased_block_cluster"}}
}
)hlo";
  EXPECT_THAT(CompileHlo(kUnsafeHlo),
              StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace xla::gpu
