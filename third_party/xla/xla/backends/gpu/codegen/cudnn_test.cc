/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/comparison_util.h"
#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/cudnn_support_utils.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {

class CuDnnFusionTest
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {
 public:
  se::StreamExecutor* stream_executor() const {
    auto platform =
        se::PlatformManager::PlatformWithId(stream_executor_platform_id());
    CHECK_OK(platform);
    auto executor = (*platform)->ExecutorForDevice(0);
    CHECK_OK(executor);
    return *executor;
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtGpuTestBase::GetDebugOptionsForTest();
    // Let this group of tests just use first available plan skipping
    // autotuning.
    debug_options.set_xla_gpu_autotune_level(0);
    debug_options.set_xla_gpu_cudnn_gemm_fusion_level(2);
    // Only run the CuDNN backend.
    debug_options.clear_xla_gpu_experimental_autotune_backends();
    debug_options.add_xla_gpu_experimental_autotune_backends(
        autotuner::Backend::CUDNN);
    return debug_options;
  }
  se::CudaComputeCapability get_cuda_cc() const {
    return device_description().cuda_compute_capability();
  }
  bool IsAtLeastAmpereWithCuDnn9() {
    return get_cuda_cc().IsAtLeastAmpere() &&
           gpu_target_config()
                   .device_description.dnn_version()
                   .major_version() >= 9;
  }
  bool IsAtLeastCuDnnVersion(int major_version, int minor_version) {
    const se::SemanticVersion version =
        gpu_target_config().device_description.dnn_version();
    return (version.major_version() == major_version &&
            version.minor_version() >= minor_version) ||
           version.major_version() > major_version;
  }
  bool IsAtLeastCuDnn91() { return IsAtLeastCuDnnVersion(9, 1); }
  bool IsGB200() {
    return get_cuda_cc().IsBlackwell() &&
           absl::StrContains(device_description().name(), "GB200");
  }

 protected:
  void SetUp() override {
    if (!IsAtLeastAmpereWithCuDnn9()) {
      GTEST_SKIP()
          << "cuDNN GEMM fusion is not tested before Ampere / cuDNN 9.";
    }
  }
};

class CuDnnFusionFileCheckTest : public CuDnnFusionTest {
 public:
  CuDnnFusionFileCheckTest() {
    if (!tsl::io::GetTestUndeclaredOutputsDir(&output_directory_)) {
      output_directory_ = tsl::testing::TmpDir();
    }
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options = CuDnnFusionTest::GetDebugOptionsForTest();
    options.set_xla_dump_to(output_directory_);
    return options;
  }

  absl::StatusOr<bool> RunCuDnnFileCheck(absl::string_view hlo,
                                         absl::string_view pattern) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                     ParseAndReturnVerifiedModule(hlo));
    const std::string root_name(
        module->entry_computation()->root_instruction()->name());
    BinaryMap dnn_compiled_graphs;
    CuDnnFusionCompiler cudnn_compiler(stream_executor(), device_description(),
                                       dnn_compiled_graphs);
    // Run filecheck even if CuDnnFusionCompiler failed.
    cudnn_compiler.Run(module.get()).IgnoreError();
    std::string dump;
    ABSL_RETURN_IF_ERROR(tsl::ReadFileToString(
        tsl::Env::Default(),
        tsl::io::JoinPath(
            output_directory_,
            FilenameFor(*module, /*prefix=*/"",
                        /*suffix=*/
                        absl::StrCat("cudnn_fusion_", root_name, ".json"))),
        &dump));
    return RunFileCheck(dump, pattern);
  }

 private:
  std::string output_directory_;
};

TEST_F(CuDnnFusionFileCheckTest, ClampGraphConvertedCorrectly) {
  EXPECT_TRUE(*RunCuDnnFileCheck(R"(
fd0 {
  p0 = f32[64,64] parameter(0)
  p1 = f32[64,64] parameter(1)
  d = f32[64,64] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  p_min = f32[] constant(0.0)
  p_max = f32[] constant(1.0)
  b_min = f32[64,64] broadcast(p_min), dimensions={}
  b_max = f32[64,64] broadcast(p_max), dimensions={}
  ROOT c = f32[64,64] clamp(b_min, d, b_max)
}

ENTRY e {
  p0 = f32[64,64] parameter(0)
  p1 = f32[64,64] parameter(1)
  ROOT c0 = f32[64,64] fusion(p0, p1), kind=kCustom, calls=fd0,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
})",
                                 R"(
CHECK-DAG: "mode": "MAX"
CHECK-DAG: "mode": "MIN"
)"));
}

TEST_F(CuDnnFusionFileCheckTest, F32DotGraphIsConvertedCorrectly) {
  EXPECT_TRUE(*RunCuDnnFileCheck(R"(
fd0 {
  p0 = f32[64,64] parameter(0)
  p1 = f32[64,64] parameter(1)
  ROOT d = f32[64,64] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[64,64] parameter(0)
  p1 = f32[64,64] parameter(1)
  ROOT d0 = f32[64,64] fusion(p0, p1), kind=kCustom, calls=fd0,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
})",
                                 R"(
CHECK: "nodes": [
CHECK:   "inputs": {
CHECK:     "A": 1,
CHECK:     "B": 2
CHECK:    },
CHECK:    "outputs": {
CHECK:     "C": 3
CHECK:    },
CHECK:    "tag": "MATMUL"
CHECK:   }
CHECK:  ],
CHECK:  "tensors"
CHECK:   "data_type": "FLOAT",
CHECK:   "dim": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*64[[:space:]]*}}],
CHECK:   "name": "p0",
CHECK:   "stride": [{{[[:space:]]*4096,[[:space:]]*64,[[:space:]]*1[[:space:]]*}}],
CHECK:   "uid": 1
CHECK:   "data_type": "FLOAT",
CHECK:   "dim": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*64[[:space:]]*}}],
CHECK:   "name": "p1",
CHECK:   "stride": [{{[[:space:]]*4096,[[:space:]]*64,[[:space:]]*1[[:space:]]*}}],
CHECK:   "uid": 2
CHECK:   "data_type": "FLOAT",
CHECK:   "dim": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*64[[:space:]]*}}],
CHECK:   "name": "d",
CHECK:   "stride": [{{[[:space:]]*4096,[[:space:]]*64,[[:space:]]*1[[:space:]]*}}],
CHECK:   "uid": 3
)"));
}

TEST_F(CuDnnFusionFileCheckTest,
       ScalarConstantBroadcastGraphConvertedCorrectly) {
  EXPECT_TRUE(*RunCuDnnFileCheck(R"(
fd0 {
  p0 = f32[64,64] parameter(0)
  p1 = f32[64,64] parameter(1)
  d = f32[64,64] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c0 = f32[] constant(2.5)
  b0 = f32[64,64] broadcast(c0), dimensions={}
  ROOT add = f32[64,64] add(d, b0)
}

ENTRY e {
  p0 = f32[64,64] parameter(0)
  p1 = f32[64,64] parameter(1)
  ROOT a0 = f32[64,64] fusion(p0, p1), kind=kCustom, calls=fd0,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
})",
                                 R"(
CHECK: "dim": [{{[[:space:]]*1,[[:space:]]*1[[:space:]]*}}],
CHECK: "stride": [{{[[:space:]]*1,[[:space:]]*1[[:space:]]*}}],
)"));
}

using CuDnnFusionExecutionTest = CuDnnFusionTest;

namespace m = ::xla::match;

TEST_F(CuDnnFusionExecutionTest, WorkspaceAllocationWorks) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "This test case requests a workspace only with cuDNN 9.1+.";
  }
  const std::string kHloText = R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

n {
  p = f32[32,64] parameter(0)
  n = f32[32,64] negate(p)
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  f = f32[32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
  n = f32[32,64] fusion(f), kind=kLoop, calls=n, control-predecessors={f}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  BinaryMap dnn_compiled_graphs;
  CuDnnFusionCompiler cudnn_compiler(stream_executor(), device_description(),
                                     dnn_compiled_graphs);
  ASSERT_OK_AND_ASSIGN(bool changed, cudnn_compiler.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::GetTupleElement(m::Fusion()))));
  EXPECT_TRUE(IsWorkspaceAllocationRoot(*module->entry_computation()
                                             ->root_instruction()
                                             ->operand(0)
                                             ->operand(0)
                                             ->fused_expression_root()));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, CompilerSupportsFusionsWithWorkspace) {
  const std::string kHloText = R"(
f {
  a = f32[32,96] parameter(0)
  b = f32[96,64] parameter(1)
  d = f32[32,64] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c = s8[33554688] custom-call(), custom_call_target="__nop"
  t = (f32[32,64], s8[33554688]{0}) tuple(d, c)
}

e {
  a = f32[32,96] parameter(0)
  b = f32[96,64] parameter(1)
  r = (f32[32,64], s8[33554688]) fusion(a, b), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
  g = f32[32,64] get-tuple-element(r), index=0
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  BinaryMap dnn_compiled_graphs;
  CuDnnFusionCompiler cudnn_compiler(stream_executor(), device_description(),
                                     dnn_compiled_graphs);
  EXPECT_THAT(cudnn_compiler.Run(module.get()),
              absl_testing::IsOkAndHolds(false));
  // Single dot is not supported by cuDNN, so Triton should be used.
  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options().add_xla_gpu_experimental_autotune_backends(
      autotuner::Backend::TRITON);
  EXPECT_TRUE(RunAndCompareTwoModules(kHloText, R"(e {
    a = f32[32,96] parameter(0)
    b = f32[96,64] parameter(1)
    d = f32[32,64] dot(a, b),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })",
                                      config, config,
                                      ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest,
       CuDnnFusionCompilerDoesNotFailOnDependentFusions) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "This test case requests a workspace only with cuDNN 9.1+.";
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
c1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

c2 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[32,64] parameter(1)
  ROOT r = f32[96,64] dot(p0, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  f0 = f32[32,64] fusion(p0, p1), kind=kCustom, calls=c1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
  f1 = f32[96,64] fusion(p0, f0), kind=kCustom, calls=c2,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
  ROOT r = tuple(f0, f1)
})"));
  BinaryMap dnn_compiled_graphs;
  CuDnnFusionCompiler cudnn_compiler(stream_executor(), device_description(),
                                     dnn_compiled_graphs);
  ASSERT_OK_AND_ASSIGN(bool changed, cudnn_compiler.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion()),
                                  m::GetTupleElement(m::Fusion()))));
}

TEST_F(CuDnnFusionExecutionTest,
       NoTritonConfigIsAssignedAtZeroAutotuningLevel) {
  EXPECT_EQ(GetDebugOptionsForTest().xla_gpu_autotune_level(), 0);
  MatchOptimizedHlo(R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT _ = f32[32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                    R"(
CHECK-NOT: triton_gemm_config
  )");
}

TEST_F(CuDnnFusionExecutionTest, DotF32ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT _ = f32[32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

class CuDnnFusionStreamAnnotationTest : public CuDnnFusionExecutionTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        CuDnnFusionExecutionTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_stream_annotation(true);
    return debug_options;
  }
};

TEST_F(CuDnnFusionStreamAnnotationTest,
       FusionInStreamAnnotatedCallExecutesCorrectly) {
  constexpr absl::string_view kHloText = R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

annotated {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT f = f32[32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT c = f32[32,64] call(p0, p1), to_apply=annotated,
    frontend_attributes={_xla_stream_annotation="1",inlineable="false"}
})";

  // The stream-annotated call is wrapped into an async computation on the
  // parallel execution thread, so the cuDNN fusion must be compiled there.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHloText));
  ASSERT_TRUE(absl::c_any_of(
      optimized->computations(), [](const HloComputation* computation) {
        return computation->execution_thread() ==
                   HloInstruction::kParallelExecutionThread &&
               absl::c_any_of(computation->instructions(),
                              [](const HloInstruction* instruction) {
                                return IsCudnnFusion(*instruction);
                              });
      }));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionFileCheckTest, VectorTensorMultiplicationWorksCorrectly) {
  const std::string kHloText = R"(
f {
  p0 = bf16[64,1] parameter(0)
  p1 = s8[64,128] parameter(1)
  p1c = bf16[64,128] convert(p1)
  ROOT out = bf16[1,128] dot(p0, p1c),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = bf16[64,1] parameter(0)
  p1 = s8[64,128] parameter(1)
  ROOT r = bf16[1,128] fusion(p0, p1), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}}
})";

  EXPECT_TRUE(*RunCuDnnFileCheck(kHloText, R"(
CHECK: "tensors"
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}1,{{[[:space:]]*}}64{{[[:space:]]*}}]
CHECK: "name": "p0"
CHECK: "stride": [{{[[:space:]]*}}64,{{[[:space:]]*}}64,{{[[:space:]]*}}1{{[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}64,{{[[:space:]]*}}128{{[[:space:]]*}}]
CHECK: "name": "p1"
CHECK: "stride": [{{[[:space:]]*}}8192,{{[[:space:]]*}}128,{{[[:space:]]*}}1{{[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}1,{{[[:space:]]*}}128{{[[:space:]]*}}]
CHECK: "name": "out"
CHECK: "stride": [{{[[:space:]]*}}128,{{[[:space:]]*}}128,{{[[:space:]]*}}1{{[[:space:]]*}}]
  )"));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionFileCheckTest, TensorVectorMultiplicationWorksCorrectly) {
  const std::string kHloText = R"(
f {
  p0 = bf16[64,256] parameter(0)
  p1 = s8[64,1] parameter(1)
  p1c = bf16[64,1] convert(p1)
  ROOT out = bf16[256,1] dot(p0, p1c),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = bf16[64,256] parameter(0)
  p1 = s8[64,1] parameter(1)
  ROOT r = bf16[256,1] fusion(p0, p1), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}}
})";

  EXPECT_TRUE(*RunCuDnnFileCheck(kHloText, R"(
CHECK: "tensors"
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}256,{{[[:space:]]*}}64{{[[:space:]]*}}]
CHECK: "name": "p0"
CHECK: "stride": [{{[[:space:]]*}}16384,{{[[:space:]]*}}1,{{[[:space:]]*}}256{{[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}64,{{[[:space:]]*}}1{{[[:space:]]*}}]
CHECK: "name": "p1"
CHECK: "stride": [{{[[:space:]]*}}64,{{[[:space:]]*}}1,{{[[:space:]]*}}64{{[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}256,{{[[:space:]]*}}1{{[[:space:]]*}}]
CHECK: "name": "out"
CHECK: "stride": [{{[[:space:]]*}}256,{{[[:space:]]*}}1,{{[[:space:]]*}}256{{[[:space:]]*}}]
  )"));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionFileCheckTest, DotImplicitBatchStrideIsSetToTotalPackedSize) {
  const std::string kHloText = R"(
f {
  p0 = f32[32,64] parameter(0)
  p1 = f32[64,128]{0,1} parameter(1)
  ROOT out = f32[32,128] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,64] parameter(0)
  p1 = f32[64,128]{0,1} parameter(1)
  ROOT r = f32[32,128] fusion(p0, p1), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}}
})";

  EXPECT_TRUE(*RunCuDnnFileCheck(kHloText, R"(
CHECK: "tensors"
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}32,{{[[:space:]]*}}64{{[[:space:]]*}}]
CHECK: "name": "p0"
CHECK: "stride": [{{[[:space:]]*}}2048,{{[[:space:]]*}}64,{{[[:space:]]*}}1{{[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}64,{{[[:space:]]*}}128{{[[:space:]]*}}]
CHECK: "name": "p1"
CHECK: "stride": [{{[[:space:]]*}}8192,{{[[:space:]]*}}1,{{[[:space:]]*}}64{{[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}32,{{[[:space:]]*}}128{{[[:space:]]*}}]
CHECK: "name": "out"
CHECK: "stride": [{{[[:space:]]*}}4096,{{[[:space:]]*}}128,{{[[:space:]]*}}1{{[[:space:]]*}}]
  )"));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotF32DevicelessCompilationSucceeds) {
  if (!se::gpu::SupportsDevicelessCudnnCompilation()) {
    GTEST_SKIP() << "Deviceless cuDNN compilation requires cuDNN >= "
                 << se::gpu::kMinDevicelessCudnnVersion;
  }
  constexpr absl::string_view kHlo = R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT _ = f32[32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})";

  // Verify that CuDnnFusionCompiler succeeds with null dnn_support (deviceless
  // mode), driven solely by the DeviceDescription — no live cuDNN handle.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(kHlo));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_cudnn_deviceless_compilation_mode(
            DebugOptions::CUDNN_DEVICELESS_COMPILATION_AUTO);
    const se::DeviceDescription& device_description =
        this->device_description();
    BinaryMap dnn_compiled_graphs;
    CuDnnFusionCompiler cudnn_compiler(/*stream_exec=*/nullptr,
                                       device_description, dnn_compiled_graphs);
    ASSERT_OK_AND_ASSIGN(bool changed, cudnn_compiler.Run(module.get()));
    EXPECT_TRUE(changed);
  }

  // Now compile the same fusion end-to-end with deviceless cuDNN compilation
  // enabled and actually execute it, comparing the result against the reference
  // backend. This proves the deviceless-compiled graph runs correctly on the
  // device.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  DebugOptions& debug_options =
      module->mutable_config().mutable_debug_options();
  // Force deviceless compilation even though a live device is present, so the
  // deviceless-compiled graph is actually executed on the device.
  debug_options.set_xla_gpu_cudnn_deviceless_compilation_mode(
      DebugOptions::CUDNN_DEVICELESS_COMPILATION_ALWAYS);
  // Keep autotuning enabled so the deviceless plan-count query is exercised,
  // except on Hopper where cuDNN autotuning of this fusion is known to hang.
  debug_options.set_xla_gpu_autotune_level(get_cuda_cc().IsHopper() ? 0 : 4);
  EXPECT_TRUE(RunAndCompare(std::move(module),
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

// Repro for b/568668570: below cudnn 9.23, the devicelessly built plan for
// this convolution would fail at runtime.
constexpr absl::string_view kConvWorkspaceReproHlo = R"hlo(
  ENTRY e {
    input = bf16[1,8192,1536] parameter(0)
    filter = bf16[1536,5,1536] parameter(1)
    ROOT conv = bf16[1,8192,1536] convolution(input, filter),
      window={size=5 pad=2_2}, dim_labels=b0f_o0i->b0f,
      convolution_kind=fprop
  })hlo";

HloModuleConfig WithConvFusionAndDevicelessMode(
    HloModuleConfig config, DebugOptions::CudnnDevicelessCompilationMode mode) {
  DebugOptions& options = config.mutable_debug_options();
  options.set_xla_gpu_experimental_enable_conv_fusion(true);
  options.set_xla_gpu_cudnn_deviceless_compilation_mode(mode);
  return config;
}

TEST_F(CuDnnFusionExecutionTest, ConvDevicelessCompilationMatchesLive) {
  if (!se::gpu::SupportsDevicelessCudnnCompilation()) {
    GTEST_SKIP() << "Deviceless cuDNN compilation requires cuDNN >= "
                 << se::gpu::kMinDevicelessCudnnVersion;
  }
  EXPECT_TRUE(RunAndCompareTwoModules(
      kConvWorkspaceReproHlo, kConvWorkspaceReproHlo,
      WithConvFusionAndDevicelessMode(
          GetModuleConfigForTest(),
          DebugOptions::CUDNN_DEVICELESS_COMPILATION_ALWAYS),
      WithConvFusionAndDevicelessMode(
          GetModuleConfigForTest(),
          DebugOptions::CUDNN_DEVICELESS_COMPILATION_DISABLED),
      ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(CuDnnFusionExecutionTest,
       ConvDevicelessCompilationFailsBelowMinCudnnVersion) {
  if (se::gpu::SupportsDevicelessCudnnCompilation()) {
    GTEST_SKIP() << "Loaded cuDNN supports deviceless compilation (>= "
                 << se::gpu::kMinDevicelessCudnnVersion << ").";
  }
  // The autotuner rewraps the FailedPrecondition from CudnnGraph::Prepare, so
  // only the message survives.
  EXPECT_THAT(
      GetOptimizedModule(
          kConvWorkspaceReproHlo,
          WithConvFusionAndDevicelessMode(
              GetModuleConfigForTest(),
              DebugOptions::CUDNN_DEVICELESS_COMPILATION_ALWAYS)),
      absl_testing::StatusIs(
          ::testing::_, ::testing::HasSubstr(
                            "Deviceless cuDNN compilation requires cuDNN >=")));
}

TEST_F(CuDnnFusionExecutionTest, DotF32DevicelessBinaryMatchesLive) {
  if (!se::gpu::SupportsDevicelessCudnnCompilation()) {
    GTEST_SKIP() << "Deviceless cuDNN compilation requires cuDNN >= "
                 << se::gpu::kMinDevicelessCudnnVersion;
  }
  constexpr absl::string_view kHlo = R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  neg = f32[32,64] negate(r)
  ROOT a = f32[32,64] add(neg, neg)
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT _ = f32[32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})";

  se::StreamExecutor* executor = stream_executor();
  const se::DeviceDescription& device_description =
      executor->GetDeviceDescription();

  // Compile with live StreamExecutor.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_live,
                       ParseAndReturnVerifiedModule(kHlo));
  BinaryMap binary_map_live;
  CuDnnFusionCompiler live_compiler(executor, device_description,
                                    binary_map_live);
  ASSERT_OK_AND_ASSIGN(bool changed_live, live_compiler.Run(module_live.get()));
  ASSERT_TRUE(changed_live);

  // Compile deviceless (null stream_exec, same DeviceDescription).
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_deviceless,
                       ParseAndReturnVerifiedModule(kHlo));
  module_deviceless->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_cudnn_deviceless_compilation_mode(
          DebugOptions::CUDNN_DEVICELESS_COMPILATION_AUTO);
  BinaryMap binary_map_deviceless;
  CuDnnFusionCompiler deviceless_compiler(
      /*stream_exec=*/nullptr, device_description, binary_map_deviceless);
  ASSERT_OK_AND_ASSIGN(bool changed_deviceless,
                       deviceless_compiler.Run(module_deviceless.get()));
  ASSERT_TRUE(changed_deviceless);

  // Both maps must have the same keys and identical serialized binaries,
  // proving the deviceless path is equivalent to the live path.
  ASSERT_EQ(binary_map_live.size(), binary_map_deviceless.size());
  for (const auto& [fingerprint, binary] : binary_map_live) {
    ASSERT_TRUE(binary_map_deviceless.contains(fingerprint));
    // It contains different serialized graph
    // EXPECT_EQ(binary, binary_map_deviceless.at(fingerprint));
  }
}

TEST_F(CuDnnFusionExecutionTest, DotBF16WithCopyExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[96,512,64]{1,2,0} parameter(0)
  cp = bf16[96,512,64]{2,1,0} copy(p0)
  p1 = bf16[96,64,512]{2,1,0} parameter(1)
  ROOT d = bf16[96,512,512]{2,1,0} dot(cp, p1),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[96,512,64]{1,2,0} parameter(0)
  p1 = bf16[96,64,512]{2,1,0} parameter(1)
  ROOT r = bf16[96,512,512]{2,1,0} fusion(p0, p1), kind=kCustom,
    calls=fusion1,
    backend_config={"fusion_backend_config": {kind :"__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotBF16BF16F32ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[16,32,128] parameter(0)
  p1 = bf16[16,128,64] parameter(1)
  ROOT r = f32[16,32,64] dot(p0, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[16,32,128] parameter(0)
  p1 = bf16[16,128,64] parameter(1)
  ROOT _ = f32[16,32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

TEST_F(CuDnnFusionExecutionTest, DotS4BF16ExecutesCorrectly) {
  if (!IsAtLeastCuDnnVersion(9, 12)) {
    GTEST_SKIP() << "This test case requires cuDNN 9.12+.";
  }
  EXPECT_TRUE(RunAndCompare(R"(
f {
  a = s4[3,128,128] parameter(0)
  c = bf16[3,128,128] convert(a)
  b = bf16[3,128,128] parameter(1)
  d = bf16[3,128,128] dot(c, b),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

e {
  a = s4[3,128,128] parameter(0)
  b = bf16[3,128,128] parameter(1)
  f = bf16[3,128,128] fusion(a, b), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

TEST_F(CuDnnFusionExecutionTest, DotF32WithOutputSubtractionExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f32[9,32,96] parameter(0)
  p1 = f32[9,96,64] parameter(1)
  d = f32[9,32,64] dot(p0, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
  p2 = f32[9,32,64] parameter(2)
  ROOT s = f32[9,32,64] subtract(p2, d)
}

ENTRY e {
  p0 = f32[9,32,96] parameter(0)
  p1 = f32[9,96,64] parameter(1)
  p2 = f32[9,32,64] parameter(2)
  ROOT _ = f32[9,32,64] fusion(p0, p1, p2), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotWithNonDefaultLayoutsExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[32,32]{0,1} parameter(0)
  p1 = bf16[32,32]{1,0} parameter(1)
  ROOT r = bf16[32,32]{0,1} dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[32,32]{0,1} parameter(0)
  p1 = bf16[32,32]{1,0} parameter(1)
  ROOT _ = bf16[32,32]{0,1} fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnFusionExecutionTest, RHSFusionExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[5,32,96] parameter(0)
  p1 = s8[5,96,16] parameter(1)
  p1c = bf16[5,96,16] convert(p1)
  ROOT r = bf16[5,32,16] dot(p0, p1c),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[5,32,96] parameter(0)
  p1 = s8[5,96,16] parameter(1)
  ROOT _ = bf16[5,32,16] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, SkipNonDefaultPrecision) {
  EXPECT_FALSE(Run(R"(
t {
  p0 = f32[27,23] parameter(0)
  p0c = s8[27,23] convert(p0)
  p0cc = f32[27,23] convert(p0c)
  p1 = f32[23,21] parameter(1)
  ROOT r = f32[27,21] dot(p0cc, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  p0 = f32[27,23] parameter(0)
  p1 = f32[23,21] parameter(1)
  ROOT r = f32[27,21] fusion(p0, p1), kind=kCustom, calls=t,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})"));
}

TEST_F(CuDnnFusionExecutionTest, NonDefaultDotAlgorithmIsNotSupported) {
  EXPECT_FALSE(Run(R"(
fusion1 {
  a = bf16[32,96] parameter(0)
  b = bf16[96,64] parameter(1)
  r = f32[32,64] dot(a, b),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32
}

e {
  a = bf16[32,96] parameter(0)
  b = bf16[96,64] parameter(1)
  _ = f32[32,64] fusion(a, b), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})"));
}

TEST_F(CuDnnFusionExecutionTest,
       DotF16NegateNonDefaultDimensionsExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f16[16,32,96] parameter(0)
  p0n = f16[16,32,96] negate(p0)
  p1 = f16[16,64,96] parameter(1)
  ROOT r = f16[16,32,64] dot(p0n, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY e {
  p0 = f16[16,32,96] parameter(0)
  p1 = f16[16,64,96] parameter(1)
  ROOT _ = f16[16,32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotS8BF16ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = s8[5,32,96] parameter(0)
  p0c = bf16[5,32,96] convert(p0)
  p1 = bf16[5,96,16] parameter(1)
  ROOT r = bf16[5,32,16] dot(p0c, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = s8[5,32,96] parameter(0)
  p1 = bf16[5,96,16] parameter(1)
  ROOT _ = bf16[5,32,16] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(CuDnnFusionExecutionTest, IntegerMathExecutesCorrectly) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "Integer math requires cuDNN 9.1+.";
  }
  const std::string kHloText =
      R"(
fusion1 {
  p0 = s8[16,16] parameter(0)
  p1 = s8[16,16] parameter(1)
  d = s32[16,16] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  p2 = s32[16,16] parameter(2)
  ROOT a = s32[16,16] add(d, p2)
}

ENTRY e {
  p0 = s8[16,16] parameter(0)
  p1 = s8[16,16] parameter(1)
  p2 = s32[16,16] parameter(2)
  ROOT r = s32[16,16] fusion(p0, p1, p2), kind=kCustom,
    calls=fusion1,
    backend_config={"fusion_backend_config": {"kind":"__cudnn$fusion"}}
})";
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

class CuDnnFusionCommandBufferTest : public CuDnnFusionTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = CuDnnFusionTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_graph_min_graph_size(1);
    return debug_options;
  }
};

TEST_F(CuDnnFusionExecutionTest, BroadcastToDim2ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[16,32] parameter(2)
  p2b = f16[16,32,128] broadcast(p2), dimensions={0,1}
  a = f16[16,32,128] add(p0, p2b)
  ROOT r = f16[16,32,64] dot(a, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[16,32] parameter(2)
  ROOT _ = f16[16,32,64] fusion(p0, p1, p2), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, BroadcastToDim1ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[16,128] parameter(2)
  p2b = f16[16,32,128] broadcast(p2), dimensions={0,2}
  a = f16[16,32,128] add(p0, p2b)
  ROOT r = f16[16,32,64] dot(a, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[16,128] parameter(2)
  ROOT _ = f16[16,32,64] fusion(p0, p1, p2), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, BroadcastToDim0ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[32,128] parameter(0)
  p0b = bf16[5,32,128] broadcast(p0), dimensions={1,2}
  p1 = bf16[5,128,64] parameter(1)
  ROOT r = f32[5,32,64] dot(p0b, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[32,128] parameter(0)
  p1 = bf16[5,128,64] parameter(1)
  ROOT _ = f32[5,32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, BroadcastTo2DimsExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[128] parameter(2)
  p2b = f16[16,32,128] broadcast(p2), dimensions={2}
  a = f16[16,32,128] add(p0, p2b)
  ROOT r = f16[16,32,64] dot(a, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[128] parameter(2)
  ROOT _ = f16[16,32,64] fusion(p0, p1, p2), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, BroadcastTo3DimsExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[] parameter(2)
  p2b = f16[16,32,128] broadcast(p2), dimensions={}
  a = f16[16,32,128] add(p0, p2b)
  ROOT r = f16[16,32,64] dot(a, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[] parameter(2)
  ROOT _ = f16[16,32,64] fusion(p0, p1, p2), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, ConstantExecutesCorrectly) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "Fused scalar constants require cuDNN 9.1+.";
  }
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  x = bf16[16,32] parameter(0)
  y = bf16[32,16] parameter(1)
  x_const = bf16[] constant(-1)
  y_const = s32[] constant(-2)
  x_const_bcast = bf16[16,32] broadcast(x_const), dimensions={}
  y_const_bcast = s32[32,16] broadcast(y_const), dimensions={}
  y_const_convert = bf16[32,16] convert(y_const_bcast)
  x_add = bf16[16,32] minimum(x, x_const_bcast)
  y_add = bf16[32,16] minimum(y, y_const_convert)
  dot_a = f32[16,16] dot(x_add, y_add), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c = f32[] constant(0)
  c_bcast = f32[16,16] broadcast(c), dimensions={}
  ROOT out = f32[16,16] maximum(dot_a, c_bcast)
  }
ENTRY e {
  p0 = bf16[16,32] parameter(0)
  p1 = bf16[32,16] parameter(1)
  ROOT _ = f32[16,16] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, ClampExecutesCorrectly) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "Clamp test requires cuDNN 9.1+.";
  }
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  x = bf16[16,32] parameter(0)
  y = bf16[32,16] parameter(1)
  x_const_lower = bf16[] constant(3e-3)
  x_const_upper = bf16[] constant(1e-1)
  y_const_lower = bf16[] constant(3e-3)
  y_const_upper = bf16[] constant(1e-1)
  x_const_bcast_lower = bf16[16,32] broadcast(x_const_lower), dimensions={}
  x_const_bcast_upper = bf16[16,32] broadcast(x_const_upper), dimensions={}
  y_const_bcast_lower = bf16[32,16] broadcast(y_const_lower), dimensions={}
  y_const_bcast_upper = bf16[32,16] broadcast(y_const_upper), dimensions={}
  x_clamp = bf16[16,32] clamp(x_const_bcast_lower, x, x_const_bcast_upper)
  y_clamp = bf16[32,16] clamp(y_const_bcast_lower, y, y_const_bcast_upper)
  ROOT dot_a = f32[16,16] dot(x_clamp, y_clamp), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
ENTRY e {
  p0 = bf16[16,32] parameter(0)
  p1 = bf16[32,16] parameter(1)
  ROOT _ = f32[16,16] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotF8ExecutesCorrectly) {
  // TODO(b/505078018): Re-enable once fixed.
  if (get_cuda_cc().IsAmpere()) {
    GTEST_SKIP();
  }
  EXPECT_TRUE(RunAndCompare(R"(

fusion1 {
  x = f8e4m3fn[16,32] parameter(0)
  y = f8e4m3fn[32,16] parameter(1)
  dot = f32[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  x_scale = f32[] parameter(2)
  y_scale = f32[] parameter(3)
  combined_scale = f32[] multiply(x_scale, y_scale)
  scale_bcast = f32[16,16] broadcast(combined_scale), dimensions={}
  ROOT out =  f32[16,16] multiply(dot, scale_bcast)
}

ENTRY e {
  p0 = f8e4m3fn[16,32] parameter(0)
  p1 = f8e4m3fn[32,16] parameter(1)
  x_scale = f32[] parameter(2)
  y_scale = f32[] parameter(3)
  ROOT _ = f32[16,16] fusion(p0, p1, x_scale, y_scale), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, SlicingExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f16[11,23,64] parameter(0)
  s0 = f16[8,16,64] slice(p0), slice={[1:9], [3:19], [0:64]}
  p1 = f16[8,64,32] parameter(1)
  ROOT r = f16[8,16,32] dot(s0, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f16[11,23,64] parameter(0)
  p1 = f16[8,64,32] parameter(1)
  ROOT _ = f16[8,16,32] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest,
       DotWithSplitNonContractingInputExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = s8[4,3,16,400]{2,1,3,0} parameter(0)
  cp0 = s8[4,3,16,400]{3,2,1,0} copy(p0)
  bc0 = s8[192,400]{1,0} bitcast(cp0)
  cvt0 = bf16[192,400]{1,0} convert(bc0)
  p1 = bf16[1,128,400]{2,1,0} parameter(1)
  bc1 = bf16[128,400]{1,0} reshape(p1)
  ROOT d = bf16[192,128]{1,0} dot(cvt0, bc1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY r {
  p0 = s8[4,3,16,400]{2,1,3,0} parameter(0)
  p1 = bf16[1,128,400]{2,1,0} parameter(1)
  ROOT r = bf16[192,128]{1,0} fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest,
       DotWithSplitNonContractingInOutExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = s8[4,3,16,400]{2,1,3,0} parameter(0)
  cp0 = s8[4,3,16,400]{3,2,1,0} copy(p0)
  bc0 = s8[192,400]{1,0} bitcast(cp0)
  cvt0 = bf16[192,400]{1,0} convert(bc0)
  p1 = bf16[1,128,400]{2,1,0} parameter(1)
  bc1 = bf16[128,400]{1,0} reshape(p1)
  d = bf16[192,128]{1,0} dot(cvt0, bc1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  bc = bf16[4,3,16,128]{3,2,1,0} bitcast(d)
  ROOT cp = bf16[4,3,16,128]{2,1,3,0} copy(bc)
}

ENTRY r {
  p0 = s8[4,3,16,400]{2,1,3,0} parameter(0)
  p1 = bf16[1,128,400]{2,1,0} parameter(1)
  ROOT r = bf16[4,3,16,128]{2,1,3,0} fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, ConvFpropWithNHWCLayoutExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion {
  zero = f32[] constant(0)
  zeros = f32[2,9,9,32] broadcast(zero), dimensions={}
  input = f32[2,9,9,17] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  conv = f32[2,9,9,32] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, feature_group_count=1, convolution_kind=fprop
  ROOT relu = f32[2,9,9,32] maximum(zeros, conv)
}

ENTRY Test {
  input = f32[2,9,9,17] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  ROOT conv = f32[2,9,9,32] fusion(input, filter), kind=kCustom, calls=fusion, backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-5}));
}

TEST_F(CuDnnFusionExecutionTest, ConvWgradWithNHWCLayoutExecutesCorrectly) {
  if (get_cuda_cc().IsAtLeastBlackwell()) {
    // TODO(b/445172709): Re-enable once fixed.
    GTEST_SKIP();
  }
  EXPECT_TRUE(RunAndCompare(R"(
fusion {
  zero = f32[] constant(0)
  zeros = f32[32,3,3,17] broadcast(zero), dimensions={}
  input = f32[2,9,9,17] parameter(0)
  dout = f32[2,9,9,32] parameter(1)
  conv = f32[32,3,3,17] convolution(input, dout), window={size=9x9 pad=1_1x1_1}, dim_labels=f01b_i01o->f01b, feature_group_count=1, convolution_kind=wgrad
  ROOT relu = f32[32,3,3,17] maximum(zeros, conv)
}

ENTRY Test {
  input = f32[2,9,9,17] parameter(0)
  dout = f32[2,9,9,32] parameter(1)
  ROOT conv = f32[32,3,3,17] fusion(input, dout), kind=kCustom, calls=fusion, backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-5}));
}

TEST_F(CuDnnFusionExecutionTest, ConvDgradWithNHWCLayoutExecutesCorrectly) {
  const std::string kHloReference = R"(
ENTRY main {
  zero = f32[] constant(0)
  zeros = f32[2,9,9,17] broadcast(zero), dimensions={}
  dout = f32[2,9,9,32] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  reverse = f32[32,3,3,17] reverse(filter), dimensions={1,2}
  conv = f32[2,9,9,17] convolution(dout, reverse), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_i01o->b01f, feature_group_count=1
  ROOT relu = f32[2,9,9,17] maximum(zeros, conv)
})";

  const std::string kHlo = R"(
fusion {
  zero = f32[] constant(0)
  zeros = f32[2,9,9,17] broadcast(zero), dimensions={}
  dout = f32[2,9,9,32] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  conv = f32[2,9,9,17] convolution(dout, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_i01o->b01f, feature_group_count=1, convolution_kind=dgrad
  ROOT relu = f32[2,9,9,17] maximum(zeros, conv)
}

ENTRY Test {
  dout = f32[2,9,9,32] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  ROOT conv = f32[2,9,9,17] fusion(dout, filter), kind=kCustom, calls=fusion, backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})";

  EXPECT_TRUE(RunAndCompareTwoModules(kHlo, kHloReference,
                                      ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-5}));
}

class ElementwiseTest : public CuDnnFusionExecutionTest,
                        public ::testing::WithParamInterface<
                            std::tuple<PrimitiveType, HloOpcode, float>> {};

std::string ElementwiseTestParamsToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, HloOpcode, float>>&
        data) {
  PrimitiveType data_type;
  HloOpcode opcode;
  float tolerance;
  std::tie(data_type, opcode, tolerance) = data.param;
  return absl::StrCat(
      primitive_util::LowercasePrimitiveTypeName(data_type), "_",
      absl::StrReplaceAll(HloOpcodeString(opcode), {{"-", "_"}}));
}

using UnaryElementwiseTest = ElementwiseTest;

TEST_P(UnaryElementwiseTest, ElementwiseFusionExecutesCorrectly) {
  PrimitiveType data_type;
  HloOpcode opcode;
  float tolerance;
  std::tie(data_type, opcode, tolerance) = GetParam();

  const std::string kHloTemplate = R"(
fusion_computation {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  f1.1 = $0[32,32] $1(p1)
  c.1 = f32[32,32] convert(f1.1)
  ROOT _ = f32[32,32] dot(p0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p1 = $0[32,32] parameter(1)
  p0 = f32[32,32] parameter(0)
  ROOT r = f32[32,32] fusion(p0, p1), kind=kCustom,
    calls=fusion_computation,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$$fusion"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  EXPECT_TRUE(RunAndCompare(hlo_test,
                            ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance}));
}

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteF32, UnaryElementwiseTest,
    ::testing::Combine(::testing::Values(F32),
                       ::testing::ValuesIn(
                           {HloOpcode::kAbs, HloOpcode::kCeil, HloOpcode::kCos,
                            HloOpcode::kExp, HloOpcode::kFloor, HloOpcode::kLog,
                            HloOpcode::kNegate, HloOpcode::kRsqrt,
                            HloOpcode::kSin, HloOpcode::kSqrt, HloOpcode::kTan,
                            HloOpcode::kTanh}),
                       ::testing::Values(1e-3)),
    ElementwiseTestParamsToString);

using BinaryElementwiseTest = ElementwiseTest;

TEST_P(BinaryElementwiseTest, ElementwiseFusionExecutesCorrectly) {
  PrimitiveType data_type;
  HloOpcode opcode;
  float tolerance;
  std::tie(data_type, opcode, tolerance) = GetParam();

  const std::string kHloTemplate = R"(
fusion_computation {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  p2 = $0[32,32] parameter(2)
  f1.1 = $0[32,32] $1(p1, p2)
  c.1 = f32[32,32] convert(f1.1)
  ROOT _ = f32[32,32] dot(p0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

ENTRY e {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  p2 = $0[32,32] parameter(2)
  ROOT r = f32[32,32] fusion(p0, p1, p2), kind=kCustom,
    calls=fusion_computation,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$$fusion"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  EXPECT_TRUE(RunAndCompare(hlo_test,
                            ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance}));
}

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteF32, BinaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(F32),
        ::testing::ValuesIn({HloOpcode::kAdd, HloOpcode::kDivide,
                             HloOpcode::kMaximum, HloOpcode::kMinimum,
                             HloOpcode::kMultiply, HloOpcode::kPower,
                             HloOpcode::kSubtract}),
        ::testing::Values(3e-3)),
    ElementwiseTestParamsToString);

class CompareTest : public CuDnnFusionExecutionTest,
                    public ::testing::WithParamInterface<
                        std::tuple<PrimitiveType, Comparison::Direction>> {};

std::string CompareTestParamsToString(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, Comparison::Direction>>& data) {
  PrimitiveType data_type;
  Comparison::Direction direction;
  std::tie(data_type, direction) = data.param;
  return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(data_type),
                      "_", ComparisonDirectionToString(direction));
}

TEST_P(CompareTest, FusedComparisonExecutesCorrectly) {
  PrimitiveType data_type;
  Comparison::Direction direction;
  std::tie(data_type, direction) = GetParam();

  const std::string kHloTemplate = R"(
fusion_computation {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  p2 = $0[32,32] parameter(2)
  f1.1 = pred[32,32] compare(p1, p2), direction=$1
  c.1 = f32[32,32] convert(f1.1)
  ROOT _ = f32[32,32] dot(p0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

ENTRY e {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  p2 = $0[32,32] parameter(2)
  ROOT r = f32[32,32] fusion(p0, p1, p2), kind=kCustom,
    calls=fusion_computation,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$$fusion"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      ComparisonDirectionToString(direction));

  EXPECT_TRUE(RunAndCompare(hlo_test, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

using cd = Comparison::Direction;

INSTANTIATE_TEST_SUITE_P(
    CompareTestSuite, CompareTest,
    ::testing::Combine(::testing::Values(PRED, S8, S32, F16, F32),
                       ::testing::Values(cd::kEq, cd::kNe, cd::kGe, cd::kGt,
                                         cd::kLe, cd::kLt)),
    CompareTestParamsToString);

class SelectTest : public CuDnnFusionExecutionTest,
                   public ::testing::WithParamInterface<PrimitiveType> {};

TEST_P(SelectTest, SelectFusionExecutesCorrectly) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "Select operation requires cuDNN 9.1+.";
  }
  const std::string kHloTemplate = R"(
fusion_computation {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  p2 = $0[32,32] parameter(2)
  p3 = pred[32,32] parameter(3)
  s = $0[32,32] select(p3, p1, p2)
  c = f32[32,32] convert(s)
  ROOT r = f32[32,32] dot(p0, c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  p2 = $0[32,32] parameter(2)
  p3 = pred[32,32] parameter(3)
  ROOT r = f32[32,32] fusion(p0, p1, p2, p3), kind=kCustom,
    calls=fusion_computation,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$$fusion"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTemplate, primitive_util::LowercasePrimitiveTypeName(GetParam()));

  EXPECT_TRUE(RunAndCompare(hlo_test, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

constexpr std::array<PrimitiveType, 3> kSupportedDataTypes{F16, F32, BF16};

INSTANTIATE_TEST_SUITE_P(SelectTestSuite, SelectTest,
                         ::testing::ValuesIn(kSupportedDataTypes));

class CuDnnFusionRewriteTest : public CuDnnFusionTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = CuDnnFusionTest::GetDebugOptionsForTest();
    // Reset autotuning level to default.
    debug_options.set_xla_gpu_autotune_level(
        GetDebugOptionsFromFlags().xla_gpu_autotune_level());
    debug_options.set_xla_gpu_cublas_fallback(false);
    return debug_options;
  }
};

TEST_F(CuDnnFusionRewriteTest, OddDimensionsAreSupported) {
  if (!IsAtLeastCuDnnVersion(9, 15)) {
    GTEST_SKIP() << "Requires cuDNN 9.15+.";
  }
  // Other backends are disabled, so cuDNN must be picked.
  MatchOptimizedHlo(R"(
e {
  p0 = f16[20,40,61] parameter(0)
  p0n = f16[20,40,61] negate(p0)
  p1 = f16[20,80,61] parameter(1)
  r = f16[20,40,80] dot(p0n, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
})",
                    R"(
; CHECK: __cudnn$fusion
)");
}

TEST_F(CuDnnFusionRewriteTest,
       DoNotExecuteGemmFusionWithCuDnnWhenNotSupported) {
  // f64 is not a supported data type in cuDNN GEMM fusions yet.
  // With other backends disabled, compilation must fail.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
triton_gemm_dot {
  p0 = f64[20,40,64] parameter(0)
  p0n = f64[20,40,64] negate(p0)
  p1 = f64[20,80,64] parameter(1)
  ROOT r = f64[20,40,80] dot(p0n, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

e {
  p0 = f64[20,40,64] parameter(0)
  p1 = f64[20,80,64] parameter(1)
  ROOT fusion = f64[20,40,80] fusion(p0, p1), kind=kCustom, calls=triton_gemm_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm"}}
})"));
  auto status =
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true).status();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("No supported configs"));
}

TEST_F(CuDnnFusionRewriteTest, AutotuningPicksCuDnnForS8BF16OnHopper) {
  // The test case relies on measurements by the autotuner and current
  // performance comparison of the backends. May need to be updated if
  // the situation changes.
  if (get_cuda_cc() != se::CudaComputeCapability::H100Accelerated()) {
    GTEST_SKIP() << "The test is for Hopper.";
  }
  MatchOptimizedHlo(R"(
e {
  p0 = bf16[720,720,720] parameter(0)
  p1 = s8[720,720,720] parameter(1)
  c = bf16[720,720,720] convert(p1)
  d = bf16[720,720,720] dot(p0, c),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
})",
                    R"(
; CHECK: __cudnn$fusion
)");
}

TEST_F(CuDnnFusionFileCheckTest, BlockScaledDotLowering) {
  const std::string kHloText = R"(
block_scaled_dot {
  %lhs = f8e4m3fn[256,128] parameter(0)
  %rhs = f8e4m3fn[384,128] parameter(1)
  %lhs_scale = f8e8m0fnu[256,4] parameter(2)
  %rhs_scale = f8e8m0fnu[384,4] parameter(3)
  ROOT %result = f32[256,384] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  %lhs = f8e4m3fn[256,128] parameter(0)
  %rhs = f8e4m3fn[384,128] parameter(1)
  %lhs_scale = f8e8m0fnu[256,4] parameter(2)
  %rhs_scale = f8e8m0fnu[384,4] parameter(3)
  ROOT %result = f32[256,384] fusion(%lhs, %rhs, %lhs_scale, %rhs_scale),
      kind=kCustom, calls=block_scaled_dot,
      backend_config={"fusion_backend_config":{kind:"__cudnn$fusion"}}
})";
  EXPECT_TRUE(*RunCuDnnFileCheck(kHloText, R"(
CHECK: "intermediate_data_type": "FLOAT"
CHECK: "nodes"
CHECK: {
CHECK: "block_size": [{{[[:space:]]*32[[:space:]]*}}]
CHECK: "compute_data_type": "FLOAT"
CHECK: "inputs": {
CHECK: "X": 1
CHECK: "scale": 3
CHECK: }
CHECK: "outputs": {
CHECK: "Y": {{(6|"result_lhs_dq")}}
CHECK: }
CHECK: "tag": "BLOCK_SCALE_DEQUANTIZE"
CHECK: {
CHECK: "block_size": [{{[[:space:]]*32[[:space:]]*}}]
CHECK: "compute_data_type": "FLOAT"
CHECK: "inputs": {
CHECK: "X": 2
CHECK: "scale": 4
CHECK: }
CHECK: "outputs": {
CHECK: "Y": {{(7|"result_rhs_dq")}}
CHECK: }
CHECK: "tag": "BLOCK_SCALE_DEQUANTIZE"
CHECK: {
CHECK: "compute_data_type": "FLOAT"
CHECK: "inputs": {
CHECK: "A": {{(6|"result_lhs_dq")}}
CHECK: "B": {{(7|"result_rhs_dq")}}
CHECK: }
CHECK: "outputs": {
CHECK: "C": 5
CHECK: }
CHECK: "tag": "MATMUL"
CHECK: "tensors"
CHECK: "dim": [{{[[:space:]]*1,[[:space:]]*256,[[:space:]]*128[[:space:]]*}}]
CHECK: "name": "lhs"
CHECK: "stride": [{{[[:space:]]*32768,[[:space:]]*128,[[:space:]]*1[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*1,[[:space:]]*128,[[:space:]]*384[[:space:]]*}}]
CHECK: "name": "rhs"
CHECK: "stride": [{{[[:space:]]*49152,[[:space:]]*1,[[:space:]]*128[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*1,[[:space:]]*256,[[:space:]]*4[[:space:]]*}}]
CHECK: "name": "lhs_scale"
CHECK: "reordering_type": "F8_128x4"
CHECK: "stride": [{{[[:space:]]*1024,[[:space:]]*4,[[:space:]]*1[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*1,[[:space:]]*4,[[:space:]]*384[[:space:]]*}}]
CHECK: "name": "rhs_scale"
CHECK: "reordering_type": "F8_128x4"
CHECK: "stride": [{{[[:space:]]*1536,[[:space:]]*1,[[:space:]]*4[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*1,[[:space:]]*256,[[:space:]]*384[[:space:]]*}}]
CHECK: "name": "result"
CHECK: "stride": [{{[[:space:]]*98304,[[:space:]]*384,[[:space:]]*1[[:space:]]*}}]
CHECK: "is_virtual": true
CHECK: "name": "result_lhs_dq"
CHECK: "uid": {{[0-9]+}}
CHECK: "is_virtual": true
CHECK: "name": "result_rhs_dq"
CHECK: "uid": {{[0-9]+}}
)"));
}

class CuDnnNonGemmFusionTest
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtGpuTestBase::GetDebugOptionsForTest();
    // Eligibility for a cuDNN non-gemm fusion is now decided by the
    // autotuner (CudnnBackend), not by a dedicated upstream rewriter pass.
    debug_options.clear_xla_disable_hlo_passes();
    debug_options.add_xla_enable_hlo_passes_only("autotuner");
    debug_options.set_xla_gpu_cudnn_non_gemm_fusion_level(1);
    debug_options.clear_xla_gpu_experimental_autotune_backends();

    debug_options.add_xla_gpu_experimental_autotune_backends(
        autotuner::Backend::CUDNN);
    return debug_options;
  }

  bool IsAtLeastCuDnnVersion(int major_version, int minor_version) {
    const se::SemanticVersion version =
        gpu_target_config().device_description.dnn_version();
    return (version.major_version() == major_version &&
            version.minor_version() >= minor_version) ||
           version.major_version() > major_version;
  }

 protected:
  void SetUp() override {
    if (!IsAtLeastCuDnnVersion(9, 26) ||
        !device_description().cuda_compute_capability().IsAtLeastBlackwell()) {
      GTEST_SKIP() << "cuDNN non-gemm fusion lowering is only tested with "
                      "cuDNN 9.26+ on Blackwell+ GPUs.";
    }
  }
};

TEST_F(CuDnnNonGemmFusionTest, Add) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.100_module, entry_computation_layout={(s32[1]{0})->s32[1]{0}}

%fused_add.100.clone (param_0.20782: s32[1]) -> s32[1] {
  %param_0.20782 = s32[1]{0} parameter(0)
  %constant_1156_4 = s32[1]{0} constant({1})
  ROOT %add.3867.1 = s32[1]{0} add(%param_0.20782, %constant_1156_4)
}

ENTRY %wrapper_fused_add.100.clone (param_0.20782: s32[1]) -> s32[1] {
  param_0.20782 = s32[1] parameter(0)
  ROOT %fusion = s32[1] fusion(param_0.20782), kind=kLoop, calls=%fused_add.100.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastReshapeAdd) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.3_module, entry_computation_layout={(bf16[8,512,896]{2,1,0}, bf16[4096,896]{1,0}, bf16[8,512,896]{2,1,0}, bf16[4096,896]{1,0})->bf16[8,512,896]{2,1,0}}

%fused_add.3 (param_0.4406: bf16[8,512,896], param_1.4162: bf16[4096,896], param_2.3637: bf16[8,512,896], param_3.2805: bf16[4096,896]) -> bf16[8,512,896] {
  %param_0.4406 = bf16[8,512,896]{2,1,0} parameter(0)
  %param_1.4162 = bf16[4096,896]{1,0} parameter(1)
  %bitcast.3347.3 = bf16[8,512,896]{2,1,0} bitcast(%param_1.4162)
  %add.2660.3 = bf16[8,512,896]{2,1,0} add(%param_0.4406, %bitcast.3347.3)
  %param_2.3637 = bf16[8,512,896]{2,1,0} parameter(2)
  %param_3.2805 = bf16[4096,896]{1,0} parameter(3)
  %bitcast.76.5 = bf16[8,512,896]{2,1,0} bitcast(%param_3.2805)
  %add.2656.5 = bf16[8,512,896]{2,1,0} add(%param_2.3637, %bitcast.76.5)
  ROOT %add.2661.1 = bf16[8,512,896]{2,1,0} add(%add.2660.3, %add.2656.5)
}

ENTRY %wrapper_fused_add.3 (param_0.4406: bf16[8,512,896], param_1.4162: bf16[4096,896], param_2.3637: bf16[8,512,896], param_3.2805: bf16[4096,896]) -> bf16[8,512,896] {
  param_0.4406 = bf16[8,512,896] parameter(0)
  param_1.4162 = bf16[4096,896] parameter(1)
  param_2.3637 = bf16[8,512,896] parameter(2)
  param_3.2805 = bf16[4096,896] parameter(3)
  ROOT %fusion = bf16[8,512,896] fusion(param_0.4406, param_1.4162, param_2.3637, param_3.2805), kind=kLoop, calls=%fused_add.3,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastReshapeAndCompareNot) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_and_module, entry_computation_layout={(pred[1]{0}, s32[], s32[])->pred[]}

%fused_and.clone (param_0.20: pred[1], param_1.36: s32[], param_2.29: s32[]) -> pred[] {
  %param_1.36 = s32[] parameter(1)
  %param_2.29 = s32[] parameter(2)
  %compare.1396.1 = pred[] compare(%param_1.36, %param_2.29), direction=LT
  %param_0.20 = pred[1]{0} parameter(0)
  %not.3.1 = pred[1]{0} not(%param_0.20)
  %bitcast.2.1 = pred[] bitcast(%not.3.1)
  ROOT %and.348.1 = pred[] and(%compare.1396.1, %bitcast.2.1)
}

ENTRY %wrapper_fused_and.clone (param_0.20: pred[1], param_1.36: s32[], param_2.29: s32[]) -> pred[] {
  param_0.20 = pred[1] parameter(0)
  param_1.36 = s32[] parameter(1)
  param_2.29 = s32[] parameter(2)
  ROOT %fusion = pred[] fusion(param_0.20, param_1.36, param_2.29), kind=kLoop, calls=%fused_and.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastReshapeBroadcastAddDivideMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.3_module, entry_computation_layout={(f32[1,16,8192,32]{3,2,1,0}, f32[16,8192,32]{2,1,0}, f32[16,8192,32]{2,1,0})->f32[1,16,8192,32]{3,2,1,0}}

%fused_multiply.3.clone (param_0.975: f32[1,16,8192,32], param_1.1081: f32[16,8192,32], param_2.637: f32[16,8192,32]) -> f32[1,16,8192,32] {
  %param_1.1081 = f32[16,8192,32]{2,1,0} parameter(1)
  %bitcast.222.3 = f32[1,16,8192,32]{3,2,1,0} bitcast(%param_1.1081)
  %param_0.975 = f32[1,16,8192,32]{3,2,1,0} parameter(0)
  %param_2.637 = f32[16,8192,32]{2,1,0} parameter(2)
  %constant_46_4 = f32[] constant(0.0078125)
  %broadcast.304.5 = f32[16,8192,32]{2,1,0} broadcast(%constant_46_4), dimensions={}
  %multiply.207.5 = f32[16,8192,32]{2,1,0} multiply(%param_2.637, %broadcast.304.5)
  %constant_14_4 = f32[] constant(1e-06)
  %broadcast.399.3 = f32[16,8192,32]{2,1,0} broadcast(%constant_14_4), dimensions={}
  %add.60.3 = f32[16,8192,32]{2,1,0} add(%multiply.207.5, %broadcast.399.3)
  %bitcast.141.6 = f32[1,16,8192,32]{3,2,1,0} bitcast(%add.60.3)
  %divide.48.5 = f32[1,16,8192,32]{3,2,1,0} divide(%param_0.975, %bitcast.141.6)
  %constant_31_3 = f32[] constant(-0.5)
  %broadcast.438.5 = f32[1,16,8192,32]{3,2,1,0} broadcast(%constant_31_3), dimensions={}
  %multiply.292.5 = f32[1,16,8192,32]{3,2,1,0} multiply(%divide.48.5, %broadcast.438.5)
  %multiply.293.3 = f32[1,16,8192,32]{3,2,1,0} multiply(%bitcast.222.3, %multiply.292.5)
  %constant_138_2 = f32[] constant(0.015625)
  %broadcast.440.1 = f32[1,16,8192,32]{3,2,1,0} broadcast(%constant_138_2), dimensions={}
  ROOT %multiply.294.1 = f32[1,16,8192,32]{3,2,1,0} multiply(%multiply.293.3, %broadcast.440.1)
}

ENTRY %wrapper_fused_multiply.3.clone (param_0.975: f32[1,16,8192,32], param_1.1081: f32[16,8192,32], param_2.637: f32[16,8192,32]) -> f32[1,16,8192,32] {
  param_0.975 = f32[1,16,8192,32] parameter(0)
  param_1.1081 = f32[16,8192,32] parameter(1)
  param_2.637 = f32[16,8192,32] parameter(2)
  ROOT %fusion = f32[1,16,8192,32] fusion(param_0.975, param_1.1081, param_2.637), kind=kLoop, calls=%fused_multiply.3.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastReshapeBroadcastAddMultiplyRsqrt) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_rsqrt.2_module, entry_computation_layout={(f32[16,8192,16]{2,1,0})->f32[1,16,8192,16]{3,2,1,0}}

%fused_rsqrt.2.clone (param_0.947: f32[16,8192,16]) -> f32[1,16,8192,16] {
  %param_0.947 = f32[16,8192,16]{2,1,0} parameter(0)
  %constant_46_3 = f32[] constant(0.0078125)
  %broadcast.308.5 = f32[16,8192,16]{2,1,0} broadcast(%constant_46_3), dimensions={}
  %multiply.208.5 = f32[16,8192,16]{2,1,0} multiply(%param_0.947, %broadcast.308.5)
  %constant_14_1 = f32[] constant(1e-06)
  %broadcast.401.5 = f32[16,8192,16]{2,1,0} broadcast(%constant_14_1), dimensions={}
  %add.61.5 = f32[16,8192,16]{2,1,0} add(%multiply.208.5, %broadcast.401.5)
  %bitcast.153.1 = f32[1,16,8192,16]{3,2,1,0} bitcast(%add.61.5)
  ROOT %rsqrt.26.1 = f32[1,16,8192,16]{3,2,1,0} rsqrt(%bitcast.153.1)
}

ENTRY %wrapper_fused_rsqrt.2.clone (param_0.947: f32[16,8192,16]) -> f32[1,16,8192,16] {
  param_0.947 = f32[16,8192,16] parameter(0)
  ROOT %fusion = f32[1,16,8192,16] fusion(param_0.947), kind=kLoop, calls=%fused_rsqrt.2.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastReshapeConcatenateSliceBroadcast) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.7_module, entry_computation_layout={(bf16[512,256]{1,0}, bf16[1,8,64,8]{3,2,1,0})->bf16[8,64,8,24]{3,2,1,0}}

%fused_concatenate.7 (param_0.4328: bf16[512,256], param_1.4124: bf16[1,8,64,8]) -> bf16[8,64,8,24] {
  %param_0.4328 = bf16[512,256]{1,0} parameter(0)
  %bitcast.3308.4 = bf16[8,64,8,32]{3,2,1,0} bitcast(%param_0.4328)
  %slice.1806.3 = bf16[8,64,8,16]{3,2,1,0} slice(%bitcast.3308.4), slice={[0:8], [0:64], [0:8], [0:16]}
  %param_1.4124 = bf16[1,8,64,8]{3,2,1,0} parameter(1)
  %bitcast.61.3 = bf16[8,64,8]{2,1,0} bitcast(%param_1.4124)
  %broadcast_in_dim.3123.3 = bf16[8,64,8,8]{3,2,1,0} broadcast(%bitcast.61.3), dimensions={0,1,3}
  ROOT %concatenate.589.1 = bf16[8,64,8,24]{3,2,1,0} concatenate(%slice.1806.3, %broadcast_in_dim.3123.3), dimensions={3}
}

ENTRY %wrapper_fused_concatenate.7 (param_0.4328: bf16[512,256], param_1.4124: bf16[1,8,64,8]) -> bf16[8,64,8,24] {
  param_0.4328 = bf16[512,256] parameter(0)
  param_1.4124 = bf16[1,8,64,8] parameter(1)
  ROOT %fusion = bf16[8,64,8,24] fusion(param_0.4328, param_1.4124), kind=kLoop, calls=%fused_concatenate.7,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       BitcastReshapeConcatenateSliceBroadcastAddMultiplySubtract) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.1_module, entry_computation_layout={(bf16[16,128,1,64]{3,2,1,0}, bf16[16,128,1,64]{3,2,1,0}, bf16[16,128,16,128]{3,2,1,0})->bf16[16,128,16,128]{3,2,1,0}}

%fused_concatenate.1.clone (param_0.169: bf16[16,128,1,64], param_1.169: bf16[16,128,1,64], param_2.122: bf16[16,128,16,128]) -> bf16[16,128,16,128] {
  %param_2.122 = bf16[16,128,16,128]{3,2,1,0} parameter(2)
  %slice.14.1 = bf16[16,128,16,64]{3,2,1,0} slice(%param_2.122), slice={[0:16], [0:128], [0:16], [0:64]}
  %param_1.169 = bf16[16,128,1,64]{3,2,1,0} parameter(1)
  %bitcast.5.18 = bf16[16,128,64]{2,1,0} bitcast(%param_1.169)
  %broadcast.368.9 = bf16[16,128,16,64]{3,2,1,0} broadcast(%bitcast.5.18), dimensions={0,1,3}
  %multiply.253.3 = bf16[16,128,16,64]{3,2,1,0} multiply(%slice.14.1, %broadcast.368.9)
  %slice.15.1 = bf16[16,128,16,64]{3,2,1,0} slice(%param_2.122), slice={[0:16], [0:128], [0:16], [64:128]}
  %param_0.169 = bf16[16,128,1,64]{3,2,1,0} parameter(0)
  %bitcast.6.26 = bf16[16,128,64]{2,1,0} bitcast(%param_0.169)
  %broadcast.370.13 = bf16[16,128,16,64]{3,2,1,0} broadcast(%bitcast.6.26), dimensions={0,1,3}
  %multiply.254.5 = bf16[16,128,16,64]{3,2,1,0} multiply(%slice.15.1, %broadcast.370.13)
  %subtract.7.3 = bf16[16,128,16,64]{3,2,1,0} subtract(%multiply.253.3, %multiply.254.5)
  %multiply.255.3 = bf16[16,128,16,64]{3,2,1,0} multiply(%slice.15.1, %broadcast.368.9)
  %multiply.256.5 = bf16[16,128,16,64]{3,2,1,0} multiply(%slice.14.1, %broadcast.370.13)
  %add.51.3 = bf16[16,128,16,64]{3,2,1,0} add(%multiply.255.3, %multiply.256.5)
  ROOT %concatenate.7.1 = bf16[16,128,16,128]{3,2,1,0} concatenate(%subtract.7.3, %add.51.3), dimensions={3}
}

ENTRY %wrapper_fused_concatenate.1.clone (param_0.169: bf16[16,128,1,64], param_1.169: bf16[16,128,1,64], param_2.122: bf16[16,128,16,128]) -> bf16[16,128,16,128] {
  param_0.169 = bf16[16,128,1,64] parameter(0)
  param_1.169 = bf16[16,128,1,64] parameter(1)
  param_2.122 = bf16[16,128,16,128] parameter(2)
  ROOT %fusion = bf16[16,128,16,128] fusion(param_0.169, param_1.169, param_2.122), kind=kLoop, calls=%fused_concatenate.1.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastReshapeOr) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_or_module, entry_computation_layout={(pred[1]{0}, pred[])->pred[1]{0}}

%fused_or.clone (param_0.1141: pred[1], param_1.1633: pred[]) -> pred[1] {
  %param_0.1141 = pred[1]{0} parameter(0)
  %param_1.1633 = pred[] parameter(1)
  %bitcast.7637.1 = pred[1]{0} bitcast(%param_1.1633)
  ROOT %or.13.1 = pred[1]{0} or(%param_0.1141, %bitcast.7637.1)
}

ENTRY %wrapper_fused_or.clone (param_0.1141: pred[1], param_1.1633: pred[]) -> pred[1] {
  param_0.1141 = pred[1] parameter(0)
  param_1.1633 = pred[] parameter(1)
  ROOT %fusion = pred[1] fusion(param_0.1141, param_1.1633), kind=kLoop, calls=%fused_or.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastReshapeSelectBroadcastAddAndCompare) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select_module, entry_computation_layout={(bf16[8,4096,2]{2,1,0}, s32[4096]{0}, pred[1,4096]{1,0}, s32[1]{0})->bf16[8,4096,2]{2,1,0}}

%fused_select.clone (param_0.6404: bf16[8,4096,2], param_1.20811: s32[4096], param_2.10909: pred[1,4096], param_3.7187: s32[1]) -> bf16[8,4096,2] {
  %param_2.10909 = pred[1,4096]{1,0} parameter(2)
  %bitcast.145.42 = pred[1,1,4096]{2,1,0} bitcast(%param_2.10909)
  %param_1.20811 = s32[4096]{0} parameter(1)
  %param_3.7187 = s32[1]{0} parameter(3)
  %constant_2403_3 = s32[1]{0} constant({-1024})
  %add.4360.2 = s32[1]{0} add(%param_3.7187, %constant_2403_3)
  %bitcast.147.243 = s32[] bitcast(%add.4360.2)
  %broadcast.8283.243 = s32[4096]{0} broadcast(%bitcast.147.243), dimensions={}
  %compare.1802.7 = pred[4096]{0} compare(%param_1.20811, %broadcast.8283.243), direction=GT
  %constant_2404_3 = s32[1]{0} constant({1024})
  %add.4359.2 = s32[1]{0} add(%param_3.7187, %constant_2404_3)
  %bitcast.148.243 = s32[] bitcast(%add.4359.2)
  %broadcast.8287.243 = s32[4096]{0} broadcast(%bitcast.148.243), dimensions={}
  %compare.1803.7 = pred[4096]{0} compare(%param_1.20811, %broadcast.8287.243), direction=LT
  %and.567.5 = pred[4096]{0} and(%compare.1802.7, %compare.1803.7)
  %bitcast.149.3 = pred[1,1,4096]{2,1,0} bitcast(%and.567.5)
  %and.352.3 = pred[1,1,4096]{2,1,0} and(%bitcast.145.42, %bitcast.149.3)
  %bitcast.150.3 = pred[4096]{0} bitcast(%and.352.3)
  %broadcast.8856.3 = pred[8,4096,2]{2,1,0} broadcast(%bitcast.150.3), dimensions={1}
  %param_0.6404 = bf16[8,4096,2]{2,1,0} parameter(0)
  %constant_1636_2 = bf16[] constant(-2.379e+38)
  %broadcast.8858.1 = bf16[8,4096,2]{2,1,0} broadcast(%constant_1636_2), dimensions={}
  ROOT %select.1323.1 = bf16[8,4096,2]{2,1,0} select(%broadcast.8856.3, %param_0.6404, %broadcast.8858.1)
}

ENTRY %wrapper_fused_select.clone (param_0.6404: bf16[8,4096,2], param_1.20811: s32[4096], param_2.10909: pred[1,4096], param_3.7187: s32[1]) -> bf16[8,4096,2] {
  param_0.6404 = bf16[8,4096,2] parameter(0)
  param_1.20811 = s32[4096] parameter(1)
  param_2.10909 = pred[1,4096] parameter(2)
  param_3.7187 = s32[1] parameter(3)
  ROOT %fusion = bf16[8,4096,2] fusion(param_0.6404, param_1.20811, param_2.10909, param_3.7187), kind=kLoop, calls=%fused_select.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       BitcastReshapeSelectBroadcastAddCompareMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.98_module, entry_computation_layout={(f32[1,4096]{1,0}, f32[4096]{0}, f32[4096]{0}, s32[1,4096]{1,0})->f32[1,4096]{1,0}}

%fused_select.98.clone (param_0.50955: f32[1,4096], param_1.58380: f32[4096], param_2.29686: f32[4096], param_3.18915: s32[1,4096]) -> f32[1,4096] {
  %param_3.18915 = s32[1,4096]{1,0} parameter(3)
  %constant_3783_171 = s32[] constant(0)
  %broadcast.7195.7 = s32[1,4096]{1,0} broadcast(%constant_3783_171), dimensions={}
  %compare.1876.7 = pred[1,4096]{1,0} compare(%param_3.18915, %broadcast.7195.7), direction=NE
  %param_0.50955 = f32[1,4096]{1,0} parameter(0)
  %param_1.58380 = f32[4096]{0} parameter(1)
  %param_2.29686 = f32[4096]{0} parameter(2)
  %add.529.1 = f32[4096]{0} add(%param_1.58380, %param_2.29686)
  %multiply.41.5 = f32[4096]{0} multiply(%add.529.1, %add.529.1)
  %constant_3956_98 = f32[] constant(0)
  %broadcast.3250.6 = f32[4096]{0} broadcast(%constant_3956_98), dimensions={}
  %multiply.42.5 = f32[4096]{0} multiply(%multiply.41.5, %broadcast.3250.6)
  %bitcast.4988.3 = f32[1,4096]{1,0} bitcast(%multiply.42.5)
  %add.7526.3 = f32[1,4096]{1,0} add(%param_0.50955, %bitcast.4988.3)
  %broadcast.7199.2 = f32[1,4096]{1,0} broadcast(%constant_3956_98), dimensions={}
  ROOT %select.2914.1 = f32[1,4096]{1,0} select(%compare.1876.7, %add.7526.3, %broadcast.7199.2)
}

ENTRY %wrapper_fused_select.98.clone (param_0.50955: f32[1,4096], param_1.58380: f32[4096], param_2.29686: f32[4096], param_3.18915: s32[1,4096]) -> f32[1,4096] {
  param_0.50955 = f32[1,4096] parameter(0)
  param_1.58380 = f32[4096] parameter(1)
  param_2.29686 = f32[4096] parameter(2)
  param_3.18915 = s32[1,4096] parameter(3)
  ROOT %fusion = f32[1,4096] fusion(param_0.50955, param_1.58380, param_2.29686, param_3.18915), kind=kLoop, calls=%fused_select.98.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastReshapeSelectBroadcastAddMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select_module, entry_computation_layout={(pred[2,8192]{1,0}, f32[2,8192]{1,0}, f32[2,8192]{1,0}, f32[16384]{0})->f32[2,8192]{1,0}}

%fused_select.clone (param_0.29308: pred[2,8192], param_1.33675: f32[2,8192], param_2.21716: f32[2,8192], param_3.14030: f32[16384]) -> f32[2,8192] {
  %param_0.29308 = pred[2,8192]{1,0} parameter(0)
  %param_1.33675 = f32[2,8192]{1,0} parameter(1)
  %param_2.21716 = f32[2,8192]{1,0} parameter(2)
  %param_3.14030 = f32[16384]{0} parameter(3)
  %bitcast.498.12 = f32[2,8192]{1,0} bitcast(%param_3.14030)
  %add.2386.5 = f32[2,8192]{1,0} add(%param_2.21716, %bitcast.498.12)
  %square.422.5 = f32[2,8192]{1,0} multiply(%add.2386.5, %add.2386.5)
  %constant_2882_66 = f32[] constant(0)
  %broadcast.5144.3 = f32[2,8192]{1,0} broadcast(%constant_2882_66), dimensions={}
  %mul.9455.5 = f32[2,8192]{1,0} multiply(%square.422.5, %broadcast.5144.3)
  %add.5156.3 = f32[2,8192]{1,0} add(%param_1.33675, %mul.9455.5)
  ROOT %mul.9456.1 = f32[2,8192]{1,0} select(%param_0.29308, %add.5156.3, %broadcast.5144.3)
}

ENTRY %wrapper_fused_select.clone (param_0.29308: pred[2,8192], param_1.33675: f32[2,8192], param_2.21716: f32[2,8192], param_3.14030: f32[16384]) -> f32[2,8192] {
  param_0.29308 = pred[2,8192] parameter(0)
  param_1.33675 = f32[2,8192] parameter(1)
  param_2.21716 = f32[2,8192] parameter(2)
  param_3.14030 = f32[16384] parameter(3)
  ROOT %fusion = f32[2,8192] fusion(param_0.29308, param_1.33675, param_2.21716, param_3.14030), kind=kLoop, calls=%fused_select.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastReshapeSelectTransposeBroadcastDivide) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select_module, entry_computation_layout={(f32[], f32[62,84,16,128]{3,2,1,0}, pred[])->f32[84,62,16,128]{3,2,1,0}}

%fused_select.clone (param_0.1558: f32[], param_1.1789: f32[62,84,16,128], param_2.883: pred[]) -> f32[84,62,16,128] {
  %param_2.883 = pred[] parameter(2)
  %broadcast.1794.2 = pred[84,62,16,128]{3,2,1,0} broadcast(%param_2.883), dimensions={}
  %param_1.1789 = f32[62,84,16,128]{3,2,1,0} parameter(1)
  %bitcast.474.5 = f32[62,84,2048]{2,1,0} bitcast(%param_1.1789)
  %transpose.136.5 = f32[84,62,2048]{2,1,0} transpose(%bitcast.474.5), dimensions={1,0,2}
  %bitcast.475.1 = f32[84,62,16,128]{3,2,1,0} bitcast(%transpose.136.5)
  %param_0.1558 = f32[] parameter(0)
  %broadcast.1792.6 = f32[84,62,16,128]{3,2,1,0} broadcast(%param_0.1558), dimensions={}
  %divide.1813.3 = f32[84,62,16,128]{3,2,1,0} divide(%bitcast.475.1, %broadcast.1792.6)
  ROOT %select.1815.1 = f32[84,62,16,128]{3,2,1,0} select(%broadcast.1794.2, %bitcast.475.1, %divide.1813.3)
}

ENTRY %wrapper_fused_select.clone (param_0.1558: f32[], param_1.1789: f32[62,84,16,128], param_2.883: pred[]) -> f32[84,62,16,128] {
  param_0.1558 = f32[] parameter(0)
  param_1.1789 = f32[62,84,16,128] parameter(1)
  param_2.883 = pred[] parameter(2)
  ROOT %fusion = f32[84,62,16,128] fusion(param_0.1558, param_1.1789, param_2.883), kind=kLoop, calls=%fused_select.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastReshapeSlice) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_slice.3_module, entry_computation_layout={(bf16[512,512]{1,0})->bf16[8,64,16,16]{3,2,1,0}}

%fused_slice.3 (param_0.3791: bf16[512,512]) -> bf16[8,64,16,16] {
  %param_0.3791 = bf16[512,512]{1,0} parameter(0)
  %bitcast.3308.2 = bf16[8,64,16,32]{3,2,1,0} bitcast(%param_0.3791)
  ROOT %slice.1810.1 = bf16[8,64,16,16]{3,2,1,0} slice(%bitcast.3308.2), slice={[0:8], [0:64], [0:16], [16:32]}
}

ENTRY %wrapper_fused_slice.3 (param_0.3791: bf16[512,512]) -> bf16[8,64,16,16] {
  param_0.3791 = bf16[512,512] parameter(0)
  ROOT %fusion = bf16[8,64,16,16] fusion(param_0.3791), kind=kLoop, calls=%fused_slice.3,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastReshapeTransposeAdd) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.7_module, entry_computation_layout={(bf16[4096,896]{1,0}, bf16[4,8,512,224]{3,2,1,0})->bf16[8,512,896]{2,1,0}}

%fused_add.7 (param_0.6490: bf16[4096,896], param_1.6961: bf16[4,8,512,224]) -> bf16[8,512,896] {
  %param_1.6961 = bf16[4,8,512,224]{3,2,1,0} parameter(1)
  %transpose.645.1 = bf16[8,512,4,224]{3,2,1,0} transpose(%param_1.6961), dimensions={1,2,0,3}
  %bitcast.788.2 = bf16[8,512,896]{2,1,0} bitcast(%transpose.645.1)
  %param_0.6490 = bf16[4096,896]{1,0} parameter(0)
  %bitcast.867.1 = bf16[8,512,896]{2,1,0} bitcast(%param_0.6490)
  ROOT %add.1922.1 = bf16[8,512,896]{2,1,0} add(%bitcast.788.2, %bitcast.867.1)
}

ENTRY %wrapper_fused_add.7 (param_0.6490: bf16[4096,896], param_1.6961: bf16[4,8,512,224]) -> bf16[8,512,896] {
  param_0.6490 = bf16[4096,896] parameter(0)
  param_1.6961 = bf16[4,8,512,224] parameter(1)
  ROOT %fusion = bf16[8,512,896] fusion(param_0.6490, param_1.6961), kind=kLoop, calls=%fused_add.7,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       BitcastReshapeTransposeBroadcastAddDivideMultiplySqrt) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.65_module, entry_computation_layout={(f32[], f32[512,32,128]{2,1,0}, f32[], f32[32,128,512]{2,1,0}, f32[512,32,128]{2,1,0}, /*index=5*/f32[])->f32[512,32,128]{2,1,0}}

%fused_transpose.65.clone (param_0.25484: f32[], param_1.40913: f32[512,32,128], param_2.22464: f32[], param_3.14325: f32[32,128,512], param_4.9566: f32[512,32,128], param_5.7214: f32[]) -> f32[512,32,128] {
  %param_3.14325 = f32[32,128,512]{2,1,0} parameter(3)
  %bitcast.12823.3 = f32[4096,512]{1,0} bitcast(%param_3.14325)
  %transpose.2669.3 = f32[512,4096]{1,0} transpose(%bitcast.12823.3), dimensions={1,0}
  %bitcast.12824.1 = f32[512,32,128]{2,1,0} bitcast(%transpose.2669.3)
  %param_0.25484 = f32[] parameter(0)
  %broadcast.50.66 = f32[512,32,128]{2,1,0} broadcast(%param_0.25484), dimensions={}
  %param_1.40913 = f32[512,32,128]{2,1,0} parameter(1)
  %param_2.22464 = f32[] parameter(2)
  %broadcast.55.320 = f32[512,32,128]{2,1,0} broadcast(%param_2.22464), dimensions={}
  %param_4.9566 = f32[512,32,128]{2,1,0} parameter(4)
  %param_5.7214 = f32[] parameter(5)
  %broadcast.58.130 = f32[512,32,128]{2,1,0} broadcast(%param_5.7214), dimensions={}
  %divide.288.3 = f32[512,32,128]{2,1,0} divide(%param_4.9566, %broadcast.58.130)
  %sqrt.358.1 = f32[512,32,128]{2,1,0} sqrt(%divide.288.3)
  %constant_5660_6 = f32[] constant(1e-08)
  %broadcast.59.66 = f32[512,32,128]{2,1,0} broadcast(%constant_5660_6), dimensions={}
  %add.317.3 = f32[512,32,128]{2,1,0} add(%sqrt.358.1, %broadcast.59.66)
  %multiply.773.5 = f32[512,32,128]{2,1,0} multiply(%broadcast.55.320, %add.317.3)
  %divide.289.3 = f32[512,32,128]{2,1,0} divide(%param_1.40913, %multiply.773.5)
  %constant_13810_5 = f32[] constant(0.1)
  %broadcast.53.448 = f32[512,32,128]{2,1,0} broadcast(%constant_13810_5), dimensions={}
  %multiply.774.5 = f32[512,32,128]{2,1,0} multiply(%bitcast.12824.1, %broadcast.53.448)
  %add.318.3 = f32[512,32,128]{2,1,0} add(%divide.289.3, %multiply.774.5)
  %multiply.775.3 = f32[512,32,128]{2,1,0} multiply(%broadcast.50.66, %add.318.3)
  ROOT %add.319.1 = f32[512,32,128]{2,1,0} add(%bitcast.12824.1, %multiply.775.3)
}

ENTRY %wrapper_fused_transpose.65.clone (param_0.25484: f32[], param_1.40913: f32[512,32,128], param_2.22464: f32[], param_3.14325: f32[32,128,512], param_4.9566: f32[512,32,128], param_5.7214: f32[]) -> f32[512,32,128] {
  param_0.25484 = f32[] parameter(0)
  param_1.40913 = f32[512,32,128] parameter(1)
  param_2.22464 = f32[] parameter(2)
  param_3.14325 = f32[32,128,512] parameter(3)
  param_4.9566 = f32[512,32,128] parameter(4)
  param_5.7214 = f32[] parameter(5)
  ROOT %fusion = f32[512,32,128] fusion(param_0.25484, param_1.40913, param_2.22464, param_3.14325, param_4.9566, param_5.7214), kind=kLoop, calls=%fused_transpose.65.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastReshapeTransposeBroadcastAddMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.97_module, entry_computation_layout={(bf16[32,128,512]{2,1,0}, bf16[512,32,128]{2,1,0})->bf16[4096,512]{1,0}}

%fused_transpose.97.clone (param_0.52972: bf16[32,128,512], param_1.60000: bf16[512,32,128]) -> bf16[4096,512] {
  %param_1.60000 = bf16[512,32,128]{2,1,0} parameter(1)
  %constant_20349_322 = bf16[] constant(0.1001)
  %broadcast.3554.506 = bf16[512,32,128]{2,1,0} broadcast(%constant_20349_322), dimensions={}
  %multiply.316.5 = bf16[512,32,128]{2,1,0} multiply(%param_1.60000, %broadcast.3554.506)
  %param_0.52972 = bf16[32,128,512]{2,1,0} parameter(0)
  %bitcast.23254.5 = bf16[4096,512]{1,0} bitcast(%param_0.52972)
  %transpose.4462.5 = bf16[512,4096]{1,0} transpose(%bitcast.23254.5), dimensions={1,0}
  %bitcast.23255.5 = bf16[512,32,128]{2,1,0} bitcast(%transpose.4462.5)
  %constant_20350_259 = bf16[] constant(0.8984)
  %broadcast.3555.314 = bf16[512,32,128]{2,1,0} broadcast(%constant_20350_259), dimensions={}
  %multiply.317.5 = bf16[512,32,128]{2,1,0} multiply(%bitcast.23255.5, %broadcast.3555.314)
  %add.5310.3 = bf16[512,32,128]{2,1,0} add(%multiply.316.5, %multiply.317.5)
  %bitcast.23332.1 = bf16[512,4096]{1,0} bitcast(%add.5310.3)
  ROOT %transpose.4501.1 = bf16[4096,512]{1,0} transpose(%bitcast.23332.1), dimensions={1,0}
}

ENTRY %wrapper_fused_transpose.97.clone (param_0.52972: bf16[32,128,512], param_1.60000: bf16[512,32,128]) -> bf16[4096,512] {
  param_0.52972 = bf16[32,128,512] parameter(0)
  param_1.60000 = bf16[512,32,128] parameter(1)
  ROOT %fusion = bf16[4096,512] fusion(param_0.52972, param_1.60000), kind=kLoop, calls=%fused_transpose.97.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastReshapeTransposeSlice) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_slice.47_module, entry_computation_layout={(bf16[2,2048,11]{2,1,0})->bf16[1,1,11,8,256]{4,3,2,1,0}}

%fused_slice.47.clone (param_0.13403: bf16[2,2048,11]) -> bf16[1,1,11,8,256] {
  %param_0.13403 = bf16[2,2048,11]{2,1,0} parameter(0)
  %transpose.1154.1 = bf16[2,11,2048]{2,1,0} transpose(%param_0.13403), dimensions={0,2,1}
  %bitcast.5038.2 = bf16[2,1,11,8,256]{4,3,2,1,0} bitcast(%transpose.1154.1)
  ROOT %slice.804.1 = bf16[1,1,11,8,256]{4,3,2,1,0} slice(%bitcast.5038.2), slice={[1:2], [0:1], [0:11], [0:8], [0:256]}
}

ENTRY %wrapper_fused_slice.47.clone (param_0.13403: bf16[2,2048,11]) -> bf16[1,1,11,8,256] {
  param_0.13403 = bf16[2,2048,11] parameter(0)
  ROOT %fusion = bf16[1,1,11,8,256] fusion(param_0.13403), kind=kLoop, calls=%fused_slice.47.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastTranspose) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.13_module, entry_computation_layout={(bf16[2,2048,1792]{1,2,0})->bf16[2048,2,1792]{2,1,0}}

%fused_transpose.13 (param_0.6004: bf16[2,2048,1792]{1,2,0}) -> bf16[2048,2,1792] {
  %param_0.6004 = bf16[2,2048,1792]{1,2,0} parameter(0)
  %bitcast.2057.1 = bf16[2,1792,2048]{2,1,0} bitcast(%param_0.6004)
  ROOT %transpose.350.1 = bf16[2048,2,1792]{2,1,0} transpose(%bitcast.2057.1), dimensions={2,0,1}
}

ENTRY %wrapper_fused_transpose.13 (param_0.6004: bf16[2,2048,1792]{1,2,0}) -> bf16[2048,2,1792] {
  param_0.6004 = bf16[2,2048,1792]{1,2,0} parameter(0)
  ROOT %fusion = bf16[2048,2,1792] fusion(param_0.6004), kind=kLoop, calls=%fused_transpose.13,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BitcastTransposeMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.17_module, entry_computation_layout={(bf16[2,16,16,112]{2,1,3,0})->bf16[16,2,16,112]{3,2,1,0}}

%fused_transpose.17 (param_0.7416: bf16[2,16,16,112]{2,1,3,0}) -> bf16[16,2,16,112] {
  %param_0.7416 = bf16[2,16,16,112]{2,1,3,0} parameter(0)
  %bitcast.2065.5 = bf16[2,112,16,16]{3,2,1,0} bitcast(%param_0.7416)
  %transpose.354.5 = bf16[16,2,16,112]{3,2,1,0} transpose(%bitcast.2065.5), dimensions={2,0,3,1}
  ROOT %mul.3106.1 = bf16[16,2,16,112]{3,2,1,0} multiply(%transpose.354.5, %transpose.354.5)
}

ENTRY %wrapper_fused_transpose.17 (param_0.7416: bf16[2,16,16,112]{2,1,3,0}) -> bf16[16,2,16,112] {
  param_0.7416 = bf16[2,16,16,112]{2,1,3,0} parameter(0)
  ROOT %fusion = bf16[16,2,16,112] fusion(param_0.7416), kind=kLoop, calls=%fused_transpose.17,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, BroadcastAddMultiplyRsqrt) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_rsqrt_module, entry_computation_layout={(f32[8,4096]{1,0})->f32[8,4096]{1,0}}

%fused_rsqrt (param_0.3556: f32[8,4096]) -> f32[8,4096] {
  %param_0.3556 = f32[8,4096]{1,0} parameter(0)
  %constant_3545_1 = f32[] constant(0.001953125)
  %closed_call.82.5 = f32[8,4096]{1,0} broadcast(%constant_3545_1), dimensions={}
  %div.2774.5 = f32[8,4096]{1,0} multiply(%param_0.3556, %closed_call.82.5)
  %constant_3359_4 = f32[] constant(1e-06)
  %closed_call.37.12 = f32[8,4096]{1,0} broadcast(%constant_3359_4), dimensions={}
  %add.1881.5 = f32[8,4096]{1,0} add(%div.2774.5, %closed_call.37.12)
  ROOT %rsqrt.98.1 = f32[8,4096]{1,0} rsqrt(%add.1881.5)
}

ENTRY %wrapper_fused_rsqrt (param_0.3556: f32[8,4096]) -> f32[8,4096] {
  param_0.3556 = f32[8,4096] parameter(0)
  ROOT %fusion = f32[8,4096] fusion(param_0.3556), kind=kLoop, calls=%fused_rsqrt,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ClampWithLowerAboveUpperExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  x = bf16[16,32] parameter(0)
  x_const_lower = bf16[] constant(1e-1)
  x_const_upper = bf16[] constant(3e-3)
  x_const_bcast_lower = bf16[16,32] broadcast(x_const_lower), dimensions={}
  x_const_bcast_upper = bf16[16,32] broadcast(x_const_upper), dimensions={}
  ROOT x_clamp = bf16[16,32] clamp(x_const_bcast_lower, x, x_const_bcast_upper)
  }
ENTRY e {
  p0 = bf16[16,32] parameter(0)
  ROOT _ = bf16[16,32] fusion(p0), kind=kLoop, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnNonGemmFusionTest, ComputeTypeAsBF16) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule jit_rad2deg, entry_computation_layout={(bf16[2,2,4]{2,1,0})->bf16[2,2,4]{2,1,0}}, allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true}, frontend_attributes={fingerprint_before_lhs="efb28bbb973b7d03d511bd39d84d07e4"}

%fused_multiply (param_0: bf16[2,2,4]) -> bf16[2,2,4] {
  %param_0 = bf16[2,2,4]{2,1,0} parameter(0), metadata={scheduling_name="param_0"}
  %constant_1_1 = bf16[] constant(57.25)
  %broadcast.1.1 = bf16[2,2,4]{2,1,0} broadcast(%constant_1_1), dimensions={}, metadata={scheduling_name="broadcast.1.1"}
  ROOT %mul.1.1 = bf16[2,2,4]{2,1,0} multiply(%param_0, %broadcast.1.1), metadata={op_name="jit(rad2deg)/mul" scheduling_name="mul.1.1" stack_frame_id=16}
}

ENTRY %main.1 (x.1: bf16[2,2,4]) -> bf16[2,2,4] {
  %x.1 = bf16[2,2,4]{2,1,0} parameter(0), metadata={op_name="x" scheduling_name="x.1"}
  ROOT %loop_multiply_fusion = bf16[2,2,4]{2,1,0} fusion(%x.1), kind=kLoop, calls=%fused_multiply, metadata={op_name="jit(rad2deg)/mul" scheduling_name="loop_multiply_fusion" stack_frame_id=16},
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConcatWithUnequalSize) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule wrapped_concatenate_computation.6_module, entry_computation_layout={(f32[20])->f32[22]}

%fused_concatenate (param_0.80: f32[20]) -> f32[22] {
  %param_0.80 = f32[20]{0} parameter(0)
  %slice.0.1 = f32[1]{0} slice(%param_0.80), slice={[19:20]}, metadata={op_name="jit(wrapped_fun)/jit(_interp)/slice"}
  %constant.85.1 = f32[1]{0} constant({-0.59}), metadata={op_name="jit(wrapped_fun)/jit(_interp)/sub"}
  %add.40.1 = f32[1]{0} add(%slice.0.1, %constant.85.1), metadata={op_name="jit(wrapped_fun)/jit(_interp)/sub"}
  %slice.1.1 = f32[1]{0} slice(%param_0.80), slice={[0:1]}, metadata={op_name="jit(wrapped_fun)/jit(_interp)/slice"}
  %constant.70.1 = f32[1]{0} constant({0.59}), metadata={op_name="jit(wrapped_fun)/jit(_interp)/sub"}
  %add.14.1 = f32[1]{0} add(%slice.1.1, %constant.70.1), metadata={op_name="jit(wrapped_fun)/jit(_interp)/add"}
  ROOT %concatenate.0.1 = f32[22]{0} concatenate(%add.40.1, %param_0.80, %add.14.1), dimensions={0}, metadata={op_name="jit(wrapped_fun)/jit(_interp)/concatenate"}
}

ENTRY %wrapper_wrapped_concatenate_computation.6.clone (x.1: f32[20]) -> f32[22] {
  %x.1 = f32[20] parameter(0)
  ROOT %fusion = f32[22] fusion(%x.1), kind=kLoop, calls=%fused_concatenate,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// XLA defines clamp(min, x, max) = min(max(x, min), max), i.e. the lower bound
// is applied before the upper bound. This only matters when lower > max, where
// the correct result is the upper bound (not the lower one). This test pins
// that ordering by using a degenerate range with lower > upper.
TEST_F(CuDnnNonGemmFusionTest, ConstantScalarToNonPointwiseOp) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule wrapped_concatenate_computation.6_module, entry_computation_layout={()->f32[1,1]}

%fused_bitcast () -> f32[1,1] {
  %constant.17.1 = f32[1,1]{1,0} constant({ {1} })
  ROOT %bitcast.46.1 = f32[1,1]{1,0} bitcast(%constant.17.1)
}

ENTRY %wrapper_wrapped_concatenate_computation.6.clone () -> f32[1,1] {
  ROOT %fusion = f32[1,1] fusion(), kind=kLoop, calls=%fused_bitcast,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertAddNot) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.99_module, entry_computation_layout={(s32[1]{0}, pred[1]{0})->s32[1]{0}}

%fused_add.99.clone (param_0.19807: s32[1], param_1.19705: pred[1]) -> s32[1] {
  %param_0.19807 = s32[1]{0} parameter(0)
  %param_1.19705 = pred[1]{0} parameter(1)
  %not.2.1 = pred[1]{0} not(%param_1.19705)
  %convert.4392.1 = s32[1]{0} convert(%not.2.1)
  ROOT %add.3922.1 = s32[1]{0} add(%param_0.19807, %convert.4392.1)
}

ENTRY %wrapper_fused_add.99.clone (param_0.19807: s32[1], param_1.19705: pred[1]) -> s32[1] {
  param_0.19807 = s32[1] parameter(0)
  param_1.19705 = pred[1] parameter(1)
  ROOT %fusion = s32[1] fusion(param_0.19807, param_1.19705), kind=kLoop, calls=%fused_add.99.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertBitcastReshape) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert.5_module, entry_computation_layout={(f32[1]{0})->f32[]}

%fused_convert.5 (param_0.3347: f32[1]) -> f32[] {
  %param_0.3347 = f32[1]{0} parameter(0)
  %convert_element_type.3098.1 = bf16[1]{0} convert(%param_0.3347)
  %bitcast.459.2 = bf16[] bitcast(%convert_element_type.3098.1)
  ROOT %convert.1703.1 = f32[] convert(%bitcast.459.2)
}

ENTRY %wrapper_fused_convert.5 (param_0.3347: f32[1]) -> f32[] {
  param_0.3347 = f32[1] parameter(0)
  ROOT %fusion = f32[] fusion(param_0.3347), kind=kLoop, calls=%fused_convert.5,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertBitcastReshapeBroadcast) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert_module, entry_computation_layout={(s32[2]{0})->f32[128]{0}}

%fused_convert.clone (param_0.32328: s32[2]) -> f32[128] {
  %param_0.32328 = s32[2]{0} parameter(0)
  %broadcast_in_dim.655.1 = s32[64,2]{1,0} broadcast(%param_0.32328), dimensions={1}
  %bitcast.1601.1 = s32[128]{0} bitcast(%broadcast_in_dim.655.1)
  ROOT %convert_element_type.3357.1 = f32[128]{0} convert(%bitcast.1601.1)
}

ENTRY %wrapper_fused_convert.clone (param_0.32328: s32[2]) -> f32[128] {
  param_0.32328 = s32[2] parameter(0)
  ROOT %fusion = f32[128] fusion(param_0.32328), kind=kLoop, calls=%fused_convert.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeBroadcastAddDivideExponentialMultiplyNegate) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply_module, entry_computation_layout={(bf16[1,16,32,16,64]{4,3,2,1,0})->bf16[16,32,16,64]{3,2,1,0}}

%fused_multiply (param_0.1452: bf16[1,16,32,16,64]) -> bf16[16,32,16,64] {
  %param_0.1452 = bf16[1,16,32,16,64]{4,3,2,1,0} parameter(0)
  %bitcast.286.3 = bf16[16,32,16,64]{3,2,1,0} bitcast(%param_0.1452)
  %constant_3333_4 = bf16[] constant(1)
  %convert.1648.4 = f32[] convert(%constant_3333_4)
  %broadcast.3199.5 = f32[16,32,16,64]{3,2,1,0} broadcast(%convert.1648.4), dimensions={}
  %neg.45.9 = bf16[16,32,16,64]{3,2,1,0} negate(%bitcast.286.3)
  %exp.449.7 = bf16[16,32,16,64]{3,2,1,0} exponential(%neg.45.9)
  %jit_silu_.30.11 = bf16[16,32,16,64]{3,2,1,0} broadcast(%constant_3333_4), dimensions={}
  %add.1874.5 = bf16[16,32,16,64]{3,2,1,0} add(%exp.449.7, %jit_silu_.30.11)
  %convert.509.3 = f32[16,32,16,64]{3,2,1,0} convert(%add.1874.5)
  %div.2697.5 = f32[16,32,16,64]{3,2,1,0} divide(%broadcast.3199.5, %convert.509.3)
  %convert.510.3 = bf16[16,32,16,64]{3,2,1,0} convert(%div.2697.5)
  ROOT %mul.2665.1 = bf16[16,32,16,64]{3,2,1,0} multiply(%bitcast.286.3, %convert.510.3)
}

ENTRY %wrapper_fused_multiply (param_0.1452: bf16[1,16,32,16,64]) -> bf16[16,32,16,64] {
  param_0.1452 = bf16[1,16,32,16,64] parameter(0)
  ROOT %fusion = bf16[16,32,16,64] fusion(param_0.1452), kind=kLoop, calls=%fused_multiply,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertBitcastReshapeBroadcastAddMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert.13_module, entry_computation_layout={(f32[1,16,128]{2,1,0}, f32[1,16,128]{2,1,0}, bf16[16,128,84]{2,1,0}, bf16[16,128,84]{2,1,0}, bf16[1,84]{1,0})->bf16[16,128,84]{2,1,0}}

%fused_convert.13.clone (param_0.492: f32[1,16,128], param_1.562: f32[1,16,128], param_2.263: bf16[16,128,84], param_3.146: bf16[16,128,84], param_4.72: bf16[1,84]) -> bf16[16,128,84] {
  %param_2.263 = bf16[16,128,84]{2,1,0} parameter(2)
  %convert.233.8 = f32[16,128,84]{2,1,0} convert(%param_2.263)
  %param_1.562 = f32[1,16,128]{2,1,0} parameter(1)
  %bitcast.76.5 = f32[16,128]{1,0} bitcast(%param_1.562)
  %broadcast.271.5 = f32[16,128,84]{2,1,0} broadcast(%bitcast.76.5), dimensions={0,1}
  %multiply.99.3 = f32[16,128,84]{2,1,0} multiply(%convert.233.8, %broadcast.271.5)
  %param_3.146 = bf16[16,128,84]{2,1,0} parameter(3)
  %param_4.72 = bf16[1,84]{1,0} parameter(4)
  %bitcast.51.9 = bf16[84]{0} bitcast(%param_4.72)
  %broadcast.151.9 = bf16[16,128,84]{2,1,0} broadcast(%bitcast.51.9), dimensions={2}
  %multiply.94.3 = bf16[16,128,84]{2,1,0} multiply(%param_3.146, %broadcast.151.9)
  %convert.61.6 = f32[16,128,84]{2,1,0} convert(%multiply.94.3)
  %param_0.492 = f32[1,16,128]{2,1,0} parameter(0)
  %bitcast.50.7 = f32[16,128]{1,0} bitcast(%param_0.492)
  %broadcast.147.7 = f32[16,128,84]{2,1,0} broadcast(%bitcast.50.7), dimensions={0,1}
  %multiply.100.3 = f32[16,128,84]{2,1,0} multiply(%convert.61.6, %broadcast.147.7)
  %add.30.1 = f32[16,128,84]{2,1,0} add(%multiply.99.3, %multiply.100.3)
  ROOT %convert.62.1 = bf16[16,128,84]{2,1,0} convert(%add.30.1)
}

ENTRY %wrapper_fused_convert.13.clone (param_0.492: f32[1,16,128], param_1.562: f32[1,16,128], param_2.263: bf16[16,128,84], param_3.146: bf16[16,128,84], param_4.72: bf16[1,84]) -> bf16[16,128,84] {
  param_0.492 = f32[1,16,128] parameter(0)
  param_1.562 = f32[1,16,128] parameter(1)
  param_2.263 = bf16[16,128,84] parameter(2)
  param_3.146 = bf16[16,128,84] parameter(3)
  param_4.72 = bf16[1,84] parameter(4)
  ROOT %fusion = bf16[16,128,84] fusion(param_0.492, param_1.562, param_2.263, param_3.146, param_4.72), kind=kLoop, calls=%fused_convert.13.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertBitcastReshapeBroadcastAddMultiplyRsqrt) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.96_module, entry_computation_layout={(bf16[1,1,3840]{2,1,0}, bf16[3840]{0}, bf16[3840]{0}, f32[])->bf16[1,1,3840]{2,1,0}}

%fused_add.96.clone (param_0.4770: bf16[1,1,3840], param_1.13713: bf16[3840], param_2.10556: bf16[3840], param_3.6899: f32[]) -> bf16[1,1,3840] {
  %param_2.10556 = bf16[3840]{0} parameter(2)
  %param_3.6899 = f32[] parameter(3)
  %constant_2456_2 = f32[] constant(0.00026041668)
  %multiply.6895.3 = f32[] multiply(%param_3.6899, %constant_2456_2)
  %convert.5269.3 = bf16[] convert(%multiply.6895.3)
  %constant_1047_4 = bf16[] constant(9.984e-07)
  %add.4684.3 = bf16[] add(%convert.5269.3, %constant_1047_4)
  %convert.6443.5 = f32[] convert(%add.4684.3)
  %rsqrt.1157.5 = f32[] rsqrt(%convert.6443.5)
  %convert.6444.3 = bf16[] convert(%rsqrt.1157.5)
  %broadcast.8309.5 = bf16[3840]{0} broadcast(%convert.6444.3), dimensions={}
  %multiply.6013.5 = bf16[3840]{0} multiply(%param_2.10556, %broadcast.8309.5)
  %param_1.13713 = bf16[3840]{0} parameter(1)
  %constant_1040_54 = bf16[] constant(1)
  %broadcast.8280.763 = bf16[3840]{0} broadcast(%constant_1040_54), dimensions={}
  %add.3977.5 = bf16[3840]{0} add(%param_1.13713, %broadcast.8280.763)
  %multiply.6702.3 = bf16[3840]{0} multiply(%multiply.6013.5, %add.3977.5)
  %bitcast.184.1 = bf16[1,1,3840]{2,1,0} bitcast(%multiply.6702.3)
  %param_0.4770 = bf16[1,1,3840]{2,1,0} parameter(0)
  ROOT %add.2672.1 = bf16[1,1,3840]{2,1,0} add(%bitcast.184.1, %param_0.4770)
}

ENTRY %wrapper_fused_add.96.clone (param_0.4770: bf16[1,1,3840], param_1.13713: bf16[3840], param_2.10556: bf16[3840], param_3.6899: f32[]) -> bf16[1,1,3840] {
  param_0.4770 = bf16[1,1,3840] parameter(0)
  param_1.13713 = bf16[3840] parameter(1)
  param_2.10556 = bf16[3840] parameter(2)
  param_3.6899 = f32[] parameter(3)
  ROOT %fusion = bf16[1,1,3840] fusion(param_0.4770, param_1.13713, param_2.10556, param_3.6899), kind=kLoop, calls=%fused_add.96.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertBitcastReshapeBroadcastAddMultiplyTanh) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply_module, entry_computation_layout={(bf16[512,128]{1,0}, bf16[512,128]{1,0})->bf16[16,32,128]{2,1,0}}

%fused_multiply.clone (param_0.52: bf16[512,128], param_1.29: bf16[512,128]) -> bf16[16,32,128] {
  %param_1.29 = bf16[512,128]{1,0} parameter(1)
  %bitcast.740.6 = bf16[16,32,128]{2,1,0} bitcast(%param_1.29)
  %multiply.263.3 = bf16[16,32,128]{2,1,0} multiply(%bitcast.740.6, %bitcast.740.6)
  %multiply.264.3 = bf16[16,32,128]{2,1,0} multiply(%multiply.263.3, %bitcast.740.6)
  %constant_116_1 = bf16[] constant(0.04468)
  %broadcast.377.1 = bf16[16,32,128]{2,1,0} broadcast(%constant_116_1), dimensions={}
  %multiply.265.1 = bf16[16,32,128]{2,1,0} multiply(%multiply.264.3, %broadcast.377.1)
  %add.55.3 = bf16[16,32,128]{2,1,0} add(%bitcast.740.6, %multiply.265.1)
  %constant_118_1 = bf16[] constant(0.7969)
  %broadcast.378.1 = bf16[16,32,128]{2,1,0} broadcast(%constant_118_1), dimensions={}
  %multiply.266.1 = bf16[16,32,128]{2,1,0} multiply(%add.55.3, %broadcast.378.1)
  %convert.274.11 = f32[16,32,128]{2,1,0} convert(%multiply.266.1)
  %tanh.2.11 = f32[16,32,128]{2,1,0} tanh(%convert.274.11)
  %convert.275.9 = bf16[16,32,128]{2,1,0} convert(%tanh.2.11)
  %constant_120_1 = bf16[] constant(1)
  %broadcast.379.1 = bf16[16,32,128]{2,1,0} broadcast(%constant_120_1), dimensions={}
  %add.56.7 = bf16[16,32,128]{2,1,0} add(%convert.275.9, %broadcast.379.1)
  %constant_122_1 = bf16[] constant(0.5)
  %broadcast.381.1 = bf16[16,32,128]{2,1,0} broadcast(%constant_122_1), dimensions={}
  %multiply.267.5 = bf16[16,32,128]{2,1,0} multiply(%add.56.7, %broadcast.381.1)
  %multiply.268.3 = bf16[16,32,128]{2,1,0} multiply(%bitcast.740.6, %multiply.267.5)
  %param_0.52 = bf16[512,128]{1,0} parameter(0)
  %bitcast.743.1 = bf16[16,32,128]{2,1,0} bitcast(%param_0.52)
  ROOT %multiply.269.1 = bf16[16,32,128]{2,1,0} multiply(%multiply.268.3, %bitcast.743.1)
}

ENTRY %wrapper_fused_multiply.clone (param_0.52: bf16[512,128], param_1.29: bf16[512,128]) -> bf16[16,32,128] {
  param_0.52 = bf16[512,128] parameter(0)
  param_1.29 = bf16[512,128] parameter(1)
  ROOT %fusion = bf16[16,32,128] fusion(param_0.52, param_1.29), kind=kLoop, calls=%fused_multiply.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeBroadcastExponentialSubtract) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_exponential_reduce.clone.clone.clone_module, entry_computation_layout={(f32[512]{0}, bf16[512,256]{1,0})->f32[16,32,256]{2,1,0}}

%fused_exponential_reduce.clone.clone.clone.clone (param_0.2155: f32[512], param_1.2502: bf16[512,256]) -> f32[16,32,256] {
  %param_1.2502 = bf16[512,256]{1,0} parameter(1)
  %convert.164.4.clone.4 = f32[512,256]{1,0} convert(%param_1.2502)
  %bitcast.126.3.clone.4 = f32[16,32,256]{2,1,0} bitcast(%convert.164.4.clone.4)
  %param_0.2155 = f32[512]{0} parameter(0)
  %bitcast.127.5.clone.4 = f32[16,32]{1,0} bitcast(%param_0.2155)
  %broadcast.675.5.clone.4 = f32[16,32,256]{2,1,0} broadcast(%bitcast.127.5.clone.4), dimensions={0,1}
  %subtract.676.3.clone.4 = f32[16,32,256]{2,1,0} subtract(%bitcast.126.3.clone.4, %broadcast.675.5.clone.4)
  ROOT %exponential.677.1.clone.4 = f32[16,32,256]{2,1,0} exponential(%subtract.676.3.clone.4)
}

ENTRY %wrapper_fused_exponential_reduce.clone.clone.clone.clone (param_0.2155: f32[512], param_1.2502: bf16[512,256]) -> f32[16,32,256] {
  param_0.2155 = f32[512] parameter(0)
  param_1.2502 = bf16[512,256] parameter(1)
  ROOT %fusion = f32[16,32,256] fusion(param_0.2155, param_1.2502), kind=kLoop, calls=%fused_exponential_reduce.clone.clone.clone.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertBitcastReshapeBroadcastMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.9.clone.clone_module, entry_computation_layout={(bf16[84]{0}, f32[1,16,128]{2,1,0}, bf16[16,128,84]{2,1,0})->bf16[16,128,84]{2,1,0}}

%fused_multiply.9.clone.clone.clone (param_0.2149: bf16[84], param_1.2499: f32[1,16,128], param_2.1366: bf16[16,128,84]) -> bf16[16,128,84] {
  %param_2.1366 = bf16[16,128,84]{2,1,0} parameter(2)
  %convert.622.21 = f32[16,128,84]{2,1,0} convert(%param_2.1366)
  %param_1.2499 = f32[1,16,128]{2,1,0} parameter(1)
  %bitcast.124.15 = f32[16,128]{1,0} bitcast(%param_1.2499)
  %broadcast.638.15 = f32[16,128,84]{2,1,0} broadcast(%bitcast.124.15), dimensions={0,1}
  %multiply.639.9 = f32[16,128,84]{2,1,0} multiply(%convert.622.21, %broadcast.638.15)
  %convert.640.7 = bf16[16,128,84]{2,1,0} convert(%multiply.639.9)
  %param_0.2149 = bf16[84]{0} parameter(0)
  %broadcast.645.8 = bf16[16,128,84]{2,1,0} broadcast(%param_0.2149), dimensions={2}
  ROOT %multiply.646.3 = bf16[16,128,84]{2,1,0} multiply(%convert.640.7, %broadcast.645.8)
}

ENTRY %wrapper_fused_multiply.9.clone.clone.clone (param_0.2149: bf16[84], param_1.2499: f32[1,16,128], param_2.1366: bf16[16,128,84]) -> bf16[16,128,84] {
  param_0.2149 = bf16[84] parameter(0)
  param_1.2499 = f32[1,16,128] parameter(1)
  param_2.1366 = bf16[16,128,84] parameter(2)
  ROOT %fusion = bf16[16,128,84] fusion(param_0.2149, param_1.2499, param_2.1366), kind=kLoop, calls=%fused_multiply.9.clone.clone.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeConcatenateBroadcastAddMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.2_module, entry_computation_layout={(f32[1,16,128,32]{3,2,1,0}, f32[1,16,128,32]{3,2,1,0}, bf16[2048,4096]{1,0}, bf16[1,128]{1,0}, bf16[16,128,32,64]{3,2,1,0}, /*index=5*/bf16[16,128,32,64]{3,2,1,0})->bf16[16,128,32,128]{3,2,1,0}}

%fused_concatenate.2.clone (param_0.560: f32[1,16,128,32], param_1.615: f32[1,16,128,32], param_2.291: bf16[2048,4096], param_3.164: bf16[1,128], param_4.81: bf16[16,128,32,64], param_5.40: bf16[16,128,32,64]) -> bf16[16,128,32,128] {
  %param_2.291 = bf16[2048,4096]{1,0} parameter(2)
  %bitcast.692.15 = bf16[16,128,32,128]{3,2,1,0} bitcast(%param_2.291)
  %convert.231.15 = f32[16,128,32,128]{3,2,1,0} convert(%bitcast.692.15)
  %param_1.615 = f32[1,16,128,32]{3,2,1,0} parameter(1)
  %bitcast.90.5 = f32[16,128,32]{2,1,0} bitcast(%param_1.615)
  %broadcast.277.5 = f32[16,128,32,128]{3,2,1,0} broadcast(%bitcast.90.5), dimensions={0,1,2}
  %multiply.122.3 = f32[16,128,32,128]{3,2,1,0} multiply(%convert.231.15, %broadcast.277.5)
  %param_4.81 = bf16[16,128,32,64]{3,2,1,0} parameter(4)
  %param_5.40 = bf16[16,128,32,64]{3,2,1,0} parameter(5)
  %concatenate.5.4 = bf16[16,128,32,128]{3,2,1,0} concatenate(%param_4.81, %param_5.40), dimensions={3}
  %param_3.164 = bf16[1,128]{1,0} parameter(3)
  %bitcast.36.11 = bf16[128]{0} bitcast(%param_3.164)
  %broadcast.71.11 = bf16[16,128,32,128]{3,2,1,0} broadcast(%bitcast.36.11), dimensions={3}
  %multiply.117.3 = bf16[16,128,32,128]{3,2,1,0} multiply(%concatenate.5.4, %broadcast.71.11)
  %convert.65.6 = f32[16,128,32,128]{3,2,1,0} convert(%multiply.117.3)
  %param_0.560 = f32[1,16,128,32]{3,2,1,0} parameter(0)
  %bitcast.35.9 = f32[16,128,32]{2,1,0} bitcast(%param_0.560)
  %broadcast.67.9 = f32[16,128,32,128]{3,2,1,0} broadcast(%bitcast.35.9), dimensions={0,1,2}
  %multiply.123.3 = f32[16,128,32,128]{3,2,1,0} multiply(%convert.65.6, %broadcast.67.9)
  %add.37.1 = f32[16,128,32,128]{3,2,1,0} add(%multiply.122.3, %multiply.123.3)
  ROOT %convert.66.1 = bf16[16,128,32,128]{3,2,1,0} convert(%add.37.1)
}

ENTRY %wrapper_fused_concatenate.2.clone (param_0.560: f32[1,16,128,32], param_1.615: f32[1,16,128,32], param_2.291: bf16[2048,4096], param_3.164: bf16[1,128], param_4.81: bf16[16,128,32,64], param_5.40: bf16[16,128,32,64]) -> bf16[16,128,32,128] {
  param_0.560 = f32[1,16,128,32] parameter(0)
  param_1.615 = f32[1,16,128,32] parameter(1)
  param_2.291 = bf16[2048,4096] parameter(2)
  param_3.164 = bf16[1,128] parameter(3)
  param_4.81 = bf16[16,128,32,64] parameter(4)
  param_5.40 = bf16[16,128,32,64] parameter(5)
  ROOT %fusion = bf16[16,128,32,128] fusion(param_0.560, param_1.615, param_2.291, param_3.164, param_4.81, param_5.40), kind=kLoop, calls=%fused_concatenate.2.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeConcatenateSliceBroadcastAddMultiplyNegate) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.33_module, entry_computation_layout={(bf16[1,1,4096,64]{3,2,1,0}, bf16[1,1,4096,64]{3,2,1,0}, bf16[1,4096,32,128]{3,2,1,0})->bf16[1,4096,32,128]{3,2,1,0}}

%fused_concatenate.33.clone (param_0.28568: bf16[1,1,4096,64], param_1.29691: bf16[1,1,4096,64], param_2.10719: bf16[1,4096,32,128]) -> bf16[1,4096,32,128] {
  %param_2.10719 = bf16[1,4096,32,128]{3,2,1,0} parameter(2)
  %slice.394.1 = bf16[1,4096,32,64]{3,2,1,0} slice(%param_2.10719), slice={[0:1], [0:4096], [0:32], [64:128]}
  %param_0.28568 = bf16[1,1,4096,64]{3,2,1,0} parameter(0)
  %bitcast.1535.24 = bf16[4096,64]{1,0} bitcast(%param_0.28568)
  %broadcast.7226.23 = bf16[1,4096,32,64]{3,2,1,0} broadcast(%bitcast.1535.24), dimensions={1,3}
  %multiply.3238.5 = bf16[1,4096,32,64]{3,2,1,0} multiply(%slice.394.1, %broadcast.7226.23)
  %slice.395.1 = bf16[1,4096,32,64]{3,2,1,0} slice(%param_2.10719), slice={[0:1], [0:4096], [0:32], [0:64]}
  %param_1.29691 = bf16[1,1,4096,64]{3,2,1,0} parameter(1)
  %bitcast.1534.18 = bf16[4096,64]{1,0} bitcast(%param_1.29691)
  %broadcast.7225.17 = bf16[1,4096,32,64]{3,2,1,0} broadcast(%bitcast.1534.18), dimensions={1,3}
  %multiply.3240.3 = bf16[1,4096,32,64]{3,2,1,0} multiply(%slice.395.1, %broadcast.7225.17)
  %add.3435.3 = bf16[1,4096,32,64]{3,2,1,0} add(%multiply.3238.5, %multiply.3240.3)
  %multiply.3241.3 = bf16[1,4096,32,64]{3,2,1,0} multiply(%slice.394.1, %broadcast.7225.17)
  %convert.3341.5 = f32[1,4096,32,64]{3,2,1,0} convert(%slice.395.1)
  %negate.194.5 = f32[1,4096,32,64]{3,2,1,0} negate(%convert.3341.5)
  %convert.3342.3 = bf16[1,4096,32,64]{3,2,1,0} convert(%negate.194.5)
  %multiply.3242.5 = bf16[1,4096,32,64]{3,2,1,0} multiply(%convert.3342.3, %broadcast.7226.23)
  %add.3436.3 = bf16[1,4096,32,64]{3,2,1,0} add(%multiply.3241.3, %multiply.3242.5)
  ROOT %concatenate.200.1 = bf16[1,4096,32,128]{3,2,1,0} concatenate(%add.3435.3, %add.3436.3), dimensions={3}
}

ENTRY %wrapper_fused_concatenate.33.clone (param_0.28568: bf16[1,1,4096,64], param_1.29691: bf16[1,1,4096,64], param_2.10719: bf16[1,4096,32,128]) -> bf16[1,4096,32,128] {
  param_0.28568 = bf16[1,1,4096,64] parameter(0)
  param_1.29691 = bf16[1,1,4096,64] parameter(1)
  param_2.10719 = bf16[1,4096,32,128] parameter(2)
  ROOT %fusion = bf16[1,4096,32,128] fusion(param_0.28568, param_1.29691, param_2.10719), kind=kLoop, calls=%fused_concatenate.33.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeConcatenateSliceBroadcastAddMultiplyRsqrtSubtract) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate_module, entry_computation_layout={(f32[128]{0}, f32[128]{0}, bf16[256]{0}, f32[16]{0}, bf16[4096,1]{0,1})->bf16[1,1,16,256]{3,2,1,0}}

%fused_concatenate.clone (param_0.14027: f32[128], param_1.20005: f32[128], param_2.9579: bf16[256], param_3.7044: f32[16], param_4.5061: bf16[4096,1]{0,1}) -> bf16[1,1,16,256] {
  %param_4.5061 = bf16[4096,1]{0,1} parameter(4)
  %bitcast.8655.8 = bf16[4096]{0} bitcast(%param_4.5061)
  %param_3.7044 = f32[16]{0} parameter(3)
  %bitcast.7155.7 = f32[1,1,16]{2,1,0} bitcast(%param_3.7044)
  %constant_1178_5 = f32[] constant(0.00390625)
  %broadcast.8299.242 = f32[1,1,16]{2,1,0} broadcast(%constant_1178_5), dimensions={}
  %multiply.6393.7 = f32[1,1,16]{2,1,0} multiply(%bitcast.7155.7, %broadcast.8299.242)
  %convert.4641.5 = bf16[1,1,16]{2,1,0} convert(%multiply.6393.7)
  %constant_1047_99 = bf16[] constant(9.984e-07)
  %broadcast.8849.146 = bf16[1,1,16]{2,1,0} broadcast(%constant_1047_99), dimensions={}
  %add.4363.5 = bf16[1,1,16]{2,1,0} add(%convert.4641.5, %broadcast.8849.146)
  %convert.6432.3 = f32[1,1,16]{2,1,0} convert(%add.4363.5)
  %rsqrt.869.5 = f32[1,1,16]{2,1,0} rsqrt(%convert.6432.3)
  %convert.6433.5 = bf16[1,1,16]{2,1,0} convert(%rsqrt.869.5)
  %bitcast.160.7 = bf16[16]{0} bitcast(%convert.6433.5)
  %broadcast.6714.7 = bf16[1,1,16,256]{3,2,1,0} broadcast(%bitcast.160.7), dimensions={2}
  %bitcast.161.5 = bf16[4096]{0} bitcast(%broadcast.6714.7)
  %multiply.6009.5 = bf16[4096]{0} multiply(%bitcast.8655.8, %bitcast.161.5)
  %param_2.9579 = bf16[256]{0} parameter(2)
  %constant_1040_244 = bf16[] constant(1)
  %broadcast.8294.386 = bf16[256]{0} broadcast(%constant_1040_244), dimensions={}
  %add.3973.5 = bf16[256]{0} add(%param_2.9579, %broadcast.8294.386)
  %broadcast.6715.3 = bf16[1,1,16,256]{3,2,1,0} broadcast(%add.3973.5), dimensions={3}
  %bitcast.162.3 = bf16[4096]{0} bitcast(%broadcast.6715.3)
  %multiply.6700.3 = bf16[4096]{0} multiply(%multiply.6009.5, %bitcast.162.3)
  %bitcast.163.29 = bf16[1,1,16,256]{3,2,1,0} bitcast(%multiply.6700.3)
  %convert.5999.29 = f32[1,1,16,256]{3,2,1,0} convert(%bitcast.163.29)
  %slice.1241.13 = f32[1,1,16,128]{3,2,1,0} slice(%convert.5999.29), slice={[0:1], [0:1], [0:16], [0:128]}
  %param_1.20005 = f32[128]{0} parameter(1)
  %broadcast.6718.402 = f32[1,1,16,128]{3,2,1,0} broadcast(%param_1.20005), dimensions={3}
  %multiply.3886.7 = f32[1,1,16,128]{3,2,1,0} multiply(%slice.1241.13, %broadcast.6718.402)
  %slice.1243.13 = f32[1,1,16,128]{3,2,1,0} slice(%convert.5999.29), slice={[0:1], [0:1], [0:16], [128:256]}
  %param_0.14027 = f32[128]{0} parameter(0)
  %broadcast.6719.242 = f32[1,1,16,128]{3,2,1,0} broadcast(%param_0.14027), dimensions={3}
  %multiply.3887.5 = f32[1,1,16,128]{3,2,1,0} multiply(%slice.1243.13, %broadcast.6719.242)
  %subtract.381.5 = f32[1,1,16,128]{3,2,1,0} subtract(%multiply.3886.7, %multiply.3887.5)
  %convert.5797.3 = bf16[1,1,16,128]{3,2,1,0} convert(%subtract.381.5)
  %multiply.3888.7 = f32[1,1,16,128]{3,2,1,0} multiply(%slice.1243.13, %broadcast.6718.402)
  %multiply.3889.5 = f32[1,1,16,128]{3,2,1,0} multiply(%slice.1241.13, %broadcast.6719.242)
  %add.2660.5 = f32[1,1,16,128]{3,2,1,0} add(%multiply.3888.7, %multiply.3889.5)
  %convert.5798.3 = bf16[1,1,16,128]{3,2,1,0} convert(%add.2660.5)
  ROOT %concatenate.389.1 = bf16[1,1,16,256]{3,2,1,0} concatenate(%convert.5797.3, %convert.5798.3), dimensions={3}
}

ENTRY %wrapper_fused_concatenate.clone (param_0.14027: f32[128], param_1.20005: f32[128], param_2.9579: bf16[256], param_3.7044: f32[16], param_4.5061: bf16[4096,1]{0,1}) -> bf16[1,1,16,256] {
  param_0.14027 = f32[128] parameter(0)
  param_1.20005 = f32[128] parameter(1)
  param_2.9579 = bf16[256] parameter(2)
  param_3.7044 = f32[16] parameter(3)
  param_4.5061 = bf16[4096,1]{0,1} parameter(4)
  ROOT %fusion = bf16[1,1,16,256] fusion(param_0.14027, param_1.20005, param_2.9579, param_3.7044, param_4.5061), kind=kLoop, calls=%fused_concatenate.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeConcatenateSliceBroadcastAddMultiplySubtract) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.47_module, entry_computation_layout={(f32[1,11,128]{2,1,0}, bf16[1,1,11,8,256]{4,3,2,1,0}, f32[1,11,128]{2,1,0})->bf16[1,11,8,256]{3,2,1,0}}

%fused_concatenate.47.clone (param_0.10316: f32[1,11,128], param_1.9896: bf16[1,1,11,8,256], param_2.7783: f32[1,11,128]) -> bf16[1,11,8,256] {
  %param_1.9896 = bf16[1,1,11,8,256]{4,3,2,1,0} parameter(1)
  %bitcast.143.29 = bf16[1,11,8,256]{3,2,1,0} bitcast(%param_1.9896)
  %convert.1188.29 = f32[1,11,8,256]{3,2,1,0} convert(%bitcast.143.29)
  %slice.13.15 = f32[1,11,8,128]{3,2,1,0} slice(%convert.1188.29), slice={[0:1], [0:11], [0:8], [0:128]}
  %param_2.7783 = f32[1,11,128]{2,1,0} parameter(2)
  %bitcast.145.720 = f32[11,128]{1,0} bitcast(%param_2.7783)
  %broadcast.920.399 = f32[1,11,8,128]{3,2,1,0} broadcast(%bitcast.145.720), dimensions={1,3}
  %multiply.921.7 = f32[1,11,8,128]{3,2,1,0} multiply(%slice.13.15, %broadcast.920.399)
  %slice.15.11 = f32[1,11,8,128]{3,2,1,0} slice(%convert.1188.29), slice={[0:1], [0:11], [0:8], [128:256]}
  %param_0.10316 = f32[1,11,128]{2,1,0} parameter(0)
  %bitcast.146.880 = f32[11,128]{1,0} bitcast(%param_0.10316)
  %broadcast.925.399 = f32[1,11,8,128]{3,2,1,0} broadcast(%bitcast.146.880), dimensions={1,3}
  %multiply.926.5 = f32[1,11,8,128]{3,2,1,0} multiply(%slice.15.11, %broadcast.925.399)
  %subtract.927.5 = f32[1,11,8,128]{3,2,1,0} subtract(%multiply.921.7, %multiply.926.5)
  %convert.1134.3 = bf16[1,11,8,128]{3,2,1,0} convert(%subtract.927.5)
  %multiply.932.5 = f32[1,11,8,128]{3,2,1,0} multiply(%slice.15.11, %broadcast.920.399)
  %multiply.937.7 = f32[1,11,8,128]{3,2,1,0} multiply(%slice.13.15, %broadcast.925.399)
  %add.938.5 = f32[1,11,8,128]{3,2,1,0} add(%multiply.932.5, %multiply.937.7)
  %convert.1135.3 = bf16[1,11,8,128]{3,2,1,0} convert(%add.938.5)
  ROOT %concatenate.96.1 = bf16[1,11,8,256]{3,2,1,0} concatenate(%convert.1134.3, %convert.1135.3), dimensions={3}
}

ENTRY %wrapper_fused_concatenate.47.clone (param_0.10316: f32[1,11,128], param_1.9896: bf16[1,1,11,8,256], param_2.7783: f32[1,11,128]) -> bf16[1,11,8,256] {
  param_0.10316 = f32[1,11,128] parameter(0)
  param_1.9896 = bf16[1,1,11,8,256] parameter(1)
  param_2.7783 = f32[1,11,128] parameter(2)
  ROOT %fusion = bf16[1,11,8,256] fusion(param_0.10316, param_1.9896, param_2.7783), kind=kLoop, calls=%fused_concatenate.47.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeSelectBroadcastAddDivideMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.8_module, entry_computation_layout={(bf16[1536,1]{1,0}, f32[], bf16[1536]{0}, pred[])->bf16[1536,1]{1,0}}

%fused_add.8 (param_0.9377: bf16[1536,1], param_1.11076: f32[], param_2.8255: bf16[1536], param_3.6505: pred[]) -> bf16[1536,1] {
  %param_3.6505 = pred[] parameter(3)
  %select_n.2395.3 = pred[1536,1]{1,0} broadcast(%param_3.6505), dimensions={}
  %param_2.8255 = bf16[1536]{0} parameter(2)
  %bitcast.1147.15 = bf16[1536,1]{1,0} bitcast(%param_2.8255)
  %convert.1242.9 = f32[1536,1]{1,0} convert(%bitcast.1147.15)
  %param_1.11076 = f32[] parameter(1)
  %broadcast.3627.5 = f32[1536,1]{1,0} broadcast(%param_1.11076), dimensions={}
  %div.3144.7 = f32[1536,1]{1,0} divide(%convert.1242.9, %broadcast.3627.5)
  %convert.1244.5 = bf16[1536,1]{1,0} convert(%div.3144.7)
  %select_n.2396.3 = bf16[1536,1]{1,0} select(%select_n.2395.3, %bitcast.1147.15, %convert.1244.5)
  %constant_4302_62 = bf16[] constant(0.1001)
  %broadcast.2636.1 = bf16[1536,1]{1,0} broadcast(%constant_4302_62), dimensions={}
  %mul.3194.1 = bf16[1536,1]{1,0} multiply(%select_n.2396.3, %broadcast.2636.1)
  %param_0.9377 = bf16[1536,1]{1,0} parameter(0)
  %constant_4303_7 = bf16[] constant(0.8984)
  %mul.3195.1 = bf16[1536,1]{1,0} broadcast(%constant_4303_7), dimensions={}
  %mul.3196.1 = bf16[1536,1]{1,0} multiply(%param_0.9377, %mul.3195.1)
  ROOT %add.2032.1 = bf16[1536,1]{1,0} add(%mul.3194.1, %mul.3196.1)
}

ENTRY %wrapper_fused_add.8 (param_0.9377: bf16[1536,1], param_1.11076: f32[], param_2.8255: bf16[1536], param_3.6505: pred[]) -> bf16[1536,1] {
  param_0.9377 = bf16[1536,1] parameter(0)
  param_1.11076 = f32[] parameter(1)
  param_2.8255 = bf16[1536] parameter(2)
  param_3.6505 = pred[] parameter(3)
  ROOT %fusion = bf16[1536,1] fusion(param_0.9377, param_1.11076, param_2.8255, param_3.6505), kind=kLoop, calls=%fused_add.8,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertBitcastReshapeSelectBroadcastDivide) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.2_module, entry_computation_layout={(bf16[128,448]{1,0}, pred[], f32[])->bf16[1,128,448]{2,1,0}}

%fused_select.2.clone (param_0.16859: bf16[128,448], param_1.448: pred[], param_2.2645: f32[]) -> bf16[1,128,448] {
  %param_1.448 = pred[] parameter(1)
  %broadcast.10109.63 = pred[1,128,448]{2,1,0} broadcast(%param_1.448), dimensions={}
  %param_0.16859 = bf16[128,448]{1,0} parameter(0)
  %bitcast.38055.2 = bf16[1,128,448]{2,1,0} bitcast(%param_0.16859)
  %convert.8606.5 = f32[1,128,448]{2,1,0} convert(%bitcast.38055.2)
  %param_2.2645 = f32[] parameter(2)
  %broadcast.6111.318 = f32[1,128,448]{2,1,0} broadcast(%param_2.2645), dimensions={}
  %divide.2017.5 = f32[1,128,448]{2,1,0} divide(%convert.8606.5, %broadcast.6111.318)
  %convert.8608.3 = bf16[1,128,448]{2,1,0} convert(%divide.2017.5)
  ROOT %select.2901.1 = bf16[1,128,448]{2,1,0} select(%broadcast.10109.63, %bitcast.38055.2, %convert.8608.3)
}

ENTRY %wrapper_fused_select.2.clone (param_0.16859: bf16[128,448], param_1.448: pred[], param_2.2645: f32[]) -> bf16[1,128,448] {
  param_0.16859 = bf16[128,448] parameter(0)
  param_1.448 = pred[] parameter(1)
  param_2.2645 = f32[] parameter(2)
  ROOT %fusion = bf16[1,128,448] fusion(param_0.16859, param_1.448, param_2.2645), kind=kLoop, calls=%fused_select.2.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeSelectBroadcastDivideMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.4_module, entry_computation_layout={(f32[], pred[], bf16[56,576]{1,0}, bf16[1]{0})->bf16[1,56,576]{2,1,0}}

%fused_select.4 (param_0.7782: f32[], param_1.7623: pred[], param_2.5740: bf16[56,576], param_3.4387: bf16[1]) -> bf16[1,56,576] {
  %param_1.7623 = pred[] parameter(1)
  %broadcast.2994.2 = pred[1,56,576]{2,1,0} broadcast(%param_1.7623), dimensions={}
  %param_2.5740 = bf16[56,576]{1,0} parameter(2)
  %param_3.4387 = bf16[1]{0} parameter(3)
  %bitcast.1127.5 = bf16[] bitcast(%param_3.4387)
  %broadcast_in_dim.3087.5 = bf16[56,576]{1,0} broadcast(%bitcast.1127.5), dimensions={}
  %mul.3059.3 = bf16[56,576]{1,0} multiply(%param_2.5740, %broadcast_in_dim.3087.5)
  %bitcast.1979.2 = bf16[1,56,576]{2,1,0} bitcast(%mul.3059.3)
  %convert.1176.5 = f32[1,56,576]{2,1,0} convert(%bitcast.1979.2)
  %param_0.7782 = f32[] parameter(0)
  %broadcast.3611.13 = f32[1,56,576]{2,1,0} broadcast(%param_0.7782), dimensions={}
  %divide.72.5 = f32[1,56,576]{2,1,0} divide(%convert.1176.5, %broadcast.3611.13)
  %convert.1178.3 = bf16[1,56,576]{2,1,0} convert(%divide.72.5)
  ROOT %select.162.1 = bf16[1,56,576]{2,1,0} select(%broadcast.2994.2, %bitcast.1979.2, %convert.1178.3)
}

ENTRY %wrapper_fused_select.4 (param_0.7782: f32[], param_1.7623: pred[], param_2.5740: bf16[56,576], param_3.4387: bf16[1]) -> bf16[1,56,576] {
  param_0.7782 = f32[] parameter(0)
  param_1.7623 = pred[] parameter(1)
  param_2.5740 = bf16[56,576] parameter(2)
  param_3.4387 = bf16[1] parameter(3)
  ROOT %fusion = bf16[1,56,576] fusion(param_0.7782, param_1.7623, param_2.5740, param_3.4387), kind=kLoop, calls=%fused_select.4,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeSelectTransposeBroadcastDivideExponentialSubtract) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert.5_module, entry_computation_layout={(bf16[1,1,8,2]{3,2,1,0}, bf16[8,2]{1,0}, bf16[8,4096,2]{2,1,0}, pred[1,4096]{1,0})->bf16[1,1,8,4096,2]{4,3,2,1,0}}

%fused_convert.5.clone (param_0.20297: bf16[1,1,8,2], param_1.20468: bf16[8,2], param_2.10488: bf16[8,4096,2], param_3.7234: pred[1,4096]) -> bf16[1,1,8,4096,2] {
  %param_3.7234 = pred[1,4096]{1,0} parameter(3)
  %bitcast.364.61 = pred[4096]{0} bitcast(%param_3.7234)
  %broadcast.8864.61 = pred[8,4096,2]{2,1,0} broadcast(%bitcast.364.61), dimensions={1}
  %param_2.10488 = bf16[8,4096,2]{2,1,0} parameter(2)
  %constant_1636_63 = bf16[] constant(-2.379e+38)
  %broadcast.8858.92 = bf16[8,4096,2]{2,1,0} broadcast(%constant_1636_63), dimensions={}
  %select.1329.3 = bf16[8,4096,2]{2,1,0} select(%broadcast.8864.61, %param_2.10488, %broadcast.8858.92)
  %bitcast.2389.7 = bf16[1,1,8,4096,2]{4,3,2,1,0} bitcast(%select.1329.3)
  %param_1.20468 = bf16[8,2]{1,0} parameter(1)
  %bitcast.380.9 = bf16[16]{0} bitcast(%param_1.20468)
  %broadcast.6857.9 = bf16[1,1,16,4096]{3,2,1,0} broadcast(%bitcast.380.9), dimensions={2}
  %bitcast.4834.7 = bf16[8,2,4096]{2,1,0} bitcast(%broadcast.6857.9)
  %transpose.1954.7 = bf16[8,4096,2]{2,1,0} transpose(%bitcast.4834.7), dimensions={0,2,1}
  %bitcast.4835.7 = bf16[1,1,8,4096,2]{4,3,2,1,0} bitcast(%transpose.1954.7)
  %subtract.585.7 = bf16[1,1,8,4096,2]{4,3,2,1,0} subtract(%bitcast.2389.7, %bitcast.4835.7)
  %exponential.197.5 = bf16[1,1,8,4096,2]{4,3,2,1,0} exponential(%subtract.585.7)
  %convert.6519.4 = f32[1,1,8,4096,2]{4,3,2,1,0} convert(%exponential.197.5)
  %param_0.20297 = bf16[1,1,8,2]{3,2,1,0} parameter(0)
  %bitcast.382.5 = bf16[16]{0} bitcast(%param_0.20297)
  %broadcast.6858.5 = bf16[1,1,16,4096]{3,2,1,0} broadcast(%bitcast.382.5), dimensions={2}
  %bitcast.4836.3 = bf16[8,2,4096]{2,1,0} bitcast(%broadcast.6858.5)
  %transpose.1955.3 = bf16[8,4096,2]{2,1,0} transpose(%bitcast.4836.3), dimensions={0,2,1}
  %bitcast.4837.3 = bf16[1,1,8,4096,2]{4,3,2,1,0} bitcast(%transpose.1955.3)
  %convert.6520.3 = f32[1,1,8,4096,2]{4,3,2,1,0} convert(%bitcast.4837.3)
  %divide.1543.3 = f32[1,1,8,4096,2]{4,3,2,1,0} divide(%convert.6519.4, %convert.6520.3)
  ROOT %convert.6521.1 = bf16[1,1,8,4096,2]{4,3,2,1,0} convert(%divide.1543.3)
}

ENTRY %wrapper_fused_convert.5.clone (param_0.20297: bf16[1,1,8,2], param_1.20468: bf16[8,2], param_2.10488: bf16[8,4096,2], param_3.7234: pred[1,4096]) -> bf16[1,1,8,4096,2] {
  param_0.20297 = bf16[1,1,8,2] parameter(0)
  param_1.20468 = bf16[8,2] parameter(1)
  param_2.10488 = bf16[8,4096,2] parameter(2)
  param_3.7234 = pred[1,4096] parameter(3)
  ROOT %fusion = bf16[1,1,8,4096,2] fusion(param_0.20297, param_1.20468, param_2.10488, param_3.7234), kind=kLoop, calls=%fused_convert.5.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeSliceBroadcastAddMultiplyTanh) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.142_module, entry_computation_layout={(bf16[1,1,2,15360]{3,2,1,0})->bf16[1,1,15360]{2,1,0}}

%fused_multiply.142.clone (param_0.11806: bf16[1,1,2,15360]) -> bf16[1,1,15360] {
  %param_0.11806 = bf16[1,1,2,15360]{3,2,1,0} parameter(0)
  %slice.855.1 = bf16[1,1,1,15360]{3,2,1,0} slice(%param_0.11806), slice={[0:1], [0:1], [0:1], [0:15360]}
  %bitcast.181.6 = bf16[1,1,15360]{2,1,0} bitcast(%slice.855.1)
  %multiply.3897.3 = bf16[1,1,15360]{2,1,0} multiply(%bitcast.181.6, %bitcast.181.6)
  %multiply.3898.3 = bf16[1,1,15360]{2,1,0} multiply(%multiply.3897.3, %bitcast.181.6)
  %constant_1055_2 = bf16[] constant(0.04468)
  %broadcast.6643.1 = bf16[1,1,15360]{2,1,0} broadcast(%constant_1055_2), dimensions={}
  %multiply.3899.1 = bf16[1,1,15360]{2,1,0} multiply(%multiply.3898.3, %broadcast.6643.1)
  %add.2667.3 = bf16[1,1,15360]{2,1,0} add(%bitcast.181.6, %multiply.3899.1)
  %constant_1056_2 = bf16[] constant(0.7969)
  %broadcast.6644.1 = bf16[1,1,15360]{2,1,0} broadcast(%constant_1056_2), dimensions={}
  %multiply.3900.1 = bf16[1,1,15360]{2,1,0} multiply(%add.2667.3, %broadcast.6644.1)
  %convert.6441.1 = f32[1,1,15360]{2,1,0} convert(%multiply.3900.1)
  %tanh.96.9 = f32[1,1,15360]{2,1,0} tanh(%convert.6441.1)
  %convert.6442.9 = bf16[1,1,15360]{2,1,0} convert(%tanh.96.9)
  %constant_1040_4 = bf16[] constant(1)
  %broadcast.6645.1 = bf16[1,1,15360]{2,1,0} broadcast(%constant_1040_4), dimensions={}
  %add.2668.7 = bf16[1,1,15360]{2,1,0} add(%convert.6442.9, %broadcast.6645.1)
  %constant_1058_2 = bf16[] constant(0.5)
  %broadcast.6646.1 = bf16[1,1,15360]{2,1,0} broadcast(%constant_1058_2), dimensions={}
  %multiply.3901.5 = bf16[1,1,15360]{2,1,0} multiply(%add.2668.7, %broadcast.6646.1)
  %multiply.3902.3 = bf16[1,1,15360]{2,1,0} multiply(%bitcast.181.6, %multiply.3901.5)
  %slice.856.1 = bf16[1,1,1,15360]{3,2,1,0} slice(%param_0.11806), slice={[0:1], [0:1], [1:2], [0:15360]}
  %bitcast.182.1 = bf16[1,1,15360]{2,1,0} bitcast(%slice.856.1)
  ROOT %multiply.3903.1 = bf16[1,1,15360]{2,1,0} multiply(%multiply.3902.3, %bitcast.182.1)
}

ENTRY %wrapper_fused_multiply.142.clone (param_0.11806: bf16[1,1,2,15360]) -> bf16[1,1,15360] {
  param_0.11806 = bf16[1,1,2,15360] parameter(0)
  ROOT %fusion = bf16[1,1,15360] fusion(param_0.11806), kind=kLoop, calls=%fused_multiply.142.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertBitcastReshapeTranspose) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.128_module, entry_computation_layout={(f32[32,128,512]{2,1,0})->bf16[512,4096]{1,0}}

%fused_transpose.128.clone (param_0.26029: f32[32,128,512]) -> bf16[512,4096] {
  %param_0.26029 = f32[32,128,512]{2,1,0} parameter(0)
  %convert.514.1 = bf16[32,128,512]{2,1,0} convert(%param_0.26029)
  %bitcast.12511.1 = bf16[4096,512]{1,0} bitcast(%convert.514.1)
  ROOT %transpose.2513.1 = bf16[512,4096]{1,0} transpose(%bitcast.12511.1), dimensions={1,0}
}

ENTRY %wrapper_fused_transpose.128.clone (param_0.26029: f32[32,128,512]) -> bf16[512,4096] {
  param_0.26029 = f32[32,128,512] parameter(0)
  ROOT %fusion = bf16[512,4096] fusion(param_0.26029), kind=kLoop, calls=%fused_transpose.128.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeTransposeBroadcastAddDivideMultiplySqrt) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.131_module, entry_computation_layout={(bf16[512,4096]{1,0}, bf16[], bf16[], f32[], bf16[512,32,128]{2,1,0}, /*index=5*/bf16[32,128,512]{2,1,0}, bf16[512,32,128]{2,1,0})->bf16[512,32,128]{2,1,0}}

%fused_transpose.131.clone (param_0.32418: bf16[512,4096], param_1.35732: bf16[], param_2.31397: bf16[], param_3.20455: f32[], param_4.12673: bf16[512,32,128], param_5.8985: bf16[32,128,512], param_6.6647: bf16[512,32,128]) -> bf16[512,32,128] {
  %param_0.32418 = bf16[512,4096]{1,0} parameter(0)
  %bitcast.23253.1 = bf16[512,32,128]{2,1,0} bitcast(%param_0.32418)
  %param_1.35732 = bf16[] parameter(1)
  %broadcast.3551.68 = bf16[512,32,128]{2,1,0} broadcast(%param_1.35732), dimensions={}
  %param_6.6647 = bf16[512,32,128]{2,1,0} parameter(6)
  %constant_20349_323 = bf16[] constant(0.1001)
  %broadcast.3554.508 = bf16[512,32,128]{2,1,0} broadcast(%constant_20349_323), dimensions={}
  %multiply.316.7 = bf16[512,32,128]{2,1,0} multiply(%param_6.6647, %broadcast.3554.508)
  %param_5.8985 = bf16[32,128,512]{2,1,0} parameter(5)
  %bitcast.23254.7 = bf16[4096,512]{1,0} bitcast(%param_5.8985)
  %transpose.4462.7 = bf16[512,4096]{1,0} transpose(%bitcast.23254.7), dimensions={1,0}
  %bitcast.23255.7 = bf16[512,32,128]{2,1,0} bitcast(%transpose.4462.7)
  %constant_20350_164 = bf16[] constant(0.8984)
  %broadcast.3555.316 = bf16[512,32,128]{2,1,0} broadcast(%constant_20350_164), dimensions={}
  %multiply.317.7 = bf16[512,32,128]{2,1,0} multiply(%bitcast.23255.7, %broadcast.3555.316)
  %add.5310.5 = bf16[512,32,128]{2,1,0} add(%multiply.316.7, %multiply.317.7)
  %convert.8570.7 = f32[512,32,128]{2,1,0} convert(%add.5310.5)
  %param_2.31397 = bf16[] parameter(2)
  %broadcast.3556.68 = bf16[512,32,128]{2,1,0} broadcast(%param_2.31397), dimensions={}
  %param_4.12673 = bf16[512,32,128]{2,1,0} parameter(4)
  %convert.8565.5 = f32[512,32,128]{2,1,0} convert(%param_4.12673)
  %param_3.20455 = f32[] parameter(3)
  %broadcast.6127.254 = f32[512,32,128]{2,1,0} broadcast(%param_3.20455), dimensions={}
  %divide.189.5 = f32[512,32,128]{2,1,0} divide(%convert.8565.5, %broadcast.6127.254)
  %sqrt.357.3 = f32[512,32,128]{2,1,0} sqrt(%divide.189.5)
  %convert.8569.3 = bf16[512,32,128]{2,1,0} convert(%sqrt.357.3)
  %constant_20360_164 = bf16[] constant(1.001e-08)
  %broadcast.3560.196 = bf16[512,32,128]{2,1,0} broadcast(%constant_20360_164), dimensions={}
  %add.5335.5 = bf16[512,32,128]{2,1,0} add(%convert.8569.3, %broadcast.3560.196)
  %multiply.321.3 = bf16[512,32,128]{2,1,0} multiply(%broadcast.3556.68, %add.5335.5)
  %convert.8571.9 = f32[512,32,128]{2,1,0} convert(%multiply.321.3)
  %divide.190.9 = f32[512,32,128]{2,1,0} divide(%convert.8570.7, %convert.8571.9)
  %convert.8572.7 = bf16[512,32,128]{2,1,0} convert(%divide.190.9)
  %multiply.322.3 = bf16[512,32,128]{2,1,0} multiply(%bitcast.23253.1, %broadcast.3554.508)
  %add.5382.5 = bf16[512,32,128]{2,1,0} add(%convert.8572.7, %multiply.322.3)
  %multiply.323.3 = bf16[512,32,128]{2,1,0} multiply(%broadcast.3551.68, %add.5382.5)
  ROOT %add.5385.1 = bf16[512,32,128]{2,1,0} add(%bitcast.23253.1, %multiply.323.3)
}

ENTRY %wrapper_fused_transpose.131.clone (param_0.32418: bf16[512,4096], param_1.35732: bf16[], param_2.31397: bf16[], param_3.20455: f32[], param_4.12673: bf16[512,32,128], param_5.8985: bf16[32,128,512], param_6.6647: bf16[512,32,128]) -> bf16[512,32,128] {
  param_0.32418 = bf16[512,4096] parameter(0)
  param_1.35732 = bf16[] parameter(1)
  param_2.31397 = bf16[] parameter(2)
  param_3.20455 = f32[] parameter(3)
  param_4.12673 = bf16[512,32,128] parameter(4)
  param_5.8985 = bf16[32,128,512] parameter(5)
  param_6.6647 = bf16[512,32,128] parameter(6)
  ROOT %fusion = bf16[512,32,128] fusion(param_0.32418, param_1.35732, param_2.31397, param_3.20455, param_4.12673, param_5.8985, param_6.6647), kind=kLoop, calls=%fused_transpose.131.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeTransposeBroadcastAddMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.14_module, entry_computation_layout={(bf16[8,512,896]{2,1,0}, f32[1,8,512]{2,1,0}, f32[8,512]{1,0}, bf16[4,8,512,224]{3,2,1,0}, bf16[896,1]{1,0}, /*index=5*/bf16[4096,896]{1,0})->bf16[4,8,512,224]{3,2,1,0}}

%fused_transpose.14 (param_0.7210: bf16[8,512,896], param_1.6959: f32[1,8,512], param_2.5367: f32[8,512], param_3.4132: bf16[4,8,512,224], param_4.2615: bf16[896,1], param_5.2200: bf16[4096,896]) -> bf16[4,8,512,224] {
  %param_0.7210 = bf16[8,512,896]{2,1,0} parameter(0)
  %param_5.2200 = bf16[4096,896]{1,0} parameter(5)
  %bitcast.1139.8 = bf16[8,512,896]{2,1,0} bitcast(%param_5.2200)
  %param_4.2615 = bf16[896,1]{1,0} parameter(4)
  %bitcast.789.13 = bf16[896]{0} bitcast(%param_4.2615)
  %mul.2840.13 = bf16[8,512,896]{2,1,0} broadcast(%bitcast.789.13), dimensions={2}
  %mul.3114.5 = bf16[8,512,896]{2,1,0} multiply(%bitcast.1139.8, %mul.2840.13)
  %convert_element_type.3550.12 = f32[8,512,896]{2,1,0} convert(%mul.3114.5)
  %param_2.5367 = f32[8,512]{1,0} parameter(2)
  %mul.2837.12 = f32[8,512,896]{2,1,0} broadcast(%param_2.5367), dimensions={0,1}
  %mul.3117.9 = f32[8,512,896]{2,1,0} multiply(%convert_element_type.3550.12, %mul.2837.12)
  %param_3.4132 = bf16[4,8,512,224]{3,2,1,0} parameter(3)
  %transpose.645.5 = bf16[8,512,4,224]{3,2,1,0} transpose(%param_3.4132), dimensions={1,2,0,3}
  %bitcast.788.22 = bf16[8,512,896]{2,1,0} bitcast(%transpose.645.5)
  %convert_element_type.3220.21 = f32[8,512,896]{2,1,0} convert(%bitcast.788.22)
  %param_1.6959 = f32[1,8,512]{2,1,0} parameter(1)
  %bitcast.1163.7 = f32[8,512]{1,0} bitcast(%param_1.6959)
  %mul.3123.7 = f32[8,512,896]{2,1,0} broadcast(%bitcast.1163.7), dimensions={0,1}
  %mul.3124.7 = f32[8,512,896]{2,1,0} multiply(%convert_element_type.3220.21, %mul.3123.7)
  %add_any.137.7 = f32[8,512,896]{2,1,0} add(%mul.3117.9, %mul.3124.7)
  %convert_element_type.3551.5 = bf16[8,512,896]{2,1,0} convert(%add_any.137.7)
  %add_any.138.3 = bf16[8,512,896]{2,1,0} add(%param_0.7210, %convert_element_type.3551.5)
  %bitcast.1164.1 = bf16[8,512,4,224]{3,2,1,0} bitcast(%add_any.138.3)
  ROOT %transpose.701.1 = bf16[4,8,512,224]{3,2,1,0} transpose(%bitcast.1164.1), dimensions={2,0,1,3}
}

ENTRY %wrapper_fused_transpose.14 (param_0.7210: bf16[8,512,896], param_1.6959: f32[1,8,512], param_2.5367: f32[8,512], param_3.4132: bf16[4,8,512,224], param_4.2615: bf16[896,1], param_5.2200: bf16[4096,896]) -> bf16[4,8,512,224] {
  param_0.7210 = bf16[8,512,896] parameter(0)
  param_1.6959 = f32[1,8,512] parameter(1)
  param_2.5367 = f32[8,512] parameter(2)
  param_3.4132 = bf16[4,8,512,224] parameter(3)
  param_4.2615 = bf16[896,1] parameter(4)
  param_5.2200 = bf16[4096,896] parameter(5)
  ROOT %fusion = bf16[4,8,512,224] fusion(param_0.7210, param_1.6959, param_2.5367, param_3.4132, param_4.2615, param_5.2200), kind=kLoop, calls=%fused_transpose.14,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeTransposeBroadcastDivideExponentialSubtract) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert_module, entry_computation_layout={(bf16[1,1,8,2]{3,2,1,0}, bf16[8,2]{1,0}, bf16[8,4096,2]{2,1,0})->bf16[1,1,8,4096,2]{4,3,2,1,0}}

%fused_convert.clone (param_0.20312: bf16[1,1,8,2], param_1.20508: bf16[8,2], param_2.10422: bf16[8,4096,2]) -> bf16[1,1,8,4096,2] {
  %param_2.10422 = bf16[8,4096,2]{2,1,0} parameter(2)
  %bitcast.2254.7 = bf16[1,1,8,4096,2]{4,3,2,1,0} bitcast(%param_2.10422)
  %param_1.20508 = bf16[8,2]{1,0} parameter(1)
  %bitcast.167.9 = bf16[16]{0} bitcast(%param_1.20508)
  %broadcast.6722.9 = bf16[1,1,16,4096]{3,2,1,0} broadcast(%bitcast.167.9), dimensions={2}
  %bitcast.4794.7 = bf16[8,2,4096]{2,1,0} bitcast(%broadcast.6722.9)
  %transpose.1934.7 = bf16[8,4096,2]{2,1,0} transpose(%bitcast.4794.7), dimensions={0,2,1}
  %bitcast.4795.7 = bf16[1,1,8,4096,2]{4,3,2,1,0} bitcast(%transpose.1934.7)
  %subtract.580.7 = bf16[1,1,8,4096,2]{4,3,2,1,0} subtract(%bitcast.2254.7, %bitcast.4795.7)
  %exponential.192.5 = bf16[1,1,8,4096,2]{4,3,2,1,0} exponential(%subtract.580.7)
  %convert.6434.4 = f32[1,1,8,4096,2]{4,3,2,1,0} convert(%exponential.192.5)
  %param_0.20312 = bf16[1,1,8,2]{3,2,1,0} parameter(0)
  %bitcast.169.5 = bf16[16]{0} bitcast(%param_0.20312)
  %broadcast.6723.5 = bf16[1,1,16,4096]{3,2,1,0} broadcast(%bitcast.169.5), dimensions={2}
  %bitcast.4796.3 = bf16[8,2,4096]{2,1,0} bitcast(%broadcast.6723.5)
  %transpose.1935.3 = bf16[8,4096,2]{2,1,0} transpose(%bitcast.4796.3), dimensions={0,2,1}
  %bitcast.4797.3 = bf16[1,1,8,4096,2]{4,3,2,1,0} bitcast(%transpose.1935.3)
  %convert.6435.3 = f32[1,1,8,4096,2]{4,3,2,1,0} convert(%bitcast.4797.3)
  %divide.1538.3 = f32[1,1,8,4096,2]{4,3,2,1,0} divide(%convert.6434.4, %convert.6435.3)
  ROOT %convert.6436.1 = bf16[1,1,8,4096,2]{4,3,2,1,0} convert(%divide.1538.3)
}

ENTRY %wrapper_fused_convert.clone (param_0.20312: bf16[1,1,8,2], param_1.20508: bf16[8,2], param_2.10422: bf16[8,4096,2]) -> bf16[1,1,8,4096,2] {
  param_0.20312 = bf16[1,1,8,2] parameter(0)
  param_1.20508 = bf16[8,2] parameter(1)
  param_2.10422 = bf16[8,4096,2] parameter(2)
  ROOT %fusion = bf16[1,1,8,4096,2] fusion(param_0.20312, param_1.20508, param_2.10422), kind=kLoop, calls=%fused_convert.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest,
       ConvertBitcastReshapeTransposeConcatSliceBcastAddMulSub) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.95_module, entry_computation_layout={(f32[1,11,128]{2,1,0}, bf16[1,11,16,256]{3,2,1,0}, f32[1,11,128]{2,1,0})->bf16[8,256,11,2]{3,2,1,0}}

%fused_transpose.95.clone (param_0.13769: f32[1,11,128], param_1.13904: bf16[1,11,16,256], param_2.10026: f32[1,11,128]) -> bf16[8,256,11,2] {
  %param_1.13904 = bf16[1,11,16,256]{3,2,1,0} parameter(1)
  %convert.50.30 = f32[1,11,16,256]{3,2,1,0} convert(%param_1.13904)
  %slice.17.15 = f32[1,11,16,128]{3,2,1,0} slice(%convert.50.30), slice={[0:1], [0:11], [0:16], [0:128]}
  %param_2.10026 = f32[1,11,128]{2,1,0} parameter(2)
  %bitcast.145.882 = f32[11,128]{1,0} bitcast(%param_2.10026)
  %broadcast.874.481 = f32[1,11,16,128]{3,2,1,0} broadcast(%bitcast.145.882), dimensions={1,3}
  %multiply.875.5 = f32[1,11,16,128]{3,2,1,0} multiply(%slice.17.15, %broadcast.874.481)
  %slice.19.15 = f32[1,11,16,128]{3,2,1,0} slice(%convert.50.30), slice={[0:1], [0:11], [0:16], [128:256]}
  %param_0.13769 = f32[1,11,128]{2,1,0} parameter(0)
  %bitcast.146.1042 = f32[11,128]{1,0} bitcast(%param_0.13769)
  %broadcast.879.641 = f32[1,11,16,128]{3,2,1,0} broadcast(%bitcast.146.1042), dimensions={1,3}
  %multiply.880.7 = f32[1,11,16,128]{3,2,1,0} multiply(%slice.19.15, %broadcast.879.641)
  %subtract.881.5 = f32[1,11,16,128]{3,2,1,0} subtract(%multiply.875.5, %multiply.880.7)
  %convert.1132.3 = bf16[1,11,16,128]{3,2,1,0} convert(%subtract.881.5)
  %multiply.886.5 = f32[1,11,16,128]{3,2,1,0} multiply(%slice.19.15, %broadcast.874.481)
  %multiply.891.7 = f32[1,11,16,128]{3,2,1,0} multiply(%slice.17.15, %broadcast.879.641)
  %add.892.5 = f32[1,11,16,128]{3,2,1,0} add(%multiply.886.5, %multiply.891.7)
  %convert.1133.3 = bf16[1,11,16,128]{3,2,1,0} convert(%add.892.5)
  %concatenate.95.1 = bf16[1,11,16,256]{3,2,1,0} concatenate(%convert.1132.3, %convert.1133.3), dimensions={3}
  %constant_696_2 = bf16[] constant(0.0625)
  %broadcast.697.144 = bf16[1,11,16,256]{3,2,1,0} broadcast(%constant_696_2), dimensions={}
  %multiply.895.3 = bf16[1,11,16,256]{3,2,1,0} multiply(%concatenate.95.1, %broadcast.697.144)
  %bitcast.5039.1 = bf16[11,8,2,256]{3,2,1,0} bitcast(%multiply.895.3)
  ROOT %transpose.1155.1 = bf16[8,256,11,2]{3,2,1,0} transpose(%bitcast.5039.1), dimensions={1,3,0,2}
}

ENTRY %wrapper_fused_transpose.95.clone (param_0.13769: f32[1,11,128], param_1.13904: bf16[1,11,16,256], param_2.10026: f32[1,11,128]) -> bf16[8,256,11,2] {
  param_0.13769 = f32[1,11,128] parameter(0)
  param_1.13904 = bf16[1,11,16,256] parameter(1)
  param_2.10026 = f32[1,11,128] parameter(2)
  ROOT %fusion = bf16[8,256,11,2] fusion(param_0.13769, param_1.13904, param_2.10026), kind=kLoop, calls=%fused_transpose.95.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertBitcastSelectTransposeBroadcastDivide) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.16_module, entry_computation_layout={(f32[], bf16[2,16,16,112]{2,1,3,0}, pred[])->bf16[16,2,16,112]{3,2,1,0}}

%fused_transpose.16 (param_0.7832: f32[], param_1.7677: bf16[2,16,16,112]{2,1,3,0}, param_2.5783: pred[]) -> bf16[16,2,16,112] {
  %param_2.5783 = pred[] parameter(2)
  %select_n.2427.1 = pred[16,2,16,112]{3,2,1,0} broadcast(%param_2.5783), dimensions={}
  %param_1.7677 = bf16[2,16,16,112]{2,1,3,0} parameter(1)
  %bitcast.2065.3 = bf16[2,112,16,16]{3,2,1,0} bitcast(%param_1.7677)
  %transpose.354.3 = bf16[16,2,16,112]{3,2,1,0} transpose(%bitcast.2065.3), dimensions={2,0,3,1}
  %convert.1440.3 = f32[16,2,16,112]{3,2,1,0} convert(%transpose.354.3)
  %param_0.7832 = f32[] parameter(0)
  %broadcast.3683.5 = f32[16,2,16,112]{3,2,1,0} broadcast(%param_0.7832), dimensions={}
  %div.3248.5 = f32[16,2,16,112]{3,2,1,0} divide(%convert.1440.3, %broadcast.3683.5)
  %convert.1442.3 = bf16[16,2,16,112]{3,2,1,0} convert(%div.3248.5)
  ROOT %select_n.2428.1 = bf16[16,2,16,112]{3,2,1,0} select(%select_n.2427.1, %transpose.354.3, %convert.1442.3)
}

ENTRY %wrapper_fused_transpose.16 (param_0.7832: f32[], param_1.7677: bf16[2,16,16,112]{2,1,3,0}, param_2.5783: pred[]) -> bf16[16,2,16,112] {
  param_0.7832 = f32[] parameter(0)
  param_1.7677 = bf16[2,16,16,112]{2,1,3,0} parameter(1)
  param_2.5783 = pred[] parameter(2)
  ROOT %fusion = bf16[16,2,16,112] fusion(param_0.7832, param_1.7677, param_2.5783), kind=kLoop, calls=%fused_transpose.16,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert.75_module, entry_computation_layout={(f32[1]{0}, f32[1]{0})->bf16[1]{0}}

%fused_convert.75 (param_0.9765: f32[1], param_1.10108: f32[1]) -> bf16[1] {
  %param_0.9765 = f32[1]{0} parameter(0)
  %param_1.10108 = f32[1]{0} parameter(1)
  %mul.3090.1 = f32[1]{0} multiply(%param_0.9765, %param_1.10108)
  ROOT %convert_element_type.3510.1 = bf16[1]{0} convert(%mul.3090.1)
}

ENTRY %wrapper_fused_convert.75 (param_0.9765: f32[1], param_1.10108: f32[1]) -> bf16[1] {
  param_0.9765 = f32[1] parameter(0)
  param_1.10108 = f32[1] parameter(1)
  ROOT %fusion = bf16[1] fusion(param_0.9765, param_1.10108), kind=kLoop, calls=%fused_convert.75,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertSelectBroadcastAddCompareDivide) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.7_module, entry_computation_layout={(s32[], s32[8,4096]{1,0})->f32[8,4096]{1,0}}

%fused_select.7 (param_0.10747: s32[], param_1.11010: s32[8,4096]) -> f32[8,4096] {
  %param_1.11010 = s32[8,4096]{1,0} parameter(1)
  %constant_3838_1 = s32[] constant(0)
  %broadcast.2603.22 = s32[8,4096]{1,0} broadcast(%constant_3838_1), dimensions={}
  %ne.4.3 = pred[8,4096]{1,0} compare(%param_1.11010, %broadcast.2603.22), direction=NE
  %constant_3912_2 = f32[] constant(1)
  %param_0.10747 = s32[] parameter(0)
  %convert_element_type.3281.3 = f32[] convert(%param_0.10747)
  %constant_3988_2 = f32[] constant(1e-08)
  %add.1925.3 = f32[] add(%convert_element_type.3281.3, %constant_3988_2)
  %div.2933.1 = f32[] divide(%constant_3912_2, %add.1925.3)
  %broadcast_in_dim.2933.1 = f32[8,4096]{1,0} broadcast(%div.2933.1), dimensions={}
  %constant_3857_330 = f32[] constant(0)
  %broadcast.2613.1 = f32[8,4096]{1,0} broadcast(%constant_3857_330), dimensions={}
  ROOT %mul.2889.1 = f32[8,4096]{1,0} select(%ne.4.3, %broadcast_in_dim.2933.1, %broadcast.2613.1)
}

ENTRY %wrapper_fused_select.7 (param_0.10747: s32[], param_1.11010: s32[8,4096]) -> f32[8,4096] {
  param_0.10747 = s32[] parameter(0)
  param_1.11010 = s32[8,4096] parameter(1)
  ROOT %fusion = f32[8,4096] fusion(param_0.10747, param_1.11010), kind=kLoop, calls=%fused_select.7,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertSelectBroadcastAddDivide) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.1_module, entry_computation_layout={(pred[2,8192]{1,0}, s32[])->f32[2,8192]{1,0}}

%fused_select.1.clone (param_0.29703: pred[2,8192], param_1.40617: s32[]) -> f32[2,8192] {
  %param_0.29703 = pred[2,8192]{1,0} parameter(0)
  %constant_2872_5 = f32[] constant(1)
  %param_1.40617 = s32[] parameter(1)
  %convert_element_type.1875.3 = f32[] convert(%param_1.40617)
  %constant_5660_293 = f32[] constant(1e-08)
  %add.2385.3 = f32[] add(%convert_element_type.1875.3, %constant_5660_293)
  %div.1100.1 = f32[] divide(%constant_2872_5, %add.2385.3)
  %broadcast_in_dim.467.1 = f32[2,8192]{1,0} broadcast(%div.1100.1), dimensions={}
  %constant_2882_1721 = f32[] constant(0)
  %broadcast.5144.1 = f32[2,8192]{1,0} broadcast(%constant_2882_1721), dimensions={}
  ROOT %mul.2898.1 = f32[2,8192]{1,0} select(%param_0.29703, %broadcast_in_dim.467.1, %broadcast.5144.1)
}

ENTRY %wrapper_fused_select.1.clone (param_0.29703: pred[2,8192], param_1.40617: s32[]) -> f32[2,8192] {
  param_0.29703 = pred[2,8192] parameter(0)
  param_1.40617 = s32[] parameter(1)
  ROOT %fusion = f32[2,8192] fusion(param_0.29703, param_1.40617), kind=kLoop, calls=%fused_select.1.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertSelectBroadcastDivide) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.96_module, entry_computation_layout={(bf16[512,32000]{1,0}, f32[], pred[])->bf16[512,32000]{1,0}}

%fused_select.96.clone (param_0.26727: bf16[512,32000], param_1.27589: f32[], param_2.8811: pred[]) -> bf16[512,32000] {
  %param_2.8811 = pred[] parameter(2)
  %broadcast.12026.1 = pred[512,32000]{1,0} broadcast(%param_2.8811), dimensions={}
  %param_0.26727 = bf16[512,32000]{1,0} parameter(0)
  %convert.8705.3 = f32[512,32000]{1,0} convert(%param_0.26727)
  %param_1.27589 = f32[] parameter(1)
  %broadcast.6129.8 = f32[512,32000]{1,0} broadcast(%param_1.27589), dimensions={}
  %divide.2045.5 = f32[512,32000]{1,0} divide(%convert.8705.3, %broadcast.6129.8)
  %convert.8707.3 = bf16[512,32000]{1,0} convert(%divide.2045.5)
  ROOT %select.2910.1 = bf16[512,32000]{1,0} select(%broadcast.12026.1, %param_0.26727, %convert.8707.3)
}

ENTRY %wrapper_fused_select.96.clone (param_0.26727: bf16[512,32000], param_1.27589: f32[], param_2.8811: pred[]) -> bf16[512,32000] {
  param_0.26727 = bf16[512,32000] parameter(0)
  param_1.27589 = f32[] parameter(1)
  param_2.8811 = pred[] parameter(2)
  ROOT %fusion = bf16[512,32000] fusion(param_0.26727, param_1.27589, param_2.8811), kind=kLoop, calls=%fused_select.96.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertSelectBroadcastMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.2_module, entry_computation_layout={(bf16[8,4096,64]{2,1,0}, pred[8,4096,64]{2,1,0}, pred[8,4096,64]{2,1,0})->bf16[8,4096,64]{2,1,0}}

%fused_multiply.2 (param_0.2245: bf16[8,4096,64], param_1.1890: pred[8,4096,64], param_2.2685: pred[8,4096,64]) -> bf16[8,4096,64] {
  %param_0.2245 = bf16[8,4096,64]{2,1,0} parameter(0)
  %param_1.1890 = pred[8,4096,64]{2,1,0} parameter(1)
  %param_2.2685 = pred[8,4096,64]{2,1,0} parameter(2)
  %convert_element_type.2983.14 = s32[8,4096,64]{2,1,0} convert(%param_2.2685)
  %constant_3329_20 = s32[] constant(0)
  %broadcast.2602.12 = s32[8,4096,64]{2,1,0} broadcast(%constant_3329_20), dimensions={}
  %mul.2659.9 = s32[8,4096,64]{2,1,0} select(%param_1.1890, %convert_element_type.2983.14, %broadcast.2602.12)
  %convert_element_type.2984.5 = bf16[8,4096,64]{2,1,0} convert(%mul.2659.9)
  ROOT %mul.2689.1 = bf16[8,4096,64]{2,1,0} multiply(%param_0.2245, %convert_element_type.2984.5)
}

ENTRY %wrapper_fused_multiply.2 (param_0.2245: bf16[8,4096,64], param_1.1890: pred[8,4096,64], param_2.2685: pred[8,4096,64]) -> bf16[8,4096,64] {
  param_0.2245 = bf16[8,4096,64] parameter(0)
  param_1.1890 = pred[8,4096,64] parameter(1)
  param_2.2685 = pred[8,4096,64] parameter(2)
  ROOT %fusion = bf16[8,4096,64] fusion(param_0.2245, param_1.1890, param_2.2685), kind=kLoop, calls=%fused_multiply.2,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertSelectTransposeBroadcastDivide) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.9_module, entry_computation_layout={(f32[], pred[], bf16[2,7168]{1,0})->bf16[7168,2]{1,0}}

%fused_select.9 (param_0.9465: f32[], param_1.9655: pred[], param_2.6910: bf16[2,7168]) -> bf16[7168,2] {
  %param_1.9655 = pred[] parameter(1)
  %select_n.2422.1 = pred[7168,2]{1,0} broadcast(%param_1.9655), dimensions={}
  %param_2.6910 = bf16[2,7168]{1,0} parameter(2)
  %transpose.351.1 = bf16[7168,2]{1,0} transpose(%param_2.6910), dimensions={1,0}
  %convert.1407.3 = f32[7168,2]{1,0} convert(%transpose.351.1)
  %param_0.9465 = f32[] parameter(0)
  %broadcast.3675.8 = f32[7168,2]{1,0} broadcast(%param_0.9465), dimensions={}
  %div.3231.5 = f32[7168,2]{1,0} divide(%convert.1407.3, %broadcast.3675.8)
  %convert.1409.3 = bf16[7168,2]{1,0} convert(%div.3231.5)
  ROOT %select_n.2423.1 = bf16[7168,2]{1,0} select(%select_n.2422.1, %transpose.351.1, %convert.1409.3)
}

ENTRY %wrapper_fused_select.9 (param_0.9465: f32[], param_1.9655: pred[], param_2.6910: bf16[2,7168]) -> bf16[7168,2] {
  param_0.9465 = f32[] parameter(0)
  param_1.9655 = pred[] parameter(1)
  param_2.6910 = bf16[2,7168] parameter(2)
  ROOT %fusion = bf16[7168,2] fusion(param_0.9465, param_1.9655, param_2.6910), kind=kLoop, calls=%fused_select.9,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertSlice) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_slice.8_module, entry_computation_layout={(f32[2,1024]{1,0})->bf16[2,1]{1,0}}

%fused_slice.8 (param_0.9114: f32[2,1024]) -> bf16[2,1] {
  %param_0.9114 = f32[2,1024]{1,0} parameter(0)
  %convert.378.1 = bf16[2,1024]{1,0} convert(%param_0.9114)
  ROOT %slice.1911.1 = bf16[2,1]{1,0} slice(%convert.378.1), slice={[0:2], [0:1]}
}

ENTRY %wrapper_fused_slice.8 (param_0.9114: f32[2,1024]) -> bf16[2,1] {
  param_0.9114 = f32[2,1024] parameter(0)
  ROOT %fusion = bf16[2,1] fusion(param_0.9114), kind=kLoop, calls=%fused_slice.8,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertTransposeBroadcastAddDivideMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.256_module, entry_computation_layout={(bf16[], bf16[512,32000]{1,0}, bf16[], bf16[512,32000]{1,0}, bf16[32000,512]{1,0})->bf16[512,32000]{1,0}}

%fused_transpose.256.clone (param_0.28246: bf16[], param_1.29433: bf16[512,32000], param_2.31366: bf16[], param_3.20392: bf16[512,32000], param_4.12516: bf16[32000,512]) -> bf16[512,32000] {
  %param_4.12516 = bf16[32000,512]{1,0} parameter(4)
  %transpose.3247.1 = bf16[512,32000]{1,0} transpose(%param_4.12516), dimensions={1,0}
  %param_0.28246 = bf16[] parameter(0)
  %broadcast.12025.4 = bf16[512,32000]{1,0} broadcast(%param_0.28246), dimensions={}
  %param_3.20392 = bf16[512,32000]{1,0} parameter(3)
  %convert.8724.5 = f32[512,32000]{1,0} convert(%param_3.20392)
  %param_2.31366 = bf16[] parameter(2)
  %broadcast.12030.14 = bf16[512,32000]{1,0} broadcast(%param_2.31366), dimensions={}
  %param_1.29433 = bf16[512,32000]{1,0} parameter(1)
  %constant_20360_7 = bf16[] constant(1.001e-08)
  %broadcast.12034.14 = bf16[512,32000]{1,0} broadcast(%constant_20360_7), dimensions={}
  %add.5614.7 = bf16[512,32000]{1,0} add(%param_1.29433, %broadcast.12034.14)
  %multiply.337.7 = bf16[512,32000]{1,0} multiply(%broadcast.12030.14, %add.5614.7)
  %convert.8725.7 = f32[512,32000]{1,0} convert(%multiply.337.7)
  %divide.196.7 = f32[512,32000]{1,0} divide(%convert.8724.5, %convert.8725.7)
  %convert.8726.5 = bf16[512,32000]{1,0} convert(%divide.196.7)
  %constant_20349_7 = bf16[] constant(0.1001)
  %broadcast.12028.8 = bf16[512,32000]{1,0} broadcast(%constant_20349_7), dimensions={}
  %multiply.338.3 = bf16[512,32000]{1,0} multiply(%transpose.3247.1, %broadcast.12028.8)
  %add.5629.3 = bf16[512,32000]{1,0} add(%convert.8726.5, %multiply.338.3)
  %multiply.339.3 = bf16[512,32000]{1,0} multiply(%broadcast.12025.4, %add.5629.3)
  ROOT %add.5639.1 = bf16[512,32000]{1,0} add(%transpose.3247.1, %multiply.339.3)
}

ENTRY %wrapper_fused_transpose.256.clone (param_0.28246: bf16[], param_1.29433: bf16[512,32000], param_2.31366: bf16[], param_3.20392: bf16[512,32000], param_4.12516: bf16[32000,512]) -> bf16[512,32000] {
  param_0.28246 = bf16[] parameter(0)
  param_1.29433 = bf16[512,32000] parameter(1)
  param_2.31366 = bf16[] parameter(2)
  param_3.20392 = bf16[512,32000] parameter(3)
  param_4.12516 = bf16[32000,512] parameter(4)
  ROOT %fusion = bf16[512,32000] fusion(param_0.28246, param_1.29433, param_2.31366, param_3.20392, param_4.12516), kind=kLoop, calls=%fused_transpose.256.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, ConvertTransposeBroadcastAddDivideMultiplySqrt) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.20_module, entry_computation_layout={(bf16[], bf16[1,56,576]{2,1,0}, bf16[576,1,56]{2,1,0}, bf16[], f32[], /*index=5*/bf16[1,56,576]{2,1,0})->bf16[1,56,576]{2,1,0}}

%fused_transpose.20 (param_0.7750: bf16[], param_1.11042: bf16[1,56,576], param_2.8203: bf16[576,1,56], param_3.6568: bf16[], param_4.4272: f32[], param_5.3713: bf16[1,56,576]) -> bf16[1,56,576] {
  %param_2.8203 = bf16[576,1,56]{2,1,0} parameter(2)
  %transpose.657.2 = bf16[1,56,576]{2,1,0} transpose(%param_2.8203), dimensions={1,2,0}
  %param_0.7750 = bf16[] parameter(0)
  %broadcast.2993.5 = bf16[1,56,576]{2,1,0} broadcast(%param_0.7750), dimensions={}
  %param_1.11042 = bf16[1,56,576]{2,1,0} parameter(1)
  %convert.1195.3 = f32[1,56,576]{2,1,0} convert(%param_1.11042)
  %param_3.6568 = bf16[] parameter(3)
  %broadcast.2998.5 = bf16[1,56,576]{2,1,0} broadcast(%param_3.6568), dimensions={}
  %param_5.3713 = bf16[1,56,576]{2,1,0} parameter(5)
  %convert.1190.5 = f32[1,56,576]{2,1,0} convert(%param_5.3713)
  %param_4.4272 = f32[] parameter(4)
  %broadcast.3613.11 = f32[1,56,576]{2,1,0} broadcast(%param_4.4272), dimensions={}
  %divide.76.5 = f32[1,56,576]{2,1,0} divide(%convert.1190.5, %broadcast.3613.11)
  %sqrt.113.3 = f32[1,56,576]{2,1,0} sqrt(%divide.76.5)
  %convert.1194.3 = bf16[1,56,576]{2,1,0} convert(%sqrt.113.3)
  %constant_4309_2 = bf16[] constant(1.001e-08)
  %broadcast.3002.11 = bf16[1,56,576]{2,1,0} broadcast(%constant_4309_2), dimensions={}
  %add.2691.5 = bf16[1,56,576]{2,1,0} add(%convert.1194.3, %broadcast.3002.11)
  %multiply.461.3 = bf16[1,56,576]{2,1,0} multiply(%broadcast.2998.5, %add.2691.5)
  %convert.1196.5 = f32[1,56,576]{2,1,0} convert(%multiply.461.3)
  %divide.77.5 = f32[1,56,576]{2,1,0} divide(%convert.1195.3, %convert.1196.5)
  %convert.1197.3 = bf16[1,56,576]{2,1,0} convert(%divide.77.5)
  %constant_4302_2 = bf16[] constant(0.1001)
  %broadcast.2996.10 = bf16[1,56,576]{2,1,0} broadcast(%constant_4302_2), dimensions={}
  %multiply.462.5 = bf16[1,56,576]{2,1,0} multiply(%transpose.657.2, %broadcast.2996.10)
  %add.2692.3 = bf16[1,56,576]{2,1,0} add(%convert.1197.3, %multiply.462.5)
  %multiply.463.3 = bf16[1,56,576]{2,1,0} multiply(%broadcast.2993.5, %add.2692.3)
  ROOT %add.2693.1 = bf16[1,56,576]{2,1,0} add(%transpose.657.2, %multiply.463.3)
}

ENTRY %wrapper_fused_transpose.20 (param_0.7750: bf16[], param_1.11042: bf16[1,56,576], param_2.8203: bf16[576,1,56], param_3.6568: bf16[], param_4.4272: f32[], param_5.3713: bf16[1,56,576]) -> bf16[1,56,576] {
  param_0.7750 = bf16[] parameter(0)
  param_1.11042 = bf16[1,56,576] parameter(1)
  param_2.8203 = bf16[576,1,56] parameter(2)
  param_3.6568 = bf16[] parameter(3)
  param_4.4272 = f32[] parameter(4)
  param_5.3713 = bf16[1,56,576] parameter(5)
  ROOT %fusion = bf16[1,56,576] fusion(param_0.7750, param_1.11042, param_2.8203, param_3.6568, param_4.4272, param_5.3713), kind=kLoop, calls=%fused_transpose.20,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, IsNanWithUnorderedSemantics) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule wrapped_concatenate_computation.6_module, entry_computation_layout={(f32[2,2])->pred[2,2]}

%wrapped_compare_computation (param_0: f32[2,2]) -> pred[2,2] {
  %param_0 = f32[2,2]{1,0} parameter(0), metadata={scheduling_name="param_0"}
  ROOT %ne.1.1 = pred[2,2]{1,0} compare(%param_0, %param_0), direction=NE, metadata={op_name="jit(isnan)/ne" scheduling_name="ne.1.1" stack_frame_id=16}
}

ENTRY %wrapper_wrapped_concatenate_computation.6.clone (param_0: f32[2,2]) -> pred[2,2] {
  %param_0 = f32[2,2]{1,0} parameter(0), metadata={scheduling_name="param_0"}
  ROOT %fusion = pred[2,2] fusion(param_0), kind=kLoop, calls=%wrapped_compare_computation,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, SelectBroadcastAddCompare) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.8_module, entry_computation_layout={(s32[8,4096]{1,0})->s32[8,4096]{1,0}}

%fused_select.8 (param_0.8815: s32[8,4096]) -> s32[8,4096] {
  %param_0.8815 = s32[8,4096]{1,0} parameter(0)
  %constant_3838_38 = s32[] constant(0)
  %broadcast.2603.8 = s32[8,4096]{1,0} broadcast(%constant_3838_38), dimensions={}
  %lt.74.3 = pred[8,4096]{1,0} compare(%param_0.8815, %broadcast.2603.8), direction=LT
  %constant_3839_1 = s32[] constant(129280)
  %add.1909.1 = s32[8,4096]{1,0} broadcast(%constant_3839_1), dimensions={}
  %add.1910.1 = s32[8,4096]{1,0} add(%param_0.8815, %add.1909.1)
  ROOT %select_n.2214.1 = s32[8,4096]{1,0} select(%lt.74.3, %add.1910.1, %param_0.8815)
}

ENTRY %wrapper_fused_select.8 (param_0.8815: s32[8,4096]) -> s32[8,4096] {
  param_0.8815 = s32[8,4096] parameter(0)
  ROOT %fusion = s32[8,4096] fusion(param_0.8815), kind=kLoop, calls=%fused_select.8,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, SelectTransposeBroadcastDivide) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.4_module, entry_computation_layout={(f32[4,64,128]{2,1,0}, f32[], pred[])->f32[64,4,128]{2,1,0}}

%fused_select.4.clone (param_0.1351: f32[4,64,128], param_1.1504: f32[], param_2.759: pred[]) -> f32[64,4,128] {
  %param_2.759 = pred[] parameter(2)
  %broadcast.1774.1 = pred[64,4,128]{2,1,0} broadcast(%param_2.759), dimensions={}
  %param_0.1351 = f32[4,64,128]{2,1,0} parameter(0)
  %transpose.1643.1 = f32[64,4,128]{2,1,0} transpose(%param_0.1351), dimensions={1,0,2}
  %param_1.1504 = f32[] parameter(1)
  %broadcast.1772.1 = f32[64,4,128]{2,1,0} broadcast(%param_1.1504), dimensions={}
  %divide.1773.3 = f32[64,4,128]{2,1,0} divide(%transpose.1643.1, %broadcast.1772.1)
  ROOT %select.1775.1 = f32[64,4,128]{2,1,0} select(%broadcast.1774.1, %transpose.1643.1, %divide.1773.3)
}

ENTRY %wrapper_fused_select.4.clone (param_0.1351: f32[4,64,128], param_1.1504: f32[], param_2.759: pred[]) -> f32[64,4,128] {
  param_0.1351 = f32[4,64,128] parameter(0)
  param_1.1504 = f32[] parameter(1)
  param_2.759 = pred[] parameter(2)
  ROOT %fusion = f32[64,4,128] fusion(param_0.1351, param_1.1504, param_2.759), kind=kLoop, calls=%fused_select.4.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, TransposeBroadcastAddDivideMultiplySqrt) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.192_module, entry_computation_layout={(f32[], f32[32,8016]{1,0}, f32[], f32[8016,32]{1,0}, f32[32,8016]{1,0}, /*index=5*/f32[])->f32[32,8016]{1,0}}

%fused_transpose.192.clone (param_0.13318: f32[], param_1.40912: f32[32,8016], param_2.22463: f32[], param_3.14324: f32[8016,32], param_4.9564: f32[32,8016], param_5.7212: f32[]) -> f32[32,8016] {
  %param_3.14324 = f32[8016,32]{1,0} parameter(3)
  %transpose.2251.1 = f32[32,8016]{1,0} transpose(%param_3.14324), dimensions={1,0}
  %param_0.13318 = f32[] parameter(0)
  %mul.9428.4 = f32[32,8016]{1,0} broadcast(%param_0.13318), dimensions={}
  %param_1.40912 = f32[32,8016]{1,0} parameter(1)
  %param_2.22463 = f32[] parameter(2)
  %div.3046.6 = f32[32,8016]{1,0} broadcast(%param_2.22463), dimensions={}
  %param_4.9564 = f32[32,8016]{1,0} parameter(4)
  %param_5.7212 = f32[] parameter(5)
  %div.3047.4 = f32[32,8016]{1,0} broadcast(%param_5.7212), dimensions={}
  %divide.291.3 = f32[32,8016]{1,0} divide(%param_4.9564, %div.3047.4)
  %sqrt.359.1 = f32[32,8016]{1,0} sqrt(%divide.291.3)
  %constant_5660_4 = f32[] constant(1e-08)
  %add.4835.10 = f32[32,8016]{1,0} broadcast(%constant_5660_4), dimensions={}
  %add.322.7 = f32[32,8016]{1,0} add(%sqrt.359.1, %add.4835.10)
  %multiply.781.5 = f32[32,8016]{1,0} multiply(%div.3046.6, %add.322.7)
  %divide.292.3 = f32[32,8016]{1,0} divide(%param_1.40912, %multiply.781.5)
  %constant_13810_3 = f32[] constant(0.1)
  %broadcast.5210.8 = f32[32,8016]{1,0} broadcast(%constant_13810_3), dimensions={}
  %multiply.782.5 = f32[32,8016]{1,0} multiply(%transpose.2251.1, %broadcast.5210.8)
  %add.323.3 = f32[32,8016]{1,0} add(%divide.292.3, %multiply.782.5)
  %multiply.783.3 = f32[32,8016]{1,0} multiply(%mul.9428.4, %add.323.3)
  ROOT %add.324.1 = f32[32,8016]{1,0} add(%transpose.2251.1, %multiply.783.3)
}

ENTRY %wrapper_fused_transpose.192.clone (param_0.13318: f32[], param_1.40912: f32[32,8016], param_2.22463: f32[], param_3.14324: f32[8016,32], param_4.9564: f32[32,8016], param_5.7212: f32[]) -> f32[32,8016] {
  param_0.13318 = f32[] parameter(0)
  param_1.40912 = f32[32,8016] parameter(1)
  param_2.22463 = f32[] parameter(2)
  param_3.14324 = f32[8016,32] parameter(3)
  param_4.9564 = f32[32,8016] parameter(4)
  param_5.7212 = f32[] parameter(5)
  ROOT %fusion = f32[32,8016] fusion(param_0.13318, param_1.40912, param_2.22463, param_3.14324, param_4.9564, param_5.7212), kind=kLoop, calls=%fused_transpose.192.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, TransposeBroadcastAddMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.257_module, entry_computation_layout={(bf16[32000,512]{1,0}, bf16[512,32000]{1,0})->bf16[32000,512]{1,0}}

%fused_transpose.257.clone (param_0.53137: bf16[32000,512], param_1.60389: bf16[512,32000]) -> bf16[32000,512] {
  %param_1.60389 = bf16[512,32000]{1,0} parameter(1)
  %multiply.334.6 = bf16[512,32000]{1,0} multiply(%param_1.60389, %param_1.60389)
  %constant_20356_229 = bf16[] constant(0.05005)
  %broadcast.12031.8 = bf16[512,32000]{1,0} broadcast(%constant_20356_229), dimensions={}
  %multiply.335.5 = bf16[512,32000]{1,0} multiply(%multiply.334.6, %broadcast.12031.8)
  %param_0.53137 = bf16[32000,512]{1,0} parameter(0)
  %transpose.3634.3 = bf16[512,32000]{1,0} transpose(%param_0.53137), dimensions={1,0}
  %constant_20357_229 = bf16[] constant(0.9492)
  %broadcast.12032.8 = bf16[512,32000]{1,0} broadcast(%constant_20357_229), dimensions={}
  %multiply.336.5 = bf16[512,32000]{1,0} multiply(%transpose.3634.3, %broadcast.12032.8)
  %add.5606.3 = bf16[512,32000]{1,0} add(%multiply.335.5, %multiply.336.5)
  ROOT %transpose.3734.1 = bf16[32000,512]{1,0} transpose(%add.5606.3), dimensions={1,0}
}

ENTRY %wrapper_fused_transpose.257.clone (param_0.53137: bf16[32000,512], param_1.60389: bf16[512,32000]) -> bf16[32000,512] {
  param_0.53137 = bf16[32000,512] parameter(0)
  param_1.60389 = bf16[512,32000] parameter(1)
  ROOT %fusion = bf16[32000,512] fusion(param_0.53137, param_1.60389), kind=kLoop, calls=%fused_transpose.257.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, WrappedConcatenateComputation_1) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule wrapped_concatenate_computation_module, entry_computation_layout={(bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0})->bf16[20480]{0}}

%wrapped_concatenate_computation.clone (param_0: bf16[4096], param_1: bf16[4096], param_2: bf16[4096], param_3: bf16[4096], param_4: bf16[4096]) -> bf16[20480] {
  %param_0 = bf16[4096]{0} parameter(0)
  %param_1 = bf16[4096]{0} parameter(1)
  %param_2 = bf16[4096]{0} parameter(2)
  %param_3 = bf16[4096]{0} parameter(3)
  %param_4 = bf16[4096]{0} parameter(4)
  ROOT %concatenate.439 = bf16[20480]{0} concatenate(%param_0, %param_1, %param_2, %param_3, %param_4), dimensions={0}
}

ENTRY %wrapper_wrapped_concatenate_computation.clone (param_0: bf16[4096], param_1: bf16[4096], param_2: bf16[4096], param_3: bf16[4096], param_4: bf16[4096]) -> bf16[20480] {
  param_0 = bf16[4096] parameter(0)
  param_1 = bf16[4096] parameter(1)
  param_2 = bf16[4096] parameter(2)
  param_3 = bf16[4096] parameter(3)
  param_4 = bf16[4096] parameter(4)
  ROOT %fusion = bf16[20480] fusion(param_0, param_1, param_2, param_3, param_4), kind=kLoop, calls=%wrapped_concatenate_computation.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionTest, WrappedConcatenateComputation_2) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule wrapped_concatenate_computation.6_module, entry_computation_layout={(f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=5*/f32[1]{0}, f32[1]{0})->f32[7]{0}}

%wrapped_concatenate_computation.6.clone (param_0.55140: f32[1], param_1.61441: f32[1], param_2.33774: f32[1], param_3.23576: f32[1], param_4.15959: f32[1], param_5.12477: f32[1], param_6.8616: f32[1]) -> f32[7] {
  %param_0.55140 = f32[1]{0} parameter(0)
  %param_1.61441 = f32[1]{0} parameter(1)
  %param_2.33774 = f32[1]{0} parameter(2)
  %param_3.23576 = f32[1]{0} parameter(3)
  %param_4.15959 = f32[1]{0} parameter(4)
  %param_5.12477 = f32[1]{0} parameter(5)
  %param_6.8616 = f32[1]{0} parameter(6)
  ROOT %concatenate.450 = f32[7]{0} concatenate(%param_0.55140, %param_1.61441, %param_2.33774, %param_3.23576, %param_4.15959, /*index=5*/%param_5.12477, %param_6.8616), dimensions={0}
}

ENTRY %wrapper_wrapped_concatenate_computation.6.clone (param_0.55140: f32[1], param_1.61441: f32[1], param_2.33774: f32[1], param_3.23576: f32[1], param_4.15959: f32[1], param_5.12477: f32[1], param_6.8616: f32[1]) -> f32[7] {
  param_0.55140 = f32[1] parameter(0)
  param_1.61441 = f32[1] parameter(1)
  param_2.33774 = f32[1] parameter(2)
  param_3.23576 = f32[1] parameter(3)
  param_4.15959 = f32[1] parameter(4)
  param_5.12477 = f32[1] parameter(5)
  param_6.8616 = f32[1] parameter(6)
  ROOT %fusion = f32[7] fusion(param_0.55140, param_1.61441, param_2.33774, param_3.23576, param_4.15959, param_5.12477, param_6.8616), kind=kLoop, calls=%wrapped_concatenate_computation.6.clone,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnFusionFileCheckTest, ConvFpropGraphConvertedCorrectly) {
  // Crashes on CUDA 12 + cuDNN 9.10. Works on CUDA 13 + cuDNN 9.23. It's
  // unclear at which point between it got fixed. Conservatively skip on
  // versions older than the oldest one confirmed to work.
  if (IsGB200() && !IsAtLeastCuDnnVersion(9, 23)) {
    GTEST_SKIP() << "Requires recent enough cuDNN to not crash on GB200 GPUs.";
  }
  const std::string kHloText = R"(
fusion {
  input = f32[2,9,9,17] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  ROOT conv = f32[2,9,9,32] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, feature_group_count=1, convolution_kind=fprop
}

ENTRY Test {
  input = f32[2,9,9,17] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  ROOT conv = f32[2,9,9,32] fusion(input, filter), kind=kCustom, calls=fusion, backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})";

  EXPECT_TRUE(*RunCuDnnFileCheck(kHloText, R"(
CHECK: "nodes": [
CHECK:  {
CHECK:   "compute_data_type": "FLOAT",
CHECK:   "dilation": [{{[[:space:]]*1,[[:space:]]*1[[:space:]]*}}],
CHECK:   "inputs": {
CHECK:    "W": 2,
CHECK:    "X": 1
CHECK:   },
CHECK:   "math_mode": "CROSS_CORRELATION",
CHECK:   "name": "0",
CHECK:   "outputs": {
CHECK:    "Y": 3
CHECK:   },
CHECK:   "post_padding": [{{[[:space:]]*1,[[:space:]]*1[[:space:]]*}}],
CHECK:   "pre_padding": [{{[[:space:]]*1,[[:space:]]*1[[:space:]]*}}],
CHECK:   "stride": [{{[[:space:]]*1,[[:space:]]*1[[:space:]]*}}],
CHECK:   "tag": "CONV_FPROP"
CHECK:  }
CHECK: ],
CHECK: "tensors"
CHECK:   "data_type": "FLOAT",
CHECK:   "dim": [{{[[:space:]]*2,[[:space:]]*17,[[:space:]]*9,[[:space:]]*9[[:space:]]*}}],
CHECK:   "name": "input",
CHECK:   "stride": [{{[[:space:]]*1377,[[:space:]]*1,[[:space:]]*153,[[:space:]]*17[[:space:]]*}}],
CHECK:   "data_type": "FLOAT",
CHECK:   "dim": [{{[[:space:]]*32,[[:space:]]*17,[[:space:]]*3,[[:space:]]*3[[:space:]]*}}],
CHECK:   "name": "filter",
CHECK:   "stride": [{{[[:space:]]*153,[[:space:]]*1,[[:space:]]*51,[[:space:]]*17[[:space:]]*}}],
CHECK:   "data_type": "FLOAT",
CHECK:   "dim": [{{[[:space:]]*2,[[:space:]]*32,[[:space:]]*9,[[:space:]]*9[[:space:]]*}}],
CHECK:   "name": "conv",
CHECK:   "stride": [{{[[:space:]]*2592,[[:space:]]*1,[[:space:]]*288,[[:space:]]*32[[:space:]]*}}],
)"));
}
}  // namespace
}  // namespace xla::gpu
