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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/comparison_util.h"
#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class CuDnnFusionTest : public GpuCodegenTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    // Let this group of tests just use first available plan skipping
    // autotuning.
    debug_options.set_xla_gpu_autotune_level(0);
    debug_options.set_xla_gpu_cudnn_gemm_fusion_level(2);
    return debug_options;
  }
  bool IsAtLeastAmpereWithCuDnn9() {
    se::StreamExecutor* executor = backend().default_stream_executor();
    return executor->GetDeviceDescription()
               .cuda_compute_capability()
               .IsAtLeastAmpere() &&
           GetDnnVersionInfoOrDefault(executor).major_version() >= 9;
  }
  bool IsAtLeastCuDnn91() {
    se::StreamExecutor* executor = backend().default_stream_executor();
    const se::dnn::VersionInfo version = GetDnnVersionInfoOrDefault(executor);
    return (version.major_version() == 9 && version.minor_version() >= 1) ||
           version.major_version() > 9;
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
    TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                        ParseAndReturnVerifiedModule(hlo));
    const std::string root_name(
        module->entry_computation()->root_instruction()->name());
    BinaryMap dnn_compiled_graphs;
    CuDnnFusionCompiler cudnn_compiler(*backend().default_stream_executor(),
                                       dnn_compiled_graphs);
    // Run filecheck even if CuDnnFusionCompiler failed.
    cudnn_compiler.Run(module.get()).IgnoreError();
    std::string dump;
    TF_RETURN_IF_ERROR(tsl::ReadFileToString(
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
CHECK:     "A": "p0",
CHECK:     "B": "p1"
CHECK:    },
CHECK:    "outputs": {
CHECK:     "C": "d"
CHECK:    },
CHECK:    "tag": "MATMUL"
CHECK:   }
CHECK:  ],
CHECK:  "tensors": {
CHECK:   "d": {
CHECK:    "data_type": "FLOAT",
CHECK:    "dim": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*64[[:space:]]*}}],
CHECK:    "stride": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*1[[:space:]]*}}],
CHECK:    "uid": 3,
CHECK:    "uid_assigned": true
CHECK:   },
CHECK:   "p0": {
CHECK:    "data_type": "FLOAT",
CHECK:    "dim": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*64[[:space:]]*}}],
CHECK:    "stride": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*1[[:space:]]*}}],
CHECK:    "uid": 1,
CHECK:    "uid_assigned": true
CHECK:   },
CHECK:   "p1": {
CHECK:    "data_type": "FLOAT",
CHECK:    "dim": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*64[[:space:]]*}}],
CHECK:    "stride": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*1[[:space:]]*}}],
CHECK:    "uid": 2,
CHECK:    "uid_assigned": true
CHECK:   }
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

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT _ = f32[32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  BinaryMap dnn_compiled_graphs;
  CuDnnFusionCompiler cudnn_compiler(*backend().default_stream_executor(),
                                     dnn_compiled_graphs);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, cudnn_compiler.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(m::Fusion())));
  EXPECT_THAT(module->entry_computation()
                  ->root_instruction()
                  ->operand(0)
                  ->fused_instructions_computation()
                  ->root_instruction(),
              GmockMatch(m::Tuple(m::Dot(), m::CustomCall())));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest,
       CuDnnFusionCompilerDoesNotFailOnDependentFusions) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "This test case requests a workspace only with cuDNN 9.1+.";
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
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
  CuDnnFusionCompiler cudnn_compiler(*backend().default_stream_executor(),
                                     dnn_compiled_graphs);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, cudnn_compiler.Run(module.get()));
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
CHECK: "out"
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}1,{{[[:space:]]*}}128{{[[:space:]]*}}]
CHECK: "stride": [{{[[:space:]]*}}1,{{[[:space:]]*}}128,{{[[:space:]]*}}1{{[[:space:]]*}}]
CHECK: "p0"
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}1,{{[[:space:]]*}}64{{[[:space:]]*}}]
CHECK: "stride": [{{[[:space:]]*}}1,{{[[:space:]]*}}64,{{[[:space:]]*}}1{{[[:space:]]*}}]
CHECK: "p1"
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}64,{{[[:space:]]*}}128{{[[:space:]]*}}]
CHECK: "stride": [{{[[:space:]]*}}1,{{[[:space:]]*}}128,{{[[:space:]]*}}1{{[[:space:]]*}}]
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
CHECK: "out"
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}256,{{[[:space:]]*}}1{{[[:space:]]*}}]
CHECK: "stride": [{{[[:space:]]*}}1,{{[[:space:]]*}}1,{{[[:space:]]*}}256{{[[:space:]]*}}]
CHECK: "p0"
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}256,{{[[:space:]]*}}64{{[[:space:]]*}}]
CHECK: "stride": [{{[[:space:]]*}}1,{{[[:space:]]*}}1,{{[[:space:]]*}}256{{[[:space:]]*}}]
CHECK: "p1"
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}64,{{[[:space:]]*}}1{{[[:space:]]*}}]
CHECK: "stride": [{{[[:space:]]*}}1,{{[[:space:]]*}}1,{{[[:space:]]*}}64{{[[:space:]]*}}]
  )"));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
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

TEST_F(CuDnnFusionCommandBufferTest, CommandBuffersAreSupported) {
  const std::string kHloText = R"(
fd0 {
  p0 = f32[64,64]{1,0} parameter(0)
  p1 = f32[64,64]{1,0} parameter(1)
  ROOT d = f32[64,64]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

fd1 {
  p0 = f32[64,64]{1,0} parameter(0)
  p1 = f32[64,64]{1,0} parameter(1)
  ROOT d = f32[64,64]{1,0} dot(p0, p1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f32[64,64]{1,0} parameter(0)
  p1 = f32[64,64]{1,0} parameter(1)
  d0 = f32[64,64]{1,0} fusion(p0, p1), kind=kCustom, calls=fd0,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
  a = f32[64,64]{1,0} add(d0, d0)
  ROOT d1 = f32[64,64]{1,0} fusion(a, d0), kind=kCustom, calls=fd1,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
})";

  se::StreamExecutorMemoryAllocator allocator(
      backend().default_stream_executor());
  // Verify that a command buffer is applied.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          backend().compiler()->RunBackend(
                              GetOptimizedModule(kHloText).value(),
                              backend().default_stream_executor(), &allocator));
  absl::StatusOr<bool> filecheck_result =
      RunFileCheck(executable->module().ToString(), R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK: command_buffer
; CHECK-NOT: fusion
)");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.value());

  // Verify that the command buffer executes correctly.
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

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

  EXPECT_TRUE(RunAndCompare(hlo_test, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
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

TEST_F(CuDnnFusionRewriteTest,
       DoNotExecuteGemmFusionWithCuDnnWhenNotSupported) {
  // Dimension size 61 does not satisfy the requirement on alignment
  // (multiple of 2).
  MatchOptimizedHlo(R"(
ENTRY e {
  p0 = f16[20,40,61] parameter(0)
  p0n = f16[20,40,61] negate(p0)
  p1 = f16[20,80,61] parameter(1)
  ROOT r = f16[20,40,80] dot(p0n, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
})",
                    R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: ROOT
; CHECK-SAME: fusion
; CHECK-NOT: cudnn
)");
}

TEST_F(CuDnnFusionRewriteTest, AutotuningPicksCuDnnForS8BF16OnHopper) {
  // The test case relies on measurements by the autotuner and current
  // performance comparison of the backends. May need to be updated if
  // the situation changes.
  if (backend()
          .default_stream_executor()
          ->GetDeviceDescription()
          .cuda_compute_capability() != se::CudaComputeCapability::Hopper()) {
    GTEST_SKIP() << "The test is for Hopper.";
  }
  MatchOptimizedHlo(R"(
e {
  p0 = bf16[720,720,720] parameter(0)
  p1 = s8[720,720,720] parameter(1)
  c = bf16[720,720,720] convert(p1)
  ROOT d = bf16[720,720,720] dot(p0, c),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
})",
                    R"(
; CHECK: __cudnn$fusion
)");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
