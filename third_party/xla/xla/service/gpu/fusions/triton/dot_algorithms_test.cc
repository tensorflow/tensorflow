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

#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusions/triton/kernel_name_tracer.h"
#include "xla/service/gpu/fusions/triton/triton_test_utils.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class AlgorithmTest : public GpuCodegenTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.set_xla_dump_to("sponge");
    debug_options.set_xla_dump_hlo_pass_re(".*");
    debug_options.set_xla_gpu_dump_autotuned_gemm_fusions(true);

    // Enable triton fusion for all supported GEMMs.
    debug_options.set_xla_gpu_triton_gemm_any(true);

    return debug_options;
  }

  std::string HloModuleTestName() const {
    auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    return absl::StrReplaceAll(
        absl::StrCat(test_info->test_suite_name(), "_", test_info->name()),
        {{"/", "_"}});
  }

  stream_executor::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

  const stream_executor::GpuComputeCapability& GpuComputeComp() {
    return device_desc().gpu_compute_capability();
  }
  stream_executor::GpuComputeCapability CudaAmpereOrRocm() {
    if (std::holds_alternative<stream_executor::RocmComputeCapability>(
            GpuComputeComp())) {
      return stream_executor::GpuComputeCapability{
          device_desc().rocm_compute_capability()};
    } else {
      return stream_executor::GpuComputeCapability{
          stream_executor::CudaComputeCapability{
              stream_executor::CudaComputeCapability::AMPERE, 0}};
    }
  }

 protected:
  const stream_executor::DeviceDescription& device_desc() {
    return backend().default_stream_executor()->GetDeviceDescription();
  }
};

// In these tests, we depend on "algorithm" annotations for selecting the 6XBF16
// algorithm.
class Triton6xBF16GemmTest : public AlgorithmTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = AlgorithmTest::GetDebugOptionsForTest();
    // These 2 flags are not strictly necessary now, but we're adding them to be
    // on the safe side against future flakiness.
    //
    // Do not fall back to cuBLAS, we are testing Triton.
    debug_options.set_xla_gpu_cublas_fallback(false);

    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    return debug_options;
  }

 protected:
  void SetUp() override {
    if (!SupportsBF16(GpuComputeComp())) {
      GTEST_SKIP() << "BF16 not supported.";
    }
  }
};

// In these tests, we depend on debug option flags for selecting the 6XBF16
// algorithm.
// TODO(b/316147294): Remove this class and the --xla_gpu_enable_bf16_6way_gemm
// flag after we will support the algorithm values through the entire stack.
class Triton6xBF16GemmTestWithFlag : public AlgorithmTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = AlgorithmTest::GetDebugOptionsForTest();
    // Do not fall back to cuBLAS, we are testing Triton.
    debug_options.set_xla_gpu_cublas_fallback(false);
    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    // Enable bf16_6way gemm to compute F32 matmul.
    debug_options.set_xla_gpu_enable_bf16_6way_gemm(true);
    return debug_options;
  }
};

class BlasAlgorithmTest : public AlgorithmTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = AlgorithmTest::GetDebugOptionsForTest();
    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    return debug_options;
  }
};

class TritonAlgorithmTest : public AlgorithmTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = AlgorithmTest::GetDebugOptionsForTest();
    // Do not fall back to cuBLAS, we are testing Triton.
    debug_options.set_xla_gpu_cublas_fallback(false);
    // Enable gemm for any hlo including pure matmuls.
    debug_options.set_xla_gpu_triton_gemm_any(true);
    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    return debug_options;
  }
};

TEST_F(AlgorithmTest, Algorithm3xBF16) {
  constexpr std::string_view kHloText = R"(
    HloModule Algorithm3xBF16

    ENTRY e {
      p0 = f32[128,128] parameter(0)
      p1 = f32[128,128] parameter(1)
      ROOT dot = f32[128,128] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0},
        algorithm=dot_bf16_bf16_f32_x3
    }
  )";
  EXPECT_TRUE(
      RunAndCompare(kHloText, ErrorSpec{/*aabs=*/0.001, /*arel=*/0.001}));
}

TEST_F(AlgorithmTest, Algorithm6xBF16) {
  constexpr std::string_view kHloText = R"(
    HloModule Algorithm6xBF16

    ENTRY e {
      p0 = f32[128,128] parameter(0)
      p1 = f32[128,128] parameter(1)
      ROOT dot = f32[128,128] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0},
        algorithm=dot_bf16_bf16_f32_x6
    }
  )";
  EXPECT_TRUE(
      RunAndCompare(kHloText, ErrorSpec{/*aabs=*/0.001, /*arel=*/0.001}));
}

TEST_F(BlasAlgorithmTest, Algorithm_BF16_BF16_F32) {
  // We check that the algorithm is propagated to the BLAS call.
  // We also check that the kernel name matches the algorithm for Ampere.
  // The algorithm for Hopper is not the one we expect because it uses TF32.

  if (!SupportsBF16(GpuComputeComp())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  constexpr std::string_view kHloText = R"(
    HloModule Algorithm_BF16_BF16_F32

    ENTRY main {
      lhs = f32[8512,256]{1,0} parameter(0)
      rhs = f32[256,8512]{1,0} parameter(1)
      ROOT dot = f32[8512,8512]{1,0} dot(lhs, rhs),
          algorithm=dot_bf16_bf16_f32,
          lhs_contracting_dims={1},
          rhs_contracting_dims={0}
    }
  )";
  const std::string pattern = R"(
    CHECK:  %convert{{.*}} = bf16[
    CHECK:  %convert{{.*}} = bf16[
    CHECK: "algorithm":"ALG_UNSET"
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), pattern));
  ASSERT_TRUE(ok);

  auto tracer = KernelNameTracer::Create();
  if (tracer == nullptr) {
    GTEST_SKIP() << "KernelNameTracer is not implemented.";
  }
  tracer->start();
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
  auto kernel_names = tracer->stop();

  auto cc = GetCudaComputeCapability();
  using CudaComputeCapabilities =
      stream_executor::CudaComputeCapability::CudaComputeCapabilities;
  switch (cc.major) {
    case CudaComputeCapabilities::BLACKWELL:
      GTEST_SKIP() << "CudaComputeCapabilities::BLACKWELL has the kernel name: "
                   << kernel_names[0];
      break;
    case CudaComputeCapabilities::AMPERE:
      EXPECT_THAT(kernel_names, ::testing::UnorderedElementsAre(
                                    ::testing::Eq("wrapped_convert"),
                                    ::testing::Eq("wrapped_convert_1"),
                                    ::testing::HasSubstr("gemm_bf16_")));
      break;
    case CudaComputeCapabilities::HOPPER:
      // Convert to bf16+cublas works faster than dot with algorithm.
      EXPECT_THAT(kernel_names,
                  ::testing::UnorderedElementsAre(
                      ::testing::Eq("wrapped_convert"),
                      ::testing::Eq("wrapped_convert_1"),
                      ::testing::HasSubstr("gemm_bf16f32_bf16f32")));
      break;
    default:
      GTEST_SKIP() << "Unsupported compute capability: " << cc.major
                   << " has the kernel name: " << kernel_names[0];
  }
}

TEST_F(BlasAlgorithmTest, Algorithm_BF16_BF16_F32_X3) {
  if (!SupportsBF16(GpuComputeComp())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  constexpr std::string_view kHloText = R"(
    HloModule Algorithm_BF16_BF16_F32_X3

    ENTRY main {
      lhs = f32[8512,256]{1,0} parameter(0)
      rhs = f32[256,8512]{1,0} parameter(1)
      ROOT dot = f32[8512,8512]{1,0} dot(lhs, rhs),
          algorithm=dot_bf16_bf16_f32_x3,
          lhs_contracting_dims={1},
          rhs_contracting_dims={0}
    }
  )";
  // Single dot was replaced with 3 dots.
  const std::string pattern = R"(
    CHECK-COUNT-3: custom_call_target="__cublas$gemm"
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), pattern));
  ASSERT_TRUE(ok);

  auto tracer = KernelNameTracer::Create();
  if (tracer == nullptr) {
    GTEST_SKIP() << "KernelNameTracer is not implemented.";
  }
  tracer->start();
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
  auto kernel_names = tracer->stop();

  auto cc = GetCudaComputeCapability();
  using CudaComputeCapabilities =
      stream_executor::CudaComputeCapability::CudaComputeCapabilities;
  switch (cc.major) {
    case CudaComputeCapabilities::BLACKWELL:
      GTEST_SKIP() << "CudaComputeCapabilities::BLACKWELL has the kernel name: "
                   << kernel_names[0];
      break;
    case CudaComputeCapabilities::AMPERE:
      ASSERT_EQ(kernel_names.size(), 1);
      EXPECT_THAT(kernel_names[0], ::testing::Eq("loop_convert_fusion_1"));
      break;
    case CudaComputeCapabilities::HOPPER:
      EXPECT_THAT(kernel_names,
                  ::testing::UnorderedElementsAre(
                      ::testing::Eq("loop_convert_fusion_1"),
                      ::testing::HasSubstr("gemm_bf16f32_bf16f32_f32_")));
      break;
    default:
      GTEST_SKIP() << "Unsupported compute capability: " << cc.major
                   << " has the kernel name: " << kernel_names[0];
  }
}

TEST_F(BlasAlgorithmTest, Algorithm_BF16_BF16_F32_X6) {
  if (!SupportsBF16(GpuComputeComp())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  constexpr std::string_view kHloText = R"(
    HloModule Algorithm_BF16_BF16_F32_X6

    ENTRY main {
      lhs = f32[8512,256]{1,0} parameter(0)
      rhs = f32[256,8512]{1,0} parameter(1)
      ROOT dot = f32[8512,8512]{1,0} dot(lhs, rhs),
          algorithm=dot_bf16_bf16_f32_x6,
          lhs_contracting_dims={1},
          rhs_contracting_dims={0}
    }
  )";
  // Single dot was replaced with 3 dots.
  const std::string pattern = R"(
    CHECK-COUNT-6: custom_call_target="__cublas$gemm"
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), pattern));
  ASSERT_TRUE(ok);

  auto tracer = KernelNameTracer::Create();
  if (tracer == nullptr) {
    GTEST_SKIP() << "KernelNameTracer is not implemented.";
  }
  tracer->start();
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
  auto kernel_names = tracer->stop();

  auto cc = GetCudaComputeCapability();
  using CudaComputeCapabilities =
      stream_executor::CudaComputeCapability::CudaComputeCapabilities;
  switch (cc.major) {
    case CudaComputeCapabilities::BLACKWELL:
      GTEST_SKIP() << "CudaComputeCapabilities::BLACKWELL has the kernel name: "
                   << kernel_names[0];
      break;
    case CudaComputeCapabilities::AMPERE:
      ASSERT_EQ(kernel_names.size(), 1);
      EXPECT_THAT(kernel_names[0], ::testing::Eq("loop_convert_fusion_1"));
      break;
    case CudaComputeCapabilities::HOPPER:
      EXPECT_THAT(kernel_names,
                  ::testing::UnorderedElementsAre(
                      ::testing::HasSubstr("loop_convert_fusion"),
                      ::testing::HasSubstr("gemm_bf16f32_bf16f32_f32_")));
      break;
    default:
      GTEST_SKIP() << "Unsupported compute capability: " << cc.major
                   << " has the kernel name: " << kernel_names[0];
  }
}

TEST_F(BlasAlgorithmTest, Algorithm_TF32_TF32_F32_X3) {
  // We check that the algorithm is propagated to the BLAS call.
  // We also check that the kernel name matches the algorithm for Ampere.

  constexpr std::string_view kHloText = R"(
    HloModule Algorithm_TF32_TF32_F32_X3

    ENTRY main {
      lhs = f32[8512,256]{1,0} parameter(0)
      rhs = f32[256,8512]{1,0} parameter(1)
      ROOT dot = f32[8512,8512]{1,0} dot(lhs, rhs),
          algorithm=dot_tf32_tf32_f32_x3,
          lhs_contracting_dims={1},
          rhs_contracting_dims={0}
    }
  )";
  const std::string pattern =
      R"(CHECK: "algorithm":"ALG_DOT_TF32_TF32_F32_X3")";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), pattern));
  ASSERT_TRUE(ok);

  auto tracer = KernelNameTracer::Create();
  if (tracer == nullptr) {
    GTEST_SKIP() << "KernelNameTracer is not implemented.";
  }
  tracer->start();
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
  auto kernel_names = tracer->stop();

  auto cc = GetCudaComputeCapability();
  using CudaComputeCapabilities =
      stream_executor::CudaComputeCapability::CudaComputeCapabilities;
  switch (cc.major) {
    case CudaComputeCapabilities::BLACKWELL:
      GTEST_SKIP() << "CudaComputeCapabilities::BLACKWELL has the kernel name: "
                   << kernel_names[0];
      break;
    case CudaComputeCapabilities::AMPERE:
      // There is no support for TF32_TF32_F32_X3 on Ampere. We use F32_F32_F32.
      EXPECT_THAT(
          kernel_names,
          ::testing::Contains(::testing::HasSubstr("ampere_sgemm_128x64_nn")));
      break;
    case CudaComputeCapabilities::HOPPER:
      // There is no support for TF32_TF32_F32_X3 on Hopper. We use F32_F32_F32.
      EXPECT_THAT(
          kernel_names,
          ::testing::Contains(::testing::HasSubstr("gemm_f32f32_f32f32_f32")));
      break;
    default:
      GTEST_SKIP() << "Unsupported compute capability: " << cc.major
                   << " has the kernel name: " << kernel_names[0];
  }
}

TEST_F(Triton6xBF16GemmTest, Emit6xBF16GemmWhenBothInputsAreF32) {
  constexpr std::string_view kHloText = R"(
HloModule Emit6xBF16GemmWhenBothInputsAreF32

triton_dot {
  p0 = f32[5,7] parameter(0)
  p1 = f32[7,33] parameter(1)
  ROOT dot = f32[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x6
}

ENTRY e {
  p0 = f32[5,7]{1,0} parameter(0)
  p1 = f32[7,33]{1,0} parameter(1)
  ROOT _ = f32[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}
}
)";
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheckForDot(this, kHloText, "triton_dot", R"(
CHECK:          %[[INFINITY:.*]] = arith.constant dense<0x7F800000> : tensor<32x32xf32>
CHECK:          %[[C_MASK:.*]] = arith.constant dense<-65536> : tensor<32x32xi32>
CHECK:          %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
CHECK:          %[[CAST_I32:.*]] = tt.bitcast %{{.*}} : tensor<32x32xf32> -> tensor<32x32xi32>
CHECK:          %[[EXTRACT_HI:.*]] = arith.andi %[[CAST_I32]], %[[C_MASK]] : tensor<32x32xi32>
CHECK:          %[[CAST_HI:.*]] = tt.bitcast %[[EXTRACT_HI]] : tensor<32x32xi32> -> tensor<32x32xf32>
CHECK:          %[[TRUNC_TO_BF16:.*]] = arith.truncf %[[CAST_HI]] : tensor<32x32xf32> to tensor<32x32xbf16>
CHECK-COUNT-5:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ABS:.*]] = math.absf
CHECK:          %[[CMP:.*]] = arith.cmpf ogt, %[[INFINITY]], %[[ABS]] : tensor<32x32xf32>
CHECK:          %[[SELECT:.*]] = arith.select %[[CMP]], %{{.*}}, %[[C0]] : tensor<32x32xi1>, tensor<32x32xf32>
CHECK:          %[[DOT_LAST:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[SELECT]] : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ACC:.*]] = arith.addf %[[DOT_LAST]], %[[C0]] : tensor<32x32xf32>
    )"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-6,
                                                           /*arel=*/1e-6}));
}

TEST_F(Triton6xBF16GemmTestWithFlag, Emit6xBF16GemmWhenBothInputsAreF32) {
  constexpr std::string_view kHloText = R"(
HloModule Emit6xBF16GemmWhenBothInputsAreF32

triton_dot {
  p0 = f32[5,7] parameter(0)
  p1 = f32[7,33] parameter(1)
  ROOT dot = f32[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[5,7]{1,0} parameter(0)
  p1 = f32[7,33]{1,0} parameter(1)
  ROOT _ = f32[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}
}
)";
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheckForDot(this, kHloText, "triton_dot", R"(
CHECK:          %[[INFINITY:.*]] = arith.constant dense<0x7F800000> : tensor<32x32xf32>
CHECK:          %[[C_MASK:.*]] = arith.constant dense<-65536> : tensor<32x32xi32>
CHECK:          %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
CHECK:          %[[CAST_I32:.*]] = tt.bitcast %{{.*}} : tensor<32x32xf32> -> tensor<32x32xi32>
CHECK:          %[[EXTRACT_HI:.*]] = arith.andi %[[CAST_I32]], %[[C_MASK]] : tensor<32x32xi32>
CHECK:          %[[CAST_HI:.*]] = tt.bitcast %[[EXTRACT_HI]] : tensor<32x32xi32> -> tensor<32x32xf32>
CHECK:          %[[TRUNC_TO_BF16:.*]] = arith.truncf %[[CAST_HI]] : tensor<32x32xf32> to tensor<32x32xbf16>
CHECK-COUNT-5:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ABS:.*]] = math.absf
CHECK:          %[[CMP:.*]] = arith.cmpf ogt, %[[INFINITY]], %[[ABS]] : tensor<32x32xf32>
CHECK:          %[[SELECT:.*]] = arith.select %[[CMP]], %{{.*}}, %[[C0]] : tensor<32x32xi1>, tensor<32x32xf32>
CHECK:          %[[DOT_LAST:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[SELECT]] : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ACC:.*]] = arith.addf %[[DOT_LAST]], %[[C0]] : tensor<32x32xf32>
    )"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-6,
                                                           /*arel=*/1e-6}));
}

TEST_F(Triton6xBF16GemmTest, Triton6xBF16GemmWorksForLongContractingDimension) {
  constexpr std::string_view kHloText = R"(
HloModule Triton6xBF16GemmWorksForLongContractingDimension

triton_dot {
  p0 = f32[5,2048] parameter(0)
  p1 = f32[2048,33] parameter(1)
  ROOT dot = f32[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x6
}

ENTRY e {
  p0 = f32[5,2048]{1,0} parameter(0)
  p1 = f32[2048,33]{1,0} parameter(1)
  ROOT _ = f32[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":64,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":4, "num_ctas":1}}}
}
)";
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheckForDot(this, kHloText, "triton_dot", R"(
CHECK-COUNT-6:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x32xbf16> * tensor<32x32xbf16> -> tensor<64x32xf32>
    )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-5,
                                                           /*arel=*/1e-5}));
}

TEST_F(Triton6xBF16GemmTest, Emit6xBF16GemmEndToEnd) {
  if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
    GTEST_SKIP() << "ALG_DOT_BF16_BF16_F32_X6 not supported on ROCM.";
  }
  constexpr std::string_view kHloText = R"(
HloModule Emit6xBF16GemmEndToEnd

ENTRY e {
  p0 = f32[5,32] parameter(0)
  p1 = f32[32,7] parameter(1)
  ROOT dot = f32[5,7] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x6
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> verified_module,
                          ParseAndReturnVerifiedModule(kHloText));
  CompileAndOptionallyVerifyPtx(std::move(verified_module),
                                R"(
CHECK: mma.sync.aligned.{{.*}}.row.col.f32.bf16.bf16.f32
CHECK-NOT: mma.sync.aligned.{{.*}}.row.col.f32.tf32.tf32.f32
)");
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-6,
                                                /*arel=*/1e-6}));
}

// In these tests, we depend on "algorithm" annotations for selecting the 3XBF16
// algorithm.
class Triton3xBF16GemmTest : public AlgorithmTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = AlgorithmTest::GetDebugOptionsForTest();
    // These 2 flags are not strictly necessary now, but we're adding them the
    // to be on the safe side against future flakiness.
    //
    // Enable triton fusion for all supported GEMMs.
    debug_options.set_xla_gpu_triton_gemm_any(true);
    // Do not fall back to cuBLAS, we are testing Triton.
    debug_options.set_xla_gpu_cublas_fallback(false);

    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    return debug_options;
  }
};

// In these tests, we depend on debug option flags for selecting the 3XBF16
// algorithm.
// TODO(b/316147294): Remove this class and the --xla_gpu_enable_bf16_3way_gemm
// flag after we will support the algorithm values through the entire stack.
class Triton3xBF16GemmTestWithFlag : public AlgorithmTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = AlgorithmTest::GetDebugOptionsForTest();
    // Enable triton fusion for all supported GEMMs.
    debug_options.set_xla_gpu_triton_gemm_any(true);
    // Do not fall back to cuBLAS, we are testing Triton.
    debug_options.set_xla_gpu_cublas_fallback(false);
    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    // Enable bf16_3way gemm to compute F32 matmul.
    debug_options.set_xla_gpu_enable_bf16_3way_gemm(true);
    return debug_options;
  }

 protected:
  void SetUp() override {
    if (!SupportsBF16(GpuComputeComp())) {
      GTEST_SKIP() << "BF16 not supported.";
    }
  }
};

TEST_F(Triton3xBF16GemmTest, Emit3xBF16GemmWhenBothInputsAreF32) {
  constexpr std::string_view kHloText = R"(
HloModule Emit3xBF16GemmWhenBothInputsAreF32

triton_dot {
  p0 = f32[5,7] parameter(0)
  p1 = f32[7,33] parameter(1)
  ROOT dot = f32[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x3
}

ENTRY e {
  p0 = f32[5,7]{1,0} parameter(0)
  p1 = f32[7,33]{1,0} parameter(1)
  ROOT _ = f32[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}
}
)";
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheckForDot(this, kHloText, "triton_dot", R"(
CHECK:          %[[INFINITY:.*]] = arith.constant dense<0x7F800000> : tensor<32x32xf32>
CHECK:          %[[C_MASK:.*]] = arith.constant dense<-65536> : tensor<32x32xi32>
CHECK:          %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
CHECK:          %[[CAST_I32:.*]] = tt.bitcast %{{.*}} : tensor<32x32xf32> -> tensor<32x32xi32>
CHECK:          %[[EXTRACT_HI:.*]] = arith.andi %[[CAST_I32]], %[[C_MASK]] : tensor<32x32xi32>
CHECK:          %[[CAST_HI:.*]] = tt.bitcast %[[EXTRACT_HI]] : tensor<32x32xi32> -> tensor<32x32xf32>
CHECK:          %[[TRUNC_TO_BF16:.*]] = arith.truncf %[[CAST_HI]] : tensor<32x32xf32> to tensor<32x32xbf16>
CHECK-COUNT-2:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ABS:.*]] = math.absf
CHECK:          %[[CMP:.*]] = arith.cmpf ogt, %[[INFINITY]], %[[ABS]] : tensor<32x32xf32>
CHECK:          %[[SELECT:.*]] = arith.select %[[CMP]], %{{.*}}, %[[C0]] : tensor<32x32xi1>, tensor<32x32xf32>
CHECK:          %[[DOT_LAST:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[SELECT]] : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ACC:.*]] = arith.addf %[[DOT_LAST]], %[[C0]] : tensor<32x32xf32>
    )"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-5,
                                                           /*arel=*/1e-5}));
}

TEST_F(Triton3xBF16GemmTestWithFlag, Emit3xBF16GemmWhenBothInputsAreF32) {
  constexpr std::string_view kHloText = R"(
HloModule Emit3xBF16GemmWhenBothInputsAreF32

triton_dot {
  p0 = f32[5,7] parameter(0)
  p1 = f32[7,33] parameter(1)
  ROOT dot = f32[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[5,7]{1,0} parameter(0)
  p1 = f32[7,33]{1,0} parameter(1)
  ROOT _ = f32[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}
}
)";
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheckForDot(this, kHloText, "triton_dot", R"(
CHECK:          %[[INFINITY:.*]] = arith.constant dense<0x7F800000> : tensor<32x32xf32>
CHECK:          %[[C_MASK:.*]] = arith.constant dense<-65536> : tensor<32x32xi32>
CHECK:          %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
CHECK:          %[[CAST_I32:.*]] = tt.bitcast %{{.*}} : tensor<32x32xf32> -> tensor<32x32xi32>
CHECK:          %[[EXTRACT_HI:.*]] = arith.andi %[[CAST_I32]], %[[C_MASK]] : tensor<32x32xi32>
CHECK:          %[[CAST_HI:.*]] = tt.bitcast %[[EXTRACT_HI]] : tensor<32x32xi32> -> tensor<32x32xf32>
CHECK:          %[[TRUNC_TO_BF16:.*]] = arith.truncf %[[CAST_HI]] : tensor<32x32xf32> to tensor<32x32xbf16>
CHECK-COUNT-2:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ABS:.*]] = math.absf
CHECK:          %[[CMP:.*]] = arith.cmpf ogt, %[[INFINITY]], %[[ABS]] : tensor<32x32xf32>
CHECK:          %[[SELECT:.*]] = arith.select %[[CMP]], %{{.*}}, %[[C0]] : tensor<32x32xi1>, tensor<32x32xf32>
CHECK:          %[[DOT_LAST:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[SELECT]] : tensor<32x32xbf16> * tensor<32x32xbf16> -> tensor<32x32xf32>
CHECK:          %[[ACC:.*]] = arith.addf %[[DOT_LAST]], %[[C0]] : tensor<32x32xf32>
    )"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-5,
                                                           /*arel=*/1e-5}));
}

TEST_F(Triton3xBF16GemmTestWithFlag, NoEmit3xBF16GemmWhenBothInputsAreNotF32) {
  constexpr std::string_view kHloText = R"(
HloModule NoEmit3xBF16GemmWhenBothInputsAreNotF32

triton_dot {
  p0 = f16[5,7] parameter(0)
  p1 = f16[7,33] parameter(1)
  ROOT dot = f16[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f16[5,7]{1,0} parameter(0)
  p1 = f16[7,33]{1,0} parameter(1)
  ROOT _ = f16[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1,"num_ctas":1}}}
}
)";
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheckForDot(this, kHloText, "triton_dot", R"(
CHECK:      tt.dot
CHECK-SAME: tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
CHECK-NOT:  tt.dot
    )"));
}

TEST_F(Triton3xBF16GemmTest, Triton3xBF16GemmWorksForLongContractingDimension) {
  constexpr std::string_view kHloText = R"(
HloModule Triton3xBF16GemmWorksForLongContractingDimension

triton_dot {
  p0 = f32[5,2048] parameter(0)
  p1 = f32[2048,33] parameter(1)
  ROOT dot = f32[5,33] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x3
}

ENTRY e {
  p0 = f32[5,2048]{1,0} parameter(0)
  p1 = f32[2048,33]{1,0} parameter(1)
  ROOT _ = f32[5,33] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config:
    {"block_m":64,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":4, "num_ctas":1}}}
}
)";
  TF_ASSERT_OK(
      CreateTritonIrAndFileCheckForDot(this, kHloText, "triton_dot", R"(
CHECK-COUNT-3:  %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x32xbf16> * tensor<32x32xbf16> -> tensor<64x32xf32>
    )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-4,
                                                           /*arel=*/1e-4}));
}

TEST_F(Triton3xBF16GemmTest, Emit3xBF16GemmEndToEnd) {
  if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
    GTEST_SKIP() << "ALG_DOT_BF16_BF16_F32_X3 not supported on ROCM.";
  }
  constexpr std::string_view kHloText = R"(
HloModule Emit3xBF16GemmEndToEnd

ENTRY e {
  p0 = f32[5,32] parameter(0)
  p1 = f32[32,7] parameter(1)
  ROOT dot = f32[5,7] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32_x3
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> verified_module,
                          ParseAndReturnVerifiedModule(kHloText));
  CompileAndOptionallyVerifyPtx(std::move(verified_module),
                                R"(
CHECK: mma.sync.aligned.{{.*}}.row.col.f32.bf16.bf16.f32
CHECK-NOT: mma.sync.aligned.{{.*}}.row.col.f32.tf32.tf32.f32
)");
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-5,
                                                /*arel=*/1e-5}));
}

TEST_F(TritonAlgorithmTest, Algorithm_BF16_BF16_F32_X3) {
  const std::string kHloText = R"(
    HloModule Algorithm_BF16_BF16_F32_X3

    ENTRY main {
      lhs = f32[8512,64]{1,0} parameter(0)
      rhs = f32[64,8512]{1,0} parameter(1)
      ROOT dot = f32[8512,8512]{1,0} dot(lhs, rhs),
          algorithm=dot_bf16_bf16_f32_x3,
          lhs_contracting_dims={1},
          rhs_contracting_dims={0}
    }
  )";
  const std::string pattern =
      R"(CHECK: "kind":"__triton_gemm","triton_gemm_config")";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), pattern));
  EXPECT_TRUE(ok);
}

TEST_F(TritonAlgorithmTest, Algorithm_BF16_BF16_F32_X6) {
  const std::string kHloText = R"(
    HloModule Algorithm_BF16_BF16_F32_X6

    ENTRY main {
      lhs = f32[8512,64]{1,0} parameter(0)
      rhs = f32[64,8512]{1,0} parameter(1)
      ROOT dot = f32[8512,8512]{1,0} dot(lhs, rhs),
          algorithm=dot_bf16_bf16_f32_x6,
          lhs_contracting_dims={1},
          rhs_contracting_dims={0}
    }
  )";
  const std::string pattern =
      R"(CHECK: "kind":"__triton_gemm","triton_gemm_config")";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), pattern));
  EXPECT_TRUE(ok);
}

TEST_F(TritonAlgorithmTest, Algorithm_TF32_TF32_F32) {
  const std::string kHloText = R"(
    HloModule Algorithm_TF32_TF32_F32

    ENTRY main {
      lhs = f32[128,1]{1,0} parameter(0)
      rhs = f32[1,128]{1,0} parameter(1)
      ROOT dot = f32[128,128]{1,0} dot(lhs, rhs),
          algorithm=dot_tf32_tf32_f32,
          lhs_contracting_dims={1},
          rhs_contracting_dims={0}
    }
  )";
  const std::string pattern =
      R"(CHECK: "kind":"__triton_gemm","triton_gemm_config")";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), pattern));
  EXPECT_TRUE(ok);
}

TEST_F(TritonAlgorithmTest, Algorithm_TF32_TF32_F32_X3) {
  const std::string kHloText = R"(
    HloModule Algorithm_TF32_TF32_F32_X3

    ENTRY main {
      lhs = f32[8512,64]{1,0} parameter(0)
      rhs = f32[64,8512]{1,0} parameter(1)
      ROOT dot = f32[8512,8512]{1,0} dot(lhs, rhs),
          algorithm=dot_tf32_tf32_f32_x3,
          lhs_contracting_dims={1},
          rhs_contracting_dims={0}
    }
  )";
  const std::string pattern =
      R"(CHECK: "kind":"__triton_gemm","triton_gemm_config")";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), pattern));
  EXPECT_TRUE(ok);
}

TEST_F(TritonAlgorithmTest, Algorithm_BF16_BF16_F32) {
  if (!SupportsBF16(GpuComputeComp())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  const std::string kHloText = R"(
    HloModule Algorithm_BF16_BF16_F32

    ENTRY main {
      lhs = f32[8512,64]{1,0} parameter(0)
      rhs = f32[64,8512]{1,0} parameter(1)
      ROOT dot = f32[8512,8512]{1,0} dot(lhs, rhs),
          algorithm=dot_bf16_bf16_f32,
          lhs_contracting_dims={1},
          rhs_contracting_dims={0}
    }
  )";
  const std::string pattern =
      R"(CHECK: "kind":"__triton_gemm","triton_gemm_config")";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), pattern));
  EXPECT_TRUE(ok);
}

using PC = PrecisionConfig;
using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::Values;
using ::testing::WithParamInterface;

struct CanHandleTestsParams {
  using TupleType = std::tuple<PrecisionConfig::Algorithm>;

  explicit CanHandleTestsParams(TupleType t) : algorithm(std::get<0>(t)) {}

  PrecisionConfig::Algorithm algorithm;
};

std::string CanHandleTestParamsToString(
    const TestParamInfo<CanHandleTestsParams::TupleType>& info) {
  const CanHandleTestsParams params(info.param);
  return AlgorithmToString(params.algorithm);
}

class CanHandleArguments {
 public:
  CanHandleArguments() {
    InitInfinityArguments();
    InitNaNArguments();
    InitLargeExponentArguments();
  }
  std::vector<Literal*> infinity_arguments() {
    return to_pointers(infinity_arguments_);
  }
  std::vector<Literal*> nan_arguments() { return to_pointers(nan_arguments_); }
  std::vector<Literal*> large_exponent_arguments() {
    return to_pointers(large_exponent_arguments_);
  }

  static constexpr float kLargeExponentFloat = 0x1.0103p72f;

 private:
  void InitInfinityArguments() {
    infinity_arguments_.push_back(LiteralUtil::CreateR2<float>(
        {{+std::numeric_limits<float>::infinity(),
          +std::numeric_limits<float>::infinity()},
         {+std::numeric_limits<float>::infinity(),
          +std::numeric_limits<float>::infinity()}}));
    infinity_arguments_.push_back(
        LiteralUtil::CreateR2<float>({{1.0f, 1.0f}, {1.0f, 1.0f}}));
  }
  void InitNaNArguments() {
    nan_arguments_.push_back(LiteralUtil::CreateR2<float>(
        {{std::numeric_limits<float>::quiet_NaN(),
          std::numeric_limits<float>::quiet_NaN()},
         {std::numeric_limits<float>::quiet_NaN(),
          std::numeric_limits<float>::quiet_NaN()}}));
    nan_arguments_.push_back(LiteralUtil::CreateR2<float>(
        {{1.0f, +std::numeric_limits<float>::infinity()},
         {1.0f, +std::numeric_limits<float>::infinity()}}));
  }

  void InitLargeExponentArguments() {
    large_exponent_arguments_.push_back(LiteralUtil::CreateR2<float>(
        {{kLargeExponentFloat, 1.0f}, {-kLargeExponentFloat, 1.0f}}));
    large_exponent_arguments_.push_back(LiteralUtil::CreateR2<float>(
        {{kLargeExponentFloat, 1.0f}, {-kLargeExponentFloat, 1.0f}}));
  }

  std::vector<Literal*> to_pointers(const std::vector<Literal>& literals) {
    std::vector<Literal*> result;
    absl::c_transform(
        literals, std::back_inserter(result),
        [](const Literal& literal) { return const_cast<Literal*>(&literal); });
    return result;
  }

  std::vector<Literal> infinity_arguments_;
  std::vector<Literal> nan_arguments_;
  std::vector<Literal> large_exponent_arguments_;
};

class BlasCanHandle
    : public BlasAlgorithmTest,
      public CanHandleArguments,
      public WithParamInterface<CanHandleTestsParams::TupleType> {
 public:
  BlasCanHandle() : BlasAlgorithmTest() {
    algorithm_ = AlgorithmToString(std::get<0>(GetParam()));
  }

  std::string HloText() const {
    return absl::StrFormat(kHloTextTemplate, HloModuleTestName(), algorithm_);
  }

  static constexpr std::string_view kPattern = R"(CHECK-NOT: __triton_gemm)";

 private:
  static constexpr std::string_view kHloTextTemplate = R"(
    HloModule %s

    ENTRY e {
      p0 = f32[2,2] parameter(0)
      p1 = f32[2,2] parameter(1)
      ROOT dot = f32[2,2] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0},
        algorithm=%s
    }
  )";

 protected:
  std::string algorithm_;
};

class TritonCanHandle
    : public TritonAlgorithmTest,
      public CanHandleArguments,
      public WithParamInterface<CanHandleTestsParams::TupleType> {
 public:
  TritonCanHandle() : TritonAlgorithmTest() {
    algorithm_ = AlgorithmToString(std::get<0>(GetParam()));
  }

  std::string HloText() const {
    return absl::StrFormat(kHloTextTemplate, HloModuleTestName(), algorithm_);
  }

  static constexpr std::string_view kPattern = R"(CHECK: __triton_gemm)";

 private:
  static constexpr std::string_view kHloTextTemplate = R"(
    HloModule %s

    triton_dot {
      p0 = f32[2,2] parameter(0)
      p1 = f32[2,2] parameter(1)
      ROOT dot = f32[2,2] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0},
        algorithm=%s
    }

    ENTRY e {
      p0 = f32[2,2]{1, 0} parameter(0)
      p1 = f32[2,2]{1, 0} parameter(1)
      ROOT _ = f32[2,2] fusion(p0, p1), kind=kCustom, calls=triton_dot,
        backend_config={"fusion_backend_config": {kind: "__triton_gemm",
        triton_gemm_config:
        {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1, "num_ctas":1}}}
    }
  )";
  std::string algorithm_;
};

TEST_P(BlasCanHandle, Infinity) {
  std::string hlo_text = HloText();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_text));
  auto module_text = module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module_text, kPattern));
  ASSERT_TRUE(ok);
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), infinity_arguments(),
                                       ErrorSpec{/*aabs=*/0, /*arel=*/0}))
      << " failed for module hlo: \n"
      << module_text;
}

TEST_P(BlasCanHandle, NaN) {
  std::string hlo_text = HloText();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_text));
  auto module_text = module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module_text, kPattern));
  ASSERT_TRUE(ok);
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), nan_arguments(),
                                       ErrorSpec{/*aabs=*/0, /*arel=*/0}))
      << " failed for module hlo: \n"
      << module_text;
}

TEST_P(BlasCanHandle, InputsWithLargeExponent) {
  std::string hlo_text = HloText();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_text));
  auto module_text = module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module_text, kPattern));
  ASSERT_TRUE(ok);
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), large_exponent_arguments(),
      ErrorSpec{/*aabs=*/kLargeExponentFloat * 1e-4, /*arel=*/1e-6}))
      << " failed for module hlo: \n"
      << module_text;
}

TEST_P(TritonCanHandle, Infinity) {
  std::string hlo_text = HloText();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_text));
  auto module_text = module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module_text, kPattern));
  ASSERT_TRUE(ok);
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), infinity_arguments(),
                                       ErrorSpec{/*aabs=*/0, /*arel=*/0}))
      << " failed for module hlo: \n"
      << module_text;
}

TEST_P(TritonCanHandle, NaN) {
  std::string hlo_text = HloText();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_text));

  auto module_text = module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module_text, kPattern));
  ASSERT_TRUE(ok);
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), nan_arguments(),
                                       ErrorSpec{/*aabs=*/0, /*arel=*/0}))
      << " failed for module hlo: \n"
      << module_text;
}

TEST_P(TritonCanHandle, InputsWithLargeExponent) {
  std::string hlo_text = HloText();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_text));
  auto module_text = module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module_text, kPattern));
  ASSERT_TRUE(ok);

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), large_exponent_arguments(),
      ErrorSpec{/*aabs=*/kLargeExponentFloat * 1e-4, /*arel=*/1e-6}))
      << " failed for module hlo: \n"
      << module_text;
}

INSTANTIATE_TEST_SUITE_P(BlasCanHandle, BlasCanHandle,
                         Combine(Values(PC::ALG_DOT_BF16_BF16_F32_X3,
                                        PC::ALG_DOT_BF16_BF16_F32_X6)),
                         CanHandleTestParamsToString);

INSTANTIATE_TEST_SUITE_P(TritonCanHandle, TritonCanHandle,
                         Combine(Values(PC::ALG_DOT_BF16_BF16_F32_X3,
                                        PC::ALG_DOT_BF16_BF16_F32_X6)),
                         CanHandleTestParamsToString);

}  // namespace
}  // namespace gpu
}  // namespace xla
