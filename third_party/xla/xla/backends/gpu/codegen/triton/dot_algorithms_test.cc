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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/codegen/triton/kernel_name_tracer.h"
#include "xla/backends/gpu/codegen/triton/test_utils.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

class AlgorithmTest : public GpuCodegenTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_autotune_level(0);
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

 protected:
  const stream_executor::DeviceDescription& device_desc() {
    return backend().default_stream_executor()->GetDeviceDescription();
  }
};

// In these tests, we depend on "algorithm" annotations for selecting the 6XBF16
// algorithm.
class Triton6xBF16GemmTest : public AlgorithmTest {
 protected:
  void SetUp() override {
    if (!SupportsBF16(GpuComputeComp())) {
      GTEST_SKIP() << "BF16 not supported.";
    }
  }
};

class BlasAlgorithmTest : public AlgorithmTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = AlgorithmTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    return debug_options;
  }
};

using TritonAlgorithmTest = AlgorithmTest;

TEST_F(AlgorithmTest, Algorithm3xBF16) {
  constexpr absl::string_view kHloText = R"(
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
  constexpr absl::string_view kHloText = R"(
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
  constexpr absl::string_view kHloText = R"(
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
  constexpr absl::string_view kPattern = R"(
    CHECK:  %convert{{.*}} = bf16[
    CHECK:  %convert{{.*}} = bf16[
    CHECK: "algorithm":"ALG_UNSET"
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), kPattern));
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
    case CudaComputeCapabilities::kBlackwell:
      GTEST_SKIP()
          << "CudaComputeCapabilities::kBlackwell has the kernel name: "
          << kernel_names[0];
      break;
    case CudaComputeCapabilities::kAmpere:
      EXPECT_THAT(kernel_names, ::testing::UnorderedElementsAre(
                                    ::testing::Eq("wrapped_convert"),
                                    ::testing::Eq("wrapped_convert_1"),
                                    ::testing::HasSubstr("gemm_bf16_")));
      break;
    case CudaComputeCapabilities::kHopper:
      // Convert to bf16+cublas works faster than dot with algorithm.
      EXPECT_THAT(kernel_names,
                  ::testing::Contains(::testing::HasSubstr("wrapped_convert"))
                      .Times(2));
      break;
    default:
      GTEST_SKIP() << "Unsupported compute capability: " << cc.major
                   << " has the kernel name: " << kernel_names[0];
  }
}

TEST_F(AlgorithmTest, Algorithm_BF16_BF16_F32_on_BF16_input_for_multiply) {
  if (!SupportsBF16(GpuComputeComp())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  constexpr absl::string_view kHloText = R"(
    HloModule Algorithm_BF16_BF16_F32_with_BF16_input

    ENTRY main {
      lhs = bf16[256,8512] parameter(0)
      rhs = bf16[256,8512] parameter(1)
      ROOT dot = f32[256] dot(lhs, rhs),
          algorithm=dot_bf16_bf16_f32,
          lhs_batch_dims={0},
          rhs_batch_dims={0},
          lhs_contracting_dims={1},
          rhs_contracting_dims={1}
    }
  )";
  // Multiply on a100 operates with f32, h100 operates with bf16.
  const std::string pattern = R"(
    CHECK:    %[[multiply:.*]] = [[type:.*]][256,8512]{1,0} multiply([[type]]
    CHECK:    %[[reduce:.*]] = f32[256]{0} reduce(
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(
      auto ok,
      RunFileCheck(
          module->ToString(HloPrintOptions().set_print_operand_shape(true)),
          pattern));
  ASSERT_TRUE(ok);
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), ErrorSpec{/*aabs=*/1e-7, /*arel=*/1e-7}));
}

TEST_F(BlasAlgorithmTest, Algorithm_BF16_BF16_F32_X3) {
  if (!SupportsBF16(GpuComputeComp())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  constexpr absl::string_view kHloText = R"(
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
  constexpr absl::string_view kPattern = R"(
    CHECK-COUNT-3: custom_call_target="__cublas$gemm"
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), kPattern));
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
    case CudaComputeCapabilities::kBlackwell:
      GTEST_SKIP()
          << "CudaComputeCapabilities::kBlackwell has the kernel name: "
          << kernel_names[0];
      break;
    case CudaComputeCapabilities::kAmpere:
      ASSERT_EQ(kernel_names.size(), 1);
      EXPECT_THAT(kernel_names[0], ::testing::Eq("loop_convert_fusion_1"));
      break;
    case CudaComputeCapabilities::kHopper:
      EXPECT_THAT(kernel_names,
                  ::testing::Contains(::testing::Eq("loop_convert_fusion_1")));
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
  constexpr absl::string_view kHloText = R"(
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
  constexpr absl::string_view kPattern = R"(
    CHECK-COUNT-6: custom_call_target="__cublas$gemm"
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), kPattern));
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
    case CudaComputeCapabilities::kBlackwell:
      GTEST_SKIP()
          << "CudaComputeCapabilities::kBlackwell has the kernel name: "
          << kernel_names[0];
      break;
    case CudaComputeCapabilities::kAmpere:
      ASSERT_EQ(kernel_names.size(), 1);
      EXPECT_THAT(kernel_names[0], ::testing::Eq("loop_convert_fusion_1"));
      break;
    case CudaComputeCapabilities::kHopper:
      EXPECT_THAT(
          kernel_names,
          ::testing::Contains(::testing::HasSubstr("loop_convert_fusion")));
      break;
    default:
      GTEST_SKIP() << "Unsupported compute capability: " << cc.major
                   << " has the kernel name: " << kernel_names[0];
  }
}

TEST_F(BlasAlgorithmTest, Algorithm_TF32_TF32_F32_X3) {
  // We check that the algorithm is propagated to the BLAS call.
  // We also check that the kernel name matches the algorithm for Ampere.

  constexpr absl::string_view kHloText = R"(
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
  constexpr absl::string_view kPattern = R"(
      CHECK: custom_call_target="__cublas$gemm"{{.*}}"algorithm":"ALG_DOT_TF32_TF32_F32"
      CHECK: custom_call_target="__cublas$gemm"{{.*}}"algorithm":"ALG_DOT_TF32_TF32_F32"
      CHECK: custom_call_target="__cublas$gemm"{{.*}}"algorithm":"ALG_DOT_TF32_TF32_F32"
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), kPattern));
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
    case CudaComputeCapabilities::kBlackwell:
      GTEST_SKIP()
          << "CudaComputeCapabilities::kBlackwell has the kernel name: "
          << kernel_names[0];
      break;
    case CudaComputeCapabilities::kAmpere:
      EXPECT_THAT(kernel_names, ::testing::Contains(::testing::HasSubstr(
                                    "bitcast_convert_subtract")));
      break;
    case CudaComputeCapabilities::kHopper:
      EXPECT_THAT(kernel_names,
                  ::testing::UnorderedElementsAre(
                      ::testing::HasSubstr("bitcast_convert_subtract"),
                      ::testing::HasSubstr("tf32f32")));
      break;
    default:
      GTEST_SKIP() << "Unsupported compute capability: " << cc.major
                   << " has the kernel name: " << kernel_names[0];
  }
}

TEST_F(Triton6xBF16GemmTest, Emit6xBF16GemmWhenBothInputsAreF32) {
  constexpr absl::string_view kHloText = R"(
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

TEST_F(Triton6xBF16GemmTest, Triton6xBF16GemmWorksForLongContractingDimension) {
  constexpr absl::string_view kHloText = R"(
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
  constexpr absl::string_view kHloText = R"(
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
using Triton3xBF16GemmTest = AlgorithmTest;

TEST_F(Triton3xBF16GemmTest, Emit3xBF16GemmWhenBothInputsAreF32) {
  constexpr absl::string_view kHloText = R"(
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

TEST_F(Triton3xBF16GemmTest, Triton3xBF16GemmWorksForLongContractingDimension) {
  constexpr absl::string_view kHloText = R"(
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
  constexpr absl::string_view kHloText = R"(
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
  if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
    GTEST_SKIP() << "Triton currently disabled on ROCM.";
  }
  constexpr absl::string_view kHloText = R"(
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
  constexpr absl::string_view kPattern =
      R"(CHECK: "kind":"__triton_gemm","triton_gemm_config")";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), kPattern));
  EXPECT_TRUE(ok);
}

TEST_F(TritonAlgorithmTest, Algorithm_BF16_BF16_F32_X6) {
  if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
    GTEST_SKIP() << "Triton currently disabled on ROCM.";
  }
  constexpr absl::string_view kHloText = R"(
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
  constexpr absl::string_view kPattern =
      R"(CHECK: "kind":"__triton_gemm","triton_gemm_config")";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), kPattern));
  EXPECT_TRUE(ok);
}

TEST_F(TritonAlgorithmTest, Algorithm_TF32_TF32_F32) {
  if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
    GTEST_SKIP() << "Triton currently disabled on ROCM.";
  }
  constexpr absl::string_view kHloText = R"(
    HloModule Algorithm_TF32_TF32_F32

    ENTRY main {
      lhs = f32[128,256]{1,0} parameter(0)
      rhs = f32[256,128]{1,0} parameter(1)
      ROOT dot = f32[128,128]{1,0} dot(lhs, rhs),
          algorithm=dot_tf32_tf32_f32,
          lhs_contracting_dims={1},
          rhs_contracting_dims={0}
    }
  )";
  constexpr absl::string_view kPattern = R"(
    CHECK: algorithm=dot_tf32_tf32_f32
    CHECK: "kind":"__triton_gemm","triton_gemm_config"
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), kPattern));
  EXPECT_TRUE(ok);
}

TEST_F(TritonAlgorithmTest, Algorithm_TF32_TF32_F32_X3) {
  if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
    GTEST_SKIP() << "Triton currently disabled on ROCM.";
  }
  constexpr absl::string_view kHloText = R"(
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
  constexpr absl::string_view kPattern =
      R"(CHECK: "kind":"__triton_gemm","triton_gemm_config")";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), kPattern));
  EXPECT_TRUE(ok);
}

TEST_F(TritonAlgorithmTest, Algorithm_BF16_BF16_F32) {
  if (!SupportsBF16(GpuComputeComp())) {
    GTEST_SKIP() << "BF16 not supported.";
  }
  if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
    GTEST_SKIP() << "Triton currently disabled on ROCM.";
  }
  constexpr absl::string_view kHloText = R"(
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
  constexpr absl::string_view kPattern =
      R"(CHECK: "kind":"__triton_gemm","triton_gemm_config")";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), kPattern));
  EXPECT_TRUE(ok);
}

TEST_F(TritonAlgorithmTest, Dot_BF16_X6_WithConst) {
  constexpr absl::string_view kHloText = R"(
    HloModule Dot_BF16_X6_WithConst

    %triton_fusion_dot (p_0: f32[1,258]) -> f32[258] {
      %c_1 = f32[] constant(-1.22474492)
      %r_1 = f32[1]{0} reshape(f32[] %c_1)
      %r_2 = f32[1,1]{1,0} reshape(f32[1]{0} %r_1)
      %p_0 = f32[1,258]{1,0} parameter(0)
      %r_3 = f32[258]{0} reshape(f32[1,258]{1,0} %p_0)
      %r_4 = f32[258,1]{1,0} reshape(f32[258]{0} %r_3)
      %dot_0 = f32[1,258]{1,0} dot(f32[1,1]{1,0} %r_2, f32[258,1]{1,0} %r_4),
          lhs_contracting_dims={0},
          rhs_contracting_dims={1},
          algorithm=dot_bf16_bf16_f32_x6
      %r_5 = f32[258]{0} reshape(f32[1,258]{1,0} %dot_0)
      %c_2 = f32[] constant(0.282094777)
      %b_0 = f32[258]{0} broadcast(f32[] %c_2), dimensions={}
      ROOT %m_0 = f32[258]{0} multiply(f32[258]{0} %r_5, f32[258]{0} %b_0)
    }

    ENTRY %entry_computation {
      %p_0 = f32[1,258]{1,0} parameter(0)
      ROOT %dot = f32[258]{0} fusion(f32[1,258]{1,0} %p_0),
        kind=kCustom,
        calls=%triton_fusion_dot,
        backend_config={
          "operation_queue_id":"0",
          "wait_on_operation_queues":[],
          "fusion_backend_config":{
            "kind":"__triton_gemm",
            "triton_gemm_config":{
              "block_m":"16",
              "block_n":"256",
              "block_k":"16",
              "split_k":"1",
              "num_stages":"4",
              "num_warps":"4",
              "num_ctas":"1"
            }
          },
          "force_earliest_schedule":false
        }
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

using PC = PrecisionConfig;
using ::testing::TestParamInfo;
using ::testing::WithParamInterface;

std::string AlgorithmTestParamToString(
    const TestParamInfo<PC::Algorithm>& info) {
  return AlgorithmToString(info.param);
}

class NumericTestsArguments {
 public:
  NumericTestsArguments() {
    InitInfinityArguments();
    InitNaNArguments();
    InitLargeExponentArguments();
  }
  std::vector<Literal*> infinity_arguments_ptrs() {
    return to_pointers(infinity_arguments_);
  }
  const std::vector<Literal>& infinity_arguments() {
    return infinity_arguments_;
  }
  std::vector<Literal*> nan_arguments_ptrs() {
    return to_pointers(nan_arguments_);
  }
  const std::vector<Literal>& nan_arguments() { return nan_arguments_; }
  std::vector<Literal*> large_exponent_arguments_ptr() {
    return to_pointers(large_exponent_arguments_);
  }
  const std::vector<Literal>& large_exponent_arguments() {
    return large_exponent_arguments_;
  }

  static constexpr float kLargeExponentFloat = 0x1.0103p72f;

 private:
  void InitInfinityArguments() {
    auto inf = +std::numeric_limits<float>::infinity();
    infinity_arguments_.push_back(LiteralUtil::CreateR2<float>(
        {{inf, inf, inf, inf, inf, inf, inf, inf},
         {inf, inf, inf, inf, inf, inf, inf, inf},
         {inf, inf, inf, inf, inf, inf, inf, inf},
         {inf, inf, inf, inf, inf, inf, inf, inf},
         {inf, inf, inf, inf, inf, inf, inf, inf},
         {inf, inf, inf, inf, inf, inf, inf, inf},
         {inf, inf, inf, inf, inf, inf, inf, inf},
         {inf, inf, inf, inf, inf, inf, inf, inf}}));
    auto one = 1.0f;
    infinity_arguments_.push_back(LiteralUtil::CreateR2<float>(
        {{one, one, one, one, one, one, one, one},
         {one, one, one, one, one, one, one, one},
         {one, one, one, one, one, one, one, one},
         {one, one, one, one, one, one, one, one},
         {one, one, one, one, one, one, one, one},
         {one, one, one, one, one, one, one, one},
         {one, one, one, one, one, one, one, one},
         {one, one, one, one, one, one, one, one}}));
  }
  void InitNaNArguments() {
    auto nan = std::numeric_limits<float>::quiet_NaN();
    auto inf = +std::numeric_limits<float>::infinity();
    auto one = 1.0f;
    nan_arguments_.push_back(LiteralUtil::CreateR2<float>(
        {{nan, nan, nan, nan, nan, nan, nan, nan},
         {nan, nan, nan, nan, nan, nan, nan, nan},
         {nan, nan, nan, nan, nan, nan, nan, nan},
         {nan, nan, nan, nan, nan, nan, nan, nan},
         {nan, nan, nan, nan, nan, nan, nan, nan},
         {nan, nan, nan, nan, nan, nan, nan, nan},
         {nan, nan, nan, nan, nan, nan, nan, nan},
         {nan, nan, nan, nan, nan, nan, nan, nan}}));
    nan_arguments_.push_back(LiteralUtil::CreateR2<float>(
        {{one, inf, inf, inf, inf, inf, inf, inf},
         {one, inf, inf, inf, inf, inf, inf, inf},
         {one, inf, inf, inf, inf, inf, inf, inf},
         {one, inf, inf, inf, inf, inf, inf, inf},
         {one, inf, inf, inf, inf, inf, inf, inf},
         {one, inf, inf, inf, inf, inf, inf, inf},
         {one, inf, inf, inf, inf, inf, inf, inf},
         {one, inf, inf, inf, inf, inf, inf, inf}}));
  }

  void InitLargeExponentArguments() {
    auto le = kLargeExponentFloat;
    auto one = 1.0f;
    large_exponent_arguments_.push_back(LiteralUtil::CreateR2<float>(
        {{le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one}}));
    large_exponent_arguments_.push_back(LiteralUtil::CreateR2<float>(
        {{le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one},
         {-le, one, one, one, one, one, one, one}}));
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

class NumericTestsForBlas : public BlasAlgorithmTest,
                            public NumericTestsArguments,
                            public WithParamInterface<PC::Algorithm> {
 public:
  NumericTestsForBlas() : BlasAlgorithmTest() {
    algorithm_ = AlgorithmToString(GetParam());
  }

  std::string HloText() const {
    return absl::StrFormat(kHloTextTemplate, HloModuleTestName(), algorithm_);
  }

  static constexpr absl::string_view kPattern = R"(CHECK: __cublas$gemm)";

  static constexpr absl::string_view kReferenceHloText = R"(
    HloModule %s

    ENTRY e {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      ROOT dot = f32[8,8] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )";

  // Takes the reference hlo and compiles it for cublas.
  std::unique_ptr<HloModule> GetReferenceModuleForCublas() {
    auto reference_options = GetDebugOptionsForTest();
    reference_options.set_xla_gpu_triton_gemm_any(false);
    reference_options.set_xla_gpu_enable_triton_gemm(false);
    reference_options.set_xla_gpu_cublas_fallback(true);

    HloModuleConfig config;
    config.set_debug_options(reference_options);
    config.set_replica_count(1);
    config.set_num_partitions(1);

    auto reference_module =
        ParseAndReturnVerifiedModule(kReferenceHloText, config);
    CHECK_OK(reference_module.status());

    auto optimized_module =
        GetOptimizedModule(std::move(reference_module.value()));
    CHECK_OK(optimized_module.status());
    return std::move(optimized_module.value());
  }

 private:
  static constexpr absl::string_view kHloTextTemplate = R"(
    HloModule %s

    ENTRY e {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      ROOT dot = f32[8,8] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0},
        algorithm=%s
    }
  )";

 protected:
  std::string algorithm_;
};

class NumericTestsForTriton : public TritonAlgorithmTest,
                              public NumericTestsArguments,
                              public WithParamInterface<PC::Algorithm> {
 public:
  NumericTestsForTriton() : TritonAlgorithmTest() {
    algorithm_ = AlgorithmToString(GetParam());
  }

  std::string HloText() const {
    return absl::StrFormat(kHloTextTemplate, HloModuleTestName(), algorithm_);
  }

  static constexpr absl::string_view kPattern = R"(CHECK: __triton_gemm)";

 private:
  static constexpr absl::string_view kHloTextTemplate = R"(
    HloModule %s

    triton_dot {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      ROOT dot = f32[8,8] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0},
        algorithm=%s
    }

    ENTRY e {
      p0 = f32[8,8]{1, 0} parameter(0)
      p1 = f32[8,8]{1, 0} parameter(1)
      ROOT _ = f32[8,8] fusion(p0, p1), kind=kCustom, calls=triton_dot,
        backend_config={"fusion_backend_config": {kind: "__triton_gemm",
        triton_gemm_config:
        {"block_m":32,"block_n":32,"block_k":32,"split_k":1,"num_stages":1,"num_warps":1, "num_ctas":1}}}
    }
  )";
  std::string algorithm_;
};

TEST_P(NumericTestsForBlas, Infinity) {
  std::string hlo_text = HloText();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_text));
  auto module_text = module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module_text, kPattern));
  ASSERT_TRUE(ok);

  auto reference_module = GetReferenceModuleForCublas();

  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(
      std::move(reference_module), std::move(module), infinity_arguments(),
      /*run_hlo_passes=*/false,
      /*use_threads=*/false, ErrorSpec{/*aabs=*/0, /*arel=*/0}))
      << " failed for module hlo: \n"
      << module_text;
}

TEST_P(NumericTestsForBlas, NaN) {
  std::string hlo_text = HloText();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_text));
  auto module_text = module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module_text, kPattern));
  ASSERT_TRUE(ok);

  auto reference_module = GetReferenceModuleForCublas();

  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(
      std::move(reference_module), std::move(module), nan_arguments(),
      /*run_hlo_passes=*/false,
      /*use_threads=*/false, ErrorSpec{/*aabs=*/0, /*arel=*/0}))
      << " failed for module hlo: \n"
      << module_text;
}

TEST_P(NumericTestsForBlas, InputsWithLargeExponent) {
  std::string hlo_text = HloText();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_text));
  auto module_text = module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module_text, kPattern));
  ASSERT_TRUE(ok);

  auto reference_module = GetReferenceModuleForCublas();

  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(
      std::move(reference_module), std::move(module),
      large_exponent_arguments(),
      /*run_hlo_passes=*/false,
      /*use_threads=*/false,
      ErrorSpec{/*aabs=*/kLargeExponentFloat * 1e-4, /*arel=*/1e-6}))
      << " failed for module hlo: \n"
      << module_text;
}

TEST_P(NumericTestsForBlas, PrecisionCheck) {
  std::string hlo_text = HloText();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_text));
  auto module_text = module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module_text, kPattern));
  ASSERT_TRUE(ok);

  auto reference_module = GetReferenceModuleForCublas();

  // No specific inputs are needed for this test.
  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(
      std::move(reference_module), std::move(module),
      /*run_hlo_passes=*/false,
      /*use_threads=*/false, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}))
      << " failed for module hlo: \n"
      << module_text;
}

TEST_P(NumericTestsForTriton, Infinity) {
  // The test proves that Triton can handle dot for one x infinity inputs.
  // It is the tricky cases for X3 and X6 algorithms. They should mask the NaN
  // intermediate results.
  std::string hlo_text = HloText();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_text));
  auto module_text = module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module_text, kPattern));
  ASSERT_TRUE(ok);
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module),
                                       infinity_arguments_ptrs(),
                                       ErrorSpec{/*aabs=*/0, /*arel=*/0}))
      << " failed for module hlo: \n"
      << module_text;
}

TEST_P(NumericTestsForTriton, NaN) {
  std::string hlo_text = HloText();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_text));

  auto module_text = module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module_text, kPattern));
  ASSERT_TRUE(ok);
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), nan_arguments_ptrs(),
                                       ErrorSpec{/*aabs=*/0, /*arel=*/0}))
      << " failed for module hlo: \n"
      << module_text;
}

TEST_P(NumericTestsForTriton, InputsWithLargeExponent) {
  std::string hlo_text = HloText();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_text));
  auto module_text = module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module_text, kPattern));
  ASSERT_TRUE(ok);

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), large_exponent_arguments_ptr(),
      ErrorSpec{/*aabs=*/kLargeExponentFloat * 1e-4, /*arel=*/1e-6}))
      << " failed for module hlo: \n"
      << module_text;
}

INSTANTIATE_TEST_SUITE_P(NumericTestsForBlas, NumericTestsForBlas,
                         ::testing::ValuesIn({PC::ALG_DOT_TF32_TF32_F32_X3,
                                              PC::ALG_DOT_BF16_BF16_F32_X3,
                                              PC::ALG_DOT_BF16_BF16_F32_X6,
                                              PC::ALG_DOT_BF16_BF16_F32_X9}),
                         AlgorithmTestParamToString);

INSTANTIATE_TEST_SUITE_P(NumericTestsForTriton, NumericTestsForTriton,
                         ::testing::ValuesIn({PC::ALG_DOT_BF16_BF16_F32_X3,
                                              PC::ALG_DOT_BF16_BF16_F32_X6,
                                              PC::ALG_DOT_BF16_BF16_F32_X9,
                                              PC::ALG_DOT_TF32_TF32_F32_X3}),
                         AlgorithmTestParamToString);

// Collects the results of a test. The results can be dumped in CSV format.
class CSVWriter {
 public:
  // Appends a value to the current row. If there is no current row, creates a
  // new one.
  template <typename V>
  void appendValue(V v) {
    if (results_.empty()) {
      results_.emplace_back();
    }
    results_.back().push_back(absl::StrCat(v));
  }

  // Appends a new empty row.
  void nextRow() { results_.emplace_back(); }

  // Appends a row with the given values.
  template <typename V>
  void appendRow(std::vector<V> v) {
    results_.emplace_back();
    for (const auto& v : v) {
      results_.back().push_back(absl::StrCat(v));
    }
  }

  // Returns the results in CSV format.
  std::string GetResult(absl::string_view title,
                        absl::string_view delimiter = ", ",
                        bool separate_first_row = true) const {
    std::vector<size_t> sizes;
    size_t columns = 0;
    for (const auto& row : results_) {
      columns = std::max(columns, row.size());
      sizes.resize(columns);
      for (int i = 0; i < row.size(); ++i) {
        sizes[i] = std::max(sizes[i], row[i].size());
      }
    }
    std::string result = absl::StrCat(title, "\n");
    bool first_row = true;
    for (const auto& row : results_) {
      for (int i = 0; i < row.size(); ++i) {
        auto format = absl::StrFormat("%%%ds", sizes[i]);
        auto format_runtime = absl::ParsedFormat<'s'>::New(format);
        absl::StrAppend(&result, absl::StrFormat(*format_runtime, row[i]),
                        delimiter);
      }
      result += "\n";
      if (first_row && separate_first_row) {
        first_row = false;
        auto total_size = delimiter.size() * (columns - 1);
        for (const auto& size : sizes) {
          total_size += size;
        }
        result += std::string(total_size, '-');
        result += "\n";
      }
    }
    return result;
  }

 private:
  std::vector<std::vector<std::string>> results_;
};

// The tests builds a matrix of MxN for different tensor sizes with the values
// Yes/No/Fail for triton and blas and dumps the results in CSV format to the
// test output.
class TritonAndBlasSupportForDifferentTensorSizes
    : public WithParamInterface<PC::Algorithm>,
      public AlgorithmTest {
 public:
  static auto GetModuleConfig(const DebugOptions& debug_options) {
    HloModuleConfig config;
    config.set_debug_options(debug_options);
    config.set_replica_count(1);
    config.set_num_partitions(1);
    return config;
  }

  absl::StatusOr<std::unique_ptr<HloModule>> GetModule(
      absl::string_view hlo_template,
      const std::vector<std::pair<std::string, std::string>>& args,
      const DebugOptions& options) {
    auto config = GetModuleConfig(options);
    auto hlo_text = absl::StrReplaceAll(hlo_template, args);

    static int counter = 0;

    DumpToFileInDirOrStdout(options, ++counter, GetTestName("_"), "",
                            "hlo_text.before_passes.txt", hlo_text);
    auto verified_module_or = ParseAndReturnVerifiedModule(hlo_text, config);
    if (!verified_module_or.ok()) {
      LOG(ERROR) << "Failed to parse module: " << verified_module_or.status();
      return verified_module_or.status();
    }
    auto module_or = backend().compiler()->RunHloPasses(
        std::move(verified_module_or.value()),
        backend().default_stream_executor(), GetAllocator());
    if (!module_or.ok()) {
      LOG(ERROR) << "Failed to compile module: " << module_or.status();
    }
    DumpToFileInDirOrStdout(options, counter, GetTestName("_"), "",
                            "hlo_text.after_passes.txt",
                            module_or.ok() ? module_or.value()->ToString()
                                           : module_or.status().message());
    return module_or;
  };

 protected:
  void SetUp() override {
    AlgorithmTest::SetUp();
    debug_options_ = GetDebugOptionsForTest();

    triton_options_ = debug_options_;

    blas_options_ = debug_options_;
    blas_options_.set_xla_gpu_enable_triton_gemm(false);

    algorithm_ = AlgorithmToString(GetParam());
  }

  std::string GetTestName(absl::string_view delimiter) const {
    auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    auto suite_name = test_info->test_suite_name();
    std::string test_name = test_info->name();
    return absl::StrReplaceAll(absl::StrCat(suite_name, delimiter, test_name),
                               {{"/", "_"}});
  }

  void DumpResults(const CSVWriter& csv, absl::string_view suffix) {
    auto title = absl::StrCat("Test name: ", GetTestName("."));
    auto result = csv.GetResult(title, ", ");
    LOG(ERROR) << "result: \n" << result;

    auto test_name = GetTestName("_");
    DumpToFileInDirOrStdout(debug_options_, 0, test_name, "", suffix, result);
  }

  DebugOptions debug_options_;

  DebugOptions triton_options_;

  DebugOptions blas_options_;

  std::string algorithm_;

  static constexpr absl::string_view kBlasPattern = "__cublas$gemm";
  static constexpr absl::string_view kTritonGemmPattern = "__triton_gemm";
  static constexpr int kMaxSize = 8192;
  static constexpr int kStepSize = 8;
  static constexpr int kMaxK = kMaxSize;
};

// The test does not fail. It just dumps the results in CSV format.
TEST_P(TritonAndBlasSupportForDifferentTensorSizes,
       DotThatWillBeConvertedToMultiply) {
  constexpr absl::string_view kHloText = R"(
    HloModule ${module_name}

    ENTRY e {
      p0 = f32[${b},${k}] parameter(0)
      p1 = f32[${b},${k}] parameter(1)
      ROOT dot = f32[${b}] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1},
        lhs_batch_dims={0},
        rhs_batch_dims={0},
        algorithm=${algorithm}
    }
  )";
  CSVWriter csv;
  csv.appendValue("M/N");
  for (int k = 1; k <= kMaxSize; k *= kStepSize) {
    csv.appendValue(k);
  }
  for (int b = 1; b <= kMaxSize; b *= kStepSize) {
    csv.nextRow();
    csv.appendValue(b);
    for (int k = 1; k <= kMaxSize; k *= kStepSize) {
      auto run = [&](absl::string_view backend, absl::string_view pattern,
                     const DebugOptions& options) -> absl::string_view {
        auto test_name = absl::StrReplaceAll(TestName(), {{"/", "_"}});
        auto module_name =
            absl::StrCat(test_name, "_", backend, "_", b, "_", k);
        auto module = GetModule(kHloText,
                                {{"${module_name}", module_name},
                                 {"${algorithm}", algorithm_},
                                 {"${b}", absl::StrCat(b)},
                                 {"${k}", absl::StrCat(k)}},
                                options);
        if (!module.ok()) {
          return "Fail";
        }
        return absl::StrContains(module.value()->ToString(), pattern) ? " Yes"
                                                                      : "  No";
      };

      csv.appendValue(absl::StrCat(
          "('triton': ", run("triton", kTritonGemmPattern, triton_options_),
          ", 'blas': ", run("blas", kBlasPattern, blas_options_), ")"));
    }
  }
  DumpResults(csv, "backend_support_matrix");
}

// The test does not fail. It just dumps the results in CSV format.
TEST_P(TritonAndBlasSupportForDifferentTensorSizes, Regular2DDot) {
  constexpr absl::string_view kHloText = R"(
    HloModule ${module_name}

    ENTRY e {
      p0 = f32[${m},${k}] parameter(0)
      p1 = f32[${k},${n}] parameter(1)
      ROOT dot = f32[${m},${n}] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0},
        algorithm=${algorithm}
    }
  )";
  CSVWriter csv;
  csv.appendValue("M/N");
  for (int n = 1; n <= kMaxSize; n *= kStepSize) {
    csv.appendValue(n);
  }
  for (int m = 1; m <= kMaxSize; m *= kStepSize) {
    csv.nextRow();
    csv.appendValue(m);
    for (int n = 1; n <= kMaxSize; n *= kStepSize) {
      LOG(INFO) << "Running test for m=" << m << ", n=" << n;
      auto run = [&](std::string backend, absl::string_view pattern,
                     const DebugOptions& options) -> absl::string_view {
        auto test_name = absl::StrReplaceAll(TestName(), {{"/", "_"}});
        auto module_name = absl::StrCat(test_name, "_", backend, "_", m, "_",
                                        kMaxK, "_", n, "_", algorithm_);
        auto module = GetModule(kHloText,
                                {{"${module_name}", module_name},
                                 {"${algorithm}", algorithm_},
                                 {"${m}", absl::StrCat(m)},
                                 {"${n}", absl::StrCat(n)},
                                 {"${k}", absl::StrCat(kMaxK)}},
                                options);
        if (!module.ok()) {
          return "Fail";
        }
        return absl::StrContains(module.value()->ToString(), pattern) ? " Yes"
                                                                      : "  No";
      };

      csv.appendValue(absl::StrCat(
          "('triton': ", run("triton", kTritonGemmPattern, triton_options_),
          ", 'blas': ", run("blas", kBlasPattern, blas_options_), ")"));
    }
  }
  DumpResults(csv, "backend_support_matrix");
}

TEST_P(TritonAndBlasSupportForDifferentTensorSizes,
       IsDotAlgorithmSupportedByTriton) {
  // Here we test which dot algorithm is supported by triton.
  // In case of a change you need to update the expected results.
  constexpr absl::string_view kHloText = R"(
    HloModule ${module_name}

    ENTRY e {
      p0 = f32[${m},${k}] parameter(0)
      p1 = f32[${k},${n}] parameter(1)
      ROOT dot = f32[${m},${n}] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0},
        algorithm=${algorithm}
    }
  )";
  auto m = 128;
  auto n = 128;
  auto k = 128;
  auto run = [&](std::string backend, absl::string_view pattern,
                 const DebugOptions& options) -> absl::StatusOr<bool> {
    auto test_name = absl::StrReplaceAll(TestName(), {{"/", "_"}});
    auto module_name = absl::StrCat(test_name, "_", backend, "_", m, "_", kMaxK,
                                    "_", n, "_", algorithm_);
    auto module = GetModule(kHloText,
                            {{"${module_name}", module_name},
                             {"${algorithm}", algorithm_},
                             {"${m}", absl::StrCat(m)},
                             {"${n}", absl::StrCat(n)},
                             {"${k}", absl::StrCat(k)}},
                            options);
    if (!module.ok()) {
      return module.status();
    }
    std::string module_text = module.value()->ToString();
    if (!Run(std::move(module.value()), false)) {
      return absl::InternalError("failed to run module");
    }
    return absl::StrContains(module_text, pattern);
  };

  auto result_or_status = run("triton", kTritonGemmPattern, triton_options_);
  switch (GetParam()) {
    case PC::ALG_UNSET:
    case PC::ALG_DOT_TF32_TF32_F32:
    case PC::ALG_DOT_TF32_TF32_F32_X3:
    case PC::ALG_DOT_BF16_BF16_F32:
    case PC::ALG_DOT_BF16_BF16_F32_X3:
    case PC::ALG_DOT_BF16_BF16_F32_X6:
    case PC::ALG_DOT_BF16_BF16_F32_X9:
    case PC::ALG_DOT_F32_F32_F32:
      ASSERT_TRUE(result_or_status.status().ok())
          << "failed to compile " << algorithm_;
      EXPECT_TRUE(result_or_status.value())
          << "wrong result for " << algorithm_;
      break;
    case PC::ALG_DOT_F64_F64_F64:
      EXPECT_EQ(result_or_status.status().code(),
                absl::StatusCode::kUnimplemented);
      break;
    default:
      EXPECT_TRUE(false) << "Uncovered algorithm. Please fix: " << algorithm_;
      break;
  }
}

// Applies elementwise absolute value to all arguments to make them
// non-negative.
void MakeNonNegative(std::vector<Literal>& fake_arguments) {
  for (Literal& literal : fake_arguments) {
    literal.MutableEachCell<float>([](absl::Span<const int64_t> indices,
                                      float value) { return std::abs(value); });
  }
}

std::vector<const Literal*> GetLiteralPointers(
    const std::vector<Literal>& fake_arguments) {
  std::vector<const Literal*> fake_argument_ptrs;
  fake_argument_ptrs.reserve(fake_arguments.size());
  for (const Literal& literal : fake_arguments) {
    fake_argument_ptrs.push_back(&literal);
  }
  return fake_argument_ptrs;
}

enum class Backend { kTriton, kBlas };

std::string BackendToString(Backend backend) {
  switch (backend) {
    case Backend::kTriton:
      return "triton";
    case Backend::kBlas:
      return "blas";
    default:
      CHECK(false) << "Uncovered backend. Please fix.";
  }
}

// Returns the maximum relative error for the algorithm, assuming that the
// majority of the error comes from rounding to narrower type, and not error
// due to floating point arithmetic calculation. I.e., we assume that:
//    <contracting dimension> << <narrowing error> / <fp arithmetic error>
// E.g., for BF16xBF16 -> F32, this would mean k << 2^-7 / 2^-23 = 64k
double GetMaxRelErrorForSmallContractingDim(Backend backend,
                                            PC::Algorithm algorithm) {
  // With `ulp` denoting the "unit in the last place", and proper floating point
  // implementation, the test does k multiplications and then k-1 additions per
  // output element. However, we also get an initial error per element due to
  // rounding to bf16, or tf32, depending on the algorithm.
  //
  // Our total error then ends up being k*ulp_f32 + 2*ulp_bf16/tf32.  We can
  // look at an example of a dot product of 2-value vectors [a,b] and [x,y], to
  // get an intuition for it:
  //  (1+ulp_f32)((1+ulp_f32)((1+ulp_bf16)a * (1+ulp_bf16)x)
  //      + (1+ulp_f32)((1+ulp_bf16)b * (1+ulp_bf16)y))
  //   = (1+ulp_f32)(1+ulp_f32)(1+ulp_bf16)(1+ulp_bf16)(ax+by)
  //  ~= (1+2ulp_f32+2ulp_bf16)(ax+by)
  //
  // In the last equality we discard any higher-order errors because they are
  // orders of magnitude smaller than the 1st-order term.
  //
  // Thus, we get 2*ulp_bf16 because the multiplication adds up the errors of
  // the factors, and addition just factors a single error term out. Then we get
  // k*ulp_f32 because each "layer" of operations adds another rounding error
  // (and we have 1 layer of multiplications and k-1 layers of additions).
  //
  // If we have a small k, such as k=8 then the error bounds are:
  //
  // BF16xBF16 -> F32: 8*2^-23 + 2*2^-7 = 2^-20 + 2^-6 ~= 1.6e-2
  // TF32xTF32 -> F32: 8*2^-23 + 2*2^-10 = 2^-20 + 2^-9 ~= 2.0e-3
  //
  // Thus, they do not actually depend on k, since f32 has much higher precision
  // than the rounding mode.
  const absl::flat_hash_map<PC::Algorithm, double> kMaxMeanRelErrorTriton = {
      {PC::ALG_DOT_BF16_BF16_F32, 1.6e-2},
      {PC::ALG_DOT_TF32_TF32_F32, 2.0e-3},
      // TODO: b/407744579 - Understand what the expected error is with various
      // precision-recovering algorithms. For now we just use the errors that
      // we got assuming that the implementation is correct.
      {PC::ALG_DOT_BF16_BF16_F32_X3, 3e-5},
      {PC::ALG_DOT_BF16_BF16_F32_X6, 4e-7},
      {PC::ALG_DOT_BF16_BF16_F32_X9, 4e-7},
      {PC::ALG_DOT_TF32_TF32_F32_X3, 5e-7}};

  const absl::flat_hash_map<PC::Algorithm, double> kMaxMeanRelErrorBlas = {
      {PC::ALG_DOT_BF16_BF16_F32, 3.3e-3},
      {PC::ALG_DOT_TF32_TF32_F32, 4.1e-4},
      {PC::ALG_DOT_BF16_BF16_F32_X3, 2.4e-5},
      {PC::ALG_DOT_TF32_TF32_F32_X3, 5e-7},
      {PC::ALG_DOT_BF16_BF16_F32_X6, 1.6e-7},
      {PC::ALG_DOT_BF16_BF16_F32_X9, 6e-8}};
  if (backend == Backend::kTriton) {
    auto max_rel_error_it = kMaxMeanRelErrorTriton.find(algorithm);
    CHECK(max_rel_error_it != kMaxMeanRelErrorTriton.end());
    return max_rel_error_it->second;
  }

  if (backend == Backend::kBlas) {
    auto max_rel_error_it = kMaxMeanRelErrorBlas.find(algorithm);
    CHECK(max_rel_error_it != kMaxMeanRelErrorBlas.end());
    return max_rel_error_it->second;
  }

  CHECK(false) << "Uncovered backend. Please fix.";
}

INSTANTIATE_TEST_SUITE_P(
    TritonAndBlasSupportForDifferentTensorSizes,
    TritonAndBlasSupportForDifferentTensorSizes,
    ::testing::ValuesIn(
        {PC::ALG_DOT_BF16_BF16_F32, PC::ALG_DOT_BF16_BF16_F32_X3,
         PC::ALG_DOT_BF16_BF16_F32_X6, PC::ALG_DOT_BF16_BF16_F32_X9,
         PC::ALG_DOT_F32_F32_F32, PC::ALG_DOT_TF32_TF32_F32,
         PC::ALG_DOT_TF32_TF32_F32_X3, PC::ALG_DOT_F64_F64_F64, PC::ALG_UNSET}),
    AlgorithmTestParamToString);

template <typename T>
void PrintHistogram(absl::string_view name, absl::Span<T> values,
                    const std::vector<double>& expected_values) {
  // Build the histogram of the relative differences.
  std::vector<double> rel_errors;
  rel_errors.reserve(values.size());
  for (int i = 0; i < values.size(); ++i) {
    double rel_difference =
        ((double)values[i] - expected_values[i]) / std::abs(expected_values[i]);
    rel_errors.push_back(rel_difference);
  }
  double max_rel_error =
      *std::max_element(rel_errors.begin(), rel_errors.end());
  double min_rel_error =
      *std::min_element(rel_errors.begin(), rel_errors.end());
  double rel_error_range = max_rel_error - min_rel_error;
  constexpr int kNumBins = 40;
  double bin_width = rel_error_range / kNumBins;
  std::vector<int> histogram(kNumBins, 0);
  double rel_error_sum = 0.0;
  for (int i = 0; i < rel_errors.size(); ++i) {
    rel_error_sum += rel_errors[i];
    int bin = static_cast<int>((rel_errors[i] - min_rel_error) / bin_width);
    if (bin >= kNumBins) {
      bin = kNumBins - 1;
    }
    histogram[bin]++;
  }
  int samples_count = values.size();
  int bar_width = 200;
  int64_t samples = 0;
  double mean_rel_error = rel_error_sum / values.size();
  bool median_found = false;
  std::tuple<int, double, double> median_bin;
  for (int i = 0; i < kNumBins; ++i) {
    samples += histogram[i];
    double bin_start = min_rel_error + i * bin_width;
    double bin_end = min_rel_error + (i + 1) * bin_width;
    std::string bar =
        std::string(histogram[i] * bar_width / samples_count, '*');
    if (!median_found && samples >= samples_count / 2) {
      median_bin = std::make_tuple(i, bin_start, bin_end);
      median_found = true;
      bar += " <--- median";
    }
    if (mean_rel_error >= bin_start && mean_rel_error < bin_end) {
      bar += " <--- mean";
    }
    if (bin_start <= 0.0 && bin_end >= 0.0) {
      bar += " <--- zero";
    }
    std::string line =
        absl::StrFormat("%2d: [% 1.3e, % 1.3e) %7d %s\n", i, bin_start, bin_end,
                        histogram[i], bar.c_str());
    std::cerr << "hist: " << line;
  }
  std::cerr << "stats: " << name << " "
            << absl::StrFormat("min rel error, %1.3e\n", min_rel_error);
  std::cerr << "stats: " << name << " "
            << absl::StrFormat("max rel error, %1.3e\n", max_rel_error);
  std::cerr << "stats: " << name << " "
            << absl::StrFormat(
                   "max abs rel error, %1.3e\n",
                   std::max(std::abs(min_rel_error), std::abs(max_rel_error)));
  std::cerr << "stats: " << name << " "
            << absl::StrFormat("rel error range, %1.3e\n",
                               max_rel_error - min_rel_error);
  std::cerr << "stats: " << name << " "
            << absl::StrFormat("median bin, %d [%1.3e - %1.3e)\n",
                               std::get<0>(median_bin), std::get<1>(median_bin),
                               std::get<2>(median_bin));
  std::cerr << "stats: " << name << " "
            << absl::StrFormat("mean rel error, %1.3e\n", mean_rel_error);
  std::cerr << "stats: \n";
}

class PrecisionTests
    : public AlgorithmTest,
      public NumericTestsArguments,
      public WithParamInterface<::testing::tuple<PC::Algorithm, Backend>> {
 public:
 protected:
  std::vector<double> RunReferenceDot(
      const std::vector<const Literal*>& fake_argument_ptrs, int m_size,
      int n_size, int k_size) {
    absl::Time start = absl::Now();
    std::vector<double> ref_result(m_size * n_size, 0.0);
    auto lhs = fake_argument_ptrs[0]->data<float>();
    auto rhs = fake_argument_ptrs[1]->data<float>();
    for (int m = 0; m < m_size; ++m) {
      for (int n = 0; n < n_size; ++n) {
        for (int k = 0; k < k_size; ++k) {
          double lhs_val = lhs[m * k_size + k];
          double rhs_val = rhs[n * k_size + k];
          ref_result[m * n_size + n] += lhs_val * rhs_val;
        }
      }
    }
    auto duration = absl::Now() - start;
    std::cerr << "Reference dot took " << duration << " for " << m_size << "x"
              << n_size << "x" << k_size << "\n";
    return ref_result;
  }

  absl::Status CheckGemmPattern(const HloModule& module,
                                absl::string_view pattern) {
    TF_ASSIGN_OR_RETURN(bool ok, RunFileCheck(module.ToString(), pattern));
    if (!ok) {
      return absl::InternalError(
          absl::StrCat("The module does not contain the pattern: ", pattern));
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::unique_ptr<HloModule>> GetSimpleDotModule(
      int lhs_outer_dim, int rhs_outer_dim, int contracting_dim,
      PC::Algorithm algorithm, Backend backend) {
    std::string hlo_text = absl::StrReplaceAll(
        kHloTextPattern, {{"${test_name}", HloModuleTestName()},
                          {"${m}", absl::StrCat(lhs_outer_dim)},
                          {"${n}", absl::StrCat(rhs_outer_dim)},
                          {"${k}", absl::StrCat(contracting_dim)},
                          {"${algorithm}", AlgorithmToString(algorithm)}});
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                        ParseAndReturnVerifiedModule(hlo_text));
    auto debug_options = module->config().debug_options();
    if (backend == Backend::kTriton) {
      debug_options.set_xla_gpu_enable_triton_gemm(true);
      debug_options.set_xla_gpu_cublas_fallback(false);
    } else if (backend == Backend::kBlas) {
      debug_options.set_xla_gpu_enable_triton_gemm(false);
      debug_options.set_xla_gpu_cublas_fallback(true);
    } else {
      return absl::InvalidArgumentError("Invalid backend");
    }
    module->mutable_config().set_debug_options(debug_options);
    TF_ASSIGN_OR_RETURN(module, GetOptimizedModule(std::move(module)));
    if (backend == Backend::kTriton) {
      TF_RETURN_IF_ERROR(CheckGemmPattern(*module, "CHECK: __triton_gemm"));
    } else if (backend == Backend::kBlas) {
      TF_RETURN_IF_ERROR(CheckGemmPattern(*module, "CHECK: __cublas$gemm"));
    } else {
      return absl::InvalidArgumentError("Invalid backend");
    }
    return module;
  }

 private:
  static constexpr absl::string_view kHloTextPattern = R"(
    HloModule ${test_name}

    ENTRY main {
      p0 = f32[${m},${k}]{1,0} parameter(0)
      p1 = f32[${n},${k}]{1,0} parameter(1)
      ROOT %dot = f32[${m},${n}]{1,0} dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1},
        algorithm=${algorithm}
    }
  )";
};

using ::testing::Combine;
using ::testing::Values;

std::string AlgorithmAndBackendTestParamToString(
    const TestParamInfo<::testing::tuple<PC::Algorithm, Backend>>& info) {
  PC::Algorithm algorithm = std::get<0>(info.param);
  Backend backend = std::get<1>(info.param);
  return absl::StrCat(BackendToString(backend), "_",
                      AlgorithmToString(algorithm));
}

MATCHER_P(RelativeDifferenceIsWithin, max_rel_difference, "") {
  double got = std::get<0>(arg);
  double expected = std::get<1>(arg);
  double rel_difference = std::abs((got - expected) / expected);
  *result_listener << "has relative difference " << rel_difference << " = ("
                   << got << " - " << expected << ") / " << expected
                   << " that should be within " << max_rel_difference;
  return rel_difference <= max_rel_difference;
}

TEST_P(PrecisionTests, PrecisionCheck) {
  if (std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp())) {
    GTEST_SKIP() << "Precision tests is unknown for ROCM.";
  }

  PC::Algorithm algorithm = std::get<0>(GetParam());
  Backend backend = std::get<1>(GetParam());
  // Use small contracting dimensions to avoid false-negatives due to changing
  // contracting dimension tiling factors.
  constexpr int kLhsOuterDim = 1024;
  constexpr int kRhsOuterDim = 1024;
  constexpr int kContractingDim = 8;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> test_module,
      GetSimpleDotModule(kLhsOuterDim, kRhsOuterDim, kContractingDim, algorithm,
                         backend));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> fake_arguments,
      MakeFakeArguments(test_module.get(), /*pseudo_random=*/true,
                        /*use_large_range=*/false,
                        /*treat_gte_as_data_formatting=*/false,
                        /*max_bits_of_precision=*/23));
  // Ensure there are no negative arguments to avoid unbounded relative errors
  // due to subtracting two similarly large numbers.
  MakeNonNegative(fake_arguments);
  std::vector<const Literal*> fake_argument_ptrs =
      GetLiteralPointers(fake_arguments);
  std::vector<double> ref_result = RunReferenceDot(
      fake_argument_ptrs, kLhsOuterDim, kRhsOuterDim, kContractingDim);
  TF_ASSERT_OK_AND_ASSIGN(
      Literal test_result,
      test_runner().Execute(std::move(test_module), fake_argument_ptrs,
                            /*run_hlo_passes=*/false));
  std::cerr << "\n";
  EXPECT_THAT(llvm::zip(test_result.data<float>(), ref_result),
              ::testing::Each(RelativeDifferenceIsWithin(
                  GetMaxRelErrorForSmallContractingDim(backend, algorithm))));
  auto name =
      absl::StrCat(BackendToString(backend), "_", AlgorithmToString(algorithm));
  PrintHistogram(name, test_result.data<float>(), ref_result);
}

INSTANTIATE_TEST_SUITE_P(
    PrecisionTests, PrecisionTests,
    Combine(Values(PC::ALG_DOT_TF32_TF32_F32, PC::ALG_DOT_TF32_TF32_F32_X3,
                   PC::ALG_DOT_BF16_BF16_F32, PC::ALG_DOT_BF16_BF16_F32_X3,
                   PC::ALG_DOT_BF16_BF16_F32_X6, PC::ALG_DOT_BF16_BF16_F32_X9),
            Values(Backend::kTriton, Backend::kBlas)),
    AlgorithmAndBackendTestParamToString);

}  // namespace
}  // namespace gpu
}  // namespace xla
