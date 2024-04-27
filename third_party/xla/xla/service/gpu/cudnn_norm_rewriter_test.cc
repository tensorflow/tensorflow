/* Copyright 2023 The OpenXLA Authors.

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

#include <string>

#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/stream_executor/device_description.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cudnn/cudnn.h"  // IWYU pragma: keep
#include "third_party/gpus/cudnn/cudnn_version.h"
#endif

#include "xla/service/gpu/tests/gpu_codegen_test.h"

namespace xla {
namespace gpu {
namespace {

class CudnnNormRewriterTest : public GpuCodegenTest {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cudnn_layer_norm(true);
    return debug_options;
  }

 protected:
  void TestNorm(std::string hlo_text, std::string optimized_hlo) {
    EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
    MatchOptimizedHlo(hlo_text, optimized_hlo);
  }
};

// The following tests evaluate LayerNormXDY configurations, with X the rank of
// the input and Y the dimensions that are normalized.
TEST_F(CudnnNormRewriterTest, LayerNorm2D1) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4] parameter(0)
        input_square = f32[2,4] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2] reduce(input_square, c0), dimensions={1}, to_apply=apply
        r_nelems = f32[] constant(0.25)
        r_nelems_bcast = f32[2] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2] multiply(input_square_sum, r_nelems_bcast)
        input_sum = f32[2] reduce(input, c0),dimensions={1}, to_apply=apply
        input_mean = f32[2] multiply(input_sum, r_nelems_bcast)
        input_mean_square = f32[2] multiply(input_mean, input_mean)
        variance = f32[2] subtract(input_square_mean, input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2] add(variance, epsilon_bcast)
        norm_factor = f32[2] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4] broadcast(norm_factor), dimensions={0}
        input_mean_bcast = f32[2,4] broadcast(input_mean), dimensions={0}
        input_center = f32[2,4] subtract(input, input_mean_bcast)
        norm = f32[2,4] multiply(norm_factor_bcast, input_center)
        scale = f32[4] parameter(1)
        scale_bcast = f32[2,4] broadcast(scale), dimensions={1}
        norm_scale = f32[2,4] multiply(norm, scale_bcast)
        bias = f32[4] parameter(2)
        bias_broadcast = f32[2,4] broadcast(bias), dimensions={1}
        ROOT out = f32[2,4] add(norm_scale, bias_broadcast)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4], {{.*}}: f32[4], {{.*}}: f32[4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4]{1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[2,4,1,1]{3,2,1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[2,4,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE:%[^ ]+]] = f32[2,4,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:  ROOT [[GTE_BITCAST:%[^ ]+]] = f32[2,4]{1,0} bitcast([[GTE]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNorm4D3) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        input_square = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,4,6] reduce(input_square, c0), dimensions={3}, to_apply=apply
        r_nelems = f32[] constant(0.125)
        r_nelems_bcast = f32[2,4,6] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,4,6] multiply(input_square_sum, r_nelems_bcast)
        input_sum = f32[2,4,6] reduce(input, c0), dimensions={3}, to_apply=apply
        input_mean = f32[2,4,6] multiply(input_sum, r_nelems_bcast)
        input_mean_square = f32[2,4,6] multiply(input_mean, input_mean)
        variance = f32[2,4,6] subtract(input_square_mean, input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,4,6] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,4,6] add(variance, epsilon_bcast)
        norm_factor = f32[2,4,6] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4,6,8] broadcast(norm_factor), dimensions={0,1,2}
        input_mean_bcast = f32[2,4,6,8] broadcast(input_mean), dimensions={0,1,2}
        input_center = f32[2,4,6,8] subtract(input, input_mean_bcast)
        norm = f32[2,4,6,8] multiply(norm_factor_bcast, input_center)
        scale = f32[8] parameter(1)
        scale_bcast = f32[2,4,6,8] broadcast(scale), dimensions={3}
        norm_scale = f32[2,4,6,8] multiply(norm, scale_bcast)
        bias = f32[8] parameter(2)
        bias_bcast = f32[2,4,6,8] broadcast(bias), dimensions={3}
        ROOT out = f32[2,4,6,8] add(norm_scale, bias_bcast)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4,6,8], {{.*}}: f32[8], {{.*}}: f32[8]) -> f32[2,4,6,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[48,8,1,1]{3,2,1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[8]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[8]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[48,8,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE:%[^ ]+]] = f32[48,8,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:  ROOT [[GTE_BITCAST:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} bitcast([[GTE]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNorm4D3Degenerate0) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[1,4,6,8] parameter(0)
        input_square = f32[1,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[1,4,6] reduce(input_square, c0), dimensions={3}, to_apply=apply
        r_nelems = f32[] constant(0.125)
        r_nelems_bcast = f32[1,4,6] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[1,4,6] multiply(input_square_sum, r_nelems_bcast)
        input_sum = f32[1,4,6] reduce(input, c0), dimensions={3}, to_apply=apply
        input_mean = f32[1,4,6] multiply(input_sum, r_nelems_bcast)
        input_mean_square = f32[1,4,6] multiply(input_mean, input_mean)
        variance = f32[1,4,6] subtract(input_square_mean, input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[1,4,6] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[1,4,6] add(variance, epsilon_bcast)
        norm_factor = f32[1,4,6] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[1,4,6,8] broadcast(norm_factor), dimensions={0,1,2}
        input_mean_bcast = f32[1,4,6,8] broadcast(input_mean), dimensions={0,1,2}
        input_center = f32[1,4,6,8] subtract(input, input_mean_bcast)
        norm = f32[1,4,6,8] multiply(norm_factor_bcast, input_center)
        scale = f32[8] parameter(1)
        scale_bcast = f32[1,4,6,8] broadcast(scale), dimensions={3}
        norm_scale = f32[1,4,6,8] multiply(norm, scale_bcast)
        bias = f32[8] parameter(2)
        bias_bcast = f32[1,4,6,8] broadcast(bias), dimensions={3}
        ROOT out = f32[1,4,6,8] add(norm_scale, bias_bcast)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[1,4,6,8], {{.*}}: f32[8], {{.*}}: f32[8]) -> f32[1,4,6,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[1,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[24,8,1,1]{3,2,1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[8]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[8]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[24,8,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE:%[^ ]+]] = f32[24,8,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:  ROOT [[GTE_BITCAST:%[^ ]+]] = f32[1,4,6,8]{3,2,1,0} bitcast([[GTE]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNorm4D2) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        input_square = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,4,8] reduce(input_square, c0), dimensions={2}, to_apply=apply
        r_nelems = f32[] constant(0.166667)
        r_nelems_bcast = f32[2,4,8] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,4,8] multiply(input_square_sum, r_nelems_bcast)
        reduce = f32[2,4,8] reduce(input, c0), dimensions={2}, to_apply=apply
        input_mean = f32[2,4,8] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2,4,8] multiply(input_mean, input_mean)
        variance = f32[2,4,8] subtract(input_square_mean, input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,4,8] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,4,8] add(variance, epsilon_bcast)
        norm_factor = f32[2,4,8] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4,6,8] broadcast(norm_factor), dimensions={0,1,3}
        input_mean_bcast = f32[2,4,6,8] broadcast(input_mean), dimensions={0,1,3}
        input_center = f32[2,4,6,8] subtract(input, input_mean_bcast)
        norm = f32[2,4,6,8] multiply(norm_factor_bcast, input_center)
        scale = f32[6] parameter(1)
        scale_bcast = f32[2,4,6,8] broadcast(scale), dimensions={2}
        norm_scale = f32[2,4,6,8] multiply(norm, scale_bcast)
        bias = f32[6] parameter(2)
        bias_broadcast = f32[2,4,6,8] broadcast(bias), dimensions={2}
        ROOT out = f32[2,4,6,8] add(norm_scale, bias_broadcast)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4,6,8], {{.*}}: f32[6], {{.*}}: f32[6]) -> f32[2,4,6,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE:%[^ ]+]] = f32[2,4,8,6]{3,2,1,0} transpose([[P0]]), dimensions={0,1,3,2}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[64,6,1,1]{3,2,1,0} bitcast([[TRANSPOSE]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[6]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[6,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[6]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[6,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[64,6,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE:%[^ ]+]] = f32[64,6,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:  ROOT [[FUSION:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} fusion([[GTE]]), kind=kLoop, calls=[[FUSED_COMPUTATION:%[^ ]+]]
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNorm4D2Degenerate1) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,1,6,8] parameter(0)
        input_square = f32[2,1,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,1,8] reduce(input_square, c0), dimensions={2}, to_apply=apply
        r_nelems = f32[] constant(0.166667)
        r_nelems_bcast = f32[2,1,8] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,1,8] multiply(input_square_sum, r_nelems_bcast)
        reduce = f32[2,1,8] reduce(input, c0), dimensions={2}, to_apply=apply
        input_mean = f32[2,1,8] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2,1,8] multiply(input_mean, input_mean)
        variance = f32[2,1,8] subtract(input_square_mean, input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,1,8] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,1,8] add(variance, epsilon_bcast)
        norm_factor = f32[2,1,8] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,1,6,8] broadcast(norm_factor), dimensions={0,1,3}
        input_mean_bcast = f32[2,1,6,8] broadcast(input_mean), dimensions={0,1,3}
        input_center = f32[2,1,6,8] subtract(input, input_mean_bcast)
        norm = f32[2,1,6,8] multiply(norm_factor_bcast, input_center)
        scale = f32[6] parameter(1)
        scale_bcast = f32[2,1,6,8] broadcast(scale), dimensions={2}
        norm_scale = f32[2,1,6,8] multiply(norm, scale_bcast)
        bias = f32[6] parameter(2)
        bias_broadcast = f32[2,1,6,8] broadcast(bias), dimensions={2}
        ROOT out = f32[2,1,6,8] add(norm_scale, bias_broadcast)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,1,6,8], {{.*}}: f32[6], {{.*}}: f32[6]) -> f32[2,1,6,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,1,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE:%[^ ]+]] = f32[1,2,8,6]{3,2,1,0} transpose([[P0]]), dimensions={1,0,3,2}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[16,6,1,1]{3,2,1,0} bitcast([[TRANSPOSE]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[6]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[6,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[6]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[6,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[16,6,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE:%[^ ]+]] = f32[16,6,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:  ROOT [[FUSION:%[^ ]+]] = f32[2,1,6,8]{3,2,1,0} fusion([[GTE]]), kind=kLoop, calls=[[FUSED_COMPUTATION:%[^ ]+]]
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNorm4D12) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        input_square = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,8] reduce(input_square, c0), dimensions={1,2}, to_apply=apply
        r_nelems = f32[] constant(0.041667)
        r_nelems_bcast = f32[2,8] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,8] multiply(input_square_sum, r_nelems_bcast)
        reduce = f32[2,8] reduce(input, c0), dimensions={1,2}, to_apply=apply
        input_mean = f32[2,8] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2,8] multiply(input_mean, input_mean)
        variance = f32[2,8] subtract(input_square_mean, input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,8] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,8] add(variance, epsilon_bcast)
        norm_factor = f32[2,8] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4,6,8] broadcast(norm_factor), dimensions={0,3}
        input_mean_bcast = f32[2,4,6,8] broadcast(input_mean), dimensions={0,3}
        input_center = f32[2,4,6,8] subtract(input, input_mean_bcast)
        norm = f32[2,4,6,8] multiply(norm_factor_bcast, input_center)
        scale = f32[4,6] parameter(1)
        scale_bcast = f32[2,4,6,8] broadcast(scale), dimensions={1,2}
        norm_scale = f32[2,4,6,8] multiply(norm, scale_bcast)
        bias = f32[4,6] parameter(2)
        bias_broadcast = f32[2,4,6,8] broadcast(bias), dimensions={1,2}
        ROOT out = f32[2,4,6,8] add(norm_scale, bias_broadcast)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4,6,8], {{.*}}: f32[4,6], {{.*}}: f32[4,6]) -> f32[2,4,6,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE:%[^ ]+]] = f32[2,8,4,6]{3,2,1,0} transpose([[P0]]), dimensions={0,3,1,2}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[16,4,6,1]{3,2,1,0} bitcast([[TRANSPOSE]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,6]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,6,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4,6]{1,0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,6,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[16,4,6,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE:%[^ ]+]] = f32[16,4,6,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:  ROOT  [[FUSION:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} fusion([[GTE]]), kind=kLoop, calls=[[FUSED_COMPUTATION:%[^ ]+]]
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNorm4D12Degenerate2) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,1,8] parameter(0)
        input_square = f32[2,4,1,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,8] reduce(input_square, c0), dimensions={1,2}, to_apply=apply
        r_nelems = f32[] constant(0.25)
        r_nelems_bcast = f32[2,8] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,8] multiply(input_square_sum, r_nelems_bcast)
        reduce = f32[2,8] reduce(input, c0), dimensions={1,2}, to_apply=apply
        input_mean = f32[2,8] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2,8] multiply(input_mean, input_mean)
        variance = f32[2,8] subtract(input_square_mean, input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,8] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,8] add(variance, epsilon_bcast)
        norm_factor = f32[2,8] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4,1,8] broadcast(norm_factor), dimensions={0,3}
        input_mean_bcast = f32[2,4,1,8] broadcast(input_mean), dimensions={0,3}
        input_center = f32[2,4,1,8] subtract(input, input_mean_bcast)
        norm = f32[2,4,1,8] multiply(norm_factor_bcast, input_center)
        scale = f32[4,1] parameter(1)
        scale_bcast = f32[2,4,1,8] broadcast(scale), dimensions={1,2}
        norm_scale = f32[2,4,1,8] multiply(norm, scale_bcast)
        bias = f32[4,1] parameter(2)
        bias_broadcast = f32[2,4,1,8] broadcast(bias), dimensions={1,2}
        ROOT out = f32[2,4,1,8] add(norm_scale, bias_broadcast)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4,1,8], {{.*}}: f32[4,1], {{.*}}: f32[4,1]) -> f32[2,4,1,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,1,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE:%[^ ]+]] = f32[1,2,8,4]{3,2,1,0} transpose([[P0]]), dimensions={2,0,3,1}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[16,4,1,1]{3,2,1,0} bitcast([[TRANSPOSE]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,1]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4,1]{1,0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[16,4,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE:%[^ ]+]] = f32[16,4,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:  ROOT  [[FUSION:%[^ ]+]] = f32[2,4,1,8]{3,2,1,0} fusion([[GTE]]), kind=kLoop, calls=[[FUSED_COMPUTATION:%[^ ]+]]
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNorm4D3IncorrectScaleBroadcast) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,2,2,2] parameter(0)
        input_square = f32[2,2,2,2] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,2,2] reduce(input_square, c0), dimensions={3}, to_apply=apply
        r_nelems = f32[] constant(0.5)
        r_nelems_bcast = f32[2,2,2] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,2,2] multiply(input_square_sum, r_nelems_bcast)
        input_sum = f32[2,2,2] reduce(input, c0), dimensions={3}, to_apply=apply
        input_mean = f32[2,2,2] multiply(input_sum, r_nelems_bcast)
        input_mean_square = f32[2,2,2] multiply(input_mean, input_mean)
        variance = f32[2,2,2] subtract(input_square_mean, input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,2,2] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,2,2] add(variance, epsilon_bcast)
        norm_factor = f32[2,2,2] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,2,2,2] broadcast(norm_factor), dimensions={0,1,2}
        input_mean_bcast = f32[2,2,2,2] broadcast(input_mean), dimensions={0,1,2}
        input_center = f32[2,2,2,2] subtract(input, input_mean_bcast)
        norm = f32[2,2,2,2] multiply(norm_factor_bcast, input_center)
        scale = f32[2] parameter(1)
        scale_bcast = f32[2,2,2,2] broadcast(scale), dimensions={2}
        norm_scale = f32[2,2,2,2] multiply(norm, scale_bcast)
        bias = f32[2] parameter(2)
        bias_bcast = f32[2,2,2,2] broadcast(bias), dimensions={3}
        ROOT out = f32[2,2,2,2] add(norm_scale, bias_bcast)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,2,2,2], {{.*}}: f32[2], {{.*}}: f32[2]) -> f32[2,2,2,2] {
; CHECK-NOT:           custom_call_target="__cudnn$norm"
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrain2D1) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4] parameter(0)
        input_square = f32[2,4] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2] reduce(input_square, c0), dimensions={1}, to_apply=apply
        r_nelems = f32[] constant(0.25)
        r_nelems_bcast = f32[2] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2] multiply(input_square_sum,r_nelems_bcast)
        reduce = f32[2] reduce(input, c0), dimensions={1}, to_apply=apply
        input_mean = f32[2] multiply(reduce,r_nelems_bcast)
        input_mean_square = f32[2] multiply(input_mean,input_mean)
        variance = f32[2] subtract(input_square_mean,input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2] add(variance, epsilon_bcast)
        norm_factor = f32[2] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4] broadcast(norm_factor), dimensions={0}
        input_mean_bcast = f32[2,4] broadcast(input_mean), dimensions={0}
        input_center = f32[2,4] subtract(input,input_mean_bcast)
        norm = f32[2,4] multiply(norm_factor_bcast,input_center)
        scale = f32[4] parameter(1)
        scale_bcast = f32[2,4] broadcast(scale), dimensions={1}
        norm_scale = f32[2,4] multiply(norm,scale_bcast)
        bias = f32[4] parameter(2)
        bias_broadcast = f32[2,4] broadcast(bias), dimensions={1}
        norm_scale_bias = f32[2,4] add(norm_scale, bias_broadcast)
        norm_factor_cube = f32[2] divide(norm_factor, variance_plus_epsilon)
        ROOT out = (f32[2,4], f32[2], f32[2], f32[2]) tuple(norm_scale_bias, input_mean, norm_factor, norm_factor_cube)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4], {{.*}}: f32[4], {{.*}}: f32[4]) -> (f32[2,4], f32[2], f32[2], f32[2]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4]{1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[2,4,1,1]{3,2,1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[2,4,1,1]{3,2,1,0}, f32[2,1,1,1]{3,2,1,0}, f32[2,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE0:%[^ ]+]] = f32[2,4,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:    [[GTE0_BITCAST:%[^ ]+]] = f32[2,4]{1,0} bitcast([[GTE0]])
; CHECK-NEXT:    [[GTE1:%[^ ]+]] = f32[2,1,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=1
; CHECK-NEXT:    [[GTE1_BITCAST:%[^ ]+]] = f32[2]{0} bitcast([[GTE1]])
; CHECK-NEXT:    [[GTE2:%[^ ]+]] = f32[2,1,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=2
; CHECK-NEXT:    [[GTE2_BITCAST:%[^ ]+]] = f32[2]{0} bitcast([[GTE2]])
; CHECK-NEXT:    [[FUSION:%[^ ]+]] = f32[2]{0} fusion([[GTE2]]), kind=kLoop, calls=[[FUSED_COMPUTATION:%[^ ]+]]
; CHECK-NEXT:  ROOT [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, f32[2]{0}, f32[2]{0}, f32[2]{0}) tuple([[GTE0_BITCAST]], [[GTE1_BITCAST]], [[GTE2_BITCAST]], [[FUSION]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrain4D3) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        input_square = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,4,6] reduce(input_square, c0), dimensions={3}, to_apply=apply
        r_nelems = f32[] constant(0.125)
        r_nelems_bcast = f32[2,4,6] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,4,6] multiply(input_square_sum, r_nelems_bcast)
        reduce = f32[2,4,6] reduce(input, c0), dimensions={3}, to_apply=apply
        input_mean = f32[2,4,6] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2,4,6] multiply(input_mean, input_mean)
        variance = f32[2,4,6] subtract(input_square_mean, input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,4,6] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,4,6] add(variance, epsilon_bcast)
        norm_factor = f32[2,4,6] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4,6,8] broadcast(norm_factor), dimensions={0,1,2}
        input_mean_bcast = f32[2,4,6,8] broadcast(input_mean), dimensions={0,1,2}
        input_center = f32[2,4,6,8] subtract(input, input_mean_bcast)
        norm = f32[2,4,6,8] multiply(norm_factor_bcast, input_center)
        scale = f32[8] parameter(1)
        scale_bcast = f32[2,4,6,8] broadcast(scale), dimensions={3}
        norm_scale = f32[2,4,6,8] multiply(norm,scale_bcast)
        bias = f32[8] parameter(2)
        bias_broadcast = f32[2,4,6,8] broadcast(bias), dimensions={3}
        norm_scale_bias = f32[2,4,6,8] add(norm_scale, bias_broadcast)
        norm_factor_cube = f32[2,4,6] divide(norm_factor, variance_plus_epsilon)
        ROOT out = (f32[2,4,6,8], f32[2,4,6], f32[2,4,6], f32[2,4,6]) tuple(norm_scale_bias, input_mean, norm_factor, norm_factor_cube)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4,6,8], {{.*}}: f32[8], {{.*}}: f32[8]) -> (f32[2,4,6,8], f32[2,4,6], f32[2,4,6], f32[2,4,6]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[48,8,1,1]{3,2,1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[8]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[8]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[48,8,1,1]{3,2,1,0}, f32[48,1,1,1]{3,2,1,0}, f32[48,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE0:%[^ ]+]] = f32[48,8,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:    [[GTE0_BITCAST:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} bitcast([[GTE0]])
; CHECK-NEXT:    [[GTE1:%[^ ]+]] = f32[48,1,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=1
; CHECK-NEXT:    [[GTE1_BITCAST:%[^ ]+]] = f32[2,4,6]{2,1,0} bitcast([[GTE1]])
; CHECK-NEXT:    [[GTE2:%[^ ]+]] = f32[48,1,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=2
; CHECK-NEXT:    [[GTE2_BITCAST:%[^ ]+]] = f32[2,4,6]{2,1,0} bitcast([[GTE2]])
; CHECK-NEXT:    [[FUSION:%[^ ]+]] = f32[2,4,6]{2,1,0} fusion([[GTE2]]), kind=kLoop, calls=[[FUSED_COMPUTATION:%[^ ]+]]
; CHECK-NEXT:  ROOT [[OUT:%[^ ]+]] = (f32[2,4,6,8]{3,2,1,0}, f32[2,4,6]{2,1,0}, f32[2,4,6]{2,1,0}, f32[2,4,6]{2,1,0}) tuple([[GTE0_BITCAST]], [[GTE1_BITCAST]], [[GTE2_BITCAST]], [[FUSION]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrain4D12) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        input_square = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,8] reduce(input_square, c0), dimensions={1,2}, to_apply=apply
        r_nelems = f32[] constant(0.041667)
        r_nelems_bcast = f32[2,8] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,8] multiply(input_square_sum, r_nelems_bcast)
        reduce = f32[2,8] reduce(input, c0), dimensions={1,2}, to_apply=apply
        input_mean = f32[2,8] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2,8] multiply(input_mean, input_mean)
        variance = f32[2,8] subtract(input_square_mean, input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,8] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,8] add(variance, epsilon_bcast)
        norm_factor = f32[2,8] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4,6,8] broadcast(norm_factor), dimensions={0,3}
        input_mean_bcast = f32[2,4,6,8] broadcast(input_mean), dimensions={0,3}
        input_center = f32[2,4,6,8] subtract(input, input_mean_bcast)
        norm = f32[2,4,6,8] multiply(norm_factor_bcast, input_center)
        scale = f32[4,6] parameter(1)
        scale_bcast = f32[2,4,6,8] broadcast(scale), dimensions={1,2}
        norm_scale = f32[2,4,6,8] multiply(norm, scale_bcast)
        bias = f32[4,6] parameter(2)
        bias_broadcast = f32[2,4,6,8] broadcast(bias), dimensions={1,2}
        norm_scale_bias = f32[2,4,6,8] add(norm_scale, bias_broadcast)
        norm_factor_cube = f32[2,8] divide(norm_factor, variance_plus_epsilon)
        ROOT out = (f32[2,4,6,8], f32[2,8], f32[2,8], f32[2,8]) tuple(norm_scale_bias, input_mean, norm_factor, norm_factor_cube)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4,6,8], {{.*}}: f32[4,6], {{.*}}: f32[4,6]) -> (f32[2,4,6,8], f32[2,8], f32[2,8], f32[2,8]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE:%[^ ]+]] = f32[2,8,4,6]{3,2,1,0} transpose([[P0]]), dimensions={0,3,1,2}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[16,4,6,1]{3,2,1,0} bitcast([[TRANSPOSE]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,6]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,6,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4,6]{1,0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,6,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[16,4,6,1]{3,2,1,0}, f32[16,1,1,1]{3,2,1,0}, f32[16,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE0:%[^ ]+]] = f32[16,4,6,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:    [[FUSION0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} fusion([[GTE0]]), kind=kLoop, calls=[[FUSED_COMPUTATION0:%[^ ]+]]
; CHECK-NEXT:    [[GTE1:%[^ ]+]] = f32[16,1,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=1
; CHECK-NEXT:    [[GTE1_BITCAST:%[^ ]+]] = f32[2,8]{1,0} bitcast([[GTE1]])
; CHECK-NEXT:    [[GTE2:%[^ ]+]] = f32[16,1,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=2
; CHECK-NEXT:    [[GTE2_BITCAST:%[^ ]+]] = f32[2,8]{1,0} bitcast([[GTE2]])
; CHECK-NEXT:    [[FUSION1:%[^ ]+]] = f32[2,8]{1,0} fusion([[GTE2]]), kind=kLoop, calls=[[FUSED_COMPUTATION1:%[^ ]+]]
; CHECK-NEXT:  ROOT [[OUT:%[^ ]+]] = (f32[2,4,6,8]{3,2,1,0}, f32[2,8]{1,0}, f32[2,8]{1,0}, f32[2,8]{1,0}) tuple([[FUSION0]], [[GTE1_BITCAST]], [[GTE2_BITCAST]], [[FUSION1]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrain4D12Degenerate2) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,1,8] parameter(0)
        input_square = f32[2,4,1,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,8] reduce(input_square, c0), dimensions={1,2}, to_apply=apply
        r_nelems = f32[] constant(0.25)
        r_nelems_bcast = f32[2,8] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,8] multiply(input_square_sum, r_nelems_bcast)
        reduce = f32[2,8] reduce(input, c0), dimensions={1,2}, to_apply=apply
        input_mean = f32[2,8] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2,8] multiply(input_mean, input_mean)
        variance = f32[2,8] subtract(input_square_mean, input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,8] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,8] add(variance, epsilon_bcast)
        norm_factor = f32[2,8] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4,1,8] broadcast(norm_factor), dimensions={0,3}
        input_mean_bcast = f32[2,4,1,8] broadcast(input_mean), dimensions={0,3}
        input_center = f32[2,4,1,8] subtract(input, input_mean_bcast)
        norm = f32[2,4,1,8] multiply(norm_factor_bcast, input_center)
        scale = f32[4,1] parameter(1)
        scale_bcast = f32[2,4,1,8] broadcast(scale), dimensions={1,2}
        norm_scale = f32[2,4,1,8] multiply(norm, scale_bcast)
        bias = f32[4,1] parameter(2)
        bias_broadcast = f32[2,4,1,8] broadcast(bias), dimensions={1,2}
        norm_scale_bias = f32[2,4,1,8] add(norm_scale, bias_broadcast)
        norm_factor_cube = f32[2,8] divide(norm_factor, variance_plus_epsilon)
        ROOT out = (f32[2,4,1,8], f32[2,8], f32[2,8], f32[2,8]) tuple(norm_scale_bias, input_mean, norm_factor, norm_factor_cube)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4,1,8], {{.*}}: f32[4,1], {{.*}}: f32[4,1]) -> (f32[2,4,1,8], f32[2,8], f32[2,8], f32[2,8]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,1,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE:%[^ ]+]] = f32[1,2,8,4]{3,2,1,0} transpose([[P0]]), dimensions={2,0,3,1}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[16,4,1,1]{3,2,1,0} bitcast([[TRANSPOSE]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,1]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4,1]{1,0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC:%[^ ]+]] = (f32[16,4,1,1]{3,2,1,0}, f32[16,1,1,1]{3,2,1,0}, f32[16,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK:           }
; CHECK-NEXT:    [[GTE0:%[^ ]+]] = f32[16,4,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=0
; CHECK-NEXT:    [[FUSION0:%[^ ]+]] = f32[2,4,1,8]{3,2,1,0} fusion([[GTE0]]), kind=kLoop, calls=[[FUSED_COMPUTATION0:%[^ ]+]]
; CHECK-NEXT:    [[GTE1:%[^ ]+]] = f32[16,1,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=1
; CHECK-NEXT:    [[GTE1_BITCAST:%[^ ]+]] = f32[2,8]{1,0} bitcast([[GTE1]])
; CHECK-NEXT:    [[GTE2:%[^ ]+]] = f32[16,1,1,1]{3,2,1,0} get-tuple-element([[CC]]), index=2
; CHECK-NEXT:    [[GTE2_BITCAST:%[^ ]+]] = f32[2,8]{1,0} bitcast([[GTE2]])
; CHECK-NEXT:    [[FUSION1:%[^ ]+]] = f32[2,8]{1,0} fusion([[GTE2]]), kind=kLoop, calls=[[FUSED_COMPUTATION1:%[^ ]+]]
; CHECK-NEXT:  ROOT [[OUT:%[^ ]+]] = (f32[2,4,1,8]{3,2,1,0}, f32[2,8]{1,0}, f32[2,8]{1,0}, f32[2,8]{1,0}) tuple([[FUSION0]], [[GTE1_BITCAST]], [[GTE2_BITCAST]], [[FUSION1]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrainBackward2D1) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4] parameter(0)
        input_square = f32[2,4] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2] reduce(input_square, c0), dimensions={1}, to_apply=apply
        reduce = f32[2] reduce(input, c0), dimensions={1}, to_apply=apply
        r_nelems = f32[] constant(0.25)
        r_nelems_bcast = f32[2] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2] multiply(input_square_sum,r_nelems_bcast)
        input_mean = f32[2] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2] multiply(input_mean,input_mean)
        variance = f32[2] subtract(input_square_mean,input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2] add(variance, epsilon_bcast)
        norm_factor = f32[2] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4] broadcast(norm_factor), dimensions={0}
        input_mean_bcast = f32[2,4] broadcast(input_mean), dimensions={0}
        input_center = f32[2,4] subtract(input, input_mean_bcast)
        norm = f32[2,4] multiply(input_center, norm_factor_bcast)
        scale = f32[4] parameter(1)
        scale_bcast = f32[2,4] broadcast(scale), dimensions={1}
        norm_scale = f32[2,4] multiply(norm, scale_bcast)
        bias = f32[4] parameter(2)
        bias_bcast = f32[2,4] broadcast(bias), dimensions={1}
        norm_scale_bias = f32[2,4] add(norm_scale, bias_bcast)
        doutput = f32[2,4] parameter(3)
        dbias = f32[4] reduce(doutput, c0), dimensions={0}, to_apply=apply
        norm_doutput = f32[2,4] multiply(norm, doutput)
        dscale = f32[4] reduce(norm_doutput, c0), dimensions={0}, to_apply=apply
        scale_doutput = f32[2,4] multiply(scale_bcast, doutput)
        input_center_scale_doutput = f32[2,4] multiply(input_center, scale_doutput)
        f0 = f32[2] reduce(input_center_scale_doutput, c0), dimensions={1}, to_apply=apply
        norm_factor_cube = f32[2] divide(norm_factor, variance_plus_epsilon)
        c1 = f32[] constant(-0.5)
        c1_bcast = f32[2] broadcast(c1), dimensions={}
        dnorm_factor = f32[2] multiply(norm_factor_cube, c1_bcast)
        f0_dnorm_factor = f32[2] multiply(f0, dnorm_factor)
        c2 = f32[] constant(0.5)
        c2_bcast = f32[2] broadcast(c2), dimensions={}
        f0_dnorm_factor_scaled = f32[2] multiply(f0_dnorm_factor, c2_bcast)
        f0_dnorm_factor_scaled_bcast = f32[2,4] broadcast(f0_dnorm_factor_scaled), dimensions={0}
        f1 = f32[2,4] multiply(input_center, f0_dnorm_factor_scaled_bcast)
        minus_f1 = f32[2,4] negate(f1)
        minus_f1_sum = f32[2] reduce(minus_f1, c0), dimensions={1}, to_apply=apply
        f2 = f32[2,4] multiply(norm_factor_bcast, scale_doutput)
        minus_f2 = f32[2,4] negate(f2)
        minus_f2_sum = f32[2] reduce(minus_f2, c0), dimensions={1}, to_apply=apply
        minus_f1_f2_sum = f32[2] add(minus_f1_sum, minus_f2_sum)
        minus_f1_f2_sum_scaled = f32[2] multiply(minus_f1_f2_sum, r_nelems_bcast)
        minus_f1_f2_sum_scaled_bcast = f32[2,4] broadcast(minus_f1_f2_sum_scaled), dimensions={0}
        f1_f2 = f32[2,4] add(f1, f2)
        dinput = f32[2,4] add(f1_f2, minus_f1_f2_sum_scaled_bcast)
        ROOT out = (f32[2,4], f32[2,4], f32[4], f32[4]) tuple(norm_scale_bias, dinput, dscale, dbias)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4], {{.*}}: f32[4], {{.*}}: f32[4], {{.*}}: f32[2,4]) -> (f32[2,4], f32[2,4], f32[4], f32[4]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4]{1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[2,4,1,1]{3,2,1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC0:%[^ ]+]] = (f32[2,4,1,1]{3,2,1,0}, f32[2,1,1,1]{3,2,1,0}, f32[2,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK-DAG:         "kind":"LAYER_FWD_TRAIN"
; CHECK:           }
; CHECK-DAG:     [[GTE0:%[^ ]+]] = f32[2,4,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=0
; CHECK-DAG:     [[GTE0_BITCAST:%[^ ]+]] = f32[2,4]{1,0} bitcast([[GTE0]])
; CHECK-DAG:     [[P3:%[^ ]+]] = f32[2,4]{1,0} parameter(3)
; CHECK-DAG:     [[P3_BITCAST:%[^ ]+]] = f32[2,4,1,1]{3,2,1,0} bitcast([[P3]])
; CHECK-DAG:     [[GTE1:%[^ ]+]] = f32[2,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = f32[2,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=2
; CHECK-NEXT:    [[CC1:%[^ ]+]] = (f32[2,4,1,1]{3,2,1,0}, f32[4,1,1,1]{3,2,1,0}, f32[4,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P3_BITCAST]], [[GTE1]], [[GTE2]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0
; CHECK-DAG:         "kind":"LAYER_BWD"
; CHECK:           }
; CHECK-DAG:     [[GTE3:%[^ ]+]] = f32[2,4,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=0
; CHECK-DAG:     [[GTE3_BITCAST:%[^ ]+]] = f32[2,4]{1,0} bitcast([[GTE3]])
; CHECK-DAG:     [[GTE4:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=1
; CHECK-DAG:     [[GTE4_BITCAST:%[^ ]+]] = f32[4]{0} bitcast([[GTE4]])
; CHECK-DAG:     [[GTE5:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=2
; CHECK-DAG:     [[GTE5_BITCAST:%[^ ]+]] = f32[4]{0} bitcast([[GTE5]])
; CHECK-DAG:  ROOT [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, f32[2,4]{1,0}, f32[4]{0}, f32[4]{0}) tuple([[GTE0_BITCAST]], [[GTE3_BITCAST]], [[GTE4_BITCAST]], [[GTE5_BITCAST]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrainBackward4D3) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        input_square = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,4,6] reduce(input_square, c0), dimensions={3}, to_apply=apply
        reduce = f32[2,4,6] reduce(input, c0), dimensions={3}, to_apply=apply
        r_nelems = f32[] constant(0.125)
        r_nelems_bcast = f32[2,4,6] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,4,6] multiply(input_square_sum,r_nelems_bcast)
        input_mean = f32[2,4,6] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2,4,6] multiply(input_mean,input_mean)
        variance = f32[2,4,6] subtract(input_square_mean,input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,4,6] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,4,6] add(variance, epsilon_bcast)
        norm_factor = f32[2,4,6] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4,6,8] broadcast(norm_factor), dimensions={0,1,2}
        input_mean_bcast = f32[2,4,6,8] broadcast(input_mean), dimensions={0,1,2}
        input_center = f32[2,4,6,8] subtract(input, input_mean_bcast)
        norm = f32[2,4,6,8] multiply(input_center, norm_factor_bcast)
        scale = f32[8] parameter(1)
        scale_bcast = f32[2,4,6,8] broadcast(scale), dimensions={3}
        norm_scale = f32[2,4,6,8] multiply(norm, scale_bcast)
        bias = f32[8] parameter(2)
        bias_bcast = f32[2,4,6,8] broadcast(bias), dimensions={3}
        norm_scale_bias = f32[2,4,6,8] add(norm_scale, bias_bcast)
        doutput = f32[2,4,6,8] parameter(3)
        dbias = f32[8] reduce(doutput, c0), dimensions={0,1,2}, to_apply=apply
        norm_doutput = f32[2,4,6,8] multiply(norm, doutput)
        dscale = f32[8] reduce(norm_doutput, c0), dimensions={0,1,2}, to_apply=apply
        scale_doutput = f32[2,4,6,8] multiply(scale_bcast, doutput)
        input_center_scale_doutput = f32[2,4,6,8] multiply(input_center, scale_doutput)
        f0 = f32[2,4,6] reduce(input_center_scale_doutput, c0), dimensions={3}, to_apply=apply
        norm_factor_cube = f32[2,4,6] divide(norm_factor, variance_plus_epsilon)
        c1 = f32[] constant(-0.5)
        c1_bcast = f32[2,4,6] broadcast(c1), dimensions={}
        dnorm_factor = f32[2,4,6] multiply(norm_factor_cube, c1_bcast)
        f0_dnorm_factor = f32[2,4,6] multiply(f0, dnorm_factor)
        c2 = f32[] constant(0.25)
        c2_bcast = f32[2,4,6] broadcast(c2), dimensions={}
        f0_dnorm_factor_scaled = f32[2,4,6] multiply(f0_dnorm_factor, c2_bcast)
        f0_dnorm_factor_scaled_bcast = f32[2,4,6,8] broadcast(f0_dnorm_factor_scaled), dimensions={0,1,2}
        f1 = f32[2,4,6,8] multiply(input_center, f0_dnorm_factor_scaled_bcast)
        minus_f1 = f32[2,4,6,8] negate(f1)
        minus_f1_sum = f32[2,4,6] reduce(minus_f1, c0), dimensions={3}, to_apply=apply
        f2 = f32[2,4,6,8] multiply(norm_factor_bcast, scale_doutput)
        minus_f2 = f32[2,4,6,8] negate(f2)
        minus_f2_sum = f32[2,4,6] reduce(minus_f2, c0), dimensions={3}, to_apply=apply
        minus_f1_f2_sum = f32[2,4,6] add(minus_f1_sum, minus_f2_sum)
        minus_f1_f2_sum_scaled = f32[2,4,6] multiply(minus_f1_f2_sum, r_nelems_bcast)
        minus_f1_f2_sum_scaled_bcast = f32[2,4,6,8] broadcast(minus_f1_f2_sum_scaled), dimensions={0,1,2}
        f1_f2 = f32[2,4,6,8] add(f1, f2)
        dinput = f32[2,4,6,8] add(f1_f2, minus_f1_f2_sum_scaled_bcast)
        ROOT out = (f32[2,4,6,8], f32[2,4,6,8], f32[8], f32[8]) tuple(norm_scale_bias, dinput, dscale, dbias)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4,6,8], {{.*}}: f32[8], {{.*}}: f32[8], {{.*}}: f32[2,4,6,8]) -> (f32[2,4,6,8], f32[2,4,6,8], f32[8], f32[8]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[48,8,1,1]{3,2,1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[8]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[8]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC0:%[^ ]+]] = (f32[48,8,1,1]{3,2,1,0}, f32[48,1,1,1]{3,2,1,0}, f32[48,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK-DAG:         "kind":"LAYER_FWD_TRAIN"
; CHECK:           }
; CHECK-DAG:     [[GTE0:%[^ ]+]] = f32[48,8,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=0
; CHECK-DAG:     [[GTE0_BITCAST:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} bitcast([[GTE0]])
; CHECK-DAG:     [[P3:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(3)
; CHECK-DAG:     [[P3_BITCAST:%[^ ]+]] = f32[48,8,1,1]{3,2,1,0} bitcast([[P3]])
; CHECK-DAG:     [[GTE1:%[^ ]+]] = f32[48,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = f32[48,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=2
; CHECK-NEXT:    [[CC1:%[^ ]+]] = (f32[48,8,1,1]{3,2,1,0}, f32[8,1,1,1]{3,2,1,0}, f32[8,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P3_BITCAST]], [[GTE1]], [[GTE2]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0
; CHECK-DAG:         "kind":"LAYER_BWD"
; CHECK:           }
; CHECK-DAG:     [[GTE3:%[^ ]+]] = f32[48,8,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=0
; CHECK-DAG:     [[GTE3_BITCAST:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} bitcast([[GTE3]])
; CHECK-DAG:     [[GTE4:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=1
; CHECK-DAG:     [[GTE4_BITCAST:%[^ ]+]] = f32[8]{0} bitcast([[GTE4]])
; CHECK-DAG:     [[GTE5:%[^ ]+]] = f32[8,1,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=2
; CHECK-DAG:     [[GTE5_BITCAST:%[^ ]+]] = f32[8]{0} bitcast([[GTE5]])
; CHECK-DAG:  ROOT [[OUT:%[^ ]+]] = (f32[2,4,6,8]{3,2,1,0}, f32[2,4,6,8]{3,2,1,0}, f32[8]{0}, f32[8]{0}) tuple([[GTE0_BITCAST]], [[GTE3_BITCAST]], [[GTE4_BITCAST]], [[GTE5_BITCAST]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrainBackward4D2) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        input_square = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,4,8] reduce(input_square, c0), dimensions={2}, to_apply=apply
        reduce = f32[2,4,8] reduce(input, c0), dimensions={2}, to_apply=apply
        r_nelems = f32[] constant(0.166667)
        r_nelems_bcast = f32[2,4,8] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,4,8] multiply(input_square_sum,r_nelems_bcast)
        input_mean = f32[2,4,8] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2,4,8] multiply(input_mean,input_mean)
        variance = f32[2,4,8] subtract(input_square_mean,input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,4,8] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,4,8] add(variance, epsilon_bcast)
        norm_factor = f32[2,4,8] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4,6,8] broadcast(norm_factor), dimensions={0,1,3}
        input_mean_bcast = f32[2,4,6,8] broadcast(input_mean), dimensions={0,1,3}
        input_center = f32[2,4,6,8] subtract(input, input_mean_bcast)
        norm = f32[2,4,6,8] multiply(input_center, norm_factor_bcast)
        scale = f32[6] parameter(1)
        scale_bcast = f32[2,4,6,8] broadcast(scale), dimensions={2}
        norm_scale = f32[2,4,6,8] multiply(norm, scale_bcast)
        bias = f32[6] parameter(2)
        bias_bcast = f32[2,4,6,8] broadcast(bias), dimensions={2}
        norm_scale_bias = f32[2,4,6,8] add(norm_scale, bias_bcast)
        doutput = f32[2,4,6,8] parameter(3)
        dbias = f32[6] reduce(doutput, c0), dimensions={0,1,3}, to_apply=apply
        norm_doutput = f32[2,4,6,8] multiply(norm, doutput)
        dscale = f32[6] reduce(norm_doutput, c0), dimensions={0,1,3}, to_apply=apply
        scale_doutput = f32[2,4,6,8] multiply(scale_bcast, doutput)
        input_center_scale_doutput = f32[2,4,6,8] multiply(input_center, scale_doutput)
        f0 = f32[2,4,8] reduce(input_center_scale_doutput, c0), dimensions={2}, to_apply=apply
        norm_factor_cube = f32[2,4,8] divide(norm_factor, variance_plus_epsilon)
        c1 = f32[] constant(-0.5)
        c1_bcast = f32[2,4,8] broadcast(c1), dimensions={}
        dnorm_factor = f32[2,4,8] multiply(norm_factor_cube, c1_bcast)
        f0_dnorm_factor = f32[2,4,8] multiply(f0, dnorm_factor)
        c2 = f32[] constant(0.333333)
        c2_bcast = f32[2,4,8] broadcast(c2), dimensions={}
        f0_dnorm_factor_scaled = f32[2,4,8] multiply(f0_dnorm_factor, c2_bcast)
        f0_dnorm_factor_scaled_bcast = f32[2,4,6,8] broadcast(f0_dnorm_factor_scaled), dimensions={0,1,3}
        f1 = f32[2,4,6,8] multiply(input_center, f0_dnorm_factor_scaled_bcast)
        minus_f1 = f32[2,4,6,8] negate(f1)
        minus_f1_sum = f32[2,4,8] reduce(minus_f1, c0), dimensions={2}, to_apply=apply
        f2 = f32[2,4,6,8] multiply(norm_factor_bcast, scale_doutput)
        minus_f2 = f32[2,4,6,8] negate(f2)
        minus_f2_sum = f32[2,4,8] reduce(minus_f2, c0), dimensions={2}, to_apply=apply
        minus_f1_f2_sum = f32[2,4,8] add(minus_f1_sum, minus_f2_sum)
        minus_f1_f2_sum_scaled = f32[2,4,8] multiply(minus_f1_f2_sum, r_nelems_bcast)
        minus_f1_f2_sum_scaled_bcast = f32[2,4,6,8] broadcast(minus_f1_f2_sum_scaled), dimensions={0,1,3}
        f1_f2 = f32[2,4,6,8] add(f1, f2)
        dinput = f32[2,4,6,8] add(f1_f2, minus_f1_f2_sum_scaled_bcast)
        ROOT out = (f32[2,4,6,8], f32[2,4,6,8], f32[6], f32[6]) tuple(norm_scale_bias, dinput, dscale, dbias)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4,6,8], {{.*}}: f32[6], {{.*}}: f32[6], {{.*}}: f32[2,4,6,8]) -> (f32[2,4,6,8], f32[2,4,6,8], f32[6], f32[6]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE0:%[^ ]+]] = f32[2,4,8,6]{3,2,1,0} transpose([[P0]]), dimensions={0,1,3,2}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[64,6,1,1]{3,2,1,0} bitcast([[TRANSPOSE0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[6]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[6,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[6]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[6,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC0:%[^ ]+]] = (f32[64,6,1,1]{3,2,1,0}, f32[64,1,1,1]{3,2,1,0}, f32[64,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK-DAG:         "kind":"LAYER_FWD_TRAIN"
; CHECK:           }
; CHECK-DAG:     [[GTE0:%[^ ]+]] = f32[64,6,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=0
; CHECK-DAG:     [[P3:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(3)
; CHECK-NEXT:    [[TRANSPOSE1:%[^ ]+]] = f32[2,4,8,6]{3,2,1,0} transpose([[P3]]), dimensions={0,1,3,2}
; CHECK-DAG:     [[P3_BITCAST:%[^ ]+]] = f32[64,6,1,1]{3,2,1,0} bitcast([[TRANSPOSE1]])
; CHECK-DAG:     [[GTE1:%[^ ]+]] = f32[64,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = f32[64,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=2
; CHECK-NEXT:    [[CC1:%[^ ]+]] = (f32[64,6,1,1]{3,2,1,0}, f32[6,1,1,1]{3,2,1,0}, f32[6,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P3_BITCAST]], [[GTE1]], [[GTE2]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0
; CHECK-DAG:         "kind":"LAYER_BWD"
; CHECK:           }
; CHECK-DAG:     [[GTE3:%[^ ]+]] = f32[64,6,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=0
; CHECK-DAG:     [[FUSION:%[^ ]+]] = (f32[2,4,6,8]{3,2,1,0}, f32[2,4,6,8]{3,2,1,0}) fusion([[GTE0]], [[GTE3]]), kind=kLoop, calls=[[FUSED_COMPUTATION:%[^ ]+]]
; CHECK-DAG:     [[GTEF0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} get-tuple-element([[FUSION]]), index=0
; CHECK-DAG:     [[GTEF1:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} get-tuple-element([[FUSION]]), index=1
; CHECK-DAG:     [[GTE4:%[^ ]+]] = f32[6,1,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=1
; CHECK-DAG:     [[GTE4_BITCAST:%[^ ]+]] = f32[6]{0} bitcast([[GTE4]])
; CHECK-DAG:     [[GTE5:%[^ ]+]] = f32[6,1,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=2
; CHECK-DAG:     [[GTE5_BITCAST:%[^ ]+]] = f32[6]{0} bitcast([[GTE5]])
; CHECK-DAG:  ROOT [[OUT:%[^ ]+]] = (f32[2,4,6,8]{3,2,1,0}, f32[2,4,6,8]{3,2,1,0}, f32[6]{0}, f32[6]{0}) tuple([[GTEF0]], [[GTEF1]], [[GTE4_BITCAST]], [[GTE5_BITCAST]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrainBackward4D12) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        input_square = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,8] reduce(input_square, c0), dimensions={1,2}, to_apply=apply
        reduce = f32[2,8] reduce(input, c0), dimensions={1,2}, to_apply=apply
        r_nelems = f32[] constant(0.041667)
        r_nelems_bcast = f32[2,8] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,8] multiply(input_square_sum,r_nelems_bcast)
        input_mean = f32[2,8] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2,8] multiply(input_mean,input_mean)
        variance = f32[2,8] subtract(input_square_mean,input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,8] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,8] add(variance, epsilon_bcast)
        norm_factor = f32[2,8] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4,6,8] broadcast(norm_factor), dimensions={0,3}
        input_mean_bcast = f32[2,4,6,8] broadcast(input_mean), dimensions={0,3}
        input_center = f32[2,4,6,8] subtract(input, input_mean_bcast)
        norm = f32[2,4,6,8] multiply(input_center, norm_factor_bcast)
        scale = f32[4,6] parameter(1)
        scale_bcast = f32[2,4,6,8] broadcast(scale), dimensions={1,2}
        norm_scale = f32[2,4,6,8] multiply(norm, scale_bcast)
        bias = f32[4,6] parameter(2)
        bias_bcast = f32[2,4,6,8] broadcast(bias), dimensions={1,2}
        norm_scale_bias = f32[2,4,6,8] add(norm_scale, bias_bcast)
        doutput = f32[2,4,6,8] parameter(3)
        dbias = f32[4,6] reduce(doutput, c0), dimensions={0,3}, to_apply=apply
        norm_doutput = f32[2,4,6,8] multiply(norm, doutput)
        dscale = f32[4,6] reduce(norm_doutput, c0), dimensions={0,3}, to_apply=apply
        scale_doutput = f32[2,4,6,8] multiply(scale_bcast, doutput)
        input_center_scale_doutput = f32[2,4,6,8] multiply(input_center, scale_doutput)
        f0 = f32[2,8] reduce(input_center_scale_doutput, c0), dimensions={1,2}, to_apply=apply
        norm_factor_cube = f32[2,8] divide(norm_factor, variance_plus_epsilon)
        c1 = f32[] constant(-0.5)
        c1_bcast = f32[2,8] broadcast(c1), dimensions={}
        dnorm_factor = f32[2,8] multiply(norm_factor_cube, c1_bcast)
        f0_dnorm_factor = f32[2,8] multiply(f0, dnorm_factor)
        c2 = f32[] constant(0.083333)
        c2_bcast = f32[2,8] broadcast(c2), dimensions={}
        f0_dnorm_factor_scaled = f32[2,8] multiply(f0_dnorm_factor, c2_bcast)
        f0_dnorm_factor_scaled_bcast = f32[2,4,6,8] broadcast(f0_dnorm_factor_scaled), dimensions={0,3}
        f1 = f32[2,4,6,8] multiply(input_center, f0_dnorm_factor_scaled_bcast)
        minus_f1 = f32[2,4,6,8] negate(f1)
        minus_f1_sum = f32[2,8] reduce(minus_f1, c0), dimensions={1,2}, to_apply=apply
        f2 = f32[2,4,6,8] multiply(norm_factor_bcast, scale_doutput)
        minus_f2 = f32[2,4,6,8] negate(f2)
        minus_f2_sum = f32[2,8] reduce(minus_f2, c0), dimensions={1,2}, to_apply=apply
        minus_f1_f2_sum = f32[2,8] add(minus_f1_sum, minus_f2_sum)
        minus_f1_f2_sum_scaled = f32[2,8] multiply(minus_f1_f2_sum, r_nelems_bcast)
        minus_f1_f2_sum_scaled_bcast = f32[2,4,6,8] broadcast(minus_f1_f2_sum_scaled), dimensions={0,3}
        f1_f2 = f32[2,4,6,8] add(f1, f2)
        dinput = f32[2,4,6,8] add(f1_f2, minus_f1_f2_sum_scaled_bcast)
        ROOT out = (f32[2,4,6,8], f32[2,4,6,8], f32[4,6], f32[4,6]) tuple(norm_scale_bias, dinput, dscale, dbias)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4,6,8], {{.*}}: f32[4,6], {{.*}}: f32[4,6], {{.*}}: f32[2,4,6,8]) -> (f32[2,4,6,8], f32[2,4,6,8], f32[4,6], f32[4,6]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE0:%[^ ]+]] = f32[2,8,4,6]{3,2,1,0} transpose([[P0]]), dimensions={0,3,1,2}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[16,4,6,1]{3,2,1,0} bitcast([[TRANSPOSE0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,6]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,6,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4,6]{1,0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,6,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC0:%[^ ]+]] = (f32[16,4,6,1]{3,2,1,0}, f32[16,1,1,1]{3,2,1,0}, f32[16,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK-DAG:         "kind":"LAYER_FWD_TRAIN"
; CHECK:           }
; CHECK-DAG:     [[GTE0:%[^ ]+]] = f32[16,4,6,1]{3,2,1,0} get-tuple-element([[CC0]]), index=0
; CHECK-DAG:     [[P3:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(3)
; CHECK-NEXT:    [[TRANSPOSE1:%[^ ]+]] = f32[2,8,4,6]{3,2,1,0} transpose([[P3]]), dimensions={0,3,1,2}
; CHECK-DAG:     [[P3_BITCAST:%[^ ]+]] = f32[16,4,6,1]{3,2,1,0} bitcast([[TRANSPOSE1]])
; CHECK-DAG:     [[GTE1:%[^ ]+]] = f32[16,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = f32[16,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=2
; CHECK-NEXT:    [[CC1:%[^ ]+]] = (f32[16,4,6,1]{3,2,1,0}, f32[4,6,1,1]{3,2,1,0}, f32[4,6,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P3_BITCAST]], [[GTE1]], [[GTE2]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0
; CHECK-DAG:         "kind":"LAYER_BWD"
; CHECK:           }
; CHECK-DAG:     [[GTE3:%[^ ]+]] = f32[16,4,6,1]{3,2,1,0} get-tuple-element([[CC1]]), index=0
; CHECK-DAG:     [[FUSION:%[^ ]+]] = (f32[2,4,6,8]{3,2,1,0}, f32[2,4,6,8]{3,2,1,0}) fusion([[GTE0]], [[GTE3]]), kind=kLoop, calls=[[FUSED_COMPUTATION:%[^ ]+]]
; CHECK-DAG:     [[GTEF0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} get-tuple-element([[FUSION]]), index=0
; CHECK-DAG:     [[GTEF1:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} get-tuple-element([[FUSION]]), index=1
; CHECK-DAG:     [[GTE4:%[^ ]+]] = f32[4,6,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=1
; CHECK-DAG:     [[GTE4_BITCAST:%[^ ]+]] = f32[4,6]{1,0} bitcast([[GTE4]])
; CHECK-DAG:     [[GTE5:%[^ ]+]] = f32[4,6,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=2
; CHECK-DAG:     [[GTE5_BITCAST:%[^ ]+]] = f32[4,6]{1,0} bitcast([[GTE5]])
; CHECK-DAG:  ROOT [[OUT:%[^ ]+]] = (f32[2,4,6,8]{3,2,1,0}, f32[2,4,6,8]{3,2,1,0}, f32[4,6]{1,0}, f32[4,6]{1,0}) tuple([[GTEF0]], [[GTEF1]], [[GTE4_BITCAST]], [[GTE5_BITCAST]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrainBackward4D12Degenerate2) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,1,8] parameter(0)
        input_square = f32[2,4,1,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,8] reduce(input_square, c0), dimensions={1,2}, to_apply=apply
        reduce = f32[2,8] reduce(input, c0), dimensions={1,2}, to_apply=apply
        r_nelems = f32[] constant(0.25)
        r_nelems_bcast = f32[2,8] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,8] multiply(input_square_sum,r_nelems_bcast)
        input_mean = f32[2,8] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2,8] multiply(input_mean,input_mean)
        variance = f32[2,8] subtract(input_square_mean,input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,8] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,8] add(variance, epsilon_bcast)
        norm_factor = f32[2,8] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4,1,8] broadcast(norm_factor), dimensions={0,3}
        input_mean_bcast = f32[2,4,1,8] broadcast(input_mean), dimensions={0,3}
        input_center = f32[2,4,1,8] subtract(input, input_mean_bcast)
        norm = f32[2,4,1,8] multiply(input_center, norm_factor_bcast)
        scale = f32[4,1] parameter(1)
        scale_bcast = f32[2,4,1,8] broadcast(scale), dimensions={1,2}
        norm_scale = f32[2,4,1,8] multiply(norm, scale_bcast)
        bias = f32[4,1] parameter(2)
        bias_bcast = f32[2,4,1,8] broadcast(bias), dimensions={1,2}
        norm_scale_bias = f32[2,4,1,8] add(norm_scale, bias_bcast)
        doutput = f32[2,4,1,8] parameter(3)
        dbias = f32[4,1] reduce(doutput, c0), dimensions={0,3}, to_apply=apply
        norm_doutput = f32[2,4,1,8] multiply(norm, doutput)
        dscale = f32[4,1] reduce(norm_doutput, c0), dimensions={0,3}, to_apply=apply
        scale_doutput = f32[2,4,1,8] multiply(scale_bcast, doutput)
        input_center_scale_doutput = f32[2,4,1,8] multiply(input_center, scale_doutput)
        f0 = f32[2,8] reduce(input_center_scale_doutput, c0), dimensions={1,2}, to_apply=apply
        norm_factor_cube = f32[2,8] divide(norm_factor, variance_plus_epsilon)
        c1 = f32[] constant(-0.5)
        c1_bcast = f32[2,8] broadcast(c1), dimensions={}
        dnorm_factor = f32[2,8] multiply(norm_factor_cube, c1_bcast)
        f0_dnorm_factor = f32[2,8] multiply(f0, dnorm_factor)
        c2 = f32[] constant(0.5)
        c2_bcast = f32[2,8] broadcast(c2), dimensions={}
        f0_dnorm_factor_scaled = f32[2,8] multiply(f0_dnorm_factor, c2_bcast)
        f0_dnorm_factor_scaled_bcast = f32[2,4,1,8] broadcast(f0_dnorm_factor_scaled), dimensions={0,3}
        f1 = f32[2,4,1,8] multiply(input_center, f0_dnorm_factor_scaled_bcast)
        minus_f1 = f32[2,4,1,8] negate(f1)
        minus_f1_sum = f32[2,8] reduce(minus_f1, c0), dimensions={1,2}, to_apply=apply
        f2 = f32[2,4,1,8] multiply(norm_factor_bcast, scale_doutput)
        minus_f2 = f32[2,4,1,8] negate(f2)
        minus_f2_sum = f32[2,8] reduce(minus_f2, c0), dimensions={1,2}, to_apply=apply
        minus_f1_f2_sum = f32[2,8] add(minus_f1_sum, minus_f2_sum)
        minus_f1_f2_sum_scaled = f32[2,8] multiply(minus_f1_f2_sum, r_nelems_bcast)
        minus_f1_f2_sum_scaled_bcast = f32[2,4,1,8] broadcast(minus_f1_f2_sum_scaled), dimensions={0,3}
        f1_f2 = f32[2,4,1,8] add(f1, f2)
        dinput = f32[2,4,1,8] add(f1_f2, minus_f1_f2_sum_scaled_bcast)
        ROOT out = (f32[2,4,1,8], f32[2,4,1,8], f32[4,1], f32[4,1]) tuple(norm_scale_bias, dinput, dscale, dbias)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4,1,8], {{.*}}: f32[4,1], {{.*}}: f32[4,1], {{.*}}: f32[2,4,1,8]) -> (f32[2,4,1,8], f32[2,4,1,8], f32[4,1], f32[4,1]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,1,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE0:%[^ ]+]] = f32[1,2,8,4]{3,2,1,0} transpose([[P0]]), dimensions={2,0,3,1}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[16,4,1,1]{3,2,1,0} bitcast([[TRANSPOSE0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,1]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4,1]{1,0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC0:%[^ ]+]] = (f32[16,4,1,1]{3,2,1,0}, f32[16,1,1,1]{3,2,1,0}, f32[16,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK-DAG:         "kind":"LAYER_FWD_TRAIN"
; CHECK:           }
; CHECK-DAG:     [[GTE0:%[^ ]+]] = f32[16,4,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=0
; CHECK-DAG:     [[P3:%[^ ]+]] = f32[2,4,1,8]{3,2,1,0} parameter(3)
; CHECK-NEXT:    [[TRANSPOSE1:%[^ ]+]] = f32[1,2,8,4]{3,2,1,0} transpose([[P3]]), dimensions={2,0,3,1}
; CHECK-DAG:     [[P3_BITCAST:%[^ ]+]] = f32[16,4,1,1]{3,2,1,0} bitcast([[TRANSPOSE1]])
; CHECK-DAG:     [[GTE1:%[^ ]+]] = f32[16,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = f32[16,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=2
; CHECK-NEXT:    [[CC1:%[^ ]+]] = (f32[16,4,1,1]{3,2,1,0}, f32[4,1,1,1]{3,2,1,0}, f32[4,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P3_BITCAST]], [[GTE1]], [[GTE2]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0
; CHECK-DAG:         "kind":"LAYER_BWD"
; CHECK:           }
; CHECK-DAG:     [[GTE3:%[^ ]+]] = f32[16,4,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=0
; CHECK-DAG:     [[FUSION0:%[^ ]+]] = (f32[2,4,1,8]{3,2,1,0}, f32[2,4,1,8]{3,2,1,0}) fusion([[GTE0]], [[GTE3]]), kind=kLoop, calls=[[FUSED_COMPUTATION0:%[^ ]+]]
; CHECK-DAG:     [[GTEF0:%[^ ]+]] = f32[2,4,1,8]{3,2,1,0} get-tuple-element([[FUSION0]]), index=0
; CHECK-DAG:     [[GTEF1:%[^ ]+]] = f32[2,4,1,8]{3,2,1,0} get-tuple-element([[FUSION0]]), index=1
; CHECK-DAG:     [[GTE4:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=1
; CHECK-DAG:     [[GTE4_BITCAST:%[^ ]+]] = f32[4,1]{1,0} bitcast([[GTE4]])
; CHECK-DAG:     [[GTE5:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=2
; CHECK-DAG:     [[GTE5_BITCAST:%[^ ]+]] = f32[4,1]{1,0} bitcast([[GTE5]])
; CHECK-DAG:  ROOT [[OUT:%[^ ]+]] = (f32[2,4,1,8]{3,2,1,0}, f32[2,4,1,8]{3,2,1,0}, f32[4,1]{1,0}, f32[4,1]{1,0}) tuple([[GTEF0]], [[GTEF1]], [[GTE4_BITCAST]], [[GTE5_BITCAST]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrainBackward4D1DoutputReshapeSplit) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        input_square = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,6,8] reduce(input_square, c0), dimensions={1}, to_apply=apply
        reduce = f32[2,6,8] reduce(input, c0), dimensions={1}, to_apply=apply
        r_nelems = f32[] constant(0.25)
        r_nelems_bcast = f32[2,6,8] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,6,8] multiply(input_square_sum,r_nelems_bcast)
        input_mean = f32[2,6,8] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2,6,8] multiply(input_mean,input_mean)
        variance = f32[2,6,8] subtract(input_square_mean,input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,6,8] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,6,8] add(variance, epsilon_bcast)
        norm_factor = f32[2,6,8] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4,6,8] broadcast(norm_factor), dimensions={0,2,3}
        input_mean_bcast = f32[2,4,6,8] broadcast(input_mean), dimensions={0,2,3}
        input_center = f32[2,4,6,8] subtract(input, input_mean_bcast)
        norm = f32[2,4,6,8] multiply(input_center, norm_factor_bcast)
        scale = f32[4] parameter(1)
        scale_bcast = f32[2,4,6,8] broadcast(scale), dimensions={1}
        norm_scale = f32[2,4,6,8] multiply(norm, scale_bcast)
        bias = f32[4] parameter(2)
        bias_bcast = f32[2,4,6,8] broadcast(bias), dimensions={1}
        norm_scale_bias = f32[2,4,6,8] add(norm_scale, bias_bcast)
        doutput = f32[2,4,48] parameter(3)
        dbias = f32[4] reduce(doutput, c0), dimensions={0,2}, to_apply=apply
        doutput_bitcast = f32[2,4,6,8] reshape(doutput) 
        norm_doutput = f32[2,4,6,8] multiply(norm, doutput_bitcast)
        dscale = f32[4] reduce(norm_doutput, c0), dimensions={0,2,3}, to_apply=apply
        scale_doutput = f32[2,4,6,8] multiply(scale_bcast, doutput_bitcast)
        input_center_scale_doutput = f32[2,4,6,8] multiply(input_center, scale_doutput)
        f0 = f32[2,6,8] reduce(input_center_scale_doutput, c0), dimensions={1}, to_apply=apply
        norm_factor_cube = f32[2,6,8] divide(norm_factor, variance_plus_epsilon)
        c1 = f32[] constant(-0.5)
        c1_bcast = f32[2,6,8] broadcast(c1), dimensions={}
        dnorm_factor = f32[2,6,8] multiply(norm_factor_cube, c1_bcast)
        f0_dnorm_factor = f32[2,6,8] multiply(f0, dnorm_factor)
        c2 = f32[] constant(0.5)
        c2_bcast = f32[2,6,8] broadcast(c2), dimensions={}
        f0_dnorm_factor_scaled = f32[2,6,8] multiply(f0_dnorm_factor, c2_bcast)
        f0_dnorm_factor_scaled_bcast = f32[2,4,6,8] broadcast(f0_dnorm_factor_scaled), dimensions={0,2,3}
        f1 = f32[2,4,6,8] multiply(input_center, f0_dnorm_factor_scaled_bcast)
        minus_f1 = f32[2,4,6,8] negate(f1)
        minus_f1_sum = f32[2,6,8] reduce(minus_f1, c0), dimensions={1}, to_apply=apply
        f2 = f32[2,4,6,8] multiply(norm_factor_bcast, scale_doutput)
        minus_f2 = f32[2,4,6,8] negate(f2)
        minus_f2_sum = f32[2,6,8] reduce(minus_f2, c0), dimensions={1}, to_apply=apply
        minus_f1_f2_sum = f32[2,6,8] add(minus_f1_sum, minus_f2_sum)
        minus_f1_f2_sum_scaled = f32[2,6,8] multiply(minus_f1_f2_sum, r_nelems_bcast)
        minus_f1_f2_sum_scaled_bcast = f32[2,4,6,8] broadcast(minus_f1_f2_sum_scaled), dimensions={0,2,3}
        f1_f2 = f32[2,4,6,8] add(f1, f2)
        dinput = f32[2,4,6,8] add(f1_f2, minus_f1_f2_sum_scaled_bcast)
        ROOT out = (f32[2,4,6,8], f32[2,4,6,8], f32[4], f32[4]) tuple(norm_scale_bias, dinput, dscale, dbias)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4,6,8], {{.*}}: f32[4], {{.*}}: f32[4], {{.*}}: f32[2,4,48]) -> (f32[2,4,6,8], f32[2,4,6,8], f32[4], f32[4]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE0:%[^ ]+]] = f32[2,6,8,4]{3,2,1,0} transpose([[P0]]), dimensions={0,2,3,1}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[96,4,1,1]{3,2,1,0} bitcast([[TRANSPOSE0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC0:%[^ ]+]] = (f32[96,4,1,1]{3,2,1,0}, f32[96,1,1,1]{3,2,1,0}, f32[96,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK-DAG:         "kind":"LAYER_FWD_TRAIN"
; CHECK:           }
; CHECK-DAG:     [[GTE0:%[^ ]+]] = f32[96,4,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=0
; CHECK-DAG:     [[P3:%[^ ]+]] = f32[2,4,48]{2,1,0} parameter(3)
; CHECK-DAG:     [[FUSION0:%[^ ]+]] = f32[2,6,8,4]{3,2,1,0} fusion([[P3]]), kind=kLoop, calls=[[FUSED_COMPUTATION0:%[^ ]+]]
; CHECK-DAG:     [[FUSION0_BITCAST:%[^ ]+]] = f32[96,4,1,1]{3,2,1,0} bitcast([[FUSION0]])
; CHECK-DAG:     [[GTE1:%[^ ]+]] = f32[96,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = f32[96,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=2
; CHECK-NEXT:    [[CC1:%[^ ]+]] = (f32[96,4,1,1]{3,2,1,0}, f32[4,1,1,1]{3,2,1,0}, f32[4,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[FUSION0_BITCAST]], [[GTE1]], [[GTE2]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0
; CHECK-DAG:         "kind":"LAYER_BWD"
; CHECK:           }
; CHECK-DAG:     [[GTE3:%[^ ]+]] = f32[96,4,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=0
; CHECK-DAG:     [[FUSION1:%[^ ]+]] = (f32[2,4,6,8]{3,2,1,0}, f32[2,4,6,8]{3,2,1,0}) fusion([[GTE0]], [[GTE3]]), kind=kLoop, calls=[[FUSED_COMPUTATION1:%[^ ]+]]
; CHECK-DAG:     [[GTEF1:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} get-tuple-element([[FUSION1]]), index=0
; CHECK-DAG:     [[GTEF2:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} get-tuple-element([[FUSION1]]), index=1
; CHECK-DAG:     [[GTE4:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=1
; CHECK-DAG:     [[GTE4_BITCAST:%[^ ]+]] = f32[4]{0} bitcast([[GTE4]])
; CHECK-DAG:     [[GTE5:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=2
; CHECK-DAG:     [[GTE5_BITCAST:%[^ ]+]] = f32[4]{0} bitcast([[GTE5]])
; CHECK-DAG:  ROOT [[OUT:%[^ ]+]] = (f32[2,4,6,8]{3,2,1,0}, f32[2,4,6,8]{3,2,1,0}, f32[4]{0}, f32[4]{0}) tuple([[GTEF1]], [[GTEF2]], [[GTE4_BITCAST]], [[GTE5_BITCAST]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

TEST_F(CudnnNormRewriterTest, LayerNormTrainBackward4D1DoutputReshapeCombine) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8905)
  GTEST_SKIP() << "Layer norm kernels require CUDA 12 and cuDNN 8.9.5.";
#endif
  if (!(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::AMPERE) &&
      !(GetCudaComputeCapability().major ==
        se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP()
        << "Layer norm kernels require Ampere or Hopper architectures.";
  }
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] add(a,b)
    }

    ENTRY test {
        input = f32[2,4,6,8] parameter(0)
        input_square = f32[2,4,6,8] multiply(input, input)
        c0 = f32[] constant(0)
        input_square_sum = f32[2,6,8] reduce(input_square, c0), dimensions={1}, to_apply=apply
        reduce = f32[2,6,8] reduce(input, c0), dimensions={1}, to_apply=apply
        r_nelems = f32[] constant(0.25)
        r_nelems_bcast = f32[2,6,8] broadcast(r_nelems), dimensions={}
        input_square_mean = f32[2,6,8] multiply(input_square_sum,r_nelems_bcast)
        input_mean = f32[2,6,8] multiply(reduce, r_nelems_bcast)
        input_mean_square = f32[2,6,8] multiply(input_mean,input_mean)
        variance = f32[2,6,8] subtract(input_square_mean,input_mean_square)
        epsilon = f32[] constant(0.001)
        epsilon_bcast = f32[2,6,8] broadcast(epsilon), dimensions={}
        variance_plus_epsilon = f32[2,6,8] add(variance, epsilon_bcast)
        norm_factor = f32[2,6,8] rsqrt(variance_plus_epsilon)
        norm_factor_bcast = f32[2,4,6,8] broadcast(norm_factor), dimensions={0,2,3}
        input_mean_bcast = f32[2,4,6,8] broadcast(input_mean), dimensions={0,2,3}
        input_center = f32[2,4,6,8] subtract(input, input_mean_bcast)
        norm = f32[2,4,6,8] multiply(input_center, norm_factor_bcast)
        scale = f32[4] parameter(1)
        scale_bcast = f32[2,4,6,8] broadcast(scale), dimensions={1}
        norm_scale = f32[2,4,6,8] multiply(norm, scale_bcast)
        bias = f32[4] parameter(2)
        bias_bcast = f32[2,4,6,8] broadcast(bias), dimensions={1}
        norm_scale_bias = f32[2,4,6,8] add(norm_scale, bias_bcast)
        doutput = f32[2,4,6,2,2,2] parameter(3)
        dbias = f32[4] reduce(doutput, c0), dimensions={0,2,3,4,5}, to_apply=apply
        doutput_bitcast = f32[2,4,6,8] reshape(doutput) 
        norm_doutput = f32[2,4,6,8] multiply(norm, doutput_bitcast)
        dscale = f32[4] reduce(norm_doutput, c0), dimensions={0,2,3}, to_apply=apply
        scale_doutput = f32[2,4,6,8] multiply(scale_bcast, doutput_bitcast)
        input_center_scale_doutput = f32[2,4,6,8] multiply(input_center, scale_doutput)
        f0 = f32[2,6,8] reduce(input_center_scale_doutput, c0), dimensions={1}, to_apply=apply
        norm_factor_cube = f32[2,6,8] divide(norm_factor, variance_plus_epsilon)
        c1 = f32[] constant(-0.5)
        c1_bcast = f32[2,6,8] broadcast(c1), dimensions={}
        dnorm_factor = f32[2,6,8] multiply(norm_factor_cube, c1_bcast)
        f0_dnorm_factor = f32[2,6,8] multiply(f0, dnorm_factor)
        c2 = f32[] constant(0.5)
        c2_bcast = f32[2,6,8] broadcast(c2), dimensions={}
        f0_dnorm_factor_scaled = f32[2,6,8] multiply(f0_dnorm_factor, c2_bcast)
        f0_dnorm_factor_scaled_bcast = f32[2,4,6,8] broadcast(f0_dnorm_factor_scaled), dimensions={0,2,3}
        f1 = f32[2,4,6,8] multiply(input_center, f0_dnorm_factor_scaled_bcast)
        minus_f1 = f32[2,4,6,8] negate(f1)
        minus_f1_sum = f32[2,6,8] reduce(minus_f1, c0), dimensions={1}, to_apply=apply
        f2 = f32[2,4,6,8] multiply(norm_factor_bcast, scale_doutput)
        minus_f2 = f32[2,4,6,8] negate(f2)
        minus_f2_sum = f32[2,6,8] reduce(minus_f2, c0), dimensions={1}, to_apply=apply
        minus_f1_f2_sum = f32[2,6,8] add(minus_f1_sum, minus_f2_sum)
        minus_f1_f2_sum_scaled = f32[2,6,8] multiply(minus_f1_f2_sum, r_nelems_bcast)
        minus_f1_f2_sum_scaled_bcast = f32[2,4,6,8] broadcast(minus_f1_f2_sum_scaled), dimensions={0,2,3}
        f1_f2 = f32[2,4,6,8] add(f1, f2)
        dinput = f32[2,4,6,8] add(f1_f2, minus_f1_f2_sum_scaled_bcast)
        ROOT out = (f32[2,4,6,8], f32[2,4,6,8], f32[4], f32[4]) tuple(norm_scale_bias, dinput, dscale, dbias)
    })";

  const char* optimized_hlo = R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,4,6,8], {{.*}}: f32[4], {{.*}}: f32[4], {{.*}}: f32[2,4,6,2,2,2]) -> (f32[2,4,6,8], f32[2,4,6,8], f32[4], f32[4]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[TRANSPOSE0:%[^ ]+]] = f32[2,6,8,4]{3,2,1,0} transpose([[P0]]), dimensions={0,2,3,1}
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[96,4,1,1]{3,2,1,0} bitcast([[TRANSPOSE0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4]{0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[P2_BITCAST:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} bitcast([[P2]])
; CHECK-NEXT:    [[CC0:%[^ ]+]] = (f32[96,4,1,1]{3,2,1,0}, f32[96,1,1,1]{3,2,1,0}, f32[96,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[P2_BITCAST]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0.001
; CHECK-DAG:         "kind":"LAYER_FWD_TRAIN"
; CHECK:           }
; CHECK-DAG:     [[GTE0:%[^ ]+]] = f32[96,4,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=0
; CHECK-DAG:     [[P3:%[^ ]+]] = f32[2,4,6,2,2,2]{5,4,3,2,1,0} parameter(3)
; CHECK-DAG:     [[FUSION0:%[^ ]+]] = f32[2,6,8,4]{3,2,1,0} fusion([[P3]]), kind=kLoop, calls=[[FUSED_COMPUTATION0:%[^ ]+]]
; CHECK-DAG:     [[FUSION0_BITCAST:%[^ ]+]] = f32[96,4,1,1]{3,2,1,0} bitcast([[FUSION0]])
; CHECK-DAG:     [[GTE1:%[^ ]+]] = f32[96,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = f32[96,1,1,1]{3,2,1,0} get-tuple-element([[CC0]]), index=2
; CHECK-NEXT:    [[CC1:%[^ ]+]] = (f32[96,4,1,1]{3,2,1,0}, f32[4,1,1,1]{3,2,1,0}, f32[4,1,1,1]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[FUSION0_BITCAST]], [[GTE1]], [[GTE2]]),
; CHECK:           custom_call_target="__cudnn$norm",
; CHECK:           backend_config={
; CHECK-DAG:         "epsilon":0
; CHECK-DAG:         "kind":"LAYER_BWD"
; CHECK:           }
; CHECK-DAG:     [[GTE3:%[^ ]+]] = f32[96,4,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=0
; CHECK-DAG:     [[FUSION1:%[^ ]+]] = (f32[2,4,6,8]{3,2,1,0}, f32[2,4,6,8]{3,2,1,0}) fusion([[GTE0]], [[GTE3]]), kind=kLoop, calls=[[FUSED_COMPUTATION1:%[^ ]+]]
; CHECK-DAG:     [[GTEF1:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} get-tuple-element([[FUSION1]]), index=0
; CHECK-DAG:     [[GTEF2:%[^ ]+]] = f32[2,4,6,8]{3,2,1,0} get-tuple-element([[FUSION1]]), index=1
; CHECK-DAG:     [[GTE4:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=1
; CHECK-DAG:     [[GTE4_BITCAST:%[^ ]+]] = f32[4]{0} bitcast([[GTE4]])
; CHECK-DAG:     [[GTE5:%[^ ]+]] = f32[4,1,1,1]{3,2,1,0} get-tuple-element([[CC1]]), index=2
; CHECK-DAG:     [[GTE5_BITCAST:%[^ ]+]] = f32[4]{0} bitcast([[GTE5]])
; CHECK-DAG:  ROOT [[OUT:%[^ ]+]] = (f32[2,4,6,8]{3,2,1,0}, f32[2,4,6,8]{3,2,1,0}, f32[4]{0}, f32[4]{0}) tuple([[GTEF1]], [[GTEF2]], [[GTE4_BITCAST]], [[GTE5_BITCAST]])
  )";

  TestNorm(hlo_text, optimized_hlo);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
