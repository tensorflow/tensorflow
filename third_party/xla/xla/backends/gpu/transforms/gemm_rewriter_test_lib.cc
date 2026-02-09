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

#include "xla/backends/gpu/transforms/gemm_rewriter_test_lib.h"

#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/gpu_codegen_test.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

const auto& GemmRewriteTestBase::device_desc() const {
  return backend().default_stream_executor()->GetDeviceDescription();
}

stream_executor::SemanticVersion GemmRewriteTestBase::GetRuntimeVersion()
    const {
  return device_desc().runtime_version();
}

const stream_executor::GpuComputeCapability& GemmRewriteTestBase::Capability()
    const {
  return device_desc().gpu_compute_capability();
}

stream_executor::SemanticVersion GemmRewriteTestBase::GetToolkitVersion()
    const {
  return backend()
      .default_stream_executor()
      ->GetDeviceDescription()
      .runtime_version();
}

bool GemmRewriteTestBase::IsCuda() const { return Capability().IsCuda(); }

bool GemmRewriteTestBase::IsRocm() const { return Capability().IsRocm(); }

bool GemmRewriteTestBase::IsBlackwell() const {
  if (IsCuda()) {
    return Capability().cuda_compute_capability()->IsBlackwell();
  }
  return false;
}

stream_executor::GpuComputeCapability
GemmRewriteTestBase::CudaHopperOrRocmCapability() {
  if (IsCuda()) {
    return se::CudaComputeCapability::Hopper();
  }
  return stream_executor::GpuComputeCapability{
      *Capability().rocm_compute_capability()};
}

DebugOptions GemmRewriteTestBase::GetDebugOptionsForTest() const {
  DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
  // These tests test the cuBLAS rewriter so we have to make sure that we use
  // cuBLAS for them.
  debug_options.set_xla_gpu_enable_triton_gemm(false);
  debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);
  return debug_options;
}

bool GemmRewriteTestBase::SkipGpuBlasLtTest() {
  return !IsCuda() &&
         !Capability().rocm_compute_capability()->has_hipblaslt() &&
         GetDebugOptionsForTest().xla_gpu_enable_cublaslt();
}

bool GemmRewriteTestBase::HasFp8Support() const {
  if (IsCuda()) {
    return Capability().cuda_compute_capability()->IsAtLeast(8, 9);
  }
  return Capability().rocm_compute_capability()->has_fp8_support();
}

bool GemmRewriteTestBase::HasCudaComputeCapability(
    const stream_executor::CudaComputeCapability& cc) const {
  return IsCuda() &&
         Capability().cuda_compute_capability()->SupportsAllFeaturesOf(cc);
}

ParameterizedGemmRewriteTestBase::ParameterizedGemmRewriteTestBase() {
  const bool kUsingCublasLt = GetParam();
  replacements_[kCustomCallTargetPlaceholder] =
      kUsingCublasLt ? "__cublas$lt$matmul" : "__cublas$gemm";
}

DebugOptions ParameterizedGemmRewriteTestBase::GetDebugOptionsForTest() const {
  DebugOptions debug_options = GemmRewriteTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_cublaslt(GetParam());
  debug_options.set_xla_gpu_enable_triton_gemm(false);
  return debug_options;
}

void ParameterizedGemmRewriteTestBase::MatchOptimizedHlo(
    absl::string_view hlo, const absl::string_view pattern,
    bool print_operand_shape) {
  GemmRewriteTestBase::MatchOptimizedHlo(
      hlo, absl::StrReplaceAll(pattern, replacements_), print_operand_shape);
}

absl::string_view ParameterizedGemmRewriteTestBase::CustomCallTarget() {
  return replacements_[kCustomCallTargetPlaceholder];
}

void ParameterizedGemmRewriteTestBase::SetUp() {
  if (SkipGpuBlasLtTest()) {
    GTEST_SKIP() << "BlasLt is not supported on this GPU architecture";
  }
}

}  // namespace xla::gpu
