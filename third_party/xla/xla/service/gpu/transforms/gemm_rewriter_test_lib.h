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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_GEMM_REWRITER_TEST_LIB_H_
#define XLA_SERVICE_GPU_TRANSFORMS_GEMM_REWRITER_TEST_LIB_H_

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// Base class for GEMM rewriter tests.
class GemmRewriteTestBase : public GpuCodegenTest {
 protected:
  const stream_executor::GpuComputeCapability& Capability() const;

  stream_executor::SemanticVersion GetToolkitVersion() const;
  stream_executor::SemanticVersion GetRuntimeVersion() const;
  bool IsCuda() const;

  bool IsRocm() const;

  bool IsBlackwell() const;

  stream_executor::GpuComputeCapability CudaHopperOrRocmCapability();

  DebugOptions GetDebugOptionsForTest() const override;

  bool SkipGpuBlasLtTest();

  bool HasFp8Support() const;

  bool HasCudaComputeCapability(
      const stream_executor::CudaComputeCapability& cc) const;

 private:
  const auto& device_desc() const;
};

// A test fixture class for tests which should have similar results with legacy
// cublas and cublasLt
class ParameterizedGemmRewriteTestBase
    : public GemmRewriteTestBase,
      public ::testing::WithParamInterface<bool> {
 public:
  ParameterizedGemmRewriteTestBase();

  DebugOptions GetDebugOptionsForTest() const override;

  void MatchOptimizedHlo(absl::string_view hlo, absl::string_view pattern,
                         bool print_operand_shape = false);

  absl::string_view CustomCallTarget();

 protected:
  void SetUp() override;

 protected:
  absl::flat_hash_map<absl::string_view, absl::string_view> replacements_;

 private:
  static constexpr absl::string_view kCustomCallTargetPlaceholder =
      "<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>";
  ;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_GEMM_REWRITER_TEST_LIB_H_
