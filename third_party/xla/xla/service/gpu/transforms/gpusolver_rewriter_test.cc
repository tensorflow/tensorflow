/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/gpusolver_rewriter.h"

#include <complex>
#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu_solver_context.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

class GpuSolverContextStub : stream_executor::GpuSolverContext {
 public:
  GpuSolverContextStub() = default;
  static absl::StatusOr<std::unique_ptr<GpuSolverContext>> Create() {
    return absl::WrapUnique(
        static_cast<GpuSolverContext*>(new GpuSolverContextStub));
  }

  absl::Status SetStream(stream_executor::Stream* stream) override {
    return UnimplementedError();
  }

  absl::Status PotrfBatched(stream_executor::blas::UpperLower uplo, int n,
                            stream_executor::DeviceMemory<float*> as, int lda,
                            stream_executor::DeviceMemory<int> lapack_info,
                            int batch_size) override {
    return UnimplementedError();
  }
  absl::Status PotrfBatched(stream_executor::blas::UpperLower uplo, int n,
                            stream_executor::DeviceMemory<double*> as, int lda,
                            stream_executor::DeviceMemory<int> lapack_info,
                            int batch_size) override {
    return UnimplementedError();
  }
  absl::Status PotrfBatched(
      stream_executor::blas::UpperLower uplo, int n,
      stream_executor::DeviceMemory<std::complex<float>*> as, int lda,
      stream_executor::DeviceMemory<int> lapack_info, int batch_size) override {
    return UnimplementedError();
  }
  absl::Status PotrfBatched(
      stream_executor::blas::UpperLower uplo, int n,
      stream_executor::DeviceMemory<std::complex<double>*> as, int lda,
      stream_executor::DeviceMemory<int> lapack_info, int batch_size) override {
    return UnimplementedError();
  }

  absl::Status Potrf(stream_executor::blas::UpperLower uplo, int n,
                     stream_executor::DeviceMemory<float> a, int lda,
                     stream_executor::DeviceMemory<int> lapack_info,
                     stream_executor::DeviceMemory<float> workspace) override {
    return UnimplementedError();
  }
  absl::Status Potrf(stream_executor::blas::UpperLower uplo, int n,
                     stream_executor::DeviceMemory<double> a, int lda,
                     stream_executor::DeviceMemory<int> lapack_info,
                     stream_executor::DeviceMemory<double> workspace) override {
    return UnimplementedError();
  }
  absl::Status Potrf(
      stream_executor::blas::UpperLower uplo, int n,
      stream_executor::DeviceMemory<std::complex<float>> a, int lda,
      stream_executor::DeviceMemory<int> lapack_info,
      stream_executor::DeviceMemory<std::complex<float>> workspace) override {
    return UnimplementedError();
  }
  absl::Status Potrf(
      stream_executor::blas::UpperLower uplo, int n,
      stream_executor::DeviceMemory<std::complex<double>> a, int lda,
      stream_executor::DeviceMemory<int> lapack_info,
      stream_executor::DeviceMemory<std::complex<double>> workspace) override {
    return UnimplementedError();
  }

  absl::StatusOr<int64_t> PotrfBufferSize(
      xla::PrimitiveType type, stream_executor::blas::UpperLower uplo, int n,
      int lda, int batch_size) override {
    return 0;
  }

 private:
  static absl::Status UnimplementedError() {
    return absl::UnimplementedError("Not needed for the unit test");
  }
};

class GpusolverRewriterTest : public HloHardwareIndependentTestBase {
 public:
  GpusolverRewriter gpusolver_rewriter_{GpuSolverContextStub::Create};
};

TEST_F(GpusolverRewriterTest, CholeskyTest) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule CholeskyTest

  ENTRY entry_computation {
    input = f32[1,256,256] parameter(0)
    ROOT decomp = f32[1,256,256] cholesky(input)
  }
)")
                    .value();

  EXPECT_TRUE(gpusolver_rewriter_.Run(module.get()).value());

  const HloInstruction* entry_root =
      module->entry_computation()->root_instruction();
  ASSERT_THAT(
      entry_root,
      GmockMatch(m::Select(
          m::Broadcast(
              m::Compare(m::GetTupleElement(), m::Broadcast(m::Constant()))),
          m::GetTupleElement(m::CustomCall()), m::Broadcast(m::Constant()))));
}
}  // namespace
}  // namespace gpu
}  // namespace xla
