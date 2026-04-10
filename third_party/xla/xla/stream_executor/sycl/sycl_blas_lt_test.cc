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

#include "xla/stream_executor/sycl/sycl_blas_lt.h"

#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/tests/hlo_test_base.h"

namespace stream_executor::sycl {
namespace {

class SyclBlasLtTest : public xla::HloTestBase {
 public:
  xla::DebugOptions GetDebugOptionsForTest() const override {
    xla::DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cublaslt(true);
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    return debug_options;
  }
};

TEST_F(SyclBlasLtTest, Plugin) {
  stream_executor::StreamExecutor* executor =
      backend().default_stream_executor();
  ASSERT_NE(executor, nullptr);

  blas::BlasSupport* blas_support = executor->AsBlas();
  ASSERT_NE(blas_support, nullptr);

  gpu::BlasLt* blas_lt = blas_support->GetBlasLt();
  ASSERT_NE(blas_lt, nullptr);
  EXPECT_NE(dynamic_cast<sycl::BlasLt*>(blas_lt), nullptr);
}

}  // namespace
}  // namespace stream_executor::sycl
